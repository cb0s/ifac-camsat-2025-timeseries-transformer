from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

from src.data.dataset import RawTrainingDataset
from src.physics.orekit_helpers import setup_orekit

setup_orekit()

# we must wait until setup_orekit() has been called before importing any physics modules
from src.physics import eclipse, msis_density, sample_orbit, solar_wind

# currently we have 880 features from this alone
F10_7_LAST_X_COUNT = 28  # 28 days -> ~27 days for one sun rotation -> capture seasonal effects
AP_LAST_X_COUNT = 0  # 24  # 2 days + 2nd order
KP_LAST_X_COUNT = 0  # 24  # 2 days -> similar to AP
AE_LAST_X_COUNT = 24  # 1 day + 2nd order
DST_LAST_X_COUNT = 24  # 1 day + 2nd order
SW_PLASMA_SPEED_LAST_X_COUNT = 24  # 1 day + 2nd order
ELECTRIC_FIELD_LAST_X_COUNT = 24  # 1 day + 2nd order
B_X_LAST_X_COUNT = 0  # 12  # 0.5 days -> not as important as by or bz
B_Y_LAST_X_COUNT = 12  # 0.5 days + 2nd order
B_Z_LAST_X_COUNT = 12  # 0.5 days + 2nd order

AE_6H_LAST_X_COUNT = 12  # 3 days + 2nd order
DST_6H_LAST_X_COUNT = 12  # 3 days + 2nd order
SW_PLASMA_SPEED_6H_LAST_X_COUNT = 12  # 3 days
SW_PROTON_DENSITY_6H_LAST_X_COUNT = 16  # 4 days + interpolate
FLOW_PRESSURE_6H_LAST_X_COUNT = 16  # 4 days
ELECTRIC_FIELD_6H_LAST_X_COUNT = 12  # 3 days
B_X_6H_LAST_X_COUNT = 10  # 2.5 days -> not as important as by or bz
B_Y_6H_LAST_X_COUNT = 10  # 2.5 days
B_Z_6H_LAST_X_COUNT = 10  # 2.5 days

GOES_LAST_X_COUNT = 24

TARGET_HORIZON = 432

# configurable knobs
MAX_MISSING = 144  # one of 3 days is missing -> columns with > n NaNs are skipped
# tested for sat_density: doesn't change max, changes mean and std after 2nd place after decimal -> should be ok
MAX_MISSING_HIGHER_ORDER = 6
DEFAULT_DOWNSAMPLE = "6h"  # e.g. "30T", "2h" …
DEFAULT_TOLERANCE = 0.1  # 0 = strictly inside neighbours

MSIS_STEPS = 1


def _clip_to_neighbours(filled: pd.DataFrame,
                        original: pd.DataFrame,
                        tolerance: float = 0.0) -> pd.DataFrame:
    """Clip filled values to neighbour range."""
    filled, original = filled.align(original, join='left', axis=0)
    common_cols = filled.columns.intersection(original.columns)
    filled = filled[common_cols]
    original = original[common_cols]

    prev_real = original.ffill()
    next_real = original.bfill()

    lo = prev_real.combine(next_real, np.minimum)
    hi = prev_real.combine(next_real, np.maximum)

    band = (hi - lo).abs() * tolerance
    lo_banded = lo - band
    hi_banded = hi + band

    return filled.clip(lower=lo_banded, upper=hi_banded)


def _interpolate_group(
        group: pd.DataFrame,
        max_missing: int = MAX_MISSING,
        downsample_rule: str = DEFAULT_DOWNSAMPLE,
        overshoot_tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Interpolate and refine a single group."""
    group = group.sort_index()
    orig_idx = group.index
    epoch = orig_idx.get_level_values("Timestamp")[0]
    ts_td = (orig_idx.get_level_values("Timestamp") - epoch)

    work = group.copy()
    work.index = ts_td

    na_cnt = work.isna().sum()
    valid_cnt = work.shape[0] - na_cnt
    skip_cols = (na_cnt > max_missing) | (valid_cnt < 3)

    untouched = work.loc[:, skip_cols]
    to_fill = work.loc[:, ~skip_cols].copy()

    to_fill = to_fill.interpolate(method="pchip", limit_direction="both", limit=MAX_MISSING_HIGHER_ORDER)
    to_fill = to_fill.interpolate(method="linear", limit_direction="both")

    prev_real_orig_idx = group.ffill()
    next_real_orig_idx = group.bfill()
    lo_orig_idx = prev_real_orig_idx.combine(next_real_orig_idx, np.minimum)
    hi_orig_idx = next_real_orig_idx.combine(prev_real_orig_idx, np.maximum)  # Fixed potential bug, ensure max

    lo_aligned, hi_aligned = lo_orig_idx.align(hi_orig_idx, join='inner')
    band = (hi_aligned - lo_aligned).abs() * overshoot_tolerance
    lo_banded_orig_idx = lo_aligned - band
    hi_banded_orig_idx = hi_aligned + band

    to_fill_orig_idx = to_fill.copy()
    to_fill_orig_idx.index = orig_idx

    temp_to_fill, temp_lo = to_fill_orig_idx.align(
        lo_banded_orig_idx.loc[:, to_fill_orig_idx.columns],
        join='left', axis=0
    )

    to_fill_aligned_for_check, hi_aligned_for_check = temp_to_fill.align(
        hi_banded_orig_idx.loc[:, to_fill_orig_idx.columns],
        join='left', axis=0
    )

    lo_aligned_for_check = temp_lo

    bad_nan = to_fill_aligned_for_check.columns[to_fill_aligned_for_check.isna().any()]
    bad_over = to_fill_aligned_for_check.columns[((to_fill_aligned_for_check < lo_aligned_for_check) |
                                                  (to_fill_aligned_for_check > hi_aligned_for_check)).any()]
    bad_cols = bad_nan.union(bad_over)

    if len(bad_cols):
        bad = to_fill.loc[:, bad_cols]
        bad_dt_index = pd.to_datetime(bad.index, unit='ns')
        bad_dt = bad.copy()
        bad_dt.index = bad_dt_index
        bad_down = bad_dt.resample(downsample_rule).mean()
        bad_down = bad_down.interpolate(method="pchip", limit_direction="both", limit=MAX_MISSING_HIGHER_ORDER)
        bad_down = bad_down.interpolate(method="akima", limit_direction="both", limit=MAX_MISSING_HIGHER_ORDER)
        bad_down = bad_down.interpolate(method="linear", limit_direction="both")
        full_td_index = work.index
        bad_up = bad_down.reindex(full_td_index).interpolate(
            method="pchip", limit_direction="both", limit=MAX_MISSING_HIGHER_ORDER
        )
        to_fill.loc[:, bad_cols] = bad_up

    to_fill = to_fill.interpolate(method="linear", limit_direction="both")
    to_fill.index = orig_idx
    to_fill = _clip_to_neighbours(to_fill, group, overshoot_tolerance)

    untouched.index = orig_idx
    result = (
        pd.concat([untouched, to_fill], axis=1)
        .reindex(columns=group.columns)
    )
    return result


def _pad_group(group: pd.DataFrame,
               columns: list[str],
               target_length: int = TARGET_HORIZON) -> pd.DataFrame:
    """Pad a single group to target_length."""
    group = group.sort_index()
    cur_len = len(group)

    pad_len = target_length - cur_len

    if pad_len <= 0:
        return group

    ts = group.index.get_level_values("Timestamp").to_series()
    file_id = group.index.get_level_values("file_id").unique()[0]

    freq = pd.Timedelta(minutes=10)

    extra_ts = pd.date_range(
        start=ts.iloc[-1] + freq, periods=pad_len, freq=freq
    )

    extra_idx = pd.MultiIndex.from_product(
        [[file_id], extra_ts], names=["file_id", "Timestamp"]
    )

    full_idx = group.index.append(extra_idx)
    padded = group.reindex(full_idx)

    if columns is not None:
        padded = padded.reindex(columns=columns)

    return padded


def _process_group_batch(
        batch_groups: list[tuple[any, pd.DataFrame]],
        columns: list[str],
        target_length: int = TARGET_HORIZON,
        max_missing: int = MAX_MISSING,
        **interp_kw
) -> pd.DataFrame:
    """Process a batch of groups."""
    processed_results = []
    for name, group in batch_groups:
        padded_group = _pad_group(group, columns, target_length)
        interpolated_group = _interpolate_group(
            padded_group,
            max_missing=max_missing,
            **interp_kw
        )
        processed_results.append(interpolated_group)

    return pd.concat(processed_results)


def _pad_and_interpolate_parallel_batched(
        df: pd.DataFrame,
        columns: list[str],
        target_length: int = TARGET_HORIZON,
        max_missing: int = MAX_MISSING,
        level: str = "file_id",
        n_jobs: int | None = -1,
        backend: str = "loky",
        prefer: str = "processes",
        batch_size: int | None = None,
        **interp_kw
) -> pd.DataFrame:
    """Run padding/interpolation on groups in parallel batches."""
    groups = list(df.groupby(level=level, sort=False))
    num_groups = len(groups)
    actual_n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    actual_n_jobs = max(1, actual_n_jobs)
    batch_size = max(1, num_groups // actual_n_jobs) if batch_size is None else batch_size

    group_batches = [groups[i:i + batch_size] for i in range(0, num_groups, batch_size)]

    processed_batches = Parallel(
        n_jobs=n_jobs, backend=backend, prefer=prefer
    )(
        delayed(_process_group_batch)(
            batch,
            columns,
            target_length,
            max_missing,
            **interp_kw
        )
        for batch in group_batches
    )

    return pd.concat(processed_batches).sort_index()


def _pad_and_interpolate_dataframe(df, columns, target_length=TARGET_HORIZON, max_missing=MAX_MISSING) \
        -> tuple[pd.DataFrame, list[int]]:
    # Step 1: Pad all groups + interpolate
    df = _pad_and_interpolate_parallel_batched(df, columns, target_length, max_missing, batch_size=50)
    # df = df.groupby(level='file_id').apply(
    #         lambda g: _pad_and_interpolate_parallel(g, columns, target_length, max_missing)
    # )

    # Step 2: Identify bad file_ids
    is_bad = df.groupby(level='file_id')[columns].apply(
        lambda g: g.isna().any().any()
    )

    bad_file_ids = is_bad[is_bad].index.tolist()

    # Step 3: Drop bad file_ids
    # cleaned_df = df[~df.index.get_level_values('file_id').isin(bad_file_ids)]

    # return cleaned_df, bad_file_ids
    return df, bad_file_ids


def _get_features(df: pd.DataFrame, key: str | list[str], last_x_count: int,
                  expected_total_items: int, resample_interval: str | None = None,
                  first_order_derivatives: bool = False,
                  second_order_derivatives: bool = False,
                  cut_last_n: int = 0, interpolate: bool = False,
                  non_nan_min_ratio: float = 0.0) -> tuple[list[np.ndarray], None | pd.Series]:
    if last_x_count == 0:
        return [], pd.Series([])

    feature_df = df[key]
    valid_ids = None

    if resample_interval is not None:
        feature_df = (
            feature_df.groupby(level='file_id')
            .resample(resample_interval, level='Timestamp').mean()
            .groupby(level='file_id').tail(expected_total_items)
        )

    if non_nan_min_ratio > 0.0:
        # Calculate fraction of non-NaNs per group
        valid_ids = (
            feature_df
            .groupby(level='file_id')
            .apply(lambda x: x.notna().sum() / len(x) >= non_nan_min_ratio)
        )

        valid_ids = valid_ids.all(axis=1)

        # Filter out file_ids that don't meet the threshold
        # feature_df = feature_df[feature_df.index.get_level_values('file_id').isin(valid_ids[valid_ids].index)]
        feature_df = feature_df[feature_df.index.get_level_values('file_id').isin(valid_ids.index)]

    if interpolate:
        feature_df = (
            feature_df.groupby(level='file_id')
            .apply(
                lambda g: g.droplevel(level='file_id')
                # try smooth interpolation
                .interpolate('pchip', limit_direction='both', limit=MAX_MISSING_HIGHER_ORDER)
                .interpolate('linear', limit_direction='both')  # fallback
            )
        )

    if cut_last_n > 0:
        feature_df = feature_df.groupby(level='file_id').head(expected_total_items - cut_last_n)

    grouped_features = feature_df.groupby(level='file_id')

    feat_dim = len(key) if isinstance(key, list) else 1
    if feat_dim == 1:
        last_x = grouped_features.tail(last_x_count).to_numpy().reshape(-1, last_x_count)
    else:
        last_x = grouped_features.tail(last_x_count).to_numpy().reshape((-1, last_x_count, feat_dim))

    mean_ = grouped_features.mean().to_numpy().reshape(-1, feat_dim)
    min_ = grouped_features.min().to_numpy().reshape(-1, feat_dim)
    max_ = grouped_features.max().to_numpy().reshape(-1, feat_dim)
    std_ = grouped_features.std().to_numpy().reshape(-1, feat_dim)
    if feat_dim != 1:
        mean_ = mean_.reshape(-1, 1, feat_dim)
        min_ = min_.reshape(-1, 1, feat_dim)
        max_ = max_.reshape(-1, 1, feat_dim)
        std_ = std_.reshape(-1, 1, feat_dim)

    features = [
        # these exclude the cut values
        mean_,
        min_,
        max_,
        std_,

        last_x,
    ]

    if first_order_derivatives:
        if feat_dim == 1:
            feature_derivative_count = last_x_count if last_x_count < expected_total_items - 1 else expected_total_items - 1

            # temp last_x t-1
            last_x_m1 = grouped_features.tail(feature_derivative_count + 1) \
                            .to_numpy().reshape(-1, feature_derivative_count + 1)[:, :-1]

            last_x_derivatives = (
                    last_x[:, last_x.shape[1] - feature_derivative_count:] -
                    last_x_m1
            )
        else:
            feature_derivative_count = last_x_count if last_x_count < expected_total_items - 1 else expected_total_items - 1

            # temp last_x t-1
            last_x_m1 = grouped_features.tail(feature_derivative_count + 1) \
                            .to_numpy().reshape(-1, feature_derivative_count + 1, feat_dim)[:, :-1, :]

            last_x_derivatives = (
                    last_x[:, last_x.shape[1] - feature_derivative_count:, :] -
                    last_x_m1
            )

        features.append(last_x_derivatives)

        # do we also want 2nd order derivatives?
        if second_order_derivatives:
            n_2nd_order_ds = last_x_count if last_x_count < expected_total_items - 2 else expected_total_items - 2
            if feat_dim == 1:
                last_x_2nd_ds = (
                        last_x_derivatives[:, feature_derivative_count - n_2nd_order_ds:] -
                        (grouped_features.tail(n_2nd_order_ds + 2).to_numpy()
                         .reshape(-1, n_2nd_order_ds + 2)[:, :-2] - last_x_m1)
                )
            else:
                last_x_2nd_ds = (
                        last_x_derivatives[:, :, feature_derivative_count - n_2nd_order_ds:] -
                        (grouped_features.tail(n_2nd_order_ds + 2).to_numpy()
                         .reshape(-1, n_2nd_order_ds + 2, feat_dim)[:, :-2, :] - last_x_m1)
                )
            features.append(last_x_2nd_ds)

    return features, valid_ids


def prepare_omni_aggregations(omni_df: pd.DataFrame) -> np.ndarray:
    interpolate = False
    # f10_7 is only available daily
    # maybe median to account for a majority of values? they are not perfectly centered
    f10_7, _ = _get_features(omni_df, 'f10_7_index', F10_7_LAST_X_COUNT, 60, 'D', interpolate=interpolate)

    # ap_index is only available in 3h resolution
    # same here -> maybe median?
    ap, _ = _get_features(omni_df, 'ap_index', AP_LAST_X_COUNT, 480, '3h', second_order_derivatives=False,
                          interpolate=interpolate)

    # kp_index is only available in 3h resolution
    kp, _ = _get_features(omni_df, 'kp_index', KP_LAST_X_COUNT, 480, '3h', interpolate=interpolate)

    # ae_index is available in 1h resolution
    ae, _ = _get_features(omni_df, 'ae_index', AE_LAST_X_COUNT, 1440, second_order_derivatives=False,
                          interpolate=interpolate)

    # dst_index is available in 1h resolution
    dst, _ = _get_features(omni_df, 'dst_index', DST_LAST_X_COUNT, 1440, second_order_derivatives=False,
                           interpolate=interpolate)

    # b_x, b_y, b_z are available in 1h resolution
    bx, _ = _get_features(omni_df, 'b_x', B_X_LAST_X_COUNT, 1440, interpolate=interpolate)
    by, _ = _get_features(omni_df, 'b_y', B_Y_LAST_X_COUNT, 1440, second_order_derivatives=False,
                          interpolate=interpolate)
    bz, _ = _get_features(omni_df, 'b_z', B_Z_LAST_X_COUNT, 1440, second_order_derivatives=False,
                          interpolate=interpolate)

    # sw_plasma_speed is available in 1h resolution
    sw_plasma_speed, _ = _get_features(omni_df, 'sw_plasma_speed', SW_PLASMA_SPEED_LAST_X_COUNT, 1440,
                                       second_order_derivatives=False, interpolate=interpolate)

    # sw_proton_density is available in 1h resolution -> some values are nan -> aggregation necessary
    # daily aggregation with 3-day interpolation would work, for instance
    sw_proton_density, _ = _get_features(omni_df, 'sw_proton_density', SW_PROTON_DENSITY_6H_LAST_X_COUNT, 240, '6h',
                                         interpolate=interpolate)

    # flow_pressure is available in 1h resolution -> some values are nan -> aggregation necessary
    flow_pressure, _ = _get_features(omni_df, 'flow_pressure', FLOW_PRESSURE_6H_LAST_X_COUNT, 240, '6h',
                                     interpolate=interpolate)

    # electric field is available in 1h resolution
    electric_field, _ = _get_features(omni_df, 'electric_field', ELECTRIC_FIELD_LAST_X_COUNT, 1440,
                                      second_order_derivatives=False, interpolate=interpolate)

    # coarser data, but longer timespan
    ae_6h, _ = _get_features(omni_df, 'ae_index', AE_6H_LAST_X_COUNT, 240, '6h',
                             second_order_derivatives=False, cut_last_n=AE_LAST_X_COUNT // 6, interpolate=interpolate)
    dst_6h, _ = _get_features(omni_df, 'dst_index', DST_6H_LAST_X_COUNT, 240, '6h',
                              second_order_derivatives=False, cut_last_n=DST_LAST_X_COUNT // 6, interpolate=interpolate)
    sw_plasma_speed_6h, _ = _get_features(omni_df, 'sw_plasma_speed', SW_PLASMA_SPEED_6H_LAST_X_COUNT, 240, '6h',
                                          cut_last_n=SW_PLASMA_SPEED_LAST_X_COUNT // 6, interpolate=interpolate)
    electric_field_6h, _ = _get_features(omni_df, 'electric_field', ELECTRIC_FIELD_6H_LAST_X_COUNT, 240, '6h',
                                         cut_last_n=ELECTRIC_FIELD_LAST_X_COUNT // 6, interpolate=interpolate)
    bx_6h, _ = _get_features(omni_df, 'b_x', B_X_6H_LAST_X_COUNT, 240, '6h',
                             cut_last_n=B_X_LAST_X_COUNT // 6, interpolate=interpolate)
    by_6h, _ = _get_features(omni_df, 'b_y', B_Y_6H_LAST_X_COUNT, 240, '6h',
                             cut_last_n=B_Y_LAST_X_COUNT // 6, interpolate=interpolate)
    bz_6h, _ = _get_features(omni_df, 'b_z', B_Z_6H_LAST_X_COUNT, 240, '6h',
                             cut_last_n=B_Z_LAST_X_COUNT // 6, interpolate=interpolate)

    features = [
        *f10_7,
        *ap,
        *kp,
        *ae,
        *dst,
        *bx,
        *by,
        *bz,
        *sw_plasma_speed,
        *sw_proton_density,
        *flow_pressure,
        *electric_field,

        *ae_6h,
        *dst_6h,
        *sw_plasma_speed_6h,
        *electric_field_6h,
        *bx_6h,
        *by_6h,
        *bz_6h
    ]

    return np.concatenate(features, axis=1)


def prepare_main_model_features(
        omni_df: pd.DataFrame,
        initial_state_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # calculate physics features
    solar_wind_df = solar_wind.calculate_solar_wind_fields(omni_df.groupby(level='file_id').last())
    eclipse_df = eclipse.calculate_eclipse_details(initial_state_df, mode='strict')
    msis_df = msis_density.get_nrlm_densities(initial_state_df, omni_df, mode='strict', n_steps=MSIS_STEPS)
    sample_orbit_df = sample_orbit.calculate_mlat_mlt_sza(initial_state_df, mode='strict')

    # omni aggregations
    omni_df_agg = prepare_omni_aggregations(omni_df)

    # full feature array
    n_dps = omni_df_agg.shape[0]
    feature_list = [
        eclipse_df.to_numpy().reshape(n_dps, -1),
        sample_orbit_df.to_numpy().reshape(n_dps, -1),
        solar_wind_df.to_numpy().reshape(n_dps, -1),
        omni_df_agg
    ]

    bad_files = np.arange(omni_df_agg.shape[0])[np.isnan(omni_df_agg).any(axis=1)]

    return np.concatenate(feature_list, axis=1), msis_df.to_numpy().reshape(n_dps, -1), bad_files


def prepare_refinement_model_features(
        goes_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    feature_list, valid_ids = _get_features(goes_df, ['xrsa_flux', 'xrsb_flux'], GOES_LAST_X_COUNT, 480, '3h',
                                            interpolate=True, non_nan_min_ratio=0.1, second_order_derivatives=False)
    # return np.concatenate(feature_list, axis=1).reshape(-1, (GOES_LAST_X_COUNT * 2 + 4) * 2), valid_ids.to_numpy()
    return np.concatenate(feature_list, axis=1).reshape(-1, (GOES_LAST_X_COUNT + 4) * 2), valid_ids.to_numpy()


def prepare_expected_outputs(sat_density_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sat_density_df, bad_file_ids = _pad_and_interpolate_dataframe(sat_density_df, sat_density_df.columns.tolist(),
                                                                  target_length=TARGET_HORIZON)
    sat_density_df = sat_density_df.groupby(level='file_id').tail(TARGET_HORIZON)

    # clipping removes potential outliers
    sat_density_np = sat_density_df.to_numpy().reshape(-1, TARGET_HORIZON).clip(1e-15, 1e-10)
    return sat_density_np, np.array(bad_file_ids)


def create_main_model_ds(
        main_features: np.ndarray,
        density_predictions_nrlm: np.ndarray,
        sat_densities: np.ndarray,
        bad_files: np.ndarray
) -> RawTrainingDataset:
    good_files_map = ~np.isin(np.arange(main_features.shape[0]), bad_files)
    main_features = main_features[good_files_map]
    density_predictions_nrlm = density_predictions_nrlm[good_files_map]
    sat_densities = sat_densities[good_files_map]

    residuals = sat_densities - density_predictions_nrlm

    return RawTrainingDataset(
        main_features, residuals,
        nrlm_predictions=density_predictions_nrlm,  # for convenience,
        bad_files=bad_files,
        main_features=main_features
    )


def create_refinement_model_ds(
        main_features: np.ndarray,
        refinement_features: np.ndarray,
        goes_good_files: np.ndarray,
        density_predictions_nrlm: np.ndarray,
        sat_densities: np.ndarray,
        bad_files: np.ndarray
) -> RawTrainingDataset:
    good_files_map = ~np.isin(np.arange(refinement_features.shape[0]), bad_files)

    goes_good_files = goes_good_files & good_files_map
    refinement_features = refinement_features[goes_good_files]
    sat_densities = sat_densities[goes_good_files]
    density_predictions_nrlm = density_predictions_nrlm[goes_good_files]
    main_features = main_features[goes_good_files]

    residuals = sat_densities - density_predictions_nrlm

    return RawTrainingDataset(
        refinement_features,
        residuals,  # this cannot be trained without the main model as the residuals are still to be calculated
        good_files_map=good_files_map,
        nrlm_predictions=density_predictions_nrlm,  # needed for OD-RMSE Loss and the back transformation
        main_features=main_features,  # needed for the calculation of the static refinement features
        good_files_goes=goes_good_files,
    )


def prepare_and_save(
        omni_df: pd.DataFrame,
        initial_state_df: pd.DataFrame,
        goes_df: pd.DataFrame,
        sat_density_df: pd.DataFrame,
        output_dir: str | Path
) -> None:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    print('Preparing main model dataset...')
    omni_df = (
        omni_df.groupby(level='file_id')
        .apply(
            lambda g: g.droplevel(level='file_id')
            # try smooth interpolation
            .interpolate('pchip', limit_direction='both', limit=MAX_MISSING_HIGHER_ORDER)
            .interpolate('linear', limit_direction='both')  # fallback
        )
    )

    print('Calculating physics features...')
    main_features, density_predictions_nrlm, bad_files_omni = prepare_main_model_features(omni_df, initial_state_df)
    refinement_features, good_files_goes = prepare_refinement_model_features(goes_df)
    sat_densities, bad_files = prepare_expected_outputs(sat_density_df)

    bad_files = np.union1d(bad_files, bad_files_omni)

    print('Creating main model features...')
    main_model_ds_file = output_dir / 'residual_model_ds-reduced.npz'
    refinement_model_ds_file = output_dir / 'refinement_model_ds-reduced.npz'
    raw_ds = create_main_model_ds(main_features, density_predictions_nrlm, sat_densities, bad_files)
    raw_ds.save(main_model_ds_file)
    np.save(output_dir / 'main_features-reduced.npy', raw_ds.other_fields['main_features'])
    print(f'Saved main model dataset to {main_model_ds_file}')
    print('Creating refinement model features...')
    create_refinement_model_ds(main_features, refinement_features, good_files_goes, density_predictions_nrlm,
                               sat_densities, bad_files).save(refinement_model_ds_file)
    print(f'Saved refinement model dataset to {refinement_model_ds_file}')


def main():
    prepare_and_save(
        pd.read_parquet('~/Projects/Uni/MIT-Aero-AI-2025/data/preprocessed/aggregated/omni2.parquet'),
        pd.read_parquet('~/Projects/Uni/MIT-Aero-AI-2025/data/preprocessed/aggregated/initial_states.parquet'),
        pd.read_parquet('~/Projects/Uni/MIT-Aero-AI-2025/data/preprocessed/aggregated/goes.parquet'),
        pd.read_parquet('~/Projects/Uni/MIT-Aero-AI-2025/data/preprocessed/aggregated/sat_density.parquet'),
        '/home/cedric/Projects/Uni/MIT-Aero-AI-2025/data/training_ds/'
    )


if __name__ == '__main__':
    main()
