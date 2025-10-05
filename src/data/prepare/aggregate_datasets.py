# some helper functions down the line
import os
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

CACHE_SIZE = 5_000
RAD_OUTLIER_VALUE = np.nan
DEFAULT_OUTLIER_VALUE = np.nan
DEFAULT_INT_OUTLIER_VALUE = -1

GOES_NAN_MAPPING = {
    'quad_diode': DEFAULT_OUTLIER_VALUE,
    'xrsa_flux': DEFAULT_OUTLIER_VALUE, 'xrsb_flux': DEFAULT_OUTLIER_VALUE,
    'xrsa_flux_observed': DEFAULT_OUTLIER_VALUE, 'xrsa_flux_electrons': DEFAULT_OUTLIER_VALUE,
    'xrsb_flux_observed': DEFAULT_OUTLIER_VALUE,
    'xrsb_flux_electrons': DEFAULT_OUTLIER_VALUE,
    'xrsa_flag': 65535, 'xrsb_flag': 65535,
    'xrsa_num': DEFAULT_INT_OUTLIER_VALUE, 'xrsb_num': DEFAULT_INT_OUTLIER_VALUE, 'xrsa_flag_excluded': 65535,
    'xrsb_flag_excluded': 65535,
    'au_factor': DEFAULT_OUTLIER_VALUE, 'corrected_current_xrsb2': DEFAULT_OUTLIER_VALUE, 'roll_angle': -4.0,
    'xrsa1_flux': DEFAULT_OUTLIER_VALUE, 'xrsa1_flux_observed': DEFAULT_OUTLIER_VALUE,
    'xrsa1_flux_electrons': DEFAULT_OUTLIER_VALUE,
    'xrsa2_flux': DEFAULT_OUTLIER_VALUE, 'xrsa2_flux_observed': DEFAULT_OUTLIER_VALUE,
    'xrsa2_flux_electrons': DEFAULT_OUTLIER_VALUE,
    'xrsb1_flux': DEFAULT_OUTLIER_VALUE, 'xrsb1_flux_observed': DEFAULT_OUTLIER_VALUE,
    'xrsb1_flux_electrons': DEFAULT_OUTLIER_VALUE,
    'xrsb2_flux': DEFAULT_OUTLIER_VALUE, 'xrsb2_flux_observed': DEFAULT_OUTLIER_VALUE,
    'xrsb2_flux_electrons': DEFAULT_OUTLIER_VALUE,
    'xrs_primary_chan': 65535, 'xrsa1_flag': 255, 'xrsa2_flag': 255, 'xrsb1_flag': 255, 'xrsb2_flag': 255,
    'xrsa1_num': DEFAULT_INT_OUTLIER_VALUE, 'xrsa2_num': DEFAULT_INT_OUTLIER_VALUE,
    'xrsb1_num': DEFAULT_INT_OUTLIER_VALUE, 'xrsb2_num': DEFAULT_INT_OUTLIER_VALUE,
    'xrsa1_flag_excluded': 65535, 'xrsa2_flag_excluded': 65535, 'xrsb1_flag_excluded': 65535,
    'xrsb2_flag_excluded': 65535,
    'yaw_flip_flag': 255
}

GOES_TYPE_MAPPING = {
    'quad_diode': np.float32,
    'xrsa_flux': np.float32,
    'xrsa_flux_observed': np.float32,
    'xrsa_flux_electrons': np.float32,
    'xrsb_flux': np.float32,
    'xrsb_flux_observed': np.float32,
    'xrsb_flux_electrons': np.float32,
    'xrsa_flag': np.uint16,
    'xrsb_flag': np.uint16,
    'xrsa_num': np.int32,
    'xrsb_num': np.int32,
    'xrsa_flag_excluded': np.uint16,
    'xrsb_flag_excluded': np.uint16,
    'au_factor': np.float32,
    'corrected_current_xrsb2': np.float32,
    'roll_angle': np.float32,
    'xrsa1_flux': np.float32,
    'xrsa1_flux_observed': np.float32,
    'xrsa1_flux_electrons': np.float32,
    'xrsa2_flux': np.float32,
    'xrsa2_flux_observed': np.float32,
    'xrsa2_flux_electrons': np.float32,
    'xrsb1_flux': np.float32,
    'xrsb1_flux_observed': np.float32,
    'xrsb1_flux_electrons': np.float32,
    'xrsb2_flux': np.float32,
    'xrsb2_flux_observed': np.float32,
    'xrsb2_flux_electrons': np.float32,
    'xrs_primary_chan': np.uint16,
    'xrsa1_flag': np.uint8,
    'xrsa2_flag': np.uint8,
    'xrsb1_flag': np.uint8,
    'xrsb2_flag': np.uint8,
    'xrsa1_num': np.int32,
    'xrsa2_num': np.int32,
    'xrsb1_num': np.int32,
    'xrsb2_num': np.int32,
    'xrsa1_flag_excluded': np.uint16,
    'xrsa2_flag_excluded': np.uint16,
    'xrsb1_flag_excluded': np.uint16,
    'xrsb2_flag_excluded': np.uint16,
    'yaw_flip_flag': np.uint8,
}

GOES_NAME_MAPPING = {}  # we don't need to rename those

GOES_NECESSARY_FIELDS = ['xrsa_flux', 'xrsb_flux']

OMNI_TO_RAD_FIELDS = [('Lat_Angle_of_B_GSE', 'Lat_Angle_of_B_GSE_rad'),
                      ('Long_Angle_of_B_GSE', 'Long_Angle_of_B_GSE_rad'),
                      ('SW_Plasma_flow_long_angle', 'SW_Plasma_flow_long_angle_rad'),
                      ('SW_Plasma_flow_lat_angle', 'SW_Plasma_flow_lat_angle_rad'),
                      ('sigma_phi_V_degrees', 'sigma_phi_V_rad'), ('sigma_theta_V_degrees', 'sigma_theta_V_rad')]

OMNI_TYPE_MAPPING = {
    'YEAR': np.uint16,
    'DOY': np.uint16,
    'Hour': np.uint8,
    'Bartels_rotation_number': np.uint16,
    'ID_for_IMF_spacecraft': np.uint16,
    'ID_for_SW_Plasma_spacecraft': np.uint16,
    'num_points_IMF_averages': np.int16,
    'num_points_Plasma_averages': np.int16,
    'Scalar_B_nT': np.float32,
    'Vector_B_Magnitude_nT': np.float32,
    'Lat_Angle_of_B_GSE_rad': np.float32,
    'Long_Angle_of_B_GSE_rad': np.float32,
    'BX_nT_GSE_GSM': np.float32,
    'BY_nT_GSE': np.float32,
    'BZ_nT_GSE': np.float32,
    'BY_nT_GSM': np.float32,
    'BZ_nT_GSM': np.float32,
    'RMS_magnitude_nT': np.float32,
    'RMS_field_vector_nT': np.float32,
    'RMS_BX_GSE_nT': np.float32,
    'RMS_BY_GSE_nT': np.float32,
    'RMS_BZ_GSE_nT': np.float32,
    'SW_Plasma_Temperature_K': np.float32,
    'SW_Proton_Density_N_cm3': np.float32,
    'SW_Plasma_Speed_km_s': np.float32,
    'SW_Plasma_flow_long_angle_rad': np.float32,
    'SW_Plasma_flow_lat_angle_rad': np.float32,
    'Alpha_Prot_ratio': np.float32,
    'sigma_T_K': np.float32,
    'sigma_n_N_cm3': np.float32,
    'sigma_V_km_s': np.float32,
    'sigma_phi_V_rad': np.float32,
    'sigma_theta_V_rad': np.float32,
    'sigma_ratio': np.float32,
    'Flow_pressure': np.float32,
    'E_electric_field': np.float32,
    'Plasma_Beta': np.float32,
    'Alfen_mach_number': np.float32,
    'Magnetosonic_Mach_number': np.float32,
    'Quasy_Invariant': np.float32,
    'Kp_index': np.float32,
    'R_Sunspot_No': np.float32,
    'Dst_index_nT': np.float32,
    'ap_index_nT': np.float32,
    'f10.7_index': np.float32,
    'AE_index_nT': np.float32,
    'AL_index_nT': np.float32,
    'AU_index_nT': np.float32,
    'pc_index': np.float32,
    'Lyman_alpha': np.float32,
    'Proton_flux_>1_Mev_10^-3': np.float32,
    'Proton_flux_>2_Mev_10^-3': np.float32,
    'Proton_flux_>4_Mev_10^-3': np.float32,
    'Proton_flux_>10_Mev_10^-3': np.float32,
    'Proton_flux_>30_Mev_10^-3': np.float32,
    'Proton_flux_>60_Mev_10^-3': np.float32,
    'Flux_FLAG': np.int8
}

OMNI_NAME_MAPPING = {
    'f10.7_index': 'f10_7_index',
    'ap_index_nT': 'ap_index',
    'Kp_index': 'kp_index',
    'Dst_index_nT': 'dst_index',
    'AE_index_nT': 'ae_index',
    'BZ_nT_GSM': 'b_z',
    'BY_nT_GSM': 'b_y',
    'BX_nT_GSE_GSM': 'b_x',
    'E_electric_field': 'electric_field',
    'Flow_pressure': 'flow_pressure',
    'SW_Plasma_Speed_km_s': 'sw_plasma_speed',
    'SW_Proton_Density_N_cm3': 'sw_proton_density'
}

OMNI_NECESSARY_FIELDS = ['f10_7_index', 'ap_index', 'kp_index', 'dst_index', 'ae_index', 'b_z', 'b_y', 'b_x',
                         'electric_field', 'flow_pressure', 'sw_plasma_speed', 'sw_proton_density']

INITIAL_STATE_TO_RAD = ['Inclination (deg)', 'RAAN (deg)', 'Argument of Perigee (deg)', 'True Anomaly (deg)',
                        'Latitude (deg)', 'Longitude (deg)']

INITIAL_STATE_TYPE_MAPPING = {
    'File ID': np.int16,
    'Semi-major Axis (1000 km)': np.float64,
    'Eccentricity': np.float32,
    'Inclination (rad)': np.float32,
    'RAAN (rad)': np.float32,
    'Argument of Perigee (rad)': np.float32,
    'True Anomaly (rad)': np.float32,
    'Latitude (rad)': np.float32,
    'Longitude (rad)': np.float32,
    'Altitude (1000 km)': np.float64,
}

INITIAL_STATE_NAME_MAPPING = {
    'Semi-major Axis (1000 km)': 'a',
    'Eccentricity': 'e',
    'Inclination (rad)': 'i',
    'RAAN (rad)': 'RAAN',
    'Argument of Perigee (rad)': 'omega',
    'True Anomaly (rad)': 'nu',
    'File ID': 'file_id'
}

INITIAL_STATE_NECESSARY_FIELDS = ['file_id', 'a', 'e', 'i', 'RAAN', 'omega', 'nu']


class DataType(Enum):
    GOES = auto()
    OMNI2 = auto()
    SAT_DENSITY = auto()
    INITIAL_STATES = auto()


def list_nested_dir(path: Path, depth: int = 1) -> list:
    """Lists all files in the given directory up to given depth (-1 for infinite depth)."""
    assert path.is_dir()
    assert path.exists()

    if depth == 0:
        return list(map(lambda _p: path / _p, os.listdir(path)))

    file_paths = []
    for p in path.iterdir():
        if p.is_file():
            file_paths.append(p)
        else:
            file_paths.extend(list_nested_dir(p, depth=depth - 1))
    return file_paths


def prepare_df(df: pd.DataFrame,
               nan_mapping: dict[str, Any],
               type_mapping: dict[str, Any],
               name_mapping: dict[str, str]) -> pd.DataFrame:
    # fill nan values
    for key, value in nan_mapping.items():
        df[key] = df[key].fillna(value)

    return df.astype(type_mapping).rename(columns=name_mapping)


def prepare_goes_df(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=',', index_col='Timestamp', parse_dates=True)
    _convert_to_rad(df, 'roll_angle')

    df = prepare_df(df, GOES_NAN_MAPPING, GOES_TYPE_MAPPING, GOES_NAME_MAPPING)
    df.loc[df['xrsa_flag'] != 0, 'xrsa_flux'] = np.nan
    df.loc[df['xrsb_flag'] != 0, 'xrsb_flux'] = np.nan
    df = df.filter(GOES_NECESSARY_FIELDS)

    # important later for merging different data sources
    file_name = file_path.name
    df['file_id'] = int(file_name[5:10])
    return df


def prepare_omni_df(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=',', index_col='Timestamp', parse_dates=True)

    # improve numerical stability
    for (key, wanted_key) in OMNI_TO_RAD_FIELDS:
        _convert_to_rad(df, key)
        df.loc[df[key] > 7, key] = RAD_OUTLIER_VALUE  # these are outliers
        df.rename(columns={key: wanted_key}, inplace=True)

    # handle other outliers
    df.loc[df['Alpha_Prot_ratio'] > 1.0, 'Alpha_Prot_ratio'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['sigma_ratio'] > 1.0, 'sigma_ratio'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['Flow_pressure'] > 90, 'Flow_pressure'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['SW_Proton_Density_N_cm3'] > 900, 'SW_Proton_Density_N_cm3'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['f10.7_index'] > 500, 'f10.7_index'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['AE_index_nT'] > 5000, 'AE_index_nT'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['BX_nT_GSE_GSM'] > 150, 'BX_nT_GSE_GSM'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['BY_nT_GSM'] > 150, 'BY_nT_GSM'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['BZ_nT_GSM'] > 150, 'BZ_nT_GSM'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['E_electric_field'] > 200, 'E_electric_field'] = DEFAULT_OUTLIER_VALUE
    df.loc[df['SW_Plasma_Speed_km_s'] > 2000, 'SW_Plasma_Speed_km_s'] = DEFAULT_OUTLIER_VALUE

    for i in [1, 2, 4, 10, 30, 60]:
        key = f'Proton_flux_>{i}_Mev'
        df[key] = df[key] / 1e3
        df.loc[df[key] > 9e2, key] = DEFAULT_OUTLIER_VALUE
        df.rename(columns={key: f'Proton_flux_>{i}_Mev_10^-3'}, inplace=True)

    # there aren't so many nan values like with GOES
    df = prepare_df(df, {}, OMNI_TYPE_MAPPING, OMNI_NAME_MAPPING)
    df = df.filter(OMNI_NECESSARY_FIELDS)

    # important later for merging different data sources
    file_name = file_path.name
    df['file_id'] = int(file_name[6:11])
    return df


def prepare_sat_density_df(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=',',
        index_col='Timestamp',
        parse_dates=True,
    )

    df[df['Orbit Mean Density (kg/m^3)'] > 1] = DEFAULT_OUTLIER_VALUE
    df = df.astype(np.float32)
    df.rename(columns={'Orbit Mean Density (kg/m^3)': 'orbit_mean_density'}, inplace=True)

    file_name = file_path.name
    start_idx = file_name.index('-', 4) + 1  # could be improved with a regex...
    df['file_id'] = int(file_name[start_idx:start_idx + 5])

    return df


def prepare_initial_state_df(file_path: Path) -> pd.DataFrame:
    def _convert_to_rad_with_rename(_df, _key):
        _convert_to_rad(_df, _key)
        _df.rename(columns={_key: _key.replace('(deg)', '(rad)')}, inplace=True)

    def _scale_km_with_rename(_df, _key):
        _scale_km(_df, _key)
        _df.rename(columns={_key: _key.replace('(km)', '(1000 km)')}, inplace=True)

    df = pd.read_csv(
        file_path,
        sep=',',
        index_col='Timestamp',
        parse_dates=True,
    )

    for key in INITIAL_STATE_TO_RAD:
        _convert_to_rad_with_rename(df, key)

    _scale_km_with_rename(df, 'Altitude (km)')
    _scale_km_with_rename(df, 'Semi-major Axis (km)')

    # Catch obvious outliers
    df.loc[df['Longitude (rad)'] > 4, 'Longitude (rad)'] = RAD_OUTLIER_VALUE
    df.loc[df['Latitude (rad)'] > 4, 'Latitude (rad)'] = RAD_OUTLIER_VALUE
    df.loc[df['Altitude (1000 km)'] > 1e+6, 'Altitude (1000 km)'] = DEFAULT_OUTLIER_VALUE

    df = prepare_df(df, {}, INITIAL_STATE_TYPE_MAPPING, INITIAL_STATE_NAME_MAPPING)
    df = df.filter(INITIAL_STATE_NECESSARY_FIELDS)

    return df


def handle_file(file_path: Path, data_type: DataType) -> pd.DataFrame:
    if data_type == DataType.GOES:
        return prepare_goes_df(file_path)
    elif data_type == DataType.OMNI2:
        return prepare_omni_df(file_path)
    elif data_type == DataType.SAT_DENSITY:
        return prepare_sat_density_df(file_path)
    elif data_type == DataType.INITIAL_STATES:
        return prepare_initial_state_df(file_path)
    else:
        raise ValueError(f'Unknown data type {data_type}')


def merge_dfs(dfs: Iterator[pd.DataFrame], total: int, cache_size=CACHE_SIZE) -> pd.DataFrame:
    final_df = next(dfs)
    final_df.reset_index(inplace=True)
    cache = []
    for iteration, df in enumerate(tqdm(dfs, total=total - 1)):
        df.reset_index(inplace=True)
        cache.append(df)
        if iteration % cache_size == 0:
            final_df = pd.concat([final_df, *cache])
            cache = []

    final_df = pd.concat([final_df, *cache])
    final_df.set_index(['file_id', 'Timestamp'], inplace=True)
    return final_df


def handle_ds(raw_data_dir: Path, aggregated_file: Path,
              df_loading_func: Callable[[list[Path]], Iterator[pd.DataFrame]],
              search_recursion_depth: int = 1,
              save_to_file: bool = True) -> None | pd.DataFrame:
    file_paths = list_nested_dir(raw_data_dir, search_recursion_depth)
    df = merge_dfs(df_loading_func(file_paths), total=len(file_paths)).sort_index(level=['file_id', 'Timestamp'],
                                                                                  ascending=[True, True])
    if save_to_file:
        aggregated_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(aggregated_file)
        return None

    return df


# helper functions
def _convert_to_rad(_df, _key):
    _df[_key] = np.radians(_df[_key])
    return _df


def _scale_km(_df, _key):
    _df[_key] = _df[_key] / 1_000.0
    return _df


def _generate_loading_func(data_type: DataType) -> Callable[[list[Path]], Iterator[pd.DataFrame]]:
    return lambda paths: iter(
        map(lambda path: handle_file(path, data_type), paths))


def main(goes_raw_data_dir: Path, goes_aggregated_file: Path,
         omni_raw_data_dir: Path, omni_aggregated_file: Path,
         sat_dens_raw_data_dir: Path, sat_dens_aggregated_file: Path,
         in_st_raw_data_dir: Path, in_st_aggregated_file: Path):
    # sat density
    sat_density_loading_func = _generate_loading_func(DataType.SAT_DENSITY)
    handle_ds(sat_dens_raw_data_dir, sat_dens_aggregated_file,
              sat_density_loading_func)

    # initial states
    initial_state_loading_func = _generate_loading_func(DataType.INITIAL_STATES)
    handle_ds(in_st_raw_data_dir, in_st_aggregated_file,
              initial_state_loading_func)

    # omni
    omni_loading_func = _generate_loading_func(DataType.OMNI2)
    handle_ds(omni_raw_data_dir, omni_aggregated_file,
              omni_loading_func)

    # goes -> I am thinking about excluding this crappy df all together...
    # too many values are missing (i.e., 1/3 of the data is not good for 60d windows)
    if goes_raw_data_dir is not None:
        goes_loading_func = _generate_loading_func(DataType.GOES)
        handle_ds(goes_raw_data_dir, goes_aggregated_file,
                  goes_loading_func)


if __name__ == "__main__":
    DATA_DIR = Path('../../../data/')
    GOES_RAW_DATA_DIR = Path(DATA_DIR / 'raw/GOES/')
    GOES_AGGREGATED_FILE = Path(DATA_DIR / 'preprocessed/aggregated/goes.parquet')

    OMNI_RAW_DATA_DIR = Path(DATA_DIR / 'raw/OMNI2/')
    OMNI_AGGREGATED_FILE = Path(DATA_DIR / 'preprocessed/aggregated/omni2.parquet')

    SAT_DENS_RAW_DATA_DIR = Path(DATA_DIR / 'raw/SAT_DENSITY/')
    SAT_DENS_AGGREGATED_FILE = Path(DATA_DIR / 'preprocessed/aggregated/sat_density.parquet')

    IN_ST_RAW_DATA_DIR = Path(DATA_DIR / 'raw/INITIAL_STATES/')
    IN_ST_AGGREGATED_FILE = Path(DATA_DIR / 'preprocessed/aggregated/initial_states.parquet')

    main(GOES_RAW_DATA_DIR, GOES_AGGREGATED_FILE,
         OMNI_RAW_DATA_DIR, OMNI_AGGREGATED_FILE,
         SAT_DENS_RAW_DATA_DIR, SAT_DENS_AGGREGATED_FILE,
         IN_ST_RAW_DATA_DIR, IN_ST_AGGREGATED_FILE)
