# implements NRLMSISE2.1 model
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# setup of orekit is kinda special... to work, this module needs physics.setup.setup_orekit() to run first
from .orekit_helpers import setup_orekit

setup_orekit()
from .orekit_helpers import multiprocess_simulate

from orekit_jpype.pyhelpers import datetime_to_absolutedate
from org.orekit.errors import OrekitException
from org.orekit.frames import FramesFactory
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.utils import Constants
from pymsis import msis, Variable

_RESULT_KEYS = []
PREDICTION_HORIZON = 432


def simulate_msis_density(
        a: float, ecc: float, i: float, raan: float, omega: float, initial_nu: float, epoch: datetime,
        n_samples: int = 384, *args
) -> dict[str, float]:
    *_, f10_7_index, ap_index = args
    assert isinstance(f10_7_index, pd.Series)
    assert isinstance(ap_index, pd.Series)

    try:
        # 1. Define the initial orbit
        initial_date = datetime_to_absolutedate(epoch)
        frame = FramesFactory.getEME2000()

        orbit = KeplerianOrbit(
            a, ecc, i, omega, raan, initial_nu,
            PositionAngleType.TRUE, frame,
            initial_date,
            Constants.EIGEN5C_EARTH_MU,
        )

        # 2. Create propagator and Earth model
        propagator = KeplerianPropagator(orbit)
        earth = ReferenceEllipsoid.getWgs84(frame)

        # 3. We only need latitude, longitude, and altitude for MSIS calculation later
        latitudes: list[np.ndarray] = []
        longitudes: list[np.ndarray] = []
        altitudes: list[np.ndarray] = []

        # 4. Compute orbital period and sampling interval
        period = orbit.getKeplerianPeriod()
        dt = period / n_samples  # seconds between samples

        # 5. Simulate orbit
        # we assume the positions can be kept constant over the prediction horizon, otherwise outer loop over
        # PREDICTION_HORIZON would be needed
        for i in range(n_samples):
            date = initial_date.shiftedBy(i * dt)
            pv = propagator.propagate(date).getPVCoordinates()
            pos = pv.getPosition()

            geo = earth.transform(pos, frame, initial_date)
            latitudes.append(np.degrees(geo.getLatitude()))
            longitudes.append(np.degrees(geo.getLongitude()))
            altitudes.append(geo.getAltitude() / 1000.0)

        # 6. Calculate MSIS density like we have n_sample satellites distributed in the same orbit
        time_delta = timedelta(minutes=10)
        dates = [f10_7_index.index[-1] for _ in range(n_samples)]
        for _ in range(1, PREDICTION_HORIZON):
            dates.extend((dates[-1] + time_delta for _ in range(n_samples)))

        resampled_f107 = f10_7_index.resample('D').mean()
        f107 = [resampled_f107.iloc[-1]] * n_samples * PREDICTION_HORIZON
        f107a = [f10_7_index.mean()] * n_samples * PREDICTION_HORIZON
        ap = [[
            ap_index.iloc[-24:].mean(),
            ap_index.iloc[-1],
            ap_index.iloc[-4],
            ap_index.iloc[-7],
            ap_index.iloc[-10],
            ap_index.iloc[-34:-13].mean(),
            ap_index.iloc[-58:-37].mean()
        ]] * n_samples * PREDICTION_HORIZON

        prepared_lats = latitudes * PREDICTION_HORIZON
        prepared_lons = longitudes * PREDICTION_HORIZON
        prepared_alts = altitudes * PREDICTION_HORIZON

        assert not np.isnan(prepared_lats).any()
        assert not np.isnan(prepared_lons).any()
        assert not np.isnan(prepared_alts).any()
        if np.isnan(f107).any():
            f107 = None
        if np.isnan(f107a).any():
            f107a = None
        if np.isnan(ap).any():
            ap = None

        outputs = msis.calculate(
            dates,
            prepared_lons, prepared_lats, prepared_alts,
            f107,
            f107a,  # not 81 days, but 60 days... Hopefully, this is enough
            ap,  # good enough? -> should be the daily mean value from the day before
            geomagnetic_activity=-1  # we use our own values, not the ones calculated automatically... is this ok?
        ).squeeze()  # usually given in [N_dates, 1, 1, 11] format, but we don't need the extra dimensions

        densities = outputs[:, Variable.MASS_DENSITY].reshape(PREDICTION_HORIZON, n_samples)  # mass density in kg/m^3
        if np.isnan(densities).any():
            print(f"something weng wrong with {epoch}")

            # try again with downloaded indices
            outputs = msis.calculate(
                dates,
                prepared_lons, prepared_lats, prepared_alts
            )
            densities = outputs[:, Variable.MASS_DENSITY].reshape(PREDICTION_HORIZON, n_samples)

        # Enforce density floor and ceiling
        # min value of sat_density DS = 2.108e-14 max value of sat_density DS = 1.759e-11
        densities = densities.mean(axis=1).clip(5e-15, 1e-10)
    except OrekitException as e:
        raise ValueError(e)

    return {f"t{i + 1}": float(densities[i]) for i in range(PREDICTION_HORIZON)}


def get_nrlm_densities(
        initial_states_df: pd.DataFrame,
        omni_df: pd.DataFrame,
        n_steps: int = 384, _: float = 0.0,  # simulation related (orekit)
        n_workers: int | None = None, mode='relaxed'  # processing related (multiprocessing)
) -> pd.DataFrame:
    return multiprocess_simulate(
        initial_states_df,
        "NRLMSISE2.1 density calculation",
        _,
        _worker_fn_simulate,
        n_steps,
        n_workers,
        mode,
        omni_df,
        ['f10_7_index', 'ap_index']
    )


def _worker_fn_simulate(orbit_arguments: tuple[float, float, float, float, float, float, datetime, int, float, Any]):
    if orbit_arguments is None:
        result_dict = {key: np.nan for key in _RESULT_KEYS}
        return result_dict

    # Unpack arguments
    a_e3km, ecc, i, raan, omega, nu, epoch, n_steps, threshold, *args = orbit_arguments
    assert all(np.isfinite([a_e3km, ecc, i, raan, omega, n_steps, threshold]))
    assert epoch is not None and isinstance(epoch, datetime)

    result = simulate_msis_density(
        a_e3km * 1_000_000, ecc, i, raan, omega, nu, epoch, n_steps, threshold, *args
    )

    return result
