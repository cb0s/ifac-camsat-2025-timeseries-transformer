from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from apexpy import Apex
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time

# setup of orekit is kinda special... to work, this module needs physics.setup.setup_orekit() to run first
from src.physics.orekit_helpers import setup_orekit

setup_orekit()
from .orekit_helpers import multiprocess_simulate

from orekit_jpype.pyhelpers import datetime_to_absolutedate
from org.orekit.errors import OrekitException
from org.orekit.frames import FramesFactory
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.time import TimeScalesFactory
from org.orekit.utils import Constants

_UTC = TimeScalesFactory.getUTC()
_RESULT_KEYS = [
    "mlat_max",
    "mlat_min",
    "mlat_mean",
    "mlat_std",
    "mlat_fraction_above_60",  # we are using a default value of 60 here, if a custom fraction is used, this is wrong
    "mlt_mean",
    "mlt_std",
    "mlt_max",
    "mlt_min",
    "sza_mean",
    "sza_std",
    "sza_max"
]


def sample_orbit_mlat_mlt_sza(
        a: float, ecc: float, i: float, raan: float, omega: float, initial_nu: float, timestamp: datetime,
        n_samples: int = 256, mlat_high_lat_threshold: float = 60.0, *_
):
    try:
        # 1. Define the initial orbit
        initial_date = datetime_to_absolutedate(timestamp)
        frame = FramesFactory.getEME2000()

        orbit = KeplerianOrbit(
            a,
            ecc,
            i,
            omega,
            raan,
            initial_nu,
            PositionAngleType.TRUE,
            frame,
            initial_date,
            Constants.EIGEN5C_EARTH_MU,
        )

        # 2. Create propagator and Earth model
        propagator = KeplerianPropagator(orbit)
        earth = ReferenceEllipsoid.getWgs84(frame)

        # 3. Compute orbital period and sampling interval
        period = orbit.getKeplerianPeriod()
        dt = period / n_samples  # seconds between samples

        # 4. Prepare Apex for MLAT conversion
        apex = Apex(date=timestamp)

        mlat_values = []
        mlt_values = []
        sza_values = []

        for i in range(n_samples):
            date = initial_date.shiftedBy(i * dt)
            pv = propagator.propagate(date).getPVCoordinates()
            pos = pv.getPosition()

            # only the position should change -> we are simulating as if the time would stand still
            geo = earth.transform(pos, frame, initial_date)
            lat = np.degrees(geo.getLatitude())
            lon = np.degrees(geo.getLongitude())
            alt_km = geo.getAltitude() / 1000.0

            mlat, mlong = apex.convert(lat, lon, 'geo', 'apex', height=alt_km)
            mlat_values.append(abs(mlat))  # absolute MLAT
            mlt_values.append(apex.mlon2mlt(mlong, timestamp))

            # Astropy time & location
            pytime = Time(initial_date.toString(), format='isot', scale='utc')
            location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt_km * u.km)

            altaz = AltAz(obstime=pytime, location=location)
            sun = get_sun(pytime).transform_to(altaz)

            sza = 90.0 - sun.alt.deg  # Zenith angle
            sza_values.append(sza)

        mlat_values = np.array(mlat_values)
        mlt_values = np.sin(np.array(mlt_values) / 12.0 * np.pi)  # we want a value between 0 and 1
        sza_values = np.radians(np.array(sza_values))
    except OrekitException as e:
        raise ValueError(e)

    return {
        "mlat_max": np.max(mlat_values),
        "mlat_min": np.min(mlat_values),
        "mlat_mean": np.mean(mlat_values),
        "mlat_std": np.std(mlat_values),
        f"mlat_fraction_above_{mlat_high_lat_threshold}": \
            (mlat_values > mlat_high_lat_threshold).sum() / len(mlat_values),
        # "mlat_all": mlat_values,  # optional: full array
        "mlt_mean": np.mean(mlt_values),
        "mlt_std": np.std(mlt_values),
        "mlt_max": np.max(mlt_values),
        "mlt_min": np.min(mlt_values),
        # "mlt_all": mlt_values,    # optional: full array
        "sza_mean": np.mean(sza_values),
        "sza_std": np.std(sza_values),
        "sza_max": np.max(sza_values),
        # "sza_all": sza_values,    # optional: full array
    }


def calculate_mlat_mlt_sza(
        df: pd.DataFrame, n_steps: int = 256, mlat_high_lat_threshold: float = 60.0,
        n_workers: int | None = None, mode='relaxed') -> pd.DataFrame:
    """
    Calculates Magnetic Apex Latitude (MLAT), Magnetic Local Time (MLT), and Solar Zenith Angle (SZA)
    for multiple orbits defined in a pandas DataFrame, using multiprocessing for parallel execution.

    Assumes standard units for DataFrame columns: 'a'[km], 'e', 'i'[rad], 'RAAN'[rad], 'omega'[rad], 'nu'[rad].
    DataFrame index must be a pandas DatetimeIndex (epoch for each orbit).

    Args:
        df (pd.DataFrame): DataFrame with required orbital elements and DatetimeIndex.
        n_steps (int): Number of steps for orbit propagation per orbit.
        mlat_high_lat_threshold (float): Latitude Threshold for which the fraction of values is calculated.
        n_workers (int, optional): Number of worker processes. Defaults to os.cpu_count().
        mode (str): either 'relaxed' (warning on NaN values) or 'strict' (fail on Nan values). Defaults to 'relaxed'.

    Returns:
        pd.DataFrame: Original DataFrame with added columns for eclipse results.
                      Rows where calculation fails will have NaN results.
    """
    return multiprocess_simulate(df, 'MLAT & SZA', mlat_high_lat_threshold, _worker_fn_sample,
                                 n_steps, n_workers, mode)


def _worker_fn_sample(orbit_arguments: tuple[float, float, float, float, float, float, datetime, int, float, Any]):
    if orbit_arguments is None:
        result_dict = {key: np.nan for key in _RESULT_KEYS}
        return result_dict

    # Unpack arguments
    a_km, ecc, i, raan, omega, nu, epoch, n_steps, threshold, *args = orbit_arguments
    assert all(np.isfinite([a_km, ecc, i, raan, omega, n_steps, threshold]))
    assert epoch is not None and isinstance(epoch, datetime)

    result = sample_orbit_mlat_mlt_sza(
        a_km * 1_000_000, ecc, i, raan, omega, nu, epoch, n_steps, threshold, *args
    )

    return result
