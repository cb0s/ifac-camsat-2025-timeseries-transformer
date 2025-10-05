from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# setup of orekit is kinda special... to work, this module needs physics.setup.setup_orekit() to run first
from .orekit_helpers import setup_orekit

setup_orekit()
from .orekit_helpers import multiprocess_simulate

from orekit_jpype.pyhelpers import datetime_to_absolutedate
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.events import EclipseDetector, EventsLogger
from org.orekit.propagation.events.handlers import ContinueOnEvent
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.errors import OrekitException

# Define reference frame and bodies globally for efficiency
GCRF = FramesFactory.getGCRF()
# Use an Earth frame for the body shape definition in EclipseDetector
ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)  # Use IERS_2010 conventions, simple EOP
UTC = TimeScalesFactory.getUTC()
# Define Earth shape model (needed for EclipseDetector)
EARTH = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                         Constants.WGS84_EARTH_FLATTENING,
                         ITRF)  # Frame used for body shape
# Define Sun model
SUN = CelestialBodyFactory.getSun()
# Earth gravitational parameter
MU = Constants.WGS84_EARTH_MU

_RESULT_KEYS = [
    'period_s', 'time_umbra_s', 'time_penumbra_s', 'fraction_umbra', 'fraction_penumbra', 'fraction_sunlight',
    'penumbra_entry_nu', 'umbra_entry_nu', 'umbra_exit_nu', 'penumbra_exit_nu'
]


# public functions to call from outside
def calculate_single_orbit_eclipse_details(
        a: float, ecc: float, i: float, raan: float, omega: float, initial_nu: float, epoch: datetime,
        n_steps: int = 10_000, threshold: float = 1.0e-9, *_
) -> dict[str, float]:
    """
    Calculates eclipse fractions, times, and entry/exit true anomalies [rad].

    Includes interpolation using SciPy's brentq for precise transition points.
    Assumes satellite at perigee (nu=0) at epoch.

    Args:
        a (float): Semi-major axis (m).
        ecc (float): Eccentricity.
        i (float): Inclination [rad].
        raan (float): Right ascension of ascending node [rad].
        omega (float): Argument of perigee [rad].
        initial_nu (float): True anomaly [rad] at the epoch.
        epoch (AbsoluteDate): Epoch for the elements as Orekit AbsoluteDate.
        n_steps (int, optional): Number of time steps (default: 1_000)
        threshold (float, optional): Convergence threshold [sec] used by the internal root-finding algorithm that
                                     pinpoints the exact time of the event (default: 1.0e-8).

    Returns:
        Dictionary with period, times, fractions, and entry/exit true anomalies [rad]
        (e.g., 'penumbra_entry_nu_deg'). Angles are NaN if no eclipse occurs
        if a specific transition doesn't happen or if interpolation fails.
    """
    assert a * (1 - ecc) > Constants.WGS84_EARTH_EQUATORIAL_RADIUS

    try:
        # 1. define orbit
        orekit_date = datetime_to_absolutedate(epoch)
        initial_orbit = KeplerianOrbit(a, ecc, i, omega, raan, initial_nu, PositionAngleType.TRUE, GCRF, orekit_date, MU)

        period_s = initial_orbit.getKeplerianPeriod()
        assert np.isfinite(period_s) and period_s > 0

        # 2. create propagator with eclipse detectors and loggers and add them to the simulation
        propagator = KeplerianPropagator(initial_orbit)

        max_check_interval = period_s / n_steps
        penumbra_logger = EventsLogger()
        penumbra_detector = penumbra_logger.monitorDetector(
            EclipseDetector(SUN, Constants.SUN_RADIUS, EARTH) \
                .withMaxCheck(max_check_interval).withThreshold(threshold) \
                .withPenumbra().withHandler(ContinueOnEvent())
        )

        umbra_logger = EventsLogger()
        umbra_detector = umbra_logger.monitorDetector(
            EclipseDetector(SUN, Constants.SUN_RADIUS, EARTH) \
                .withMaxCheck(max_check_interval).withThreshold(threshold) \
                .withUmbra().withHandler(ContinueOnEvent())
        )

        propagator.addEventDetector(penumbra_detector)
        propagator.addEventDetector(umbra_detector)

        # 3. propagate for one period
        end_date = orekit_date.shiftedBy(period_s)
        _ = propagator.propagate(end_date)

        # 4. process results
        penumbra_events = penumbra_logger.getLoggedEvents()
        umbra_events = umbra_logger.getLoggedEvents()

        # initialize results
        transition_times = {'pen_entry': [], 'pen_exit': [], 'umb_entry': [], 'umb_exit': []}
        transition_nus = {'pen_entry': np.nan, 'pen_exit': np.nan, 'umb_entry': np.nan,
                          'umb_exit': np.nan}  # Store only first Nu


        # process penumbra events
        for event in penumbra_events:
            state = event.getState()
            event_time = state.getDate()
            is_increasing = event.isIncreasing()  # True if exiting penumbra -> Sunlight

            event_orbit = KeplerianOrbit(state.getOrbit())
            nu = event_orbit.getTrueAnomaly()

            if is_increasing:  # Exiting Penumbra (Pen -> Sun)
                transition_times['pen_exit'].append(event_time)
                if np.isnan(transition_nus['pen_exit']):
                    transition_nus['pen_exit'] = nu
            else:  # Entering Penumbra (Sun -> Pen)
                transition_times['pen_entry'].append(event_time)
                if np.isnan(transition_nus['pen_entry']):
                    transition_nus['pen_entry'] = nu

        # process umbra events
        for event in umbra_events:
            state = event.getState()
            event_time = state.getDate()
            is_increasing = event.isIncreasing()

            event_orbit = KeplerianOrbit(state.getOrbit())
            nu = event_orbit.getTrueAnomaly()

            if is_increasing:  # Exiting Umbra (Umb -> Pen)
                transition_times['umb_exit'].append(event_time)
                if np.isnan(transition_nus['umb_exit']):
                    transition_nus['umb_exit'] = nu
            else:  # Entering Umbra (Pen -> Umb)
                transition_times['umb_entry'].append(event_time)
                if np.isnan(transition_nus['umb_entry']):
                    transition_nus['umb_entry'] = nu

        # sort times for duration calculation
        for key in transition_times:
            transition_times[key].sort()

        # 5. calculate durations
        def _total_duration(_entry_list, _exit_list):
            # assert len(_entry_list) == len(_exit_list)
            _duration = 0.0
            for _entry, _exit in zip(_entry_list, _exit_list):
                # durationFrom gives signed duration, use abs or check order?
                # Assume exit always follows entry within one period propagation
                _duration = _exit.durationFrom(_entry)
                if _duration >= 0:  # Add only positive durations
                    _duration += _duration

            return _duration

        time_penumbra_s = _total_duration(transition_times['pen_entry'], transition_times['pen_exit'])
        time_umbra_s = _total_duration(transition_times['umb_entry'], transition_times['umb_exit'])

        # clamp to zero in case of float issues or very short umbra
        time_only_penumbra_s = max(0.0, time_penumbra_s - time_umbra_s)

        # calculate fractions
        fraction_umbra = time_umbra_s / period_s if period_s > 0 else 0.0
        fraction_penumbra = time_only_penumbra_s / period_s if period_s > 0 else 0.0
        fraction_umbra = max(0.0, min(1.0, fraction_umbra))
        fraction_penumbra = max(0.0, min(1.0, fraction_penumbra))
        fraction_sunlight = max(0.0, min(1.0, 1.0 - fraction_umbra - fraction_penumbra))

        # handle the edge case where there is no eclipse, the eclipse is too short to capture, or the orbit is only in
        # (umbra/)penumbra
        umbra_nu_entry = transition_nus['umb_entry']
        umbra_nu_exit = transition_nus['umb_exit']
        if np.isnan(umbra_nu_entry):
            if np.isnan(umbra_nu_exit):
                umbra_nu_exit = 0.0
            umbra_nu_entry = umbra_nu_exit
        elif np.isnan(umbra_nu_exit):
            umbra_nu_exit = umbra_nu_entry

        penumbra_nu_entry = transition_nus['pen_entry']
        penumbra_nu_exit = transition_nus['pen_exit']
        if np.isnan(penumbra_nu_entry):
            if np.isnan(penumbra_nu_exit):
                penumbra_nu_exit = 0.0
            penumbra_nu_entry = penumbra_nu_exit
        elif np.isnan(penumbra_nu_exit):
            penumbra_nu_exit = penumbra_nu_entry

        results = {
            'period_s': period_s,
            'time_umbra_s': time_umbra_s,
            'time_penumbra_s': time_only_penumbra_s,  # Report only penumbra time
            'fraction_umbra': fraction_umbra,
            'fraction_penumbra': fraction_penumbra,  # Report only penumbra fraction
            'fraction_sunlight': fraction_sunlight,
            'penumbra_entry_nu': penumbra_nu_entry,
            'umbra_entry_nu': umbra_nu_entry,
            'umbra_exit_nu': umbra_nu_exit,
            'penumbra_exit_nu': penumbra_nu_exit,
        }
    except OrekitException as e:
        raise ValueError(e)

    return results


def calculate_eclipse_details(
        df: pd.DataFrame, n_steps: int = 10_000, threshold: float = 1.0e-9,
        n_workers: int | None = None, mode='relaxed') -> pd.DataFrame:
    """
    Calculates eclipse fractions for multiple orbits defined in a pandas DataFrame,
    using multiprocessing for parallel execution.

    Assumes standard units for DataFrame columns: 'a'[km], 'e', 'i'[rad], 'RAAN'[rad], 'omega'[rad].
    DataFrame index must be a pandas DatetimeIndex (epoch for each orbit).

    Args:
        df (pd.DataFrame): DataFrame with required orbital elements and DatetimeIndex.
        n_steps (int): Number of steps for orbit propagation per orbit.
        threshold (float): Convergence threshold [sec] used by the internal root-finding algorithm that pinpoints the
                           exact time of the event (default: 1.0e-8).
        n_workers (int, optional): Number of worker processes. Defaults to os.cpu_count().
        mode (str): either 'relaxed' (warning on NaN values) or 'strict' (fail on Nan values). Defaults to 'relaxed'.

    Returns:
        pd.DataFrame: Original DataFrame with added columns for eclipse results.
                      Rows where calculation fails will have NaN results.
    """
    return multiprocess_simulate(df, "eclipse details", threshold, _worker_fn_simulate,
                                 n_steps, n_workers, mode)


def _worker_fn_simulate(orbit_arguments: tuple[float, float, float, float, float, float, datetime, int, float, Any]):
    if orbit_arguments is None:
        result_dict = {key: np.nan for key in _RESULT_KEYS}
        return result_dict

    # Unpack arguments
    a_km, ecc, i, raan, omega, nu, epoch, n_steps, threshold, *args = orbit_arguments
    assert all(np.isfinite([a_km, ecc, i, raan, omega, n_steps, threshold]))
    assert epoch is not None and isinstance(epoch, datetime)

    result = calculate_single_orbit_eclipse_details(
        a_km * 1_000_000, ecc, i, raan, omega, nu, epoch, n_steps, threshold, *args
    )

    return result
