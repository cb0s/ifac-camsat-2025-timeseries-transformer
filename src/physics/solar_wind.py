import numpy as np
import pandas as pd

_NECESSARY_KEYS = [
    'b_y', 'b_z',  # necessary for Sin/CosClockAngle
]


def calculate_solar_wind_fields(df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame)
    assert all(key in df.keys() for key in _NECESSARY_KEYS)

    theta = np.atan2(df['b_y'], df['b_z'])
    sin_clock_angle, cos_clock_angle = np.sin(theta), np.cos(theta)

    result_df = pd.DataFrame(
        {
            'sin_clock_angle': sin_clock_angle, 'cos_clock_angle': cos_clock_angle,
        },
        index=df.index)

    return result_df
