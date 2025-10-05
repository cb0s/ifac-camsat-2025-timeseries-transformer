import logging
import multiprocessing
from datetime import datetime
from typing import Callable, Any

import jpype
import numpy as np
import orekit_jpype
import pandas as pd
from tqdm import tqdm


def setup_orekit():
    if not jpype.isJVMStarted():
        orekit_jpype.initVM()
        from orekit_jpype.pyhelpers import setup_orekit_curdir
        setup_orekit_curdir(from_pip_library=True)


def multiprocess_simulate(df: pd.DataFrame, job_name: str, threshold: float,
                          simulation_fn: Callable[
                              [tuple[float, float, float, float, float, float, datetime, int, float, Any]], dict[
                                  str, float]],
                          n_steps: int = 1_000, n_workers: int | None = None,
                          mode='relaxed',
                          second_source: pd.DataFrame | None = None,
                          additional_cols: list[str] | None = None):
    logging.info(f"Calculating {job_name}...")

    # 1. input check
    _required_cols = ['a', 'e', 'i', 'RAAN', 'omega', 'nu']
    assert isinstance(df, pd.DataFrame)
    assert set(_required_cols).issubset(df.columns)

    if additional_cols is None:
        additional_cols = []

    # 2. prepare execution pool for multiprocessing
    tasks = []
    for index, row in df[_required_cols].iterrows():
        if not all(np.isfinite(row[column]) for column in _required_cols):
            error = f"NaN values found at time {index}"
            if mode == 'strict':
                raise ValueError(error)

            tasks.append(None)
            logging.warning(" > " + error)
            continue

        # we don't want timezone info
        if index[1].tzinfo is not None and index[1].tzinfo.utcoffset(index) is not None:
            epoch_dt = index[1].tz_convert('UTC').to_pydatetime()
        else:
            epoch_dt = index[1].to_pydatetime()

        task_args = (
            row['a'], row['e'], row['i'], row['RAAN'], row['omega'], row['nu'],  # kepler elements
            epoch_dt,  # time info
            n_steps, threshold,
            # when a second column is used we assume that the 2 dataframes are connected through their respective first
            # index (i.e. file_id)
            *(second_source.loc[index[0]][key] for key in additional_cols)
        )
        tasks.append(task_args)

    # 3. execute in threadpool with multithreading support
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    logging.info(f" > Using {n_workers} workers.")
    with multiprocessing.get_context('spawn').Pool(n_workers) as processing_pool:
        results_iterator = processing_pool.imap(simulation_fn, tasks, chunksize=max(1, len(tasks) // (n_workers * 4)))
        results_list = list(tqdm(results_iterator, total=len(tasks), desc=f" > Processing orbits for {job_name}"))

    # 4. process results
    assert len(results_list) == len(tasks)
    results_df = pd.DataFrame(results_list, index=df.index)

    logging.info(" > Finished calculating eclipse details!")
    return results_df
