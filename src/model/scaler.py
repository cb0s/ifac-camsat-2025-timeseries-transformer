from pathlib import Path

import numpy as np


def z_transform(data, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=axis)
    std = data.std(axis=axis)
    return (data - mean) / std, mean, std


def reverse_z_transform(data, mean, std) -> np.ndarray:
    return data * std + mean


class ZTransform:
    def __init__(self, data: np.ndarray = None,
                 mean: np.ndarray = None, std: np.ndarray = None, axis: int | tuple[int, ...] = 0):
        self._axis = axis

        if mean is None or std is None:
            assert data is not None, 'Either mean and std, or data must be provided'
            mean = data.mean(axis=self._axis)
            std = data.std(axis=self._axis)

        self._mean = mean
        self._std = std

    def save(self, path: Path | str):
        to_save = {
            'mean': self._mean,
            'std': self._std,
        }
        np.savez_compressed(path, **to_save)

    @classmethod
    def load(cls, path: Path | str, axis: int = 0):
        file_content = np.load(path)
        return cls(mean=file_content['mean'], std=file_content['std'], axis=axis)

    def z_transform(self, data) -> np.ndarray:
        return (data - self._mean) / self._std

    def reverse_z_transform(self, data) -> np.ndarray:
        return data * self._std + self._mean
