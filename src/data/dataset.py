import os
import random
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

# needed for the random shuffle algorithm -> we always want to have the same order and don't mix up training and
# validation sets
SPLIT_SEED = 0xC0FFEEBABE
TRAINING_PERCENTAGE = 0.85 # 0.75   # we need as much training data as possible for somewhat good results...
MAX_CACHE_SIZE = 10_000  # should easily be enough -> we have 8119 elements in total ;)

N_FEATURES = 904
N_PREDICTION_HORIZON = 432

INITIAL_CALCULATION_BATCH_SIZE = 256
CACHE_REFINE_INITIAL_FEATURES = True  # we don't want to have to compute this every time we train the refinement model


def apply_z_transform(array, axis=-1,
                      mean: np.ndarray = None,
                      std: np.ndarray = None) -> tuple[np.ndarray, float | np.ndarray, float | np.ndarray]:
    if mean is None or std is None:
        mean = np.mean(array, axis=axis)
        std = np.std(array, axis=axis)

    return (array - mean) / std, mean, std


def reverse_z_transform(array: np.ndarray, mean: float | np.ndarray, std: float | np.ndarray) -> np.ndarray:
    return array * std + mean


class RawTrainingDataset:
    """ A class that stores and loads raw training data into and from npz files. Also handles z-transforming. """

    def __init__(
            self,
            features: np.ndarray,  # expected in [n_samples, n_features]
            true_values: np.ndarray,  # expected in [n_samples, n_true_values]

            # if one is provided, all must be provided
            feature_mean: np.ndarray | None = None,
            feature_std: np.ndarray | None = None,
            true_value_mean: np.ndarray | None = None,
            true_value_std: np.ndarray | None = None,

            needs_z_transform: bool = True,

            **other_fields: Any  # only caveat: must be pickle-able!
    ):
        assert features.shape[0] == true_values.shape[0]

        if needs_z_transform and any(
                inp is None for inp in [feature_mean, feature_std, true_value_mean, true_value_std]):
            assert feature_mean is None and feature_std is None and true_value_mean is None and true_value_std is None
            self._features, self._feature_means, self._features_stds = apply_z_transform(features, axis=0)
            self._true_values, self._tv_means, self._tv_stds = apply_z_transform(true_values, axis=0)
        else:
            self._features = features
            self._feature_means = feature_mean
            self._features_stds = feature_std
            self._true_values = true_values
            self._tv_means = true_value_mean
            self._tv_stds = true_value_std

        self._other_fields = other_fields

    def save(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        assert file_path.parent.exists()

        np.savez_compressed(file_path,
                            x=self._features, y=self._true_values,
                            x_mean=self._feature_means, x_std=self._features_stds,
                            y_mean=self._tv_means, y_std=self._tv_stds,
                            **self._other_fields)

    @classmethod
    def load(cls, file_path: str | Path, auto_reverse_z: bool = False) -> 'RawTrainingDataset':
        file_path = Path(file_path)
        assert isinstance(file_path, Path)
        assert file_path.is_file()
        assert file_path.exists()

        data: dict[str, np.ndarray] = dict(np.load(file_path))

        if auto_reverse_z:
            x = reverse_z_transform(data.pop('x'), data.pop('x_mean'), data.pop('x_std'))
            y = reverse_z_transform(data.pop('y'), data.pop('y_mean'), data.pop('y_std'))
            return cls(x, y, **data, needs_z_transform=False)
        else:
            return cls(
                data.pop('x'), data.pop('y'),
                data.pop('x_mean'), data.pop('x_std'), data.pop('y_mean'), data.pop('y_std'),
                **data, needs_z_transform=False
            )

    # getters
    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def true_values(self) -> np.ndarray:
        return self._true_values

    @property
    def feature_means(self) -> float | np.ndarray:
        return self._feature_means

    @property
    def feature_stds(self) -> float | np.ndarray:
        return self._features_stds

    @property
    def true_value_means(self) -> float | np.ndarray:
        return self._tv_means

    @property
    def true_value_stds(self) -> float | np.ndarray:
        return self._tv_stds

    @property
    def other_fields(self) -> dict[str, Any]:
        return self._other_fields


class TrainingMode(Enum):
    TRAIN = 0
    VALIDATION = 1


class TrainingDataset(Dataset):
    """ Small wrapper around the raw training dataset to provide a torch-friendly interface. """

    def __init__(
            self,
            raw_dataset: RawTrainingDataset,
            input_noise_std: float = 0.0,
            output_noise_std: float = 0.0,
            training_mode: TrainingMode = TrainingMode.TRAIN,
            device: str | torch.device = 'cpu',
            n_features: int = 904,  # at least for the residual model
    ):
        self._raw_dataset = raw_dataset
        self._input_noise_std = input_noise_std
        self._output_noise_std = output_noise_std
        self._training_mode = training_mode
        self._device = device
        self._n_features = n_features

        train_length = round(self._raw_dataset.features.shape[0] * TRAINING_PERCENTAGE)
        if training_mode == TrainingMode.TRAIN:
            self._length = train_length
            self._index_mapping = list(range(len(self)))
        else:
            self._length = round(self._raw_dataset.features.shape[0] * (1 - TRAINING_PERCENTAGE))
            self._index_mapping = list(range(train_length, train_length + len(self)))

        # we need to shuffle the data always in the same way for splitting between training and validation sets
        self._index_mapping = list(range(len(self)))
        random.Random(SPLIT_SEED).shuffle(self._index_mapping)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        feature_tensor, ground_truth_tensor = self._get_raw_item(idx)

        # these however must be created each time
        if self._input_noise_std > 0:
            feature_tensor += torch.randn_like(feature_tensor) * self._input_noise_std

        if self._output_noise_std > 0:
            ground_truth_tensor += torch.randn_like(ground_truth_tensor) * self._output_noise_std

        return feature_tensor, ground_truth_tensor

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _get_raw_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 <= idx < len(self)
        idx = self._index_mapping[idx]
        # as we use caching, the tensors are actually only created once
        feature_tensor = torch.tensor(self._raw_dataset.features[idx], dtype=torch.float32, device=self._device)
        ground_truth_tensor = torch.tensor(self._raw_dataset.true_values[idx], dtype=torch.float32, device=self._device)

        return feature_tensor, ground_truth_tensor

    @property
    def training_mode(self) -> TrainingMode:
        return self._training_mode

    @property
    def raw_dataset(self) -> RawTrainingDataset:
        return self._raw_dataset

    @property
    def input_noise_std(self) -> float:
        return self._input_noise_std

    @property
    def output_noise_std(self) -> float:
        return self._output_noise_std

    @property
    def device(self) -> str | torch.device:
        return self._device

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def get_index_mapping(self) -> list[int]:
        return self._index_mapping


class RefinementDataset(TrainingDataset):
    """ Small wrapper around the training dataset to provide a torch-friendly interface for the refinement model. """

    def __init__(
            self,
            raw_dataset: RawTrainingDataset,
            residual_model: nn.Module,
            initial_inputs_mean: float | np.ndarray,
            initial_inputs_std: float | np.ndarray,
            input_noise_std: float = 0.0,
            output_noise_std: float = 0.0,
            training_mode: TrainingMode = TrainingMode.TRAIN,
            device: str | torch.device = 'cpu',
            n_static_features: int = 432,  # at least for the residual model
            n_dynamic_features: int = 104,
    ):
        super().__init__(
            raw_dataset=raw_dataset, input_noise_std=input_noise_std, output_noise_std=output_noise_std,
            training_mode=training_mode, device=device, n_features=n_dynamic_features
        )

        self._n_static_features = n_static_features
        self._initial_inputs = torch.tensor(apply_z_transform(
            raw_dataset.other_fields['main_features'], axis=0, mean=initial_inputs_mean, std=initial_inputs_std
        )[0], dtype=torch.float32, device=self._device)
        self._initial_inputs_mean = initial_inputs_mean
        self._initial_inputs_std = initial_inputs_std

        self._model = residual_model
        length = self._initial_inputs.shape[0]
        self._initial_features = torch.empty((length, N_PREDICTION_HORIZON, 1), dtype=torch.float32,
                                             device=self._device)

        if CACHE_REFINE_INITIAL_FEATURES and os.path.exists('initial_features.pt'):
            self._initial_features = torch.load('initial_features.pt')
            return

        for i in range(0, length - INITIAL_CALCULATION_BATCH_SIZE, INITIAL_CALCULATION_BATCH_SIZE):
            self._initial_features[i:i + INITIAL_CALCULATION_BATCH_SIZE] = self._model.predict(
                self._initial_inputs[i:i + INITIAL_CALCULATION_BATCH_SIZE]
            )

        self._initial_features[length - INITIAL_CALCULATION_BATCH_SIZE:] = self._model.predict(
            self._initial_inputs[length - INITIAL_CALCULATION_BATCH_SIZE:]
        )

        if CACHE_REFINE_INITIAL_FEATURES:
            torch.save(self._initial_features, 'initial_features.pt')

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_tensor, ground_truth_tensor, initial_inputs_tensor = self._get_raw_item(idx)

        # these however must be created each time
        if self._input_noise_std > 0:
            feature_tensor += torch.randn_like(feature_tensor) * self._input_noise_std
            # This is an interesting question: should we add noise to the raw input of the residual model, or maybe to
            # the output of it? Or both? Or add output noise to the output of the residual model? We only add it to the
            # input of the residual model for now. Which introduced new noise for each training call but also makes us
            # recalculate the static features each time.
            initial_inputs_tensor += torch.randn_like(initial_inputs_tensor) * self._input_noise_std

        if self._output_noise_std > 0:
            ground_truth_tensor += torch.randn_like(ground_truth_tensor) * self._output_noise_std

        return feature_tensor, ground_truth_tensor, initial_inputs_tensor

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _get_raw_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert 0 <= idx < len(self)
        idx = self._index_mapping[idx]
        # as we use caching, the tensors are actually only created once
        feature_tensor = torch.tensor(self._raw_dataset.features[idx], dtype=torch.float32, device=self._device)
        ground_truth_tensor = torch.tensor(self._raw_dataset.true_values[idx], dtype=torch.float32, device=self._device)

        return feature_tensor, ground_truth_tensor, self._initial_features[idx]

    @property
    def n_static_features(self):
        return self._n_static_features


# because pickling functions is stupid in python -.-
class RefinementCollator:
    def __init__(self, model: nn.Module):
        super().__init__()
        # Store the model. PyTorch modules generally handle pickling correctly.
        self.model = model

    @torch.no_grad()
    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unzip and stack the batch elements
        features_list, ground_truths_list, initial_inputs_list = zip(*batch)

        features = torch.stack(features_list, dim=0)
        ground_truths = torch.stack(ground_truths_list, dim=0)

        initial_features = self.model.predict(initial_inputs_list)

        return features, ground_truths, initial_features
