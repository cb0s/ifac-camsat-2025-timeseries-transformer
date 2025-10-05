import numpy as np
import torch
from torch import nn


class ODRMSELoss(nn.Module):
    """
    Custom PyTorch Loss function based on the Orbital Density RMSE (OD-RMSE) metric.
    Minimizes the weighted ratio of the test RMSE to the baseline RMSE.
    """

    def __init__(self, total_duration: float, times: torch.Tensor, min_weight_epsilon: float = 1e-5,
                 numerical_stability_delta: float = 1e-6):
        """
        Initializes the ODRMSE Loss function.

        Args:
            total_duration (float): The total duration of the evaluation period in seconds (T).
            times (torch.Tensor): The time points corresponding to each timestep, in seconds from the start.
                                  Expected shape (timesteps,).
            min_weight_epsilon (float): The small value the weight approaches at the end of the period (epsilon).
                                        Must be between 0 and 1. Default is 1e-5.
            numerical_stability_delta (float): A small value added to the denominator of the RMSE ratio for stability.
                                             Default is 1e-6.
        """
        super().__init__()
        self.total_duration = float(total_duration)
        self.times = times
        self.min_weight_epsilon = float(min_weight_epsilon)
        self.numerical_stability_delta = float(numerical_stability_delta)

        if self.total_duration <= 0:
            raise ValueError("Total duration must be positive.")
        if self.min_weight_epsilon <= 0 or self.min_weight_epsilon >= 1:
            raise ValueError("Minimum weight epsilon must be between 0 and 1.")
        if self.numerical_stability_delta < 0:
            raise ValueError("Numerical stability delta cannot be negative.")

        # Calculate decay constant gamma
        # Use torch.tensor for gamma so it's registered as part of the module state if needed,
        # though it's a constant here. Using numpy and converting is fine too.
        self.gamma = -np.log(self.min_weight_epsilon) / self.total_duration
        # self.gamma = torch.tensor(self.gamma, dtype=torch.float32) # Optional: store as tensor

        # Calculate weights for each timestep
        self.weights = torch.exp(-self.gamma * self.times)  # Shape (timesteps,)

    def to(self, device, *args, **kwargs):
        self.times = self.times.to(device)
        self.weights = self.weights.to(device)
        return super().to(device, *args, **kwargs)

    def forward(self, predictions: torch.Tensor, true_values: torch.Tensor, baseline: torch.Tensor = None):
        """
        Calculates the OD-RMSE Loss. We don't need the baseline predictions as we assume residual values from which the
        baseline predictions would cancel out.

        Args:
            predictions (torch.Tensor): The model's time series predictions.
                                        Expected shape (batch_size, timesteps).
            true_values (torch.Tensor): The true time series values.
                                      Expected shape (batch_size, timesteps).
            baseline (torch.Tensor): The baseline predictions.

        Returns:
            torch.Tensor: The scalar OD-RMSE Loss for the batch.
        """
        assert predictions.shape == true_values.shape, \
            (f"Input tensors must have the same shape. Got predictions:{predictions.shape}, "
             f"true_values:{true_values.shape}")
        assert predictions.shape[1] == self.times.shape[0], \
            f"Number of timesteps in predictions ({predictions.shape[1]}) and times ({self.times.shape[0]}) must match."

        # timesteps = predictions.shape[1]
        # batch_size = predictions.shape[0] # Not needed for calculation over batch+features
        # num_features = predictions.shape[2] # Not needed for calculation over batch+features

        # Ensure times tensor is on the same device as input data
        # times = times.to(predictions.device)

        # Calculate MSE and RMSE ratio for each timestep

        # We can optimize this loop using broadcasting or tensor operations
        # Calculate squared errors for all timesteps, batch items, and features
        squared_error_test = (predictions - true_values).square()  # Shape (batch_size, timesteps, num_features)
        if baseline is None:
            squared_error_msis = true_values.square()
        else:
            squared_error_msis = (true_values - baseline).square()  # Shape (batch_size, timesteps, num_features)

        # Calculate MSE for each timestep by taking mean over batch and features
        # Resulting shape (timesteps,)
        mse_test_per_timestep = torch.mean(squared_error_test, dim=0)
        mse_msis_per_timestep = torch.mean(squared_error_msis, dim=0)

        # Calculate RMSE for each timestep
        rmse_test_per_timestep = torch.sqrt(mse_test_per_timestep)
        rmse_msis_per_timestep = torch.sqrt(mse_msis_per_timestep)

        # Calculate the ratio for each timestep, adding delta for stability
        ratio_per_timestep = rmse_test_per_timestep / (
                    rmse_msis_per_timestep + self.numerical_stability_delta)  # Shape (timesteps,)

        # Calculate the weighted sum of ratios
        # Element-wise multiplication and then sum over timesteps
        weighted_ratio_sum = torch.sum(self.weights * ratio_per_timestep)

        # Sum of weights
        sum_weights = torch.sum(self.weights)

        # Calculate the final loss
        # Add a small epsilon to the sum of weights denominator in case all weights are zero
        # (though with positive gamma and positive duration/epsilon this shouldn't happen with non-empty times)
        loss = weighted_ratio_sum / (sum_weights + self.numerical_stability_delta)

        return loss
