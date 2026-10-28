import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        squared_error = (y_pred - y_true) ** 2
        example_means = np.mean(squared_error, axis = 1)

        full_error_calculation = np.mean(example_means)

        return Tensor(full_error_calculation)

    def get_input_gradients(self) -> list[Tensor]:
        y_prediction, y_true = self.inputs

        prediction_gradients = (2 * (y_prediction  - y_true)) / y_prediction.size
        true_gradients = np.zeros_like(y_true)

        return [Tensor(prediction_gradients), Tensor(true_gradients)]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        
        epsilon = 1e-7
        
        y_prediction = np.clip(y_pred, epsilon, 1 - epsilon)
        sample_losses = -np.sum(y_true * np.log(y_prediction), axis = 1)

        return Tensor(np.mean(sample_losses))

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        
        y_prediction, y_true = self.inputs
        epsilon = 1e-7

        y_prediction_clipped = np.clip(y_prediction, epsilon, 1 - epsilon)
        batch_size = y_prediction.shape[0]

        prediction_gradients = -y_true / (batch_size * y_prediction_clipped)
        true_gradients = np.zeros_like(y_true)

        return [Tensor(prediction_gradients), Tensor(true_gradients)]
