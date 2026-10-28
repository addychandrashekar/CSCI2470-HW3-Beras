import numpy as np
from typing import List

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        return Tensor(np.where(x > 0, x, x * self.alpha))

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        x = self.inputs[0]
        input_gradients = np.zeros_like(x)
        
        for index in np.ndindex(x.shape):
            if x[index] > 0: input_gradients[index] = 1
            elif x [index] < 0: input_gradients[index] = self.alpha
            else: input_gradients[index] = 0
        
        return [Tensor(input_gradients)]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)

class GELU(Activation):
    """GELU using the tanh approximation.

      Equivalent to torch.nn.GELU(approximate="tanh").
      Reference: https://arxiv.org/abs/1606.08415
      """
    ## TODO: Implement GELU for intermediate activation. Hint: Use the tanh approximation for GELU, which is faster to compute than the exact formula.
    
    def forward(self, x: Tensor) -> Tensor:
        u = np.sqrt(2 / np.pi) * (x + (0.044715 * (x ** 3)))
        forward_pass_result = 0.5 * x * (1 + np.tanh(u))

        return Tensor(forward_pass_result)

    def get_input_gradients(self) -> List[Tensor]:
        x = self.inputs[0]

        u = np.sqrt(2 / np.pi) * (x + (0.044715 * (x ** 3)))
        du = np.sqrt(2 / np.pi) * (1 + (3 * (0.044715 * (x**2))))

        input_gradients = 0.5 * (1 + np.tanh(u)) + 0.5 * x * (1 - (np.tanh(u)**2)) * du

        return [Tensor(input_gradients)]


    def compose_input_gradients(self, J: Tensor) -> Tensor:
        return self.get_input_gradients()[0] * J


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def forward(self, x) -> Tensor:
        forward_pass_result = 1 / (1 + np.exp(-x))

        return Tensor(forward_pass_result)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        forward_pass_result = self.outputs[0]
        input_gradients = forward_pass_result * (1 - forward_pass_result)

        return [Tensor(input_gradients)]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def forward(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        
        x_modified = x - np.max(x, axis=-1, keepdims=True)
        exponents = np.exp(x_modified)
        forward_pass_result = exponents / np.sum(exponents, axis=-1, keepdims=True)

        return Tensor(forward_pass_result)

    def get_input_gradients(self):
        """Softmax input gradients!"""
        x, y = self.inputs + self.outputs
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)
        
        # TODO: Implement softmax gradient
        
        for batch in range(bn):
            grad[batch] = -np.outer(y[batch], y[batch])
            np.fill_diagonal(grad[batch], y[batch] * (1 - y[batch]))
        
        return [Tensor(grad)]
