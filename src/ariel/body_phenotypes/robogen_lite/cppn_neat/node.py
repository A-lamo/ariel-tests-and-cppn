from typing import Callable as function

class Node:
    """A node gene in a NEAT genome."""
    def __init__(self,
                  _id: int,
                  typ: str, 
                  activation: function,
                  bias: float
                ):
        self._id = _id
        self.typ = typ
        self.activation = activation
        self.bias = bias

    