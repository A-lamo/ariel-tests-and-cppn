import math

def sigmoid(x):
    """The standard logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))

def tanh(x):
    """Hyperbolic tangent, maps values to (-1, 1)."""
    return math.tanh(x)

def sin_act(x):
    """Sine function, useful for generating repeating patterns."""
    return math.sin(x)

def gaussian(x):
    """Gaussian (bell curve) function."""
    return math.exp(-(x**2))

def relu(x):
    """Rectified Linear Unit."""
    return max(0.0, x)

# A list of all available activation functions
# Note: Functions are first-class objects in Python, so we can store them in a list.
ACTIVATION_FUNCTIONS = [
    sigmoid,
    tanh,
    sin_act,
    gaussian,
    relu,
    # Add more functions here (e.g., 'cos_act', 'identity', 'step')
]

DEFAULT_ACTIVATION = tanh