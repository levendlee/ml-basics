"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others
    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # Might we need to store something before returning?
        self.state = x
        return 1.0 / (1 + np.exp(-x))

    def derivative(self):
        # Maybe something we need later in here...
        x = self.state
        raise np.exp(-x) / np.power((np.exp(-x) + 1), 2)


class Tanh(Activation):
    """
    Tanh non-linearity
    """

    # This one's all you!
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = x
        return np.tanh(x)

    def derivative(self):
        x = self.state
        return 1.0 / np.power(np.cosh(x), 2)


class ReLU(Activation):
    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x
        raise np.where(x >= 0, x, 0)

    def derivative(self):
        x = self.state
        raise np.where(x >= 0, 1, 0)


# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        self.sm = np.exp(x) / np.sum(np.exp(x))
        self.loss = np.sum(-y * np.log(self.sm) - (1 - y) * np.log(1 - self.sm))
        return self.sm

    def derivative(self):
        # self.sm might be useful here...
        return self.sm - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):
        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.fan_in = fan_in
        self.n_batches = 0
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        if eval:
            norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * norm + self.beta
            return out

        self.x = x

        self.mean = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)
        self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        self.out = self.gamma * self.norm + self.beta

        # update running batch statistics
        self.running_mean = \
            (self.running_mean * self.n_batches + self.mean) / (self.n_batches + 1)
        coeff0 = self.fan_in / ((self.fan_in - 1) * self.n_batches)
        coeff1 = self.fan_in / ((self.fan_in - 1) * (self.n_batches + 1))
        self.running_var = \
            ((self.running_var / coeff0) * self.n_batches + self.var) * coeff1
        self.n_batches += 1

        return self.out

    def backward(self, delta):
        self.dgamma = self.out * delta
        self.dbeta = delta
        dnorm = self.gamma * delta

        dmean = -(1.0 / np.sqrt(self.var + self.eps)) * np.sum(dnorm, axis=1, keepdims=True)
        dvar = -0.5 * np.power(self.var + self.eps, -1.5) * np.sum(dnorm * (self.x - self.mean))
        dx = (dnorm / np.sqrt(self.var + self.eps) +
              dvar * 2 * (self.x - self.mean) / self.fan_in +
              dmean / self.fan_in)
        return dx


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.rand(d0, d1)


def zeros_bias_init(d):
    return np.zeros(d)


class MLP(object):
    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn,
                 criterion, lr, momentum=0.0, num_bn_layers=0):
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = np.random.rand(self.nlayers, input_size)
        self.dW = np.zeros([self.nlayers, input_size])
        self.b = np.zeros(self.nlayers)
        self.db = np.zeros(self.nlayers)
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = [BatchNorm(self.bn) for i in range(self.nlayers)]

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        z = x
        for i in range(self.nlayers):
            y = self.W[i, :] * z + self.b[i]
            if self.bn:
                y = self.bn_layers[i](y)
            z = self.activations[i](y)
        return self.criterion(z)

    def zero_grads(self):
        raise NotImplemented

    def step(self):
        raise NotImplemented

    def backward(self, labels):
        raise NotImplemented

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):
            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):
            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):
        pass  # Remove this line when you start implementing this
        # Test ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
