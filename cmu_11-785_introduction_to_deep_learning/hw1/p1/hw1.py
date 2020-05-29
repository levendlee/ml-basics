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
import abc
import typing


class Activation(abc.ABC):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """

        :param x: z. (n_samples, n_out_features)
        :return: y. (n_samples, n_out_features)
        """
        raise NotImplemented

    @abc.abstractmethod
    def derivative(self) -> np.ndarray:
        """

        :return: dy/dz. (n_samples, n_out_features)
        """
        raise NotImplemented


class Identity(Activation):
    """
    Identity function (already implemented).
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self) -> float:
        return 1.0


class Sigmoid(Activation):
    """
    Sigmoid non-linearity
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: np.ndarray):
        self.state = 1.0 / (1 + np.exp(-x))
        self.x = x
        return self.state

    def derivative(self):
        x = self.x
        return np.exp(-x) / np.power((np.exp(-x) + 1.0), 2)
        # return self.state * (1 - self.state)


class Tanh(Activation):
    """
    Tanh non-linearity
    """

    # This one's all you!
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        self.x = x
        return self.state

    def derivative(self):
        x = self.x
        return 1.0 / np.power(np.cosh(x), 2)


class ReLU(Activation):
    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.where(x > 0, x, 0.0)
        self.x = x
        return self.state

    def derivative(self):
        x = self.x
        return np.where(x > 0, 1.0, 0.0)


# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(abc.ABC):
    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.forward(x, y)

    @abc.abstractmethod
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """

        :param x: output. (n_samples, output_size)
        :param y: target. (n_samples,)
        :return: loss. (n_samples,)
        """
        raise NotImplemented

    @abc.abstractmethod
    def derivative(self) -> np.ndarray:
        """

        :return: (n_samples, output_size)
        """
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
        # self.sm shape: (n_samples, output_size)
        self.sm = np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
        # TODO: I don't understand why the autograder take the following formula as wrong
        # self.loss = np.sum(-y * np.log(self.sm) - (1 - y) * np.log(1 - self.sm), axis=1)
        self.loss = np.sum(-y * np.log(self.sm), axis=1)
        return self.loss

    def derivative(self):
        return self.sm - self.labels


class BatchNorm(object):
    """
    Batch normalization
    """

    def __init__(self, fan_in: int, alpha: float = 0.9):
        """

        :param fan_in: n_features of previous layer.
        :param alpha: momentum for running mean and running variance.
        """
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((fan_in,))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((fan_in,))

        # inference parameters
        self.fan_in = fan_in
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x: np.ndarray, eval=False) -> np.ndarray:
        return self.forward(x, eval)

    def forward(self, x: np.ndarray, eval=False) -> np.ndarray:
        """
        
        :param x: z. linear output. (n_samples, n_features)
        :param eval: train or test.
        :return: \hat{z}. shifted output. (n_samples, n_features)
        """
        if eval:
            norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * norm + self.beta
            return out

        self.x = x

        # following arrays shape: (1, n_features)
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.var = np.var(x, axis=0, keepdims=True)
        # following arrays shape: (n_samples, n_features)
        self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        self.out = self.gamma * self.norm + self.beta

        # update running batch statistics.
        # following arrays shape: (1, n_features)
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        return self.out

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """

        :param delta: \frac{dDiv}{d\hat{z}}. (n_samples, n_features)
        :return: \frac{dDiv}{dz}. (n_samples, n_features)
        """
        # (1, n_features)
        # Here we take sum, not average
        self.dgamma = np.sum(self.norm * delta, axis=0)
        self.dbeta = np.sum(delta, axis=0)

        dnorm = self.gamma * delta
        # followings are the formula from the slides
        '''
        # (1, n_features)
        dmean = (-(1.0 / np.sqrt(self.var + self.eps)) *
                 np.sum(dnorm, axis=0, keepdims=True))
        dvar = (-0.5 * np.power(self.var + self.eps, -1.5) *
                np.sum(dnorm * (self.x - self.mean), axis=0, keepdims=True))
        # (n_samples, n_features)
        dx = (dnorm / np.sqrt(self.var + self.eps) +
              dvar * 2 * (self.x - self.mean) / self.fan_in +
              dmean / self.fan_in)
        '''
        dx = (1.0 / (self.fan_in * np.sqrt(self.var + self.eps)) *
              (self.fan_in * dnorm
               - np.sum(dnorm, axis=0)
               - dnorm * np.sum(dnorm * self.norm, axis=0)))
        return dx


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0: int, d1: int) -> np.ndarray:
    """
    Create a randomized 2D array.
    :param d0:
    :param d1:
    :return:
    """
    return np.random.rand(d0, d1)


def zeros_bias_init(d: int) -> np.ndarray:
    """
    Create a zeroed 1D array.
    :param d:
    :return:
    """
    return np.zeros((1, d))


def check_shape(arr, dim):
    assert arr.shape == dim, \
        "Expect shape: {}; \n Actual shape: {};\n".format(dim, arr.shape)


class MLP(object):
    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size: int,
                 output_size: int,
                 hiddens: typing.Sequence[int],
                 activations: typing.Sequence[Activation],
                 weight_init_fn: typing.Callable[[int, int], np.ndarray],
                 bias_init_fn: typing.Callable[[int], np.ndarray],
                 criterion: Criterion,
                 lr: float,  # learning rate
                 momentum: float = 0.0,
                 num_bn_layers: int = 0):
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

        self.sizes = sizes = [input_size] + hiddens + [output_size]
        # self.W / self.dW: A list of matrices. L * D_{i} * D_{i+1}
        #                   Each matrix contains weights rom i th layer to i+1 th layer.
        self.W = [weight_init_fn(sizes[i], sizes[i + 1]) for i in range(self.nlayers)]
        self.dW = [np.zeros((self.sizes[i], self.sizes[i + 1])) for i in range(self.nlayers)]
        # self.b / self.db: A list of vectors. L * 1 * D_{i+1}
        self.b = [bias_init_fn(sizes[i + 1]) for i in range(self.nlayers)]
        self.db = [np.zeros((self.sizes[i + 1],)) for i in range(self.nlayers)]
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        # NOTE: Assume we add batch normalization since the first layer.
        if self.bn:
            self.bn_layers = [BatchNorm(self.sizes[i + 1]) for i in range(self.num_bn_layers)]
        else:
            self.bn_layers = []
        self.zero_grads()

        # cache intermediate results for gradient calculation
        self.y = [None for _ in range(self.nlayers + 1)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """

        :param x: input. (n_samples, input_size)
        :return: output. (n_samples, output_size)
        """
        # cache input. (n_samples, input_size / n_features)
        n_samples = x.shape[0]
        y = self.y[0] = x
        for i in range(self.nlayers):
            check_shape(y, (n_samples, self.sizes[i]))
            check_shape(self.W[i], (self.sizes[i], self.sizes[i + 1]))
            check_shape(self.b[i], (1, self.sizes[i + 1]))
            z = np.matmul(y, self.W[i]) + self.b[i]
            check_shape(z, (n_samples, self.sizes[i + 1]))

            if i < self.num_bn_layers:
                z = self.bn_layers[i](z, eval=not self.train_mode)
                check_shape(z, (n_samples, self.sizes[i + 1]))

            y = self.y[i + 1] = self.activations[i](z)
            check_shape(y, (n_samples, self.sizes[i + 1]))
        return self.y[-1]

    def zero_grads(self):
        """
        Reset all gradients.
        :return:
        """
        self.dW_prev = self.dW
        self.db_prev = self.db
        self.dW = [np.zeros((self.sizes[i], self.sizes[i + 1])) for i in range(self.nlayers)]
        self.db = [np.zeros((self.sizes[i + 1],)) for i in range(self.nlayers)]
        '''
        for layer in self.bn_layers:
            layer.dgamma_prev = layer.dgamma
            layer.dbeta_prev = layer.dbeta
            layer.dgamma = np.zeros((layer.fan_in,))
            layer.dbeta = np.zeros((layer.fan_in,))
        '''

    def step(self):
        """
        Update gradient.
        :return:
        """
        for i in range(self.nlayers):
            self.dW[i] = self.momentum * self.dW_prev[i] - self.lr * self.dW[i]
            self.W[i] = self.W[i] + self.dW[i]
            self.db[i] = self.momentum * self.db_prev[i] - self.lr * self.db[i]
            self.b[i] = self.b[i] + self.db[i]

        for layer in self.bn_layers:
            # layer.dgamma = self.momentum * layer.dgamma_prev - self.lr * layer.dgamma
            layer.gamma += layer.dgamma
            # layer.dbeta = self.momentum * layer.dbeta_prev - self.lr * layer.dbeta
            layer.beta += layer.dbeta

    def backward(self, labels: np.ndarray) -> np.ndarray:
        """

        :param labels: expected output. (n_samples,)
        :return: loss. (n_samples,)
        """
        n_samples = labels.shape[0]
        # (n_samples, output_size)
        y = self.y[-1]
        # (n_samples,)
        loss = self.criterion(y, labels)
        # (n_samples, output_size)
        delta = self.criterion.derivative()

        for i in range(-1, -self.nlayers - 1, -1):
            check_shape(loss, (n_samples,))
            check_shape(delta, (n_samples, self.sizes[i]))
            delta = self.activations[i].derivative() * delta

            if (i + self.nlayers) < self.num_bn_layers:
                delta = self.bn_layers[i + self.nlayers].backward(delta)

            y = self.y[i - 1]

            check_shape(delta, (n_samples, self.sizes[i]))
            self.dW[i] = np.matmul(np.transpose(y), delta) / n_samples
            self.db[i] = np.mean(delta, axis=0)
            check_shape(self.dW[i], (self.sizes[i - 1], self.sizes[i]))
            check_shape(self.db[i], (self.sizes[i],))

            delta = np.matmul(delta, np.transpose(self.W[i]))
            check_shape(delta, (n_samples, self.sizes[i - 1]))

        return loss

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    train = np.random.shuffle(train)
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    '''
    def pad_data(trainx):
        raw_size = len(trainx)
        padded_size = (raw_size // batch_size + 1) * batch_size
        trainx[raw_size:padded_size] = np.random.choice(trainx, padded_size - raw_size)
        return trainx
    
    pad_data(trainx)
    pad_data(val)
    '''

    norm_mean, norm_var = np.mean(trainx, axis=1), np.var(trainx, axis=1)
    trainx = (trainx - norm_mean) / norm_var

    for e in range(nepochs):
        mlp.zero_grads()
        mlp.train()
        cur_training_loss = []
        cur_training_error = 0
        for b in range(0, len(trainx), batch_size):
            x, y = trainx[b: b + batch_size], trainy[b: b + batch_size]
            output = mlp.forward(x)
            loss = mlp.criterion(output, y)
            mlp.backward(y)
            mlp.step()
            cur_training_loss.append(loss)
            cur_training_error += np.sum(np.argmax(output) != y)

        mlp.eval()
        cur_validation_loss = []
        cur_validation_error = 0
        for b in range(0, len(valx), batch_size):
            x, y = valx[b: b + batch_size], valy[b: b + batch_size]
            output = mlp.forward(x)
            loss = mlp.criterion(output, y)
            cur_validation_loss.append(loss)
            cur_validation_error += np.sum(np.argmax(output) != y)

        # Accumulate data...
        training_losses.append(np.mean(cur_training_loss))
        training_errors.append(cur_training_error / len(trainx))

    # Cleanup ...
    mlp.eval()
    test_error = 0
    for b in range(0, len(testx), batch_size):
        x, y = testx[b: b + batch_size], testy[b: b + batch_size]
        output = mlp.forward(x)
        test_error += np.sum(np.argmax(output) != y)
    test_error /= len(test_error)

    return (training_losses, training_errors, validation_losses, validation_errors)
