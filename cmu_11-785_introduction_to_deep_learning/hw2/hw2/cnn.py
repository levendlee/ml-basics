from mlp import *


class CNN_B():
    """
    Input width: 128.
    Channel: 24.
    Kernel Size: 8.
    Stride: 4.
    """

    def __init__(self):
        # Your initialization code goes here
        self.layers = [
            # Input: 1 x 24 x 128
            Conv1D(in_channel=24,
                   out_channel=8,
                   kernel_size=8,
                   stride=4),
            # Output: 1 x 8 x 31
            ReLU(),
            Conv1D(in_channel=8,
                   out_channel=16,
                   kernel_size=1,
                   stride=1),
            # Output: 1 x 16 x 31
            ReLU(),
            Conv1D(in_channel=16,
                   out_channel=4,
                   kernel_size=1,
                   stride=1),
            # Output: 1 x 4 x 31
            Flatten()
            # Output: 1 x 124
        ]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        self.layers[0].W = weights[0].reshape((8, 24, 8)).T
        self.layers[2].W = weights[1].reshape((1, 8, 16)).T
        self.layers[4].W = weights[2].reshape((1, 16, 4)).T

    def forward(self, x):
        print(x.shape)
        # You do not need to modify this method
        out = x
        for i, layer in enumerate(self.layers):
            print("Forwarding at layer {} {}".format(i, layer.__class__.__name__))
            out = layer(out)
            print(out.shape)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        self.layers = []

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        raise NotImplemented

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
