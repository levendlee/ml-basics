import numpy as np


class Linear():
    # DO NOT DELETE
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx


class Conv1D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def add_padding(self, x):
        batch, in_channel, width = x.shape
        padded_width = (((width - self.kernel_size - 1) // self.stride + 1) * self.stride +
                        self.kernel_size)
        padded_x = np.zeros((batch, in_channel, padded_width))
        padded_x[:, :, 0:width] = x
        return padded_x

    def forward(self, x):
        ## Your codes here
        self.x = x
        self.batch, __, self.width = x.shape
        assert __ == self.in_channel, \
            'Expected the inputs to have {} channels'.format(self.in_channel)

        self.output_width = (self.width - self.kernel_size) // self.stride + 1
        z = np.zeros([self.batch, self.out_channel, self.output_width])
        for i in range(self.output_width):
            start = i * self.stride
            end = start + self.kernel_size
            # x[,,]: batch, in_channel, kernel_size
            # W: out_channel, in_channel, kernel_size
            # z[,,]: batch, output_channel
            z[:, :, i] = np.tensordot(x[:, :, start:end], self.W,
                                      axes=([1, 2], [1, 2])) + self.b
        return z

    def backward(self, delta):
        ## Your codes here
        dx = np.zeros([self.batch, self.in_channel, self.width])

        for k in range(self.width):
            i = max((k - self.kernel_size + 1) // self.stride, 0)
            while i * self.stride + self.kernel_size <= k:
                i += 1
            j = k // self.stride
            while j * self.stride + self.kernel_size > self.width or j >= self.output_width:
                j -= 1
            if i > j:
                continue
            wi = k - j * self.stride
            wj = k - i * self.stride
            # delta[,,]: batch, out_channel, impacted_weights_size
            # W[,,]: out_channel, in_channel, impacted_weights_size
            # dx[,,]: batch, in_channel
            dx[:, :, k] = \
                np.tensordot(delta[:, :, i:j + 1],
                             self.W[:, :, wi:wj + 1:self.stride][:, :, ::-1],
                             axes=([1, 2], [0, 2]))

        # delta[,,]: batch, out_channel, output_width
        # x[,,]: batch, in_channel, output_width
        # dW: output_channel, in_channel, kernel_size
        for k in range(self.kernel_size):
            stop = k + self.output_width * self.stride
            self.dW[:, :, k] = \
                np.tensordot(delta[:, :, :],
                             self.x[:, :, k:stop:self.stride],
                             axes=([0, 2], [0, 2]))

        self.db = np.sum(delta, axis=(0, 2))
        return dx


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.batch = x.shape[0]
        self.channel = x.shape[1]
        self.width = x.shape[2]
        return x.reshape((self.batch, self.channel * self.width))

    def backward(self, x):
        return x.reshape((self.batch, self.channel, self.width))


class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x >= 0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta
