import numpy as np
import theano
import theano.tensor as theanoTensor

class LinearRegression(object):
    """Linear Regression Class with Theano

    The regression is fully described by a weight matrix `W`
    and bias vector `b`.
    """

    def __init__(self, n_in):
        self.X = theanoTensor.matrix('X', dtype=theano.config.floatX)
        self.y = theanoTensor.vector('y', dtype=theano.config.floatX)

        self.W = theano.shared(name='W',
                               value=np.ones((n_in, ), dtype=theano.config.floatX),
                               borrow=True)

        self.b = theano.shared(name='b',
                               value=np.cast[theano.config.floatX](0.0),
                               borrow=True)

        y_pred = theanoTensor.dot(self.X, self.W) + self.b
        self.MSe = theanoTensor.mean(theanoTensor.pow(y_pred - self.y, 2))
        self.MSy = theanoTensor.mean(theanoTensor.pow(self.y, 2))
        self.R2 = 1 - (self.MSe / self.MSy)

        paramList = [self.W, self.b]
        grad_wrtParams = theanoTensor.grad(self.getMSE(), wrt=paramList)
        learning_rate = 1e-3
        updates = [(p[0], p[0] - learning_rate * p[1]) for p in zip(paramList, grad_wrtParams)]

        self.train_model = theano.function(
            inputs=[self.X, self.y],
            outputs=[self.getMSE()],
            updates=updates
        )

        self.test_model = theano.function(
            inputs=[self.X, self.y],
            outputs=[self.getR2()],
        )

    def fit(self, dataX, dataY):
        maxIter = 10000
        for p in range(maxIter):
            self.train_model(dataX, dataY)
            if (p % 100) == 0:
                print('{}/{}, R^2 = {}'.format(p + 1, maxIter, self.test_model(dataX, dataY)))

    def getR2(self):
        return self.R2

    def getMSE(self):
        return self.MSe
