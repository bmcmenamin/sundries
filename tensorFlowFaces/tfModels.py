import numpy as np
import tensorflow as tf

"""
code for running tests of each model

import numpy as np
X = np.random.randn(10000, 5)
Y = np.random.randn(10000, 1) + 0.5 * X[:, 0:1] - 0.25 * X[:, 1:2]
m = linReg(X[0::2, :], Y[0::2], X[1::2, :], Y[1::2])

"""

################################
#
# Linear regression functions (iterative)
#


def linReg_iter(Xtrain, Ytrain, Xtest, Ytest):
    """
    Linear regression built in tensor flow
    """

    numIter = 5000
    dim = Xtrain.shape[1]
    totalTrainVar = np.mean(Ytrain**2)
    totalTestVar = np.mean(Ytest**2)
    with tf.Session() as sess:
        x = tf.placeholder("float", shape=[None, dim])
        y_ = tf.placeholder("float", shape=[None, 1])

        W = tf.Variable(tf.random_uniform([dim, 1]))
        # W = tf.Print(W, [W], 'W:')

        b = tf.Variable(tf.random_uniform([1, 1]))
        # b = tf.Print(b, [b], 'b:')

        y = tf.matmul(x, W) + b
        # y = tf.Print(y, [y], 'y:')

        mse = tf.reduce_mean(tf.square(y_ - y))
        sess.run(tf.initialize_all_variables())

        # train_step = tf.train.MomentumOptimizer(1.0e-2, 0.1).minimize(mse)
        train_step = tf.train.AdamOptimizer().minimize(mse)
        sess.run(tf.initialize_all_variables())

        for i in range(numIter):
            train_step.run(feed_dict={x: Xtrain, y_: Ytrain}, session=sess)
            if (i % 500) == 0:
                trainError = mse.eval(feed_dict={x: Xtrain, y_: Ytrain}, session=sess)
                trainR2 = (totalTrainVar - trainError) / totalTrainVar
                print("Train r2: {:.2f}".format(trainR2))
        testError = mse.eval(feed_dict={x: Xtest, y_: Ytest}, session=sess)
        testR2 = (totalTestVar - testError) / totalTestVar
        print("Test r2: {:.2f}".format(testR2))

        statModel = [W.eval(session=sess), b.eval(session=sess), testR2]
    return statModel


################################
#
# Linear regression functions (analytical)
#


def linReg(Xtrain, Ytrain, Xtest, Ytest):
    """
    Linear regression built in tensor flow
    """

    dim = Xtrain.shape[1]
    totalTrainVar = np.mean(Ytrain**2)
    totalTestVar = np.mean(Ytest**2)
    with tf.Session() as sess:
        x = tf.placeholder("float", shape=[None, dim + 1])
        y_ = tf.placeholder("float", shape=[None, 1])

        hatMat = tf.matmul(tf.matrix_inverse(tf.matmul(x, x, transpose_a=True)), x, transpose_b=True)
        W = tf.matmul(hatMat, y_)

        Xtrain_int = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
        Xtest_int = np.hstack([Xtest, np.ones((Xtest.shape[0], 1))])

        obsW = W.eval(feed_dict={x: Xtrain_int, y_: Ytrain}, session=sess)

        predY = Xtrain_int.dot(obsW)
        rmse = np.mean((predY - Ytrain)**2)
        trainR2 = (totalTrainVar - rmse) / totalTrainVar
        print("Train R2: {:.2f}".format(trainR2))

        predY = Xtest_int.dot(obsW)
        rmse = np.mean((predY - Ytest)**2)
        testR2 = (totalTestVar - rmse) / totalTestVar
        print("Test R2: {:.2f}".format(testR2))
    return obsW


##############################
#
# Convolutional net functions
#



"""
code for running tests of each model

import numpy as np
numData = 400
Xtrain = np.random.randn(numData, 128*128)
Ytrain = np.random.randn(numData, 1)
Ytest = np.random.randn(numData, 1)

"""


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_NxN(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def convNet(Xtrain, Ytrain, Xtest, Ytest):
    """
    Convolutional netowrk built in tensor flow
    """

    numIter = 10000
    dim = Xtrain.shape[1]
    imageDim = int(np.sqrt(dim))
    totalTrainVar = np.mean(Ytrain**2)
    totalTestVar = np.mean(Ytest**2)
    with tf.Session() as sess:
        x = tf.placeholder("float", shape=[None, dim])
        y_ = tf.placeholder("float", shape=[None, 1])

        x_image = tf.reshape(x, [-1, imageDim, imageDim, 1])

        # Kernels per layer
        W_conv1 = weight_variable([12, 12, 1, 16])
        W_conv2 = weight_variable([8, 8, 16, 32])
        W_conv3 = weight_variable([4, 4, 32, 64])

        # Bias per layer
        b_conv1 = bias_variable(W_conv1.get_shape().as_list()[-1:])
        b_conv2 = bias_variable(W_conv2.get_shape().as_list()[-1:])
        b_conv3 = bias_variable(W_conv3.get_shape().as_list()[-1:])

        # Layer outputs
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_NxN(h_conv1, 2)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_NxN(h_conv2, 2)

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_NxN(h_conv3, 2)


        # Fully-connected Dense Layer with dropout
        tmpDim = np.prod([h for h in h_pool3.get_shape().as_list() if h is not None])
        denseSize = 256

        W_fc1 = weight_variable([tmpDim, denseSize])
        W_fc2 = weight_variable([denseSize, 1])

        b_fc1 = bias_variable([denseSize])
        b_fc2 = bias_variable([1])

        h_poolX_flat = tf.reshape(h_pool3, [-1, tmpDim])
        h_fc1 = tf.nn.relu(tf.matmul(h_poolX_flat, W_fc1) + b_fc1)

        # Dropout
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Readout layer
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Loss function
        mse = tf.reduce_mean(tf.square(y_ - y_conv))
        sess.run(tf.initialize_all_variables())

        train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)
        sess.run(tf.initialize_all_variables())

        toKeep = [i <= 100 for i in range(Xtrain.shape[0])]
        dataDict_full = {x: Xtrain, y_: Ytrain, keep_prob: 1.0}
        for i in range(numIter):
            toKeep = np.random.permutation(toKeep)
            dataDict = {x: Xtrain[toKeep, :], y_: Ytrain[toKeep, :], keep_prob: 0.5}
            train_step.run(feed_dict=dataDict, session=sess)
            if (i % 100) == 0:
                trainError = mse.eval(feed_dict=dataDict_full, session=sess)
                trainR2 = (totalTrainVar - trainError) / totalTrainVar
                print("Train r2: {:.2f}".format(trainR2))
        testError = mse.eval(feed_dict={x: Xtest, y_: Ytest, keep_prob: 1.0}, session=sess)
        testR2 = (totalTestVar - testError) / totalTestVar
        print("Test r2: {:.2f}".format(testR2))

        # statModel = [W.eval(session=sess), b.eval(session=sess), testR2]
    return testR2
