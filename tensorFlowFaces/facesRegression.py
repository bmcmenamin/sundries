import os as os
import glob as glob

import skimage.data as skimData
from skimage.color import rgb2grey
from skimage.transform import resize

import tensorflow as tf

import matplotlib.pyplot as plt

rootDir = '/Users/mcmenamin/GitHub/tensorFlowFaces/'

os.chdir(rootDir)
import tfModels as tfModels

######################################
#
# Reading in all face images, resizing
#


def imageProcess(x):
    return rgb2grey(skimData.load(x))


def _scaleImg(X, dim):
    tmpIm = resize(X, [dim, dim]).ravel()
    tmpIm /= np.sqrt(np.mean(tmpIm**2))
    return tmpIm


trainFnames = glob.glob(rootDir + 'SaidAndTodorov_Model/faces_training_jpg/[m,f]*.jpg')
testFnames = glob.glob(rootDir + 'SaidAndTodorov_Model/faces_testing_jpg/[m,f]*.jpg')

train_images = [imageProcess(f) for f in trainFnames]
test_images = [imageProcess(f) for f in testFnames]

train_images_hidim = [_scaleImg(f, 128) for f in train_images]
test_images_hidim = [_scaleImg(f, 128) for f in test_images]

U, S, Vt = np.linalg.svd(np.vstack(train_images_hidim), full_matrices=False)
toKeep = S > 1
print('Keeping {} features ({:.1f}%)'.format(np.sum(toKeep), 100 * np.mean(toKeep)))

train_images_hidim_pca = np.vstack(train_images_hidim).dot(Vt[toKeep, :].T) / S[toKeep].reshape(1, -1)
test_images_hidim_pca = np.vstack(test_images_hidim).dot(Vt[toKeep, :].T) / S[toKeep].reshape(1, -1)







train_images_lodim = [_scaleImg(f, 64) for f in train_images]
test_images_lodim = [_scaleImg(f, 64) for f in test_images]

U, S, Vt = np.linalg.svd(np.vstack(train_images_lodim), full_matrices=False)
toKeep = S > 1
print('Keeping {} features ({:.1f}%)'.format(np.sum(toKeep), 100 * np.mean(toKeep)))

train_images_lodim_pca = np.vstack(train_images_lodim).dot(Vt[toKeep, :].T) / S[toKeep].reshape(1, -1)
test_images_lodim_pca = np.vstack(test_images_lodim).dot(Vt[toKeep, :].T) / S[toKeep].reshape(1, -1)



######################################
#
# Reading in attractiveness ratings for each face
#

attr_maleFaces_train = np.loadtxt(rootDir + 'SaidAndTodorov_Model/FrM_attractivenessratings_formatlab.csv', delimiter=',')[:, 0]
attr_femlFaces_train = np.loadtxt(rootDir + 'SaidAndTodorov_Model/MrF_attractivenessratings_formatlab.csv', delimiter=',')[:, 0]
attr_maleFaces_test = np.loadtxt(rootDir + 'SaidAndTodorov_Model/validationresultsFrM_formatlab.csv', delimiter=',')[1:, 0]
attr_femlFaces_test = np.loadtxt(rootDir + 'SaidAndTodorov_Model/validationresultsMrF_formatlab.csv', delimiter=',')[1:, 0]


train_attr = []
for f in trainFnames:
    f = f.split('/')[-1]
    num = int(f[1:-4])
    if f[0] == 'm':
        train_attr.append(attr_maleFaces_train[num])
    elif f[0] == 'f':
        train_attr.append(attr_femlFaces_train[num])
    else:
        print('bad gender?')

test_attr = []
for f in testFnames:
    f = f.split('/')[-1]
    num = int(f[1:-4])
    if f[0] == 'm':
        test_attr.append(attr_maleFaces_test[num])
    elif f[0] == 'f':
        test_attr.append(attr_femlFaces_test[num])
    else:
        print('bad gender?')


Ytrain = np.vstack(train_attr)
Ytest = np.vstack(test_attr)





####################################
#
# Linear regression
#

import importlib
importlib.reload(tfModels)

# Use linear regression to find 'attractive' face dimensions on 'high' and 'low'
# resolution images

linearModel_hidim = tfModels.linReg(train_images_hidim_pca, Ytrain,
                                    test_images_hidim_pca, Ytest)


linearModel_lodim = tfModels.linReg(train_images_lodim_pca, Ytrain,
                                    test_images_lodim_pca, Ytest)


"""

With simple linear regression models, we find that:
    -The result is pretty good if the faces are really downsampled (i.e., LSF features
    are useful for simple 'template matching'), with x-vaidated r2 ~ 0.40

    -The result is terrible if we allow hi-dimensional features (i.e., LSF+HSF features
    lead to an over-fit 'template'), with x-vaidated r2 ~ -0.6

Let's see what we can accomplish with a deep convolutional network!!

"""




####################################
#
# Multilayer convolutional network
#

import importlib
importlib.reload(tfModels)


deepModel_lodim = tfModels.convNet(np.vstack(train_images_lodim), Ytrain,
                                   np.vstack(test_images_lodim), Ytest)

deepModel_hidim = tfModels.convNet(np.vstack(train_images_hidim), Ytrain,
                                   np.vstack(test_images_hidim), Ytest)



"""

With hierarchical convolutional nets,
    -The result is pretty good if the faces are really downsampled (i.e., LSF features
    are useful for simple 'template matching'), with x-vaidated r2 ~ 0.60 on an undertrained
    network.

    -I didn't let the high dimensional one run all the way because I was impatient, so I'll
    save that for when I'm running on an actual GPU.
    
"""
