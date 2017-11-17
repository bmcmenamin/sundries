import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding

from matplotlib import cm

#############################
#
# simulating data as a network that develops two modular communities
# over time
#


def setDiagToOne(X):
    '''
    This takes a square matrix and normalizes rows/columns to make the
    diagonal entries equal to 1. Use this to turn covariance matrices into
    correlation matrices
    '''
    tmp = 1 / np.sqrt(np.diag(X)).reshape(-1, 1)
    return tmp * X * tmp.T


def forcePosDef(X):
    '''
    This takes a square matrix and forces it to be positive semidefinite
    '''
    D, V = np.linalg.eigh(X)
    posD = D > 0
    Xpos = (V[:, posD] * D[posD].reshape(1, -1)).dot(V[:, posD].T)
    Xpos /= np.sum(D[posD])
    return Xpos


nNodes_comm1 = 12
nNodes_comm2 = 12
nNodes = nNodes_comm1 + nNodes_comm2

noiseStrength = 0.05
noiseMat_end = np.random.randn(nNodes, nNodes)
noiseMat_end = noiseMat_end.dot(noiseMat_end.T) + np.eye(nNodes) * noiseStrength
adjMat_end = setDiagToOne(forcePosDef(noiseMat_end))

withinNetStrength = 0.3
noiseMat_start = np.copy(adjMat_end)
noiseMat_start[:nNodes_comm1, :nNodes_comm1] += withinNetStrength
noiseMat_start[-nNodes_comm2:, -nNodes_comm2:] += withinNetStrength
noiseMat_start[:nNodes_comm1, -nNodes_comm2:] -= withinNetStrength
noiseMat_start[-nNodes_comm2:, :nNodes_comm1] -= withinNetStrength
adjMat_start = setDiagToOne(forcePosDef(noiseMat_start))


#############################
#
# interpolating networks between start and end times
#

numInterSteps = 1

groupD, groupV = np.linalg.eigh(adjMat_start + adjMat_end)

D_start = np.diag(groupV.T.dot(adjMat_start).dot(groupV))
D_end = np.diag(groupV.T.dot(adjMat_end).dot(groupV))

D_timeseries = []
adjMat_timeseries = []
for tStep in np.linspace(0, 1, numInterSteps + 2):
    tmpD = tStep * D_start + (1 - tStep) * D_end
    D_timeseries.append(tmpD)
    tmpAdj = (groupV * tmpD).dot(groupV.T)
    adjMat_timeseries.append(setDiagToOne(tmpAdj))


#############################
#
# Using TSNE to visualize network layout for each timepoint
#

simMatAllTimes = []
for key1, val1 in enumerate(D_timeseries):
    tmpRowMatrix = []
    for key2, val2 in enumerate(D_timeseries):
        geomMeanOfD = np.array(np.sqrt(val1 * val2)).reshape(1, -1)
        tmpAdj = (groupV * geomMeanOfD).dot(groupV.T)
        tmpRowMatrix.append(tmpAdj)
    tmpRowMatrix = np.hstack(tmpRowMatrix)
    simMatAllTimes.append(tmpRowMatrix)
simMatAllTimes = np.vstack(simMatAllTimes)

simMatAllTimes = setDiagToOne(forcePosDef(simMatAllTimes))

# tsneModel = TSNE(n_components=2, metric='precomputed',
#                 method='barnes_hut', perplexity=30.0)
# #method='barnes_hut'
# #method='exact'
# distMat = 1.0 / (simMatAllTimes + 1.01)

tsneModel = SpectralEmbedding(n_components=2, affinity='precomputed',
                              gamma=1.0, n_neighbors=6)

distMat = simMatAllTimes + 1.0


tsneModel = tsneModel.fit(distMat)
sizeScale = np.abs(tsneModel.embedding_.ravel()).max()
tsneModel.embedding_ /= sizeScale

#############################
#
# Plotting
#

# set plot parameters

comm1_str = np.vstack([np.sum(a[:, :nNodes_comm1], axis=1, keepdims=True)
                       for a in adjMat_timeseries])

comm2_str = np.vstack([np.sum(a[:, -nNodes_comm2:], axis=1, keepdims=True)
                       for a in adjMat_timeseries])

relativeCommStr = comm1_str - comm2_str
relativeCommStr *= 128 / np.max(np.abs(relativeCommStr))
relativeCommStr += 128

nodeColors = [cm.winter(int(r[0])) for r in relativeCommStr]

nodeKWargs = [{'markerfacecolor': c,
               'marker': 'o',
               'markersize': 8,
               'linestyle': 'None',
               'zorder': 2}
              for c in nodeColors]

edgeKWargs = {'color': 'k',
              'marker': 'None',
              'linestyle': '-',
              'linewidth': 1,
              'zorder': 1}

# Initialize plots
plt.close('all')
fig, figAxes = plt.subplots(len(adjMat_timeseries), 2)
for i in range(len(adjMat_timeseries)):
    figAxes[i, 1].set_yticks([])
    figAxes[i, 1].set_xticks([])
    figAxes[i, 1].set(axis_bgcolor='w')
    figAxes[i, 1].set_ylim([-1.3, 1.3])
    figAxes[i, 1].set_xlim([-1.3, 1.3])

# Plots nodes
for key, val in enumerate(nodeKWargs):
    x = tsneModel.embedding_[key, 0]
    y = tsneModel.embedding_[key, 1]
    figAxes[key // nNodes, 1].plot(x, y, **val)

# Plots edges

edgeCutoff = 75
goodEdgeList = []
for key, val in enumerate(adjMat_timeseries):
    tmpCutoff = np.percentile(val.ravel(), edgeCutoff)
    hasEdge = np.where(np.triu(val > tmpCutoff, k=1))
    for i in zip(*hasEdge):
        goodEdgeList.append((i[0] + key * nNodes, i[1] + key * nNodes))

for e in goodEdgeList:
    x1 = tsneModel.embedding_[e[0], 0]
    y1 = tsneModel.embedding_[e[0], 1]
    x2 = tsneModel.embedding_[e[1], 0]
    y2 = tsneModel.embedding_[e[1], 1]
    figAxes[e[0] // nNodes, 1].plot([x1, x2], [y1, y2], **edgeKWargs)

# Plot Adjacency matrices
for key, val in enumerate(adjMat_timeseries):
    sb.heatmap(1 - val, annot=False, linecolor='black', linewidths=0.1,
               square=True, cbar=False, vmin=0, vmax=1, cmap='gray',
               xticklabels=False, yticklabels=False, ax=figAxes[key, 0])
