import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


class TwoClassTwoDimensionDataset(object):
    """Dataset with two Gaussian pointclouds in a 2D plane
    """

    @staticmethod
    def rotation_matrix(theta):
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        return rot_mat

    def get_rotated_data(self, rot_degrees):
        return self.X.dot(self.rotation_matrix(rot_degrees))

    def __init__(self, num_samples=2000, sep_dist=2.0, grid_steps=50):
        """ Initialize dataclouds with Gaussian data in clouds that are
            sep_dist apart from each other
        """

        samp_per_class = num_samples//2
        self.y = np.ones(num_samples,)
        self.y[:samp_per_class] = 0

        self.X = np.random.randn(num_samples, 2)
        self.X[:, 1] *= 3
        self.X[:, 0] += sep_dist * (self.y - 0.5)

        self.plot_width = np.sqrt(np.max(np.sum(self.X ** 2, axis=1)))
        
        _x, _y = np.meshgrid(
            np.linspace(-self.plot_width, self.plot_width, grid_steps),
            np.linspace(-self.plot_width, self.plot_width, grid_steps)
        )

        self.grid_x = _x
        self.grid_y = _y
        self.grid_xy = np.c_[self.grid_x.ravel(), self.grid_y.ravel()]


class RotatingClassifierAnimation(object):

    CMAP = cmap=plt.get_cmap('plasma')

    def __init__(self, model, dataset, min_auc=0.85):
        """ Initialize dataclouds with Gaussian data in clouds that are
            sep_dist apart from each other

            Args:
                model: SKLearn classifier model
                dataset: A TwoClassTwoDimensionDataset
        """

        self.model = model
        self.data = dataset
        self.axes = None
        self.min_auc = min_auc

    def fit_model(self, newX):
        self.model = self.model.fit(newX[0::2], self.data.y[0::2])
        preds = self.model.predict_proba(newX[1::2])[:, 1]
        auc = roc_auc_score(self.data.y[1::2], preds)
        fpr, tpr, _ = roc_curve(self.data.y[1::2], preds)
        return auc, list(fpr), list(tpr)

    def _init_point_clouds(self, ax):
        ax.grid(False)
        ax.set_title("Data clouds")
        ax.set_xlim(-self.data.plot_width, self.data.plot_width)
        ax.set_ylim(-self.data.plot_width, self.data.plot_width)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

        point_cloud = [
            Circle(
                center,
                radius=0.1,
                color=self.CMAP.colors[-1] if cat > 0 else self.CMAP.colors[0],
                alpha=0.3,
                edgecolor=None,
                visible=False
            )
            for center, cat in zip(self.data.X, self.data.y)
        ]

        for p in point_cloud:
            ax.add_patch(p)

        return ax.patches

    @staticmethod
    def _update_point_clouds(ax, rotX):
        for p, center in zip(ax.patches, rotX):
            p.set_center(center)
            p.set_visible(True)
        return ax.patches

    def _init_decision_boundaries(self, ax):
        ax.grid(False)
        ax.set_title("Learned Decision Boundary")
        ax.set_xlim(-self.data.plot_width, self.data.plot_width)
        ax.set_ylim(-self.data.plot_width, self.data.plot_width)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

    def _update_decision_boundaries(self, ax):

        preds = (
            self.model.predict_proba(self.data.grid_xy)[:, 1]
            .reshape(self.data.grid_x.shape)
        )

        ax.contourf(
            self.data.grid_x,
            self.data.grid_y,
            preds,
            levels=24,
            cmap=self.CMAP
        )

    @staticmethod
    def _init_roc_curve(ax):
        ax.set_title("ROC on rotated dataset")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        _ticks = [np.round(x, 2) for x in np.arange(0.0, 1.0 + 0.01, 0.25)]

        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        ax.set_ylim(_ticks[0], _ticks[1])
        ax.set_xlim(_ticks[0], _ticks[1])
        ax.set_yticks(_ticks)
        ax.set_xticks(_ticks)
        ax.set_yticklabels(_ticks)
        ax.set_xticklabels(_ticks)
        ax.set_aspect('equal')

        ax.plot([0, 1], [0, 1], '--k', linewidth=0.5)
        ax.plot([0, 1], [0, 1], '-k')

        for line in ax.lines:
            line.set_visible(False)

        return ax.lines

    @staticmethod
    def _update_roc_curve(ax, fpr, tpr):
        for line in ax.lines:
            line.set_visible(True)
        ax.lines[1].set_xdata([0] + fpr + [1.0])
        ax.lines[1].set_ydata([0] + tpr + [1.0])
        return ax.lines

    def _init_auc_plot(self, ax):
        ax.set_title("Classification as function of rotation")
        ax.set_ylabel("Classification (AUC)")
        ax.set_xlabel("Input rotation (deg)")
        ax.get_yaxis().set_visible(True)
        ax.get_xaxis().set_visible(True)

        yticks = [
            np.round(x, 2) for
            x in np.arange(0.05 * np.floor(self.min_auc / 0.05), 1.0, 0.05)
        ]

        xticks = [0, 90, 180, 270, 360]

        ax.set_ylim(yticks[0], yticks[-1])
        ax.set_xlim(0, 360)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

        ax.plot([-1], [0], '--k')
        for line in ax.lines:
            line.set_visible(False)
        return ax.lines

    @staticmethod
    def _update_auc_plot(ax, deg, auc):
        for line in ax.lines:
            _deg = list(line.get_xdata())
            _deg.append(180 * deg / np.pi)
            _auc = list(line.get_ydata())
            _auc.append(auc)

            line.set_xdata(_deg)
            line.set_ydata(_auc)
            line.set_visible(True)
        return ax.lines

    def frame_init(self):
        point_patches = self._init_point_clouds(self.axes[0][0])
        self._init_decision_boundaries(self.axes[0][1])
        roc_curves = self._init_roc_curve(self.axes[1][0])
        auc_plots = self._init_auc_plot(self.axes[1][1])

        return point_patches + roc_curves + auc_plots

    def update_frame(self, deg):

        rotX = self.data.get_rotated_data(deg)
        auc, fpr, tpr = self.fit_model(rotX)

        to_redraw_points = self._update_point_clouds(self.axes[0][0], rotX)
        self._update_decision_boundaries(self.axes[0][1])
        to_redraw_roc = self._update_roc_curve(self.axes[1][0], fpr, tpr)
        to_redraw_auc = self._update_auc_plot(self.axes[1][1], deg, auc)
        return to_redraw_points + to_redraw_roc + to_redraw_auc

    def animate(self, num_frames=90):

        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.tight_layout()
        self.axes = axes

        ani = FuncAnimation(
            fig,
            self.update_frame,
            frames=np.linspace(0, 2 * np.pi, num_frames),
            init_func=self.frame_init,
            blit=True,
            interval=125,
            repeat=True
        )
        return ani.to_html5_video();