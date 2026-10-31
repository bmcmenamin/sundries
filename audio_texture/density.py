"""Diagonal Gaussian mixtures, to model where a stem's frames go and how they move, as a smooth density the profile can hold in
place of frames (a handful of means and variances per stem, none of them a frame of the source).

`Mixture.fit` is EM. `blocked_scores` says how many components a stretch of audio supports: it holds out contiguous stretches
(neighbouring frames are nearly the same, so held-out frames picked at random would be copies of training frames and reward
memorising), and the variance floor is part of the same capacity control.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

DENSITY_DIMS = 16  # leading coefficients the densities cover
DENSITY_WINDOW = 8  # frames in a run of the "positions" density
VARIANCE_FLOOR = 1e-2  # of the unit variance the features are scaled to: stops a component collapsing onto a few frames
EM_ITERS = 100
EM_TOLERANCE = 1e-5  # change in mean log-likelihood per feature that counts as converged


def windows(x: torch.Tensor, length: int) -> torch.Tensor:
    """(T - length + 1, k * length): every run of `length` frames of x (k, T), flattened frame after frame."""
    return x.unfold(1, length, 1).permute(1, 2, 0).flatten(1)


def transitions(x: torch.Tensor) -> torch.Tensor:
    """(T - 1, 2k): each frame of x (k, T) with the step to the next: where it is, and where it goes from there."""
    return torch.cat([x[:, :-1], x[:, 1:] - x[:, :-1]]).T


@dataclass(frozen=True)
class Mixture:
    weights: torch.Tensor  # (M,)
    means: torch.Tensor  # (M, D)
    variances: torch.Tensor  # (M, D)

    @classmethod
    def fit(cls, x: torch.Tensor, components: int, seed: int = 0) -> Mixture:
        """EM from `components` of the frames of x (N, D), chosen at random, each with the variance of the whole."""
        picked = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:components]
        start = cls(torch.full((len(picked),), 1 / len(picked)), x[picked].clone(), x.var(0).expand(len(picked), -1).clone())
        mixture, previous = start, -math.inf
        for _ in range(EM_ITERS):
            mixture, score = mixture._step(x)
            if abs(score - previous) < EM_TOLERANCE:
                break
            previous = score
        return mixture

    def _log_joint(self, x: torch.Tensor) -> torch.Tensor:
        """(N, M): log weight + log density of each component at each frame."""
        squared = (x[:, None] - self.means).pow(2) / self.variances
        return self.weights.log() - 0.5 * (squared + self.variances.log() + math.log(2 * math.pi)).sum(-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """(N,): log density of each frame of x (N, D); differentiable in x."""
        return self._log_joint(x).logsumexp(-1)

    def _step(self, x: torch.Tensor) -> tuple[Mixture, float]:
        """One EM step, and the mean log-likelihood per feature of the mixture it started from."""
        joint = self._log_joint(x)
        total = joint.logsumexp(-1, keepdim=True)
        resp = (joint - total).exp()
        mass = resp.sum(0) + 1e-8
        means = resp.T @ x / mass[:, None]
        variances = (resp.T @ x.pow(2) / mass[:, None] - means.pow(2)).clamp(min=VARIANCE_FLOOR)
        return Mixture(mass / len(x), means, variances), float(total.mean()) / x.shape[1]

    def parameters(self) -> int:
        return self.weights.numel() - 1 + self.means.numel() + self.variances.numel()


FEATURES = {
    "positions": lambda x: windows(x, DENSITY_WINDOW),
    "transitions": transitions,
}
COMPONENTS = {"positions": 16, "transitions": 8}  # from the blocked cross-validation of `blocked_scores` on the guitar stems


@dataclass(frozen=True)
class Density:
    """A stem's frames as a mixture, kept in place of the frames: `kind` says what is modelled ("positions": runs of
    DENSITY_WINDOW frames of the leading DENSITY_DIMS coefficients; "transitions": a frame with the step to the next)."""

    kind: str
    mixture: Mixture
    mean: torch.Tensor  # (D,): the features are scaled to zero mean and unit variance over the source
    scale: torch.Tensor  # (D,)
    reference: float  # the source's own mean negative log-likelihood per feature under the mixture

    @classmethod
    def of(cls, coefficients: torch.Tensor, kind: str) -> Density:
        """The density of the source's coefficients (k, T)."""
        features = FEATURES[kind](coefficients[:DENSITY_DIMS])
        mean, scale = features.mean(0), features.std(0) + 1e-6
        mixture = Mixture.fit((features - mean) / scale, COMPONENTS[kind])
        return cls(kind, mixture, mean, scale, 0.0).measured(coefficients)

    def measured(self, coefficients: torch.Tensor) -> Density:
        return Density(self.kind, self.mixture, self.mean, self.scale, float(self.nll(coefficients)))

    def nll(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood per feature of coefficients (k, T); differentiable."""
        features = (FEATURES[self.kind](coefficients[:DENSITY_DIMS]) - self.mean) / self.scale
        return -self.mixture.log_prob(features).mean() / features.shape[1]

    def penalty(self, coefficients: torch.Tensor) -> torch.Tensor:
        """How much less likely than the source's own frames the coefficients' frames are; none when as likely or more."""
        return (self.nll(coefficients) - self.reference).clamp(min=0)

    def state(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state(cls, s: dict) -> Density:
        return cls(s["kind"], Mixture(**s["mixture"]), s["mean"], s["scale"], s["reference"])


def blocked_scores(x: torch.Tensor, spans: int, components: int, folds: int = 5, seed: int = 0) -> tuple[float, float, float]:
    """(mean, standard error, training mean) of the held-out log-likelihood per feature of a `components`-component mixture,
    over `folds` contiguous stretches of x (N, D) held out in turn. Rows within `spans` of a held-out stretch (windows that
    share frames with it) are left out of training too."""
    edges = torch.linspace(0, len(x), folds + 1).round().long()
    index = torch.arange(len(x))
    held, train = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        test = (index >= lo) & (index < hi)
        fit = Mixture.fit(x[(index < lo - spans) | (index >= hi + spans)], components, seed)
        held.append(float(fit.log_prob(x[test]).mean()) / x.shape[1])
        train.append(float(fit.log_prob(x[~test]).mean()) / x.shape[1])
    scores = torch.tensor(held)
    return float(scores.mean()), float(scores.std() / math.sqrt(folds)), float(torch.tensor(train).mean())
