"""Event VAE: learn a generator for the events of a stem, then place sampled events on a timeline.

    venv/bin/python event_vae.py walk_1 --stems guitar --post 300
    venv/bin/python event_vae.py hanging_1 --stems drums --post 225

Per stem:
  1. parse events: onsets from the amplitude map, one patch of PRE + post frames around each (raw, so events may
     overlap and carry their neighbours, as they do in the stem)
  2. features per event, perceptually compressed:
       sqrt amplitude / event rms   loudness-like compression, gain stored apart
       frequency offset             phase gradient along time      (weighted by amplitude in the loss)
       group delay / DELAY_SCALE    phase gradient along frequency (weighted by amplitude in the loss)
  3. convolutional VAE on the (3, bands, frames) patches, trained with band-shift (transposition) and small time
     jitter augmentation; `--beta` keeps the latent smooth enough to sample from with so few events
  4. generate: draw latents, decode, give each event a gain from the stem's own gain distribution, place on a
     timeline (shuffled inter-onset intervals for now), blend overlaps by tapered weights, rebuild phase
     (plain.render_coefficients) and synthesise
Writes out/event_vae/01_train_<tag>/<input>/<stem>/model.pt, 02_generate_<tag>/.../<uuid>.wav (10 s samples) and
03_resynthesis_<tag>/.../<uuid>.wav (the stem's own onsets and gains, sampled events instead of the real ones).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import plain
from paths import EVENT_VAE, Stem, new_sample_path, step_dir, stems_of
from phase_structure import z_scores
from synthesize import SAMPLE_RATE, write_wav
from wavelet import N_BANDS, Wavelet
from wavelet_code import trim

PRE = 24  # frames kept before the onset
FADE = 24  # frames of taper at each end of a placed event
GAP = 60  # onsets closer than this (40 ms) keep only the strongest
LAG = 4
DELAY_SCALE = 20.0  # samples per unit of the delay feature
MAX_SHIFT = 12  # augmentation: band shifts of +-12 (one octave)
POOL = 4  # time step of the event features, in frames (375 per second)
ENVELOPE = 5  # frames averaged into each feature step
JITTER = 2  # augmentation: time shifts of +-2 feature steps (+-8 frames)
TONAL_WEIGHT = 4.0  # extra loss weight on bands with structured phase: the ring must not be blurred away
BANDS_PADDED = 80  # N_BANDS rounded up to a multiple of 16 for four stride-2 convolutions


def detect_onsets(amplitude: torch.Tensor, z: float) -> torch.Tensor:
    """Frames where the broadband amplitude jumps: peaks of the positive sqrt-amplitude flux above z sigma."""
    root = amplitude.sqrt()
    flux = (root[:, LAG:] - root[:, :-LAG]).clamp(min=0).sum(0)
    peak = F.max_pool1d(flux[None, None], 2 * GAP + 1, stride=1, padding=GAP)[0, 0]
    return ((flux == peak) & (flux > flux.mean() + z * flux.std())).nonzero().flatten() + LAG // 2


def shift_bands(x: torch.Tensor, n: int) -> torch.Tensor:
    """Content moved up `n` bands (down if negative), zero fill; (..., B, T)."""
    out = torch.zeros_like(x)
    if n == 0:
        return x
    if n > 0:
        out[..., n:, :] = x[..., :-n, :]
    else:
        out[..., :n, :] = x[..., -n:, :]
    return out


def shift_time(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    if n == 0:
        return x
    if n > 0:
        out[..., n:] = x[..., :-n]
    else:
        out[..., :n] = x[..., -n:]
    return out


@dataclass(frozen=True)
class Events:
    shapes: torch.Tensor  # (E, 3, B, T): sqrt(amplitude / gain), frequency offset, delay / DELAY_SCALE
    gains: torch.Tensor  # (E,) rms amplitude of each patch
    starts: torch.Tensor  # (E,) onset frame in the stem
    band_weight: torch.Tensor  # (B, 1) loss weight: 1 / the band's rms, so quiet low bands count as much as loud ones
    tonal: torch.Tensor  # (B,) bool: bands whose phase is structured (the ring of a drum body)

    @property
    def span(self) -> int:
        return self.shapes.shape[-1]

    @classmethod
    def from_stem(cls, stem: Stem, post: int, z: float) -> Events:
        """Smooth envelope features at 1 / POOL of the frame rate: the stem's fine structure (cell-to-cell amplitude
        flicker) is noise-like, cannot be predicted from a latent, and is regenerated at render time instead."""
        audio, wavelet = plain.load_stem(stem)
        maps = plain.plain_maps(wavelet.analyse(audio))
        pool = lambda x: F.avg_pool2d(x[None, None], (1, ENVELOPE), stride=(1, POOL), padding=(0, ENVELOPE // 2), count_include_pad=False)[0, 0]  # noqa: E731
        weight = pool(maps[0]).clamp(min=1e-9)
        amplitude = pool(maps[0].pow(2)).sqrt()
        live = amplitude > 0.02 * amplitude.mean()
        features = torch.stack([
            amplitude, pool(maps[1]) / weight * live, pool(maps[2]) / weight / DELAY_SCALE * live
        ])
        starts = detect_onsets(maps[0], z)
        pre, span = PRE // POOL, (PRE + post) // POOL
        padded = F.pad(features, (span, span))
        patches = torch.stack([padded[..., int(s) // POOL - pre + span : int(s) // POOL - pre + 2 * span] for s in starts])
        gains = patches[:, 0].pow(2).mean((1, 2)).sqrt()
        shapes = torch.cat([(patches[:, :1] / gains[:, None, None, None]).sqrt(), patches[:, 1:]], dim=1)
        rms = shapes[:, 0].pow(2).mean((0, 2)).sqrt()
        tonal = z_scores(stem)["structure"] > plain.STRUCTURE_Z
        band_weight = (1 / (rms + 0.1 * rms.mean())) * torch.where(tonal, TONAL_WEIGHT, 1.0)
        return cls(shapes, gains, starts, (band_weight / band_weight.mean())[:, None], tonal)

    def batch(self, size: int, generator: torch.Generator, max_shift: int) -> torch.Tensor:
        """Random events with a random band shift (up to `max_shift`) and time jitter each."""
        picks = torch.randint(0, len(self.starts), (size,), generator=generator)
        shifts = torch.randint(-max_shift, max_shift + 1, (size,), generator=generator)
        jitters = torch.randint(-JITTER, JITTER + 1, (size,), generator=generator)
        return torch.stack([
            shift_time(shift_bands(self.shapes[int(p)], int(s)), int(j)) for p, s, j in zip(picks, shifts, jitters)
        ])


class EventVAE(nn.Module):
    def __init__(self, frames: int, latent: int = 16, width: int = 16) -> None:
        super().__init__()
        self.frames = frames
        self.padded = -(-frames // 16) * 16
        flat = 4 * width * (BANDS_PADDED // 16) * (self.padded // 16)
        down = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.GELU())  # noqa: E731
        up = lambda i, o: nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(i, o, 3, padding=1), nn.GELU())  # noqa: E731
        self.encoder = nn.Sequential(down(3, width), down(width, 2 * width), down(2 * width, 4 * width), down(4 * width, 4 * width), nn.Flatten())
        self.to_latent = nn.Linear(flat, 2 * latent)
        self.from_latent = nn.Linear(latent, flat)
        self.grid = (4 * width, BANDS_PADDED // 16, self.padded // 16)
        self.decoder = nn.Sequential(up(4 * width, 4 * width), up(4 * width, 2 * width), up(2 * width, width), up(width, width), nn.Conv2d(width, 3, 1))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padded = F.pad(x, (0, self.padded - self.frames, 0, BANDS_PADDED - N_BANDS))
        return self.to_latent(self.encoder(padded)).chunk(2, dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(self.from_latent(z).unflatten(-1, self.grid))[:, :, :N_BANDS, : self.frames]
        return torch.cat([F.softplus(raw[:, :1]), raw[:, 1:]], dim=1)


def log_profile(shape: torch.Tensor) -> torch.Tensor:
    """(N, B): log share of an event's energy in each band (energy = amplitude^2 = shape^4, summed over time)."""
    energy = shape.pow(4).sum(-1)
    return torch.log(energy / energy.sum(-1, keepdim=True) + 1e-4)


def elbo(
    model: EventVAE, x: torch.Tensor, sigma: float, beta: float, profile_weight: float, band_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loss, reconstruction and KL (per event). Frequency and delay errors count where the event has amplitude.

    The band-profile term keeps the spectral balance: squared errors in sqrt amplitude blur the loud, narrow
    parts of an event (a kick's low bands) into its many quiet cells, which flattens the spectrum.
    """
    mu, logvar = model.encode(x)
    y = model.decode(mu + torch.randn_like(mu) * (0.5 * logvar).exp())
    weight = x[:, :1].detach() * band_weight
    reconstruction = (
        (band_weight * (y[:, 0] - x[:, 0]).pow(2)).sum((1, 2)) + (weight * (y[:, 1:] - x[:, 1:]).pow(2)).sum((1, 2, 3))
    ) / (2 * sigma**2)
    profile = (log_profile(y[:, 0]) - log_profile(x[:, 0])).pow(2).sum(-1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
    return (reconstruction + beta * kl + profile_weight * profile).mean(), reconstruction.mean(), kl.mean()


@dataclass(frozen=True)
class Prior:
    """Gaussian fitted to where the encoder puts real (augmented) events: sampling it lands where the decoder is trained."""

    mean: torch.Tensor
    factor: torch.Tensor  # Cholesky factor of the covariance

    @classmethod
    def fit(cls, model: EventVAE, events: Events, generator: torch.Generator, max_shift: int, batches: int = 40) -> Prior:
        with torch.no_grad():
            mus = torch.cat([model.encode(events.batch(32, generator, max_shift))[0] for _ in range(batches)])
        covariance = torch.cov(mus.T) + 1e-3 * torch.eye(mus.shape[1])
        return cls(mus.mean(0), torch.linalg.cholesky(covariance))

    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        return self.mean + torch.randn(n, len(self.mean), generator=generator) @ self.factor.T


def train(
    events: Events, latent: int, steps: int, beta: float, sigma: float, profile_weight: float, max_shift: int, seed: int
) -> EventVAE:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = EventVAE(events.span, latent)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        warm = min(1.0, step / (steps / 5))  # KL warm-up
        loss, reconstruction, kl = elbo(
            model, events.batch(16, generator, max_shift), sigma, beta * warm, profile_weight, events.band_weight
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"  step {step:5d}  reconstruction {reconstruction.item():9.1f}  KL {kl.item():6.1f}", flush=True)
    return model.eval()


def taper(frames: int) -> torch.Tensor:
    t = torch.arange(frames).float()
    rise = torch.sin((t / FADE).clamp(0, 1) * math.pi / 2).pow(2)
    fall = torch.sin(((frames - 1 - t) / FADE).clamp(0, 1) * math.pi / 2).pow(2)
    return rise * fall


def to_maps(shapes: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """(N, 3, B, T) decoded shapes and (N,) gains -> (N, 3, B, T * POOL) maps in plain_maps units."""
    shapes = F.interpolate(shapes.flatten(0, 1), scale_factor=POOL, mode="linear").unflatten(0, shapes.shape[:2])
    amplitude = gains[:, None, None] * shapes[:, 0].pow(2)
    return torch.stack([amplitude, amplitude * shapes[:, 1], amplitude * shapes[:, 2] * DELAY_SCALE], dim=1)


def fine_structure(wavelet: Wavelet, seed: int) -> torch.Tensor:
    """(B, W) cell-to-cell amplitude flicker of band-limited noise, unit rms in every band (white noise through
    constant-Q bands is blue, so each band is levelled)."""
    noise = wavelet.analyse(torch.randn(wavelet.n_samples, generator=torch.Generator().manual_seed(seed))).abs()
    return noise / noise.pow(2).mean(-1, keepdim=True).sqrt()


def compose(patches: torch.Tensor, starts: torch.Tensor, frames: int) -> torch.Tensor:
    """(N, 3, B, T) patches placed with their pre-roll at `starts`; overlaps are blended by tapered weights."""
    span = patches.shape[-1]
    weight = taper(span)
    numerator = torch.zeros(3, N_BANDS, frames + 2 * span)
    denominator = torch.zeros(frames + 2 * span)
    for patch, start in zip(patches, starts):
        low = int(start) - PRE + span
        numerator[..., low : low + span] += patch * weight
        denominator[low : low + span] += weight
    return (numerator / denominator.clamp(min=1))[..., span : span + frames]


def render(maps: torch.Tensor, mode: str, wavelet: Wavelet, seed: int, tonal: torch.Tensor) -> torch.Tensor:
    """Smooth event maps -> complex coefficients. Noisy stems get band-limited noise flicker and noise phase back
    (the flicker also scales the amplitude-weighted channels), except in their `tonal` bands, which keep smooth
    amplitude and integrated phase so a drum body still rings. Tonal stems stay smooth throughout."""
    if mode != "noise":
        return plain.render_coefficients(maps, mode, wavelet, seed)
    noisy = plain.render_coefficients(maps * fine_structure(wavelet, seed), "noise", wavelet, seed)
    ringing = plain.render_coefficients(maps, "integrated", wavelet, seed)
    return torch.where(tonal[:, None], ringing, noisy)


def shuffled_starts(starts: torch.Tensor, frames: int, generator: torch.Generator) -> torch.Tensor:
    """Onsets for a new timeline of `frames`: the stem's inter-onset intervals, drawn with replacement."""
    intervals = (starts[1:] - starts[:-1]).float()
    draws = intervals[torch.randint(0, len(intervals), (frames // int(intervals.min()) + 2,), generator=generator)]
    times = PRE + torch.cumsum(draws, 0)
    return times[times < frames].long()


def draw_gains(gains: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    log = gains.log()
    return (log.mean() + log.std() * torch.randn(n, generator=generator)).exp()


def nearest_distance(shapes: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """(N,) relative L2 distance of the sqrt-amplitude channel to the closest reference event, over band shifts."""
    ref = reference[:, 0]
    best = torch.full((len(shapes),), float("inf"))
    for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
        moved = shift_bands(ref, shift)
        dist = torch.cdist(shapes[:, 0].flatten(1), moved.flatten(1)) / moved.flatten(1).norm(dim=1)[None]
        best = torch.minimum(best, dist.amin(1))
    return best


def report(model: EventVAE, events: Events, prior: Prior, generator: torch.Generator) -> None:
    """How different are sampled events from the real ones? Real events set the scale: each one's distance to the
    nearest *other* real event. Reconstructions show what the model keeps of an event it was shown."""
    with torch.no_grad():
        mu, _ = model.encode(events.shapes)
        reconstructed = model.decode(mu)
        sampled = model.decode(prior.sample(len(events.starts), generator))
        leave_one_out = torch.stack([
            nearest_distance(events.shapes[i : i + 1], torch.cat([events.shapes[:i], events.shapes[i + 1 :]]))[0]
            for i in range(len(events.starts))
        ])
        own = (reconstructed[:, 0] - events.shapes[:, 0]).flatten(1).norm(dim=1) / events.shapes[:, 0].flatten(1).norm(dim=1)
    print(f"  distance to nearest real event (median): other real events {float(leave_one_out.median()):.2f}   "
          f"sampled events {float(nearest_distance(sampled, events.shapes).median()):.2f}   "
          f"reconstruction error of a shown event {float(own.median()):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, a folder under out/separation/01_demucs/")
    parser.add_argument("--stems", nargs="+", help="stems to model (default: all)")
    parser.add_argument("--post", type=int, default=225, help="frames kept after the onset (1500 per second)")
    parser.add_argument("--z", type=float, default=2.0, help="onset threshold in standard deviations of the flux")
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--max-shift", type=int, default=12, help="band-shift augmentation; 0 for fixed-pitch stems like drums")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=10.0, help="KL weight; samples come from a Gaussian fitted to the posterior")
    parser.add_argument("--sigma", type=float, default=1.0, help="assumed reconstruction noise")
    parser.add_argument("--profile-weight", type=float, default=50.0, help="weight of the per-band energy share loss")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wavelet = Wavelet.for_length(trim(torch.zeros(round(args.seconds * SAMPLE_RATE))).numel())
    tag = f"z{args.latent}_b{args.beta:g}_w{args.profile_weight:g}_p{args.post}_s{POOL}_m{args.max_shift}"
    for stem in (s for s in stems_of(args.source) if args.stems is None or s.name in args.stems):
        events = Events.from_stem(stem, args.post, args.z)
        print(f"{stem.source}/{stem.name}: {len(events.starts)} events of {events.span} frames", flush=True)
        directory = step_dir(EVENT_VAE, f"01_train_{tag}", stem.source, stem.name)
        if (directory / "model.pt").exists():
            model = EventVAE(events.span, args.latent).eval()
            model.load_state_dict(torch.load(directory / "model.pt"))
        else:
            model = train(events, args.latent, args.steps, args.beta, args.sigma, args.profile_weight, args.max_shift, args.seed)
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), directory / "model.pt")
        generator = torch.Generator().manual_seed(args.seed)
        prior = Prior.fit(model, events, generator, args.max_shift)
        report(model, events, prior, generator)

        mode, _ = plain.detect_phase_mode(stem)
        audio, own_wavelet = plain.load_stem(stem)
        with torch.no_grad():
            for i in range(args.n):
                # resynthesis: the stem's own onsets and gains, sampled events
                z = prior.sample(len(events.starts), generator)
                maps = compose(to_maps(model.decode(z), events.gains), events.starts, own_wavelet.frames)
                path = new_sample_path(step_dir(EVENT_VAE, f"03_resynthesis_{tag}", stem.source, stem.name))
                write_wav(path, own_wavelet.synthesize(render(maps, mode, own_wavelet, args.seed + i, events.tonal)))
                # generation: a new timeline from shuffled inter-onset intervals, gains from the stem's distribution
                starts = shuffled_starts(events.starts, wavelet.frames, generator)
                z = prior.sample(len(starts), generator)
                maps = compose(to_maps(model.decode(z), draw_gains(events.gains, len(starts), generator)), starts, wavelet.frames)
                out = new_sample_path(step_dir(EVENT_VAE, f"02_generate_{tag}", stem.source, stem.name))
                write_wav(out, wavelet.synthesize(render(maps, mode, wavelet, args.seed + i, events.tonal)))
                print(f"  {out}  ({len(starts)} events)", flush=True)


if __name__ == "__main__":
    main()
