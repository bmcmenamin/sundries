"""Texture generation in the non-negative (NMF-style) code of perceptually weighted amplitude maps.

    venv/bin/python nmf_texture.py walk_1 piano --n 2

Needs out/nmf/01_dictionary/<input>/<stem>/dictionary.pt (nmf_code.py). Same two stages as
ampfreq_texture.py (code statistics incl. cross-feature population sparsity, then a linear
histogram critic on the decoded maps), but the code is non-negative by construction
(optimised through a square), so generated sounds are sums of parts, never differences.
Writes out/nmf/02_generate/<input>/<stem>/<uuid>.wav.
"""

from __future__ import annotations

import argparse

import torch

from ampfreq import FREQ_CLIP
from ampfreq_code import Operator
from ampfreq_texture import (
    ENCODE_ITERS, AmplitudeReference, Target, generate_code, map_features, refine_with_critic,
)
from histogram_critic import FeatureConfig
from nmf_code import amplitude_maps, encode, fit_weights, hybrid_coefficients, phase_policy
from nmf_sweep import activation_report
from paths import NMF, Stem, new_sample_path, step_dir
from synthesize import SAMPLE_RATE, decode_mono, write_wav
from wavelet import Wavelet
from wavelet_code import trim
from wavelet_texture import CodeStats


def critic_view(maps: torch.Tensor) -> torch.Tensor:
    """(P, P * offset) -> (compressed amplitude, frequency offset), the maps the histogram features read."""
    amplitude = maps[0].clamp(min=0)
    return torch.stack([(amplitude + 1e-6).sqrt(), (maps[1] / (amplitude + 1e-3)).clamp(-FREQ_CLIP, FREQ_CLIP)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, e.g. walk_1")
    parser.add_argument("stem", help="stem name, e.g. piano")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--iters", type=int, default=300, help="stage 1 (code statistics) iterations")
    parser.add_argument("--hist-iters", type=int, default=100, help="stage 2 (histogram critic) iterations")
    parser.add_argument("--hist-weight", type=float, default=1.0)
    parser.add_argument("--hist-ridge", type=float, default=0.3)
    parser.add_argument("--hist-group", type=int, default=4)
    parser.add_argument("--hist-bins", type=int, default=20)
    parser.add_argument("--hist-joint-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="", help="dictionary step suffix used by nmf_code.py, e.g. k4")
    args = parser.parse_args()

    stem = Stem(args.source, args.stem)
    suffix = f"_{args.tag}" if args.tag else ""
    directory = step_dir(NMF, "01_dictionary" + suffix, args.source, args.stem)
    saved = torch.load(directory / "dictionary.pt")
    atoms, scales, lam_rel, stride = saved["atoms"], saved["scales"], saved["lam_rel"], saved["stride"]

    audio = decode_mono(stem.path)
    audio = trim(audio / audio.pow(2).mean().sqrt())
    source_maps = amplitude_maps(Wavelet.for_length(audio.numel()).analyse(audio))
    scaled = source_maps / scales

    cache = directory / "codes.pt"
    if cache.exists():
        source_code = torch.load(cache)
    else:
        operator = Operator.build(atoms, (scaled.shape[1], scaled.shape[2]), stride)
        weights = fit_weights(scaled[0], saved.get("weighting", "masking"))
        lam = lam_rel * float(operator.analyse(weights * scaled).max())
        source_code = encode(scaled, weights, operator, lam, 6 * 25)
        torch.save(source_code, cache)
    code_target = Target.from_code(source_code)
    keep = phase_policy(stem, directory)
    print(f"{args.stem}: keeping integrated phase in {int(keep.sum())}/{keep.numel()} bands, noise phase in the rest")

    n_samples = trim(torch.zeros(round(args.seconds * SAMPLE_RATE))).numel()
    wavelet = Wavelet.for_length(n_samples)
    operator = Operator.build(atoms, (scaled.shape[1], wavelet.frames), stride)
    view = critic_view(source_maps)
    reference = AmplitudeReference.from_maps(view)
    config = FeatureConfig(args.hist_group, args.hist_bins, args.hist_joint_bins)

    def features_of(code: torch.Tensor) -> torch.Tensor:
        return map_features(critic_view(operator.decode(code) * scales), reference, config)

    with torch.no_grad():
        real_features = map_features(view, reference, config)
    print(f"{args.stem}: {real_features.shape[1]} histogram features, {real_features.shape[0]} real windows", flush=True)

    stage = "" if args.hist_iters else "_stats_only"  # no critic: code statistics + population sparsity only
    out_dir = step_dir(NMF, "02_generate" + stage + suffix, args.source, args.stem)
    for i in range(args.n):
        seed = args.seed + i
        code = generate_code(code_target, operator.code_shape, args.iters, seed, positive=True)
        report = ""
        if args.hist_iters:
            code, (before, after) = refine_with_critic(
                code, code_target, real_features, features_of, args.hist_iters, args.hist_weight, args.hist_ridge,
                positive=True,
            )
            report = f"  critic {before:.2f} -> {after:.2f}"
        with torch.no_grad():
            out = new_sample_path(out_dir)
            decoded = operator.decode(code) * scales
            write_wav(out, wavelet.synthesize(hybrid_coefficients(decoded, keep, wavelet, seed)))
            torch.save(code, out.with_suffix(".pt"))  # lets nmf_render.py re-render this sample with other phase choices
            terms = {k: round(float(v), 3) for k, v in CodeStats.from_code(code).terms(code_target.code).items()}
        print(f"{out}  {terms}{report}", flush=True)
        for label, c in (("source", source_code), ("generated", code)):
            r = activation_report(c)
            print(
                f"  {label:9s} off {r['off']:.1f}%  mid {r['mid']:.2f}%  on {r['on']:.2f}%  clean {r['clean']:.2f}  "
                f"eta {r['eta']:.2f}  clearly-on per frame {r['per_frame']:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
