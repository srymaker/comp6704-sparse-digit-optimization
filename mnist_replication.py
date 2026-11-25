"""Repeat the Lasso comparison on all examples in the official MNIST split."""

import argparse
import gzip
import hashlib
import json
import platform
import struct
import tempfile
import urllib.request
from pathlib import Path
from time import perf_counter

# experiment configures single-thread BLAS before importing NumPy.
from experiment import METHODS, RATIOS, TOL, Problem, direct_check, solve, write_csv
import numpy as np


HERE = Path(__file__).resolve().parent
BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# TensorFlow Datasets' published SHA-256 checksums for the CVDF gzip files.
DIGESTS = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def read_file(cache, name):
    """Validate a cached archive or retrieve and verify it before saving."""
    archive = cache / name
    if archive.exists():
        compressed = archive.read_bytes()
    else:
        with urllib.request.urlopen(BASE + name, timeout=90) as response:
            compressed = response.read()
    if hashlib.sha256(compressed).hexdigest() != DIGESTS[name]:
        raise ValueError(f"Incorrect SHA-256 for {name}. Check the downloaded archive.")
    if not archive.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
            tmp.write(compressed)
        Path(tmp.name).replace(archive)
    return gzip.decompress(compressed)


def read_partition(cache, split, n):
    image_bytes = read_file(cache, f"{split}-images-idx3-ubyte.gz")
    label_bytes = read_file(cache, f"{split}-labels-idx1-ubyte.gz")
    if struct.unpack_from(">IIII", image_bytes) != (2051, n, 28, 28):
        raise ValueError(f"Invalid image header for {split}")
    if struct.unpack_from(">II", label_bytes) != (2049, n):
        raise ValueError(f"Invalid label header for {split}")
    images = np.frombuffer(image_bytes, np.uint8, offset=16).reshape(n, 7, 4, 7, 4)
    digits = np.frombuffer(label_bytes, np.uint8, offset=8).reshape(n)
    # Each 7x7 feature is the mean intensity of one nonoverlapping 4x4 block.
    features = images.mean(axis=(2, 4)).reshape(n, 49)
    return features, np.where(digits % 2, 1.0, -1.0)


def run(cache, out, repeats):
    train, raw_y = read_partition(cache, "train", 60000)
    test, test_y = read_partition(cache, "t10k", 10000)
    started = perf_counter()
    mu, sd = train.mean(axis=0), train.std(axis=0)
    keep = sd > 0
    X = np.ascontiguousarray((train[:, keep] - mu[keep]) / sd[keep])
    X_test = np.ascontiguousarray((test[:, keep] - mu[keep]) / sd[keep])
    ybar = raw_y.mean()
    y = raw_y - ybar
    n = len(y)
    G, h, yy = X.T @ X / n, X.T @ y / n, y @ y / n
    eig = np.linalg.eigvalsh(G)
    lmax = np.abs(h).max()
    prep_seconds = perf_counter() - started
    rng = np.random.default_rng(2026)
    summary, timings, curves, problems = [], [], [], []
    for ratio in RATIOS:
        prob = Problem(G, h, yy, eig[-1], ratio * lmax)
        ref_w, ref_hist = solve(prob, "CD", 1e-12)
        direct_check(X, y, ref_w, prob.lam, ref_hist)
        ref = ref_hist[-1][2]
        active = np.abs(ref_w) > 1e-7
        active_eigs = np.linalg.eigvalsh(G[np.ix_(active, active)])
        problems.append(dict(ratio=ratio, n_train=n, n_test=len(test_y), d=X.shape[1],
                             lambda_max=lmax, lam=prob.lam, L=eig[-1], min_eigenvalue=eig[0],
                             ref_objective=ref, ref_gap=ref_hist[-1][3], ref_nonzero=int(active.sum()),
                             active_min_eig=active_eigs[0], active_max_eig=active_eigs[-1]))
        for method in METHODS:
            solve(prob, method, TOL)
        trials = {method: [] for method in METHODS}
        for rep in range(repeats):
            for method in rng.permutation(METHODS):
                w, hist = solve(prob, method, TOL)
                trials[method].append((w, hist))
                timings.append(dict(ratio=ratio, method=method, trial=rep+1, elapsed_ms=hist[-1][1]))
        for method, runs in trials.items():
            w, hist = sorted(runs, key=lambda run: run[1][-1][1])[repeats // 2]
            k, ms, obj, gap, rst = hist[-1]
            for other, history in runs:
                assert history[-1][0] == k
                np.testing.assert_allclose(other, w, atol=1e-13, rtol=0)
            direct_check(X, y, w, prob.lam, hist)
            assert -2e-12 <= obj - ref <= TOL + 2e-12
            rises = sum(b[2] > a[2] + 1e-14 for a, b in zip(hist, hist[1:]))
            assert method == "FISTA" or rises == 0
            scores = X_test @ w + ybar
            accuracy = np.mean(np.where(scores >= 0, 1, -1) == test_y)
            summary.append(dict(ratio=ratio, method=method, epochs=k, time_ms=ms, gap=gap,
                                primal=obj, ref_difference=obj-ref, restarts=rst, increases=rises,
                                nonzero=int((np.abs(w) > 1e-7).sum()), test_accuracy=accuracy,
                                test_rmse=np.sqrt(np.mean((scores-test_y)**2))))
            for epoch, elapsed, primal, gap, restarts in hist:
                curves.append(dict(ratio=ratio, method=method, epoch=epoch, elapsed_ms=elapsed,
                                   primal=primal, gap=gap, restarts=restarts))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summary), ("problems", problems),
                       ("curves", curves), ("timings", timings)):
        write_csv(out / f"{name}.csv", rows)
    meta = dict(dataset="MNIST, original 60,000/10,000 split", input_checksums=DIGESTS,
                features="7x7 nonoverlapping 4x4 arithmetic means from 28x28 8-bit input",
                train_odd=int((raw_y > 0).sum()), test_odd=int((test_y > 0).sum()),
                preprocessing_and_gram_seconds=prep_seconds, methods=METHODS,
                tolerance=TOL, timing_repetitions=repeats, python=platform.python_version(),
                numpy=np.__version__, platform=platform.platform(),
                blas=np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}))
    (out / "environment.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Completed {len(summary)} fits and {len(timings)} timed trials in {out}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / ".mnist-cache")
    parser.add_argument("--out", type=Path, default=HERE / "mnist")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats % 2 == 0:
        parser.error("--repeats must be a positive odd integer")
    run(args.cache, args.out, args.repeats)
