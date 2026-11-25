"""Benchmark five Lasso solvers on the bundled UCI digits subset."""
import argparse
import csv
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

# Configure supported BLAS runtimes before importing NumPy.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
SHA256 = "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8"
SEEDS = (11, 29, 47)
RATIOS = (0.02, 0.10, 0.40)
METHODS = ("ISTA", "FISTA", "Restart-FISTA", "CD", "Restart+CD")
TOL = 1e-8


def read_digits():
    path = HERE / "data" / "optdigits.tes"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("The input file differs from the archived UCI copy.")
    data = np.loadtxt(path, delimiter=",", dtype=np.int16)
    assert data.shape == (1797, 65)
    return data[:, :-1].astype(float), data[:, -1]


def split_by_digit(labels, seed):
    """Proportional allocation with largest remainders and ties by digit."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels)
    target = int(np.ceil(len(labels) / 4))
    quotas = counts * target / len(labels)
    sizes = np.floor(quotas).astype(int)
    order = np.argsort(-(quotas - sizes), kind="stable")
    sizes[order[:target - sizes.sum()]] += 1
    test = np.zeros(len(labels), dtype=bool)
    for digit, size in enumerate(sizes):
        candidates = np.flatnonzero(labels == digit)
        test[rng.permutation(candidates)[:size]] = True
    return ~test, test


@dataclass
class Problem:
    G: np.ndarray       # X.T X / n
    h: np.ndarray       # X.T y / n
    yy: float           # y.T y / n
    L: float            # largest eigenvalue of G
    lam: float


def soft(v, tau):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0)


def prox_step(problem, center):
    gradient = problem.G @ center - problem.h
    return soft(center - gradient / problem.L, problem.lam / problem.L)


def cd_sweep(problem, w):
    """Exact successive minimizers, visiting features in original pixel order."""
    for j, row in enumerate(problem.G):
        diagonal = row[j]
        partial = problem.h[j] - row @ w + diagonal * w[j]
        w[j] = np.copysign(max(abs(partial) - problem.lam, 0.0), partial) / diagonal


def certificate(problem, w):
    """Compute objective and gap for the scaled-residual feasible dual point."""
    gw = problem.G @ w
    yw = problem.h @ w
    residual_sq = problem.yy - 2 * yw + w @ gw    # ||y-Xw||^2 / n
    primal = 0.5 * residual_sq + problem.lam * np.abs(w).sum()
    dual_norm = np.max(np.abs(problem.h - gw))
    scale = problem.lam / max(problem.lam, dual_norm)
    dual = scale * (problem.yy - yw) - 0.5 * scale**2 * residual_sq
    gap = primal - dual
    if gap < -1e-12:
        raise ArithmeticError("Weak duality check failed")
    return primal, max(gap, 0.0)


def solve(problem, method, tolerance):
    w = np.zeros_like(problem.h)
    z, t = w.copy(), 1.0
    restarts = 0
    started = perf_counter_ns()
    objective, gap = certificate(problem, w)
    history = [(0, 0.0, objective, gap, 0)]
    for k in range(1, 20001):
        if method == "CD":
            cd_sweep(problem, w)
            objective, gap = certificate(problem, w)
        else:
            previous = w
            w = prox_step(problem, z)
            candidate, gap = certificate(problem, w)
            if method in ("Restart-FISTA", "Restart+CD") and candidate > objective:
                # Discard the extrapolated candidate and retry from w_{k-1}.
                w = prox_step(problem, previous)
                if method == "Restart+CD":
                    # The hybrid spends one full coordinate sweep at a restart.
                    cd_sweep(problem, w)
                candidate, gap = certificate(problem, w)
                t = 1.0
                restarts += 1
            objective = candidate
            if method == "ISTA":
                z = w
            else:
                next_t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
                z = w + ((t - 1.0) / next_t) * (w - previous)
                t = next_t
        elapsed_ms = (perf_counter_ns() - started) / 1e6
        history.append((k, elapsed_ms, objective, gap, restarts))
        if gap <= tolerance:
            return w, history
    raise RuntimeError(f"{method} failed to reach tolerance {tolerance}")


def direct_check(X, y, w, lam, history):
    """Independent sample-space evaluation of the Gram-space stopping rule."""
    n = len(y)
    r = y - X @ w
    dual = r / n
    dual *= lam / max(lam, np.abs(X.T @ dual).max())
    primal = 0.5 * (r @ r) / n + lam * np.abs(w).sum()
    gap = primal - y @ dual + 0.5 * n * (dual @ dual)
    assert np.abs(X.T @ dual).max() <= lam + 1e-12
    np.testing.assert_allclose([primal, gap], history[-1][2:4], atol=2e-13, rtol=1e-12)


def write_csv(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(out, repeats):
    raw, digits = read_digits()
    labels = np.where(digits % 2, 1.0, -1.0)
    summary, traces, timings, references = [], [], [], []
    reference_weights = []
    rng = np.random.default_rng(2026)
    for seed in SEEDS:
        train, test = split_by_digit(digits, seed)
        mean, std = raw[train].mean(axis=0), raw[train].std(axis=0)
        keep = std > 0
        standard = (raw[:, keep] - mean[keep]) / std[keep]
        X = np.ascontiguousarray(standard[train])
        ybar = labels[train].mean()
        y = labels[train] - ybar
        G = X.T @ X / len(y)
        h = X.T @ y / len(y)
        eig = np.linalg.eigvalsh(G)
        for ratio in RATIOS:
            problem = Problem(G, h, y @ y / len(y), eig[-1], ratio * np.abs(h).max())
            key = dict(seed=seed, ratio=ratio)
            ref_w, ref_history = solve(problem, "CD", 1e-12)
            direct_check(X, y, ref_w, problem.lam, ref_history)
            reference = ref_history[-1][2]
            active = np.abs(ref_w) > 1e-7
            active_eigenvalues = np.linalg.eigvalsh(G[np.ix_(active, active)])
            for pixel, weight in zip(np.flatnonzero(keep), ref_w):
                reference_weights.append(dict(**key, pixel=int(pixel), weight=weight,
                                              active=int(abs(weight) > 1e-7)))
            references.append(dict(**key, ref_nonzero=int(active.sum()),
                                   active_min_eig=active_eigenvalues[0],
                                   active_max_eig=active_eigenvalues[-1], n_train=int(train.sum()), n_test=int(test.sum()),
                                   d=int(keep.sum()), lambda_max=np.abs(h).max(), lam=problem.lam,
                                   L=eig[-1], min_eigenvalue=eig[0], ref_objective=reference,
                                   ref_gap=ref_history[-1][3], ref_epochs=ref_history[-1][0]))
            # Each implementation receives one complete untimed warm-up solve.
            for method in METHODS:
                solve(problem, method, TOL)
            runs = {method: [] for method in METHODS}
            for rep in range(repeats):
                for method in rng.permutation(METHODS):
                    w, hist = solve(problem, method, TOL)
                    runs[method].append((w, hist))
                    timings.append(dict(**key, method=method, trial=rep + 1,
                                        elapsed_ms=hist[-1][1]))
            for method, trials in runs.items():
                elapsed = [hist[-1][1] for _, hist in trials]
                median_index = np.argsort(elapsed)[len(elapsed) // 2]
                w, hist = trials[median_index]
                k, ms, primal, gap, rst = hist[-1]
                for other_w, other_hist in trials:
                    assert other_hist[-1][0] == k
                    np.testing.assert_allclose(other_w, w, rtol=0, atol=1e-13)
                direct_check(X, y, w, problem.lam, hist)
                assert -2e-12 <= primal - reference <= TOL + 2e-12
                increases = sum(b[2] > a[2] + 1e-14 for a, b in zip(hist, hist[1:]))
                if method != "FISTA":
                    assert increases == 0
                scores = standard[test] @ w + ybar
                accuracy = np.mean(np.where(scores >= 0, 1, -1) == labels[test])
                rmse = np.sqrt(np.mean((scores - labels[test]) ** 2))
                summary.append(dict(**key, method=method, epochs=k, time_ms=ms, gap=gap,
                                    primal=primal, ref_difference=primal-reference, restarts=rst,
                                    increases=increases, nonzero=int(np.sum(np.abs(w) > 1e-7)),
                                    test_accuracy=accuracy, test_rmse=rmse))
                for epoch, time_ms, obj, cert, restarts in hist:
                    traces.append(dict(**key, method=method, epoch=epoch, elapsed_ms=time_ms,
                                       primal=obj, gap=cert, restarts=restarts))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summary), ("curves", traces),
                       ("timings", timings), ("problems", references),
                       ("reference_weights", reference_weights)):
        write_csv(out / f"{name}.csv", rows)
    metadata = dict(platform=platform.platform(), machine=platform.machine(),
                    python=platform.python_version(), numpy=np.__version__,
                    timing_repetitions=repeats, split_seeds=SEEDS, tolerance=TOL,
                    blas=np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}))
    (out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {len(summary)} fits, {len(timings)} timed trials. Results in {out}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--repeats", type=int, default=7, help="Odd number of timed trials")
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats % 2 == 0:
        parser.error("--repeats must be a positive odd integer")
    run(args.out, args.repeats)
