"""Simulate label-private sufficient statistics for an MNIST parity Lasso."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiment import cd_sweep, soft, write_csv
from mnist_replication import read_partition
import numpy as np


HERE = Path(__file__).resolve().parent
# Append the stronger-budget sensitivity check so initial streams stay identical.
EPSILONS = (0.5, 2.0, 8.0, 0.05)
REPLICATES = 30
LAMBDA = 0.06
KKT_TOL = 1e-9
SIMULATION_ENTROPY = 67042026


def kkt_inf(G, h, w, lam):
    """Infinity norm of the coordinatewise minimum-norm Lasso subgradient."""
    grad = G @ w - h
    residual = np.where(w != 0, np.abs(grad + lam * np.sign(w)),
                        np.maximum(np.abs(grad) - lam, 0))
    return np.max(residual)


def fit_released(G, released_query, X_test, y_test):
    # No unperturbed training labels or moments enter fitting or its stop test.
    problem = SimpleNamespace(G=G, h=released_query[:-1], lam=LAMBDA)
    w = np.zeros_like(problem.h)
    for sweeps in range(1, 10001):
        cd_sweep(problem, w)
        residual = kkt_inf(G, problem.h, w, LAMBDA)
        if residual <= KKT_TOL:
            break
    else:
        raise RuntimeError("KKT tolerance was not met")
    # A separate proximal fixed-point identity checks the implementation.
    fixed = soft(w - (G @ w - problem.h), LAMBDA)
    assert np.max(np.abs(fixed - w)) <= KKT_TOL + 1e-14
    scores = X_test @ w + released_query[-1]
    return dict(sweeps=sweeps, kkt_inf=residual, nonzero=int(np.sum(np.abs(w) > 1e-7)),
                accuracy=np.mean(np.where(scores >= 0, 1, -1) == y_test),
                rmse=np.sqrt(np.mean((scores-y_test)**2)))


def simulate(cache, out):
    train, train_labels = read_partition(cache, "train", 60000)
    test, test_labels = read_partition(cache, "t10k", 10000)
    mean, std = train.mean(axis=0), train.std(axis=0)
    assert np.all(std > 0)
    X = (train-mean)/std
    X_test = (test-mean)/std
    n, d = X.shape
    G = X.T @ X / n
    assert np.linalg.eigvalsh(G)[0] > 0
    assert np.max(np.abs(X.mean(axis=0))) < 1e-12
    # This vector contains every label-dependent training statistic used below.
    q = np.r_[X.T @ train_labels / n, train_labels.mean()]
    row_norms = np.abs(X).sum(axis=1)
    extremal = np.argmax(row_norms)
    delta = (2/n) * (1 + row_norms[extremal])
    # On this public benchmark, verify sharpness at a maximizing row.
    alternate_labels = train_labels.copy()
    alternate_labels[extremal] *= -1
    alternate = np.r_[X.T @ alternate_labels / n, alternate_labels.mean()]
    np.testing.assert_allclose(np.abs(q-alternate).sum(), delta, rtol=1e-11)

    rows = [dict(epsilon="inf", draw=0, scale=0.0,
                 **fit_released(G, q, X_test, test_labels))]
    component_rows = []
    component_groups = []
    groups = [dict(epsilon="inf", count=1, scale=0.0)]
    measures = ("accuracy", "rmse", "nonzero", "sweeps")
    for metric in measures:
        for suffix in ("p25", "median", "p75"):
            groups[0][f"{metric}_{suffix}"] = rows[0][metric]

    # Distinct, reproducibly spawned streams for simulation on public MNIST.
    # A private deployment must not reveal or fix the mechanism's randomness.
    streams = np.random.SeedSequence(SIMULATION_ENTROPY).spawn(len(EPSILONS)*REPLICATES)
    for i, eps in enumerate(EPSILONS):
        scale = delta/eps
        current = []
        for j in range(REPLICATES):
            rng = np.random.default_rng(streams[i*REPLICATES+j])
            released = q + rng.laplace(0, scale, size=d + 1)
            joint = fit_released(G, released, X_test, test_labels)
            current.append(dict(epsilon=eps, draw=j+1, scale=scale, **joint))
            # Public-benchmark diagnostics: restore one exact component while
            # keeping the identical draw in the other. Neither hybrid release
            # implements the specified joint label-private mechanism.
            for term, query in (("h_only", np.r_[released[:-1], q[-1]]),
                                ("b_only", np.r_[q[:-1], released[-1]])):
                fit = fit_released(G, query, X_test, test_labels)
                component_rows.append(dict(epsilon=eps, draw=j+1, perturbed=term,
                                           **fit, paired_accuracy_minus_joint=
                                           fit["accuracy"] - joint["accuracy"],
                                           paired_rmse_minus_joint=fit["rmse"] - joint["rmse"]))
        rows.extend(current)
        aggregate = dict(epsilon=eps, count=REPLICATES, scale=scale)
        for metric in measures:
            values = [r[metric] for r in current]
            for suffix, sample in zip(("p25", "median", "p75"),
                                      np.quantile(values, [.25, .5, .75])):
                aggregate[f"{metric}_{suffix}"] = sample
        groups.append(aggregate)
        for term in ("h_only", "b_only"):
            subset = [r for r in component_rows if r["epsilon"] == eps and r["perturbed"] == term]
            agg = dict(epsilon=eps, perturbed=term, count=len(subset))
            for metric in (*measures, "paired_accuracy_minus_joint", "paired_rmse_minus_joint"):
                values = [r[metric] for r in subset]
                for suffix, val in zip(("p25", "median", "p75"), np.quantile(values, [.25, .5, .75])):
                    agg[f"{metric}_{suffix}"] = val
            component_groups.append(agg)

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"trials.csv", rows)
    write_csv(out/"summary.csv", groups)
    write_csv(out/"component_trials.csv", component_rows)
    write_csv(out/"component_summary.csv", component_groups)
    design = dict(n_train=n, n_test=len(test_labels), pixels=d, query_dimension=d+1,
                  max_training_feature_l1=row_norms[extremal], label_replacement_l1=delta,
                  measured_max_row_flip_l1=np.abs(q-alternate).sum(), lambda_absolute=LAMBDA,
                  kkt_tolerance=KKT_TOL, max_observed_kkt=max(r["kkt_inf"] for r in rows),
                  epsilon_values=EPSILONS, draws_per_epsilon=REPLICATES,
                  simulation_root_entropy=SIMULATION_ENTROPY)
    (out/"design.json").write_text(json.dumps(design, indent=2) + "\n")
    print(f"Completed {len(rows)} main fits and {len(component_rows)} paired component diagnostics. Wrote {out}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE/".mnist-cache")
    parser.add_argument("--out", type=Path, default=HERE/"privacy")
    args = parser.parse_args()
    simulate(args.cache, args.out)
