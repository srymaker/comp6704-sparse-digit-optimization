"""Read the benchmark output and check cross-split support and stopping-time comparisons."""
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def compare_supports():
    supports = defaultdict(lambda: defaultdict(set))
    for row in read_csv(ROOT / "results" / "reference_weights.csv"):
        if int(row["active"]):
            supports[row["ratio"]][row["seed"]].add(int(row["pixel"]))
    accuracy = defaultdict(list)
    for row in read_csv(ROOT / "results" / "summary.csv"):
        if row["method"] == "CD":
            accuracy[row["ratio"]].append(100 * float(row["test_accuracy"]))
    for ratio, by_split in supports.items():
        sets = list(by_split.values())
        jaccard = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
        print(f"Penalty {ratio}: common/union = {len(set.intersection(*sets))}/{len(set.union(*sets))}, "
              f"Jaccard {min(jaccard):.3f}-{max(jaccard):.3f}, "
              f"accuracy {min(accuracy[ratio]):.2f}-{max(accuracy[ratio]):.2f}%")


def compare_stopping_times():
    traces = defaultdict(list)
    for design in ("results", "mnist"):
        for row in read_csv(ROOT / design / "curves.csv"):
            case = (design, row.get("seed", "official"), row["ratio"])
            traces[case, row["method"]].append((int(row["epoch"]), float(row["gap"])))
    cases = {case for case, method in traces}
    for tolerance in (1e-4, 1e-5, 1e-6, 1e-8):
        first = {key: min(epoch for epoch, gap in series if gap <= tolerance)
                 for key, series in traces.items()}
        relative = [first[case, "FISTA"] / first[case, "ISTA"] for case in cases]
        hybrid_leads = sum(first[case, "Restart+CD"] < first[case, "Restart-FISTA"]
                           for case in cases)
        print(f"Gap {tolerance:g}: median FISTA/ISTA step ratio = {median(relative):.3f}; "
              f"hybrid reaches tolerance before restart in {hybrid_leads}/{len(cases)} cases")


if __name__ == "__main__":
    compare_supports()
    compare_stopping_times()
