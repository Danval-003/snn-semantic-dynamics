from __future__ import annotations

import argparse
import json
from pathlib import Path

from .e11 import bootstrap_interval


SEEDS = (7, 19, 31, 43, 59)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired(values_snn, values_ann, samples=10000):
    return {
        "snn": bootstrap_interval(values_snn, samples),
        "ann": bootstrap_interval(values_ann, samples),
        "snn_minus_ann": bootstrap_interval(
            [left - right for left, right in zip(values_snn, values_ann, strict=True)], samples
        ),
    }


def run(
    output="reports/benchmark_summary.json",
    raw_output="reports/raw_metrics.jsonl",
):
    snn_runs = [read(f"runs/e12b/heterogeneous_learnable/seed_{seed}.json") for seed in SEEDS]
    ann_runs = [read(f"runs/ann/seed_{seed}.json") for seed in SEEDS]
    snn_factorial = [read(f"runs/factorial/seed_{seed}.json") for seed in SEEDS]
    snn_order = [read(f"runs/e12c/seed_{seed}.json")["evaluation"] for seed in SEEDS]
    gen_snn = [read(f"runs/generalization/snn/context_exposed/seed_{seed}.json") for seed in SEEDS]
    gen_ann = [read(f"runs/generalization/ann/context_exposed/seed_{seed}.json") for seed in SEEDS]
    gen_snn_control = [
        read(f"runs/generalization/snn/no_context/seed_{seed}.json") for seed in SEEDS
    ]
    gen_ann_control = [
        read(f"runs/generalization/ann/no_context/seed_{seed}.json") for seed in SEEDS
    ]

    metrics = {}
    metrics["iid_population_sta"] = paired(
        [r["evaluation"]["subsets"]["all"][2]["rate_only"]["sta"] for r in snn_runs],
        [r["evaluation"]["subsets"]["all"][2]["integrated"]["sta"] for r in ann_runs],
    )
    metrics["hard_orthographic_population_sta"] = paired(
        [
            r["evaluation"]["subsets"]["hard_orthographic"][2]["rate_only"]["sta"]
            for r in snn_runs
        ],
        [
            r["evaluation"]["subsets"]["hard_orthographic"][2]["integrated"]["sta"]
            for r in ann_runs
        ],
    )
    metrics["factorial_abstraction_index_l3"] = paired(
        [r["effects"]["rate"][2]["abstraction_index_relative"] for r in snn_factorial],
        [r["factorial"]["effects"]["rate"][2]["abstraction_index_relative"] for r in ann_runs],
    )
    snn_order_drop = [
        r["correct"]["all"][2]["rate_only"]["sta"]
        - r["permuted_order"]["all"][2]["rate_only"]["sta"]
        for r in snn_order
    ]
    ann_order_drop = [
        r["evaluation"]["subsets"]["all"][2]["integrated"]["sta"]
        - r["evaluation"]["permuted_integrated_sta"]
        for r in ann_runs
    ]
    metrics["ordered_input_dependency_drop"] = paired(snn_order_drop, ann_order_drop)
    metrics["relation_disjoint_population_sta"] = paired(
        [r["evaluation"]["relation_disjoint"]["population"]["sta"] for r in gen_snn],
        [r["evaluation"]["relation_disjoint"]["population"]["sta"] for r in gen_ann],
    )
    metrics["contextual_lexical_transfer_population_sta"] = paired(
        [r["evaluation"]["lexical_context_transfer"]["population"]["sta"] for r in gen_snn],
        [r["evaluation"]["lexical_context_transfer"]["population"]["sta"] for r in gen_ann],
    )
    ann_activity = [
        r["evaluation"]["activity"][2]["active_fraction_1e-3"] for r in ann_runs
    ]
    e11_runs = [read(f"runs/e11/seed_{seed}.json") for seed in SEEDS]
    snn_spike_rates = [row["evaluation"]["all"]["spike_rate"] for row in e11_runs]
    summary = {
        "experiment": "E1-consolidated-SNN-vs-ANN",
        "seeds": SEEDS,
        "metrics": metrics,
        "activity": {
            "snn_final_spike_rate": bootstrap_interval(snn_spike_rates, 10000),
            "ann_final_active_fraction_abs_gt_1e-3": bootstrap_interval(ann_activity, 10000),
            "warning": "Spike rate and ANN thresholded active fraction are operational, not energetic equivalents.",
        },
        "interpretation": [
            "The orthographic-to-semantic layerwise shift is not unique to the SNN.",
            "The ANN has higher global IID and generalization performance.",
            "The SNN is stronger on hard orthographic distractors and more order-dependent.",
            "The SNN state is sparse while the ANN state is effectively dense.",
        ],
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_rows = []
    for index, seed in enumerate(SEEDS):
        snn_lexical = gen_snn[index]["evaluation"]["lexical_context_transfer"][
            "population"
        ]["sta"]
        snn_control = gen_snn_control[index]["evaluation"]["lexical_context_transfer"][
            "population"
        ]["sta"]
        raw_rows.append(
            {
                "model": "snn",
                "seed": seed,
                "iid_population_sta": snn_runs[index]["evaluation"]["subsets"]["all"][2][
                    "rate_only"
                ]["sta"],
                "hard_orthographic_population_sta": snn_runs[index]["evaluation"]["subsets"][
                    "hard_orthographic"
                ][2]["rate_only"]["sta"],
                "factorial_abstraction_index_l3": snn_factorial[index]["effects"]["rate"][2][
                    "abstraction_index_relative"
                ],
                "ordered_input_dependency_drop": snn_order[index]["correct"]["all"][2][
                    "rate_only"
                ]["sta"]
                - snn_order[index]["permuted_order"]["all"][2]["rate_only"]["sta"],
                "relation_disjoint_population_sta": gen_snn[index]["evaluation"][
                    "relation_disjoint"
                ]["population"]["sta"],
                "contextual_lexical_transfer_population_sta": snn_lexical,
                "no_context_lexical_transfer_population_sta": snn_control,
                "paired_context_gain": snn_lexical - snn_control,
                "final_spike_rate": snn_spike_rates[index],
            }
        )

        ann_lexical = gen_ann[index]["evaluation"]["lexical_context_transfer"][
            "population"
        ]["sta"]
        ann_control = gen_ann_control[index]["evaluation"]["lexical_context_transfer"][
            "population"
        ]["sta"]
        ann_iid = ann_runs[index]["evaluation"]["subsets"]["all"][2]["integrated"]["sta"]
        raw_rows.append(
            {
                "model": "ann",
                "seed": seed,
                "iid_population_sta": ann_iid,
                "hard_orthographic_population_sta": ann_runs[index]["evaluation"]["subsets"][
                    "hard_orthographic"
                ][2]["integrated"]["sta"],
                "factorial_abstraction_index_l3": ann_runs[index]["factorial"]["effects"][
                    "rate"
                ][2]["abstraction_index_relative"],
                "ordered_input_dependency_drop": ann_iid
                - ann_runs[index]["evaluation"]["permuted_integrated_sta"],
                "relation_disjoint_population_sta": gen_ann[index]["evaluation"][
                    "relation_disjoint"
                ]["population"]["sta"],
                "contextual_lexical_transfer_population_sta": ann_lexical,
                "no_context_lexical_transfer_population_sta": ann_control,
                "paired_context_gain": ann_lexical - ann_control,
                "final_active_fraction_abs_gt_1e-3": ann_activity[index],
            }
        )
    raw_path = Path(raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in raw_rows
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Consolida comparacion SNN vs ANN")
    parser.add_argument("--output", default="reports/benchmark_summary.json")
    parser.add_argument("--raw-output", default="reports/raw_metrics.jsonl")
    args = parser.parse_args()
    run(args.output, args.raw_output)


if __name__ == "__main__":
    main()
