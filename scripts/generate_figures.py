"""Generate research figures from archived experimental results."""

import hashlib
import json
import platform
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT = Path("docs/figures")

SOURCES = {
    "rq1": Path("results/rq1/metrics_v0_1.json"),
    "rq1_stats": Path("results/rq1/statistics_v0_1.json"),
    "rq2": Path("results/heldout/rq2_comparison_v0_1.json"),
    "rq3": Path("results/rq3/metrics_v0_1.json"),
}

NAVY = "#142B45"
BLUE = "#2563A6"
TEAL = "#16857A"
GRAY = "#8493A5"
ORANGE = "#D97706"

METHODS = [
    "recency",
    "exhaustive",
    "semantic",
    "oracle",
]

METHOD_NAMES = [
    "Recency",
    "Exhaustive",
    "Semantic",
    "Oracle",
]

COLORS = [GRAY, ORANGE, TEAL, NAVY]

GENERATED = []


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    plt.switch_backend("Agg")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.titlecolor": NAVY,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.alpha": 0.20,
            "legend.frameon": False,
            "svg.hashsalt": "agent-memory-auditor-v0.1",
            "savefig.dpi": 300,
        }
    )


def save(fig, name, note):
    fig.text(
        0.5,
        0.025,
        note,
        ha="center",
        fontsize=9,
        color="#526174",
        wrap=True,
    )

    for extension in ("png", "svg"):
        path = OUTPUT / f"{name}.{extension}"

        metadata = (
            {"Date": None}
            if extension == "svg"
            else {"Software": "Agent Memory Auditor"}
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.25,
            metadata=metadata,
        )

        if extension == "svg":
            content = path.read_text(encoding="utf-8")
            normalized = "\n".join(line.rstrip(" \t") for line in content.split("\n"))
            path.write_text(normalized, encoding="utf-8")

        GENERATED.append(path)

    plt.close(fig)
    print("Generated:", name)


def grouped_bars(title, names, series, filename, note):
    fig, ax = plt.subplots(figsize=(11, 6))

    fig.subplots_adjust(bottom=0.17, top=0.83)

    x = np.arange(len(names))
    width = 0.72 / len(series)

    for i, (label, values, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            color=color,
            label=label,
        )

        ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=4,
            fontsize=10,
        )

    ax.set_title(title, pad=18)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 115)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Score (%)")
    ax.grid(axis="y")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(series),
    )

    save(fig, filename, note)


def architecture():
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.set_title(
        "Agent Memory Auditor — NLP Architecture",
        pad=15,
    )

    def node(x, y, w, h, title, detail, color):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=color,
            edgecolor="none",
        )

        ax.add_patch(patch)

        ax.text(
            x + w / 2,
            y + h * 0.65,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )

        ax.text(
            x + w / 2,
            y + h * 0.30,
            detail,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "color": GRAY,
                "lw": 2,
            },
        )

    node(
        0.2,
        3.8,
        2.7,
        1.35,
        "Memory pair",
        "Texts and timestamps",
        NAVY,
    )

    node(
        3.5,
        3.8,
        2.7,
        1.35,
        "Sentence-BERT",
        "Cosine similarity",
        BLUE,
    )

    node(
        6.8,
        3.8,
        2.7,
        1.35,
        "DeBERTa NLI",
        "Forward and reverse",
        BLUE,
    )

    node(
        10.1,
        3.8,
        2.7,
        1.35,
        "Temporal rules",
        "Change evidence",
        TEAL,
    )

    arrow(2.9, 4.48, 3.5, 4.48)
    arrow(6.2, 4.48, 6.8, 4.48)
    arrow(9.5, 4.48, 10.1, 4.48)

    node(
        3.6,
        1.85,
        5.8,
        1.1,
        "Frozen decision policy",
        "Similarity gate + NLI consensus + temporal evidence",
        NAVY,
    )

    arrow(11.45, 3.8, 9.4, 2.4)

    labels = [
        "Consistent",
        "New",
        "Updated",
        "Contradictory",
    ]

    for i, label in enumerate(labels):
        x = 0.2 + i * 3.3

        node(
            x,
            0.2,
            2.7,
            0.85,
            label,
            "Memory label",
            TEAL,
        )

        arrow(6.5, 1.85, x + 1.35, 1.05)

    save(
        fig,
        "01_system_architecture",
        "Frozen pair classifier: C_consensus. "
        "Similarity is a decision gate after pair scoring.",
    )


def retrieval_performance(data):
    overall = data["overall"]

    series = [
        (
            "Macro F1",
            [100 * overall[m]["macro_f1"] for m in METHODS],
            BLUE,
        ),
        (
            "Conflict retrieval recall",
            [100 * overall[m]["conflict_retrieval_recall"] for m in METHODS],
            TEAL,
        ),
        (
            "Target conflict recall",
            [100 * overall[m]["target_conflict_recall"] for m in METHODS],
            GRAY,
        ),
    ]

    grouped_bars(
        "RQ1 — Retrieval Strategy Performance",
        METHOD_NAMES,
        series,
        "02_rq1_retrieval_performance",
        "60 DNLI source cases; 540 controlled histories per method. "
        "Oracle receives the known target.",
    )


def scalability(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    fig.subplots_adjust(bottom=0.17)

    sizes = [20, 50, 100]
    results = data["by_history_size"]

    for method, name, color in zip(METHODS, METHOD_NAMES, COLORS, strict=True):
        values = [results[str(size)][method]["mean_latency_ms"] for size in sizes]

        ax.plot(
            sizes,
            values,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=name,
            color=color,
        )

    ax.set_title("RQ1 — Processing Latency and History Size")
    ax.set_xlabel("Number of memories")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_xticks(sizes)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid()

    save(
        fig,
        "03_rq1_scalability",
        "CPU inference; model loading excluded. "
        "Benchmark completed across two sessions.",
    )


def temporal_policies(data):
    policies = data["policies"]

    keys = [
        "C_consensus",
        "most_recent_wins",
        "never_update",
    ]

    names = [
        "NLP Pipeline",
        "Most Recent Wins",
        "Never Update",
    ]

    series = [
        (
            "Accuracy",
            [100 * policies[k]["accuracy"] for k in keys],
            BLUE,
        ),
        (
            "Updated recall",
            [
                100 * policies[k]["classification_report"]["updated"]["recall"]
                for k in keys
            ],
            TEAL,
        ),
        (
            "Contradictory recall",
            [
                100 * policies[k]["classification_report"]["contradictory"]["recall"]
                for k in keys
            ],
            GRAY,
        ),
    ]

    grouped_bars(
        "RQ2 — Temporal Resolution Policies",
        names,
        series,
        "04_rq2_temporal_policies",
        "200 held-out pairs; shared NLI conflicts. "
        "Neither McNemar comparison reached p < 0.05.",
    )


def class_recall(data):
    labels = data["label_order"]

    series = [
        (
            "NLP Pipeline",
            [
                100 * data["pipeline"]["classification_report"][label]["recall"]
                for label in labels
            ],
            BLUE,
        ),
        (
            "Qwen3 8B",
            [
                100 * data["llm"]["classification_report"][label]["recall"]
                for label in labels
            ],
            TEAL,
        ),
    ]

    grouped_bars(
        "RQ3 — Per-class Recall",
        [label.title() for label in labels],
        series,
        "05_rq3_class_recall",
        "Same 200 held-out pairs; 50 examples per class.",
    )


def confusion_matrices(data):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
    )

    fig.subplots_adjust(
        wspace=0.38,
        bottom=0.23,
        top=0.82,
    )

    fig.suptitle(
        "RQ3 — Classification Confusion Matrices",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )

    labels = [label.title() for label in data["label_order"]]

    for ax, key, name in zip(
        axes,
        ("pipeline", "llm"),
        ("NLP Pipeline", "Qwen3 8B"),
        strict=True,
    ):
        matrix = np.array(data[key]["confusion_matrix"])

        assert matrix.shape == (4, 5)
        assert matrix[:, -1].sum() == 0

        matrix = matrix[:, :4]

        ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=50,
        )

        for i in range(4):
            for j in range(4):
                value = int(matrix[i, j])

                ax.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color="white" if value >= 30 else NAVY,
                )

        ax.set_title(name, fontsize=16)

        ax.set_xticks(
            range(4),
            labels,
            rotation=30,
            ha="right",
        )

        ax.set_yticks(range(4), labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    save(
        fig,
        "06_rq3_confusion_matrices",
        "No invalid responses. The all-zero invalid prediction column is omitted.",
    )


def accuracy_latency(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    fig.subplots_adjust(bottom=0.17)

    models = [
        ("pipeline", "NLP Pipeline (CPU)", BLUE),
        ("llm", "Qwen3 8B (GPU)", TEAL),
    ]

    for key, name, color in models:
        latency = data[key]["mean_latency_ms"]
        accuracy = 100 * data[key]["accuracy"]

        ax.scatter(
            latency,
            accuracy,
            s=220,
            color=color,
            zorder=3,
        )

        offset = (10, -40) if key == "pipeline" else (-10, 15)

        ax.annotate(
            f"{name}\n{accuracy:.2f}% | {latency:,.2f} ms",
            (latency, accuracy),
            xytext=offset,
            textcoords="offset points",
            ha="left" if key == "pipeline" else "right",
            fontsize=12,
            color=NAVY,
        )

    ax.set_xscale("log")
    ax.set_xlim(20, 2500)
    ax.set_ylim(50, 100)
    ax.set_title("RQ3 — Accuracy and Latency Trade-off")
    ax.set_xlabel("Mean latency (ms, logarithmic scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.20)

    save(
        fig,
        "07_accuracy_latency_tradeoff",
        "Same Mac mini M4, different acceleration. "
        "Latency is not an energy or monetary cost measurement.",
    )


def confidence_intervals(rq1_stats, rq3):
    comparisons = rq1_stats["comparisons"]

    recent = comparisons["recency"]["macro_f1"]
    exhaustive = comparisons["exhaustive"]["macro_f1"]

    difference = rq3["llm_minus_pipeline"]

    llm = {
        "difference": difference["macro_f1_difference"],
        **difference["paired_ci95"],
    }

    rows = [
        ("RQ1: Semantic vs. Recency", recent),
        ("RQ1: Semantic vs. Exhaustive", exhaustive),
        ("RQ3: Qwen3 vs. NLP Pipeline", llm),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))

    fig.subplots_adjust(left=0.32, bottom=0.19)

    for i, (name, result) in enumerate(rows):
        point = result["difference"]
        lower = result["lower"]
        upper = result["upper"]

        ax.errorbar(
            point,
            i,
            xerr=[[point - lower], [upper - point]],
            fmt="o",
            color=TEAL if i < 2 else BLUE,
            capsize=5,
            markersize=8,
            linewidth=2.5,
        )

        ax.text(
            upper + 0.012,
            i,
            f"{point:+.4f}",
            va="center",
            fontsize=11,
            color=NAVY,
        )

    ax.axvline(
        0,
        linestyle="--",
        color=GRAY,
    )

    ax.set_yticks(
        range(len(rows)),
        [name for name, _ in rows],
    )

    ax.set_xlim(-0.05, 0.40)
    ax.set_ylim(-0.6, 2.6)
    ax.invert_yaxis()
    ax.set_xlabel("Difference in macro F1")
    ax.set_title("Paired Bootstrap Confidence Intervals")
    ax.grid(axis="x")

    save(
        fig,
        "08_statistical_confidence_intervals",
        "95% individual intervals; 5,000 resamples. "
        "RQ1 uses clustered resampling by source case.",
    )


def main():
    configure()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    hashes = {str(path): sha256(path) for path in SOURCES.values()}

    data = {key: json.loads(path.read_text()) for key, path in SOURCES.items()}

    architecture()
    retrieval_performance(data["rq1"])
    scalability(data["rq1"])
    temporal_policies(data["rq2"])
    class_recall(data["rq3"])
    confusion_matrices(data["rq3"])
    accuracy_latency(data["rq3"])

    confidence_intervals(
        data["rq1_stats"],
        data["rq3"],
    )

    assert len(GENERATED) == 16

    assert hashes == {str(path): sha256(path) for path in SOURCES.values()}

    manifest = {
        "version": "research-figures-v0.1",
        "python": platform.python_version(),
        "matplotlib": mpl.__version__,
        "numpy": np.__version__,
        "png_dpi": 300,
        "figures": 8,
        "generated_files": 16,
        "source_sha256": hashes,
        "architecture_source_sha256": {
            str(path): sha256(path)
            for path in (
                Path("scripts/compare_policies.py"),
                Path("scripts/predict_heldout.py"),
            )
        },
        "generator_sha256": sha256(Path(__file__)),
        "figure_sha256": {str(path): sha256(path) for path in GENERATED},
    }

    manifest_path = OUTPUT / "manifest_v0_1.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== SCIENTIFIC FIGURES ===")
    print("Figures:", 8)
    print("PNG files:", 8)
    print("SVG files:", 8)
    print("Source results preserved: PASS")
    print("New model inference: NONE")
    print("Manifest:", manifest_path)
    print("SCIENTIFIC FIGURES: PASS")


if __name__ == "__main__":
    main()
