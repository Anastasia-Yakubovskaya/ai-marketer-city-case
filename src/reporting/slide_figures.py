from typing import Optional
from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs" / "slide_figures"

BLUE = "#2563eb"
GRAY = "#8a8f98"
ORANGE = "#f97316"
DARK = "#111827"
LIGHT_GRAY = "#e5e7eb"

plt.rcParams.update(
    {
        "figure.figsize": (8.5, 4.8),
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
    }
)


def load_table(name: str):
    path = TABLES / name
    if not path.exists():
        print(f"[skip] missing table: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[ok] loaded {name}: {len(df)} rows")
        return df
    except Exception as exc:
        print(f"[error] failed to load {name}: {exc}")
        return None


def save_fig(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png_path = OUT / f"{name}.png"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")


def human_tornado_label(param: str) -> str:
    mapping = {
        "delta": "Communication uplift delta",
        "roi0": "Baseline ROI0",
        "capacity_monetization": "Capacity monetization",
        "labor_share_removed": "Labor share removed",
        "post_pilot_opex": "Post-pilot OPEX",
        "pilot_cost": "Pilot cost",
    }
    return mapping.get(param, param)


def short_tier_label(tier: str) -> str:
    mapping = {
        "yellow_one_click": "Yellow\none-click",
        "green_limited": "Green\nlimited",
        "green_standard": "Green\nstandard",
    }
    return mapping.get(tier, tier)


def make_s1_cycle_time(process) -> None:
    if process is None:
        return

    df = process[process["metric"] == "time_to_launch"].copy()
    if df.empty:
        print("[skip] s1: no time_to_launch rows")
        return

    order = ["as_is", "to_be"]
    labels = ["As-is launch", "To-be AI production"]
    colors = [GRAY, BLUE]

    df = df.set_index("scenario").loc[order].reset_index()

    p50 = df["p50"].to_numpy(dtype=float)
    p10 = df["p10"].to_numpy(dtype=float)
    p90 = df["p90"].to_numpy(dtype=float)

    err_low = p50 - p10
    err_high = p90 - p50

    fig, ax = plt.subplots()
    x = np.arange(len(labels))

    ax.bar(x, p50, color=colors, alpha=0.9, width=0.55)
    ax.errorbar(
        x,
        p50,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor=DARK,
        elinewidth=1.4,
        capsize=6,
        alpha=0.8,
    )

    for xi, v in zip(x, p50):
        ax.text(xi, v + 0.25, f"{v:.1f} d", ha="center", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Working days")
    ax.set_ylim(0, max(p90) * 1.25)
    ax.set_title("AI compresses campaign production time")
    ax.text(
        0.02,
        0.95,
        "Error bars: p10-p90 range",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=GRAY,
    )

    save_fig(fig, "s1_cycle_time")


def make_s2_time_to_result(process) -> None:
    if process is None:
        return

    df = process[process["metric"] == "time_to_evidence"].copy()
    if df.empty:
        print("[skip] s2: no time_to_evidence rows")
        return

    windows = sorted(df["observation_window_days"].unique())
    scenarios = ["as_is", "to_be"]
    colors = {"as_is": GRAY, "to_be": BLUE}
    labels = {"as_is": "As-is", "to_be": "To-be AI"}

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(windows))

    for i, scenario in enumerate(scenarios):
        sub = df[df["scenario"] == scenario].set_index("observation_window_days").loc[windows]
        pos = x + (i - 0.5) * width
        p50 = sub["p50"].to_numpy(dtype=float)
        err_low = p50 - sub["p10"].to_numpy(dtype=float)
        err_high = sub["p90"].to_numpy(dtype=float) - p50

        ax.bar(pos, p50, width=width, color=colors[scenario], alpha=0.9, label=labels[scenario])
        ax.errorbar(
            pos,
            p50,
            yerr=[err_low, err_high],
            fmt="none",
            ecolor=DARK,
            elinewidth=1.2,
            capsize=4,
            alpha=0.75,
        )

        for xi, v in zip(pos, p50):
            ax.text(xi, v + 0.7, f"{v:.1f}", ha="center", fontsize=9)


    as_is = df[df["scenario"] == "as_is"].set_index("observation_window_days").loc[windows]
    to_be = df[df["scenario"] == "to_be"].set_index("observation_window_days").loc[windows]

    for xi, w in zip(x, windows):
        a = float(as_is.loc[w, "p50"])
        b = float(to_be.loc[w, "p50"])
        if a > 0:
            reduction = 1 - b / a
            ax.text(
                xi + width / 2,
                max(a, b) + 2.2,
                f"-{reduction:.0%}",
                ha="center",
                fontsize=10,
                color=BLUE,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"W={int(w)} days" for w in windows])
    ax.set_ylabel("Days from hypothesis to evidence")
    ax.set_ylim(0, max(df["p90"]) * 1.22)
    ax.set_title("Observation window limits total speed gain")
    ax.legend(loc="upper left")

    save_fig(fig, "s2_time_to_result")


def make_s3_judge_evidence(judge_ev) -> None:
    if judge_ev is None:
        return

    df = judge_ev.copy()
    df["n"] = df["n_clean_zero_misses"].astype(float)
    df["miss_pct"] = df["upper_miss_rate"].astype(float) * 100

    fig, ax = plt.subplots()

    ax.plot(df["n"], df["miss_pct"], marker="o", color=BLUE, linewidth=2.2, markersize=6)
    ax.axhline(1.0, color=ORANGE, linestyle="--", linewidth=1.4, label="1% miss-rate target")
    ax.axhline(2.0, color=GRAY, linestyle=":", linewidth=1.4, label="2% miss-rate target")


    highlights = {50: "50", 149: "149", 299: "299"}
    for n_key, label in highlights.items():
        row = df[np.isclose(df["n"], n_key)]
        if not row.empty:
            x = float(row.iloc[0]["n"])
            y = float(row.iloc[0]["miss_pct"])
            ax.scatter([x], [y], s=80, color=DARK, zorder=5)
            ax.annotate(
                f"n={label}\n<= {y:.1f}%",
                xy=(x, y),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=9,
                color=DARK,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Clean creatives with zero critical misses")
    ax.set_ylabel("Upper bound on miss rate, %")
    ax.set_title("Autonomy is earned by evidence, not by trust")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(df["miss_pct"]) * 1.15)

    save_fig(fig, "s3_judge_evidence")


def make_s4_contact_policy(contact) -> None:
    if contact is None:
        return

    local = contact[contact["strategy"] == "local_checks"].copy()
    central = contact[contact["strategy"] == "central_arbiter"].copy()

    if local.empty or central.empty:
        print("[skip] s4: missing local_checks or central_arbiter rows")
        return

    local = local.sort_values("weekly_pressure")
    central = central.sort_values("weekly_pressure")

    fig, ax = plt.subplots()

    ax.plot(
        local["weekly_pressure"],
        local["any_policy_violation_rate"] * 100,
        marker="o",
        color=ORANGE,
        linewidth=2.3,
        label="Local checks: policy violations",
    )

    ax.plot(
        central["weekly_pressure"],
        central["value_loss"] * 100,
        marker="s",
        color=BLUE,
        linewidth=2.3,
        label="Central arbiter: value loss",
    )

    ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.0, alpha=0.7)


    for df, col, offset in [
        (local, "any_policy_violation_rate", (8, 8)),
        (central, "value_loss", (8, -14)),
    ]:
        row = df[np.isclose(df["weekly_pressure"], 3.0)]
        if not row.empty:
            x = float(row.iloc[0]["weekly_pressure"])
            y = float(row.iloc[0][col]) * 100
            ax.scatter([x], [y], s=70, color=DARK, zorder=5)
            ax.annotate(
                f"{y:.1f}%",
                xy=(x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=DARK,
            )

    ax.set_xlabel("Average candidate marketing messages per client per week")
    ax.set_ylabel("Percent of clients")
    ax.set_title("Central arbiter trades small value loss for zero violations")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)

    save_fig(fig, "s4_contact_policy")


def make_s5_payback(traj, payback) -> None:
    if traj is None:
        return

    df = traj.copy()
    if "cumulative_net_mln" not in df.columns:
        print("[skip] s5: cumulative_net_mln missing")
        return

    fig, ax = plt.subplots()

    ax.plot(df["month"], df["cumulative_net_mln"], color=BLUE, linewidth=2.5, label="Median cumulative net")
    ax.axhline(0, color=DARK, linestyle="--", linewidth=1.2, alpha=0.8)


    positive = df[df["cumulative_net_mln"] >= 0]
    if not positive.empty:
        cross = positive.iloc[0]
        ax.scatter(
            [cross["month"]],
            [cross["cumulative_net_mln"]],
            s=90,
            color=ORANGE,
            zorder=5,
            label=f"Median payback: month {int(cross['month'])}",
        )
        ax.annotate(
            f"Month {int(cross['month'])}",
            xy=(cross["month"], cross["cumulative_net_mln"]),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=ORANGE,
        )


    if payback is not None and not payback.empty:
        base = payback[payback["scenario"] == "base"]
        cons = payback[payback["scenario"] == "conservative"]

        text_lines = []
        if not base.empty:
            b = base.iloc[0]
            text_lines.append(
                f"Base payback: p10 {b['payback_p10']:.0f}, "
                f"p50 {b['payback_p50']:.0f}, p90 {b['payback_p90']:.0f} months"
            )
        if not cons.empty:
            c = cons.iloc[0]
            text_lines.append(
                f"Conservative payback: p10 {c['payback_p10']:.0f}, "
                f"p50 {c['payback_p50']:.0f}, p90 {c['payback_p90']:.0f} months"
            )

        if text_lines:
            ax.text(
                0.02,
                0.03,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="bottom",
                fontsize=9,
                color=DARK,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=LIGHT_GRAY, alpha=0.95),
            )

    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative net, mln RUB")
    ax.set_title("Pilot cost is recovered as coverage ramps up")
    ax.legend(loc="upper left")

    save_fig(fig, "s5_payback")


def make_s6_tornado(tornado) -> None:
    if tornado is None:
        return

    df = tornado.copy()
    if "range" not in df.columns:
        print("[skip] s6: range column missing")
        return

    df["range"] = df["range"].abs()
    df = df.sort_values("range", ascending=True)
    df["label"] = df["parameter"].map(human_tornado_label)

    fig, ax = plt.subplots()

    ax.barh(df["label"], df["range"], color=BLUE, alpha=0.88, height=0.62)

    for y, v in zip(df["label"], df["range"]):
        ax.text(v + max(df["range"]) * 0.015, y, f"{v:.1f}", va="center", fontsize=9, color=DARK)

    ax.set_xlabel("NPV swing over break-even horizon, mln RUB")
    ax.set_title("Delta and ROI0 dominate economic uncertainty")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    save_fig(fig, "s6_tornado")


def make_s7_promotion_rule(aut_thr) -> None:
    if aut_thr is None:
        return

    df = aut_thr.copy()

    needed_cols = {"tier", "required_creatives_n", "max_contacts_without_canary"}
    if not needed_cols.issubset(df.columns):
        print("[skip] s7: autonomy_thresholds missing columns")
        return

    df["n"] = df["required_creatives_n"].astype(float)
    df["reach_mln"] = df["max_contacts_without_canary"].astype(float) / 1_000_000.0
    df = df.sort_values("n")

    fig, ax = plt.subplots()

    ax.plot(df["n"], df["reach_mln"], marker="o", color=BLUE, linewidth=2.3, markersize=7)

    for _, row in df.iterrows():
        ax.annotate(
            short_tier_label(str(row["tier"])),
            xy=(row["n"], row["reach_mln"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=DARK,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Clean creatives required for tier")
    ax.set_ylabel("Max contacts without canary, mln")
    ax.set_title("Allowed reach grows with accumulated evidence")
    ax.set_ylim(bottom=0)

    save_fig(fig, "s7_promotion_rule")


def make_s8_state_diagram() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.55, 4.15, "Shadow\nAI prepares,\nnothing sends", GRAY),
        (2.45, 4.15, "One-click\nhuman approves\nready card", BLUE),
        (4.35, 4.15, "Canary\nsmall share,\nguardrails", BLUE),
        (6.25, 4.15, "Limited Green\nproven tier,\ncontrolled reach", BLUE),
        (8.15, 4.15, "Extended Green\nmore evidence,\nmore reach", BLUE),
    ]

    box_w = 1.45
    box_h = 1.05

    for x, y, label, color in boxes:
        rect = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.08",
            linewidth=1.4,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2,
            y + box_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=DARK,
            fontweight="bold",
        )


    for x_start in [2.0, 3.9, 5.8, 7.7]:
        arrow = FancyArrowPatch(
            (x_start, 4.67),
            (x_start + 0.45, 4.67),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color=DARK,
        )
        ax.add_patch(arrow)


    ax.text(
        5.0,
        5.55,
        "Promotion: Clopper-Pearson evidence + OC curve <= 5% false promotion",
        ha="center",
        va="center",
        fontsize=10,
        color=BLUE,
        fontweight="bold",
    )


    demotion = FancyArrowPatch(
        (8.8, 3.75),
        (1.3, 3.75),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2.0,
        color=ORANGE,
        connectionstyle="arc3,rad=0.18",
    )
    ax.add_patch(demotion)

    ax.text(
        5.0,
        2.55,
        "Demotion: SPRT monitoring + error budget breach\nreturn to human control until root cause is fixed",
        ha="center",
        va="center",
        fontsize=10,
        color=ORANGE,
        fontweight="bold",
    )

    ax.text(
        5.0,
        1.35,
        "In pilot: Green expands eligible reach and automation depth.\n"
        "Prod-send is always confirmed by a human.",
        ha="center",
        va="center",
        fontsize=10,
        color=DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f9fafb", edgecolor=LIGHT_GRAY),
    )

    ax.set_title("Earned autonomy ladder", pad=18)

    save_fig(fig, "s8_state_diagram")


def make_s9_waterfall(econ) -> None:
    if econ is None:
        return

    df = econ.set_index("metric")

    needed = {
        "communications_mln_month",
        "labor_capacity_mln_month",
        "errors_avoided_mln_month",
        "total_mln_month",
    }
    if not needed.issubset(df.index):
        print("[skip] s9: economics_summary missing expected metrics")
        return

    comm = float(df.loc["communications_mln_month", "p50"])
    labor = float(df.loc["labor_capacity_mln_month", "p50"])
    errors = float(df.loc["errors_avoided_mln_month", "p50"])
    total = float(df.loc["total_mln_month", "p50"])

    labels = ["Communications", "Labor capacity", "Errors avoided", "Total median"]
    values = [comm, labor, errors, total]
    bottoms = [0, comm, comm + labor, 0]
    colors = [BLUE, GRAY, ORANGE, DARK]

    fig, ax = plt.subplots()

    bars = ax.bar(labels, values, bottom=bottoms, color=colors, alpha=0.9, width=0.58)

    for bar, value, bottom in zip(bars, values, bottoms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bottom + value + 0.12,
            f"{value:.1f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=DARK,
        )

    ax.set_ylabel("mln RUB / month at full coverage")
    ax.set_title("Main value is communication uplift, not FTE savings")
    ax.set_ylim(0, max(total, comm + labor + errors) * 1.22)


    if total > 0:
        share = comm / total
        ax.text(
            0.02,
            0.95,
            f"Communications share: {share:.0%} of median benefit",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            color=BLUE,
            fontweight="bold",
        )

    save_fig(fig, "s9_waterfall")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    print("Generating slide figures from existing output tables...")
    print(f"Reading tables from: {TABLES}")
    print(f"Writing figures to: {OUT}")
    print()

    process = load_table("process_summary.csv")
    judge_ev = load_table("judge_upper_bounds.csv")
    contact = load_table("contact_policy_violations.csv")
    econ = load_table("economics_summary.csv")
    payback = load_table("payback_scenarios.csv")
    tornado = load_table("tornado.csv")
    traj = load_table("economics_trajectory.csv")
    aut_thr = load_table("autonomy_thresholds.csv")

    make_s1_cycle_time(process)
    make_s2_time_to_result(process)
    make_s3_judge_evidence(judge_ev)
    make_s4_contact_policy(contact)
    make_s5_payback(traj, payback)
    make_s6_tornado(tornado)
    make_s7_promotion_rule(aut_thr)
    make_s8_state_diagram()
    make_s9_waterfall(econ)

    print()
    print("Done. Check outputs/slide_figures/.")


if __name__ == "__main__":
    main()
