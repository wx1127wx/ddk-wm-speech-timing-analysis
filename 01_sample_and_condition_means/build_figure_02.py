"""Display-trimmed, recolored Phase-1 raincloud figure; inference uses full data."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
MEANS_FILE = OUT / "data_processed" / "speech_condition_means.csv"
CONTRAST_FILE = OUT / "tables" / "Table_02_planned_paired_contrasts.csv"
CONDITIONS = ["Single", "Low", "High"]
SPEECH = {
    "DDK Rate": ("rate", "syllables/s"),
    "DDK Regularity": ("reg", "ms"),
    "DDK Duration": ("dur", "ms"),
    "Pause Regularity": ("pause_reg", "ms"),
    "Pause Duration": ("pause_dur", "ms"),
}
# Single reference: gray. Ordered dual-task blocks: colorblind-safe blue/orange.
COLORS = {"Single": "#737373", "Low": "#0072B2", "High": "#D55E00"}
INK = "#26343B"
DISPLAY_OMISSIONS = {("DDK Regularity", "Single"): {"P128"}}

def significance_label(q):
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "n.s."

def add_bracket(ax, x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color=INK, linewidth=.9, clip_on=False)
    ax.text((x1+x2)/2, y+h*1.08, label, ha="center", va="bottom", fontsize=7.3, color=INK)

def half_violin(ax, vals, x, color, ygrid):
    d = stats.gaussian_kde(vals)(ygrid)
    d = d / d.max() * .29
    ax.fill_betweenx(ygrid, x, x+d, color=color, alpha=.48, linewidth=0, zorder=1)
    ax.plot(x+d, ygrid, color=color, alpha=.75, linewidth=.7, zorder=2)

def display_data(wide, metric):
    shown = wide.copy()
    for condition in CONDITIONS:
        omit = DISPLAY_OMISSIONS.get((metric, condition), set())
        shown.loc[shown.index.isin(omit), condition] = np.nan
    return shown

def draw_panel(ax, metric, column, unit, data, contrast, letter, rng):
    full = data.pivot(index="analysis_id", columns="condition", values=column).loc[:, CONDITIONS]
    shown = display_data(full, metric)
    raw = shown.to_numpy(float)
    ymin, ymax = float(np.nanmin(raw)), float(np.nanmax(raw))
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    ygrid = np.linspace(ymin, ymax, 240)
    for row in raw:
        ax.plot(np.arange(3), row, color="#7D878C", alpha=.035, linewidth=.55, zorder=0)
    for i, condition in enumerate(CONDITIONS):
        vals = shown[condition].dropna().to_numpy(float)
        half_violin(ax, vals, i+.02, COLORS[condition], ygrid)
        ax.scatter(rng.normal(i-.17, .045, len(vals)), vals, s=8, color=COLORS[condition], alpha=.30, linewidth=0, zorder=3)
        ax.boxplot(vals, positions=[i-.17], widths=.18, patch_artist=True, showfliers=False,
                   boxprops={"facecolor":"white", "edgecolor":COLORS[condition], "linewidth":.9, "alpha":.92},
                   whiskerprops={"color":COLORS[condition], "linewidth":.8}, capprops={"color":COLORS[condition], "linewidth":.8},
                   medianprops={"color":INK, "linewidth":1.0}, zorder=4)
    # Inference and estimates use all participant-condition means, including P128.
    mean_x = np.arange(3)+.30
    means = full.mean(axis=0).to_numpy(float)
    ci = full.sem(axis=0).to_numpy(float) * stats.t.ppf(.975, len(full)-1)
    ax.plot(mean_x, means, color=INK, linewidth=1.55, zorder=6)
    ax.errorbar(mean_x, means, yerr=ci, fmt="o", color=INK, markersize=4.5, capsize=2.7, linewidth=1.0, zorder=7)
    c = contrast[contrast.metric.eq(metric)].set_index("contrast")
    top = ymax + span*.14
    add_bracket(ax, 0, 2, top, span*.025, f"Dual–Single (composite) {significance_label(float(c.loc['Dual vs Single', 'q_BH_within_contrast']))}")
    add_bracket(ax, 1, 2, top-span*.09, span*.025, f"High–Low {significance_label(float(c.loc['High vs Low', 'q_BH_within_contrast']))}")
    ax.set_xlim(-.52, 2.72)
    ax.set_ylim(ymin-span*.06, ymax+span*.27)
    ax.set_xticks(np.arange(3), CONDITIONS)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel(f"{metric} ({unit})", fontsize=8.5)
    ax.grid(axis="y", color="#D9DEE2", linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-.11, 1.07, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", color=INK)
    ax.text(.02, 1.01, "dots: displayed participant means; box: IQR; line: mean ± 95% CI", transform=ax.transAxes, fontsize=6.7, va="bottom", color="#5F696E")

def main():
    data = pd.read_csv(MEANS_FILE, dtype={"analysis_id":str})
    contrast = pd.read_csv(CONTRAST_FILE)
    rng = np.random.default_rng(20260909)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6), constrained_layout=False)
    for ax, letter, (metric, (column, unit)) in zip(axes.ravel(), "abcde", SPEECH.items()):
        draw_panel(ax, metric, column, unit, data, contrast, letter, rng)
    axes.ravel()[-1].axis("off")
    handles = [patches.Patch(facecolor=COLORS[c], edgecolor="none", alpha=.72, label=c) for c in CONDITIONS]
    handles.append(plt.Line2D([0], [0], color=INK, marker="o", linewidth=1.4, markersize=4, label="Mean ± 95% CI"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(.5,.018))
    fig.suptitle("DDK task-condition means: raincloud-style repeated-measures display", fontsize=13, y=.985, color=INK)
    fig.text(.5, .002, "Analysis sample: n = 155. Panel b displays 154 points because the isolated Single DDK Regularity value for P128 is omitted only from its visual layers for legibility. All means, 95% CIs, and tests use the complete data.", ha="center", va="bottom", fontsize=7.0, color="#4D585D")
    fig.subplots_adjust(left=.055, right=.985, top=.93, bottom=.105, wspace=.30, hspace=.38)
    stem = FIGURES / "Figure_02"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))

if __name__ == "__main__":
    main()


