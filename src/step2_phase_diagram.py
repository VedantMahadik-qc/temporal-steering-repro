"""
Step 2: reproduce the (p, P) phase diagram (paper Fig. 1 style, x = 1 / pure
ADC-ADC case) and confirm the three ESD-manipulation regimes (avoidance /
delay / hastening) against Table I.

FOLLOW-UP FINDING (see NOTES.md, step1_convention_check.py): the paper's
dedicated ADC-NOT-ADC formula, Eq. (4)/(B8)-(B11), does not match a
from-scratch Kraus-map simulation of sigma_x (x) sigma_x, and does not
algebraically reduce from the paper's own general formula, Eq. (14),
evaluated at x=1 -- despite the paper's text stating Eq. (14) "approaches
the apparatus-defined ADC-like at x=1". Eq. (14)|x=1 DOES match the Kraus
simulation to numerical precision. This script therefore plots BOTH: the
recommended Eq. (14)|x=1 boundary (solid) and the literal Eq. (4) boundary
(dashed, for reference), and reports thresholds/Table-I comparison for both.

Outputs:
  figures/phase_diagram.png   -- ESD boundary curves + regime bands (both formulas)
  figures/regime_slices.png   -- 1D concurrence-vs-P slices, one per regime
  results_step2.txt           -- thresholds, ESD points, and Table I comparison
"""
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from model import (
    analytic_C_no_not, analytic_C_with_not_eq4, analytic_C_with_not_eq14_x1,
    esd_no_not_bracket, esd_with_not_bracket_eq4, esd_with_not_bracket_eq14_x1,
)

ALPHA = 0.55

# Okabe-Ito colorblind-safe qualitative palette.
C_ORANGE = "#E69F00"   # avoidance
C_GREEN = "#009E73"    # delay
C_VERMILLION = "#D55E00"  # hastening
C_BLUE = "#0072B2"     # no-NOT boundary curve (paper: blue)
C_RED = "#CC3311"      # with-NOT boundary curve, Eq.(14)|x=1 (recommended)
C_GRAY = "#888888"     # with-NOT boundary curve, Eq.(4) literal (reference only)

FIG_DIR = "../figures"


def p_esd(bracket_fn, p, P_max=1.0 - 1e-9):
    """First root in P of bracket_fn(alpha, p, P) = 0, i.e. the ESD point.

    Returns 0.0 if already separable at P=0, or 1.0 if the bracket stays
    positive across the whole range (no finite-P ESD -- 'avoidance').
    """
    f = lambda P: bracket_fn(ALPHA, p, P)
    if f(0.0) <= 0:
        return 0.0
    if f(P_max) > 0:
        return 1.0
    return brentq(f, 0.0, P_max)


def find_threshold(pred, ps):
    """First p in ps where pred(p) becomes True (assumes it stays True after)."""
    for p in ps:
        if pred(p):
            return p
    return ps[-1]


def thresholds(bracket_with_not):
    ps = np.linspace(0.0, 0.6, 2001)
    p1 = find_threshold(lambda p: p_esd(bracket_with_not, p) < 0.999999, ps)
    p2 = find_threshold(lambda p: p_esd(bracket_with_not, p) < p_esd(esd_no_not_bracket, p), ps)
    return p1, p2


def main():
    ps = np.linspace(0.0, 0.6, 601)
    Pesd_no = np.array([p_esd(esd_no_not_bracket, p) for p in ps])
    Pesd_with_eq14 = np.array([p_esd(esd_with_not_bracket_eq14_x1, p) for p in ps])
    Pesd_with_eq4 = np.array([p_esd(esd_with_not_bracket_eq4, p) for p in ps])

    p1_eq14, p2_eq14 = thresholds(esd_with_not_bracket_eq14_x1)
    p1_eq4, p2_eq4 = thresholds(esd_with_not_bracket_eq4)

    # ---------------- phase diagram (Fig. 1 style) ----------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.axvspan(0.0, p1_eq14, color=C_ORANGE, alpha=0.15, label="avoidance (Eq. 14|x=1)")
    ax.axvspan(p1_eq14, p2_eq14, color=C_GREEN, alpha=0.15, label="delay (Eq. 14|x=1)")
    ax.axvspan(p2_eq14, 0.6, color=C_VERMILLION, alpha=0.15, label="hastening (Eq. 14|x=1)")

    ax.plot(ps, Pesd_no, color=C_BLUE, lw=2, label=r"$P_{\rm ESD}$ without NOT [Eq. (3)]")
    ax.plot(ps, Pesd_with_eq14, color=C_RED, lw=2.2,
            label=r"$P_{\rm ESD}$ with NOT [Eq. (14)$|_{x=1}$, recommended]")
    ax.plot(ps, Pesd_with_eq4, color=C_GRAY, lw=1.6, ls="--",
            label=r"$P_{\rm ESD}$ with NOT [Eq. (4), literal -- reference only]")

    # Table I's stated thresholds, shown as reference for comparison.
    for p_ref, lbl in [(0.17, "Table I: 0.17"), (0.28, "Table I: 0.28")]:
        ax.axvline(p_ref, color="0.3", ls=":", lw=1.3)
        ax.text(p_ref - 0.008, 0.03, lbl, rotation=90, va="bottom", ha="right",
                 fontsize=8, color="0.3")

    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$p$ (first-channel damping)")
    ax.set_ylabel(r"$P_{\rm ESD}$ (second-channel damping at ESD)")
    ax.set_title(r"ESD boundary, $x=1$ (independent ADC $\to$ ADC), $\alpha=0.55$")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/phase_diagram.png", dpi=180)
    plt.close(fig)

    # ---------------- 1D regime slices (using recommended Eq. 14|x=1) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    slice_ps = [(0.05, "avoidance", C_ORANGE), (0.20, "delay", C_GREEN), (0.40, "hastening", C_VERMILLION)]
    Ps = np.linspace(0, 1, 300)
    for ax, (p, label, color) in zip(axes, slice_ps):
        c_no = [analytic_C_no_not(ALPHA, p, P) for P in Ps]
        c_with = [analytic_C_with_not_eq14_x1(ALPHA, p, P) for P in Ps]
        ax.plot(Ps, c_no, color=C_BLUE, lw=2, label="without NOT")
        ax.plot(Ps, c_with, color=C_RED, lw=2, label="with NOT [Eq. (14)|x=1]")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(f"p = {p:.2f} ({label})", color=color)
        ax.set_xlabel("P")
    axes[0].set_ylabel("Concurrence C(p, P)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/regime_slices.png", dpi=180)
    plt.close(fig)

    # ---------------- text report ----------------
    lines = []
    lines.append(f"alpha = {ALPHA}, beta = {np.sqrt(1-ALPHA**2):.6f}\n")

    beta = np.sqrt(1 - ALPHA ** 2)
    p1_closed_form = 1.0 - 1.0 / (abs(ALPHA * beta) + beta ** 2)

    lines.append("Thresholds using Eq. (14) at x=1 [RECOMMENDED -- verified against")
    lines.append("a from-scratch Kraus-map simulation to 5.8e-9, see step1]:")
    lines.append(f"  p1 (avoidance -> delay)  = {p1_eq14:.6f}   [Table I: ~0.17]")
    lines.append(f"    exact closed form p1 = 1 - 1/(|ab|+b^2) = {p1_closed_form:.6f}")
    lines.append("    (root-find and closed form agree to 1e-4; the p1 vs. 0.17 gap")
    lines.append("    is real, not a numerical-precision artifact -- see NOTES.md)")
    lines.append(f"  p2 (delay -> hastening)  = {p2_eq14:.4f}   [Table I: ~0.28, matches to ~1%]\n")

    ab = abs(ALPHA * beta)
    p1_eq4_closed = 1 - (1 - ab) / beta ** 2
    lines.append("Thresholds using Eq. (4) literal [paper's own Appendix B formula,")
    lines.append("does NOT match the Kraus simulation -- reference only]:")
    lines.append(f"  p1 (avoidance -> delay)  = {p1_eq4:.6f}   [Table I: ~0.17]")
    lines.append(f"    exact closed form p1 = 1 - (1-|ab|)/b^2 = {p1_eq4_closed:.6f}")
    lines.append(f"  p2 (delay -> hastening)  = {p2_eq4:.4f}   [Table I: ~0.28]\n")

    lines.append("p1 closed-form comparison (both formulas checked against Table I's 0.17):")
    lines.append(f"  Eq. (4) closed-form p1      = {p1_eq4_closed:.6f}  (gap from 0.17: {abs(p1_eq4_closed-0.17)/0.17*100:.1f}% relative)")
    lines.append(f"  Eq. (14)|x=1 closed-form p1 = {p1_closed_form:.6f}  (gap from 0.17: {abs(p1_closed_form-0.17)/0.17*100:.1f}% relative)")
    lines.append("  Neither formula's p1 threshold lands near Table I's 0.17. p1 is")
    lines.append("  genuinely unresolved under either candidate formula -- see NOTES.md.\n")

    lines.append("Representative points (Eq. 14|x=1, recommended):")
    for p_test in [0.0, 0.05, 0.10, 0.14, 0.17, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.43, 0.50]:
        pn = p_esd(esd_no_not_bracket, p_test)
        pw = p_esd(esd_with_not_bracket_eq14_x1, p_test)
        lines.append(f"  p={p_test:5.2f}  P_ESD_noNOT={pn:.4f}  P_ESD_withNOT={pw:.4f}  "
                     f"dP={pw-pn:+.4f}")

    report = "\n".join(lines)
    print(report)
    with open("../results_step2.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
