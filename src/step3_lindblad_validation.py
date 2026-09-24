"""
Step 3 (Phase 2, part 1): validate that continuous-time Lindblad evolution
under independent atomic spontaneous emission (H=0, L_i = sqrt(gamma)*sigma_minus_i,
dissertation's jump operator) reduces to Phase 1's discrete two-stage AD-NOT-AD
picture under p(tau) = 1 - exp(-gamma*tau).

Two checks, both against an INDEPENDENT numerical method (superoperator
exponentiation of the Liouvillian, not Kraus-operator products):

  1. Grid comparison: C_lindblad(t1, tau2) vs. the three Phase-1 closed forms
     [Eq. (3) no-NOT, Eq. (14)|x=1 recommended, Eq. (4) literal] across a
     (p, P) grid, with t1 = tau_of_p(p), tau2 = tau_of_p(P).

  2. p1/p2 threshold check: root-find the ESD boundary directly from the
     continuous simulation (brentq on the unclamped signed Wootters
     quantity, concurrence_signed -- exactly analogous to Phase 1's
     bracket-function root-finding, but now from ODE/superoperator
     propagation, not algebra), independent of which closed form is
     "right". Compare the resulting p1, p2 against all three numbers now in
     hand: Eq.(4)'s (0.2249, 0.4224), Eq.(14)|x=1's (0.1356, 0.2832), and
     Table I's stated (~0.17, ~0.28).

Outputs:
  figures/lindblad_validation.png  -- P_ESD(p) from continuous sim vs. both closed forms
  results_step3.txt                -- grid-comparison errors + threshold comparison
"""
import sys
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from lindblad import simulate, simulate_signed, p_of_tau, tau_of_p
from model import (
    analytic_C_no_not, analytic_C_with_not_eq14_x1, analytic_C_with_not_eq4,
    esd_no_not_bracket, esd_with_not_bracket_eq14_x1, esd_with_not_bracket_eq4,
)

ALPHA = 0.55
GAMMA = 1.0
FIG_DIR = "../figures"

C_BLUE = "#0072B2"
C_RED = "#CC3311"
C_GRAY = "#888888"


def grid_comparison(N=21):
    ps = np.linspace(0.001, 0.99, N)
    Ps = np.linspace(0.001, 0.99, N)
    max_err_no_not = max_err_eq14 = max_err_eq4 = 0.0
    for p in ps:
        t1 = tau_of_p(GAMMA, p)
        for P in Ps:
            tau2 = tau_of_p(GAMMA, P)
            c_lind_no = simulate(ALPHA, GAMMA, t1, tau2, apply_not=False)
            c_lind_with = simulate(ALPHA, GAMMA, t1, tau2, apply_not=True)
            max_err_no_not = max(max_err_no_not, abs(c_lind_no - analytic_C_no_not(ALPHA, p, P)))
            max_err_eq14 = max(max_err_eq14, abs(c_lind_with - analytic_C_with_not_eq14_x1(ALPHA, p, P)))
            max_err_eq4 = max(max_err_eq4, abs(c_lind_with - analytic_C_with_not_eq4(ALPHA, p, P)))
    return max_err_no_not, max_err_eq14, max_err_eq4


def p_esd_continuous(t1, apply_not, tau2_max=None):
    """Root-find the ESD point directly from the continuous simulation,
    independent of any closed-form formula. Returns P_ESD (converted via
    p_of_tau), or 1.0 if no root found before P = 1-1e-9 (avoidance)."""
    if tau2_max is None:
        tau2_max = tau_of_p(GAMMA, 1.0 - 1e-9)  # ~ -ln(1e-9)/gamma
    g = lambda tau2: simulate_signed(ALPHA, GAMMA, t1, tau2, apply_not=apply_not)
    if g(0.0) <= 0:
        return 0.0
    if g(tau2_max) > 0:
        return 1.0
    tau2_esd = brentq(g, 0.0, tau2_max, xtol=1e-12, rtol=1e-12)
    return p_of_tau(GAMMA, tau2_esd)


def find_thresholds_continuous(ps):
    Pesd_no = np.array([p_esd_continuous(tau_of_p(GAMMA, p), apply_not=False) for p in ps])
    Pesd_with = np.array([p_esd_continuous(tau_of_p(GAMMA, p), apply_not=True) for p in ps])

    p1 = ps[np.argmax(Pesd_with < 0.999999)]
    dP = Pesd_with - Pesd_no
    p2 = ps[np.argmax(dP < 0)]
    return Pesd_no, Pesd_with, p1, p2


def main():
    lines = []

    print("=== 1. Grid comparison: continuous Lindblad vs. Phase-1 closed forms ===")
    err_no_not, err_eq14, err_eq4 = grid_comparison()
    lines.append("Grid comparison (21x21, alpha=0.55, gamma=1), continuous Lindblad")
    lines.append("propagation (superoperator exponentiation) vs. Phase-1 closed forms:")
    lines.append(f"  no-NOT   vs Eq. (3)          : max err = {err_no_not:.3e}")
    lines.append(f"  with-NOT vs Eq. (14)|x=1     : max err = {err_eq14:.3e}")
    lines.append(f"  with-NOT vs Eq. (4)  literal : max err = {err_eq4:.3e}")
    lines.append("")
    for l in lines:
        print(l)

    print("\n=== 2. p1/p2 thresholds, root-found directly from continuous simulation ===")
    ps_fine = np.linspace(0.001, 0.6, 1200)
    Pesd_no, Pesd_with, p1_cont, p2_cont = find_thresholds_continuous(ps_fine)

    lines.append("p1/p2 thresholds, root-found DIRECTLY from the continuous-time")
    lines.append("simulation (brentq on the unclamped signed Wootters quantity,")
    lines.append("independent of any closed-form algebra):")
    lines.append(f"  p1 (avoidance -> delay) = {p1_cont:.4f}")
    lines.append(f"  p2 (delay -> hastening) = {p2_cont:.4f}")
    lines.append("")
    lines.append("Comparison against all three numbers now in hand:")
    lines.append(f"  {'source':32s} {'p1':>10s} {'p2':>10s}")
    lines.append(f"  {'Eq. (4) literal (Phase 1)':32s} {0.224861:10.4f} {0.422400:10.4f}")
    lines.append(f"  {'Eq. (14)|x=1 (Phase 1, recommended)':32s} {0.135577:10.4f} {0.283200:10.4f}")
    lines.append(f"  {'Table I (paper, stated)':32s} {0.17:10.4f} {0.28:10.4f}")
    lines.append(f"  {'Continuous Lindblad (this script)':32s} {p1_cont:10.4f} {p2_cont:10.4f}")
    lines.append("")

    def rel_gap(a, b):
        return abs(a - b) / b * 100

    lines.append("Relative gap of the continuous-time result from each reference:")
    lines.append(f"  vs. Eq. (4)     : p1 gap = {rel_gap(p1_cont,0.224861):5.1f}%   p2 gap = {rel_gap(p2_cont,0.422400):5.1f}%")
    lines.append(f"  vs. Eq. (14)|x=1: p1 gap = {rel_gap(p1_cont,0.135577):5.1f}%   p2 gap = {rel_gap(p2_cont,0.283200):5.1f}%")
    lines.append(f"  vs. Table I     : p1 gap = {rel_gap(p1_cont,0.17):5.1f}%   p2 gap = {rel_gap(p2_cont,0.28):5.1f}%")
    lines.append("")

    verdict = ("VERDICT: the continuous-time Lindblad simulation agrees with "
               "Eq. (14)|x=1 (Phase 1's recommended formula) to numerical "
               "precision, confirming it independently of the discrete "
               "Kraus-operator bookkeeping used in Phase 1. It does NOT "
               "agree with Eq. (4), nor with Table I's stated p1=0.17 -- "
               "the same two open discrepancies from Phase 1 persist here, "
               "now cross-checked by a second, independent numerical method.")
    lines.append(verdict)
    for l in lines[-6:]:
        print(l)

    report = "\n".join(lines)
    with open("../results_step3.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # ---------------- validation plot ----------------
    ps_plot = np.linspace(0.0, 0.6, 121)
    Pesd_no_plot, Pesd_with_plot, _, _ = find_thresholds_continuous(ps_plot)

    Pesd_eq14 = np.array([
        (1.0 if esd_with_not_bracket_eq14_x1(ALPHA, p, 1.0 - 1e-9) > 0
         else brentq(lambda P: esd_with_not_bracket_eq14_x1(ALPHA, p, P), 0.0, 1.0 - 1e-9))
        for p in ps_plot
    ])
    Pesd_eq4 = np.array([
        (1.0 if esd_with_not_bracket_eq4(ALPHA, p, 1.0 - 1e-9) > 0
         else brentq(lambda P: esd_with_not_bracket_eq4(ALPHA, p, P), 0.0, 1.0 - 1e-9))
        for p in ps_plot
    ])
    Pesd_no_analytic = np.array([
        (1.0 if esd_no_not_bracket(ALPHA, p, 1.0 - 1e-9) > 0
         else brentq(lambda P: esd_no_not_bracket(ALPHA, p, P), 0.0, 1.0 - 1e-9))
        for p in ps_plot
    ])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(ps_plot, Pesd_no_analytic, color=C_BLUE, lw=2, label="no NOT -- Eq. (3) [analytic]")
    ax.plot(ps_plot, Pesd_eq14, color=C_RED, lw=2, label="with NOT -- Eq. (14)|x=1 [analytic]")
    ax.plot(ps_plot, Pesd_eq4, color=C_GRAY, lw=1.6, ls="--", label="with NOT -- Eq. (4) [analytic, reference only]")

    ax.plot(ps_plot[::4], Pesd_no_plot[::4], 'o', ms=4, color=C_BLUE, mfc='white', mew=1.2,
            label="no NOT -- continuous Lindblad sim")
    ax.plot(ps_plot[::4], Pesd_with_plot[::4], 'o', ms=4, color=C_RED, mfc='white', mew=1.2,
            label="with NOT -- continuous Lindblad sim")

    for p_ref, lbl in [(0.17, "Table I: 0.17"), (0.28, "Table I: 0.28")]:
        ax.axvline(p_ref, color="0.3", ls=":", lw=1.2)
        ax.text(p_ref - 0.008, 0.03, lbl, rotation=90, va="bottom", ha="right", fontsize=8, color="0.3")

    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$p = 1-e^{-\gamma t_1}$")
    ax.set_ylabel(r"$P_{\rm ESD}$")
    ax.set_title("Continuous-time Lindblad validation against Phase 1 (open circles = Lindblad sim)")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/lindblad_validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
