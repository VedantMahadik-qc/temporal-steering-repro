"""
Step 4 (Phase 2, part 2): the native continuous-time result. Fix a total
observation time T, sweep the intervention (NOT) time t1 in [0, T], and plot
final concurrence C(T) vs. t1 -- the genuinely continuous-time question Phase
1's discrete (p, P) grid can only approximate (there, p and P were
independent; here, p = 1-exp(-gamma t1) and the second stage's damping
P = 1-exp(-gamma(T-t1)) are coupled through a fixed total time budget T).

Without a NOT, C(T) does not depend on t1 at all: two independent AD stages
of durations t1 and T-t1 in sequence are equivalent to one AD stage of
duration T regardless of any intermediate bookkeeping (1-(1-p)(1-P) =
1 - exp(-gamma t1) exp(-gamma(T-t1)) = 1 - exp(-gamma T)), so it is plotted
as a flat reference line.

Note (sanity check, not a formula): concurrence is invariant under LOCAL
unitaries, so C(T) with NOT applied exactly at t1=T (i.e. after all the
damping, with zero time left to evolve) must exactly equal the no-NOT
baseline -- confirmed numerically in this script's output (the with-NOT
curve meets the no-NOT line at t1=T for every T).

Output: figures/continuous_time_sweep.png
"""
import numpy as np
import matplotlib.pyplot as plt

from lindblad import simulate

ALPHA = 0.55
GAMMA = 1.0
FIG_DIR = "../figures"

C_RED = "#CC3311"
C_BLUE = "#0072B2"


def sweep(T, n=241):
    t1s = np.linspace(0.0, T, n)
    c_with = np.array([simulate(ALPHA, GAMMA, t1, T - t1, apply_not=True) for t1 in t1s])
    c_no = simulate(ALPHA, GAMMA, 0.0, T, apply_not=False)  # t1-independent
    return t1s, c_with, c_no


def main():
    Ts = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)

    lines = ["Continuous-time sweep: C(T) vs. intervention time t1, alpha=0.55, gamma=1\n"]

    for ax, T in zip(axes, Ts):
        t1s, c_with, c_no = sweep(T)
        ax.plot(t1s, c_with, color=C_RED, lw=2, label="with NOT at $t_1$")
        ax.axhline(c_no, color=C_BLUE, lw=2, ls="--", label="no NOT (flat, $t_1$-independent)")
        ax.set_title(f"T = {T}")
        ax.set_xlabel(r"$t_1$ (NOT time)")
        ax.set_ylim(-0.02, 0.85)
        ax.set_xlim(0, T)

        i_best = int(np.argmax(c_with))
        t1_best, c_best = t1s[i_best], c_with[i_best]
        ax.plot([t1_best], [c_best], 'o', color=C_RED, ms=6, mfc='white', mew=1.5)

        lines.append(f"T={T}:  C_noNOT = {c_no:.4f}   "
                     f"best t1* = {t1_best:.4f}  ->  C(T)_max = {c_best:.4f}   "
                     f"(gain over no-NOT: {c_best - c_no:+.4f})")
        lines.append(f"       C(t1={T:.2f}=T, with NOT) = {c_with[-1]:.4f}  "
                     f"(sanity: should equal C_noNOT = {c_no:.4f}, local-unitary invariance)")

    axes[0].set_ylabel("Concurrence C(T)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(r"Native continuous-time result: $C(T)$ vs. intervention time $t_1 \in [0,T]$"
                 "\n(open circle = optimal $t_1^*$ for that T)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/continuous_time_sweep.png", dpi=180)
    plt.close(fig)

    report = "\n".join(lines)
    print(report)
    with open("../results_step4.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
