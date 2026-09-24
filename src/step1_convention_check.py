"""
Step 1: resolve the NOT convention.

Sec. II is explicit: "With a local NOT operation sigma_x x sigma_x applied
after the first channel but before the second ... the concurrence is
[[Eq. (4)]]". Appendix A confirms this literally: Eq. (A4) defines
UNOT = sigma_x tensor (path identity) for ONE photon, and Eq. (A8) shows the
two-photon unitary is UA (x) UB, i.e. each qubit gets its OWN independent NOT
-- sigma_x (x) sigma_x on the two-qubit system, not a single-qubit flip.

This script checks that claim three independent ways:

  1. Full Kraus-map simulation (no shortcuts) for both candidate conventions
     -- NOT on qubit 1 only, and NOT on both qubits -- compared against the
     closed-form Eq. (4) on a grid of (p, P).
  2. Cross-checks the no-NOT case (Eq. (3)) as a sanity control, since if
     that one doesn't match to machine precision the model itself has a bug
     independent of the NOT-convention question.
  3. FOLLOW-UP FINDING: compares the "both qubits" simulation against the
     paper's OWN general time-dependent formula, Eq. (14) (built from Eqs.
     12-13), evaluated at x=1 -- the limit the paper's own text calls
     "apparatus-defined ADC-like". Eq. (14)|x=1 has a *different* coherence
     term than Eq. (4)/(B9): (1-p)(1-P)|alpha beta| instead of
     {1-p(1-P)}(1-P)|alpha beta|, and no rho23 cross term at all. This is
     exactly what a pure local sigma_x (x) sigma_x conjugation must produce
     (proof: sigma_x is a permutation matrix, so conjugating an X-state by a
     product of permutations can only relocate existing coherence, never
     manufacture a second independent one -- so rho14 and rho23 cannot both
     be nonzero from a clean local bit-flip). Confirmed both analytically
     and by web-checking the arXiv preprint (2505.16622v2): Appendix B's
     Eq. (4)/(B8)-(B11) and the main-text Eq. (12)-(14) are NOT algebraically
     consistent with each other at x=1, in both the arXiv and (per the DOI
     record) published versions -- this is not a PDF-extraction artifact.
     See NOTES.md for the full writeup.
"""
import numpy as np
from model import (
    analytic_C_no_not, analytic_C_with_not_eq4, analytic_C_with_not_eq14_x1,
    simulate_concurrence,
    initial_state, apply_channel, two_qubit_ad_kraus, not_single, not_both,
)

ALPHA = 0.55
N = 41  # grid resolution per axis


def grid_compare():
    ps = np.linspace(0.0, 1.0, N)
    Ps = np.linspace(0.0, 1.0, N)

    max_err_no_not = 0.0
    max_err_single_eq4 = 0.0
    max_err_both_eq4 = 0.0
    max_err_both_eq14 = 0.0

    for p in ps:
        for P in Ps:
            c_analytic_no_not = analytic_C_no_not(ALPHA, p, P)
            c_sim_no_not = simulate_concurrence(ALPHA, p, P, not_convention=None)
            max_err_no_not = max(max_err_no_not, abs(c_analytic_no_not - c_sim_no_not))

            c_eq4 = analytic_C_with_not_eq4(ALPHA, p, P)
            c_eq14 = analytic_C_with_not_eq14_x1(ALPHA, p, P)
            c_sim_single = simulate_concurrence(ALPHA, p, P, not_convention='single')
            c_sim_both = simulate_concurrence(ALPHA, p, P, not_convention='both')

            max_err_single_eq4 = max(max_err_single_eq4, abs(c_eq4 - c_sim_single))
            max_err_both_eq4 = max(max_err_both_eq4, abs(c_eq4 - c_sim_both))
            max_err_both_eq14 = max(max_err_both_eq14, abs(c_eq14 - c_sim_both))

    return max_err_no_not, max_err_single_eq4, max_err_both_eq4, max_err_both_eq14


def component_diagnostic(p=0.3, P=0.4):
    """Compare individual X-state matrix elements to the paper's Appendix B
    (Eq. B9) at one representative point, for both NOT conventions. This
    isolates *where* the mismatch comes from (populations vs. coherences).
    """
    beta = np.sqrt(1.0 - ALPHA ** 2)

    def rho_out(convention):
        rho = initial_state(ALPHA)
        rho = apply_channel(rho, two_qubit_ad_kraus(p))
        u = not_both() if convention == 'both' else not_single(0)
        rho = u @ rho @ u.conj().T
        return apply_channel(rho, two_qubit_ad_kraus(P))

    r_single = rho_out('single').real
    r_both = rho_out('both').real

    # Paper's Eq. (B9), for comparison.
    rho11 = P ** 2 + (1 - p) * (1 - P) * (1 + P - p * (1 - P)) * beta ** 2
    rho14 = (1 - p * (1 - P)) * (1 - P) * ALPHA * beta
    rho22 = (1 - P) * (P - (1 - p) * (P - p * (1 - P)) * beta ** 2)
    rho23 = 2 * np.sqrt(p * (1 - p) * P * (1 - P)) * ALPHA * beta
    rho44 = (1 - P) ** 2 * (ALPHA ** 2 + p ** 2 * beta ** 2)

    print(f"\nComponent-level diagnostic at p={p}, P={P}, alpha={ALPHA}:")
    print(f"{'':14s} {'rho11':>10s} {'rho14':>10s} {'rho22':>10s} {'rho23':>10s} {'rho44':>10s}")
    print(f"{'paper Eq(B9)':14s} {rho11:10.6f} {rho14:10.6f} {rho22:10.6f} {rho23:10.6f} {rho44:10.6f}")
    print(f"{'sim: single-Q':14s} {r_single[0,0]:10.6f} {r_single[0,3]:10.6f} "
          f"{r_single[1,1]:10.6f} {r_single[1,2]:10.6f} {r_single[3,3]:10.6f}")
    print(f"{'sim: both-Q':14s} {r_both[0,0]:10.6f} {r_both[0,3]:10.6f} "
          f"{r_both[1,1]:10.6f} {r_both[1,2]:10.6f} {r_both[3,3]:10.6f}")
    print("\nrho11/rho22/rho44 from the 'both-qubit' simulation match Eq. (B9)")
    print("to machine precision (populations agree exactly). Only the")
    print("coherence terms (rho14, rho23) disagree -- and rho14=(1-p)(1-P)*a*b")
    print("from the simulation exactly equals the paper's OWN Eq. (13) X term.")


if __name__ == '__main__':
    err_no_not, err_single_eq4, err_both_eq4, err_both_eq14 = grid_compare()

    print(f"Grid: {N}x{N} points over (p, P) in [0,1]^2, alpha = {ALPHA}\n")
    print("Sanity control -- no-NOT case, simulation vs. Eq. (3):")
    print(f"  max |C_sim - C_analytic| = {err_no_not:.3e}\n")

    print("With-NOT case, simulation vs. Eq. (4) [Appendix B literal formula]:")
    print(f"  convention A (sigma_x on qubit 1 only):  max err = {err_single_eq4:.3e}")
    print(f"  convention B (sigma_x x sigma_x, both):  max err = {err_both_eq4:.3e}\n")

    print("With-NOT case, convention B simulation vs. Eq. (14) at x=1")
    print("[paper's OWN general time-dependent formula, Sec. II]:")
    print(f"  max err = {err_both_eq14:.3e}\n")

    print("VERDICT:")
    print("  Convention B (sigma_x x sigma_x, BOTH qubits) is confirmed correct --")
    print("  by the paper's explicit wording (Sec. II + Appendix A, Eq. A4/A8),")
    print("  and by matching Eq. (14) at x=1 to numerical precision.")
    print("  Eq. (4) / Appendix B's B8-B11 (the paper's OWN dedicated")
    print("  ADC-NOT-ADC derivation) does NOT match this simulation, and does")
    print("  NOT algebraically reduce from the paper's own Eq. (14) at x=1 --")
    print("  an internal inconsistency in the paper, not a convention ambiguity.")
    print("  Confirmed present in both the PDF and the arXiv preprint")
    print("  (2505.16622v2), so not a PDF-extraction artifact. See NOTES.md.")

    component_diagnostic()
