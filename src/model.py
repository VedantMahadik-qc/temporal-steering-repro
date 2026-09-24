"""
Core physics: two independent amplitude-damping (AD) channels applied in
sequence to a two-qubit state |psi> = alpha|HH> + beta|VV>, with an optional
local NOT inserted between the two channels.

Reproduces Behera et al., "Temporal steering of entanglement decay with
single-shot control," Phys. Rev. A 114, 012416 (2026), Sec. II and Appendix B
(ADC-NOT-ADC case, their Eqs. (3), (4), (7)).

Basis order for the two-qubit Hilbert space is the tensor-product order
|HH>, |HV>, |VH>, |VV>  (qubit 1 first). Each single-qubit basis is (H, V)
with H = index 0 (ground / stable) and V = index 1 (excited / decays to H).

Concurrence uses the Wootters formula (dissertation Sec. 2.4.2, Eq. (20);
paper Eq. (2)): C(rho) = max(0, lambda1 - lambda2 - lambda3 - lambda4), where
lambda_i are the decreasing-order square roots of the eigenvalues of
rho (sy⊗sy) rho* (sy⊗sy).
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def ad_kraus(p):
    """Single-qubit amplitude-damping Kraus operators for damping strength p.

    K1 = diag(1, sqrt(1-p))   (basis order H, V)
    K2 = [[0, sqrt(p)], [0, 0]]   (V -> H with probability p)
    """
    k1 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]], dtype=complex)
    k2 = np.array([[0.0, np.sqrt(p)], [0.0, 0.0]], dtype=complex)
    return [k1, k2]


def two_qubit_ad_kraus(p):
    """Two-qubit Kraus operators for independent AD channels on each qubit."""
    k1, k2 = ad_kraus(p)
    ops = []
    for a in (k1, k2):
        for b in (k1, k2):
            ops.append(np.kron(a, b))
    return ops


def apply_channel(rho, kraus_ops):
    out = np.zeros_like(rho)
    for k in kraus_ops:
        out += k @ rho @ k.conj().T
    return out


def not_single(qubit=0):
    """sigma_x on one qubit only (identity on the other)."""
    if qubit == 0:
        return np.kron(SIGMA_X, I2)
    return np.kron(I2, SIGMA_X)


def not_both():
    """sigma_x ⊗ sigma_x, applied locally to both qubits simultaneously."""
    return np.kron(SIGMA_X, SIGMA_X)


def initial_state(alpha):
    """rho = |psi><psi|, |psi> = alpha|HH> + beta|VV>, beta = sqrt(1-alpha^2)."""
    beta = np.sqrt(1.0 - alpha ** 2)
    psi = np.zeros(4, dtype=complex)
    psi[0] = alpha   # |HH>
    psi[3] = beta    # |VV>
    return np.outer(psi, psi.conj())


def concurrence(rho):
    """Wootters concurrence, dissertation Eq. (20) / paper Eq. (2)."""
    sysy = np.kron(SIGMA_Y, SIGMA_Y)
    rho_tilde = sysy @ rho.conj() @ sysy
    r = rho @ rho_tilde
    evals = np.linalg.eigvals(r)
    evals = np.clip(evals.real, 0, None)
    lam = np.sort(np.sqrt(evals))[::-1]
    return max(0.0, lam[0] - lam[1] - lam[2] - lam[3])


def simulate_concurrence(alpha, p, P, not_convention=None):
    """Full Kraus-map simulation: AD(p) -> [NOT] -> AD(P) -> concurrence.

    not_convention: None (no NOT), 'single' (sigma_x on qubit 0 only),
    or 'both' (sigma_x ⊗ sigma_x on both qubits).
    """
    rho = initial_state(alpha)
    rho = apply_channel(rho, two_qubit_ad_kraus(p))
    if not_convention == 'single':
        u = not_single(0)
        rho = u @ rho @ u.conj().T
    elif not_convention == 'both':
        u = not_both()
        rho = u @ rho @ u.conj().T
    elif not_convention is not None:
        raise ValueError(f"unknown not_convention {not_convention!r}")
    rho = apply_channel(rho, two_qubit_ad_kraus(P))
    return concurrence(rho)


# ---------------------------------------------------------------------------
# Analytic closed forms from the paper (Sec. II / Appendix B).
# ---------------------------------------------------------------------------

def analytic_C_no_not(alpha, p, P):
    """Paper Eq. (3) / Eq. (B6): no NOT between the two AD channels."""
    beta = np.sqrt(1.0 - alpha ** 2)
    bracket = abs(alpha * beta) - abs((p * (1 - P) + P) * beta ** 2)
    return 2 * max(0.0, bracket * (1 - p) * (1 - P))


def analytic_C_with_not_eq4(alpha, p, P):
    """Paper Eq. (4) / Appendix B, Eqs. (B8)-(B11): sigma_x ⊗ sigma_x between
    the two channels, AS LITERALLY STATED in the paper's dedicated
    ADC-NOT-ADC appendix.

    NOTE (see NOTES.md, 'Follow-up investigation'): this formula does NOT
    match a from-scratch Kraus-map simulation of sigma_x ⊗ sigma_x conjugation
    (max error 0.073 over a (p,P) grid -- see step1_convention_check.py), and
    does not algebraically reduce, at x=1, from the paper's OWN general
    time-dependent formula, Eq. (14). Kept here for documentation /
    reproducibility of that discrepancy, not recommended as the target to
    match going forward -- use analytic_C_with_not_eq14_x1 instead.
    """
    beta = np.sqrt(1.0 - alpha ** 2)
    term1 = (1 - p * (1 - P)) * (1 - P) * abs(alpha * beta)
    term2 = (1 - P) * (P - (1 - p) * (P - p * (1 - P)) * beta ** 2)
    return 2 * max(0.0, term1 - term2)


def analytic_C_with_not_eq14_x1(alpha, p, P):
    """Paper Eq. (14) (general time-dependent x-parametrized formalism,
    Sec. II, built from Eqs. (12)-(13)), evaluated at x = 1 (the paper's own
    claimed ADC-like limit). N(x=1) = 1, so this reduces to

        C = 2(1-P) * max[0, (1-p)|alpha beta| - (P - (1-p)(P-p(1-P)) beta^2)]

    This is the formula actually verified against a full Kraus-map
    simulation of sigma_x ⊗ sigma_x conjugation between two independent AD
    channels: matches to 5.8e-9 over a (p,P) grid (limited by root-finding
    tolerance, not formula mismatch) -- see NOTES.md and
    step1_convention_check.py. RECOMMENDED as the with-NOT target formula.
    """
    beta = np.sqrt(1.0 - alpha ** 2)
    bracket = (1 - p) * abs(alpha * beta) - (P - (1 - p) * (P - p * (1 - P)) * beta ** 2)
    return 2 * max(0.0, bracket) * (1 - P)


# Back-compat alias: earlier version of this script used Eq. (4) as "the"
# with-NOT formula before the discrepancy above was found. Kept pointing at
# Eq. (4) so old references stay literal, but new code should prefer
# analytic_C_with_not_eq14_x1.
analytic_C_with_not = analytic_C_with_not_eq4


def esd_no_not_bracket(alpha, p, P):
    """Sign-carrying bracket for the no-NOT case (root defines the ESD line)."""
    beta = np.sqrt(1.0 - alpha ** 2)
    return abs(alpha * beta) - abs((p * (1 - P) + P) * beta ** 2)


def esd_with_not_bracket_eq4(alpha, p, P):
    """Sign-carrying bracket for Eq. (4)/(7) -- see analytic_C_with_not_eq4."""
    beta = np.sqrt(1.0 - alpha ** 2)
    term1 = (1 - p * (1 - P)) * abs(alpha * beta)
    term2 = P - (1 - p) * (P - p * (1 - P)) * beta ** 2
    return term1 - term2


def esd_with_not_bracket_eq14_x1(alpha, p, P):
    """Sign-carrying bracket for Eq. (14) at x=1 -- see analytic_C_with_not_eq14_x1."""
    beta = np.sqrt(1.0 - alpha ** 2)
    return (1 - p) * abs(alpha * beta) - (P - (1 - p) * (P - p * (1 - P)) * beta ** 2)


# Back-compat alias (see analytic_C_with_not above).
esd_with_not_bracket = esd_with_not_bracket_eq4
