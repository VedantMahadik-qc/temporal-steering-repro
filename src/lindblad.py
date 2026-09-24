"""
Phase 2: continuous-time Lindblad model for independent atomic spontaneous
emission, no cavity, no coherent Hamiltonian -- dissertation's jump operator
L_{gamma,i} = sqrt(gamma) * sigma_minus_i for each of the two qubits/atoms.

Master equation (H = 0):
    drho/dt = sum_i [ L_i rho L_i^dagger - 1/2 {L_i^dagger L_i, rho} ]

sigma_minus = |H><V| (V = excited/index 1 decays to H = ground/index 0),
matching model.py's convention exactly, so this is the textbook spontaneous-
emission Lindblad equation for the SAME two-level system Phase 1's discrete
amplitude-damping (AD) Kraus map describes.

Exact solution (single qubit, standard textbook result): tracing out the
vacuum field reservoir gives exactly the AD Kraus map with
    p(tau) = 1 - exp(-gamma * tau)
This is not an approximation -- integrating the Lindblad equation over a
duration tau is mathematically identical to applying the discrete AD Kraus
map with that p(tau). This module solves the ODE independently (via
superoperator exponentiation, not by invoking the Kraus formulas), so
comparing its output to model.py's closed forms is a genuine independent
check, not a tautology.

Propagation is done by exponentiating the vectorized Liouvillian
superoperator (exact to machine precision, not a finite-step integrator).
"""
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)  # |H><V|: V -> H


def liouvillian(lindblad_ops, dim=4):
    """Superoperator D (dim^2 x dim^2) such that d(vec rho)/dt = D vec(rho),
    for drho/dt = sum_k [L_k rho L_k^dagger - 1/2{L_k^dagger L_k, rho}],
    using column-stacking vec (vec(A X B) = (B^T kron A) vec(X)).
    """
    D = np.zeros((dim * dim, dim * dim), dtype=complex)
    Iop = np.eye(dim, dtype=complex)
    for L in lindblad_ops:
        LdL = L.conj().T @ L
        D += np.kron(L.conj(), L)
        D -= 0.5 * np.kron(Iop, LdL)
        D -= 0.5 * np.kron(LdL.T, Iop)
    return D


def two_qubit_lindblad_ops(gamma):
    """L1 = sqrt(gamma) * sigma_minus (x) I,  L2 = sqrt(gamma) * I (x) sigma_minus
    -- independent spontaneous emission on each qubit, no cavity, no coherent H.
    """
    sm = np.sqrt(gamma) * SIGMA_MINUS
    return [np.kron(sm, I2), np.kron(I2, sm)]


def propagator(gamma, tau, dim=4):
    """Exact superoperator exp(D * tau) for duration tau at rate gamma."""
    D = liouvillian(two_qubit_lindblad_ops(gamma), dim=dim)
    return expm(D * tau)


def propagate(rho, gamma, tau):
    """Evolve rho for duration tau under independent spontaneous emission."""
    if tau <= 0:
        return rho
    U = propagator(gamma, tau)
    vec = rho.reshape(-1, order='F')
    vec_out = U @ vec
    return vec_out.reshape(rho.shape, order='F')


def initial_state(alpha):
    beta = np.sqrt(1.0 - alpha ** 2)
    psi = np.zeros(4, dtype=complex)
    psi[0] = alpha   # |HH>
    psi[3] = beta    # |VV>
    return np.outer(psi, psi.conj())


def concurrence(rho):
    sysy = np.kron(SIGMA_Y, SIGMA_Y)
    rho_tilde = sysy @ rho.conj() @ sysy
    r = rho @ rho_tilde
    evals = np.clip(np.linalg.eigvals(r).real, 0, None)
    lam = np.sort(np.sqrt(evals))[::-1]
    return max(0.0, lam[0] - lam[1] - lam[2] - lam[3])


def concurrence_signed(rho):
    """Wootters quantity WITHOUT the max(0, ...) clamp: lambda1-lambda2-lambda3-lambda4.

    Positive before ESD, crosses zero exactly at the ESD point, stays
    negative (unphysical as "concurrence") after -- used purely as a smooth,
    sign-changing function for root-finding the ESD boundary directly from
    the continuous-time simulation, independent of any algebraic formula.
    """
    sysy = np.kron(SIGMA_Y, SIGMA_Y)
    rho_tilde = sysy @ rho.conj() @ sysy
    r = rho @ rho_tilde
    evals = np.clip(np.linalg.eigvals(r).real, 0, None)
    lam = np.sort(np.sqrt(evals))[::-1]
    return lam[0] - lam[1] - lam[2] - lam[3]


def not_both():
    return np.kron(SIGMA_X, SIGMA_X)


def simulate(alpha, gamma, t1, tau2, apply_not=True):
    """Full continuous-time protocol: free decay for t1, optional instantaneous
    sigma_x (x) sigma_x kick, free decay for tau2, then concurrence.

    t1: duration of the first damping stage (p-equivalent = 1 - exp(-gamma*t1))
    tau2: duration of the second damping stage (P-equivalent = 1 - exp(-gamma*tau2))
    """
    rho = initial_state(alpha)
    rho = propagate(rho, gamma, t1)
    if apply_not:
        U = not_both()
        rho = U @ rho @ U.conj().T
    rho = propagate(rho, gamma, tau2)
    return concurrence(rho)


def simulate_signed(alpha, gamma, t1, tau2, apply_not=True):
    """Same protocol as simulate(), but returns the unclamped signed Wootters
    quantity (see concurrence_signed) for root-finding the ESD boundary."""
    rho = initial_state(alpha)
    rho = propagate(rho, gamma, t1)
    if apply_not:
        U = not_both()
        rho = U @ rho @ U.conj().T
    rho = propagate(rho, gamma, tau2)
    return concurrence_signed(rho)


def p_of_tau(gamma, tau):
    return 1.0 - np.exp(-gamma * tau)


def tau_of_p(gamma, p):
    return -np.log(1.0 - p) / gamma
