"""
Phase 3: each atom (qubit) coupled to its own lossy cavity mode via the
Jaynes-Cummings interaction, with BOTH atomic spontaneous emission (gamma)
and cavity photon loss (kappa) as Lindblad dissipators:

    H = g (sigma_+ (x) a + sigma_- (x) a_dagger)          [resonant JC, hbar=1]
    drho/dt = -i[H, rho]
              + gamma * D[sigma_- (x) I](rho)              [atomic emission]
              + kappa * D[I (x) a](rho)                    [cavity loss]
    D[L](rho) = L rho L^dagger - 1/2 {L^dagger L, rho}

Each qubit has its OWN independent cavity (no coupling between qubit A's
cavity and qubit B's) -- only the two atoms are entangled, in the initial
state alpha|HH> + beta|VV>, cavities start in vacuum.

Hilbert-space truncation: this system starts with AT MOST ONE excitation
per atom-cavity subsystem (the atom is H or V, the cavity starts empty; H
and the two dissipators only ever remove excitation, never add it). So a
2-level Fock truncation (n=0,1) is EXACT for this problem, not an
approximation -- the atom-cavity subsystem is exactly 4-dimensional
(2 atom x 2 cavity Fock levels).

Locality argument (why this reduces to a per-qubit 4x4 process matrix,
exactly as Phase 1/2's single-qubit AD channel did): the NOT operation is
applied to the ATOM only, leaving that atom's own cavity mode untouched;
qubit A's (atom+cavity) subsystem never couples to qubit B's. So the whole
t1 -> NOT -> tau2 protocol, run independently and identically on each
(atom_i, cavity_i) pair starting from atom_i (x) |0><0|_cavity_i, and then
tracing out cavity_i at the very end, is a well-defined local CPTP map
Lambda on atom_i's 2x2 operator space alone -- regardless of any
atom-cavity entanglement or memory effects built up along the way. The
joint two-atom map for the WHOLE protocol is then exactly Lambda (x) Lambda,
built and applied exactly as in lindblad.py, just with a richer Lambda.
"""
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)   # atom: |H><V|
SIGMA_PLUS = SIGMA_MINUS.conj().T                          # atom: |V><H|
A_OP = np.array([[0, 1], [0, 0]], dtype=complex)           # cavity (2-level): a|1>=|0>
A_DAG = A_OP.conj().T

# atom (x) cavity, dim 4: basis order (H,0)=0,(H,1)=1,(V,0)=2,(V,1)=3
SIGMA_X_AC = np.kron(SIGMA_X, np.eye(2, dtype=complex))
NOT_ATOM_CAVITY = SIGMA_X_AC  # NOT on atom only, identity on cavity


def hamiltonian(g):
    """Resonant Jaynes-Cummings H = g(sigma_+ (x) a + sigma_- (x) a_dagger)."""
    return g * (np.kron(SIGMA_PLUS, A_OP) + np.kron(SIGMA_MINUS, A_DAG))


def liouvillian(g, gamma, kappa, dim=4):
    """d(vec rho)/dt = D vec(rho), atom(x)cavity Hilbert space, dim=4."""
    Iop = np.eye(dim, dtype=complex)
    H = hamiltonian(g)
    D = -1j * (np.kron(Iop, H) - np.kron(H.T, Iop))

    L_gamma = np.sqrt(gamma) * np.kron(SIGMA_MINUS, np.eye(2, dtype=complex))
    L_kappa = np.sqrt(kappa) * np.kron(np.eye(2, dtype=complex), A_OP)
    for L in (L_gamma, L_kappa):
        LdL = L.conj().T @ L
        D += np.kron(L.conj(), L)
        D -= 0.5 * np.kron(Iop, LdL)
        D -= 0.5 * np.kron(LdL.T, Iop)
    return D


def ptrace_cavity(rho4):
    """Partial trace over the cavity (2nd) factor of a 4x4 atom(x)cavity rho."""
    r = rho4.reshape(2, 2, 2, 2)  # [atom_row, cav_row, atom_col, cav_col]
    return np.einsum('anbn->ab', r)


def atom_process_matrix(t1, tau2, g, gamma, kappa, apply_not=True):
    """The single-atom 4x4 superoperator Lambda for the FULL t1 -> [NOT] -> tau2
    protocol, correctly propagated through one continuous atom+cavity
    evolution (no unjustified vacuum reset at t1). vec convention: column-major.
    """
    D = liouvillian(g, gamma, kappa)
    P1 = expm(D * t1) if t1 > 0 else np.eye(16, dtype=complex)
    P2 = expm(D * tau2) if tau2 > 0 else np.eye(16, dtype=complex)
    if apply_not:
        U = NOT_ATOM_CAVITY
        Unot_super = np.kron(U.conj(), U)
        full = P2 @ Unot_super @ P1
    else:
        full = P2 @ P1

    atom_basis = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (row, col), column-major order
    Lambda = np.zeros((4, 4), dtype=complex)
    for col_idx, (a, b) in enumerate(atom_basis):
        e_ab = np.zeros((2, 2), dtype=complex)
        e_ab[a, b] = 1.0
        rho_in = np.kron(e_ab, np.array([[1, 0], [0, 0]], dtype=complex))  # (x) |0><0|_cavity
        vec_in = rho_in.reshape(-1, order='F')
        vec_out = full @ vec_in
        rho_out_ac = vec_out.reshape(4, 4, order='F')
        atom_out = ptrace_cavity(rho_out_ac)
        Lambda[:, col_idx] = atom_out.reshape(-1, order='F')
    return Lambda


def initial_state(alpha):
    beta = np.sqrt(1.0 - alpha ** 2)
    psi = np.zeros(4, dtype=complex)
    psi[0] = alpha
    psi[3] = beta
    return np.outer(psi, psi.conj())


def concurrence(rho):
    sysy = np.kron(SIGMA_Y, SIGMA_Y)
    rho_tilde = sysy @ rho.conj() @ sysy
    r = rho @ rho_tilde
    evals = np.clip(np.linalg.eigvals(r).real, 0, None)
    lam = np.sort(np.sqrt(evals))[::-1]
    return max(0.0, lam[0] - lam[1] - lam[2] - lam[3])


def kraus_from_superop(Lambda, tol=1e-12):
    """Extract genuine 2x2 Kraus operators from a single-qubit 4x4
    superoperator (column-major natural representation) via its Choi
    matrix. NEEDED because the joint two-qubit map for two independent
    local channels Lambda (x) Lambda is NOT simply np.kron(Lambda, Lambda)
    applied to vec(rho_2qubit) -- that naive shortcut mixes up the tensor
    legs (vec of a 4x4 joint matrix is not vec_A (x) vec_B). Going through
    actual Kraus operators and applying kron(K_i, K_j) directly to the 4x4
    two-qubit rho, exactly as model.py/lindblad.py do, sidesteps that
    reshuffle entirely and was validated directly against the known AD
    Kraus operators at g=0 (recovers K1=diag(1,sqrt(1-p)),
    K2=offdiag(0,sqrt(p)) exactly, up to an irrelevant global phase).
    """
    T = np.zeros((2, 2, 2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    T[a, b, c, d] = Lambda[a + 2 * b, c + 2 * d]
    J = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    J[a * 2 + c, b * 2 + d] = T[a, b, c, d]
    evals, evecs = np.linalg.eigh(J)
    Ks = []
    for k in range(4):
        lam = evals[k]
        if lam < tol:
            continue
        v = evecs[:, k]
        Ks.append(np.sqrt(lam) * v.reshape(2, 2))
    return Ks


def simulate(alpha, t1, tau2, g, gamma, kappa, apply_not=True):
    """Full two-qubit protocol: each atom independently coupled to its own
    cavity, joint map built from Lambda's actual Kraus operators
    (Sum_ij (K_i (x) K_j) rho (K_i (x) K_j)^dagger), applied to the initial
    two-atom state; returns final concurrence."""
    Lambda = atom_process_matrix(t1, tau2, g, gamma, kappa, apply_not=apply_not)
    Ks = kraus_from_superop(Lambda)
    rho0 = initial_state(alpha)
    rho_out = np.zeros_like(rho0)
    for Ki in Ks:
        for Kj in Ks:
            KK = np.kron(Ki, Kj)
            rho_out += KK @ rho0 @ KK.conj().T
    return concurrence(rho_out)
