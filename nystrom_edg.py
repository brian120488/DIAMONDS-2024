import argparse
import numpy as np
import cvxpy as cvx
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes


def _truncated_psd_root(M, rank=None):
    lamb, U = np.linalg.eigh(M)

    idx = lamb.argsort()[::-1]

    if rank:
        lamb = lamb[idx][:rank]
        U = U[:,idx][:,:rank]
    else:
        lamb = lamb[idx]
        U = U[:,idx]

    lamb = np.diag(lamb)

    M = U @ np.sqrt(lamb)

    return M


def recover_nystrom_edg(D, m: int, r: int, sampling_rate: float = 0.4):
    """
    Input:
        D:  m+n by m+n distance matrix
        m:  Nystrom parameter (# rows/cols)
        r:  ground truth dimension of points that 
            created D
        sampling_rate: rate at which to sample elements of F in D

    Returns: 
        K: estimate for the kernel matrix.
    """
    n = D.shape[0] - m

    E = D[:m, :m]
    F = D[:m, m:]

    # We use the particular centering from Platt
    # which lets us recover A from only E.
    J = np.eye(m)-(1/m)*np.ones((m,m))
    
    # sanity check: A = P[:m,:] @ P[:m,:].T agrees with this
    A = -0.5*((J @ E) @ J)
    
    # The last row of F is considered known, so we 
    # only sample the first (m-1) rows... 
    F_mask = np.random.binomial(1, sampling_rate, (m-1, n))
    
    # ...then add on a zero row so the sizes match.
    F_mask = np.vstack((F_mask, np.zeros(n)))
    
    # Using our mask for F, 
    # make indices we'll need for the CVX constraints
    I,J = np.nonzero(F_mask)
    
    M = np.repeat(m-1, J.size)
    
    # make constant matrix C for row offsets
    E_sums = (1/m)*np.sum(E, axis=1)
    
    # subtract off the last sum
    E_sums -= E_sums[-1]
    
    # m x n
    C = np.tile(E_sums[:,np.newaxis], n)

    # -- Start CVX program

    # X is (r, m), P is (m+n, r), P = [X' ; Y']
    X = P[:m, :].T

    Y = cvx.Variable(shape=(r,n))

    Xty = X.T @ Y

    constraints = [Xty[I, J] - Xty[M, J] == -0.5*(F[I, J] - F[M, J]) + 0.5 * C[I, J]]

    cost = cvx.norm(X.T @ Y, "nuc")

    p = cvx.Problem(cvx.Minimize(cost), constraints)
 
    p.solve()

    if p.status in ["infeasible", "unbounded"]:
        raise Exception(p.status)

    # -- End CVX program

    B = X.T @ Y.value

    C = (B.T @ np.linalg.pinv(A)) @ B

    # K = [A B; B' C];
    K = np.block([[A, B], [B.T, C]])

    return K



def evaluate_recovery(K, P, show_plot=False):
    """
    Input:
        K: estimated kernel matrix, (m+n, m+n)
        P: ground truth points, (m+n, r)
        show_plot: plot recovery comparison? (r=2 only)

    Output:
        RMSE
    """
    r = P.shape[1]

    show_plot = (show_plot and r < 3)

    # Eigendecomposition to recover points
    P_cvx = _truncated_psd_root(K, rank=r)
    
    # Procrustes analysis to align points
    R, _ = orthogonal_procrustes(P_cvx, P)

    P_cvx = P_cvx @ R
    
    RMSE = np.sqrt(np.mean(np.sum((P_cvx - P)**2, axis=1)))

    if show_plot:
        import matplotlib.pyplot as plt

        plt.scatter(P_cvx[:,0], P_cvx[:, 1], 10, 'r', '.')
        plt.scatter(P[:,0], P[:, 1], 10, 'b', 'o')
       
        plt.show()

    return RMSE


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Nystrom EDG recovery")
    
    parser.add_argument('-m', '--m-block-size', type=int, default=50)
    parser.add_argument('-n', '--n-block-size', type=int, default=200)
    parser.add_argument('-r', '--rank', type=int, default=2)
    parser.add_argument('-sr', '--sampling-rate', type=float, default=0.4)
    parser.add_argument('-sp', '--show-plot', action="store_true")

    args = parser.parse_args()

    m = args.m_block_size
    n = args.n_block_size
    r = args.rank
    sampling_rate = args.sampling_rate
    show_plot = args.show_plot

    print(f"Opts: m={m}, n={n}, r={r}, sampling_rate={sampling_rate}, show_plot={show_plot}")

    P = np.random.random((m+n, r))

    # This centering assumes the Platt choice of s.
    # Sanity check: the reconstruction is indeed worse
    # if you don't do this.
    P = P - np.mean(P[:m,:], axis=0)
    
    D = squareform(pdist(P))**2

    K = recover_nystrom_edg(D, m, r, sampling_rate)

    RMSE = evaluate_recovery(K, P, show_plot)

    print(RMSE)

