import numpy as np
import cvxpy 

m = 30
n = 1000
r = 10

P = np.random.random((m+n, r))
P = P - P.mean(axis=1)[:,None]

# r x m
X = P[:m, :].T

# r x n
Y = P[m:,:].T

# m x m
A = X.T @ X

# m x n
B = X.T @ Y

# n x n
C = Y.T @ Y

# (m + n) x (m + n)
G = np.block([[A, B], [B.T, C]])




#




# Nystrom approximation

C_nys = B.T @ np.linalg.pinv(A) @ B

G_nys = np.block([[A, B], [B.T, C_nys]])


# Modified Nystrom approximation
# Generate the partial inner product matrix for the "B" block
Weight_B1 = np.zeros((m-1, n))
for i in range(n):
    Weight_B1[:r, i] = 1

# m x n
Weight = np.vstack((Weight_B1, np.zeros(n)))

I,J = np.nonzero(Weight)
M = np.repeat(m-1, J.size)

Xr = X
Yr = cvxpy.Variable(shape=(r,n))

Xty = Xr.T @ Yr

constraints = [Xty[I, J] - Xty[M, J] == B[I, J] - B[M, J]]

# dummy minimization
p = cvxpy.Problem(cvxpy.Minimize(0), constraints)

p.solve()

if p.status in ["infeasible", "unbounded"]:
    raise Exception(p.status)

Yr = Yr.value
Br = Xr.T @ Yr

# Approximation of G using modified Nystrom
G_mod1 = np.block([[A, Br], [Br.T, Yr.T @ Yr]])
G_mod2 = np.block([[A, Br], [Br.T, Br.T @ np.linalg.pinv(A) @ Br]])

error1 = np.linalg.norm(G-G_nys,'fro')/np.linalg.norm(G,'fro')
error2 = np.linalg.norm(G-G_mod1,'fro')/np.linalg.norm(G,'fro')
error3 = np.linalg.norm(G-G_mod2,'fro')/np.linalg.norm(G,'fro')

print(error1)
print(error2)
print(error3)