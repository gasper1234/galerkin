import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta

def A_ij(m, n, m_l, n_l):
	if m == m_l:
		konst = -np.pi/2 * (n*n_l*(3+4*m)) / (2+4*m+n+n_l)
		return konst*beta(n+n_l-1, 3+4*m)
	else:
		return 0


def matrix_A(N, xM):
	A = np.zeros((N*xM, N*xM))

	for m in range(xM):
		for n in range(1, N+1):
			for m_l in range(xM):
				for n_l in range(1, N+1):
					A[m*N+n-1, m_l*N+n_l-1] = A_ij(m, n, m_l, n_l)

	return A


def b_i(m, n):
	return -2 / (2*m+1) * beta(2*m+3, n+1)


def vec_b(N, xM):
	b = np.zeros(N*xM)

	for m in range(xM):
		for n in range(1, N+1):
			b[m*N+n-1] = b_i(m, n)

	return b


def psi(x, fi, m, n):
	return x**(2*m+1) * (1-x)**n * np.sin((2*m+1)*fi)


def u(x, fi, a, N, xM):

	sum = 0
	for m in range(xM):
		for n in range(1, N+1):
			sum += a[N*m+n-1]*psi(x, fi, m, n)

	return sum

print(u(1, np.pi/2, [1, 2], 1, 2))

N = 10
xM = 10
N_p = 100

kot = np.linspace(0, np.pi , N_p)
r = np.linspace(0, 1, N_p)

res = np.zeros((N_p, N_p))

res_true = np.zeros((N_p, 2*N_p))

A = matrix_A(N, xM)
b = vec_b(N, xM)

a = np.linalg.solve(A, b)


for i in range(len(kot)):
	for j in range(len(r)):
		res[i][j] = u(r[j], kot[i], a, N, xM)

for i in range(N_p):
	for j in range(2*N_p):
		if j < N_p:
			if np.sqrt( (1-i/N_p)**2 + (1-j/N_p)**2 ) < 1:
				res_true[i][j] = u( np.sqrt( (1-i/N_p)**2 + (1-j/N_p)**2 ), np.pi/2+np.arcsin( (1-j/N_p) / np.sqrt((1-i/N_p)**2+(1-j/N_p)**2) ), a, N, xM)
		else:
			if np.sqrt( (1-i/N_p)**2 + ((j-N_p)/N_p)**2 ) < 1:
				res_true[i][j] = u( np.sqrt( (1-i/N_p)**2 + ((j-N_p)/N_p)**2 ), np.arctan( (1-i/N_p) / np.sqrt(((j-N_p)/N_p)**2 + (1-i/N_p)**2) ), a, N, xM)


plt.imshow(res_true)
plt.yticks([i for i in range(0, 100, 10)], [round(i*np.pi/100, 2) for i in range(0, 100, 10)])
plt.xticks([i for i in range(0, 100, 10)], [i/100 for i in range(0, 100, 10)])
plt.show()