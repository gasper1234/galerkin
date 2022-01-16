import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta
from matplotlib import cm
import timeit

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

def cev(x):
	return N_p-np.sqrt(N_p**2-(x-N_p)**2)

print(u(1, np.pi/2, [1, 2], 1, 2))

N = 10
xM = 10
N_p = 200

kot = np.linspace(0, np.pi , N_p)
r = np.linspace(0, 1, N_p)

res = np.zeros((N_p, N_p))

res_true = np.zeros((N_p, 2*N_p))

tic=timeit.default_timer()
A = matrix_A(N, xM)
b = vec_b(N, xM)
tic1=timeit.default_timer()
print('matrike', tic-tic1)

tic=timeit.default_timer()
a = np.linalg.solve(A, b)
tic1=timeit.default_timer()
print('sistem', tic-tic1)


tic=timeit.default_timer()
for i in range(len(kot)):
	for j in range(len(r)):
		res[i][j] = u(r[j], kot[i], a, N, xM)
tic1=timeit.default_timer()
print('funckije', tic-tic1)

#naredi resitve v obliki cevi

tic=timeit.default_timer()
for i in range(N_p):
	for j in range(2*N_p):
		if j < N_p:
			if np.sqrt( (1-i/N_p)**2 + (1-j/N_p)**2 ) < 1:
				res_true[i][j] = u( np.sqrt( (1-i/N_p)**2 + (1-j/N_p)**2 ), np.pi/2+np.arcsin( (1-j/N_p) / np.sqrt((1-i/N_p)**2+(1-j/N_p)**2) ), a, N, xM)
		else:
			if np.sqrt( (1-i/N_p)**2 + ((j-N_p)/N_p)**2 ) < 1:
				res_true[i][j] = u( np.sqrt( (1-i/N_p)**2 + ((j-N_p)/N_p)**2 ), np.arcsin( (1-i/N_p) / np.sqrt(((j-N_p)/N_p)**2 + (1-i/N_p)**2) ), a, N, xM)
tic1=timeit.default_timer()
print('preslikava', tic-tic1)

#normirano

max_val = max(res[N_p//2,:])
res /= max_val


max_val = max(res_true[:,N_p])
res_true /= max_val

Z = [0.15]+[np.sqrt(i/8) for i in range(1, 8)]+[0.98]

Z_mod = [10**(-10)]+Z+[1]

#naredi imshow kot contourf - neuporabno
'''
res_color = np.copy(res_true)
for y in range(len(res_true)):
	for x in range(len(res_true[0])):
		for i in range(len(Z_mod)-1):
			if res_true[y, x] >= Z_mod[i] and res_true[y, x] <= Z_mod[i+1]:
				if res_true[y, x] != 0:
					res_color[y, x] = Z_mod[i+1]
				else:
					res_color[y, x] = np.nan
'''

fig, ax = plt.subplots()
#im = plt.imshow(res_color, origin='upper')
#cb = plt.colorbar(im)

#plot cev

def plot_cev():
	fig, ax = plt.subplots()
	cmap = cm.viridis
	cset1 = ax.contourf(np.linspace(0, 2*N_p, 2*N_p), r*N_p, res_true, Z_mod, cmap=cm.get_cmap(cmap, len(Z_mod) - 1))
	ct = ax.contour(np.linspace(0, 2*N_p, 2*N_p), r*N_p, res_true, Z, colors='k',origin='upper')
	cb = plt.colorbar(cset1, orientation='horizontal')
	ax.plot(np.linspace(0, 2*N_p, 2*N_p), cev(np.linspace(0, 2*N_p, 2*N_p)), 'k', linewidth=3)
	ax.axis('equal')
	ax.set_xlim(-1, 2*N_p+1)
	ax.set_ylim(N_p, -1)
	ax.set_yticks([])
	ax.set_xlabel(r'$v/v_{max}$')
	ax.set_xticks([0, N_p, 2*N_p])
	ax.set_xticklabels(['r', '0', 'r'])
	ax.hlines(N_p, 0, 2*N_p,'k', linewidth=3)



#običen graf
def plot_navaden():
	Z_mod = [10**(-10)]+Z+[1]

	cmap = cm.viridis
	cset1 = ax.contourf(np.linspace(0, N_p, N_p), r*N_p, res, Z_mod, cmap=cm.get_cmap(cmap, len(Z_mod) - 1))
	ct = ax.contour(np.linspace(0, N_p, N_p), r*N_p, res, Z, colors='k',origin='upper')
	cb = plt.colorbar(cset1)
	ax.set_yticks([i for i in range(0, N_p, N_p//5)]+[len(res[0])-1])
	ax.set_yticklabels([str(round(i/N_p, 2))+r'$\pi$' for i in range(0, N_p+1, N_p//5)])
	ax.set_xticks([i for i in range(0, N_p, N_p//5)]+[len(res[0])-1])
	ax.set_xticklabels([i/N_p for i in range(0, N_p+1, N_p//5)])


plt.show()