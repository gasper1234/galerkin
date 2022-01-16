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

def data_save(name):
	np.save(name, RES)

N = 0

C_ref = 0.7577218702419775
def calculate():
	N_f = 80

	RES = np.zeros((N_f-1, N_f-1))

	for N in range(1, N_f):
		xM = 80
		A = matrix_A(N, xM)
		b = vec_b(N, xM)

		def coef(xM_v):
			A_n = A[:xM_v*N,:xM_v*N]
			b_n = b[:xM_v*N]
			T1 = timeit.default_timer()
			a = np.array([])
			for i in range(xM_v):
				a_bl = np.linalg.solve(A_n[i*N:(i+1)*N, i*N:(i+1)*N], b_n[i*N:(i+1)*N])
				a = np.concatenate([a, a_bl])
			T2 = timeit.default_timer()
			t = T2-T1
			return -np.dot(a, b_n)*32/np.pi, t

		def coef_simple(xM_v):
			A_n = A[:xM_v*N,:xM_v*N]
			b_n = b[:xM_v*N]
			T1 = timeit.default_timer()
			a = np.linalg.solve(A_n, b_n)
			T2 = timeit.default_timer()
			t = T2-T1
			return -np.dot(a, b_n)*32/np.pi, t


		for k in range(1, xM):
			C, t = coef(k)
			print(N)
			RES[N-1,k-1] = C
	return RES

def data_save(name):
	try:
		T = np.load(name)
	except:
		print('exception!!!')
		T = calculate()
		np.save(name, T)
	return T

T = data_save('data2.npy')

a = plt.imshow(np.log(abs(T-C_ref)))
plt.colorbar(a)
plt.ylim(0, 78)
plt.xlabel('M')
plt.ylabel('N')
plt.title(r'$ln(|C-C_{ref}|)$')
plt.show()

'''
C_100 = 0.7577211820889066
C_100_simple = 0.7577211820889066
plt.plot(x, abs(C_list-C_100), label='blocno')
plt.plot(x, abs(C_simple_list-C_100_simple), label='navadno')
plt.plot(x, abs(abs(C_simple_list-C_100_simple)-abs(C_list-C_100)), label='razlika')
plt.xlabel('M')
plt.ylabel(r'$|C-C_{100}|$')
plt.yscale('log')
plt.legend()
plt.show()
'''

'''
plt.plot(x, t_list, label='blocno')
plt.plot(x, t_simple_list, label='navadno')
plt.ylabel('t [s]')
plt.xlabel('M')
plt.legend()
plt.show()
'''