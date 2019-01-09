#####
##Analytical Linear Evolution of Spherical Density Perturbations
##Michelle Xu
##Last updated: 01/03/19
#####

#Note: the theoretical basis for this script can be found in the write up, "Linear Evolution: Analytical Pipeline"

import numpy as np
import matplotlib.pyplot as plt
from math import *
import numpy as np
import argparse
import csv
from scipy.interpolate import CubicSpline
from scipy import special
from finitediff import Derivative

def lin_evol(data, t0 = 15., m_0=10., m_psi=0.1, N=1, n=10, M=100, order=7, test=False):
	#inputs
	'''
	alpha is the value for alpha in the Bloomfield-Bulhosa-Face paper; in radiation dominated eras its value 0.5

	m_0 is the dimensionless waterfall field mass (divided by H)

	m_psi is the dimensionless timer field mass (divided by H)

	N is the number of waterfall fields

	n is the number of efolds between the waterfall transition and the end of inflation
    
	M is the number of interpolation points

	order is the order of the spline used for finitediff

	data is the file path for the csv of tilde{Phi}(tilde{chi}) samples

	test is the indication whether or not we're running through a test Gaussian and need lambda=2

	'''

	#set/calculate constants
	'''
	t0 is the end of inflation (in e-folds, so H is already factored in)

	p is needed to compute lambda

	lambda is needed to turn phi into delta_U 
	'''
	p =  1.5 - sqrt(2.25 - m_psi**2)
	lam = -1.5 + sqrt(2.25 + (1-e**(-2*p*t0))*(m_0)**2)
	if test: lam=2. #to test gaussian

	#process  zander's  csv
	'''
	chi and phi are numpy arrays of Zander's bar{chi} and bar{Phi} values respectively
	'''
	chi, phi = process_csv(data, N)

	#calculate delta_u0(bar{A})
	'''
	delta_u0 is an array of values of delta_u0(bar{A}) corresponding to A at xi=0

	A is an array of bar{A} values with corresponding delta_u0 values
	'''
	delta_u0, A = calc_delta_u0(chi, phi, lam, n, order)

	#calculate delta_rho_dot(bar{A})
	'''
	delta_rho_dot is an array of values of delta_rho_dot(bar{A}) corresponding to A at xi=0
	'''
	delta_rho_dot = calc_delta_rho_dot(delta_u0, A, chi, order)
	u = sqrt(3)*A
	L = u[u.size-1]

	#b_n = b_coeffs[n+1]; n goes from 1 to N/2-1
	b_coeffs = find_b_coeffs(delta_rho_dot, M, u, L) 

	#for Bn and Cn, n still goes from 1 to N/2-1
	Bn, Cn = find_general_coeffs(b_coeffs, L)

	#finding the xi when n=1 mode peaks
	xi_max = 2*log(2.7437072699922695*L/pi)

	#xi0 is the xi we will evolve to, either the point where delta_rho~0.05 or xi_max
	#maxornot=True means xi0=xi_max
	xi0, maxornot = find_xi0(Bn, Cn, L, xi_max)

	#evolving delta_rho, delta_m, delta_u and delta_r forward to xi0
	#they are all functions of A
	delta_rho, delta_m, delta_u, delta_r = find_deltas(xi0, Bn, Cn, L, A)

	#outputting results
	if maxornot: print "NOTE: n=1 peaked before delta_rho became nonlinear at the origin."
	print "Linear Evolution Completed."
	return xi0, A, delta_rho, delta_m, delta_u, delta_r

def process_csv(data, N):
	#reading in Zander's values from his csv
	'''
	according to someone on stack overflow, it's faster to create a list and then numpy array the list, then to directly append to numpy arrays; thus, the below implementation

	chi_list is the list of Zander's bar{chi} values

	phi_list is the list of Zander's tilde{Phi} values
	'''
	phi_list = []
	chi_list = []
	blah = 0
	with open(data, 'rb') as f:
		mycsv = csv.reader(f)
		for row in mycsv:
			if blah == 0:
				for col in row:
					chi_list.append(float(col))
				blah += 1
			else:
				for col in row:
					phi_list.append(float(col)/N)
	return np.array(chi_list), np.array(phi_list)


def calc_delta_u0(chi, phi, lam, n, order):
	#calculate delta_u0(r)
	'''
	r is a list of bar{r} values that align with the bar{chi} values from above
	delta_u0 is the function delta_u0(bar{r})
	'''
	#we implement a shift so that the derivatives will evaluate at zero
	shift = (chi[1]-chi[0])/2.
	chi = chi + shift
	diff = Derivative(order)
	diff.set_x(chi)
	diff.apply_boundary(1) #even boundary, since radial
	dphidchi = diff.dydx(phi)
	r = e**(n) * chi

	delta_u0 = -lam * (dphidchi) * ((phi)**(lam-1)) / (2*r*(exp(n)))

	#numerically invert to get \bar{r}(\bar{A})
	'''
	delta_t is a list of delta_t values which aligns with the bar{chi}, and thus ba{r}, values from above

	A is a list of bar{A} values that align with the bar{r} and thus bar{chi] values from above 

	this means delta_u0 is now also representative of the function delta_u0(bar{A})
	'''
	delta_t = - np.log(phi) / (2*lam) 
	A = r * e**(delta_t)


	'''
	#don't need to do this anymore because of the shift

	#fix delta_u0 at the origin
	#applying l'Hopital's rule; otherwise, at the origin the value will be 'inf'
	diff.apply_boundary(-1) #odd second derivative because first was even
	d2phidchi2=diff.dydx(dphidchi)
	delta_u0[0] = -lam * (d2phidchi2[0]) * ((phi[0])**(lam-1)) / (2.*(exp(2*n)))

	plt.figure()
	plt.plot(chi, d2phidchi2)
	plt.show()
	print d2phidchi2[0], d2phidchi2[1]
	'''


	return delta_u0, A

def calc_delta_rho_dot(delta_u0, A, chi, order):
	#calculate delta_rho_dot at xi=0 as a function of A
	'''
	chi is there so we can check against Mathematica plots
	'''
	diffA = Derivative(order)
	diffA.set_x(A)
	diffA.apply_boundary(1) #even boundary, since radial
	ddelta_u0dA = diffA.dydx(delta_u0)
	A6ddelta_u0dA = diffA.dydx((A**6)*delta_u0)

	delta_rho_dot = -2*delta_u0-A/3*ddelta_u0dA 
	delta_rho_dot_new = -A6ddelta_u0dA/(3*A**5) #old expression:

	#plot that shows our calculated delta_rho_dot (in 2 different ways)
	plt.figure()
	plt.plot(A, delta_rho_dot)
	plt.plot(A, delta_rho_dot_new)
	plt.show()

	return delta_rho_dot

def find_b_coeffs(delta_rho_dot, N, u, L):
	#find coefficients that allow us to interpolate delta_rho_dot
	'''
	delta_rho_dot is the function we're interpolating coefficients for

	N dictates the number of interpolation points (N/2-1)

	u is the x-axis and will provide both points and length

  	L is the length of the x-axis that we're interpolating 
	'''
	#create a cubic spline for delta_rho_dot so we can pull values of delta_rho_dot
	#at the interpolation points, not just the values of A
	first = (delta_rho_dot[1]-delta_rho_dot[0])/(u[1]-u[0])
	last = (delta_rho_dot[delta_rho_dot.size-1]-delta_rho_dot[delta_rho_dot.size-2])/(L-u[u.size-2])
	cs = CubicSpline(u, delta_rho_dot, bc_type=((1,first),(1,last)))

	#make a list of interpolation points
	x_list = []
	for i in range(1, N+1):
		val = -L + 2*L*i/N
		x_list.append(val)
  	x_n = np.array(x_list)
        
	#find interpolation coeffs
	b_coeffs_list = []
	for i in range(1, N/2):
		sum = 0.
		for k in range(0,N):
			x = x_n[k]
			if x < 0:
				x=x*-1 #because the interpolation is bad when x is negative, and j_0 and x^2 are even
			f = special.spherical_jn(0, i*pi/L*x) #the j_0 value
			g = cs(x) #the delta_rho_dot value
			sum += g * x**2 * f
		interpol = L * sum / N
		b = 2 * (pi*i)**2 / (L**3) * interpol
		b_coeffs_list.append(b)
  	b_coeffs = np.array(b_coeffs_list)
    
  	return b_coeffs

def find_general_coeffs(b_coeffs, L):
	#convert the interpolation coefficients to coefficients of our functional form
	'''
	b_coeffs is the array of coefficients that describe delta_dot_rho in the spherical bessel function basis

	the n that correspond to each b_n go from 1 to N/2-1
	'''
	Bn_list = []
	Cn_list = []
	#We basically need to solve the following linear system of equations for Bn and Cn (for each n)
	#Bn*j_1(n*pi/L)+Cn*y_1(n*pi/L) = 0
	#Bn(j_1(n*pi/L)+L/(n*pi)j_1'(n*pi/L)) + Cn(y_1(n*pi/L)+L/(n*pi)y_1'(n*pi/L)) = bn
	for i in range(b_coeffs.size):
		n = i + 1
		#breaking the expression down into pieces
		eq1 = [special.spherical_jn(1, n*pi/L), special.spherical_yn(1, n*pi/L)]
		eq2 = [special.spherical_jn(1, n*pi/L) + L/(n*pi)*special.spherical_jn(1, n*pi/L, True), special.spherical_yn(1, n*pi/L) + L/(n*pi)*special.spherical_yn(1, n*pi/L, True)]
		#LHS of above equations
		left = np.array([eq1, eq2])
		#RHS of above quations
		right = np.array([0., b_coeffs[i]])
		sol = np.linalg.solve(left, right)
		Bn_list.append(sol[0])
		Cn_list.append(sol[1])
	Bn = np.array(Bn_list)
	Cn = np.array(Cn_list)

	print Bn
	return Bn, Cn

def delta_rho_origin(x, Bn, Cn, L):
	#value of delta_rho(xi, A=0)
	'''
	x is xi, the time variable
	'''
	sum = 0
	for i in range(Bn.size):
		n = i+1
		first = Bn[i]*special.spherical_jn(1,n*pi/L*(np.exp(x/2)))
		second = Cn[i]*special.spherical_yn(1,n*pi/L*(np.exp(x/2)))
		sum += (first+second)
	func = sum*np.exp(x/2)*special.spherical_jn(0,0) #- 0.05
	return func

def find_xi0(Bn, Cn, L, xi_max):
	#our goal is to find the cutoff xi0, whether that will equal xi_max or if delta_rho gets to 0.05 first
	'''
	xi_max is the xi where the n=1 mode peaks
	'''
	#make a list of xis of small step size to test
	#recall s = exp(xi/2)
	#want points to be even in s, not in xi, since delta_rho is really a function of s
	s_max = exp(xi_max/2)
	s_array = np.linspace(0.,s_max,num=10000) 
	s_array[0] = 1 #will take a log to get back to xi and log(0) is not a happy number
	xis = 2*np.log(s_array)

	#search for xi s.t. either delta_rho(xi, A=0)>0.05 or xi>xi_max
	i = 0
	if delta_rho_origin(0, Bn, Cn, L) < 0.05:
		while(i < xis.size and delta_rho_origin(xis[i], Bn, Cn, L) < 0.05):
			i+=1
	else:
		while(i < xis.size and delta_rho_origin(xis[i], Bn, Cn, L) > 0.05):
			i+=1

	#determine whether we have xi0=xi_max or found a xi where delta_rho(xi, A=0)>0.05
	maxornot=False
	xi0 = 0.
	if i == xis.size:
		maxornot = True #true means that it didn't form a black hole/did not reach 0.05
		xi0 = xis[i-1]
	else:
		xi0 = xis[i]

	#plot of evolution of delta_rho at the origin through time
	plt.figure()
	plt.plot(xis, delta_rho_origin(xis, Bn, Cn, L))
	plt.show()
    
	return xi0, maxornot

def find_deltas(xi, Bn, Cn, L, A):
	#the goal is to evolve delta_rho, delta_m, delta_u and delta_r up to xi
	'''
	xi in this function is previously defined as xi0 in other parts of the code
	'''
	#find a few quantities not dependent on A 
	#each is a function of n, represented as an array corresponding to integer n
	#1 the time-dependent part of m (same as time-dependent part of rho)
	delta_m_t = []
	#2 the time-dependent part of delta_m_dot
	delta_m_dt = []
	#3 the time-dependent part of the time-integral of delta_m
	delta_m_intt = []

	for k in range(Bn.size):#looping through n
		n = k+1
		temp = Bn[k]*special.spherical_jn(1,n*pi/L*exp(xi/2))+Cn[k]*special.spherical_yn(1,n*pi/L*exp(xi/2))
		delta_m_t.append(temp)
        
		#same beta and gamma in the write up
		beta = 1./2.*(exp(xi/2)*special.spherical_jn(1,n*pi/L*exp(xi/2))+exp(xi)*n*pi/L*special.spherical_jn(1,n*pi/L*exp(xi/2), True))
		gamma = 1./2.*(exp(xi/2)*special.spherical_yn(1,n*pi/L*exp(xi/2))+exp(xi)*n*pi/L*special.spherical_yn(1,n*pi/L*exp(xi/2), True))
		temp2 = Bn[k]*beta+Cn[k]*gamma
		delta_m_dt.append(temp2)

		#same phi and  psi in the write up
		phi = -2*exp(-xi/2)*(L**2)/((n*pi)**2)*sin(n*pi/L*exp(xi/2))
		psi = 2*exp(-xi/2)*(L**2)/((n*pi)**2)*cos(n*pi/L*exp(xi/2))
		temp3 = Bn[k]*phi+Cn[k]*psi
		delta_m_intt.append(temp3)
        
	#now to find the deltas
	#each is a function of A, represented as an array corresponding to previously established points of A
	delta_rho_list = []
	delta_m_list = []
	delta_u_list = []
	delta_r_list = []

	for i in range(A.size):#looping through points in A
		#find various sums
		rho_sum = 0.
		m_sum = 0.
		m_dot_sum = 0.
		r_integral_sum = 0.

		for j in range(Bn.size):
			n = j+1
            
			#delta_rho
			rho_sum += special.spherical_jn(0,n*pi/L*sqrt(3)*A[i])*delta_m_t[j]
            
			#delta_m
			cos_part = -3*n*pi*A[i]*cos(sqrt(3)*n*pi/L*A[i])
			sin_part = sqrt(3)*L*sin(sqrt(3)*n*pi/L*A[i])
			if i==0: sigma=special.spherical_jn(0,0)/3.
			else: sigma = L**2/(9*(pi*n)**3)*(cos_part+sin_part)
			m_sum += sigma*delta_m_t[j]

			#delta_m_dot
			m_dot_sum += sigma*delta_m_dt[j]
            
			#that integral in r
			if i==0: r_integral_sum += delta_m_intt[j]*(special.spherical_jn(0, n*pi/L*sqrt(3)*A[i])-3*sigma)
			else: r_integral_sum += delta_m_intt[j]*(special.spherical_jn(0, n*pi/L*sqrt(3)*A[i])-3/(A[i]**3)*sigma)
			
		delta_rho_list.append(exp(xi/2)*rho_sum)
		
		if i==0: m_sum = exp(xi/2)*3*m_sum
		else: m_sum = exp(xi/2)*3/(A[i]**3)*m_sum
		delta_m_list.append(m_sum)

		if i==0: m_dot_sum = 3*m_dot_sum
		else: m_dot_sum = 3/(A[i]**3)*m_dot_sum
		u_total = -m_dot_sum/2 + m_sum/4
		delta_u_list.append(u_total)

		r_total = -(m_sum+r_integral_sum/2.)/4.
		delta_r_list.append(r_total)
	delta_rho = np.array(delta_rho_list)
	delta_m = np.array(delta_m_list)
	delta_u = np.array(delta_u_list)
	delta_r = np.array(delta_r_list)

	return delta_rho, delta_m, delta_u, delta_r

#--------------------------------------------------
#testing purposes

a,b,c,d,e,f = lin_evol("gaussian_sigma10to13.csv", test = True)

#--------------------------------------------------
