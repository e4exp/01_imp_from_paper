#coding:utf-8

import numpy as np
import math
import random

#definition
ALPHA=0.001 #step size
BETA_1=0.9 #exponential decay rates for the moment estimates
BETA_2=0.999 #exponential decay rates for the moment estimates
THETA_0=random.random() #initial parameter vector
EPS=1e-08 #epsilon

#initialize
m_0=0 #1st moment 
m_t_old=m_0

v_0=0 #2nd moment
v_t_old=v_0

t=0 #time step

#theta
theta_t=THETA_0
theta_t_old=THETA_0



#放物線
def parabola(x, theta):

	result=math.pow(x,theta)
	return result

#2乗誤差
def error_func(x, y):
	subs=x-y
	result=math.pow(subs, 2)
	return result

f=error_func

#define dataset
DS_X=[]
DS_y=[]
NUM_DATA=1000
for d in range(NUM_DATA):
	DS_X.append(d)
	DS_y.append(parabola(d, 2)) # get d^2 as y


# while theta not converged do
while theta_t - theta_t_old > EPS:
	# get data
	idx=random.randint(NUM_DATA)
	x=DS_X[idx]
	y=DS_y[idx]

	#increase time 
	t=t+1
	#get gradients w.r.t stochastic objective at timestep t
	g_t=2*(parabola(x, theta_t_old)-y)*theta_t_old*math.pow(x,theta_t_old-1)

	#update biased first moment estimate
	m_t=BETA_1*m_t_old + (1- BETA_1)* g_t 
	#update biased second moment estimate
	v_t=BETA_2*v_t_old + (1-BETA_2) *math.pow(g_t,2)

	#compute bias corrected first moment estimate
	m_t_hat=m_t / (1-math.pow(BETA_1, t))
	#compute bias corrected second moment estimate
	v_t_hat=v_t / (1-math.pow(BETA_2, t))

	#update parameters
	theta_t=theta_t_old -ALPHA * m_t_hat / (math.sqrt(v_t_hat + EPS)) 
	#prepare for the next iteration
	theta_t_old=theta_t
	print("t:",t)
	print("theta_t:", theta_t)

#show the resulting parameter
print("result:")
print(theta_t)

