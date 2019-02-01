# -*- coding: utf-8 -*-

import numpy as np
import math
import random

#definition
ALPHA=0.001 #step size
BETA_1=0.9 #exponential decay rates for the moment estimates
BETA_2=0.999 #exponential decay rates for the moment estimates
THETA_0=random.random() #initial parameter vector
EPS=1e-08 #epsilon
EPOCH=10000
NINF=-10**10


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
    result=pow(x,theta)
    return result

def dx_dtheta(x,theta):
    #if x<0:
    #    result=NINF#*pow(x,theta)
    #else:
    result=math.log(x+EPS)*pow(x,theta)

    return result

#2乗誤差
def error_func(x, y):
    subs=x-y
    result=pow(subs, 2)
    return result

def df_dx(x, y):
    result=2*(x-y)

    return result


#define dataset
DS_X=[]
DS_y=[]
NUM_DATA=1000
#for d in range(-NUM_DATA//2, NUM_DATA//2):
for d in range( NUM_DATA):
    #print(d,flush=True)
    x=d*0.1
    DS_X.append(x)
    DS_y.append(parabola(x, 2)) # get d^2 as y



# while theta not converged do
while  t<EPOCH :

    # get data
    idx=random.randint(0,NUM_DATA-1)
    x=DS_X[idx]
    y=DS_y[idx]

    #increase time 
    t=t+1
    #get gradients w.r.t stochastic objective at timestep t

    print("x",x)
    print("y",y)
    print("theta_t_old",theta_t_old)
    #g_t=2*(parabola(x, theta_t_old)-y)*theta_t_old*pow(x,theta_t_old-1)
    #g_t=2*(parabola(x, theta_t_old)-y)*math.log(x)*pow(x,theta_t_old)
    print("loss:",error_func(parabola(x,theta_t_old),y))
    g_t=df_dx(parabola(x,theta_t_old),y)*dx_dtheta(x,theta_t_old)


    print("pow(x,theta-1)", pow(x,theta_t_old-1))
    print("g_t", g_t)
    print("t:",t,flush=True)

    #update biased first moment estimate
    m_t=BETA_1*m_t_old + (1- BETA_1)* g_t 
    #update biased second moment estimate
    v_t=BETA_2*v_t_old + (1-BETA_2) * pow(g_t,2)
    print("v_t",v_t)
    print("v_t_old", v_t_old)

    #compute bias corrected first moment estimate
    m_t_hat=m_t / (1-pow(BETA_1, t))
    #compute bias corrected second moment estimate
    v_t_hat=v_t / (1-pow(BETA_2, t))

    #update parameters
    print(EPS)
    print(v_t_hat)
    print(math.sqrt(v_t_hat + EPS))
    theta_t=theta_t_old -ALPHA * m_t_hat / (math.sqrt(v_t_hat + EPS)) 
    #prepare for the next iteration
    theta_t_old=theta_t

    print("theta_t:", theta_t,flush=True)


#show the resulting parameter
print("result:",flush=True)
print(theta_t,flush=True)
