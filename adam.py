# -*- coding: utf-8 -*-
import math
import random

#定義
ALPHA=0.001 #ステップサイズ
BETA_1=0.9 #一次モーメントの指数減衰率
BETA_2=0.999 #二次モーメントの指数減衰率
THETA_0=random.random() #最適化するθの初期値
EPS=1e-08 #イプシロン
EPOCH=1000 #エポック数

#初期化------
m_0=0 #一次モーメント
m_t_old=m_0 #１時刻前の一次モーメント

v_0=0 #二次モーメント
v_t_old=v_0 #１時刻前の二次モーメント

t=0 #タイムステップ

#θ
theta_t=THETA_0
theta_t_old=THETA_0


#関数の定義------
#y=x^θ
def parabola(x, theta):
    result=pow(x,theta)
    return result

#上記の微分
def dy_dtheta(x,theta):
    result=math.log(x+EPS)*pow(x,theta)
    return result

#2乗誤差
def error_func(x, y):
    subs=x-y
    result=pow(subs, 2)
    return result

#２乗誤差の微分
def df_dy(f, y):
    result=2*(f-y)
    return result

#可視化用------
gts=[]
mts=[]
vts=[]
mthats=[]
vthats=[]
thetas=[]
losses=[]

#データセット------
DS_X=[]
DS_y=[]
NUM_DATA=1000
# yとして、x^2 を使う。
# 0<= x < 100
for d in range(NUM_DATA):
    x=d*0.1
    DS_X.append(x)
    DS_y.append(parabola(x, 2)) 

# 実行------
# 収束までではなくエポック数だけ回す
while  t<EPOCH :

    #データはランダムに送る
    idx=random.randint(0,NUM_DATA-1)
    x=DS_X[idx]
    y=DS_y[idx]

    #時刻増やす 
    t=t+1

    #目的関数の勾配g_t
    y_prime=parabola(x,theta_t_old)
    g_t=df_dy(y_prime,y)*dy_dtheta(x,theta_t_old)
    gts.append(g_t)

    #誤差
    losses.append(error_func(y_prime, y))

    # 一次モーメントの更新
    # beta_1が大きいと過去の値が支配的になる
    m_t=BETA_1*m_t_old + (1- BETA_1)* g_t 
    mts.append(m_t)
    # 二次モーメントの更新
    v_t=BETA_2*v_t_old + (1-BETA_2) * pow(g_t,2)
    vts.append(v_t)


    #一次モーメントのバイアス補正
    m_t_hat=m_t / (1-pow(BETA_1, t))
    mthats.append(m_t_hat)
    #二次モーメントのバイアス補正
    v_t_hat=v_t / (1-pow(BETA_2, t))
    vthats.append(v_t_hat)

    #パラメータθを更新
    theta_t=theta_t_old -ALPHA * m_t_hat / (math.sqrt(v_t_hat + EPS)) 
    #古いθとして保持
    theta_t_old=theta_t
    thetas.append(theta_t)


#結果
print("result:",theta_t)
