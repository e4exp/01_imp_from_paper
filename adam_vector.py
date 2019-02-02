# -*- coding: utf-8 -*-
import math
import random

#定義
ALPHA=0.001 #ステップサイズ
BETA_1=0.9 #一次モーメントの指数減衰率
BETA_2=0.999 #二次モーメントの指数減衰率
THETA_0=random.random() #最適化するθの初期値
EPS=1e-08 #イプシロン
EPOCH=2000 #エポック数

#初期化------
m_0=0 #一次モーメント
m_t_old1=m_t_old2=m_0 #１時刻前の一次モーメント

v_0=0 #二次モーメント
v_t_old1=v_t_old2=v_0 #１時刻前の二次モーメント

t=0 #タイムステップ

#θ
theta_t1=theta_t2=THETA_0
theta_t_old1=theta_t_old2=THETA_0
theta_t_old=[theta_t_old1, theta_t_old2]


#モデル定義
#y(x1,x2,θ1,θ2 )=θ1* x1 + θ2* x2 
def calc_y(x,theta):
  """
  x: float list length=2
  theta: float list length=2
  """
  result =theta[0]*x[0]+theta[1]*x[1]
  return result

#上記のθ1微分
def dy_dtheta_1(x):
  """
  x: float list length=2
  """
  result=x[0]
  return result
  
#上記のθ2微分
def dy_dtheta_2(x):
  """
  x: float list length=2
  """
  result=x[1]
  return result

#2乗誤差
def error_func(x, y):
    """
      x: float
      y: float
    """
    subs=x-y
    result=pow(subs, 2)
    return result

#２乗誤差の微分
def df_dy(f, y):
    """
      f: float
      y: float
    """
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
steps=[]

#データセット------
DS_X=[]
DS_y=[]
NUM_DATA=1000
RAND_MAX=10.0

# yとして、x1*0.9 + x2*2.9 を使う。
# θ1=0.9,  θ2=2.9
THETA_1=0.9
THETA_2=2.9
THETA=[THETA_1,THETA_2]

for i in range(NUM_DATA):
    x1=random.uniform(-RAND_MAX,RAND_MAX)
    x2=random.uniform(-RAND_MAX,RAND_MAX)
    x_=[x1,x2]
    ans=calc_y(x_,THETA)
    DS_X.append(x_)
    DS_y.append(ans) 

    
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
    y_prime=calc_y(x,theta_t_old)
    g_t1=df_dy(y_prime,y)*dy_dtheta_1(x)
    g_t2=df_dy(y_prime,y)*dy_dtheta_2(x)
    gts.append([g_t1, g_t2])

    #誤差
    losses.append(error_func(y_prime, y))

    # 一次モーメントの更新
    # beta_1が大きいと過去の値が支配的になる
    m_t1=BETA_1*m_t_old1 + (1- BETA_1)* g_t1
    m_t2=BETA_1*m_t_old2 + (1- BETA_1)* g_t2
    mts.append([m_t1, m_t2])
    # 二次モーメントの更新
    v_t1=BETA_2*v_t_old1 + (1-BETA_2) * pow(g_t1, 2)
    v_t2=BETA_2*v_t_old2 + (1-BETA_2) * pow(g_t2, 2)
    vts.append([v_t1,v_t2])

    #一次モーメントのバイアス補正
    m_t_hat1=m_t1 / (1-pow(BETA_1, t))
    m_t_hat2=m_t2 / (1-pow(BETA_1, t))
    mthats.append([m_t_hat1, m_t_hat2])
    
    #二次モーメントのバイアス補正
    v_t_hat1=v_t1 / (1-pow(BETA_2, t))
    v_t_hat2=v_t2 / (1-pow(BETA_2, t))
    vthats.append([v_t_hat1, v_t_hat2])

    #パラメータθを更新
    theta_t1=theta_t_old1 -ALPHA * m_t_hat1 / (math.sqrt(v_t_hat1 + EPS)) 
    theta_t2=theta_t_old2 -ALPHA * m_t_hat2 / (math.sqrt(v_t_hat2 + EPS)) 
    steps.append(-ALPHA * m_t_hat1 / (math.sqrt(v_t_hat1 + EPS)) )#, m_t_hat2 / (math.sqrt(v_t_hat2 + EPS))])
    #古いθとして保持
    theta_t_old1=theta_t1
    theta_t_old2=theta_t2
    theta_t_old=[theta_t_old1,theta_t_old2]
    thetas.append([theta_t1,theta_t2])


#結果
print("theta_t1:",theta_t1)
print("theta_t2:",theta_t2)
