import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def SEIRV(y,births,beta,alpha,mu,gamma,theta,sigma,k,lambda_,rho,rate):
    [S, V, E, I, R,Q]=y
    dSdt =max(S+births-beta*S*I-alpha*S-mu*S,0)
    dEdt=max(E+beta*S*I-gamma*E+theta*beta*V*I-mu*E,0)
    dIdt=max(I+gamma*E*(1-rate)-sigma*I-mu*I,100)
    dQdt=max(Q+gamma*E*rate+sigma*I-(1-k)*lambda_*Q-k*rho*Q-mu*Q,0)
    dVdt=max(V+alpha*S-theta*beta*V*I-mu*V,0)
    dRdt=max(R+(1-k)*(lambda_*Q)-mu*R,0)
    new_I=gamma*E
    return [dSdt,dVdt,dEdt,dIdt,dRdt,dQdt],new_I
def fit_rate(a,b,hk_):
    S0=0.1*(pop-1200000)
    V0=0.9*(pop-1200000)
    # S0=0.6*pop
    # V0=0.4*pop
    E0=3000
    I0=500
    R0=0
    Q0=0
    H0=0
    y=[S0, V0, E0, I0, R0,Q0]
    alpha=0.001
    mu=0.0069/365
    gamma=1/5.5
    theta=0.5
    sigma=1/3.8

    k=0.01
    lambda_=1/10
    rho=1/15
    I_l=[]
    rate=0.68
    new_I=0

    for i in range(89):
        action=0
        beta_change=0.375
        if new_I>4000:
            beta_change=0.3

        rate=np.round((-0.0000001*b*(new_I)+0.01*a),2)
        beta=7.2*beta_change/pop
        y,new_I=SEIRV(y,births,beta,alpha,mu,gamma,theta,sigma,k,lambda_,rho,rate)
        I_l.append(new_I)
    err=sum((np.array(I_l[:])-np.array(hk_['incre']))**2)
    return err,I_l
if __name__ == '__main__':
    pop = 7200000
    births = 159
    hk_ = pd.read_excel("data/covid_hk.xlsx")
    al = []
    bl = []
    errl = []
    for a in range(50, 100):
        for b in range(100):
            error, I_l = fit_rate(a, b, hk_)
            errl.append(error)
            al.append(a)
            bl.append(b)
    a_star=al[np.argmin(errl)]
    b_star=bl[np.argmin(errl)]
    print(a_star,b_star)
    error, I_l = fit_rate(a_star,b_star,hk_)
    print(error)
    plt.figure(figsize=[15, 4])
    plt.title('', fontsize=10)
    plt.plot(np.array(I_l[:]), label='predict daily increase')
    plt.plot(hk_['incre'].tolist(), label='real daily increase')
    plt.show()
