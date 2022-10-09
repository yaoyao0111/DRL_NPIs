# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 20:05:05 2021

@author: Hanchu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math


def predtsirControl(times, births, beta, alpha, S0, I0, nround, stochastic, controlStart, controlEnd, betachange):
    global b
    I_mat = np.zeros([len(times) + 1, nround])
    S_mat = np.zeros([len(times) + 1, nround])
    alpha = alpha
    if len(beta) < len(times):
        beta = np.tile(beta, len(times))
    if len([births]) < len(times):
        births = np.tile([births], len(times))

    for i in range(nround):
        S = np.zeros(len(times) + 1)
        I = np.zeros(len(times) + 1)
        S[0] = np.round(S0)
        I[0] = np.round(I0)
        for t in np.arange(1, len(times) + 1):
            lambd = min(S[t - 1], pow(beta[t - 1] * S[t - 1] * (I[t - 1]), alpha))
            if (t >= controlStart) & (t <= controlEnd):
                lambd = min(S[t - 1], pow(beta[t - 1] * S[t - 1] * I[t - 1] * betachange, alpha))
            if t > controlEnd:
                b = lambd
            if stochastic:
                I[t] = sum(np.random.negative_binomial(1, I[t - 1] / (I[t - 1] + lambd), size=int(I[t - 1])))
            else:
                I[t] = lambd
            S[t] = max(S[t - 1] + births[t - 1] - I[t], 0)
            if (t >= controlStart) & (t <= controlEnd):
                S[t] = max(S[t - 1] * (1 - (89200 + 50400) / 7200000) + births[t - 1] * 0.6 - I[t],
                           0)  # birth rate have to less than 0.3
        I_mat[:, i] = I
        S_mat[:, i] = S

    return S_mat, I_mat


def sis_model(y: tuple,
              N: int,
              beta: float,
              L: int,
              D: int,
              birth: float,
              changebeta: float):
    """


    Parameters
    ----------
    y : tuple
        Current states SEIRAH.
       y = [S, I]
       S: # susceptible individuals
       I: # infected individuals
    t : int
        Timestep.
    beta : float
        Transmission rate.
    L: float
        Infectious period (days).
    D: float
        Immune period (days).
    N : int
        Population size.

    Returns
    -------
    dydt: tuple, next state.

    """
    S, I = y

    dSdt = (N - S - I) / L - (beta * I * S) / N * changebeta + S + birth

    dIdt = (beta * I * S) / N * changebeta - I / D + I

    dydt = (dSdt[0], dIdt[0])

    return dydt


def new_infection(dIdt, dSdt, influenza_alpha, seasonal_fluc, week):
    pinf = influenza_alpha * np.log(dIdt) + seasonal_fluc[week % 52]
    pinf = 1 / (1 + np.exp(-pinf))  # logit transform
    pinf = np.min([1, pinf + 0.001])
    dIdt = dSdt * pinf
    return dIdt


def seasonal(seasonality_param):
    seasonal_fluc = np.zeros(52)
    for k in range(52):
        seasonal_fluc[k] = seasonality_param[0] + seasonality_param[1] * math.sin(2 * math.pi * (k + 1) / 52) + \
                           seasonality_param[2] * math.cos(2 * math.pi * (k + 1) / 52) + seasonality_param[
                               3] * math.sin(4 * math.pi * (k + 1) / 52) + seasonality_param[4] * math.cos(
            4 * math.pi * (k + 1) / 52)
    return seasonal_fluc


if __name__ == '__main__':
    b = 0
    RealData = pd.read_excel('data/Realfit.xlsx', sheet_name='Sheet1')
    ChangedBeta = np.loadtxt('data/ChangedBeta.csv')
    NID = ['Influenza', 'RSV', 'Rhinovirus_Enterovirus', 'Adenovirus', 'Parainfluenza', 'Mycoplasma_Pneumoniae']
    seasonality_param = pd.read_excel('data/Influenza_param.xlsx', sheet_name='Sheet1', header=None).values
    seasonal_fluc = seasonal(seasonality_param)
    influenza_alpha = 0.1503
    MCMCpredict = pd.read_excel('data/New_Infectious.xlsx', sheet_name='Sheet2')

    # Data for influx
    RH = pd.read_excel('data/HumidityHK.xlsx', sheet_name='Sheet2').values
    temperature = pd.read_excel('data/HumidityHK.xlsx', sheet_name='Sheet3').values
    # RH = humidity.values
    es = 611.2 * np.exp(17.67 * temperature / (temperature + 243.5))
    e = RH / 100 * es
    SH = (1 - 0.378) * e / (101325 - (0.378 * e))
    Dw = 18.016 * es * (RH / 100) / ((temperature + 273.1) * 8.314472)
    R0 = np.exp(-180 * SH + np.log(3 - 1.2)) + 1.2

    simulationyear = 58
    times = np.arange(0, simulationyear, 1 / 52)
    controlWeekStart = 7
    controlWeekLength = 156  # important for sensitive analysis
    controlStart = 50 * 52 + controlWeekStart
    controlEnd = 50 * 52 + controlWeekStart + controlWeekLength
    pop = 7200000
    births = 1099
    nround = 1
    stochastic = 0

    I0 = 0.8 * pop
    S0 = 0.2 * pop
    StateRecord = []
    for i in np.arange(0, 5):
        CumlativeIData = []
        Data = pd.read_csv('data/NID_betas/' + str(NID[i]) + '_parms.csv')
        beta = Data['beta'].values
        betahigh = Data['betahigh'].values
        betalow = Data['betalow'].values
        betachange = 1 - ChangedBeta[i]

        plotstart = 44  # the year that applied data
        finalrate = 0
        real_control = 88  # the time contorl last
        if i == 0:

            I = 0.2 * pop
            S = pop - I
            state = []
            dIdt = I
            dSdt = S
            for j in range(simulationyear * 52):
                # changebeta = 1
                state.append((dSdt, dIdt))  # using pop/100 in MCMC simulation
                dIdt = new_infection(dIdt, dSdt, influenza_alpha, seasonal_fluc, j)
                dSdt = dSdt - dIdt + births
                if (j >= (controlStart)) & (j <= controlEnd):
                    changebeta = 0.01
                    dIdt = new_infection(dIdt, dSdt, influenza_alpha, seasonal_fluc, j) * changebeta
                    dSdt = dSdt - dIdt + births
            state.append((dSdt, dIdt))
            Stateplot = np.array(state)
            # Stateplot = np.delete(Stateplot, 0, axis=0)
            I_mean = Stateplot[:, 1]
            S_mean = Stateplot[:, 0]
            # A = I_mean
        else:
            Resultsave = []
            ResultIsave = []
            Resultalphasave = []
            alpha = [0, 0.97, 0.97, 0.97, 0.97, 0.97]
            # for alpha in np.arange(0.1, 2.01, 0.001):
            S_mat, I_mat = predtsirControl(times, births, beta, alpha[i], S0, I0, nround, stochastic, controlStart,
                                           controlEnd, betachange)
            S_mean = np.mean(S_mat, axis=1)
            I_mean = np.mean(I_mat, axis=1)
        realdata = np.array(RealData[str(NID[i])])
        NAcount = len(realdata[np.isnan(realdata)])
        if i == 0:
            MSEsave = []
            ratesave = []
            weightrange = np.arange(0.001, 20.1, 0.001)
            for rate in weightrange:  # weight equal to the constant term in Poisson model
                MSE = np.sum(np.square(I_mean[plotstart * 52 + NAcount:plotstart * 52 - real_control + len(
                    RealData[str(NID[i])])] - realdata[NAcount:-real_control] * rate))
                MSEsave.append(MSE)
            MSEsave = np.array(MSEsave)
            minloc = np.argmin(MSEsave)
            finalrate = 1
            I_mean[plotstart * 52: plotstart * 52 + 320] = np.array(MCMCpredict.iloc[:, 10])
        else:
            MSEsave = []
            ratesave = []
            weightrange = np.arange(0.001, 20.1, 0.001)
            for rate in weightrange:  # weight equal to the constant term in Poisson model
                MSE = np.sum(np.square(I_mean[plotstart * 52 + NAcount:plotstart * 52 - real_control + len(
                    RealData[str(NID[i])])] - realdata[NAcount:-real_control] * rate))
                MSEsave.append(MSE)
                # print(np.sum(I_mean[plotstart*52 + NAcount:plotstart*52-controlWeekLength + len(RealData[str(NID[i])])]), np.sum(realdata[NAcount:-controlWeekLength]*rate))
            MSEsave = np.array(MSEsave)
            minloc = np.argmin(MSEsave)
        contant_adjustment = np.max(I_mean[plotstart * 52:]) / np.max(I_mean[plotstart * 52:(plotstart + 6) * 52]) - 1
        print(str(NID[i]), ':')
        print('contant_adjustment:', contant_adjustment)
        print('Susceptible population after control:', S_mean[controlEnd])
        print('Infected population after control:', I_mean[controlEnd])