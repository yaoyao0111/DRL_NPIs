import os
import pandas as pd
from environment import *
import math
import itertools
import torch
import matplotlib.pyplot as plt
from RL_runner import plot_demand_policy



def  generate_tra_zero(env):
    tra = []
    state = env.reset()
    for _ in range(1000):
        action=0
        next_state, reward, done = env.step(state, action,0)
        state_l=env.featurize(state)
        state_l.append(reward)
        tra.append(state_l)
        if done:
            break
        state = next_state
    return tra
def generate_thre_tra(env,num):
    tra = []
    state = env.reset()
    for _ in range(1000):
        state_l = env.featurize(state)
        if state_l[0] + state_l[1] > num:
            action = 1
        else:
            if state['continue_t'] < 6:
                action=int(state['last_act'])
            else:
                action= 0
        # or state_l[0] + state_l[1] > 171

        next_state, reward, done = env.step(state, action,0)
        state_l = env.featurize(state)
        state_l.append(reward)
        tra.append(state_l)
        if done:
            break
        state = next_state
    return tra


def plot_no_control(tra):
    plt.figure(figsize=(12, 4))
    plt.plot(tra.iloc[:, 1], color='r', label='covid beds demand')
    plt.plot(tra.iloc[:, 0], color='b', label='other dis beds demand')
    plt.plot(np.ones(len(tra)) * 244, color='g', label='beds capacity')
    xlabel = [0, 0, 200, 400, 600, 800, 1000]
    ylabel = [0, 0, 200, 500, 1000, 2000, 3000]
    plt.xticks(xlabel, size=12)
    plt.yticks(ylabel, size=12)
    plt.xlabel('Time [d]', size=12)
    plt.ylabel('Beds demand', size=12)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    NID = ['Influenza', 'RSV', 'Rhinovirus_Enterovirus', 'Adenovirus', 'Parainfluenza', 'Mycoplasma_Pneumoniae']
    const_adj = [1 / 1.0559, 1 / 7.7695, 1 / 7.6026, 1 / 8.1670, 1 / 7.1098]
    # initialize the Infected population and Susceptible population for other RIDs
    I_l = [58,
           5.41396557666954e-43,
           1.1412200467416312e-37,
           1.3124570402881998e-36,
           4.8168524769612755e-45]

    S_l = [347091,
           33280,
           36058,
           36045,
           35894]
    # hospitalization rate of other RIDs
    Bed_usage = [131 / 10000, 1617 / 10000, 887 / 10000, 1072 / 10000, 734 / 10000]

    # import the fitted beta of other RIDs
    beta_dic = {}
    for i in range(1, 5):
        Data = pd.read_csv('data/NID_betas/' + str(NID[i]) + '_parms.csv')
        beta_v = Data['beta'].tolist()
        beta_dic[str(NID[i])] = beta_v


    # load the parameters for prediction of influenza's transmission
    def seasonal(seasonality_param):
        seasonal_fluc = np.zeros(52)
        for k in range(52):
            seasonal_fluc[k] = seasonality_param[0] + seasonality_param[1] * math.sin(2 * math.pi * (k + 1) / 52) + \
                               seasonality_param[2] * math.cos(2 * math.pi * (k + 1) / 52) + seasonality_param[
                                   3] * math.sin(4 * math.pi * (k + 1) / 52) + seasonality_param[4] * math.cos(
                4 * math.pi * (k + 1) / 52)
        return seasonal_fluc


    seasonality_param = pd.read_excel('data/Influenza_param.xlsx', sheet_name='Sheet1', header=None).values
    seasonal_fluc = seasonal(seasonality_param)

    # initialize the parameter for environment setting
    theta = 0.5
    new_R = 7.2 * 0.5
    rate = 0.71
    start_day = 151
    is_other_dis_considered = 1

    env = environment(theta, new_R, I_l, S_l, beta_dic, NID, start_day, Bed_usage, const_adj,
                      seasonal_fluc, rate, is_other_dis_considered)

    # predict the transmission of COVID-19's and other RIDs' transmission without control
    tra_ = generate_tra_zero(env)
    tra_temp_ = pd.DataFrame(tra_)
    plot_no_control(tra_temp_)

    # predict the transmission of COVID-19's and other RIDs' transmission by setting the threshold at 220
    num=220
    tra_ = generate_thre_tra(env, num)
    tra_temp_ = pd.DataFrame(tra_)
    covid_demand = tra_temp_[1].tolist()
    other_dis_demand = tra_temp_[0].tolist()
    all_demand = tra_temp_[0] + tra_temp_[1].tolist()
    no_hos=0
    for i in all_demand:
        if i>244:
            no_hos+=(i-244)
    print(no_hos)
    policy_level = tra_temp_[2].tolist()
    plot_demand_policy(covid_demand, other_dis_demand, all_demand, policy_level)
    tra_temp_.to_csv('/Users/yaoyao/Desktop/research/r_disease/Code_for_paper/tra_thre.csv')
