
from environment import *
from ddqn import *
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import itertools
import matplotlib.ticker as mticker

def generate_tra(env,agent):
    tra = []
    state = env.reset()
    for _ in range(1000):
        best_action, state_l= agent.step(state,env)
        next_state, reward, done = env.step(state, best_action,0)
        next_state_l=env.featurize(next_state)
        data = [state_l,state['continue_t'],int(state['last_act']), best_action, reward, next_state_l, done]
        tra.append(data)
        if done:
            break
        state = next_state
    return tra
def record_tra(env,agent):

    tra = []
    state = env.reset()
    for _ in range(1000):
        best_action= agent.eval_step(state,env)
        next_state, reward, done = env.step(state, best_action,0)
        if reward<0:
            print(best_action,reward)
        state_l=env.featurize(state)
        state_l.append(reward)
        tra.append(state_l)
        if done:
            break
        state = next_state
    return tra

def get_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    return device


def tournament(env, num, agent):
    reward_l = 0
    for i in range(num):
        state = env.reset()
        reward_sum=0
        for t in itertools.count():
            action = agent.eval_step(state, env)
            next_state, reward, done = env.step(state, action,0)
            reward_sum += reward
            if done:
                break
            state=next_state
        reward_l+=reward_sum
    return reward_l / num

def  generate_tra_act(env,act_l):
    tra = []
    state = env.reset()
    for _ in range(1000):
        action=act_l[_]
        Bed_usage = [131 / 10000, 1617 / 10000, 887 / 10000, 1072 / 10000, 734 / 10000]
        const_adj = [1 / 1.0559, 1 / 7.7695, 1 / 7.6026, 1 / 8.1670, 1 / 7.1098]
        rid_beds = np.array(state['I_inc']) * np.array(Bed_usage) * np.array(const_adj)
        next_state, reward, done = env.step(state, action,0)
        tra.append(sum(rid_beds))
        if done:
            break
        state = next_state
    return tra

def plot_demand_policy(covid_demand,other_dis_demand,all_demand, policy_level):
    fig, axs = plt.subplots(2, 1, figsize=(12, 5), tight_layout=True)

    axs[0].plot(covid_demand, color='r', label='covid bed demands')
    axs[0].plot(other_dis_demand, color='b', label='other dis ded demands')
    axs[0].plot(all_demand, color='y', label='all demands')
    axs[0].plot(np.ones(len(covid_demand)) * 244, color='g', label='beds capacity')

    xlabel = [0, 0, 200, 400, 600, 800, 1000]
    ylabel = [0, 0, 50, 100, 150, 200, 250]
    axs[0].xaxis.set_major_locator(mticker.FixedLocator(xlabel))
    axs[0].set_xticklabels(xlabel, fontsize=12)

    axs[0].yaxis.set_major_locator(mticker.FixedLocator(ylabel))
    axs[0].set_yticklabels(ylabel, fontsize=12)

    axs[1].plot(policy_level, label='policy level')

    axs[1].xaxis.set_major_locator(mticker.FixedLocator(xlabel))
    axs[1].set_xticklabels(xlabel, fontsize=12)
    ylabel = [0, 0, 1,2,3,4,5]
    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlabel('Time [d]', fontsize='large')
    axs[0].set_ylabel('beds demands', fontsize='large')

    axs[1].yaxis.set_major_locator(mticker.FixedLocator(ylabel))
    axs[1].set_yticklabels(ylabel, fontsize=12)
    axs[1].set_xlabel('Time [d]', fontsize='large')
    axs[1].set_ylabel('Level', fontsize='large')
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

    # set the parameter for agent training
    num_episode = 200
    evaluate_every=10

    device = get_device()

    # initialize the parameter for environment setting
    theta = 0.5
    new_R = 7.2 * 0.5
    rate = 0.71
    start_day = 151
    is_other_dis_considered = 0

    env = environment(theta, new_R, I_l, S_l, beta_dic, NID, start_day, Bed_usage, const_adj,
                      seasonal_fluc, rate, is_other_dis_considered)


    if is_other_dis_considered == 1:
        agent = DQNAgent(
            num_actions=6,
            state_shape=[3],
            mlp_layers=[64, 64],
            device=device
        )
    else:
        agent = DQNAgent(
            num_actions=6,
            state_shape=[2],
            mlp_layers=[64, 64],
            device=device
        )

    reward_l = []
    max_re = -100000
    for episode in range(num_episode):
        # Generate data from the environment
        tra= generate_tra(env,agent)
        for ts in tra:
            agent.feed(ts)
        if episode % evaluate_every == 0:
            re=tournament(env,1,agent)
            reward_l.append(re)
            if re>max_re:
                max_re=re
                tra = record_tra(env, agent)
                tra_temp = pd.DataFrame(tra)
                if is_other_dis_considered==1:
                    tra_temp.to_csv('/Users/yaoyao/Desktop/tra_considered.csv')
                else:
                    tra_temp.to_csv('/Users/yaoyao/Desktop/tra_not_considered.csv')
                print(re)
    if is_other_dis_considered == 1:
        covid_demand = tra_temp[1].tolist()
        other_dis_demand = tra_temp[0].tolist()
        all_demand = tra_temp[0]+tra_temp[1].tolist()
        policy_level = tra_temp[2].tolist()
        plot_demand_policy(covid_demand,other_dis_demand,all_demand, policy_level)
    else:
        act_l = tra_temp[1].tolist()
        other_tra = generate_tra_act(env, act_l)
        covid_demand = tra_temp[0].tolist()
        other_dis_demand = other_tra
        all_demand = +tra_temp[0] +other_dis_demand
        policy_level = tra_temp[1].tolist()
        plot_demand_policy(covid_demand, other_dis_demand, all_demand, policy_level)


