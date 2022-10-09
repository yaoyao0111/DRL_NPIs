# theta:vaccine inefficacy
# new_R: the reproduction number of the new mutant

import numpy as np
import copy
import math

class environment():
    def __init__(self, theta, new_R, I_l, S_l, beta_dict, NID, start_day, Bed_usage, const_adj,
                 seasonal_fluc,rate,is_other_dis_considered):
        pop = 7200000
        self.beta_dict = beta_dict
        self.NID = NID
        self.Bed_usage = Bed_usage
        self.seasonal_fluc = seasonal_fluc
        self.const_adj = const_adj
        self.start_day = start_day
        self.rate=rate
        self.theta = theta
        self.new_beta = new_R / pop
        self.I_l = I_l
        self.S_l = S_l
        # everyday birth in Hong Kong
        self.births = 159
        self.lambd_s = np.zeros(5)
        # initialize the state of COVID-19 transmission by data at 31/05/2022
        self.covid_s = [0.10 * (pop-1200000), 0.90 * (pop-1200000), 3000, 500, 1200000, 0]
        self.awarence=1
        self.is_other_dis_considered=is_other_dis_considered

        # gamma:average latent time
        # beta:R/N
        # alpha:vaccination rate
        # mu:natural death rate
        # theta:vaccine inefficacy
        # sigma:average quarantine time
        # k:mortality rate
        # lambda:average days until recovery
        # rho:average time until death

        self.alpha = 0.001
        self.mu = 0.0069/365
        self.gamma = 1 / 5.5
        self.sigma = 1 / 3.8
        self.k = 0.01
        self.lambda_ = 1 / 10
        self.rho = 1 / 15
        self.influenza_alpha = 0.1503


    def reset(self):
        state = {'I': self.I_l, 'S': self.S_l, 'day': self.start_day, 'continue_t': 0, 'last_act': 0,
                 'covid': self.covid_s,'covid_inf':0,'R':[0],'I_inc':np.ones(5),'beds':0}
        self.rate=0.68
        self.awarence = 1
        for i in range(88):
            state, reward, done=self.step(state,0,0)
        return state

    def step(self, state, action,reinfect_rate):

        next_state = copy.deepcopy(state)
        # Record the length of days a kind of control level lasts.
        continue_t = state['continue_t']
        if action <= state['last_act']:
            next_state['continue_t'] += 1
            continue_t+=1
        else:
            next_state['continue_t'] = 0
            continue_t=0
        next_state['last_act'] = action

        # Identify the effects of different level of control on transmission of RIDs
        if action==0:
            beta_change = 0.75
        elif action ==1:
            beta_change = 0.53
        elif action ==2:
            beta_change = 0.45
        elif action ==3:
            beta_change = 0.39
        elif action==4:
            beta_change = 0.23
        else:
            beta_change = 0.15


        day = state['day']
        week_abs = int(day / 7)%52
        count=day%7
        all_beds = 0

        # update other diseases transmission
        for i in range(1, 5):
            nid = self.NID[i]
            S = state['S'][i]
            I = state['I'][i]
            beta = self.beta_dict[nid][week_abs] * beta_change*0.5
            I_t, S_t, lambd = self.nid_sim(beta, S, I, count,i)
            beds_usage_i = lambd * self.Bed_usage[i] * self.const_adj[i]
            all_beds += beds_usage_i
            next_state['S'][i] = S_t
            next_state['I'][i] = I_t
            next_state['I_inc'][i] = lambd

        # update the flu transmission
        dIdt = state['I'][0]
        dSdt = state['S'][0]
        lambd = self.new_infection(dIdt, dSdt, week_abs) * beta_change/7
        self.lambd_s[0]+=lambd
        count+=1
        if count==7:
            dIdt_t = self.lambd_s[0]
            dSdt_t =  dSdt - dIdt_t + self.births*7
            self.lambd_s[0] = 0
        else:
            dIdt_t = dIdt
            dSdt_t = dSdt
        next_state['I_inc'][0] = dIdt_t

        next_state['I'][0] = dIdt_t
        next_state['S'][0] = dSdt_t
        beds_usage_0 = lambd * self.Bed_usage[0] * self.const_adj[0]
        all_beds += beds_usage_0

        # update the covid transmission
        new_beta = self.new_beta * beta_change* self.awarence
        y_new,I_new= self.SEIRV(state['covid'], new_beta,reinfect_rate)
        next_state['covid'] = y_new
        next_state['covid_inf'] = I_new

        # use the citizen's awarence to explain the reduction of daily infection when it hits 4000
        if I_new > 4000:
            self.awarence = 0.8

        # calculate the contact tracing rate
        self.rate=np.round((-0.0000073*(I_new)+0.71),2)
        #self.rate = np.round((-0.0000075 * (I_new) + 0.71), 2)
        # hospital beds demanded from the COVID-19
        beds_covid = I_new * 0.02
        next_state['R'].append(y_new[4])
        beds_day=244
        next_state['beds']=beds_day
        if self.is_other_dis_considered==1:
            if all_beds+beds_covid> beds_day:
                no_hos = (all_beds+beds_covid- beds_day)
            else:
                no_hos=0
        else:
            if beds_covid> beds_day:
               no_hos = (beds_covid- beds_day)
            else:
                no_hos=0

        # calculate the economic cost related to its effect on transmission reduction.
        # sum up the economic cost and health cost
        eco_cost = (0.75-beta_change)
        cost=no_hos*100+eco_cost
        reward =-cost

        # whether the process ends
        next_state['day']= state['day']+1
        if next_state['day'] >= 52 * 3*7+89+151:
            done = 1
        else:
            done = 0
        return next_state, reward, done

    def nid_sim(self,beta, S, I, count,i):
        lambd = min(S, pow(beta * S * (I), 0.97))/7
        self.lambd_s[i]+=lambd
        count+=1
        if count==7:
            I_t = self.lambd_s[i]
            S_t = max(S + self.births*7- I_t, 0)
            self.lambd_s[i]=0
        else:
            I_t= I
            S_t= S
        return I_t, S_t, lambd

    def new_infection(self, dIdt, dSdt, week):
        pinf = self.influenza_alpha * np.log(dIdt) + self.seasonal_fluc[week % 52]
        pinf = 1 / (1 + np.exp(-pinf))  # logit transform
        pinf = np.min([1, pinf + 0.001])
        dIdt = dSdt * pinf
        return dIdt

    def SEIRV(self, y, new_beta,reinfect_rate):
        [S, V, E, I, R, Q] = y

        S_ = max(S + self.births - new_beta * S * I - self.alpha * S - self.mu * S, 0)
        E_ = max(E + new_beta * S * I - self.gamma * E + self.theta * new_beta * V * I - self.mu * E, 0)
        I_ = max(I + self.gamma * E*(1-self.rate) - self.sigma * I - self.mu * I, 100)
        Q_ = max(Q + self.sigma * I +self.gamma * E*(self.rate)- (1 - self.k) * self.lambda_ * Q - self.k * self.rho * Q - self.mu * Q, 0)
        V_ = max(V + self.alpha * S - self.theta * new_beta * V * I - self.mu * V+ reinfect_rate*(1 - self.k) * self.lambda_ * Q, 0)
        R_ = max(R + (1 - self.k) * self.lambda_ * Q - self.mu * R, 0)

        new_I = self.gamma * E
        return [S_, V_, E_, I_, R_, Q_],new_I

    def featurize(self,state):
        state_l=[]
        # calculate the beds demanded by other RIDs
        if self.is_other_dis_considered==1:
            rid_beds = np.array(state['I_inc']) * np.array(self.Bed_usage) * np.array(self.const_adj)
            state_l.append(sum(rid_beds))
        else:
            pass
        # calculate the beds demanded by COVID-19
        covid_beds = state['covid_inf'] * 0.02
        state_l.append(covid_beds)
        state_l.append(state['last_act'])

        return state_l