import gym 
from gym import spaces 
import numpy as np
import math 
from typing import Optional

def parcelout(ptconnect):
    XP = np.zeros((1,ptconnect.shape[1]), dtype=int)
    for i in range(ptconnect.shape[1]):
        if np.any(ptconnect[:,i] != 0):
            XP[:,i] = 1
    return XP

def tankin(ptconnect):
    XT = np.zeros((ptconnect.shape[0],1), dtype=int)
    for i in range(ptconnect.shape[0]):
        if np.any(ptconnect[i,:] != 0):
            XT[i,:] = 1
    return XT

def tankout(tcduconnect):
    XD = np.zeros((tcduconnect.shape[0],1), dtype=int)
    for i in range(tcduconnect.shape[0]):
        if np.any(tcduconnect[i,:] != 0):
            XD[i,:] = 1
    return XD

def range_restrict(xmin,xmax,real):
    difference = np.zeros((real.shape[0],real.shape[1]))
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if real[i,j] < xmin[i,j]:
                difference[i,j] = xmin[i,j] - real[i,j]
            elif real[i,j] > xmax[i,j]:
                difference[i,j] = real[i,j] - xmax[i,j]
            else:
                difference[i,j] = 0
    return difference

def change(time,tcduconnect,old_tcduconnect):
    CO = np.zeros((1,tcduconnect.shape[1]), dtype=int)
    for i in range(tcduconnect.shape[1]):
        if time == 0 or np.array_equal(tcduconnect[:,i],old_tcduconnect[:,i]):
            CO[:,i] = 0
        else:
            CO[:,i] = 1
    return CO

def supply(time,tcdurate,old_tcdurate,tcduconnect,old_tcduconnect):
    new_tcdurate = np.zeros((tcdurate.shape[0],tcdurate.shape[1]))
    new_tcduconnect = np.zeros((tcduconnect.shape[0],tcduconnect.shape[1]))
    difference = np.zeros((tcdurate.shape[0],tcdurate.shape[1]))
    for i in range(tcduconnect.shape[1]):
        if time == 0:
            difference[:,i] = 0
            new_tcdurate[:,i] = tcdurate[:,i]
        elif np.array_equal(tcduconnect[:,i],old_tcduconnect[:,i]):
            for j in range(tcduconnect.shape[0]):
                if tcdurate[j,i] == old_tcdurate[j,i]:
                    difference[j,i] = 0
                else:
                    difference[j,i] = 1
            new_tcdurate[:,i] = old_tcdurate[:,i]
            new_tcduconnect[:,i] = old_tcduconnect[:,i]
        else:
            difference[:,i] = 0
            new_tcdurate[:,i] = tcdurate[:,i]
            new_tcduconnect[:,i] = tcduconnect[:,i]
    return [difference,new_tcdurate,new_tcduconnect]

def correctrate(rate,row_lim,col_lim,size):

    #print("111")
    for i in range(rate.shape[0]):
        noncon_count = np.count_nonzero(rate[i,:])
        lim = row_lim[i]
        if noncon_count > lim:
            row = np.where(rate[i,:] != 0)[0]
            max_index = row[np.argsort(rate[i,:][row])[-lim:]]
            rate[i,row[~np.isin(row,max_index)]] = 0
    for j in range(rate.shape[1]):
        col_sum = np.sum(rate[:,j])
        if col_sum != 0:
            noncon_count = np.count_nonzero(rate[:,j])
            lim = col_lim[j]
            if noncon_count > lim:             
                col = np.where(rate[:,j] != 0)[0]
                max_index = col[np.argsort(rate[col,j])[-lim:]]
                rate[col[~np.isin(col,max_index)],j] = 0#应该是在这里操作
                col_sum = np.sum(rate[:,j])
            scale = size[j] / col_sum
            rate[:,j] = rate[:,j] * scale

    return rate

def pttransfer(rate,row_lim,col_lim,size):
    new_rate = rate.copy()
    new_rate = correctrate(new_rate,row_lim,col_lim,size)
    new_connect = np.zeros((new_rate.shape[0],new_rate.shape[1]),dtype = int)
    new_connect = np.where(new_rate != 0,1,new_rate)
    return [new_connect,new_rate]

def  tcdutransfer(rate,row_lim,col_lim,size_low,size_up):
    new_rate = rate.copy()
    rate_sum = np.sum(new_rate,axis = 0)
    num = len(rate_sum)
    difference = np.zeros(num)
    for i in range(num):
        if rate_sum[i] > size_up[:,i]:
            difference[i] = rate_sum[i] - size_up[:,i]
            rate_sum[i] = size_up[:,i]
        elif rate_sum[i] < size_low[:,i]:
            difference[i] = rate_sum[i] - size_low[:,i]
            rate_sum[i] = size_low[:,i]
    new_rate = correctrate(new_rate,row_lim,col_lim,rate_sum)
    new_connect = np.zeros((new_rate.shape[0],new_rate.shape[1]),dtype = int)
    new_connect = np.where(new_rate != 0,1,new_rate)
    return [new_connect,new_rate]

class ECS_Env(gym.Env):
    
    def __init__(self,random_seed_ref=0):
        self.random_seed_ref = random_seed_ref
        self.action_space = spaces.Box(low=0,high=1,shape=(18,),dtype=np.float32) # FPT,FTU
        self.observation_space = spaces.Box(low=-100,high=100,shape=(79,),dtype=np.float32)
        self.state = None
        self.trajectory = []
        self.jetty = np.array([1])
        self.A = np.full((3),2)
        self.B = np.full((6),2)
        self.P = np.full((6),2)
        self.Q = np.full((2),3)
        self.ps = np.array([[80,100,60]])
        self.psf = np.array([[0,1,0],[0,0,1],[0,0,0],[1,0,0]])
        self.fptl = np.full((6,3),5)
        self.fptu = np.full((6,3),120)
        self.wl = np.full((6,1),10)
        self.wu = np.full((6,1),100)
        self.ftul = np.full((6,2),2.99)
        self.ftuu = np.hstack((np.full((6,1),21.60),np.full((6,1),17.60)))
        self.ful = np.array([[21.50,17.50]])
        self.fuu = np.array([[21.60,17.60]])
        self.pgl = np.array([[1.44,0.96]])
        self.pgu = np.array([[5.28,2.88]])
        self.pdl = np.array([[2.4,1.92]])
        self.pdu = np.array([[5.28,4.08]])
        self.prl = np.zeros((1,2))
        self.pru = np.array([[6.48,6.00]])
        self.price = np.array([[358,337,322,312]])
        self.coc = 300
        self.swc = np.array([[1000,1000,1000]])
        self.xcl = np.zeros((4,2))
        self.xcu = np.array([[0.8,0.8],[1,1],[0.8,0.8],[0.8,0.8]])
        self.xkl = np.array([[0.84,0.84],[0,0],[0,0]])
        self.xku = np.array([[0.89,0.89],[2.50,2.50],[1.50,2.00]])
        self.xk = np.array([[0.858,0.875,0.938,0.911],[1.65,2.78,1.91,0.12],[0.16,0.18,0.22,4.38]])
        self.yieldg = np.array([[0.1720,0.1710],[0.1550,0.1548],[0.1220,0.1221],[0.0370,0.0365]])
        self.yieldd = np.array([[0.2051,0.2407],[0.1938,0.1942],[0.1724,0.1726],[0.1071,0.1073]]) 
        self.yieldr = np.array([[0.2165,0.2200],[0.2965,0.2968],[0.3615,0.3611],[0.5310,0.5314]])
        self.WCO = np.array([[0,0,0,60],[0,20,0,0],[0,0,80,0],[0,80,0,0],[60,0,0,0],[30,0,0,0]])
        self.WO = np.sum(self.WCO,axis=1).reshape(-1,1)
        self.fO = self.WCO/self.WO    
        self.XTO = np.zeros((6,1))
        self.YO = np.zeros((6,2))
        self.FTUO = np.zeros((6,2))
        self.expenseO = np.zeros((1,1))     
        self.penalty = 5000
        self.i_day = None
        self.ep_len = 7
        self.coming = np.zeros((3,self.ep_len))
        self.info = {}

    def reset(self, *, seed: Optional[int] = None):
        self.i_day = 0 
        # We need the following line to seed self.np_random
        #'''
        self.coming = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0]])
        self.arrival_day = np.array([[0,1,4]])
        '''
        super().reset(seed=seed)
        self.coming = np.zeros((3,self.ep_len))
        remain = [np.random.randint(0,7)]
        self.coming[0,remain] = 1
        remain = np.where(self.coming[0] == 0)[0]
        self.coming[1,np.random.choice(remain)] = 1
        remain = np.where((self.coming[0] == 0) & (self.coming[1] == 0))[0]
        self.coming[2,np.random.choice(remain)] = 1
        print(self.coming)
        self.arrival_day = np.zeros((1, 3))
        for i in range(3):
            self.arrival_day[0, i] = np.where(self.coming[i] == 1)[0][0]
        print(self.arrival_day)
        '''
        self.state = np.hstack((np.reshape(self.XTO,(1,6)),np.reshape(self.WCO,(1,24)),np.reshape(self.fO,(1,24)),np.reshape(self.YO,(1,12)),np.reshape(self.FTUO,(1,12)),self.expenseO)).flatten()
        self.trajectory.append(self.state)
        return self.state

    def step(self,action):

        arrive_ship = self.coming[:,self.i_day].reshape(-1,1)
        parcel = np.dot(self.ps,arrive_ship)
        parcelf = np.dot(self.psf,arrive_ship)
        fptlow = np.dot(self.fptl,arrive_ship)
        fptup = np.dot(self.fptu,arrive_ship)
        init_FPT = np.reshape(action[0:6],(6,1)) * (parcel > 0) * fptup
        init_FTU = np.reshape(action[6:18],(6,2)) * self.ftuu
        init_FPT[init_FPT < fptlow] = 0
        init_FTU[init_FTU < self.ftul] = 0
        XT_old = np.reshape(self.state[0:6],(6,1))
        WC_old = np.reshape(self.state[6:30],(6,4))
        f_old = np.reshape(self.state[30:54],(6,4))
        Y_old = np.reshape(self.state[54:66],(6,2))
        FTU_old = np.reshape(self.state[66:78],(6,2))
        expense_old = self.state[78]
        [terX,terFPT] = pttransfer(init_FPT,self.B,self.A,parcel)
        terFPT[terFPT < fptlow] = 0
        [X,FPT] = pttransfer(terFPT,self.B,self.A,parcel)
        XT = tankin(X)
        [terY,terFTU] = tcdutransfer(init_FTU,self.P,self.Q,self.ful,self.fuu)
        terFTU[terFTU < self.ftul] = 0
        [terY,terFTU] = tcdutransfer(terFTU,self.P,self.Q,self.ful,self.fuu)        
        [supplychange,FTU,Y] = supply(self.i_day,terFTU,FTU_old,terY,Y_old)
        XP = parcelout(X)
        
        XD = tankout(Y)
        FCPT1 = FPT * parcelf[0,:]
        FCPT2 = FPT * parcelf[1,:]
        FCPT3 = FPT * parcelf[2,:]
        FCPT4 = FPT * parcelf[3,:]
        FCTU1 = FTU * f_old[:,0].reshape(-1,1)
        FCTU2 = FTU * f_old[:,1].reshape(-1,1)
        FCTU3 = FTU * f_old[:,2].reshape(-1,1)
        FCTU4 = FTU * f_old[:,3].reshape(-1,1)

        carry1 = np.sum(FCPT1,axis=1).reshape(-1,1)
        carry2 = np.sum(FCPT2,axis=1).reshape(-1,1)
        carry3 = np.sum(FCPT3,axis=1).reshape(-1,1)
        carry4 = np.sum(FCPT4,axis=1).reshape(-1,1)
        FCPT = np.hstack((carry1,carry2,carry3,carry4))
        delay_time = self.i_day - np.dot(self.arrival_day,arrive_ship)
        delay_price = np.dot(self.swc,arrive_ship)
        delay_penalty = np.sum(delay_time * delay_price) + (parcel > np.sum(init_FPT)) * self.penalty
        extra_penalty = max(np.sum(XP,axis = 1) - self.jetty,0) * self.penalty

        carry1 = np.sum(FCTU1,axis=1).reshape(-1,1)
        carry2 = np.sum(FCTU2,axis=1).reshape(-1,1)
        carry3 = np.sum(FCTU3,axis=1).reshape(-1,1)
        carry4 = np.sum(FCTU4,axis=1).reshape(-1,1)
        FCTU = np.hstack((carry1,carry2,carry3,carry4))
        WCT = WC_old + FCPT - FCTU
        WT = np.sum(WCT,axis=1).reshape(-1,1)
        f = WCT/WT
        tstorage = range_restrict(self.wl,self.wu,WT)
        tankstorage_penalty = np.sum(tstorage) * self.penalty#储罐油量不处于上下限内
        # initial_penalty_coefficient = 10
        # final_penalty_coefficient = 100
        # penalty_coefficient = min(final_penalty_coefficient, initial_penalty_coefficient * (1 + 2*self.i_day ))
        brine_penalty = (np.sum(XT+XD)>1+np.sum(XT_old + XD > 1)) * self.penalty

        FU = np.array([np.sum(FTU,axis=0)])
        carry1 = np.sum(FCTU1,axis=0)
        carry2 = np.sum(FCTU2,axis=0)
        carry3 = np.sum(FCTU3,axis=0)
        carry4 = np.sum(FCTU4,axis=0)
        FCTUS = np.vstack((carry1,carry2,carry3,carry4))
        FCTUSL = FU * self.xcl
        FCTUSU = FU * self.xcu
        FUk = np.dot(self.xk,FCTUS)
        FUkl = FU * self.xkl
        FUku = FU * self.xku
        pg = np.array([np.sum(FCTUS * self.yieldg,axis=0)])
        pd = np.array([np.sum(FCTUS * self.yieldd,axis=0)])
        pr = np.array([np.sum(FCTUS * self.yieldr,axis=0)])
        CO = change(self.i_day,Y,Y_old)
        cdurate = range_restrict(self.ful,self.fuu,np.array([np.sum(init_FTU,axis=0)]))
        compostion = range_restrict(FCTUSL,FCTUSU,FCTUS)
        oil_property = range_restrict(FUkl,FUku,FUk)
        productg = range_restrict(self.pgl,self.pgu,pg)
        productd = range_restrict(self.pdl,self.pdu,pd)
        productr = range_restrict(self.prl,self.pru,pr)
        cdurate_penalty = np.sum(cdurate) * self.penalty#处理量不处于上下限内
        composition_penalty = np.sum(compostion) * self.penalty#原油组成不处于上下限内
        property_penalty = np.sum(oil_property) * self.penalty#原油性质不处于上下限内
        product_penalty = (np.sum(productg) + np.sum(productd) + np.sum(productr))*self.penalty#产品量不处于上下限内
        change_penalty = np.sum(CO) * self.coc 
        source_penalty = np.sum(supplychange) * self.penalty / 2
        oil_cost = np.dot(self.price,np.sum(FCTUS,axis = 1))
        limit_penalty = extra_penalty + tankstorage_penalty + brine_penalty
        process_penalty = composition_penalty + property_penalty + product_penalty + source_penalty
        expense = delay_penalty + change_penalty + oil_cost
 
        reward =  - 0.3 * limit_penalty - 0.3 * process_penalty - 0.4 * expense
        total_expense = expense + expense_old

        self.state = np.hstack((np.reshape(XT,(1,6)),np.reshape(WCT,(1,24)),np.reshape(f,(1,24)),np.reshape(Y,(1,12)),np.reshape(FTU,(1,12)),total_expense)).flatten()
        self.trajectory.append(self.state)
        done = bool(self.i_day>=self.ep_len-1)
        reward = reward / 100000
        total_expense = expense + expense_old
        print(extra_penalty)
        #print(tankstorage_penalty)
        print(brine_penalty)
        #print("流量{}".format(cdurate_penalty))
        #print("*")
        #print(composition_penalty)
        #print(property_penalty)
        #print(product_penalty)
        #print("*")
        #print(delay_penalty)
        print(change_penalty)
        #print(expense)
        #print(reward)
        print("FPT")
        print(FPT)
        print("FTU")
        print(FTU)
        print("WT")
        print(WT)
        print("FCTUS")
        print(FCTUS)
        #print("init_FTU")
        #print(init_FTU)
        print(total_expense)
        print('----------------------------------------------------------')
        self.i_day += 1

        return self.state, reward, done, self.info 

    def seed(self, seed):
        np.random.seed(seed)

'''
a = ECS_Env()

a.__init__()
a.reset()

FPT = np.array([[0],[17.9],[0],[1],[0],[62.1]])/120
ll = np.hstack((np.full((6,1),21.60),np.full((6,1),17.60)))
FTU = np.array([[0,0],[0,0],[5.310,4.98],[13.19,9.52],[3.00,3.00],[0,0]])/ll
action1 = np.hstack((np.reshape(FPT,(1,6)),np.reshape(FTU,(1,12)))).flatten()
a.step(action1)

FPT = np.array([[40],[60],[0],[0],[2],[0]])/120
FTU = np.array([[0,0],[0,0],[5.310,4.98],[13.19,9.52],[3.00,3.00],[0,0]])/ll
action2 = np.hstack((np.reshape(FPT,(1,6)),np.reshape(FTU,(1,12)))).flatten()
a.step(action2)

FPT = np.array([[0],[0],[0],[0],[0],[2]])/120
FTU = np.array([[0,0],[0,0],[5.310,4.98],[13.19,9.52],[3.00,3.00],[0,0]])/ll
action3 = np.hstack((np.reshape(FPT,(1,6)),np.reshape(FTU,(1,12)))).flatten()
a.step(action3)

FPT = np.array([[0],[10],[0],[5],[0],[2]])/120
FTU = np.array([[0,0],[12.05,9.43],[6.450,3.33],[0,0],[3,0],[0,4.74]])/ll
action4 = np.hstack((np.reshape(FPT,(1,6)),np.reshape(FTU,(1,12)))).flatten()
a.step(action4)

FPT = np.array([[0],[1],[0],[60],[0],[2]])/120
FTU = np.array([[0,0],[12.05,9.43],[6.450,3.33],[0,0],[3,0],[0,4.74]])/ll
action5 = np.hstack((np.reshape(FPT,(1,6)),np.reshape(FTU,(1,12)))).flatten()
a.step(action5)

FPT = np.array([[0],[1],[0],[0],[0],[2]])/120
FTU = np.array([[0,0],[12.05,9.43],[6.450,3.33],[0,0],[3,0],[0,4.74]])/ll
action6 = np.hstack((np.reshape(FPT,(1,6)),np.reshape(FTU,(1,12)))).flatten()
a.step(action6)

FPT = np.array([[0],[1],[0],[0],[0],[2]])/120
FTU = np.array([[0,0],[12.05,9.43],[6.450,3.33],[0,0],[3,0],[0,4.74]])/ll
action7 = np.hstack((np.reshape(FPT,(1,6)),np.reshape(FTU,(1,12)))).flatten()
a.step(action7)
'''