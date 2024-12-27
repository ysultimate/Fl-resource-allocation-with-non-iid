import abc
import numpy as np
import math
from scipy.special import comb

class Bandit(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def requireArms(self):
        pass

    @abc.abstractclassmethod
    def updateWithRewards(self,rewards):
        pass

class Rexp3Bandit(Bandit):
    def __init__(self,args):
        self.num_clients = args.num_users
        self.batch_size = 250
        self.gamma = 0.2
        self.round_idx = 0
        self.clients = np.arange(self.num_clients,dtype='int')
        self.weights = np.ones((self.num_clients),dtype='int')



    def requireArms(self,num_picked):
        if self.round_idx >= self.batch_size:
            self.__init_batch()

        possi = self.__possibilities()
        draw = np.random.choice(self.clients,num_picked,p=possi,replace=False)
        return draw

    def updateWithRewards(self,rewards):
        possi = self.__possibilities(reweight=True)
        for client, reward in rewards.items():
            xhat = reward/possi[client]
            self.weights[client] = self.weights[client] * math.exp(self.gamma * xhat / self.num_clients)
        

    def __init_batch(self):
        self.round_idx = 0
        self.weights = np.ones((self.num_clients),dtype='float')
        return

    def __possibilities(self,reweight=False):
        if reweight == True:
            self.weights = self.weights * (self.num_clients / sum(self.weights))
        weights_sum = sum(self.weights)
        possi = np.zeros((self.num_clients),dtype='float')
        for i in range(self.num_clients):
            possi[i] = (1-self.gamma)*(self.weights[i]/weights_sum) + (self.gamma/self.num_clients)
        return possi


class MoveAvgBandit(Bandit):
    def __init__(self,args):
        self.num_clients = args.num_users
        self.avgs = {}
        for client in range(self.num_clients):
            self.avgs[client] = 0
        self.clients = np.arange(self.num_clients,dtype='int')
        self.uninitialized = [i for i in range(self.num_clients)]
        self.lastinit = 0 # 0: initializing, 1: the last round for initialization, 2: working

        # Parameters
        self.learningRate = 0.05
        self.epsilon = 0.05

    def requireArms(self,num_picked):
        # All required arms is uninitialized
        if len(self.uninitialized) >= num_picked:
            if len(self.uninitialized) == num_picked and self.lastinit == 0:
                self.lastinit = 1
            result = np.random.choice(self.uninitialized,num_picked,replace=False)
            for i in result:
                self.uninitialized.remove(i)
            return result

        if self.lastinit == 0:
            self.lastinit = 1

        # Part of arms is uninitialized
        if len(self.uninitialized) > 0:
            reserved = np.array(self.uninitialized,dtype='int')
            num_left = num_picked - len(self.uninitialized)
            self.uninitialized.clear()
            temp = self.clients.copy()
            for i in reserved:
                temp = np.delete(temp, np.argwhere(temp == i))
            newpicked = np.random.choice(temp,num_left,replace=False)
            result = np.concatenate([reserved,newpicked])
            return result

        # All arms initialized
        sortarms = sorted(self.avgs.items(),key=lambda x:x[1],reverse=True)
        results = []
        idx = 0
        for i in range(num_picked):
            draw = np.random.ranf()
            if draw < self.epsilon:
                while True:
                    client = np.random.choice(self.clients,1)
                    if client not in results:
                        results.append(client)
                        break
            else:
                while sortarms[idx][0] in results:
                    idx += 1
                results.append(sortarms[idx][0])
        return np.array(results,dtype='int')

    def updateWithRewards(self,rewards):
        for client,reward in rewards.items():
            if self.lastinit <= 1:
                self.avgs[client] = reward
            else:
                self.avgs[client] = (1-self.learningRate) * self.avgs[client] + self.learningRate * reward

        if self.lastinit == 1:
            self.lastinit = 2
        return

# fedacs
class SelfSparringBandit(Bandit):
    def __init__(self,args):
        if args.sampling == 'my_fourclass_client':
            self.num_clients = 48
        else:
            self.num_clients = args.num_users
        # s,f用来记录各个client被选择和不被选择的“次数”(不一定是实际次数)，是beta分布的超参数
        self.s = [0] * self.num_clients
        self.f = [0] * self.num_clients
        self.extension = args.extension
        # historical_rounds means -- How many rounds are remenbered by bandit for historical comparison
        self.historical_rounds = args.historical_rounds
        self.history = []
        if self.historical_rounds > 0:
            # 衰减？ 类似于前期多探索后期多利用？
            self.lr = float(1/self.historical_rounds)
        else:
            self.lr = 1

    # num_picked是fl在本轮需要选择的client数， 这个方法是用来选择client的
    # 目的学习最优的各个client的beta分布，弄清楚多臂老虎机是怎样探索和利用的
    # 检测选完后的各个client的多样性 or 补充在main_fed文件中
    def requireArms(self, num_picked):
        # extension 是为了扩充被选择范围，对应于论文中的lambda, 用来框定联邦中的总数据量，比如extension=8，那么框定的candidate就是80，即总数100的80%
        num_candidate = int(num_picked * self.extension)
        # 测试 四类用户
        # num_candidate = 20
        candidates = [i for i in range(self.num_clients)]
        picked = []
        for j in range(num_candidate):
            record = {}
            for candidate in candidates:
                # 从每个candidate的beta分布中进行采样
                record[candidate] = np.random.beta(self.s[candidate]+1,self.f[candidate]+1)
            # 根据采样结果将 record分从大到小排，sorted返回值是一个元组：[0][0]即为元组中第一个元素的的第一个值，即其id（id，record）
            sortedRecord = sorted(record.items(),key=lambda x:x[1],reverse=True)
            # winner应该是client的id
            winner = sortedRecord[0][0]
            picked.append(winner)
            candidates.remove(winner)
        # 从待选空间中的用户随机选择，extension=4 的时候性能最好
        return np.random.choice(picked,num_picked,replace=False)

    # rewards是字典，里面存放了“本轮”参与fl的client的Ri  {client_id ：Ri}
    # 用这一轮的rewards值来更新beta分布
    def updateWithRewards(self, rewards):
        # x 要取Ri，即我们设计的client的该轮评分, x在这里是 [(id,reward_i),()]列表中存放的是元组对
        x = list(rewards.items())
        y = list(rewards.items())

        # x is current round rewards, y includes historical records (no duplicate clients, use set to delete the same client)

        usedclients = set(rewards.keys())

        # 这里的目的是将过去几轮的ri也纳入观察的范围，但是代码很奇怪，第一次的时候是空的直接跳过了
        for hist in self.history:
            # hist是指 某个epoch的reward的list=[（id：ri）,]， onerecord 是指 此个epoch的list中的某个此（id：ri）元素
            # onerecord[0] 即为某client的id
            for onerecord in hist:
                # 若 当前client不在usedclients中则添加进去；这里是history中的故这部分代码运行结束之后usedclients中存放的是近historical_rounds中所有参与过fl的用户
                # 这里只会将距离现在最老的 （id,ri）添加进y中，因为添加新client后 就进不去下面的if判断了
                # ⭐可以考虑改：利用最新的记录试试
                if onerecord[0] not in usedclients:
                    usedclients.add(onerecord[0])
                    y.append(onerecord)

        # history = [[(id,ri),()],[epoch2],[epoch3]...]  列表中的元素是各个epoch中的id和reward的值
        # 将本次的值记录进 history中
        self.history.append(x.copy())
        # historical_rounds 是用来框定保存近几轮 历史record值的超参数 h_r=5时，若len（history）=6 则会把最早的h[0]删除
        if len(self.history) > self.historical_rounds:
            del self.history[0]
        # 更新各个client的beta分布参数，需要该轮每个参与过fl的用户相互之间都对比一次，谁分高谁取得优势 其对应的s[i]扩大，beta分布更容易取得大值
        # 这里的 i j都是 用户的id索引， i是记录本轮的id ， j是包含本轮+以前参与的id
        for i, ir in x:
            for j, jr in y:
                if i == j:
                    continue
                if ir > jr:
                    self.s[i] += self.lr
                else:
                    self.f[i] += self.lr
        return

class OortBandit(Bandit):
    def __init__(self,args):
        self.num_clients = args.num_users

        self.clients = np.arange(self.num_clients,dtype='int')
        self.uninitialized = [i for i in range(self.num_clients)]
        self.lastinit = 0 # 0: initializing, 1: the last round for initialization, 2: working
        self.participation = [0] * self.num_clients
        self.available_clients = set([i for i in range(self.num_clients)])

        self.u = [0] * self.num_clients
        self.lastround = [1] * self.num_clients
        self.round = 0

        self.lamb = 0.2
        self.clientpoolsize = int(self.num_clients*self.lamb)
        self.maxparticipation = 100
    
    def requireArms(self,num_picked):
        self.round += 1

        if num_picked > self.num_clients:
            print('Too much clients picked')
            exit(0)

        # All required arms is uninitialized
        if len(self.uninitialized) >= num_picked:
            if len(self.uninitialized) == num_picked and self.lastinit == 0:
                self.lastinit = 1
            result = np.random.choice(self.uninitialized,num_picked,replace=False)
            for i in result:
                self.uninitialized.remove(i)
            return result

        if self.lastinit == 0:
            self.lastinit = 1

        # Part of arms is uninitialized
        if len(self.uninitialized) > 0:
            reserved = np.array(self.uninitialized,dtype='int')
            num_left = num_picked - len(self.uninitialized)
            self.uninitialized.clear()
            temp = self.clients.copy()
            for i in reserved:
                temp = np.delete(temp, np.argwhere(temp == i))
            newpicked = np.random.choice(temp,num_left,replace=False)
            result = np.concatenate([reserved,newpicked])
            return result

        # All arms initialized
        clientpoolsize = max(self.clientpoolsize,num_picked)
        util = self.__util()
        sortarms = sorted(util.items(),key=lambda x:x[1],reverse=True)
        clientpool = np.zeros(clientpoolsize,dtype='int')
        clientutil = np.zeros(clientpoolsize,dtype='float')
        for i in range(clientpoolsize):
            clientpool[i] = sortarms[i][0]
            clientutil[i] = sortarms[i][1]
        clientutil = clientutil / clientutil.sum()
        draw = np.random.choice(clientpool,num_picked,p=clientutil,replace=False)
        return draw

    def updateWithRewards(self,loss):
        for arm, reward in loss.items():
            self.lastround[arm] = self.round
            self.u[arm] = reward
            self.participation[arm] += 1
            if self.participation[arm] >= self.maxparticipation and arm in self.available_clients and len(self.available_clients) > 10:
                self.available_clients.remove(arm)

    def __util(self):
        util = {}
        for i in self.available_clients:
            util[i] = self.u[i] + math.sqrt(0.1 * math.log(self.round) / self.lastround[i])
        return util



def tryBandit(bandit,iter,verbose=True):
    record = []
    for r in range(iter):
        arms = bandit.requireArms(10)
        realuti = sum(arms)/len(arms)
        record.append(realuti)
        if verbose:
            print("Round %3d: reward %.1f" % (r,realuti))
        rewards = {}
        for i in arms:
            flowing = 20*np.random.ranf()- 10
            rewards[i] = (i+1) * (1-0.001*r) + flowing
            #rewards[i] = (i+1) * flowing
        bandit.updateWithRewards(rewards)
    if verbose:
        print("Average Utility: %1f" % (sum(record)/len(record)) )

    return float(sum(record)/len(record))


class argument():
    def __init__(self):
        self.extension = 1
        self.num_users = 200
        self.historical_rounds = 5

if __name__ == "__main__":
    args = argument()
    record1 = []
    for i in range(10):
        bandit1 = OortBandit(args)
        print("Try: %3d" % (i+1))
        record1.append(tryBandit(bandit1,500))

    print("ss : %.1f" % (sum(record1)/len(record1)) )
