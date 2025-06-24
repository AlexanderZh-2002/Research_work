import pandas as pd
import numpy as np
import torch
from torchmetrics import Accuracy, F1Score, Precision
#from torchmetrics.classification import BinaryPrecision
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def check_grads(model):
    for layer in model.layers:
        if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
            return layer.weight.grad
        

def proj(x):
    n = weight_constraints
    if abs(x) > n:
        return (x/abs(x))*n
    return x

def active(x):# 0 stands for active
    n = weight_constraints
    if abs(x) == n:
        return 0
    return 1

def count_active(act):
    res = 0
    for a in act:
        res += torch.sum(torch.where(a == 0 , 1, 0)).item()
    return res

def is_equal(A):#not prooved
    for i in range(0, len(A) - 1):
        if A[i] != 0:
            for j in range(0, len(A[0])):
                if not torch.equal(A[i][j], A[i+1][j]):
                    return False
    return True

def get_U(model, error_estimator):
    alpha = 0.1
    beta = 1.9
    U = False
    for layer in model.layers:
        if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
            x = torch.where(layer.weight.grad.clone().detach().apply_(abs) >= error_estimator**alpha, 1.0, 0.0)
            y = torch.where(layer.weight.clone().detach().apply_(abs) >= error_estimator**beta, 1.0, 0.0)
            if (x*y).nonzero().nelement() != 0:
                U = True
                #print("UUUUUUUUUUUUUUUUUUUUUUUUUUUU")
                break
    return U
    
def get_active_grad(model): #count active gradient
    ac_grad = []
    
    for layer in model.layers:
        if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
            cur_act = layer.weight.clone().detach().apply_(active)
            ac_grad.append(layer.weight.grad.clone()*cur_act)
            
    return ac_grad
    

def get_measures(model): #count error_estimator and A and norm_gl
        
    error_estimator = 0
    last_active = []
    norm_gl = 0
        
    for layer in model.layers:
        if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
            #print("Layer :", layer)
            error_estimator += (torch.linalg.vector_norm((layer.weight - layer.weight.grad).clone().detach().apply_(proj)-layer.weight)).item()**2
            #print("weight\n",layer.weight)
            cur_act = layer.weight.clone().detach().apply_(active)
            #print("cur\n", cur_act)
            #print("grad:\n", layer.weight.grad)
            #print("active g\n", layer.weight.grad*cur_act)
            norm_gl += torch.linalg.vector_norm(layer.weight.grad*cur_act)**2
            last_active.append(cur_act)
    error_estimator = error_estimator**0.5
    norm_gl = norm_gl**0.5
    #print("error_estimator", error_estimator)
    #print("last_active", last_active)
    #print("norm_gl", norm_gl)
    return error_estimator, norm_gl, last_active
        
    #active number

class aNGPA:
    def __init__(self, model, criterion, optimizer, min_step, max_step, Lmin, Lmax, Mmin, Mmax, cycle_legth, sigma, data, target):
        self.iteration = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.min_step = min_step
        self.max_step = max_step
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.Mmin = Mmin
        self.Mmax = Mmax
        
        self.cycle_length = cycle_legth
        self.sigma = sigma
        self.CBB_step = 1
        
        self.data = data
        self.target = target
        self.E_history = [0]*self.Mmax
        self.L_history = [0]*3
        self.prev_M = 0

        
        self.prev_grad = []
        self.prev_weight = []
    
    def make_step(self):
        
        
        
        #self.optimizer.zero_grad()
        for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                    layer.zero_grad()
        result = self.model(self.data)
        loss = self.criterion(result, self.target)
        initial_loss = loss.item()
        loss.backward()
        
        #save model state
        initial_model_state = self.model.state_dict()
        initial_optim_state = self.optimizer.state_dict()
        
        #testing
        #print("aNGPA iteration = ", self.iteration)
        #print("in_loss", initial_loss)
        
        #update E-history
        for i in range(1, len(self.E_history)):
            self.E_history[i-1] = self.E_history[i]
        self.E_history[-1] = initial_loss
        
        if self.iteration == 0:
            
            step = self.CBB_step
            for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                    with torch.no_grad():
                        self.prev_weight.append(layer.weight.clone())
                        self.prev_grad.append(layer.weight.grad.clone())
                        layer.weight = torch.nn.Parameter((layer.weight - layer.weight.grad.clone()*self.CBB_step).apply_(proj))
            #print("prev grad", self.prev_grad)
            #print("prev weight", self.prev_weight)
            
            #self.optimizer.zero_grad()
            for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                    layer.zero_grad()
            result = self.model(self.data)
            loss = self.criterion(result, self.target)
            loss.backward()
            
            
    
        else:
            
            #get s, y norms
            ss = 0
            ys = 0
            yy = 0
            idx = 0
            for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                    
                    #print("layer ", layer)
                    
                    #print("weight\n", layer.weight)
                    #print("prev_weight\n", self.prev_weight[idx])
                    
                    #print("grad\n", layer.weight.grad)
                    
                    s = layer.weight.clone() - self.prev_weight[idx]
                    y = layer.weight.grad.clone() - self.prev_grad[idx]
                    ss += torch.sum(s*s).item()
                    ys += torch.sum(y*s).item()
                    yy += torch.sum(y*y).item()
                    
                    self.prev_weight[idx] = layer.weight.clone()
                    self.prev_grad[idx] = layer.weight.grad.clone()
                    idx += 1
            #print("ss ",ss, " yy " , yy, " ys ", ys)
                    
                    
            #count new CBB
            if self.iteration % self.cycle_length == 0:
                self.CBB_step = ss/ys
                
            #count step
            step = max(self.min_step, min(self.CBB_step, self.max_step))
            #print("step = ", step)
            
            #count L
            if ss == 0:
                L = self.Lmax
            else:
                L = max(self.Lmin, min((yy**0.5/ss**0.5), self.Lmax))
            self.L_history[0] = self.L_history[1]
            self.L_history[1] = self.L_history[2]
            self.L_history[2] = L
            
            #count M
            if self.iteration < 2:
                M = int((self.Mmax + self.Mmin)/2)
                self.prev_M = M
            else:
                if self.L_history[0] > self.L_history[1] and self.L_history[1] > self.L_history[2]:
                    M = max(self.Mmin, min(self.Mmax, self.prev_M + 1))
                elif self.L_history[0] < self.L_history[1] and self.L_history[1] < self.L_history[2]:
                    M = max(self.Mmin, min(self.Mmax, self.prev_M - 1))
                else:
                    M = self.prev_M
                    
            #count Emax
            Emax = max(self.E_history[len(self.E_history)-M:])
            
            
            #Armicho
            alpha = 0.5
            nu = 1
            
            while True:
                #print("in while")
                self.model.load_state_dict(initial_model_state)
                self.optimizer.load_state_dict(initial_optim_state)
                
                dg = 0
                #print("nu = ", nu)
                idx = 0
                for layer in self.model.layers:
                    if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                        #print(self.iteration)
                        d = (self.prev_weight[idx].clone() - self.prev_grad[idx].clone()*step).detach().apply_(proj) - self.prev_weight[idx].clone()
                        #print("d =\n",d) 
                        dg += torch.sum(d.clone()*self.prev_grad[idx].clone()).item()
                        
                        with torch.no_grad():
                            layer.weight = torch.nn.Parameter((layer.weight.clone() + nu*d).apply_(proj))
                        #print("layer_weight\n", layer.weight)
                        idx += 1
                #self.optimizer.zero_grad()
                for layer in self.model.layers:
                    if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                        layer.zero_grad()
                result = self.model(self.data)
                loss = self.criterion(result, self.target)
                probe_loss = loss.item()
                loss.backward()
                
                #print("pr loss", probe_loss)
                
                if probe_loss <= Emax + self.sigma*nu*dg:
                    #print("end of armicho", check_grads(self.model))
                    break
                nu *= alpha
        self.iteration+=1
        
        #print("after_angpa", check_grads(self.model))
        
            
        
        

class CG_descent:
    def __init__(self, model, criterion, optimizer, sigma1, sigma2, delta, c1, c2, data, target):
        self.iteration = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.delta = delta
        self.c1 = c1
        self.c2 = c2
        
        self.data = data
        self.target = target
        
        self.prev_grad = []
        self.prev_weight = []
        self.prev_direction = []
    
    def make_step(self):
        
        #self.optimizer.zero_grad()
        for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                    layer.zero_grad()
        
        result = self.model(self.data)
        loss = self.criterion(result, self.target)
        initial_loss = loss.item()
        loss.backward()
        
        #save model state
        initial_model_state = self.model.state_dict()
        initial_optim_state = self.optimizer.state_dict()
        
        #testing
        #print("CG_descent iteration = ", self.iteration)
        
        #get active grad
        active_grad = get_active_grad(self.model)
        d = [0]*len(active_grad)
        
        if self.iteration == 0:
            #save prev weights
            for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                    with torch.no_grad():
                        self.prev_weight.append(layer.weight.clone())
            #get direction
            for i in range(len(active_grad)):
                d[i] = active_grad[i].clone()*(-1)
            # save prev grad
            self.prev_grad = active_grad
            #save prev direction
            self.prev_direction = d
        else:
            yd = 0
            yy = 0
            dd = 0
            gg = 0
            arr_y = []
            
            #count scalar multiplications
            for idx in range(len(active_grad)):
                y = active_grad[idx].clone() - self.prev_grad[idx].clone()
                arr_y.append(y)
                yy += torch.sum(y*y).item()
                dd += torch.sum(self.prev_direction[idx].clone()*self.prev_direction[idx].clone()).item()
                yd += torch.sum(self.prev_direction[idx].clone()*y.clone()).item()
                gg += torch.sum(self.prev_grad[idx].clone()*self.prev_grad[idx].clone()).item()
                idx += 1
                
            #count beta
            beta = 0
            for idx in range(len(active_grad)):
                beta += torch.sum((arr_y[idx].clone() - self.prev_direction[idx].clone()*(yy/yd))*active_grad[idx].clone()).item()
            beta /= yd
            
            #print("yy", yy, "dd", dd, "yd", yd, "gg", gg)
            
            #count teta
            teta = (-1)/((dd**0.5)*min(self.delta, gg**0.5))
            
            #print("beta", beta, "teta", teta)
            
            #count final step
            fin_step = max(beta, teta)
            """
            if fin_step == beta:
                print("beta")
            else:
                print("teta")
            #print("fin_step", fin_step)
            """
            
            #count direction
            for idx in range(len(active_grad)):
                d[idx] = (-1)*(active_grad[idx].clone()) + fin_step*(self.prev_direction[idx].clone())
                
            #save prev weights
            idx = 0
            for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                    with torch.no_grad():
                        self.prev_weight[idx] = layer.weight.clone()
                    idx += 1
                        
            # save prev grad
            self.prev_grad = active_grad
            
            #save prev direction
            self.prev_direction = d
            
                    
            
        nu = 1
        a = 0.8
        b = 1.2
        
        """
        print("before wolfe")
        idx = 0
        for layer in self.model.layers:
            if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':                        
                if torch.equal(layer.weight, self.prev_weight[idx]):
                    print("EQUAL")
                else:
                    print("MISTAKE")
                idx += 1
        """
                
        #Wolf
        #print("start wolf")   
        while True:
            self.model.load_state_dict(initial_model_state)
            self.optimizer.load_state_dict(initial_optim_state)
            
            """
            idx = 0
            print("in wolfe")
            for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':                        
                    if torch.equal(layer.weight, self.prev_weight[idx]):
                        print("EQUAL")
                    else:
                        print("MISTAKE")
                    idx += 1
            """
            
            
        
            idx = 0
            
            #testing
            #print("nu =", nu)
            
            # make step
            for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':                        
                    with torch.no_grad():
                        layer.weight = torch.nn.Parameter((self.prev_weight[idx].clone() + nu*self.prev_direction[idx].clone()).apply_(proj))
                    idx += 1
            
            #self.optimizer.zero_grad()
            for layer in self.model.layers:
                if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                    layer.zero_grad()
            result = self.model(self.data)
            loss = self.criterion(result, self.target)
            probe_loss = loss.item()
            loss.backward()
            
            #get active grad
            active_grad = get_active_grad(self.model)
            
            #testing
            #print("probe_loss", probe_loss, "initial_loss", initial_loss)
                
            d_prev_g = 0
            dp = 0
            idx = 0
            for idx in range(len(active_grad)):
                d_prev_g += torch.sum(self.prev_grad[idx].clone()*self.prev_direction[idx].clone()).item()
                dp += torch.sum(active_grad[idx].clone()*self.prev_direction[idx].clone()).item()  
                idx += 1 
            
            #testing
            #print("d_pred_g", d_prev_g, "dp", dp)
            #print("c1", self.c1, "c2", self.c2)
            #print("initial loss", initial_loss, "probe loss", probe_loss)
            
            first_condition = (probe_loss <= (initial_loss + self.c1*d_prev_g))
            second_condition = (-dp <= (-1)*self.c2 * d_prev_g)
            """
            first_condition = (probe_loss <= (initial_loss + self.c1*nu*d_prev_g))
            second_condition = (-abs(dp) <= (-1)*self.c2 * abs(d_prev_g))
            """
            #print(first_condition, second_condition)
                
            if first_condition and second_condition:
                #print("break")
                break
            elif not first_condition:
                #print("not first")
                nu *= a
            elif not second_condition:
                #print("not second")
                nu *= b
            if nu == 0:
                raise Error
        self.iteration += 1
        #print("after wolfe")
        #print("after_cg", check_grads(self.model))

            
class AANN:
    def __init__(self, model, criterion, optimizer, n1, n2, mu, ro, data, target):
        self.iteration = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.mu = mu
        self.ro = ro
        
        self.n1 = n1
        self.n2 = n2
        
        self.data = data
        self.target = target
        
        self.A = [0] * (n1+1)
        self.Status = "aNGPA"
        
        self.angpa = aNGPA(model, criterion, optimizer, 10**(-5), 10**(5), 10**(-3), 10**(8), 3, 15, 4, 10**(-4), data, target)
        self.cgd = CG_descent(model, criterion, optimizer, 0.1, 0.9, 0.4, 0.001, 0.9, data, target)
    
    def restart_aNGPA(self):
        self.angpa = aNGPA(self.model, self.criterion, self.optimizer, 10**(-2), 10**(5), 10**(-3), 10**(8), 3, 15, 4, 10**(-4), self.data, self.target)
        
    def restart_CG_descent(self):
        self.cgd = CG_descent(self.model, self.criterion, self.optimizer, 0.1, 0.9, 0.4, 0.001, 0.9, self.data, self.target)
    
    def make_step(self):
        #print("status:", self.Status)
        if self.Status == "aNGPA":
            #print("start aNGPA")
            self.angpa.make_step()
            
            
            error_estimator, norm_gl, last_active = get_measures(self.model)
            U = get_U(self.model, error_estimator)
            a_n = count_active(last_active)
            
            for i in range(1, len(self.A)):
                self.A[i-1] = self.A[i]
            self.A[-1] = last_active
            
            
            if not U:
                if norm_gl < self.mu * error_estimator:
                    self.mu *= self.ro
                    #print("mu = ", self.mu)
                else:
                    print("change to CG-decent 1 iter", self.iteration)
                    self.Status = "CG-decent"
                    self.restart_CG_descent()
            elif(is_equal(self.A)):
                if norm_gl >= self.mu * error_estimator:
                    print("change to CG-decent 2 iter", self.iteration)
                    self.Status = "CG-decent"
                    self.restart_CG_descent()
        else:
            #print("start CGD")
            self.cgd.make_step()
            
            error_estimator, norm_gl, last_active = get_measures(self.model)
            U = get_U(self.model, error_estimator)
            a_n = count_active(last_active)
            a_n_prev = count_active(self.A[-1])
            
            for i in range(1, len(self.A)):
                self.A[i-1] = self.A[i]
            self.A[-1] = last_active
            
            if norm_gl < self.mu * error_estimator:
                #print("change to aNGPA 1")
                self.Status = "aNGPA"
                self.restart_aNGPA()
            elif(a_n > a_n_prev):
                if U and a_n >= a_n_prev + n2:
                    #print("change to aNGPA 2")
                    self.Status = "aNGPA"
                    self.restart_aNGPA()
                self.restart_CG_descent()
        self.iteration += 1

class Network_1_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(5, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1),
            torch.nn.Sigmoid()
        )
        
        
        print("do init")
        for layer in self.layers:
            if str(layer) != 'ReLU()' and str(layer) != 'Sigmoid()':
                layer.weight.data.normal_(mean=0.00, std=0.1)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        
        




    def forward(self, x):
        out = x
        out = self.layers(out)
        return out
        
class MyDataset(torch.utils.data.Dataset):
 
  def __init__(self,df, target):
 
    self.x_train=torch.tensor(df.iloc[:, :].values,dtype=torch.float32)
    self.y_train=torch.tensor(target.iloc[:, :].values,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

def prepare_dataset(path, dict_):
    unformated_df = pd.read_csv(path)

    df = unformated_df

    for key in dict_.keys():
        #print(key)
        if dict_[key][0] == 'categorial':
            #print("yes")
            y = pd.get_dummies(df[key], prefix=key)
            #print(y)
            df = df.drop(columns = [key])
            df = df.join(y, how = 'left')

    target = pd.DataFrame(df['default'])
    del df['default']
    Ds = MyDataset(df, target)
    return Ds

def train_evaluate(model):
    global train_loader
    model.eval()
    epoch_loss = []
    ac = Accuracy(task="binary", treshold = 0.5)
    f = F1Score(task="binary", treshold = 0.5)

    with torch.no_grad():
        for i, (data, target) in enumerate(train_loader):
            t = target.int()
            result = model(data)
            ac_score = ac(result, t)
            f_score = f(result, t)
    #print("te ac:", ac_score.item(), "F:", f_score.item())

def evaluate(model):
    global test_loader, scores
    model.eval()
    epoch_loss = []
    ac = Accuracy(task="binary", treshold = 0.5)
    f = F1Score(task="binary", treshold = 0.5)
    precision =  Precision(task="binary", treshold = 0.5)

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            t = target.int()
            result = model(data)
            ac_score = ac(result, t)
            f_score = f(result, t)
            pc_score = precision(result, t)
    #print("ev ac:", ac_score.item(), "F:", f_score.item())
    scores[0].append(ac_score.item())
    scores[1].append(f_score.item())
    scores[2].append(pc_score.item())

dict2 = {
    "norm":["normal",[0.5,0.25],("treshold_good",)],
    "weib  k = 1.5":["weibull",[0.5,1.5],("treshold_bad",)],
    "weib  k = 1":["weibull",[0.25,1],("treshold_good",)],
    "log":["lognormal",[0.5,0.25],("treshold_bad",)],
    "chi":["chi-square",[3, 0.1],("treshold_bad",)]
    }

train_ds = prepare_dataset("data/train2_75.csv", dict2)
train_loader=torch.utils.data.DataLoader(train_ds,batch_size=len(train_ds),shuffle=False)

test_ds = prepare_dataset("data/test2_75.csv", dict2)
test_loader=torch.utils.data.DataLoader(test_ds,batch_size=len(test_ds),shuffle=False)

data, target = 0, 0
for i, (inp, otp) in enumerate(train_loader):
    print(i)
    data = inp
    target = otp
print("data",data.shape)#,"\n",data,"\n")
print("target",target.shape)#,"\n",target,"\n")


net = Network_1_layer()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)

#save model state
initial_model_state = net.state_dict()
initial_optim_state = optimizer.state_dict()

iter_10 = []
iter_15 = []
iter_9 = []
iter_5 = []
name = "img.png"

mu = [0.1, 1, 3]
weight_constraints = 1

for k in range(3):
    scores = [[],[],[]]
    print("K = ",k,"mu = ", mu[k], " \n\n")

    loss_fn = torch.nn.BCELoss()
    net.load_state_dict(initial_model_state)
    optimizer.load_state_dict(initial_optim_state)
    
    print("first evaluate")
    #evaluate(net)
    
    #an = aNGPA(net, loss_fn, optimizer, 10**(-5), 10**(5), 10**(-3), 10**(8), 3, 15, 4, 10**(-4), data, target)
    #first_phase = CG_descent(net1, loss_fn, optimizer1, 0.1, 0.9, 0.4, 0.001, 0.9, data, target)
    #main_alg = AANN(net, loss_fn, optimizer, 2, 1, mu[k], 0.1, data, target)
    main_alg = aNGPA(net, loss_fn, optimizer, 10**(-5), 10**(5), 10**(-3), 10**(8), 3, 15, 4, 10**(-4), data, target)
    for i in range(50):
        main_alg.make_step()
        if i%1 == 0:
            evaluate(net)
    xp = np.arange(1,len(scores[0])+1)
    yp = np.array(scores[0])
    plt.plot(xp,yp)
    plt.title("accuracy")
    #plt.savefig('ac_ver3_'+name, bbox_inches = 'tight')
    plt.show()
            
    xp = np.arange(1,len(scores[1])+1)
    yp = np.array(scores[1])
    plt.plot(xp,yp)
    plt.title("f - score")
    #plt.savefig(str(k)+'f_'+name, bbox_inches = 'tight')
    plt.show()
    del main_alg
    
"""
    scores = [[],[],[]]
    print("first evaluate")
    evaluate(net)
    weight_constraints = 0.4 + k*0.05

    for i in range(20):
        main_alg.make_step()
        #an.make_step()
        if i%1 == 0:
            evaluate(net)
        if i == 5:
            iter_5.append(scores[2][-1])
        if i == 9:
            iter_9.append(scores[2][-1])
        if i == 10:
            iter_10.append(scores[2][-1])
        if i == 15:
            iter_15.append(scores[2][-1])
    
    
    if k == 0:
        xp = np.arange(1,len(scores[0])+1)
        yp = np.array(scores[0])
        plt.plot(xp,yp)
        plt.title("accuracy")
        #plt.savefig('ac_ver3_'+name, bbox_inches = 'tight')
        plt.show()
            
        xp = np.arange(1,len(scores[1])+1)
        yp = np.array(scores[1])
        plt.plot(xp,yp)
        plt.title("f - score")
        #plt.savefig(str(k)+'f_'+name, bbox_inches = 'tight')
        plt.show()
        xp=[]

    xp.append(weight_constraints)
    del main_alg

print("iter 5\n", iter_5)
yp = np.array(iter_5)
plt.plot(xp,yp)
plt.title("precision")
plt.savefig("iter_5_ver3_"+'pc_'+name, bbox_inches = 'tight')
plt.show()
print("iter 9\n", iter_9)
yp = np.array(iter_9)
plt.plot(xp,yp)
plt.title("precision")
plt.savefig("iter_9_ver3_"+'pc_'+name, bbox_inches = 'tight')
plt.show()
print("iter 10\n", iter_10)
yp = np.array(iter_10)
plt.plot(xp,yp)
plt.title("precision")
plt.savefig("iter_10_ver3_"+'pc_'+name, bbox_inches = 'tight')
plt.show()
print("iter 15\n", iter_15)
yp = np.array(iter_15)
plt.plot(xp,yp)
plt.title("precision")
plt.savefig("iter_15_ver3_"+'pc_'+name, bbox_inches = 'tight')
plt.show()
"""
