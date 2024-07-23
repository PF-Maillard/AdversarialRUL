import random
from scipy.optimize import curve_fit
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


import Sources.Utils as UtilsTool

def Fgsm(model, Objective, X, y, epsilon, device):
    model.train()
    Adv = X.clone()
    Adv.requires_grad = True
    criterion = nn.MSELoss()
    
    output = model(Adv)
    Target = torch.full_like(y, Objective)
    loss = criterion(output, Target)
    loss.backward()
    Grad = Adv.grad.data
    SX = Grad.sign().to(device)
    Adv = Adv + epsilon * SX
    Adv = torch.clamp(Adv, min=0.0, max=1.0)
    return Adv

def Bim(model, Objective, X, y, epsilon, epochs, device):
    model.train()
    Adv = X.clone()
    criterion = nn.MSELoss()
    
    for i in range(epochs):
        Adv.requires_grad = True
        output = model(Adv)
        Target = torch.full_like(y, Objective)
        loss = criterion(output, Target)
        loss.backward()
        Grad = Adv.grad.data
        
        SX = Grad.sign().to(device)
        with torch.no_grad():
            Adv = Adv + epsilon * SX
            
    Adv = torch.clamp(Adv, min=0.0, max=1.0)
    return Adv

def CW(model, Objective, X, y, LearingRate, c, epochs, device):
    model.train()
    Adv = X.clone()
    Adv.requires_grad = True
    Optimizer = optim.Adam([Adv], lr=LearingRate)
    MseLoss = nn.MSELoss()
    
    for i in range(epochs):
        Optimizer.zero_grad()
        Output = model(Adv)
        Target = torch.full_like(y, Objective)
        TargetLoss = MseLoss(Output, Target)
        DiffLoss = MseLoss(Adv, X)
        loss = c * TargetLoss +  DiffLoss
        
        loss.backward()
        Optimizer.step()
        with torch.no_grad():
            Adv.clamp_(0.0, 1.0)
    return Adv

def TestAttacks(model, X, y, AttacksParameters, device):
    
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)

    AdversarialDataFgsm = Fgsm(model, AttacksParameters["FGSM"]["Objective"], X, y, AttacksParameters["FGSM"]["Epsilon"], device)
    AdversarialDataBim = Bim(model, AttacksParameters["BIM"]["Objective"], X, y, AttacksParameters["BIM"]["Epsilon"], AttacksParameters["BIM"]["Iterations"], device)
    AdversarialDataCW = CW(model, AttacksParameters["CW"]["Objective"], X, y, AttacksParameters["CW"]["LearningRate"], AttacksParameters["CW"]["c"], AttacksParameters["CW"]["Iterations"], device)
    
    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataFgsm, model)
    print("FGSM:", Infos)

    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataBim, model)
    print("BIM:", Infos)

    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataCW, model)
    print("CW:", Infos)
    
    return AdversarialDataFgsm, AdversarialDataBim, AdversarialDataCW


class ChoiceLoss(nn.Module):
    def __init__(self, device):
        self.device = device
        super(ChoiceLoss, self).__init__()

    def forward(self, X, Z):
        Focus = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        Focus2D = Focus.unsqueeze(0).repeat(len(X[0]), 1)
        Focus2D = Focus2D.to(self.device)
        
        L2 = ((X - Z) ** 2)
        L2*= Focus2D
        
        loss = torch.sum(L2)
        
        return loss
 
class L0Loss(nn.Module):
    
    def tanh(self, x, c):
        return torch.tanh(x*c)
    
    def __init__(self, k, device):
        self.k=k
        self.device = device
        super(L0Loss, self).__init__()

    def forward(self, X, Z):
        Focus = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        Focus2D = Focus.unsqueeze(0).repeat(len(X[0]), 1)
        Focus2D = Focus2D.to(self.device)
        
        L2 = ((X - Z) ** 2)
        L2*= Focus2D
        Result = self.tanh(L2, self.k)
        Result2 = torch.sum(Result, dim=1)
        Result3 = self.tanh(Result2, self.k)

        loss = torch.sum(Result3)
        return loss

def MaillardL2(model, Objective, X, y, LearningRate, c, epochs, device):
    model = model.to(device)
    X = X.to(device)
    
    model.train()
    Adv = X.clone()
    Adv.requires_grad = True
    Optimizer = optim.Adam([Adv], lr=LearningRate)
    MyLoss = ChoiceLoss(device)
    MseLoss = nn.MSELoss()
    
    Adv = Adv.to(device)
    for i in range(epochs):
        Optimizer.zero_grad()
        Output = model(Adv)
        Target = torch.full_like(y, Objective)
        
        TargetLoss = MseLoss(Output, Target)
        DiffLoss = MyLoss(Adv, X)
        
        loss = c * TargetLoss +  DiffLoss

        loss.backward()
        Optimizer.step()
        with torch.no_grad():
            Adv.clamp_(0.0, 1.0)
            
    return Adv

def Project(Adv, X, S, Loss):
    L2 = ((Adv - X) ** 2)
    Result = Loss.tanh(L2, Loss.k)
    Result2 = torch.sum(Result, dim=1)
    Result3 = Loss.tanh(Result2, Loss.k)
    
    ToProject = []
    for i in range(len(Result3[0])):
        if Result3[0][i] < S:
            ToProject.append(i)
    
    for Ins in range(len(Adv[0])):
        for i in ToProject:
            Adv[0][Ins][i] = X[0][Ins][i]
    return Adv

def IterativeMaillardL0(model, Objective, X, y, LearningRate, Init, Min, Steps, epochs, k, P, S, device):
    c = Init 
    LInfos = []
    while c > Min:
        Adversarials = MaillardL0(model, Objective, X, y, LearningRate, c, epochs, k, P, S, device)
        Infos = UtilsTool.GetInfos(X, y, 0, Adversarials, model)
        Infos["c"] = c
        LInfos.append(Infos)
        print(c, Infos)
        c/=Steps
    return LInfos


def IterativeMaillardL2(model, Objective, X, y, LearningRate, Init, Min, Steps, epochs, device):
    c = Init
    LInfos = []
    while c > Min:
        Adversarials = MaillardL2(model, Objective, X, y, LearningRate, c, epochs, device)
        Infos = UtilsTool.GetInfos(X, y, 0, Adversarials, model)
        Infos["c"] = c
        LInfos.append(Infos)
        c/=Steps
    return LInfos

def MaillardL0(model, Objective, X, y, LearningRate, c, epochs, k, P, S, device):
    model = model.to(device)
    model.train()
    
    LAdversarial =[]
    for Instance in range(len(X)):
        Adv = X[Instance:Instance+1].clone()
        Adv.requires_grad = True
        Optimizer = optim.Adam([Adv], lr=LearningRate)
        MyLoss = L0Loss(k, device)
        MseLoss = nn.MSELoss()
        
        MinLoss = 100000000
        Target = torch.full_like(y[Instance:Instance+1], Objective)
        SavedAdv = Adv.clone()
        
        for i in range(epochs):
            token = 0
            
            with torch.no_grad():
                Adv.clamp_(0.0, 1.0)
                if random.randint(0, P) == 0:
                    Adv = Project(Adv, X[Instance:Instance+1], S, MyLoss)    
                    token = 1   
            
            Output = model(Adv)
            
            TargetLoss = MseLoss(Output, Target)
            DiffLoss = MyLoss(Adv, X[Instance:Instance+1])
            
            loss = c * TargetLoss +  DiffLoss
            
            if token == 1 and loss.item() < MinLoss:
                MinLoss = loss.item()
                SavedAdv = Adv.clone()
                
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
        
        LAdversarial.append(SavedAdv) 
    FAdversarial = torch.cat(LAdversarial, dim=0)
    
    return FAdversarial


def OptimizeParameters(X, y, Model, OptimalParametersPath, TempoK, Name, device):
    
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
    
    Epochs = 1000
    LearningRate = 0.01
    Objective = 0 
    
    KList = {1, 2, 3, 4, 5, 8, 10, 12, 15}
    FixP = 5
    FixS = 0.1
    c = 1

    with open(OptimalParametersPath + "OptimizeK" + Name + ".txt", "w") as file:
        while c > 0.0001:
            for k in KList:
                CurrentAdversarial = MaillardL0(Model, Objective, X, y, LearningRate, c, Epochs, k, FixP, FixS, device)
                info = UtilsTool.GetInfos(X, y, 0, CurrentAdversarial, Model)
                result = f"{c}, {k}, {info}\n"
                file.write(result)
                file.flush()
                print(result)
            print()
            c/=2
    print("Created at: " + OptimalParametersPath + "OptimizeK" + Name + ".txt")
   
    FixK = TempoK
    PList = [1, 3, 5, 8]
    SList = [0.03, 0.06, 0.1, 0.3]
    c = 1
    with open(OptimalParametersPath + "OptimizePS" + Name + ".txt", "w") as file:
        while c > 0.0001:
            for P in PList:
                for S in SList:
                    CurrentAdversarial = MaillardL0(Model, Objective, X, y, LearningRate, c, Epochs, FixK, P, S, device)
                    info = UtilsTool.GetInfos(X, y, 0, CurrentAdversarial, Model)
                    result = f"{c}, {P}, {S}, {info}\n"
                    file.write(result)
                    file.flush()
                    print(result)
            print()
            c/=2
    print("Created at: " + OptimalParametersPath + "OptimizePS" + Name + ".txt")

def GetL0(X, Z):
    L2 = ((X - Z) ** 2)
    Result = torch.sum(L2, dim=1)
    Result = torch.sum(Result != 0, dim=1)
    Result = Result.float().mean()
    return Result

def TestNewAttacks(model, X, y, AttacksParameters, device, Path, Name):
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
    LInfos = IterativeMaillardL2(model, AttacksParameters["Maillard"]["Objective"], X, y, AttacksParameters["Maillard"]["LearningRate"], AttacksParameters["Maillard"]["Initial_c"], AttacksParameters["Maillard"]["Final_c"], AttacksParameters["Maillard"]["Steps_c"], AttacksParameters["Maillard"]["Iterations"], device)
    A = [item['c'] for item in LInfos]
    B = [item['PredRUL'] for item in LInfos]
    C = [item['RMSE_adversarial'] for item in LInfos]
    UtilsTool.DisplayCgraph(A, B, C, 'L2 Attack', 'RUL', 'RMSE', Path, Name + "L2")
    
    LInfos2 = IterativeMaillardL0(model, AttacksParameters["Maillard"]["Objective"], X, y, AttacksParameters["Maillard"]["LearningRate"], AttacksParameters["Maillard"]["Initial_c"], AttacksParameters["Maillard"]["Final_c"], AttacksParameters["Maillard"]["Steps_c"], AttacksParameters["Maillard"]["Iterations"], AttacksParameters["Maillard"]["k"], AttacksParameters["Maillard"]["P"], AttacksParameters["Maillard"]["S"], device)
    A = [item['c'] for item in LInfos2]
    B = [item['PredRUL'] for item in LInfos2]
    C = [item['L0'] for item in LInfos2]
    UtilsTool.DisplayCgraph(A, B, C, ' L0 Attack', 'RUL', 'L0', Path, Name + "L0")
    
def Torch_exp_degradation(x, D0, lambda_ ):
    return D0 * torch.exp(lambda_  * x)

def CreateAdversarialStats(X, y, Param, c, Epochs, LearningRate):
    Xt = torch.tensor(X, dtype=torch.float64)
    Adv = Xt.clone()
    Adv = Adv.requires_grad_(True)
    Optimizer = optim.Adam([Adv], lr=LearningRate)
    MseLoss = torch.nn.MSELoss()
    for i in range(Epochs):
        Optimizer.zero_grad() 
        A = Adv.view(Adv.size(0), -1)
        A = torch.mean(A, axis=1)
        y_pred = Torch_exp_degradation(A, Param[0], Param[1])
        LossPred = MseLoss(y_pred, torch.zeros_like(y_pred))
        LossClose = MseLoss(Adv, Xt)
        Loss = LossPred * c + LossClose
        Loss.backward()
        Optimizer.step()
    return Adv