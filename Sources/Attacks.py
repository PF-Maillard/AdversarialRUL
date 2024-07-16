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

def TestNewAttacks(model, X, y, AttacksParameters, device):
    
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)

    AdversarialDataCW = CW(model, AttacksParameters["CW"]["Objective"], X, y, AttacksParameters["CW"]["LearningRate"], AttacksParameters["CW"]["c"], AttacksParameters["CW"]["Iterations"], device)

    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataCW, model)
    print("CW:", Infos)
    
    return AdversarialDataCW