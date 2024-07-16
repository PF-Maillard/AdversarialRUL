import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import sleep
import pandas as pd
import numpy as np

import Sources.Data as DataTool

def InitModel(model, device):
    model = model.to(device)
    ks = [key for key in model.state_dict().keys() if 'linear' in key and '.weight' in key]
    for k in ks:
        nn.init.kaiming_uniform_(model.state_dict()[k])
    bs = [key for key in model.state_dict().keys() if 'linear' in key and '.bias' in key]
    for b in bs:
        nn.init.constant_(model.state_dict()[b], 0)
        
def PredRmse(tensor1, tensor2):
    mse = torch.mean((tensor1 - tensor2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def validation(model, valloader, device):
    model.eval()
    Loss_MSE = nn.MSELoss()
    
    X, y = next(iter(valloader))
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
    
    with torch.no_grad():
        y_pred = model(X)
        val_loss = Loss_MSE(y_pred, y).item()
        
    return val_loss
    
def test(MyModel, testloader, device):
    MyModel.eval()
    
    X, y = next(iter(testloader))
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
    
    loss_L1 = nn.L1Loss()
    Loss_MSE = nn.MSELoss()
    with torch.no_grad():
        y_pred = MyModel(X)
        test_loss_RMSE = PredRmse(y_pred, y).item()
        test_loss_MSE = Loss_MSE(y_pred, y).item()
        test_loss_L1 = loss_L1(y_pred, y).item()
        
    return test_loss_MSE, test_loss_L1, test_loss_RMSE, y_pred, y

def TrainModel(model, trainloader, valloader,epochs, learning_rate, device):
    T = []
    V = []
    
    Criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for i in tqdm(range(epochs)):
        
        L = 0
        model.train()
        for batch, (X,y) in enumerate(trainloader):
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            
            y_pred = model(X)
            loss = Criterion(y_pred, y)
            L += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss = validation(model, valloader, device)
        model.train()
        
        T.append(L/len(trainloader))
        V.append(val_loss)
        
        if (i+1) % 1 == 0:
            sleep(0.5)
            print(f'epoch:{i+1}, avg_train_loss:{L/len(trainloader)}, val_loss:{val_loss}')
            
def DisplayGraph(y_pred, y):
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(np.arange(1,101), y_pred.detach().cpu().numpy(), label = 'predictions', c = 'salmon')
    ax.plot(np.arange(1,101), y.detach().cpu().numpy(), label = 'true values', c = 'lightseagreen')
    ax.set_xlabel('Instance', fontsize = 16)
    ax.set_ylabel('RUL', fontsize = 16)
    ax.grid(True)
    ax.legend()
    plt.show()
    
def GetL0(X, Z):
    L2 = ((X - Z) ** 2)
    Result = torch.sum(L2, dim=1)
    Result = torch.sum(Result != 0, dim=1)
    Result = Result.float().mean()
    return Result
    
def GetInfos(X, y, Objective, AX, model):
    X, AX, y, = X.to(torch.float32), AX.to(torch.float32), y.to(torch.float32)
    rmse_adversarials = PredRmse(X, AX).item()
    output = model(AX)
    average = torch.mean(output.float()).item()
    averagey = torch.mean(y.float()).item()
    rmse_pred = PredRmse(y.float(), output.float()).item()
    L0 = GetL0(X.float(), AX.float()).cpu()
    extra_info = {
        'RealRUL': averagey,
        'Objective': Objective,
        'PredRUL': average,
        'RMSE_adversarial': rmse_adversarials,
        'RMSE_pred': rmse_pred,
        'L0': L0
    }
    return extra_info

def SaveDFS(df, i, prefix, extra_info, Path):
    extra_info_df = pd.DataFrame([extra_info])
    result_df = pd.concat([df, extra_info_df], ignore_index=True, sort=False)
    result_df.to_csv(f'{Path}/{prefix}_{i}.csv', index=False)

def CreateDFSFiles(testloaderAttack, X, AdvX, y, model, minmax_dict, NormalOutputPath, AdversarialOutputPath, device):
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
    
    columns_to_update = ['os1', 'os2', 'os3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    CurrentDataset = testloaderAttack.dataset

    for i in range(len(X)):
        DF = CurrentDataset.__getdf__(i)
        NDF = DataTool.UnnormalizeDataset(DF, minmax_dict)
        
        DF[columns_to_update] = AdvX[i].cpu().detach().numpy()
        ADF = DataTool.UnnormalizeDataset(DF, minmax_dict)
    
        extra_info = GetInfos(X[i:i+1], y[i:i+1], "NA", X[i:i+1], model)    
        SaveDFS(NDF, i, 'NormalDF', extra_info, NormalOutputPath)
        
        extra_info = GetInfos(X[i:i+1], y[i:i+1], 0, AdvX[i:i+1], model)   
        SaveDFS(ADF, i, 'AdversarialDF', extra_info, AdversarialOutputPath)