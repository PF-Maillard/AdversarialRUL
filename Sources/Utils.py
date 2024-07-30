from tqdm import tqdm
import matplotlib.pyplot as plt
from time import sleep
import pandas as pd
import numpy as np
import math
from decimal import Decimal, ROUND_HALF_UP
from scipy.optimize import curve_fit
import ast
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.optimize import least_squares

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
            
def DisplayGraph(y_pred, y, Path, Name):
    Size = len(y_pred) + 1
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(np.arange(1,Size), y_pred.detach().cpu().numpy(), label = 'predictions', c = 'salmon')
    ax.plot(np.arange(1,Size), y.detach().cpu().numpy(), label = 'true values', c = 'lightseagreen')
    ax.set_xlabel('Instance', fontsize = 16)
    ax.set_ylabel('RUL', fontsize = 16)
    ax.grid(True)
    ax.legend()
    plt.savefig(Path + Name, dpi=300)
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
        

def round_column(series, decimal_places):
    return series.apply(lambda x: float(Decimal(str(x)).quantize(Decimal('1e-{0}'.format(decimal_places)), rounding=ROUND_HALF_UP)))

def ApplyDiscrete(DF, minmax_dict):
    for c in DF.columns:
        if c + 'unit' in minmax_dict:
            Dec = minmax_dict[c + 'unit']
            if math.isnan(Dec):
                Dec = 0
            DF[c] = round_column(DF[c], Dec)       
    return DF 

def TestProjectionCost(testloaderAttack, X, Adversarial, MyModel, minmax_dict, window, device):
    columns_to_update = ['os1', 'os2', 'os3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    ListADF = []

    CurrentDataset = testloaderAttack.dataset
 
    for i in range(len(Adversarial)):
        DF = CurrentDataset.__getdf__(i)
        DF[columns_to_update] = Adversarial[i].cpu().detach().numpy()
        ADF = DataTool.UnnormalizeDataset(DF, minmax_dict)
        PDF = ApplyDiscrete(ADF, minmax_dict)
        NPDF = DataTool.NormalizeDataset(PDF, minmax_dict)
        ListADF.append(NPDF)

    FADF = pd.concat(ListADF, ignore_index=True)
    FADF = FADF.drop('index', axis=1)
    
    testdataset2 = DataTool.test(FADF, window)
    testloader2 = DataLoader(testdataset2, batch_size = 100)
    
    for Adv, y in testloader2:
        X, Adv, y = X.to(device).to(torch.float32), Adv.to(device).to(torch.float32), y.to(device).to(torch.float32)
        Infos = GetInfos(X, y, 0, Adversarial, MyModel)
        print("Initial:", Infos)
        Infos = GetInfos(X, y, 0, Adv, MyModel)
        print("Formated:", Infos)
    print()
    
    return Adv
    
def DisplayCgraph(A, B, C, MainLabel, label1, label2, Path, Name):
    fig, ax1 = plt.subplots()
    ax1.set_xscale('log')

    ax1.plot(A, B, 'b-', label=label1)
    ax1.set_xlabel('c')
    
    ax1.set_ylabel(label1, color='b')
    ax1.tick_params(axis='y', labelcolor='b')


    ax2 = ax1.twinx()
    ax2.plot(A, C, 'r-', label=label2)
    
    ax2.set_ylabel(label2, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(MainLabel)
    fig.tight_layout()

    plt.savefig(Path + Name, dpi=300)
    plt.show()
    
    
def exp_degradation(x, D0, lambda_):
    return D0 * np.exp(lambda_ * x)

def prepare_batch(X_batch, y_batch):
    X_batch_np = X_batch.view(X_batch.size(0), -1).numpy()
    y_batch_np = y_batch.numpy()
    
    X_batch_np = np.mean(X_batch_np, axis=1)
    
    return X_batch_np, y_batch_np

def prepare_data(loader):
    X_list = []
    y_list = []
    for X_batch, y_batch in loader:
        X_list.append(X_batch.view(X_batch.size(0), -1).numpy())
        y_list.append(y_batch.numpy())
    X_np = np.vstack(X_list)
    y_np = np.hstack(y_list)
    
    X_np = np.mean(X_np, axis=1)
    
    return X_np, y_np

def TrainStatModel(trainloader):
    X_train_np, y_train_np = prepare_data(trainloader)
    Param, pcov = curve_fit(exp_degradation, X_train_np, y_train_np, maxfev=10000)
    return Param

def GetStatsInfos(X, y, Objective, AX, Parameters):
    X, y, AX = X.cpu(), y.cpu(), AX.cpu()
    rmse_adversarials = PredRmse(X, AX).item()
    X_adv_np, y_adv_np = prepare_batch(AX.detach(), y)
    output = exp_degradation(X_adv_np, *Parameters)
    average = torch.mean(torch.tensor(output).float()).item()
    averagey = torch.mean(y.float()).item()
    rmse_pred = PredRmse(y.float(), torch.tensor(output).float()).item()
    extra_info = {
        'RealRUL': averagey,
        'Objective': Objective,
        'PredRUL': average,
        'RMSE_adversarial': rmse_adversarials,
        'RMSE_pred': rmse_pred
    }
    return extra_info

def AnalyzeK(Intersection, Path, PathSaving):
    with open(Path, 'r') as file:
        Data = defaultdict(list)
        
        for line in file:
            parts = line.split(", ", 2)
            c = float(parts[0])
            k = float(parts[1])
            MyDict = parts[2].replace("tensor", "")
            MyDict = ast.literal_eval(MyDict)
            Data[k].append({'c': c, 'k': k, 'PredRUL': MyDict["PredRUL"], 'L0': MyDict["L0"]})
        
    for k_value, rows in Data.items():
        c_values = [row['c'] for row in rows]
        RUL_values = [row['PredRUL'] for row in rows]
        L0_values = [row['L0'] for row in rows]
        
        filtered_c_values = [c for c, L0 in zip(c_values, L0_values) if L0 != 0]
        filtered_L0_values = [L0 for L0 in L0_values if L0 != 0]
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        ax1.set_xlabel('c')
        ax1.set_ylabel('RUL', color='tab:blue')
        ax1.plot(c_values, RUL_values, marker='x', color='tab:blue', label='RUL')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xscale('log')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('L0', color='tab:red')
        ax2.plot(filtered_c_values, filtered_L0_values, marker='x', color='tab:red', linestyle='--', label='L0')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_xscale('log')
        ax2.set_ylim(0, 24)
        
        ax1.axhline(y=Intersection, color='black')
        
        MinC = None
        L0_min = 24
        for c, RUL, L0 in zip(c_values, RUL_values, L0_values):
            if RUL <= Intersection and L0 < L0_min:
                L0_min = L0
                MinC = c

        if MinC is not None:
            ax1.axvline(x=MinC, color='green', linestyle='--')
            
            L0_at_c = None
            for c, L0 in zip(filtered_c_values, filtered_L0_values):
                if c == MinC:
                    L0_at_c = L0
                    break
            
            if L0_at_c is not None:
                ax2.annotate(f'L0={L0_at_c:.2f}', xy=(MinC, L0_at_c), xytext=(MinC, L0_at_c+2),
                            arrowprops=dict(facecolor='black', shrink=0.05))        

        plt.title(f'k = {k_value}')
        fig.tight_layout() 
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.savefig(f'{PathSaving}Optimization_{k_value}.png', dpi=300)
        plt.show()
        
        
def AnalyzePS(Intersection, Path, PathSaving):
    with open(Path, 'r') as file:
        Data = defaultdict(list)
        
        for line in file:
            parts = line.split(", ", 3)
            c = float(parts[0])
            P = float(parts[1])
            S = float(parts[2])
            MyDict = parts[3].replace("tensor", "")
            MyDict = ast.literal_eval(MyDict)
            Data[str(P) + " " + str(S)].append({'c': c, 'PS': str(P) + ", " + str(S), 'PredRUL': MyDict["PredRUL"], 'L0': MyDict["L0"]})
        
    for PS, rows in Data.items():
        c_values = [row['c'] for row in rows]
        RUL_values = [row['PredRUL'] for row in rows]
        L0_values = [row['L0'] for row in rows]
        
        filtered_c_values = [c for c, L0 in zip(c_values, L0_values) if L0 != 0]
        filtered_L0_values = [L0 for L0 in L0_values if L0 != 0]
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        ax1.set_xlabel('c')
        ax1.set_ylabel('RUL', color='tab:blue')
        ax1.plot(c_values, RUL_values, marker='x', color='tab:blue', label='RUL')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xscale('log')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('L0', color='tab:red')
        ax2.plot(filtered_c_values, filtered_L0_values, marker='x', color='tab:red', linestyle='--', label='L0')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_xscale('log')
        ax2.set_ylim(0, 24)
        
        ax1.axhline(y=Intersection, color='black')
        
        MinC = None
        L0_min = 24
        for c, RUL, L0 in zip(c_values, RUL_values, L0_values):
            if RUL <= Intersection and L0 < L0_min:
                L0_min = L0
                MinC = c

        if MinC is not None:
            ax1.axvline(x=MinC, color='green', linestyle='--')
            
            L0_at_c = None
            for c, L0 in zip(filtered_c_values, filtered_L0_values):
                if c == MinC:
                    L0_at_c = L0
                    break
            
            if L0_at_c is not None:
                ax2.annotate(f'L0={L0_at_c:.2f}', xy=(MinC, L0_at_c), xytext=(MinC, L0_at_c+2),
                            arrowprops=dict(facecolor='black', shrink=0.05))        

        plt.title(f'P, S = {PS}')
        fig.tight_layout() 
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.savefig(f'{PathSaving}Optimization{PS}.png', dpi=300)
        plt.show()
        
def model_predict(X, a, *b, CurrentModel):
    with torch.no_grad():
        CurrentModel.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        CurrentModel.b = nn.Parameter(torch.tensor(b, dtype=torch.float64))
        ypred = CurrentModel(torch.tensor(X).double()).numpy()
    return ypred

def residuals(params, X, y, CurrentModel):
    a = params[0]
    b = params[1:]
    y_pred = model_predict(X, a, *b, CurrentModel=CurrentModel)
    return (y - y_pred)

def TrainStatsModelLM(MyModel, trainloader, device):

    FullLoader = DataLoader(trainloader.dataset, batch_size=len(trainloader.dataset), shuffle=False)
    X, y = next(iter(FullLoader))
    X, y = X.to(device).float(), y.to(device).float()

    initial_params = [MyModel.a.item()] + MyModel.b.cpu().detach().numpy().tolist()
    initial_params = np.hstack(initial_params)

    result = least_squares(residuals, initial_params, args=(X.cpu().detach().numpy(), y.cpu().detach().numpy(), MyModel), method='lm')

    optimized_a = result.x[0]
    optimized_b = result.x[1:]

    with torch.no_grad():
        MyModel.a = nn.Parameter(torch.tensor(optimized_a, dtype=torch.float))
        MyModel.b = nn.Parameter(torch.tensor(optimized_b, dtype=torch.float))