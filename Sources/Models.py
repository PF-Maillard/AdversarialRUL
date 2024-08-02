import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, n_features, hidden_units):
        super().__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.n_layers = 1
        self.rnn = nn.RNN(input_size=n_features, hidden_size=self.hidden_units, batch_first=True, num_layers=self.n_layers)
        self.linear1 = nn.Linear(in_features=self.hidden_units, out_features=12)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=12, out_features=12)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=12, out_features=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        
        _, hn = self.rnn(x, h0)
        out = self.linear1(hn[0])
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out).flatten()
        return out

class GRUModel(nn.Module):
    def __init__(self, n_features, hidden_units):
        super().__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.n_layers = 1
        self.gru = nn.GRU(input_size=n_features, hidden_size=self.hidden_units, batch_first=True, num_layers=self.n_layers)
        self.linear1 = nn.Linear(in_features=self.hidden_units, out_features=12)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=12, out_features=12)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=12, out_features=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        
        _, hn = self.gru(x, h0)
        out = self.linear1(hn[0])
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out).flatten()
        return out

class LSTMModel(nn.Module):
    
    def __init__(self, n_features, hidden_units):
        super().__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.n_layers = 1
        self.lstm = nn.LSTM(input_size = n_features, hidden_size = self.hidden_units, batch_first = True, num_layers = self.n_layers)
        self.linear1 = nn.Linear(in_features=self.hidden_units, out_features=12)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=12, out_features=12)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=12, out_features=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear1(hn[0])
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out).flatten()
        
        return out
    
class LinearDegradationModel(nn.Module):
    def __init__(self, n_sensors, window_size):
        super(LinearDegradationModel, self).__init__()
        self.a = nn.Parameter(torch.ones(window_size))
        self.b = nn.Parameter(torch.ones(n_sensors))

    def forward(self, x):
        device = x.device
        sum_y = torch.zeros(x.size(0)).to(device)
        for i in range(len(self.a)):
            A = x[:,i,:] @ self.b
            y = self.a[i] * A
            sum_y += y
        average_y = sum_y / len(self.a)
        average_y = torch.relu(average_y)
        return average_y

class LogDegradationModel(nn.Module):
    def __init__(self, n_sensors, window_size):
        super(LogDegradationModel, self).__init__()
        self.a = nn.Parameter(torch.ones(window_size))
        self.b = nn.Parameter(torch.ones(n_sensors))

    def forward(self, x):
        device = x.device
        sum_y = torch.zeros(x.size(0)).to(device)
        for i in range(len(self.a)):
            A = x[:,i,:] @ self.b
            A = A**2
            y = self.a[i] * torch.log(A + 1)
            sum_y += y
        average_y = sum_y / len(self.a)
        average_y = torch.relu(average_y)
        return average_y

class ExpDegradationModel(nn.Module):
    def __init__(self, n_sensors, window_size):
        super(ExpDegradationModel, self).__init__()
        self.a = nn.Parameter(torch.ones(window_size))
        self.b = nn.Parameter(torch.ones(n_sensors))

    def forward(self, x):
        device = x.device
        sum_y = torch.zeros(x.size(0)).to(device)
        for i in range(len(self.a)):
            A = x[:,i,:] @ self.b
            y = self.a[i] * torch.exp(A) 
            sum_y += y 
        average_y = sum_y / len(self.a)
        average_y = torch.relu(average_y)
        return average_y