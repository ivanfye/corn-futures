import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


cn_corn = pd.read_csv('data/cn_corn.csv')
us_corn = pd.read_csv('data/us_corn.csv')
us_hardwheat = pd.read_csv('data/us_hardwheat.csv')
br_corn = pd.read_csv('data/br_corn.csv')
us_oat = pd.read_csv('data/us_oat.csv')
us_rice = pd.read_csv('data/us_rice.csv')
us_softwheat = pd.read_csv('data/us_softwheat.csv')

il_temp = pd.read_csv('data/il_temp.csv')
io_temp = pd.read_csv('data/io_temp.csv')
cn_temp = pd.read_csv('data/cn_temp.csv')
br_temp = pd.read_csv('data/br_temp.csv')

cn_humidity = pd.read_csv('data/cn_humidity.csv')
il_humidity = pd.read_csv('data/il_humidity.csv')
io_humidity = pd.read_csv('data/io_humidity.csv')
br_humidity = pd.read_csv('data/br_humidity.csv')

cn_corn['Date'] = pd.to_datetime(cn_corn['Date'])
us_corn['Date'] = pd.to_datetime(us_corn['Date'])
us_hardwheat['Date'] = pd.to_datetime(us_hardwheat['Date'])
br_corn['Date'] = pd.to_datetime(br_corn['Date'])
us_oat['Date'] = pd.to_datetime(us_oat['Date'])
us_rice['Date'] = pd.to_datetime(us_rice['Date'])
us_softwheat['Date'] = pd.to_datetime(us_softwheat['Date'])
io_temp['Date'] = pd.to_datetime(io_temp['Date'])
il_temp['Date'] = pd.to_datetime(il_temp['Date'])
cn_temp['Date'] = pd.to_datetime(cn_temp['Date'])
br_temp['Date'] = pd.to_datetime(br_temp['Date'])
cn_humidity['Date'] = pd.to_datetime(cn_humidity['Date'])
io_humidity['Date'] = pd.to_datetime(io_humidity['Date'])
il_humidity['Date'] = pd.to_datetime(il_humidity['Date'])
br_humidity['Date'] = pd.to_datetime(br_humidity['Date'])

cn_corn = cn_corn[['Date', 'Close']].rename(columns={'Date': 'Date', 'Close': 'cn_corn'})
us_corn = us_corn[['Date', 'Close']].rename(columns={'Date': 'Date', 'Close': 'us_corn'})
us_hardwheat = us_hardwheat[['Date', 'Close']].rename(columns={'Date': 'Date', 'Close': 'us_hardwheat'})
br_corn = br_corn[['Date', 'Close']].rename(columns={'Date': 'Date', 'Close': 'br_corn'})
us_oat = us_oat[['Date', 'Close']].rename(columns={'Date': 'Date', 'Close': 'us_oat'})
us_rice = us_rice[['Date', 'Close']].rename(columns={'Date': 'Date', 'Close': 'us_rice'})
us_softwheat = us_softwheat[['Date', 'Close']].rename(columns={'Date': 'Date', 'Close': 'us_softwheat'})
io_temp = io_temp[['Date', 'Temp']].rename(columns={'Date': 'Date', 'Temp': 'io_temp'})
il_temp = il_temp[['Date', 'Temp']].rename(columns={'Date': 'Date', 'Temp': 'il_temp'})
cn_temp = cn_temp[['Date', 'Temp']].rename(columns={'Date': 'Date', 'Temp': 'cn_temp'})
br_temp = br_temp[['Date', 'Temp']].rename(columns={'Date': 'Date', 'Temp': 'br_temp'})
cn_humidity = cn_humidity[['Date', 'Humid']].rename(columns={'Date': 'Date', 'Humid': 'cn_humidity'})
io_humidity = io_humidity[['Date', 'Humid']].rename(columns={'Date': 'Date', 'Humid': 'io_humidity'})
il_humidity = il_humidity[['Date', 'Humid']].rename(columns={'Date': 'Date', 'Humid': 'il_humidity'})
br_humidity = br_humidity[['Date', 'Humid']].rename(columns={'Date': 'Date', 'Humid': 'br_humidity'})


datelist = us_corn['Date'].values
dataframes = [cn_corn, us_corn, us_hardwheat, br_corn, us_oat, us_rice, us_softwheat, io_temp, il_temp, cn_temp, br_temp, cn_humidity, br_humidity, io_humidity, il_humidity]

# get intersecting dates
for df in dataframes:
    datelist = np.intersect1d(datelist, df['Date'].values)

# filter by intersecting dates
filtered_df = []
for df in dataframes:
    filtered_df.append(df.loc[df['Date'].isin(datelist)])

dfs = filtered_df[0]
for df in filtered_df[1:]:
    dfs = pd.merge(dfs, df, on='Date', how='outer')

# split data into target, crops, weather, international 
target = dfs['us_corn']
crops = dfs[['us_hardwheat', 'us_oat', 'us_rice', 'us_softwheat']]
weather = dfs[['io_temp', 'io_humidity', 'il_temp', 'il_humidity']]
international = dfs[['br_corn', 'br_temp', 'br_humidity', 'cn_corn', 'cn_temp', 'cn_humidity']]

# normalize 
scaler_target = MinMaxScaler()
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))
scaler_crops = MinMaxScaler()
crops_scaled = scaler_crops.fit_transform(crops)
scaler_weather = MinMaxScaler()
weather_scaled = scaler_weather.fit_transform(weather)
scaler_inter = MinMaxScaler()
inter_scaled = scaler_inter.fit_transform(international)

# train test split
split = int(len(target) * 0.85)
target_train, target_test = target_scaled[:split], target_scaled[split:]
crops_train, crops_test = crops_scaled[:split], crops_scaled[split:]
weather_train, weather_test = weather_scaled[:split], weather_scaled[split:]
inter_train, inter_test = inter_scaled[:split], inter_scaled[split:]

target_train_tensor = torch.tensor(target_train, dtype=torch.float32)
target_test_tensor = torch.tensor(target_test, dtype=torch.float32)
crops_train_tensor = torch.tensor(crops_train, dtype=torch.float32).view(-1,1,4)
crops_test_tensor = torch.tensor(crops_test, dtype=torch.float32).view(-1,1,4)
weather_train_tensor = torch.tensor(weather_train, dtype=torch.float32).view(-1,1,4)
weather_test_tensor = torch.tensor(weather_test, dtype=torch.float32).view(-1,1,4)
inter_train_tensor = torch.tensor(inter_train, dtype=torch.float32).view(-1,1,6)
inter_test_tensor = torch.tensor(inter_test, dtype=torch.float32).view(-1,1,6)

crops_train_dataset = TensorDataset(crops_train_tensor, target_train_tensor)
crops_test_dataset = TensorDataset(crops_test_tensor, target_test_tensor)
weather_train_dataset = TensorDataset(weather_train_tensor, target_train_tensor)
weather_test_dataset = TensorDataset(weather_test_tensor, target_test_tensor)
inter_train_dataset = TensorDataset(inter_train_tensor, target_train_tensor)
inter_test_dataset = TensorDataset(inter_test_tensor, target_test_tensor)

# RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5) 
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.tanh(out[:, -1, :])  
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
def train_model(model, train_loader, test_loader, criterion, num_epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        model.train()
        for inputs, target in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        
        train_loss.append(eval_model(model, train_loader))
        test_loss.append(eval_model(model, test_loader))

        if (epoch % 10 == 0):
          print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss[-1]}, Test Loss: {test_loss[-1]}")

    return train_loss, test_loss

def eval_model(model, test_loader):
    model.eval()
    pred, actual = [], []

    with torch.no_grad():
        for inputs, target in test_loader:
            outputs = model(inputs)
            pred.append(outputs.detach().numpy())
            actual.append(target.detach().numpy())
    
    pred = np.concatenate(pred, axis=0)
    actual = np.concatenate(actual, axis=0)

    pred_inv = scaler_target.inverse_transform(pred)
    actual_inv = scaler_target.inverse_transform(actual)

    loss = mean_squared_error(actual_inv, pred_inv)
    return loss

def plot_figure(loss, num, title):
    train_loss = loss[0]
    test_loss = loss[1]
    plt.figure(num=num)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.title(f'{title} Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

criterion = nn.MSELoss()
batch_size = 64
hidden_size = 100

crops_train_loader = DataLoader(crops_train_dataset, batch_size=batch_size) 
crops_test_loader = DataLoader(crops_test_dataset, batch_size=batch_size)
weather_train_loader = DataLoader(weather_train_dataset, batch_size=batch_size)
weather_test_loader = DataLoader(weather_test_dataset, batch_size=batch_size)
inter_train_loader = DataLoader(inter_train_dataset, batch_size=batch_size)
inter_test_loader = DataLoader(inter_test_dataset, batch_size=batch_size)

crops_model = RNN(input_size=4, hidden_size=hidden_size, output_size=1)
weather_model = RNN(input_size=4, hidden_size=hidden_size, output_size=1)
inter_model = RNN(input_size=6, hidden_size=hidden_size, output_size=1)
losses = []

# hyperparams
lr = .0001
num_epochs = 200

crops_loss = train_model(crops_model, crops_train_loader, crops_test_loader, criterion, num_epochs, lr)
losses.append(f"crops final train loss {crops_loss[0][-1]} test loss {crops_loss[1][-1]}")
plot_figure(crops_loss, 1, "Crops")

# hyperparams
lr = .0001
num_epochs = 200

weather_loss = train_model(weather_model, weather_train_loader, weather_test_loader, criterion, num_epochs, lr)
losses.append(f"weather final train loss {weather_loss[0][-1]} test loss {weather_loss[1][-1]}")
plot_figure(weather_loss, 2, "Weather")

# hyperparams
lr = .0001
num_epochs = 200

inter_loss = train_model(inter_model, inter_train_loader, inter_test_loader, criterion, num_epochs, lr)
losses.append(f"inter final train loss {inter_loss[0][-1]} test loss {inter_loss[1][-1]}")
plot_figure(inter_loss, 3, "Int")

print(losses)
plt.show()
