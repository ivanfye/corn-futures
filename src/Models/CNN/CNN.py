import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy


def get_cleaned_data(data_path: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Clean and return all data.
    Parameters:
        `data_path` - The path to the data.
    Returns: A tuple containing inputs, labels, and dates (in that order).
    """

    #############
    # Load data #
    #############

    # US corn prices
    price_us_corn = pd.read_csv(f'{data_path}/US corn/us_corn_price.csv')

    # US related crop prices
    price_us_red_wheat = pd.read_csv(f'{data_path}/Related Crops/ke.csv')
    price_us_oat = pd.read_csv(f'{data_path}/Related Crops/zo.csv')
    price_us_rough_rice = pd.read_csv(f'{data_path}/Related Crops/zr.csv')
    price_us_wheat = pd.read_csv(f'{data_path}/Related Crops/zw.csv')

    # US weather
    weather_us_il_temperature = pd.read_csv(f'{data_path}/Weather/IL_Temp.csv')
    weather_us_il_humidity = pd.read_csv(f'{data_path}/Weather/IL_Humidity.csv')
    weather_us_ia_temperature = pd.read_csv(f'{data_path}/Weather/IOWA_Temp.csv')
    weather_us_ia_humidity = pd.read_csv(f'{data_path}/Weather/IOWA_Humidity.csv')

    # Foreign weather
    weather_br_temperature = pd.read_csv(f'{data_path}/Weather/BR_Temp.csv')
    weather_br_humidity = pd.read_csv(f'{data_path}/Weather/BR_Humidity.csv')
    weather_cn_temperature = pd.read_csv(f'{data_path}/Weather/CH_Temp.csv')
    weather_cn_humidity = pd.read_csv(f'{data_path}/Weather/CH_Humidity.csv')

    # Foreign corn prices
    price_br_corn = pd.read_csv(f'{data_path}/int corn currency raw/Corn Prices Brazil.csv')
    price_cn_corn = pd.read_csv(f'{data_path}/int corn currency raw/Corn Prices China.csv')

    # Exchange rates
    exchange_usd_to_brl = pd.read_csv(f'{data_path}/int corn currency raw/currency conversion/usdtobrl.csv')
    exchange_usd_to_cny = pd.read_csv(f'{data_path}/int corn currency raw/currency conversion/usdtocny.csv')

    #########################################################
    # Create standardized date columns containing datetimes #
    #########################################################

    # US corn prices
    price_us_corn['date'] = pd.to_datetime(price_us_corn['date'])

    # US related crop prices
    price_us_red_wheat['date'] = pd.to_datetime(price_us_red_wheat['date'])
    price_us_oat['date'] = pd.to_datetime(price_us_oat['date'])
    price_us_rough_rice['date'] = pd.to_datetime(price_us_rough_rice['date'])
    price_us_wheat['date'] = pd.to_datetime(price_us_wheat['date'])

    # US weather
    weather_us_il_temperature['date'] = pd.to_datetime(weather_us_il_temperature['date'])
    weather_us_il_humidity['date'] = pd.to_datetime(weather_us_il_humidity['date'])
    weather_us_ia_temperature['date'] = pd.to_datetime(weather_us_ia_temperature['date'])
    weather_us_ia_humidity['date'] = pd.to_datetime(weather_us_ia_humidity['date'])

    # Foreign weather
    weather_br_temperature['date'] = pd.to_datetime(weather_br_temperature['date'])
    weather_br_humidity['date'] = pd.to_datetime(weather_br_humidity['date'])
    weather_cn_temperature['date'] = pd.to_datetime(weather_cn_temperature['date'])
    weather_cn_humidity['date'] = pd.to_datetime(weather_cn_humidity['date'])

    # Foreign corn prices
    price_br_corn['date'] = pd.to_datetime(price_br_corn['Date'])
    price_cn_corn['date'] = pd.to_datetime(price_cn_corn['Date'])

    # Exchange rates
    exchange_usd_to_brl['date'] = pd.to_datetime(exchange_usd_to_brl['Date'])
    exchange_usd_to_cny['date'] = pd.to_datetime(exchange_usd_to_cny['Date'])

    #########################################################
    # Create standardized value columns containing float64s #
    #########################################################

    # US corn prices
    price_us_corn['value'] = price_us_corn['Close']

    # US related crop prices
    price_us_red_wheat['value'] = price_us_red_wheat['Close']
    price_us_oat['value'] = price_us_oat['Close']
    price_us_rough_rice['value'] = price_us_rough_rice['Close']
    price_us_wheat['value'] = price_us_wheat['Close']

    # US weather
    weather_us_il_temperature['value'] = weather_us_il_temperature['tempC']
    weather_us_il_humidity['value'] = weather_us_il_humidity['relh']
    weather_us_ia_temperature['value'] = weather_us_ia_temperature['tempC']
    weather_us_ia_humidity['value'] = weather_us_ia_humidity['relh']

    # Foreign weather
    weather_br_temperature['value'] = weather_br_temperature['tempC']
    weather_br_humidity['value'] = weather_br_humidity['relh']
    weather_cn_temperature['value'] = weather_cn_temperature['tempC']
    weather_cn_humidity['value'] = weather_cn_humidity['relh']

    # Foreign corn prices
    price_br_corn['value'] = price_br_corn['Close']
    price_cn_corn['value'] = price_cn_corn['Close'].str.replace(',', '').astype('float64')

    # Exchange rates
    exchange_usd_to_brl['value'] = exchange_usd_to_brl[' Close']
    exchange_usd_to_cny['value'] = exchange_usd_to_cny[' Close']

    ####################
    # Get common dates #
    ####################

    # List of all data
    data = [
        price_us_corn,
        price_us_red_wheat, price_us_oat, price_us_rough_rice, price_us_wheat,
        weather_us_il_temperature, weather_us_il_humidity, weather_us_ia_temperature, weather_us_ia_humidity,
        weather_br_temperature, weather_br_humidity, weather_cn_temperature, weather_cn_humidity,
        price_br_corn, price_cn_corn,
        exchange_usd_to_brl, exchange_usd_to_cny
    ]

    # Find dates with data present in each frame
    for i in range(len(data)):
        if i == 0:
            dates = set(data[i]['date'].to_list())
        else:
            dates = dates & set(data[i]['date'].to_list())
        dates = dates - set(data[i].loc[pd.isna(data[i]['value']), 'date'].to_list())
    dates = pd.Series(list(dates)).sort_values()

    #######################################################
    # Keep shared dates only and sort data frames by date #
    #######################################################

    # US corn prices
    price_us_corn = price_us_corn[price_us_corn['date'].isin(dates)].sort_values('date')

    # US related crop prices
    price_us_red_wheat = price_us_red_wheat[price_us_red_wheat['date'].isin(dates)].sort_values('date')
    price_us_oat = price_us_oat[price_us_oat['date'].isin(dates)].sort_values('date')
    price_us_rough_rice = price_us_rough_rice[price_us_rough_rice['date'].isin(dates)].sort_values('date')
    price_us_wheat = price_us_wheat[price_us_wheat['date'].isin(dates)].sort_values('date')

    # US weather
    weather_us_il_temperature = weather_us_il_temperature[weather_us_il_temperature['date'].isin(dates)].sort_values('date')
    weather_us_il_humidity = weather_us_il_humidity[weather_us_il_humidity['date'].isin(dates)].sort_values('date')
    weather_us_ia_temperature = weather_us_ia_temperature[weather_us_ia_temperature['date'].isin(dates)].sort_values('date')
    weather_us_ia_humidity = weather_us_ia_humidity[weather_us_ia_humidity['date'].isin(dates)].sort_values('date')

    # Foreign weather
    weather_br_temperature = weather_br_temperature[weather_br_temperature['date'].isin(dates)].sort_values('date')
    weather_br_humidity = weather_br_humidity[weather_br_humidity['date'].isin(dates)].sort_values('date')
    weather_cn_temperature = weather_cn_temperature[weather_cn_temperature['date'].isin(dates)].sort_values('date')
    weather_cn_humidity = weather_cn_humidity[weather_cn_humidity['date'].isin(dates)].sort_values('date')

    # Foreign corn prices
    price_br_corn = price_br_corn[price_br_corn['date'].isin(dates)].sort_values('date')
    price_cn_corn = price_cn_corn[price_cn_corn['date'].isin(dates)].sort_values('date')

    # Exchange rates
    exchange_usd_to_brl = exchange_usd_to_brl[exchange_usd_to_brl['date'].isin(dates)].sort_values('date')
    exchange_usd_to_cny = exchange_usd_to_cny[exchange_usd_to_cny['date'].isin(dates)].sort_values('date')

    ###############
    # Final steps #
    ###############

    # Modify foregin corn prices based on exchange rates
    price_br_corn['value'] = np.round(price_br_corn['value'].to_numpy() / exchange_usd_to_brl['value'].to_numpy(), decimals=2)
    price_cn_corn['value'] = np.round(price_cn_corn['value'].to_numpy() / exchange_usd_to_cny['value'].to_numpy(), decimals=2)

    # Get inputs
    inputs = [
        price_us_red_wheat, price_us_oat, price_us_rough_rice, price_us_wheat,
        weather_us_il_temperature, weather_us_il_humidity, weather_us_ia_temperature, weather_us_ia_humidity,
        weather_br_temperature, weather_br_humidity, weather_cn_temperature, weather_cn_humidity,
        price_br_corn, price_cn_corn,
    ]
    for i in range(len(inputs)):
        inputs[i] = inputs[i]['value'].to_numpy()

    # Get labels
    labels = price_us_corn['value'].to_numpy()

    # Get dates
    dates = dates.to_numpy()

    # Return cleaned data
    return inputs, labels, dates


def create_dataset(inputs: list[np.ndarray], labels: np.ndarray, dates: np.ndarray,
                   start: np.datetime64, end: np.datetime64, input_length: int, offset: int,
                   overlap: bool=False) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
    """
    Creates a dataset.
    Parameters:
        `inputs` - The list of inputs.
        `labels` - The labels.
        `dates` - The dates.
        `start` - The earliest date to be included in the datset.
        `end` - The latest date to be included in the dataset.
        `input_length` - The number of consecutive values to include per input.
        `offset` - The number of days between the latest input and the label.
        `overlap` - Whether the dates of training inputs are allowed to overlap.
    Returns: A list of (input, label) pairs. Each input is a list of tensors.
    """

    # Indices into inputs and labels defined earlier
    start_index = np.argmin(np.abs(dates - start))
    end_index = np.argmin(np.abs(dates - end))

    # Lists to store results
    inputs_dataset = []
    labels_dataset = []

    # Iterate over inputs and labels
    i = start_index
    while i <= end_index + 1:

        # Create a list of arrays as a single input
        input_current = []
        for j in range(len(inputs)):
            input_current.append(torch.from_numpy(inputs[j][i:i + input_length]).float())

        # Calculate the expected date and actual index of the label
        label_date = dates[i + input_length] + np.timedelta64(offset, 'D')
        label_index = np.argmin(np.abs(dates - label_date))

        # Stop if the actual date of the label is past the end
        if dates[label_index] > end:
            break

        # Get the label
        label_current = torch.tensor(labels[label_index]).float()

        # Store the current input and label
        inputs_dataset.append(input_current)
        labels_dataset.append(label_current)

        # Stop if the last index has been reached
        if label_index == dates.size - 1:
            break

        # Set the index of the first value of the next input
        if overlap:
            i += 1
        else:
            i += input_length

    # Return a tuple of inputs and labels
    return list(zip(inputs_dataset, labels_dataset))


class CNN(nn.Module):
    """
    A convolutional neural network for predicting US corn prices.
    """

    def __init__(self, input_sources: int, input_length: int) -> None:
        """
        Create a CNN.
        Parameters:
            `input_sources` - The number of distinct sources of data in a single
                              input.
            `input_length` - The length of a single input.
        """
        super().__init__()
        self.conv1 = nn.ModuleList([nn.Conv1d(1, 2, 4) for _ in range(input_sources)])
        self.pool1 = nn.AvgPool1d(2)
        self.conv2 = nn.ModuleList([nn.Conv1d(2, 4, 2) for _ in range(input_sources)])
        self.pool2 = nn.AvgPool1d(2)
        self.conv_end_size = int(4 * torch.floor(torch.tensor(input_length / 4 - 1.25)))
        self.fc_start_size = self.conv_end_size * input_sources
        self.fc1 = nn.Linear(self.fc_start_size, self.fc_start_size // 2)
        self.fc2 = nn.Linear(self.fc_start_size // 2, 1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        Parameters:
            `x` - Input data.
        Return: Predicted price.
        """
        out = []
        for i in range(len(x)):
            out.append(self.conv1[i](x[i].unsqueeze(1)))
            out[i] = self.pool1(out[i])
        for i in range(len(out)):
            out[i] = self.conv2[i](out[i])
            out[i] = self.pool2(out[i])
            out[i] = out[i].view(-1, self.conv_end_size)
        out = torch.cat(out, dim=1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out).squeeze(dim=1)
        return out


def train_model(model: CNN, train_data: list, val_data: list, batch_size: int=64, learning_rate: int=0.0001,
                weight_decay: int=0.0, num_epochs: int=100, plot_every: int=25) -> CNN:
    """
    Train a convolutional neural network.
    Parameters:
        `model` - The CNN.
        `train_data` - The training set.
        `val_data` - The validation set.
        `batch_size` - The batch size.
        `learning_rate` - The learning rate.
        `weight_decay` - The weight decay.
        `num_epochs` - The number of epochs.
        `plot_every` - How often to plot metrics.
    Returns: The model with the lowest validation loss.
    """

    # Setup
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data), shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Track metrics for plotting
    iters, train_loss, val_loss = [], [], []
    iter_count = 0
    print(f'Total Iterations: {len(train_data) // batch_size * num_epochs}')

    # Training loop
    try:
        for epoch in range(num_epochs):
            for x, t in iter(train_loader):

                # Check if the batch size is incorrect
                if t.shape[0] < batch_size:
                    continue

                # Perform SGD
                y = model(x)
                loss = criterion(y, t)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Update metrics
                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    train_loss.append(loss.detach().item())
                    for x, t in iter(val_loader):
                        val_loss.append(criterion(model(x), t).item())

                        # Track best model
                        if val_loss[-1] == min(val_loss):
                            model_best = copy.deepcopy(model)
                            iter_best = iter_count
                    
                    # Display metrics
                    print(f'{iter_count} Train Loss: {train_loss[-1]} Val Loss: {val_loss[-1]}')

    # Generate training curve
    finally:
        plt.plot(iters, train_loss)
        plt.plot(iters, val_loss)
        plt.axvline(iter_best, c='g')
        plt.legend(['Training', 'Validation', 'Best Model'])
        plt.title('Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

        # Model with best metrics
        print(f'{iter_best} Train Loss: {train_loss[(iter_best // plot_every) - 1]} Val Loss: {val_loss[(iter_best // plot_every) - 1]}')
        return model_best


def test_loss(model: CNN, test_data: list) -> None:
    """
    Get the loss on the test set for a CNN.
    Parameters:
        `model` - The CNN.
        `test_data` - The test set.
    """

    # Calculate test loss
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)
    for x, t in iter(test_loader):
        print(f'Test Loss: {nn.MSELoss()(model(x), t).item()}')
