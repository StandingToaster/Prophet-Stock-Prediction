import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class LSTMStockPredictor(nn.Module):
    """Defines an LSTM model for stock price prediction."""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Predict next stock price

def train_lstm_model(data):
    # Step 1: Detect GPU and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a copy of your data to preserve the original for plotting
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = data.copy()
    data_scaled['Close'] = scaler.fit_transform(data_scaled[['Close']])
    
    X_train = []
    y_train = []

    # Create sequences for training
    for i in range(len(data_scaled) - 30):
        X_train.append(data_scaled['Close'].values[i:i+30])
        y_train.append(data_scaled['Close'].values[i+30])
    
    # Step 2: Convert to PyTorch tensors and move them to the device
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32).view(-1, 30, 1).to(device)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1).to(device)
    
    # Step 3: Initialize your LSTM model and move it to the device
    model = LSTMStockPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(50):  # Reduced epochs for faster training
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    # Optional Step 4: Compile with TorchScript to optimize prediction speed
    scripted_model = torch.jit.script(model)
    return scripted_model, scaler



def predict_lstm(model, data, scaler, future_days=365):
    # Step 1: Detect GPU and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    original_last_30 = data[['Close']].tail(30)  # This is a DataFrame with column "Close"
    last_30_days_scaled = scaler.transform(original_last_30)

    predictions = []
    # Step 2: Create the input sequence tensor and move it to the device
    input_seq = torch.tensor(last_30_days_scaled, dtype=torch.float32).view(1, 30, 1).to(device)
    
    for _ in range(future_days):
        with torch.no_grad():
            pred = model(input_seq)
        # Step 3: Move prediction to CPU for inverse transformation
        pred = pred.cpu()
        pred_price = scaler.inverse_transform([[pred.item()]])[0][0]
        predictions.append(pred_price)
        
        # Scale the predicted price (still using the scaler)
        pred_scaled = scaler.transform([[pred_price]])[0][0]
        # Step 4: Update the input sequence and ensure it remains on the device
        input_seq = torch.cat(
            (input_seq[:, 1:, :], torch.tensor([[[pred_scaled]]], dtype=torch.float32).to(device)),
            dim=1
        )
    
    return predictions


