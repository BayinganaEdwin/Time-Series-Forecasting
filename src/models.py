import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# PyTorch LSTM Model
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# PyTorch GRU Model
class GRUForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(GRUForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.gru(x, h0)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Statistical Model Wrapper (SARIMA or Holt-Winters)
class StatisticalForecaster:
    def __init__(self, model_type='sarima', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.fitted_model = None
        self.train_data = None
        
    def fit(self, train_data):
        """
        Fits the statistical model on the training data.
        """
        self.train_data = np.asarray(train_data, dtype=np.float64)
        
        if self.model_type == 'sarima':
            # For SARIMA, default to ARIMA(1, 1, 1) x (1, 0, 0, 24) for speed and stability
            order = self.kwargs.get('order', (1, 1, 1))
            seasonal_order = self.kwargs.get('seasonal_order', (1, 0, 0, 24))
            
            # To speed up training, we only fit on the last 1008 samples (1 week) of train data
            # since statsmodels SARIMAX on 6000+ samples with seasonality can take a long time.
            fit_data = self.train_data[-1008:] if len(self.train_data) > 1008 else self.train_data
            
            model = SARIMAX(fit_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            self.fitted_model = model.fit(disp=False)
            
        elif self.model_type == 'holtwinters':
            # Holt-Winters Exponential Smoothing
            # seasonal_periods = 144 (24 hours * 6 intervals)
            seasonal_periods = self.kwargs.get('seasonal_periods', 144)
            model = ExponentialSmoothing(
                self.train_data,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods,
                initialization_method='estimated'
            )
            self.fitted_model = model.fit()
            
    def predict_one_step_ahead(self, history, actual_test_data):
        """
        Performs one-step-ahead rolling prediction on the test dataset.
        For each step in the test dataset, it receives the history up to that step and predicts the next value.
        """
        predictions = []
        
        if self.model_type == 'sarima':
            # To perform rolling one-step-ahead forecasting in statsmodels:
            # We can use the apply() method to extend the fitted model with new observations
            # without re-estimating the parameters.
            full_series = np.concatenate([self.train_data[-1008:], actual_test_data])
            # Create a new results object with all data
            res = self.fitted_model.apply(full_series)
            
            # The predictions for the test portion (the last len(actual_test_data) elements)
            # represent the one-step-ahead forecasts.
            # E.g. index -len(actual_test_data) to -1 predicted the corresponding test values
            pred = res.predict(start=len(self.train_data[-1008:]), end=len(full_series)-1)
            predictions = list(pred)
            
        elif self.model_type == 'holtwinters':
            # For Holt-Winters rolling prediction, we can run a loop or apply filtering.
            # A standard way is to update the level, trend, and seasonal components at each step.
            # Statsmodels does not have an easy .apply() for ExponentialSmoothing, but since we want
            # rolling 1-step forecasts, we can re-initialize or do a rolling update.
            # To be efficient, we can use the fitted parameters to project 1 step ahead rolling.
            # Let's do it using a simple simulation or rolling update:
            alpha = self.fitted_model.params['smoothing_level']
            beta = self.fitted_model.params['smoothing_trend']
            gamma = self.fitted_model.params['smoothing_seasonal']
            
            # Simple implementation of rolling Holt-Winters prediction
            # We can also just use the statsmodels predict() or simulate()
            # For simplicity and correctness, we can fit on history + test data (excluding current point)
            # or use the fitted parameters.
            # Let's write a simple simulation loop:
            # Level (l), Trend (b), Season (s)
            p = self.fitted_model.model.seasonal_periods
            l = self.fitted_model.level
            b = self.fitted_model.trend
            s = self.fitted_model.season
            
            # We will copy the final state of training
            curr_l = l[-1]
            curr_b = b[-1]
            curr_s = list(s[-p:])
            
            # Iterate through the test set and compute 1-step forecasts
            for i in range(len(actual_test_data)):
                # Predict 1-step ahead: y_hat = l_t + b_t + s_{t-p+1}
                pred_val = curr_l + curr_b + curr_s[0]
                predictions.append(pred_val)
                
                # Get the actual observation
                y = actual_test_data[i]
                
                # Update equations
                prev_l = curr_l
                curr_l = alpha * (y - curr_s[0]) + (1 - alpha) * (curr_l + curr_b)
                curr_b = beta * (curr_l - prev_l) + (1 - beta) * curr_b
                s_new = gamma * (y - prev_l - curr_b) + (1 - gamma) * curr_s[0]
                
                # Update seasonal buffer (rolling list)
                curr_s.pop(0)
                curr_s.append(s_new)
                
        return np.array(predictions)
