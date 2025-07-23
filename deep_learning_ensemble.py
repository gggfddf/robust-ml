#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - DEEP LEARNING ENSEMBLE
Complete Deep Learning Model Training Pipeline
Production-Grade Implementation with Advanced Architectures
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ML Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Custom imports
from data.live_data_loader import DataLoader
from technical_analysis.advanced_indicators import TechnicalAnalyzer
from config import get_config

class DeepLearningModels:
    """Deep Learning Model Architectures"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
    
    def build_lstm_model(self):
        """Build LSTM model for time series prediction"""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_first=True
                )
                
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc2 = nn.Linear(hidden_size // 2, 1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # Initialize hidden state
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                
                # Forward propagate LSTM
                out, _ = self.lstm(x, (h0, c0))
                
                # Get the last time step output
                out = out[:, -1, :]
                
                # Apply dropout and fully connected layers
                out = self.dropout(out)
                out = self.relu(self.fc1(out))
                out = self.dropout(out)
                out = self.fc2(out)
                
                return out
        
        return LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def build_cnn_model(self):
        """Build CNN model for pattern recognition"""
        class CNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, dropout):
                super(CNNModel, self).__init__()
                
                # Calculate sequence length (assuming square input)
                self.sequence_length = int(np.sqrt(input_size))
                if self.sequence_length * self.sequence_length != input_size:
                    self.sequence_length = input_size
                
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                
                self.pool = nn.MaxPool1d(2)
                self.dropout = nn.Dropout(dropout)
                
                # Calculate the size after convolutions and pooling
                conv_output_size = 128 * (self.sequence_length // 8)
                
                self.fc1 = nn.Linear(conv_output_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, 1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # Reshape input for 1D convolution
                batch_size = x.size(0)
                x = x.view(batch_size, 1, -1)
                
                # Convolutional layers
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.dropout(x)
                
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.dropout(x)
                
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = self.dropout(x)
                
                # Flatten and fully connected layers
                x = x.view(batch_size, -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                
                return x
        
        return CNNModel(self.input_size, self.hidden_size, self.dropout)
    
    def build_transformer_model(self):
        """Build Transformer model for sequence modeling"""
        class TransformerModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(TransformerModel, self).__init__()
                
                self.input_size = input_size
                self.hidden_size = hidden_size
                
                # Embedding layer
                self.embedding = nn.Linear(input_size, hidden_size)
                
                # Positional encoding
                self.pos_encoding = nn.Parameter(torch.randn(1, 100, hidden_size))
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Output layers
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc2 = nn.Linear(hidden_size // 2, 1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # Embed input
                x = self.embedding(x)
                
                # Add positional encoding
                if x.size(1) <= self.pos_encoding.size(1):
                    x = x + self.pos_encoding[:, :x.size(1), :]
                
                # Transformer encoding
                x = self.transformer(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Output layers
                x = self.dropout(x)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                
                return x
        
        return TransformerModel(self.input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def build_autoencoder_model(self):
        """Build AutoEncoder model for feature learning"""
        class AutoEncoderModel(nn.Module):
            def __init__(self, input_size, hidden_size, dropout):
                super(AutoEncoderModel, self).__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU()
                )
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_size // 4, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, input_size)
                )
                
                # Prediction head
                self.prediction_head = nn.Sequential(
                    nn.Linear(hidden_size // 4, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1)
                )
                
            def forward(self, x):
                # Encode
                encoded = self.encoder(x)
                
                # Decode (for reconstruction loss)
                decoded = self.decoder(encoded)
                
                # Predict (for prediction loss)
                predicted = self.prediction_head(encoded)
                
                return decoded, predicted
        
        return AutoEncoderModel(self.input_size, self.hidden_size, self.dropout)

class DeepLearningTrainer:
    """Deep Learning Model Trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        self.patience = 10
        
        # Results storage
        self.training_results = {}
        self.models = {}
        
    def prepare_data(self, data, sequence_length=10):
        """Prepare data for deep learning models"""
        print("📊 Preparing data for deep learning...")
        
        # Create sequences
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"  📊 Training samples: {len(X_train)}")
        print(f"  📊 Test samples: {len(X_test)}")
        print(f"  📊 Input shape: {X_train.shape}")
        
        return train_loader, test_loader, X_train.shape[2]
    
    def train_model(self, model, train_loader, test_loader, model_name):
        """Train a single model"""
        print(f"🤖 Training {model_name}...")
        
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training history
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if model_name == 'AutoEncoder':
                    decoded, predicted = model(batch_X)
                    # Combined loss: reconstruction + prediction
                    recon_loss = criterion(decoded, batch_X)
                    pred_loss = criterion(predicted.squeeze(), batch_y)
                    loss = recon_loss + pred_loss
                else:
                    predicted = model(batch_X)
                    loss = criterion(predicted.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            test_loss = 0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if model_name == 'AutoEncoder':
                        decoded, predicted = model(batch_X)
                    else:
                        predicted = model(batch_X)
                    
                    loss = criterion(predicted.squeeze(), batch_y)
                    test_loss += loss.item()
                    
                    predictions.extend(predicted.squeeze().cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())
            
            # Calculate average losses
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            
            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'models/ml_models/{model_name.lower()}_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"    ⏹️ Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        # Calculate final metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        results = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_test_loss': best_test_loss,
            'final_metrics': {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            },
            'predictions': predictions,
            'actuals': actuals
        }
        
        self.training_results[model_name] = results
        self.models[model_name] = model
        
        print(f"  ✅ {model_name} Training Complete:")
        print(f"    Final R²: {r2:.4f}")
        print(f"    Final RMSE: {np.sqrt(mse):.6f}")
        print(f"    Best Test Loss: {best_test_loss:.6f}")
        
        return results
    
    def train_all_models(self, data):
        """Train all deep learning models"""
        print("🚀 Training all deep learning models...")
        
        # Prepare data
        train_loader, test_loader, input_size = self.prepare_data(data)
        
        # Initialize model builder
        model_builder = DeepLearningModels(input_size=input_size)
        
        # Model configurations
        models_config = {
            'LSTM': model_builder.build_lstm_model(),
            'CNN': model_builder.build_cnn_model(),
            'Transformer': model_builder.build_transformer_model(),
            'AutoEncoder': model_builder.build_autoencoder_model()
        }
        
        # Train each model
        for model_name, model in models_config.items():
            try:
                self.train_model(model, train_loader, test_loader, model_name)
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
        
        return self.training_results
    
    def create_ensemble_predictions(self, data):
        """Create ensemble predictions from all trained models"""
        print("🤝 Creating ensemble predictions...")
        
        # Prepare data for prediction
        sequence_length = 10
        X = []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
        
        X = torch.FloatTensor(np.array(X)).to(self.device)
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                if model_name == 'AutoEncoder':
                    decoded, pred = model(X)
                else:
                    pred = model(X)
                predictions[model_name] = pred.squeeze().cpu().numpy()
        
        # Create ensemble (simple average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        predictions['Ensemble'] = ensemble_pred
        
        return predictions
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        print("📋 Generating training report...")
        
        # Calculate overall performance
        model_performance = {}
        for model_name, results in self.training_results.items():
            metrics = results['final_metrics']
            model_performance[model_name] = {
                'R²': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae']
            }
        
        # Find best model
        best_model = max(model_performance.items(), key=lambda x: x[1]['R²'])
        
        # Overall score calculation
        avg_r2 = np.mean([metrics['R²'] for metrics in model_performance.values()])
        overall_score = min(10, max(0, avg_r2 * 10))  # Convert to 0-10 scale
        
        report = {
            'overall_score': overall_score,
            'best_model': best_model[0],
            'best_model_r2': best_model[1]['R²'],
            'model_performance': model_performance,
            'total_models_trained': len(self.training_results),
            'training_completion': 'SUCCESS',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 DEEP LEARNING TRAINING - FINAL RESULTS")
        print("=" * 60)
        print(f"🏆 Overall Score: {overall_score:.2f}/10")
        print(f"🎯 Best Model: {best_model[0]} (R²: {best_model[1]['R²']:.4f})")
        print(f"🤖 Models Trained: {len(self.training_results)}")
        print(f"📊 Average R²: {avg_r2:.4f}")
        print(f"⏰ Completion Time: {report['timestamp']}")
        print("=" * 60)
        
        return report

def main():
    """Main execution function"""
    print("🚀 ULTIMATE MARKET AI ENGINE - DEEP LEARNING TRAINING")
    print("=" * 60)
    
    try:
        # Initialize components
        config = get_config()
        data_loader = DataLoader(config)
        technical_analyzer = TechnicalAnalyzer(config)
        
        # Load and prepare data
        print("📊 Loading market data...")
        data = data_loader.fetch_single_timeframe_data(config.SYMBOL, '1d')
        
        if data is None or len(data) == 0:
            print("❌ No data available for training")
            return None
        
        # Add technical indicators
        print("🔧 Adding technical indicators...")
        indicators = technical_analyzer.calculate_all_indicators(data)
        data = pd.concat([data, indicators], axis=1)
        
        # Prepare target variable
        data['target'] = data['Close'].shift(-1) / data['Close'] - 1
        
        # Remove NaN values
        data = data.dropna()
        
        # Prepare features for deep learning
        feature_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
        X = data[feature_cols].values
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize trainer
        trainer = DeepLearningTrainer(config)
        
        # Train all models
        training_results = trainer.train_all_models(X_scaled)
        
        # Create ensemble predictions
        ensemble_predictions = trainer.create_ensemble_predictions(X_scaled)
        
        # Generate report
        report = trainer.generate_training_report()
        
        print("\n✅ Deep Learning Training completed successfully!")
        return report
        
    except Exception as e:
        print(f"❌ Training Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()