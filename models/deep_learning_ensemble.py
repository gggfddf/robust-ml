"""
Ultimate Market AI Engine - Deep Learning Prediction System
==========================================================

This module implements a multi-architecture deep learning ensemble
for market prediction with confidence scoring and feature attribution.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
import os
from pathlib import Path
import warnings
from datetime import datetime, timedelta

# Import configuration
import sys
sys.path.append('..')
from config import get_config

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketDataset(Dataset):
    """Custom dataset for market data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMModel(nn.Module):
    """LSTM model for sequential pattern recognition."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out


class CNNModel(nn.Module):
    """CNN model for visual chart pattern recognition."""
    
    def __init__(self, input_size: int, num_filters: List[int], dropout: float = 0.2):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for filters in num_filters:
            self.conv_layers.append(
                nn.Conv1d(in_channels, filters, kernel_size=3, padding=1)
            )
            in_channels = filters
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters[-1], 1)
    
    def forward(self, x):
        # Reshape for CNN: (batch, channels, sequence_length)
        x = x.unsqueeze(1)
        
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
        
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class TransformerModel(nn.Module):
    """Transformer model for long-range dependencies."""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.2):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        batch_size, seq_len, _ = x.shape
        
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, seq, d_model)
        x = x[:, -1, :]  # Take last sequence element
        
        x = self.dropout(x)
        x = self.fc(x)
        return x


class AutoEncoderModel(nn.Module):
    """Autoencoder for anomaly detection."""
    
    def __init__(self, input_size: int, encoding_dim: int, dropout: float = 0.2):
        super(AutoEncoderModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, input_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class DeepLearningEnsemble:
    """
    Multi-architecture deep learning ensemble for market prediction.
    
    Features:
    - LSTM Networks: Sequential pattern recognition
    - CNN Layers: Visual chart pattern recognition
    - Transformer/Attention: Long-range dependencies
    - AutoEncoders: Anomaly detection
    - Ensemble Methods: Combine all models
    - Confidence scoring and feature attribution
    """
    
    def __init__(self, config=None):
        """Initialize the deep learning ensemble."""
        self.config = config or get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.models = {}
        self.scalers = {}
        self.model_weights = {}
        
        # Training history
        self.training_history = {}
        
        logger.info(f"Deep Learning Ensemble initialized on {self.device}")
    
    def _prepare_features(self, df: pd.DataFrame, technical_indicators: Dict) -> np.ndarray:
        """
        Prepare features for deep learning models.
        
        Args:
            df: OHLCV data
            technical_indicators: Technical indicator values
        
        Returns:
            Feature matrix
        """
        features = []
        
        # Price features
        features.append(df['Close'].values)
        features.append(df['Volume'].values if 'Volume' in df.columns else np.zeros(len(df)))
        features.append(df['High'].values)
        features.append(df['Low'].values)
        features.append(df['Open'].values)
        
        # Technical indicators
        for indicator_name, indicator_data in technical_indicators.items():
            if isinstance(indicator_data, pd.Series):
                features.append(indicator_data.values)
            elif isinstance(indicator_data, dict):
                for sub_name, sub_data in indicator_data.items():
                    if isinstance(sub_data, pd.Series):
                        features.append(sub_data.values)
        
        # Convert to numpy array
        feature_matrix = np.column_stack(features)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        return feature_matrix
    
    def _prepare_targets(self, df: pd.DataFrame, prediction_horizon: int = 5) -> np.ndarray:
        """
        Prepare target variables for prediction.
        
        Args:
            df: OHLCV data
            prediction_horizon: Number of periods ahead to predict
        
        Returns:
            Target array
        """
        # Predict future price change
        future_returns = df['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Convert to binary classification (up/down)
        targets = (future_returns > 0).astype(float)
        
        return targets.values
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            features: Feature matrix
            targets: Target array
            sequence_length: Length of input sequences
        
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def _train_model(self, model: nn.Module, train_loader: TorchDataLoader, val_loader: TorchDataLoader, 
                    model_name: str, epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train a single model.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name of the model
            epochs: Number of training epochs
        
        Returns:
            Training history
        """
        model.to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= 20:  # Early stopping patience
                    logger.info(f"Early stopping for {model_name} at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"{model_name} - Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def train_ensemble(self, df: pd.DataFrame, technical_indicators: Dict) -> Dict[str, Any]:
        """
        Train the complete ensemble of models.
        
        Args:
            df: OHLCV data
            technical_indicators: Technical indicator values
        
        Returns:
            Training results
        """
        logger.info("Starting ensemble training...")
        
        # Prepare features and targets
        features = self._prepare_features(df, technical_indicators)
        targets = self._prepare_targets(df)
        
        # Create sequences
        X, y = self._create_sequences(features, targets, sequence_length=60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        self.scalers['main'] = scaler
        
        # Create data loaders
        train_dataset = MarketDataset(X_train_scaled, y_train)
        val_dataset = MarketDataset(X_val_scaled, y_val)
        test_dataset = MarketDataset(X_test_scaled, y_test)
        
        train_loader = TorchDataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        test_loader = TorchDataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Train LSTM model
        logger.info("Training LSTM model...")
        lstm_model = LSTMModel(
            input_size=features.shape[1],
            hidden_size=self.config.LSTM_UNITS,
            num_layers=self.config.LSTM_LAYERS,
            dropout=self.config.DROPOUT_RATE
        )
        
        lstm_history = self._train_model(lstm_model, train_loader, val_loader, "lstm")
        self.models['lstm'] = lstm_model
        self.training_history['lstm'] = lstm_history
        
        # Train CNN model
        logger.info("Training CNN model...")
        cnn_model = CNNModel(
            input_size=features.shape[1],
            num_filters=self.config.CNN_FILTERS,
            dropout=self.config.DROPOUT_RATE
        )
        
        cnn_history = self._train_model(cnn_model, train_loader, val_loader, "cnn")
        self.models['cnn'] = cnn_model
        self.training_history['cnn'] = cnn_history
        
        # Train Transformer model
        logger.info("Training Transformer model...")
        transformer_model = TransformerModel(
            input_size=features.shape[1],
            d_model=128,
            nhead=self.config.TRANSFORMER_HEADS,
            num_layers=self.config.TRANSFORMER_LAYERS,
            dropout=self.config.DROPOUT_RATE
        )
        
        transformer_history = self._train_model(transformer_model, train_loader, val_loader, "transformer")
        self.models['transformer'] = transformer_model
        self.training_history['transformer'] = transformer_history
        
        # Train AutoEncoder model
        logger.info("Training AutoEncoder model...")
        autoencoder_model = AutoEncoderModel(
            input_size=features.shape[1],
            encoding_dim=32,
            dropout=self.config.DROPOUT_RATE
        )
        
        # Autoencoder training (reconstruction loss)
        autoencoder_model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder_model.parameters(), lr=self.config.LEARNING_RATE)
        
        autoencoder_losses = []
        for epoch in range(50):  # Fewer epochs for autoencoder
            autoencoder_model.train()
            epoch_loss = 0.0
            
            for batch_X, _ in train_loader:
                batch_X = batch_X.to(self.device)
                batch_X_flat = batch_X.view(batch_X.size(0), -1)  # Flatten for autoencoder
                
                optimizer.zero_grad()
                reconstructed = autoencoder_model(batch_X_flat)
                loss = criterion(reconstructed, batch_X_flat)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            autoencoder_losses.append(epoch_loss / len(train_loader))
            
            if epoch % 10 == 0:
                logger.info(f"AutoEncoder - Epoch {epoch}: Loss: {epoch_loss/len(train_loader):.4f}")
        
        self.models['autoencoder'] = autoencoder_model
        self.training_history['autoencoder'] = {'train_losses': autoencoder_losses}
        
        # Calculate model weights based on validation performance
        self._calculate_model_weights()
        
        # Evaluate ensemble
        ensemble_results = self._evaluate_ensemble(test_loader)
        
        logger.info("Ensemble training completed!")
        return ensemble_results
    
    def _calculate_model_weights(self):
        """Calculate ensemble weights based on validation performance."""
        weights = {}
        
        for model_name, history in self.training_history.items():
            if 'best_val_loss' in history:
                # Lower validation loss = higher weight
                weights[model_name] = 1.0 / (1.0 + history['best_val_loss'])
            else:
                weights[model_name] = 1.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.model_weights = {name: weight / total_weight for name, weight in weights.items()}
        
        logger.info(f"Model weights: {self.model_weights}")
    
    def _evaluate_ensemble(self, test_loader: TorchDataLoader) -> Dict[str, float]:
        """Evaluate ensemble performance on test set."""
        predictions = []
        actuals = []
        
        for model_name, model in self.models.items():
            if model_name == 'autoencoder':
                continue  # Skip autoencoder for classification evaluation
            
            model.eval()
            model_preds = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    probs = torch.sigmoid(outputs)
                    model_preds.extend(probs.cpu().numpy())
            
            predictions.append(model_preds)
            if not actuals:  # Only need to collect actuals once
                actuals = [batch_y.cpu().numpy() for _, batch_y in test_loader]
                actuals = np.concatenate(actuals)
        
        # Ensemble prediction
        ensemble_preds = np.zeros_like(predictions[0])
        for i, preds in enumerate(predictions):
            model_name = list(self.models.keys())[i]
            if model_name != 'autoencoder':
                weight = self.model_weights.get(model_name, 1.0 / len(predictions))
                ensemble_preds += weight * np.array(preds)
        
        # Calculate metrics
        ensemble_preds_binary = (ensemble_preds > 0.5).astype(int)
        accuracy = np.mean(ensemble_preds_binary == actuals)
        
        return {
            'accuracy': accuracy,
            'ensemble_predictions': ensemble_preds,
            'actual_values': actuals
        }
    
    def predict(self, df: pd.DataFrame, technical_indicators: Dict) -> Dict[str, Any]:
        """
        Generate predictions using the trained ensemble.
        
        Args:
            df: Recent market data
            technical_indicators: Technical indicator values
        
        Returns:
            Prediction dictionary with all required fields
        """
        if not self.models:
            logger.warning("Models not trained. Run train_ensemble() first.")
            return {}
        
        # Prepare features
        features = self._prepare_features(df, technical_indicators)
        
        if len(features) < 60:
            logger.warning("Insufficient data for prediction. Need at least 60 periods.")
            return {}
        
        # Create sequence
        sequence = features[-60:].reshape(1, 60, -1)
        
        # Scale features
        if 'main' in self.scalers:
            sequence_scaled = self.scalers['main'].transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)
        else:
            sequence_scaled = sequence
        
        # Get predictions from each model
        predictions = {}
        confidence_scores = {}
        
        for model_name, model in self.models.items():
            if model_name == 'autoencoder':
                # Autoencoder for anomaly detection
                model.eval()
                with torch.no_grad():
                    sequence_tensor = torch.FloatTensor(sequence_scaled).to(self.device)
                    sequence_flat = sequence_tensor.view(sequence_tensor.size(0), -1)
                    reconstructed = model(sequence_flat)
                    anomaly_score = torch.mean((sequence_flat - reconstructed) ** 2).item()
                    predictions[model_name] = anomaly_score
                    confidence_scores[model_name] = 1.0 / (1.0 + anomaly_score)  # Lower anomaly = higher confidence
            else:
                # Classification models
                model.eval()
                with torch.no_grad():
                    sequence_tensor = torch.FloatTensor(sequence_scaled).to(self.device)
                    outputs = model(sequence_tensor)
                    probs = torch.sigmoid(outputs)
                    predictions[model_name] = probs.item()
                    confidence_scores[model_name] = abs(probs.item() - 0.5) * 2  # Distance from 0.5
        
        # Calculate ensemble prediction
        ensemble_prob = 0.0
        total_weight = 0.0
        
        for model_name, pred in predictions.items():
            if model_name != 'autoencoder':
                weight = self.model_weights.get(model_name, 1.0 / len(predictions))
                ensemble_prob += weight * pred
                total_weight += weight
        
        if total_weight > 0:
            ensemble_prob /= total_weight
        
        # Determine prediction direction and confidence
        direction = "up" if ensemble_prob > 0.5 else "down"
        confidence = abs(ensemble_prob - 0.5) * 2
        
        # Calculate target price
        current_price = df['Close'].iloc[-1]
        avg_return = 0.02  # 2% average move
        target_price = current_price * (1 + avg_return) if direction == "up" else current_price * (1 - avg_return)
        
        # Risk assessment
        atr = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        stop_loss = current_price - (atr.iloc[-1] * 2) if direction == "up" else current_price + (atr.iloc[-1] * 2)
        
        # Feature attribution (simplified)
        feature_importance = self._calculate_feature_importance(sequence_scaled)
        
        return {
            'movement_direction': f"{confidence*100:.1f}% chance {direction}",
            'target_price': target_price,
            'confidence_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low',
            'confidence_percentage': confidence * 100,
            'time_horizon': '5 days',
            'risk_assessment': {
                'stop_loss': stop_loss,
                'risk_reward_ratio': abs(target_price - current_price) / abs(stop_loss - current_price),
                'max_loss': abs(stop_loss - current_price)
            },
            'feature_attribution': feature_importance,
            'model_predictions': predictions,
            'ensemble_probability': ensemble_prob,
            'anomaly_score': predictions.get('autoencoder', 0)
        }
    
    def _calculate_feature_importance(self, sequence: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for the prediction."""
        # Simplified feature importance calculation
        feature_names = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower',
            'ATR', 'Stochastic_K', 'Williams_R', 'CCI', 'ADX'
        ]
        
        # Use the last timestep for feature importance
        last_features = sequence[0, -1, :]
        
        # Normalize feature values
        feature_importance = {}
        for i, name in enumerate(feature_names[:len(last_features)]):
            feature_importance[name] = abs(last_features[i])
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:5])  # Top 5 features
    
    def save_models(self, filepath: str):
        """Save trained models."""
        try:
            model_states = {}
            for name, model in self.models.items():
                model_states[name] = model.state_dict()
            
            torch.save({
                'model_states': model_states,
                'model_weights': self.model_weights,
                'scalers': self.scalers,
                'training_history': self.training_history
            }, filepath)
            
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Recreate models
            self.models = {}
            features_dim = 20  # Default, should match training data
            
            self.models['lstm'] = LSTMModel(features_dim, self.config.LSTM_UNITS, self.config.LSTM_LAYERS)
            self.models['cnn'] = CNNModel(features_dim, self.config.CNN_FILTERS)
            self.models['transformer'] = TransformerModel(features_dim, 128, self.config.TRANSFORMER_HEADS, self.config.TRANSFORMER_LAYERS)
            self.models['autoencoder'] = AutoEncoderModel(features_dim, 32)
            
            # Load states
            for name, model in self.models.items():
                if name in checkpoint['model_states']:
                    model.load_state_dict(checkpoint['model_states'][name])
                    model.to(self.device)
            
            self.model_weights = checkpoint['model_weights']
            self.scalers = checkpoint['scalers']
            self.training_history = checkpoint['training_history']
            
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history for all models."""
        if not self.training_history:
            logger.warning("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (model_name, history) in enumerate(self.training_history.items()):
            if i >= 4:
                break
            
            ax = axes[i]
            
            if 'train_losses' in history:
                ax.plot(history['train_losses'], label='Train Loss')
            if 'val_losses' in history:
                ax.plot(history['val_losses'], label='Val Loss')
            
            ax.set_title(f'{model_name.upper()} Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test the deep learning ensemble
    from data.live_data_loader import DataLoader
    from technical_analysis.advanced_indicators import TechnicalAnalyzer
    
    print("Testing Deep Learning Ensemble...")
    
    # Load sample data
    loader = DataLoader()
    data = loader.fetch_single_timeframe_data(timeframe="1d")
    
    if data is not None and not data.empty:
        # Calculate technical indicators
        analyzer = TechnicalAnalyzer()
        indicators = analyzer.calculate_all_indicators(data)
        
        # Initialize ensemble
        ensemble = DeepLearningEnsemble()
        
        # Train ensemble (commented out for testing - takes time)
        # results = ensemble.train_ensemble(data, indicators)
        # print(f"Training results: {results}")
        
        # Generate prediction
        prediction = ensemble.predict(data, indicators)
        print(f"Prediction: {prediction}")
        
    else:
        print("No data available for testing")
