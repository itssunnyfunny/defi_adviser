import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
import json
import os

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # [risk_score, expected_return, volatility]
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc_layers(last_hidden)

class PortfolioAnalyzer:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            try:
                self._load_model(model_path)
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                self._build_default_model()
        else:
            self._build_default_model()

    def _build_default_model(self):
        """Build a simple LSTM model for portfolio analysis"""
        self.model = LSTMModel().to(self.device)
        self.model.eval()  # Set to evaluation mode
        logger.info("Built default portfolio analysis model")

    def _load_model(self, model_path: str):
        """Load model from file"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = LSTMModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler if available
        if 'scaler_params' in checkpoint:
            self.scaler.mean_ = checkpoint['scaler_params']['mean']
            self.scaler.scale_ = checkpoint['scaler_params']['scale']

    def analyze_portfolio(
        self,
        historical_data: np.ndarray,
        current_allocations: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Analyze portfolio performance and risk
        
        Args:
            historical_data: Array of shape (n_days, n_features) with price/volume data
            current_allocations: Dictionary of asset -> allocation percentage
            
        Returns:
            Tuple of (risk_score, metrics_dict)
        """
        try:
            # Validate input
            if historical_data.size == 0 or not current_allocations:
                raise ValueError("Invalid input data")
            
            # Prepare input data
            x = self._prepare_input_data(historical_data)
            
            # Get model predictions
            with torch.no_grad():
                predictions = self.model(x)
                predictions = predictions.cpu().numpy()[0]
            
            # Calculate portfolio metrics
            risk_score = float(predictions[0])
            expected_return = float(predictions[1])
            volatility = float(predictions[2])
            
            # Calculate additional metrics
            sharpe_ratio = self._calculate_sharpe_ratio(expected_return, volatility)
            max_drawdown = self._calculate_max_drawdown(historical_data)
            
            metrics = {
                "risk_score": risk_score,
                "expected_return": expected_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown
            }
            
            return risk_score, metrics
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {str(e)}")
            return 0.5, {}  # Return neutral risk score on failure

    def _prepare_input_data(self, historical_data: np.ndarray) -> torch.Tensor:
        """Prepare historical data for model input"""
        # Normalize data
        normalized_data = self.scaler.fit_transform(historical_data)
        
        # Get last 30 days and reshape for LSTM input
        sequence = normalized_data[-30:]
        sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        return sequence.to(self.device)

    def _calculate_sharpe_ratio(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        return (expected_return - risk_free_rate) / volatility if volatility > 0 else 0

    def _calculate_max_drawdown(self, historical_data: np.ndarray) -> float:
        """Calculate maximum drawdown from historical data"""
        cumulative_returns = np.cumprod(1 + historical_data[:, 0])  # Using first column as returns
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        return float(np.min(drawdowns))

    def suggest_rebalancing(
        self,
        current_allocations: Dict[str, float],
        risk_score: float,
        target_risk: float
    ) -> Dict[str, float]:
        """
        Suggest portfolio rebalancing based on risk analysis
        
        Args:
            current_allocations: Current portfolio allocations
            risk_score: Current portfolio risk score
            target_risk: Target risk score
            
        Returns:
            Dictionary of suggested allocations
        """
        try:
            # Simple rebalancing strategy - adjust based on risk difference
            risk_diff = target_risk - risk_score
            
            # Sort assets by volatility (assuming more volatile assets are riskier)
            sorted_assets = sorted(
                current_allocations.items(),
                key=lambda x: x[1],  # Sort by current allocation as proxy for risk
                reverse=True
            )
            
            # Adjust allocations
            new_allocations = current_allocations.copy()
            
            if risk_diff > 0:  # Need to increase risk
                # Increase allocation to riskier assets
                for asset, _ in sorted_assets[:2]:  # Top 2 riskiest assets
                    new_allocations[asset] *= (1 + risk_diff * 0.1)
            else:  # Need to decrease risk
                # Increase allocation to safer assets
                for asset, _ in sorted_assets[-2:]:  # Top 2 safest assets
                    new_allocations[asset] *= (1 - risk_diff * 0.1)
                    
            # Normalize allocations to sum to 1
            total = sum(new_allocations.values())
            new_allocations = {k: v/total for k, v in new_allocations.items()}
            
            return new_allocations
            
        except Exception as e:
            logger.error(f"Rebalancing calculation failed: {str(e)}")
            return current_allocations  # Return current allocations on failure

    def save_model(self, save_path: str):
        """Save model and scaler parameters"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'scaler_params': {
                    'mean': self.scaler.mean_,
                    'scale': self.scaler.scale_
                }
            }
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")

    def train_model(
        self,
        training_data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """Train the model on new data"""
        try:
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Prepare training data
            X = torch.FloatTensor(training_data).to(self.device)
            y = torch.FloatTensor(labels).to(self.device)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
            self.model.eval()
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise 