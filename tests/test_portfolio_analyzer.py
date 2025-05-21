import pytest
import numpy as np
import torch
from models.portfolio_analyzer import PortfolioAnalyzer, LSTMModel

@pytest.fixture
def analyzer():
    return PortfolioAnalyzer()

@pytest.fixture
def sample_historical_data():
    # Generate 100 days of sample data with 5 features
    np.random.seed(42)
    return np.random.randn(100, 5) * 0.1 + 1.0  # Mean return of 1.0 with 0.1 std

@pytest.fixture
def sample_allocations():
    return {
        "SOL": 0.4,
        "USDC": 0.3,
        "RAY": 0.2,
        "ORCA": 0.1
    }

@pytest.fixture
def sample_model():
    model = LSTMModel()
    model.eval()
    return model

def test_lstm_model_forward(sample_model):
    # Test model forward pass
    batch_size = 1
    seq_length = 30
    input_size = 5
    
    x = torch.randn(batch_size, seq_length, input_size)
    output = sample_model(x)
    
    assert output.shape == (batch_size, 3)  # [risk_score, expected_return, volatility]
    assert not torch.isnan(output).any()

def test_portfolio_analysis(analyzer, sample_historical_data, sample_allocations):
    risk_score, metrics = analyzer.analyze_portfolio(
        sample_historical_data,
        sample_allocations
    )
    
    # Check risk score is in valid range
    assert 0 <= risk_score <= 1
    
    # Check all required metrics are present
    required_metrics = {
        "risk_score",
        "expected_return",
        "volatility",
        "sharpe_ratio",
        "max_drawdown"
    }
    assert all(metric in metrics for metric in required_metrics)
    
    # Check metric values are reasonable
    assert -1 <= metrics["expected_return"] <= 1
    assert 0 <= metrics["volatility"] <= 1
    assert -5 <= metrics["sharpe_ratio"] <= 5
    assert -1 <= metrics["max_drawdown"] <= 0

def test_rebalancing_suggestion(analyzer, sample_allocations):
    current_risk = 0.7
    target_risk = 0.5
    
    new_allocations = analyzer.suggest_rebalancing(
        sample_allocations,
        current_risk,
        target_risk
    )
    
    # Check allocations sum to 1
    assert abs(sum(new_allocations.values()) - 1.0) < 1e-6
    
    # Check all allocations are positive
    assert all(v >= 0 for v in new_allocations.values())
    
    # Check all original assets are present
    assert set(new_allocations.keys()) == set(sample_allocations.keys())

def test_sharpe_ratio_calculation(analyzer):
    # Test with known values
    expected_return = 0.10  # 10% return
    volatility = 0.20      # 20% volatility
    risk_free_rate = 0.02  # 2% risk-free rate
    
    sharpe = analyzer._calculate_sharpe_ratio(
        expected_return,
        volatility,
        risk_free_rate
    )
    
    # Expected Sharpe ratio = (0.10 - 0.02) / 0.20 = 0.4
    assert abs(sharpe - 0.4) < 1e-6

def test_max_drawdown_calculation(analyzer):
    # Create a price series with a known drawdown
    prices = np.array([
        [1.0],  # Start
        [1.1],  # Peak
        [0.9],  # 18.18% drawdown from peak
        [1.0]   # Recovery
    ])
    
    max_dd = analyzer._calculate_max_drawdown(prices)
    
    # Expected max drawdown ≈ -0.1818
    assert -0.19 <= max_dd <= -0.18

def test_input_data_preparation(analyzer, sample_historical_data):
    prepared_data = analyzer._prepare_input_data(sample_historical_data)
    
    # Check shape is correct (batch_size=1, timesteps=30, features=5)
    assert prepared_data.shape == (1, 30, 5)
    
    # Check data is normalized
    data_numpy = prepared_data.cpu().numpy()
    assert -3 <= data_numpy.mean() <= 3  # Roughly normalized
    assert 0.1 <= data_numpy.std() <= 10

def test_model_save_load(analyzer, tmp_path):
    # Save model
    save_path = tmp_path / "test_model.pt"
    analyzer.save_model(str(save_path))
    
    # Load model in new analyzer
    new_analyzer = PortfolioAnalyzer(model_path=str(save_path))
    
    # Check both models produce similar outputs
    test_input = torch.randn(1, 30, 5)
    with torch.no_grad():
        output1 = analyzer.model(test_input)
        output2 = new_analyzer.model(test_input)
    
    assert torch.allclose(output1, output2, rtol=1e-4)

def test_error_handling(analyzer):
    with pytest.raises(ValueError):
        # Test with invalid input
        analyzer.analyze_portfolio(
            np.array([]),  # Empty array
            {}  # Empty allocations
        )
        
    with pytest.raises(Exception):
        # Test with mismatched dimensions
        analyzer.analyze_portfolio(
            np.random.randn(10, 3),  # Wrong number of features
            {"SOL": 1.0}
        )

def test_model_training(analyzer):
    # Generate dummy training data
    X = np.random.randn(100, 30, 5)  # 100 samples, 30 timesteps, 5 features
    y = np.random.randn(100, 3)      # 100 samples, 3 outputs
    
    # Train model
    analyzer.train_model(X, y, epochs=2)  # Just 2 epochs for testing
    
    # Check model is in eval mode after training
    assert not analyzer.model.training 