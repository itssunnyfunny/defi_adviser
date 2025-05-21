# DeFi Advisor AI Agent

An autonomous AI-powered DeFi advisor built on Solana using Fetch.ai's uAgents framework. This agent helps users optimize their DeFi portfolio through automated analysis, rebalancing, and risk management.

## Features

- 🤖 Autonomous agent-based architecture using Fetch.ai uAgents
- 📊 Real-time portfolio analysis and rebalancing
- 🔗 Integration with Solana DeFi protocols (Jupiter, Raydium, Orca)
- 🧠 AI/ML-powered decision making with LSTM models
- 🔒 Policy-controlled secure wallet management
- 📈 Real-time monitoring and alerts via Telegram

## Prerequisites

- Python 3.9+
- Solana CLI tools
- Node.js 16+ (for web3 interactions)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/defi_advisor.git
cd defi_advisor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env-example .env
# Edit .env with your configuration
```

## Configuration

The agent can be configured through:
- Environment variables (.env file)
- `config/agent_config.yaml` - Agent behavior settings
- `config/policy_rules.yaml` - Security and risk management policies

### Key Configuration Parameters

- `RISK_THRESHOLD`: Target risk score (0-1)
- `REBALANCE_THRESHOLD`: Portfolio deviation trigger
- `MAX_TRANSACTION_VALUE_USD`: Maximum single transaction value
- `DAILY_TRANSACTION_LIMIT_USD`: Maximum daily trading volume

## Usage

1. Start the agent:
```bash
python main.py
```

2. Monitor metrics (Prometheus):
```bash
curl localhost:9090/metrics
```

3. View logs:
```bash
tail -f defi_advisor.log
```

## Architecture

### Components

1. **Agent Framework (`main.py`)**
   - Core agent implementation
   - Portfolio monitoring and rebalancing
   - Event handling and decision making

2. **Portfolio Analysis (`models/portfolio_analyzer.py`)**
   - LSTM model for risk assessment
   - Performance metrics calculation
   - Rebalancing suggestions

3. **Blockchain Integration (`protocols/solana_client.py`)**
   - Solana RPC interactions
   - Token account management
   - Transaction handling

4. **Monitoring (`utils/monitoring.py`)**
   - Prometheus metrics
   - Telegram notifications
   - Performance tracking

### Security Features

- Policy-driven transaction limits
- Multi-signature support for large transactions
- Circuit breakers for market volatility
- Continuous monitoring and alerts

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Features

1. Create feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Implement changes following project structure
3. Add tests
4. Submit pull request

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security

- Uses policy-controlled wallets
- Implements rate limiting and circuit breakers
- Regular security audits
- Monitoring for suspicious activities

## License

MIT License - see LICENSE file for details

## Support

For support, please:
1. Check the documentation
2. Open an issue
3. Join our community chat 