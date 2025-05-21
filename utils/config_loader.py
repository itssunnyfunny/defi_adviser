import yaml
import os
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Agent configuration container"""
    agent: Dict[str, Any]
    portfolio: Dict[str, Any]
    risk: Dict[str, Any]
    allocation: Dict[str, Any]
    protocols: Dict[str, Any]
    oracles: Dict[str, Any]
    monitoring: Dict[str, Any]
    security: Dict[str, Any]
    model: Dict[str, Any]

@dataclass
class PolicyConfig:
    """Policy configuration container"""
    transaction_policies: Dict[str, Any]
    protocol_policies: Dict[str, Any]
    risk_policies: Dict[str, Any]
    security_policies: Dict[str, Any]
    monitoring_policies: Dict[str, Any]
    emergency_policies: Dict[str, Any]

class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.agent_config: Optional[AgentConfig] = None
        self.policy_config: Optional[PolicyConfig] = None

    def load_configs(self) -> tuple[AgentConfig, PolicyConfig]:
        """Load all configuration files"""
        try:
            # Load agent configuration
            agent_config = self._load_yaml("agent_config.yaml")
            self.agent_config = AgentConfig(**agent_config)

            # Load policy configuration
            policy_config = self._load_yaml("policy_rules.yaml")
            self.policy_config = PolicyConfig(**policy_config)

            # Validate configurations
            self._validate_configs()

            return self.agent_config, self.policy_config

        except Exception as e:
            logger.error(f"Failed to load configurations: {str(e)}")
            raise

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load and parse YAML file"""
        try:
            file_path = self.config_dir / filename
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {str(e)}")
            raise

    def _validate_configs(self):
        """Validate configuration values and relationships"""
        if not self.agent_config or not self.policy_config:
            raise ValueError("Configurations not loaded")

        try:
            # Validate portfolio limits
            self._validate_portfolio_limits()

            # Validate risk parameters
            self._validate_risk_parameters()

            # Validate protocol settings
            self._validate_protocol_settings()

            # Validate security settings
            self._validate_security_settings()

            logger.info("Configuration validation successful")

        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

    def _validate_portfolio_limits(self):
        """Validate portfolio-related configuration values"""
        portfolio = self.agent_config.portfolio
        risk_policies = self.policy_config.risk_policies

        # Check position size limits
        if portfolio["min_position_size_usd"] >= portfolio["max_position_size_usd"]:
            raise ValueError("min_position_size must be less than max_position_size")

        # Check allocation limits
        allocation = self.agent_config.allocation
        if sum(allocation["asset_classes"].values()) != 1.0:
            raise ValueError("Asset class allocations must sum to 1.0")

        # Check risk limits consistency
        if portfolio["target_risk_score"] > risk_policies["portfolio_limits"]["max_drawdown"]:
            raise ValueError("Target risk score exceeds maximum drawdown limit")

    def _validate_risk_parameters(self):
        """Validate risk-related configuration values"""
        risk = self.agent_config.risk
        risk_policies = self.policy_config.risk_policies

        # Check risk thresholds
        if risk["max_drawdown"] > 1.0 or risk["max_drawdown"] < 0:
            raise ValueError("max_drawdown must be between 0 and 1")

        if risk["volatility_threshold"] <= 0:
            raise ValueError("volatility_threshold must be positive")

        # Check correlation limits
        if not 0 <= risk["correlation_threshold"] <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")

    def _validate_protocol_settings(self):
        """Validate protocol-related configuration values"""
        protocols = self.agent_config.protocols
        protocol_policies = self.policy_config.protocol_policies

        # Check protocol consistency
        for protocol in protocols:
            if protocol not in [p["name"] for p in protocol_policies["allowed_protocols"]]:
                raise ValueError(f"Protocol {protocol} not found in policy rules")

        # Validate protocol addresses
        for protocol in protocol_policies["allowed_protocols"]:
            if not protocol["contract_address"]:
                raise ValueError(f"Missing contract address for {protocol['name']}")

    def _validate_security_settings(self):
        """Validate security-related configuration values"""
        security = self.agent_config.security
        security_policies = self.policy_config.security_policies

        # Check transaction limits
        if security["max_transaction_value"] > security_policies["authentication"]["multi_sig_threshold"]:
            raise ValueError("max_transaction_value exceeds multi-sig threshold")

        # Validate network settings
        if not security_policies["network"]["allowed_networks"]:
            raise ValueError("No allowed networks specified")

    def get_protocol_config(self, protocol_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific protocol"""
        if not self.agent_config or not self.policy_config:
            raise ValueError("Configurations not loaded")

        # Get protocol settings
        protocol_config = self.agent_config.protocols.get(protocol_name.lower())
        if not protocol_config:
            return None

        # Get protocol policies
        protocol_policies = next(
            (p for p in self.policy_config.protocol_policies["allowed_protocols"]
             if p["name"].lower() == protocol_name.lower()),
            None
        )

        if not protocol_policies:
            return None

        # Combine settings and policies
        return {
            "settings": protocol_config,
            "policies": protocol_policies
        }

    def get_risk_limits(self) -> Dict[str, float]:
        """Get combined risk limits from agent and policy configs"""
        if not self.agent_config or not self.policy_config:
            raise ValueError("Configurations not loaded")

        return {
            "max_drawdown": min(
                self.agent_config.risk["max_drawdown"],
                self.policy_config.risk_policies["portfolio_limits"]["max_drawdown"]
            ),
            "volatility_threshold": min(
                self.agent_config.risk["volatility_threshold"],
                self.policy_config.risk_policies["volatility_controls"]["portfolio_volatility_limit"]
            ),
            "max_single_asset": min(
                self.agent_config.allocation["max_single_asset"],
                self.policy_config.transaction_policies["asset_rules"]["max_allocation_per_token"]
            )
        }

    def update_config_value(self, config_type: str, path: list[str], value: Any):
        """Update a specific configuration value"""
        if not self.agent_config or not self.policy_config:
            raise ValueError("Configurations not loaded")

        try:
            # Select configuration object
            config = self.agent_config if config_type == "agent" else self.policy_config

            # Navigate to the target dictionary
            target = config
            for key in path[:-1]:
                target = target[key]

            # Update value
            target[path[-1]] = value

            # Validate updated configurations
            self._validate_configs()

            # Save updated configuration
            self._save_config(config_type)

            logger.info(f"Updated configuration value: {'.'.join(path)} = {value}")

        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            raise

    def _save_config(self, config_type: str):
        """Save configuration to file"""
        try:
            filename = "agent_config.yaml" if config_type == "agent" else "policy_rules.yaml"
            config = self.agent_config if config_type == "agent" else self.policy_config

            with open(self.config_dir / filename, 'w') as f:
                yaml.safe_dump(config.__dict__, f, default_flow_style=False)

        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise 