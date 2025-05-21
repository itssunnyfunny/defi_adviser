from solana.rpc.api import Client
from solana.transaction import Transaction
from solana.system_program import TransactionInstruction
from typing import Dict, List, Optional
import logging
import os
from base58 import b58encode, b58decode

logger = logging.getLogger(__name__)

class SolanaClient:
    def __init__(self):
        self.client = Client(os.getenv("SOLANA_RPC_URL"))
        self.wallet_private_key = b58decode(os.getenv("SOLANA_WALLET_PRIVATE_KEY"))
        
    async def get_token_accounts(self, owner_pubkey: str) -> List[Dict]:
        """Get all token accounts for a wallet address"""
        try:
            response = await self.client.get_token_accounts_by_owner(
                owner_pubkey,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}
            )
            
            if "result" in response and "value" in response["result"]:
                return response["result"]["value"]
            return []
            
        except Exception as e:
            logger.error(f"Failed to get token accounts: {str(e)}")
            return []

    async def get_token_balance(self, account_pubkey: str) -> Optional[float]:
        """Get balance for a specific token account"""
        try:
            response = await self.client.get_token_account_balance(account_pubkey)
            
            if "result" in response and "value" in response["result"]:
                return float(response["result"]["value"]["uiAmount"])
            return None
            
        except Exception as e:
            logger.error(f"Failed to get token balance: {str(e)}")
            return None

    async def get_pool_info(self, pool_address: str) -> Optional[Dict]:
        """Get information about a liquidity pool"""
        try:
            response = await self.client.get_account_info(pool_address)
            
            if "result" in response and "value" in response["result"]:
                # Parse pool data based on specific DEX format
                # This is a simplified example - actual implementation would need
                # to handle specific pool layouts for Raydium, Orca, etc.
                return {
                    "data": response["result"]["value"]["data"],
                    "lamports": response["result"]["value"]["lamports"],
                    "owner": response["result"]["value"]["owner"]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get pool info: {str(e)}")
            return None

    async def submit_transaction(
        self,
        instructions: List[TransactionInstruction],
        signers: List[bytes]
    ) -> Optional[str]:
        """Submit a transaction to the Solana network"""
        try:
            transaction = Transaction()
            
            # Add recent blockhash
            recent_blockhash = await self.client.get_recent_blockhash()
            transaction.recent_blockhash = recent_blockhash["result"]["value"]["blockhash"]
            
            # Add instructions
            for instruction in instructions:
                transaction.add(instruction)
            
            # Sign transaction
            transaction.sign(*signers)
            
            # Send transaction
            result = await self.client.send_transaction(
                transaction,
                *signers,
                opts={"skip_preflight": False, "preflightCommitment": "confirmed"}
            )
            
            if "result" in result:
                return result["result"]
            return None
            
        except Exception as e:
            logger.error(f"Failed to submit transaction: {str(e)}")
            return None

    async def get_token_price(self, token_mint: str) -> Optional[float]:
        """Get token price from Pyth oracle"""
        try:
            # This is a simplified example - actual implementation would need
            # to handle Pyth program calls and price feed accounts
            response = await self.client.get_account_info(token_mint)
            
            if "result" in response and "value" in response["result"]:
                # Parse Pyth price data
                # Actual implementation would decode the specific Pyth account layout
                return 0.0  # Placeholder
            return None
            
        except Exception as e:
            logger.error(f"Failed to get token price: {str(e)}")
            return None

    def create_swap_instruction(
        self,
        from_token: str,
        to_token: str,
        amount: float,
        slippage: float = 0.01
    ) -> Optional[TransactionInstruction]:
        """Create instruction for token swap"""
        try:
            # This is a simplified example - actual implementation would need
            # to handle specific DEX instruction creation (Jupiter, Raydium, etc.)
            instruction_data = {
                "from_token": from_token,
                "to_token": to_token,
                "amount": amount,
                "slippage": slippage
            }
            
            # Create and return instruction
            # Actual implementation would construct proper Solana instruction
            return TransactionInstruction(
                keys=[],  # Add proper account metas
                program_id=b"",  # Add proper program ID
                data=b""  # Add proper instruction data
            )
            
        except Exception as e:
            logger.error(f"Failed to create swap instruction: {str(e)}")
            return None 