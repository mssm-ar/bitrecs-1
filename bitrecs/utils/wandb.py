import wandb
import bittensor as bt
from typing import Dict, Any, Optional

class WandbHelper:
    def __init__(
        self,
        project_name: str,
        entity: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None
    ):
        """
        Initialize WandB tracking
        """
        self.default_config = {
            "network": "testnet",
            "neuron_type": "validator",
            "sample_size": 5,
            "num_concurrent_forwards": 1,
            "vpermit_tao_limit": 1024,
            "run_name": f"validator_{wandb.util.generate_id()}"
        }
        
        if config:
            self.default_config.update(config)

        try:
            self.run = wandb.init(
                project=project_name,
                entity=entity,
                config=self.default_config,
                tags=tags,
                reinit=True
            )
        except Exception as e:
            bt.logging.error(f"Error initializing wandb: {e}")
            self.run = None
        
    def log_weights(self, step: int, weights: Dict[str, float], prefix: str = "weights"):
        """
        Log weight updates to wandb
        """
        try:

            metrics = {f"{prefix}/{k}": v for k, v in weights.items()}
            metrics["step"] = step
            wandb.log(metrics)
        except Exception as e:
            bt.logging.error(f"Error logging weights to wandb: {e}")

    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log arbitrary metrics to wandb
        """
        try:
            wandb.log(metrics)
        except Exception as e:
            bt.logging.error(f"Error logging metrics to wandb: {e}")
    
    def finish(self):
        """
        Close wandb run
        """
        if self.run:
            self.run.finish()