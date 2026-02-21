"""
MSNN (Multi-Stage Neural Networks) configuration for DIFFICE X-PINNs.

Based on: Wang & Lai (2024), J. Comput. Phys. 504, 112865
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MSNNConfig:
    """Configuration for multi-stage neural network training.
    
    Attributes:
        n_stages: Number of correction stages (Stage 0 is the baseline PINN).
                  Total stages = 1 + n_stages.
        stage_epochs: Adam epochs per correction stage.
                      Length should equal n_stages.
        use_lbfgs: Whether to run L-BFGS after Adam per stage.
                   Length should equal n_stages. Recommended: only for Stage 0.
        lbfgs_epochs: L-BFGS epochs per stage (when use_lbfgs[k] is True).
        correction_n_hl: Number of hidden layers in correction networks.
        correction_n_unit: Number of units per hidden layer in correction networks.
        kappa_multiplier: Safety factor for scale factor: κ = multiplier * π * f_d.
        use_sgd_resampling: Whether to re-sample collocation points during
                            higher-stage Adam training (beneficial for high-freq).
        resample_interval: Re-sample every N Adam iterations.
        pretrained_params_path: Optional path to .pkl file with pre-trained
                                Stage 0 params. If provided, Stage 0 training
                                is skipped entirely.
    """
    n_stages: int = 1
    stage_epochs: Optional[List[int]] = None
    use_lbfgs: Optional[List[bool]] = None
    lbfgs_epochs: int = 10000
    correction_n_hl: int = 2
    correction_n_unit: int = 30
    kappa_multiplier: float = 1.2
    use_sgd_resampling: bool = True
    resample_interval: int = 100
    pretrained_params_path: Optional[str] = None

    def __post_init__(self):
        if self.stage_epochs is None:
            self.stage_epochs = [30000] * self.n_stages
        if self.use_lbfgs is None:
            self.use_lbfgs = [False] * self.n_stages
        assert len(self.stage_epochs) == self.n_stages, \
            f"stage_epochs length ({len(self.stage_epochs)}) != n_stages ({self.n_stages})"
        assert len(self.use_lbfgs) == self.n_stages, \
            f"use_lbfgs length ({len(self.use_lbfgs)}) != n_stages ({self.n_stages})"
