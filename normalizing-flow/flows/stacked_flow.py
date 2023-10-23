from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from .nf_utils import Flow


class StackedFlows(nn.Module):
    """Stack a list of transformations with a given based distribtuion.

    Args:
        transforms: list fo stacked transformations. list of Flows
        dim: dimension of input/output data. int
        base_dist: name of the base distribution. options: ['Normal']
    """

    def __init__(
        self,
        transforms: List[Flow],
        dim: int = 2,
        base_dist: str = "Normal",
        device="cpu",
    ):
        super().__init__()

        if isinstance(transforms, Flow):
            self.transforms = nn.ModuleList(
                [
                    transforms,
                ]
            )
        elif isinstance(transforms, list):
            if not all(isinstance(t, Flow) for t in transforms):
                raise ValueError("transforms must be a Flow or a list of Flows")
            self.transforms = nn.ModuleList(transforms)
        else:
            raise ValueError(
                f"transforms must a Flow or a list, but was {type(transforms)}"
            )

        self.dim = dim
        if base_dist == "Normal":
            self.base_dist = MultivariateNormal(
                torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device)
            )
        else:
            raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of a batch of data (slide 27).

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            log_prob: Log probability of the data, shape [batch_size]
        """

        B, D = x.shape

        ##########################################################
        # YOUR CODE HERE
        z = x
        log_det_jac_sum = 0
        for transform in reversed(self.transforms):
            z, log_det_jac = transform.inverse(z)
            log_det_jac_sum += log_det_jac

        log_prob = self.base_dist.log_prob(z) + log_det_jac_sum
        ##########################################################

        assert log_prob.shape == (B,)

        return log_prob

    def rsample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from the transformed distribution (slide 31).

        Returns:
            x: sample after forward transformation, shape [batch_size, dim]
            log_prob: Log probability of x, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE
        z_0 = self.base_dist.rsample((batch_size,))
        z = z_0
        log_det_jac_rcp_sum = 0
        for transform in self.transforms:
            z, log_det_jac = transform.forward(z)
            log_det_jac_rcp_sum -= log_det_jac

        x = z
        log_prob = self.base_dist.log_prob(z_0) + log_det_jac_rcp_sum
        ##########################################################

        assert x.shape == (batch_size, self.dim)
        assert log_prob.shape == (batch_size,)

        return x, log_prob
