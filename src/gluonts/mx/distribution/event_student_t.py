# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Dict, List, Optional, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .deterministic import DeterministicOutput
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput
from .mixture import MixtureDistributionOutput

from .student_t import StudentT


class EventStudentT(StudentT):
    r"""


    """

    is_reparameterizable = False

    @validated()
    # def __init__(self, theta_bar: Tensor, **kwargs) -> None:
    #     print(kwargs)
    #     super().__init__(**kwargs)
    # def __init__(self, theta_bar, *args) -> None:
    #     print(args)
    #     super().__init__(*args)
    #     self.theta_bar = theta_bar
    # TODO: proper initiation with args and kwargs
    def __init__(self, theta_bar: Tensor, mu: Tensor, sigma: Tensor, nu: Tensor, F=None) -> None:
        super().__init__(mu, sigma, nu)
        self.theta_bar = theta_bar

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        ll_event = F.log(self.theta_bar) + super().log_prob(x)
        ll_no_event = F.log(
            (1 - self.theta_bar) + self.theta_bar * (F.exp(super().log_prob(x)))
        )
        events = (x != 0)
        ll = (1 - events) * ll_no_event + events * ll_event
        return ll

    @property
    def mean(self) -> Tensor:
        # return (1 - self.theta) * self.mu
        return self.theta_bar * super().mean

    @property
    def stddev(self) -> Tensor:
        # TODO: using conditional variance Var[Y] = V[E[Y|X]] + E[V[Y|X]]
        # TODO: sqrt and square?
        # return self.F.sqrt(self.mu * (1.0 + self.mu * self.alpha))
        return (
                (1 - self.theta) * (self.mean ** 2) + self.theta * ((1 - self.theta)**2) * (super().mean ** 2)
                + self.theta * (super().stddev**2)
        )

    def one_sample(self, *args):
        theta_bar = args[0]
        args_ = args[1:]
        F = self.F
        events = (F.random.uniform(shape=self.mean.shape) < theta_bar)
        return events * super().one_sample(*args_)
    def sample(
            self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        args = tuple(self.args)  + (dtype, )
        return _sample_multiple(
            self.one_sample, *args, num_samples=num_samples
        )


    @property
    def args(self) -> List:
        # print(super().args)
        return [self.theta_bar] +  super().args



class EventStudentTOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"theta_bar":1, "mu": 1, "sigma": 1, "nu": 1}
    distr_cls: type = EventStudentT

    @classmethod
    def domain_map(cls, F, theta_bar, mu, sigma, nu):
        theta_bar = F.sigmoid(theta_bar) # F.maximum(F.sigmoid(theta), epsilon)
        sigma = softplus(F, sigma)
        nu = 2.0 + softplus(F, nu)
        return theta_bar.squeeze(axis=-1), mu.squeeze(axis=-1), sigma.squeeze(axis=-1), nu.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
