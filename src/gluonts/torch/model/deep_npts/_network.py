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

from functools import partial
from typing import Optional, Callable, List, Union

import torch
from torch import nn
from torch.distributions import (
    Categorical,
    MixtureSameFamily,
    Normal,
)
from gluonts.core.component import validated

from gluonts.torch.distributions.discrete_distribution import (
    DiscreteDistribution,
)
from .scaling import (
    min_max_scaling,
    standard_normal_scaling,
)

INPUT_SCALING_MAP = {
    "min_max_scaling": partial(min_max_scaling, dim=1, keepdim=True),
    "standard_normal_scaling": partial(
        standard_normal_scaling, dim=1, keepdim=True
    ),
}


def init_weights(module: nn.Module, scale: float = 1.0):
    if type(module) == nn.Linear:
        nn.init.uniform_(module.weight, -scale, scale)
        nn.init.zeros_(module.bias)


class FeatureEmbedder(nn.Module):
    @validated()
    def __init__(
        self,
        cardinalities: List[int],
        embedding_dimensions: List[int],
    ):
        super().__init__()

        assert (
            len(cardinalities) > 0
        ), "Length of `cardinalities` list must be greater than zero"
        assert len(cardinalities) == len(
            embedding_dimensions
        ), "Length of `embedding_dims` and `embedding_dims` should match"
        assert all(
            [c > 0 for c in cardinalities]
        ), "Elements of `cardinalities` should be > 0"
        assert all(
            [d > 0 for d in embedding_dimensions]
        ), "Elements of `embedding_dims` should be > 0"

        self.embedders = [
            torch.nn.Embedding(num_embeddings=card, embedding_dim=dim)
            for card, dim in zip(cardinalities, embedding_dimensions)
        ]
        for embedder in self.embedders:
            embedder.apply(init_weights)

    def forward(self, features):
        """

        :param features: (-1, num_features)
        :return:
            Embedding with shape (-1, sum([self.embedding_dimensions]))
        """
        embedded_features = torch.cat(
            [
                embedder(features[:, i].long())
                for i, embedder in enumerate(self.embedders)
            ],
            dim=-1,
        )
        return embedded_features


class DeepNPTSNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        context_length: int,
        num_hidden_nodes: List[int],
        cardinality: List[int],
        embedding_dimension: List[int],
        num_time_features: int,
        batch_norm: bool = False,
        input_scaling: Optional[Union[Callable, str]] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.context_length = context_length
        self.num_hidden_nodes = num_hidden_nodes
        self.batch_norm = batch_norm
        self.input_scaling = (
            INPUT_SCALING_MAP[input_scaling]
            if isinstance(input_scaling, str)
            else input_scaling
        )
        self.dropout = dropout

        # Embedding for categorical features
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality, embedding_dimensions=embedding_dimension
        )
        total_embedding_dim = sum(embedding_dimension)

        # We have two target related features: past_target and observed value indicator each of length `context_length`.
        # Also, +1 for the static real feature.
        dimensions = [
            context_length * (num_time_features + 2) + total_embedding_dim + 1
        ] + num_hidden_nodes
        modules = []
        for in_features, out_features in zip(dimensions[:-1], dimensions[1:]):
            if self.dropout:
                modules.append(nn.Dropout(self.dropout))
            modules += [nn.Linear(in_features, out_features), nn.ReLU()]
            if self.batch_norm:
                modules.append(nn.BatchNorm1d(in_features))

        self.model = nn.Sequential(*modules)
        self.model.apply(partial(init_weights, scale=0.07))

    def forward(
        self,
        feat_static_cat,
        feat_static_real,
        past_target,
        past_observed_values,
        past_time_feat,
    ):
        """
        TODO: Handle missing values using the observed value indicator.

        :param feat_static_cat: shape (-1, num_features)
        :param feat_static_real: shape (-1, num_features)
        :param past_target: shape (-1, context_length)
        :param past_observed_values: shape (-1, context_length)
        :param past_time_feat: shape (-1, context_length, self.num_time_features)
        :return:
        """
        x = past_target
        if self.input_scaling:
            loc, scale = self.input_scaling(x)
            x_scaled = (x - loc) / scale
        else:
            x_scaled = x

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, torch.tensor(feat_static_real, requires_grad=True)),
            dim=1,
        )
        time_features = torch.cat(
            [
                x_scaled.unsqueeze(dim=-1),
                past_observed_values.unsqueeze(dim=-1),
                past_time_feat,
            ],
            dim=-1,
        )

        features = torch.cat(
            [
                time_features.reshape(time_features.shape[0], -1),
                static_feat,
            ],
            dim=-1,
        )
        return self.model(features)


class DeepNPTSNetworkSmooth(DeepNPTSNetwork):
    @validated()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        modules = [] if self.dropout is None else [nn.Dropout(self.dropout)]
        modules += [
            nn.Linear(self.num_hidden_nodes[-1], self.context_length + 1),
            nn.Softplus(),
        ]
        self.output_layer = nn.Sequential(*modules)
        self.output_layer.apply(init_weights)

    def forward(
        self,
        feat_static_cat,
        feat_static_real,
        past_target,
        past_observed_values,
        past_time_feat,
    ):
        h = super().forward(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
        )
        outputs = self.output_layer(h)
        probs = outputs[:, :-1]
        kernel_width = outputs[:, -1:]
        mix = Categorical(probs)
        components = Normal(loc=past_target, scale=kernel_width)
        return MixtureSameFamily(
            mixture_distribution=mix, component_distribution=components
        )


class DeepNPTSNetworkDiscrete(DeepNPTSNetwork):
    @validated()
    def __init__(self, *args, use_softmax: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_softmax = use_softmax
        modules = [] if self.dropout is None else [nn.Dropout(self.dropout)]
        modules.append(
            nn.Linear(self.num_hidden_nodes[-1], self.context_length)
        )
        self.output_layer = nn.Sequential(*modules)
        self.output_layer.apply(init_weights)

    def forward(
        self,
        feat_static_cat,
        feat_static_real,
        past_target,
        past_observed_values,
        past_time_feat,
    ):
        h = super().forward(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
        )
        outputs = self.output_layer(h)
        probs = (
            nn.functional.softmax(outputs, dim=1)
            if self.use_softmax
            else nn.functional.normalize(
                nn.functional.softplus(outputs), p=1, dim=1
            )
        )
        return DiscreteDistribution(values=past_target, probs=probs)


class DeepNPTSMultiStepPredictor(nn.Module):
    @validated()
    def __init__(
        self,
        net: DeepNPTSNetwork,
        prediction_length: int,
        num_parallel_samples: int = 100,
    ):
        super().__init__()
        self.net = net
        self.prediction_length = prediction_length
        self.num_parallel_samples = num_parallel_samples

    def forward(
        self,
        feat_static_cat,
        feat_static_real,
        past_target,
        past_observed_values,
        past_time_feat,
        future_time_feat,
    ):
        """
        Generates samples from the forecast distribution.

        :param feat_static_cat: shape (-1, num_features)
        :param feat_static_real: shape (-1, num_features)
        :param past_target: shape (-1, context_length)
        :param past_observed_values: shape (-1, context_length)
        :param past_time_feat: shape (-1, context_length, self.num_time_features)
        :param future_time_feat: shape (-1, prediction_length, self.num_time_features)
        :return:
        :return:
        """
        # Blow up the initial `x` by the number of parallel samples required.
        # (batch_size * num_parallel_samples, context_length)
        past_target = past_target.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        # Note that gluonts returns empty future_observed_values.
        future_observed_values = torch.ones(
            (past_observed_values.shape[0], self.prediction_length)
        )
        observed_values = torch.cat(
            [past_observed_values, future_observed_values], dim=1
        )
        observed_values = observed_values.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        time_feat = torch.cat([past_time_feat, future_time_feat], dim=1)
        time_feat = time_feat.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        feat_static_cat = feat_static_cat.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        feat_static_real = feat_static_real.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        future_samples = []
        for t in range(self.prediction_length):
            distr = self.net(
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                past_target=past_target,
                past_observed_values=observed_values[
                    :, t : -self.prediction_length + t
                ],
                past_time_feat=time_feat[
                    :, t : -self.prediction_length + t, :
                ],
            )
            samples = distr.sample()
            if past_target.dim() != samples.dim():
                samples = samples.unsqueeze(dim=-1)

            future_samples.append(samples)
            past_target = torch.cat([past_target[:, 1:], samples], dim=1)

        # (batch_size * num_parallel_samples, prediction_length)
        samples_out = torch.stack(future_samples, dim=1)

        # (batch_size, num_parallel_samples, prediction_length)
        return samples_out.reshape(
            -1, self.num_parallel_samples, self.prediction_length
        )
