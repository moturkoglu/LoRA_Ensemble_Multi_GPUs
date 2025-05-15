#!/usr/bin/env python

"""
Implements a Low Rank Adaptation module
"""

### IMPORTS ###
# Built-in imports
from typing import List, Dict
import enum

# Lib imports
from torch import nn, Tensor, vmap
from torch.func import stack_module_state, functional_call
import copy
import torch

# Custom imports
from utils_GPU import DEVICE


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### CLASS DEFINITION ###
class Init_Weight(enum.Enum):
    NORMAL = 0
    KAIMING_UNIFORM = 1
    XAVIER_UNIFORM = 2
    DEFAULT = NORMAL


class LoRA(nn.Module):
    def __init__(
            self,
            w: nn.Module,
            rank: int,
            dim: int,
            initialize: bool = True,
            init_type: Init_Weight = Init_Weight.DEFAULT,
            init_settings: dict = None,
            out_dim: int = None
    ) -> None:
        """
        Implements a Low Rank Adaptation module

        Parameters
        ----------
        w : nn.Module
            The original projection layer
        rank : int
            The rank of the Low Rank Adaptation module
        dim : List[int]
            The dimension of the weight matrix
        initialize : bool, optional
            Whether to initialize the weights, by default True
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        out_dim : int, optional
            The output dimension of the LoRA layer, by default None
        """

        super().__init__()

        # LoRA rank
        self.rank = rank

        # Original projection layer weights
        self.w = w

        if out_dim is None:
            out_dim = dim

        # LoRA matrices
        self.w_a = nn.Linear(dim, rank, bias=False)
        self.w_b = nn.Linear(rank, out_dim, bias=False)

        # Initialize the weights if needed
        if initialize:
            self.initialize_weights(init_type=init_type, init_settings=init_settings)

    def forward(self, x):
        """
        Forward pass for the LoRA Layer

        Parameters
        ----------
        x : Tensor
            The input tensor

        Returns
        -------
        out : Tensor
            The output tensor
        """

        out = self.w(x) + self.w_b(self.w_a(x))

        return out

    def initialize_weights(self, init_type: Init_Weight = Init_Weight.DEFAULT, init_settings: dict = None) -> None:
        """
        Initialize the weights of the LoRA matrices

        Parameters
        ----------
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        """

        if init_type == Init_Weight.NORMAL:
            # Set the mean and standard deviation
            if init_settings is None:
                mean = 0
                std = 0.02
            else:
                mean = init_settings["mean"]
                std = init_settings["std"]

            # Initialize all weights separately
            nn.init.normal_(self.w_a.weight, mean=mean, std=std)
            nn.init.zeros_(self.w_b.weight)

        elif init_type == Init_Weight.KAIMING_UNIFORM:
            # Set the a_squared parameter
            if init_settings is None:
                a_squared = 5
            else:
                a_squared = init_settings["a_squared"]

            # Initialize all weights separately
            from math import sqrt
            nn.init.kaiming_uniform_(self.w_a.weight, a=sqrt(a_squared))
            nn.init.zeros_(self.w_b.weight)

        elif init_type == Init_Weight.XAVIER_UNIFORM:
            # Set the a_squared parameter
            if init_settings is None:
                gain  = 1
            else:
                gain = init_settings["gain"]

            # Initialize all weights separately
            nn.init.xavier_uniform_(self.w_a.weight, gain=gain)
            nn.init.zeros_(self.w_b.weight)
        else:
            raise ValueError("Invalid initialization type")


# class EnsembleLoRA(nn.Module):
#     """
#     Class to ensemble the LoRA layers

#     Credit
#     ------
#     https://pytorch.org/tutorials/intermediate/ensembling.html
#     """

#     def __init__(
#             self,
#             w: nn.Module,
#             rank: int,
#             dim: int,
#             n_members: int,
#             initialize: bool = True,
#             init_type: Init_Weight = Init_Weight.DEFAULT,
#             init_settings: dict = None,
#             out_dim: int = None,
#             chunk_size: int = None
#     ):
#         """
#         Class to ensemble the LoRA layers

#         Parameters
#         ----------
#         w : nn.Module
#             The original projection layer
#         rank : int
#             The rank of the Low Rank Adaptation module
#         dim : List[int]
#             The dimension of the weight matrix
#         n_members : int
#             The number of ensemble members
#         initialize : bool, optional
#             Whether to initialize the weights, by default True
#         init_type : INIT_WEIGHT, optional
#             The type of initialization to use, by default INIT_WEIGHT.DEFAULT
#         init_settings : dict, optional
#             Settings for the initialization method, by default None
#         out_dim : int, optional
#             The output dimension of the LoRA layer, by default None
#         chunk_size : int, optional
#             The chunk size for the vmap function, by default None
#             If None all members are processed in parallel, otherwise the chunk size is used
#             If 1 all members are processed sequentially, like a for loop
#         """

#         super().__init__()

#         # If out_dim is not set, set it to dim
#         if out_dim is None:
#             out_dim = dim

#         # Initialize all the LoRA models
#         self.lora_models =  nn.ModuleList([LoRA(w, rank, dim, initialize, init_type, init_settings, out_dim).to(DEVICE)
#                             for _ in range(n_members)])
        
#         # Set the output dimension
#         self.out_dim = out_dim

#         # Stack the module state
#         self.params, self.buffers = stack_module_state(self.lora_models)

#         # Set the number of members
#         self.n_members = n_members

#         # Set the base model
#         self.base_model = copy.deepcopy(self.lora_models[0])
#         # self.base_model = self.base_model.to('meta')

#         # Set base model to not require gradients
#         # This can be done because meta tensors do not carry weights, they only include the model structure
#         # for param in self.base_model.parameters():
#         #     param.requires_grad = False

#         self.chunk_size = chunk_size

#     def _functional_call(
#             self,
#             x: Tensor,
#             params: Dict[str, Tensor],
#             buffers: Dict[str, Tensor],
#     ) -> callable:
#         """
#         Function to call the LoRA models per member with their own parameters and buffers
#         as well as their own input.

#         Parameters
#         ----------
#         x : Tensor
#             The input tensor
#         params : Dict[str, Tensor]
#             The parameters of the LoRA models
#         buffers : Dict[str, Tensor]
#             The buffers of the LoRA models

#         Returns
#         -------
#         callable
#             The functional call for the mapping of values to LoRA Models
#         """

#         return functional_call(self.base_model, (params, buffers), x)

#     def forward(self, x):
#         """
#         Forward pass for the LoRA Ensemble module

#         Parameters
#         ----------
#         x : Tensor
#             The input tensor

#         Returns
#         -------
#         out : Tensor
#             The output tensor
#         """
#         # Extract the necessary dimensions
#         sequence_length = x.shape[0]
#         batch_size = x.shape[1] // self.n_members

#         # Reshape the input tensor to have the ensemble members as the first dimension
#         ensemble_input = x.view(sequence_length, batch_size, self.n_members, -1)
#         ensemble_input = ensemble_input.movedim(2, 0)

#         # Call the actual models
#         out = vmap(self._functional_call, chunk_size=self.chunk_size)(ensemble_input, self.params, self.buffers)

#         # Move the dimensions back to the original shape
#         out = out.movedim(0, 2)
#         out = out.contiguous().view(sequence_length, -1, self.out_dim)

#         return out


# class EnsembleLoRA(nn.Module):
#     """
#     Ensembles multiple LoRA modules in parallel via torch.vmap,
#     ensuring everything is on DEVICE to avoid cross‐device errors.
#     """

#     def __init__(
#         self,
#         w: nn.Module,
#         rank: int,
#         dim: int,
#         n_members: int,
#         initialize: bool = True,
#         init_type=None,
#         init_settings: dict = None,
#         out_dim: int = None,
#         chunk_size: int = None
#     ):
#         super().__init__()
#         if out_dim is None:
#             out_dim = dim

#         # 1) Build & register each LoRA module *on DEVICE*
#         self.lora_models = nn.ModuleList([
#             LoRA(w, rank, dim, initialize, init_type, init_settings, out_dim)
#             .to(DEVICE)
#             for _ in range(n_members)
#         ])

#         # 2) Stack their parameters & buffers for functional vmap
#         params, buffers = stack_module_state(self.lora_models)

#         # 3) Move the stacked params & buffers to DEVICE as well
#         self.params = {k: t.to(DEVICE) for k, t in params.items()}
#         self.buffers = {k: t.to(DEVICE) for k, t in buffers.items()}

#         self.n_members = n_members
#         self.chunk_size = chunk_size
#         self.out_dim = out_dim

#     def _functional_call(
#         self,
#         x: Tensor,
#         params: dict,
#         buffers: dict
#     ) -> Tensor:
#         # Calls the *first* LoRA module structure with its own params/buffers
#         return functional_call(self.lora_models[0], (params, buffers), x)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         x: [seq_len, batch*n_members, dim]
#         returns: [seq_len, batch*n_members, out_dim]
#         """
#         # Parallel call across ensemble members
#         out = vmap(
#             self._functional_call,
#             in_dims=(None, 0, 0),
#             chunk_size=self.chunk_size
#         )(x, self.params, self.buffers)

#         # vmap returns [n_members, seq_len, batch, out_dim]
#         # Move member-dim back into the batch-dim:
#         out = out.movedim(0, 2).reshape(x.shape[0], -1, self.out_dim)
#         return out



class EnsembleLoRA(nn.Module):
    """
    Ensembles multiple LoRA modules by splitting the batch for each member,
    then applying each member's LoRA parameters to its slice of the sequence.
    """

    def __init__(
        self,
        w: nn.Module,
        rank: int,
        dim: int,
        n_members: int,
        initialize: bool = True,
        init_type=None,
        init_settings: dict = None,
        out_dim: int = None,
        chunk_size: int = None
    ):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        # Build & register each LoRA module on DEVICE
        self.lora_models = nn.ModuleList([
            LoRA(w, rank, dim, initialize, init_type, init_settings, out_dim)
            .to(DEVICE)
            for _ in range(n_members)
        ])

        # Stack their parameters & buffers for easy slicing
        params, buffers = stack_module_state(self.lora_models)
        self.params = {k: t.to(DEVICE) for k, t in params.items()}
        self.buffers = {k: t.to(DEVICE) for k, t in buffers.items()}

        self.n_members = n_members
        self.out_dim = out_dim
        self.chunk_size = chunk_size  # unused here but kept for API compatibility

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [seq_len, batch*n_members, dim]
        returns: [seq_len, batch*n_members, out_dim]
        """
        seq_len, batch_nm, _ = x.shape
        if batch_nm % self.n_members != 0:
            raise ValueError(f"Batch size ({batch_nm}) not divisible by n_members ({self.n_members})")
        batch = batch_nm // self.n_members

        # reshape to group each member's slice: [seq_len, batch, n_members, dim]
        x_grouped = x.view(seq_len, batch, self.n_members, -1)
        # bring member dim first: [n_members, seq_len, batch, dim]
        x_grouped = x_grouped.permute(2, 0, 1, 3)

        # apply each member's LoRA parameters to its slice
        outs = []
        for i in range(self.n_members):
            # extract params/buffers for member i
            pi = {k: v[i] for k, v in self.params.items()}
            bi = {k: v[i] for k, v in self.buffers.items()}
            # functional call on the base LoRA module structure
            out_i = functional_call(self.lora_models[0], (pi, bi), x_grouped[i])
            outs.append(out_i)  # each out_i: [seq_len, batch, out_dim]

        # stack outputs: [n_members, seq_len, batch, out_dim]
        out_stack = torch.stack(outs, dim=0)
        # permute to [seq_len, batch, n_members, out_dim]
        tmp = out_stack.permute(1, 2, 0, 3)
        # flatten back to [seq_len, batch*n_members, out_dim]
        out = tmp.reshape(seq_len, batch_nm, self.out_dim)
        return out


class BERTEnsembleLoRA(EnsembleLoRA):
    """
    Wraps EnsembleLoRA to permute BERT's (batch, seq_len, dim)
    ↔ (seq_len, batch*n_members, dim) for applying LoRA.
    """

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch*n_members, seq_len, dim]
        batch_nm, seq_len, _ = x.shape
        # to [seq_len, batch*n_members, dim]
        x2 = x.permute(1, 0, 2)
        # apply ensemble LoRA
        out = super().forward(x2)  # [seq_len, batch_nm, out_dim]
        # back to [batch*n_members, seq_len, out_dim]
        return out.permute(1, 0, 2).contiguous()


class SimpleBERTEnsembleLoRA(nn.Module):
    """
    Sequential ensemble of LoRA adapters for a single projection (query/key/value or out).
    Input:  [batch*n_members, seq_len, dim]
    Output: [batch*n_members, seq_len, out_dim]
    """
    def __init__(
        self,
        w: nn.Module,
        rank: int,
        dim: int,
        n_members: int,
        initialize: bool = True,
        init_type: Init_Weight = Init_Weight.DEFAULT,
        init_settings: dict = None,
        out_dim: int = None
    ):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        # One LoRA adapter per ensemble member
        self.adapters = nn.ModuleList([
            LoRA(w, rank, dim, initialize, init_type, init_settings, out_dim).to(DEVICE)
            for _ in range(n_members)
        ])
        self.n_members = n_members
        self.out_dim     = out_dim

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch*n_members, seq_len, dim]
        batch_nm, seq_len, _ = x.shape
        if batch_nm % self.n_members != 0:
            raise ValueError(f"batch {batch_nm} not divisible by n_members {self.n_members}")
        batch = batch_nm // self.n_members

        # reshape → [batch, n_members, seq_len, dim]
        xg = x.view(batch, self.n_members, seq_len, -1)

        # apply each adapter to its slice
        outs = []
        for i, adapter in enumerate(self.adapters):
            # adapter expects [batch, seq_len, dim]
            out_i = adapter(xg[:, i])  # → [batch, seq_len, out_dim]
            outs.append(out_i)

        # stack → [batch, n_members, seq_len, out_dim]
        stacked = torch.stack(outs, dim=1)
        # flatten → [batch*n_members, seq_len, out_dim]
        return stacked.view(batch_nm, seq_len, self.out_dim)




class ASTEnsembleLoRA(EnsembleLoRA):
    """
    Class to ensemble the LoRA layers for AST.
    This just overrides the forward pass to permute the channels to match the ViT implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # Permute channels to match ViT Implementation
        x = x.permute(1, 0, 2)

        # Call the original forward method from EnsembleLoRA
        out = super().forward(x)

        # Permute channels back to continue pass through AST
        out = out.permute(1, 0, 2)

        return out
