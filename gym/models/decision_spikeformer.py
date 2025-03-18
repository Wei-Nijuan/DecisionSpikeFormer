import numpy as np
import torch
import torch.nn as nn
import math
import transformers
import torch.nn.functional as F

import warnings
from typing import Optional, Tuple, Union

from git import Commit
from numpy.f2py.auxfuncs import throw_error
from torch.autograd import Variable


import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


thresh = 0.5
lens = 0.5
alpha = 1





# input shape: [T, B, L, D]
class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = x.permute(0, 3, 2, 1)  # T D L B
        x = self.bn(x) + self.alpha * x
        # x = x.transpose(1, 2)
        x = x.permute(0, 3, 2, 1)  # T B L D
        return x


class LN(nn.Module):
    """self.norm = LN([T, dim])"""

    def __init__(self, dim):
        super(LN, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # T B L D
        x = x.permute(2, 1, 0, 3).contiguous()  # L B T D
        x = self.ln(x)
        x = x.permute(2, 1, 0, 3).contiguous()  # T B L D
        return x


class BN(nn.Module):
    """self.norm=BN(dim)"""

    def __init__(self, dim):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # T D L B
        x = self.bn(x)
        x = x.permute(0, 3, 2, 1)  # T B L D
        return x


class PTNorm(nn.Module):
    """
    input: [T, B, L, D]
    dim: the dimension of input
    norm1: the first norm layer, used initially
    norm2: the second norm layer, used later
    step: the total steps of training
    warm: the warmup steps
    r0: the initial ratio of norm1, default is 1.0
    """

    def __init__(self, dim, T, step, warm=0, r0=1.0):
        super(PTNorm, self).__init__()
        self.register_buffer("warm", torch.tensor(warm, dtype=torch.long))
        self.register_buffer("iter", torch.tensor(step, dtype=torch.long))
        self.register_buffer("total_step", torch.tensor(step, dtype=torch.long))
        self.r0 = r0
        # self.norm1 = LN(dim)
        self.norm1 = LN(dim)
        self.norm2 = BN(dim)

    def forward(self, x):
        if self.training:
            # Compute lamda based on warmup phase
            if self.warm > 0:
                lamda = 1.0
            elif self.total_step == 0:
                lamda = 0
            else:
                lamda = self.r0 * self.iter.float() / self.total_step.float()
                # print(lamda)

            # # Ensure lamda is within [0, 1]
            # lamda = lamda.clamp(0.0, 1.0)

            # Compute outputs from both norm layers
            x1 = self.norm1(x)
            x2 = self.norm2(x)

            # Combine the outputs
            x = lamda * x1 + (1 - lamda) * x2
            if self.warm > 0:
                self.warm -= 1
            if self.iter > 0:
                self.iter -= 1
        else:
            x = self.norm2(x)
            # Decrement warmup and iteration counters
        return x


class PTNorm_Advanced(nn.Module):
    """
    another version of PTNorm
    this behavior well in some case (Adroit)
    still keep spike nature in this version
    """

    def __init__(self, dim, T, step, warm=0, r0=1.0):
        super(PTNorm_Advanced, self).__init__()
        self.register_buffer("warm", torch.tensor(warm, dtype=torch.long))
        self.register_buffer("iter", torch.tensor(step, dtype=torch.long))
        self.register_buffer("total_step", torch.tensor(step, dtype=torch.long))
        self.r0 = r0
        self.norm1 = LN([T, dim])
        self.norm2 = RepBN(dim)  # TODO

    def forward(self, x):
        if self.training:
            # Compute lamda based on warmup phase
            if self.warm > 0:
                lamda = 1.0
            elif self.total_step == 0:
                lamda = 0
            else:
                lamda = self.r0 * self.iter.float() / self.total_step.float()
                # print(lamda)
            # # Ensure lamda is within [0, 1]
            # lamda = lamda.clamp(0.0, 1.0)
            # Compute outputs from both norm layers
            x1 = self.norm1(x)
            x2 = self.norm2(x)

            # Combine the outputs
            x = lamda * x1 + (1 - lamda) * x2
            if self.warm > 0:
                self.warm -= 1
            if self.iter > 0:
                self.iter -= 1
        else:
            x = self.norm2(x)
            # Decrement warmup and iteration counters
        return x


class Norm(nn.Module):
    def __init__(self, dim, T, step, warm=0, r0=1.0, norm_type=3):
        super().__init__()
        if norm_type == 1:
            self.norm = LN(dim)
            print("Using LN norm.")
        elif norm_type == 2:
            self.norm = BN(dim)
            print("Using BN norm.")
        elif norm_type == 3:
            self.norm = PTNorm(dim, T, step, warm, r0)
            # self.norm = PTNorm_Advanced(dim, T, step, warm, r0)
            print("Using PTNorm norm.")
        else:
            print(f"Invalid norm type:{norm_type}")
            raise ValueError("Invalid norm type.")

    def forward(self, x):
        return self.norm(x)


class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = (input > thresh).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        # Compute the gradient where input is close to thresh within lens
        temp = ((input - thresh).abs() < lens).float() / (2 * lens)
        grad_input = grad_output * temp
        return grad_input


act_fun = ActFun.apply





class LIFNode(nn.Module):
    def __init__(
        self,
        act=False,
        init_thresh=1.0,
        init_decay=0.25,
        store_fire_rate=False,
    ):
        super(LIFNode, self).__init__()
        self.thresh = init_thresh
        self.decay = init_decay
        self.actFun = (
            nn.SiLU() if act else act_fun
        )  # Ensure `act_fun` is defined elsewhere
        self.store_fire_rate = store_fire_rate
        self.fire_rate = []  # List to store fire rate for each time step

    def forward(self, x):
        # Input shape: T x B x L x D
        T, B, L, D = x.size()
        mem = x[0]
        spike = self.actFun(mem)
        outputs = [spike]

        # Initialize fire rate list if enabled
        if self.store_fire_rate and not self.training:
            step_fire_rates = [spike.mean().item()]

        for i in range(1, T):
            # Update membrane potential
            mem = mem * self.decay * (1 - spike.detach()) + x[i]
            # Generate spike output
            spike = self.actFun(mem)
            outputs.append(spike)

            # Calculate and store fire rate for this time step
            if self.store_fire_rate and not self.training:
                step_fire_rates.append(spike.mean().item())

        # Stack outputs along the time dimension
        output = torch.stack(outputs, dim=0)

        # Store fire rates for each time step
        if self.store_fire_rate and not self.training:
            self.fire_rate.append(step_fire_rates)
        if self.store_fire_rate and self.training:
            # Clear fire rate during training
            self.fire_rate = []

        return output

    def reset_fire_rate(self):
        """Clear the fire rate statistics."""
        self.fire_rate = []

    def get_fire_rate(self):
        """Get average fire rate over all stored time steps."""
        if len(self.fire_rate) == 0:
            return []
        return np.mean(self.fire_rate, axis=0)  # Mean over batches if needed

    def get_fire_rate_list(self):
        """Get average fire rate over all stored time steps."""
        if len(self.fire_rate) == 0:
            return []
        return self.fire_rate # Mean over batches if needed


class positional_spiking_attention(nn.Module):
    def __init__(
        self,
        dim,
        T,
        num_training_steps,
        heads=8,
        seq_len=64,
        norm_type=3,
        window_size=8,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.norm_type = norm_type

        self.L = seq_len
        self.dim = dim
        self.heads = heads

        self.q_m = nn.Linear(dim, dim)
        self.q_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.q_ln = LN([T, dim])
        # self.q_ln = BN(dim)
        self.q_lif = LIFNode(act=False)

        self.k_m = nn.Linear(dim, dim)
        self.k_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.k_ln = LN([T, dim])
        # self.k_ln = BN(dim)
        self.k_lif = LIFNode(act=False)  # 使用第一种形式

        self.v_m = nn.Linear(dim, dim)
        self.v_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.v_ln = LN([T, dim])
        # self.v_ln = BN(dim)
        self.v_lif = LIFNode(act=False)

        self.attn_lif = LIFNode(act=False)

        self.last_m = nn.Linear(dim, dim)
        self.last_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.last_ln = LN([T, dim])
        # self.last_ln = BN(dim)
        if self.norm_type == 3:
            self.last_ln.norm.norm1.is_shortcut = True
            self.last_ln.norm.norm2.is_shortcut = True
        # self.last_ln.norm1.is_shortcut = True
        # self.last_ln.norm2.is_shortcut = True
        # self.last_ln.ln.is_shortcut = True
        # self.last_ln.bn.is_shortcut = True

        self.first_lif = LIFNode(act=False)

        local_window_size = window_size
        # self.pos_vector_a = nn.Parameter(torch.ones(seq_len), requires_grad=True)
        # self.pos_vector_b = nn.Parameter(torch.ones(seq_len), requires_grad=True)
        # This part can be optimized, because we just need parameter in local window size rather all
        self.pos_bias = nn.Parameter(torch.ones(seq_len, seq_len), requires_grad=True)
        self.register_buffer(
            "local_mask", self.create_local_mask(seq_len, local_window_size)
        )
        print(f"使用spiking free attention!!!!!!!!!!!!!!!!!!!!")

    @staticmethod
    def create_local_mask(seq_len, local_window_size):
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        mask = torch.tril(mask, local_window_size - 1)
        mask = torch.triu(mask, -local_window_size + 1)
        return mask

    def print_fire_rate(self):
        print(f"Attn1 fire rate: {self.first_lif.get_fire_rate()}")
        print(f"Attn2 fire rate: {self.attn_lif.get_fire_rate()}")

    def forward(self, x, attention_mask=None):  # T B L D
        # pos_bias = torch.ger(self.pos_vector_a, self.pos_vector_b)
        pos_bias = self.pos_bias
        L = x.size(2)
        # 根据输入长度生成位置偏差
        pos_bias = pos_bias[:L, :L] * self.local_mask[:L, :L]
        pos_bias = pos_bias.unsqueeze(0).unsqueeze(0)  # B H L L
        # 获取因果掩码
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        pos_bias = pos_bias.masked_fill(mask == 0, 0)  # B H L L
        # 获取填充掩码
        if attention_mask is not None:
            batch_size = x.shape[1]
            attention_mask = attention_mask.view(batch_size, -1) # B, L,
            # B 1 1 L , 只在to使用attention_mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # B 1 1 L
            pos_bias = pos_bias.masked_fill(attention_mask == 0, 0)
        # 生成q/k/v矩阵,其中k矩阵无需进行lif脉冲化
        x = self.first_lif(x)
        T, B, L, D = x.shape
        q_m_out = self.q_m(x)  # T B L D
        q_m_out = self.q_ln(q_m_out)
        q_m_out = self.q_lif(q_m_out)
        q = (
            q_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B heads L D//heads
        q = q.permute(1, 2, 3, 0, 4).reshape(
            B, self.heads, L, T * D // self.heads
        )  # B heads L T*D//heads

        k_m_out = self.k_m(x)  # T B L D
        k_m_out = self.k_ln(k_m_out)
        k_m_out = self.k_lif(k_m_out)

        v_m_out = self.v_m(x)
        v_m_out = self.v_ln(v_m_out)
        v_m_out = self.v_lif(v_m_out)  # T,B,L,D

        # 计算注意力,这里的逐点乘法并不会导致整数脉冲的出现
        kv = (
            (k_m_out * v_m_out)
            .reshape(T, B, L, self.heads, D // self.heads)
            .permute(1, 3, 2, 0, 4)
        )  # B H L T D//H
        kv = kv.reshape(B, self.heads, L, T * D // self.heads)  # B H L T*D//H
        attn = torch.einsum("bhij,bhjd->bhid", pos_bias, kv)
        x = q * attn
        x = (
            x.reshape(B, self.heads, L, T, D // self.heads)
            .permute(3, 0, 2, 1, 4)
            .reshape(T, B, L, D)
        )
        x = self.attn_lif(x)

        x = self.last_m(x)
        x = self.last_ln(x)
        return x


class step_spiking_attention(nn.Module):
    """
    qk_scale: 1/head_num
    """

    def __init__(self, dim, T, num_training_steps, heads=8, seq_len=64, norm_type=3):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.norm_type = norm_type

        self.L = seq_len
        self.dim = dim
        self.heads = heads

        self.q_m = nn.Linear(dim, dim)
        self.q_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.q_ln = LN([T, dim])
        # self.q_ln = BN(dim)
        self.q_lif = LIFNode(act=False)

        self.k_m = nn.Linear(dim, dim)
        self.k_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.k_ln = LN([T, dim])
        # self.k_ln = BN(dim)
        self.k_lif = LIFNode(act=False)

        self.v_m = nn.Linear(dim, dim)
        self.v_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.v_ln = LN([T, dim])
        # self.v_ln = BN(dim)
        self.v_lif = LIFNode(act=False)

        self.attn_lif = LIFNode(act=False)

        self.last_m = nn.Linear(dim, dim)
        self.last_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.last_ln = LN([T, dim])
        # self.last_ln = BN(dim)
        if self.norm_type == 3:
            self.last_ln.norm.norm1.is_shortcut = True
            self.last_ln.norm.norm2.is_shortcut = True
        # self.last_ln.norm1.is_shortcut = True
        # self.last_ln.norm2.is_shortcut = True
        # self.last_ln.ln.is_shortcut = True
        # self.last_ln.bn.is_shortcut = True
        self.first_lif = LIFNode(act=False)
        # self.fire_rate_1 = []
        # self.fire_rate_2 = []

    def get_fire_rate(self):
        return np.mean(self.fire_rate_1) + np.mean(self.fire_rate_2)

    def print_fire_rate(self):
        print(f"Attn1 fire rate: {self.first_lif.get_fire_rate()}")
        print(f"Attn2 fire rate: {self.get_fire_rate()}")
        print(f"Attn3 fire rate: {self.attn_lif.get_fire_rate()}")

    def forward(self, x, attention_mask=None):
        # x : T B L D
        x = self.first_lif(x)

        T, B, L, D = x.shape

        q_m_out = self.q_m(x)  # T B L D
        q_m_out = self.q_ln(q_m_out)
        q_m_out = self.q_lif(q_m_out)
        # q: T B heads L D//heads
        q = (
            q_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_m_out = self.k_m(x)  # T B L D
        k_m_out = self.k_ln(k_m_out)
        k_m_out = self.k_lif(k_m_out)
        # k: T B heads L D//heads
        k = (
            k_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_m_out = self.v_m(x)
        v_m_out = self.v_ln(v_m_out)
        v_m_out = self.v_lif(v_m_out)
        # v: T B heads L D//heads
        v = (
            v_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        attn = q @ k.transpose(-2, -1)
        # -----------------------fire rate 1-------------------------
        # # attn is not spike, T B heads L L
        # if not self.training:
        #     # calculate the fire rate
        #     fire_rate_1 = attn.sum().item() / (T * B * D * L * L)
        #     self.fire_rate_1.append(fire_rate_1)
        # if self.training:
        #     # to avoid test data too much, we clear the fire rate
        #     # we just use it in inference not test and train
        #     self.fire_rate_1 = []
        # ------------------------------------------------
        # 生成因果mask
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, 1, L, L)
        attn = attn.masked_fill(mask == 0, 0)  # T B heads L L
        if attention_mask is not None:
            batch_size = attn.shape[1]
            attention_mask = attention_mask.view(batch_size, -1) # B, L,
            # B 1 1 L , 只在to使用attention_mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # 1 B 1 1 L
            attention_mask = attention_mask.unsqueeze(0)
            attn = attn.masked_fill(attention_mask == 0, 0)
        # # 将mask的部分设置为0,因为我们没有时候softmax,因此不需要设置为负无穷
        x = attn @ v  # T B heads L D//heads
        # # -----------------------fire rate 2-------------------------
        # if not self.training:
        #     # calculate the fire rate
        #     # transform attn to spike
        #     attn_spike = (attn > 0).float()
        #     y = attn_spike @ v
        #     fire_rate_2 = y.sum().item() / (T * B * D * L * L)
        #     self.fire_rate_2.append(fire_rate_2)
        # if self.training:
        #     # to avoid test data too much, we clear the fire rate
        #     # we just use it in inference not test and train
        #     self.fire_rate_2 = []
        # -----------------------------------------------------------
        # x = x.reshape(B, self.heads, L, T, D // self.heads).permute(3, 0, 2, 1, 4).reshape(T, B, L, D).contiguous()
        x = (
            x.reshape(T, B, self.heads, L, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .reshape(T, B, L, D)
        )
        x = self.attn_lif(x)

        x = self.last_m(x)
        # x = x.reshape(T, B, L, D)
        x = self.last_ln(x)
        # x = x.transpose(0, 1)  # B T L D
        return x


class temporal_spiking_attention(nn.Module):
    """
    qk_scale: 1/head_num
    """

    def __init__(self, dim, T, num_training_steps, heads=8, seq_len=64, norm_type=3):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.norm_type = norm_type

        self.L = seq_len
        self.dim = dim
        self.heads = heads

        self.q_m = nn.Linear(dim, dim)
        self.q_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.q_ln = LN([T, dim])
        # self.q_ln = BN(dim)
        self.q_lif = LIFNode(act=False)

        self.k_m = nn.Linear(dim, dim)
        self.k_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.k_ln = LN([T, dim])
        # self.k_ln = BN(dim)
        self.k_lif = LIFNode(act=False)

        self.v_m = nn.Linear(dim, dim)
        self.v_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.v_ln = LN([T, dim])
        # self.v_ln = BN(dim)
        self.v_lif = LIFNode(act=False)

        self.attn_lif = LIFNode(act=False)
        self.last_m = nn.Linear(dim, dim)
        self.last_ln = Norm(
            dim, T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.last_ln = LN([T, dim])
        # self.last_ln = BN(dim)
        if self.norm_type == 3:
            self.last_ln.norm.norm1.is_shortcut = True
            self.last_ln.norm.norm2.is_shortcut = True
        # self.last_ln.norm1.is_shortcut = True
        # self.last_ln.norm2.is_shortcut = True
        # self.last_ln.ln.is_shortcut = True
        # self.last_ln.bn.is_shortcut = True
        self.first_lif = LIFNode(act=False)
        self.fire_rate_1 = []
        self.fire_rate_2 = []

    def get_fire_rate(self):
        return np.mean(self.fire_rate_1) + np.mean(self.fire_rate_2)

    def print_fire_rate(self):
        print(f"Attn1 fire rate: {self.first_lif.get_fire_rate()}")
        print(f"Attn2 fire rate: {self.get_fire_rate()}")
        print(f"Attn3 fire rate: {self.attn_lif.get_fire_rate()}")
        # print(self.attn_lif.get_fire_rate_list())

    def forward(self, x, attention_mask=None, output_attentions=False):  # T B L D
        x = self.first_lif(x)

        T, B, L, D = x.shape

        q_m_out = self.q_m(x)  # T B L D
        # 转为 T D B L
        q_m_out = self.q_ln(q_m_out)
        q_m_out = self.q_lif(q_m_out)
        q = (
            q_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B heads L D//heads
        q = q.permute(1, 2, 3, 0, 4).reshape(
            B, self.heads, L, T * D // self.heads
        )  # B heads L T*D//heads

        k_m_out = self.k_m(x)  # T B L D
        k_m_out = self.k_ln(k_m_out)
        k_m_out = self.k_lif(k_m_out)
        k = (
            k_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        k = k.permute(1, 2, 3, 0, 4).reshape(
            B, self.heads, L, T * D // self.heads
        )  # B heads L T*D//heads

        v_m_out = self.v_m(x)
        v_m_out = self.v_ln(v_m_out)
        v_m_out = self.v_lif(v_m_out)
        v = (
            v_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        v = v.permute(1, 2, 3, 0, 4).reshape(
            B, self.heads, L, T * D // self.heads
        )  # B heads L T*D//heads

        attn = q @ k.transpose(-2, -1)  # attn is not spike, T B heads L L
        if output_attentions:
            print("Q shape= ", q.shape)
            print("attn1:",attn.shape)
        # -----------------------fire rate 1-------------------------
        if not self.training:
            # calculate the fire rate
            fire_rate_1 = attn.sum().item() / (T * B * D * L * L)
            self.fire_rate_1.append(fire_rate_1)
        if self.training:
            # to avoid test data too much, we clear the fire rate
            # we just use it in inference not test and train
            self.fire_rate_1 = []
        # ------------------------------------------------
        # 生成因果mask
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, 1, L, L)
        attn = attn.masked_fill(mask == 0, 0)  # T B heads L L


        if attention_mask is not None:
            batch_size = attn.shape[1]
            attention_mask = attention_mask.view(batch_size, -1) # B, L,
            # B 1 1 L , 只在to使用attention_mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # 1 B 1 1 L
            attention_mask = attention_mask.unsqueeze(0)
            attn = attn.masked_fill(attention_mask == 0, 0)
        # # 将mask的部分设置为0,因为我们没有时候softmax,因此不需要设置为负无穷
        x = attn @ v  # B heads L T*D//heads
        # -----------------------fire rate 2-------------------------
        if not self.training:
            # calculate the fire rate
            # transform attn to spike
            attn_spike = (attn > 0).float()
            y = attn_spike @ v
            fire_rate_2 = y.sum().item() / (T * B * D * L * L)
            self.fire_rate_2.append(fire_rate_2)
        if self.training:
            # to avoid test data too much, we clear the fire rate
            # we just use it in inference not test and train
            self.fire_rate_2 = []
        # -----------------------------------------------------------
        x = (
            x.reshape(B, self.heads, L, T, D // self.heads)
            .permute(3, 0, 2, 1, 4)
            .reshape(T, B, L, D)
            .contiguous()
        )
        x = self.attn_lif(x)
        x = self.last_m(x)
        x = self.last_ln(x)
        return x


class mlp(nn.Module):
    def __init__(
        self,
        T,
        num_training_steps,
        in_features,
        hidden_features=None,
        out_features=None,
        norm_type=3,
    ):
        super().__init__()
        # self.length = length
        self.norm_type = norm_type
        out_features = out_features or in_features
        hidden_features = hidden_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = Norm(
            hidden_features,
            T,
            step=int(num_training_steps * 0.10),
            norm_type=self.norm_type,
        )
        # self.ln1 = LN([T, hidden_features])
        # self.ln1 = BN(hidden_features)
        self.lif1 = LIFNode(act=False)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.ln2 = Norm(
            out_features,
            T,
            step=int(num_training_steps * 0.10),
            norm_type=self.norm_type,
        )
        # self.ln2 = LN([T, out_features])
        # self.ln2 = BN(out_features)
        if self.norm_type == 3:
            self.ln2.norm.norm1.is_shortcut = True
            self.ln2.norm.norm2.is_shortcut = True
        # self.ln2.norm1.is_shortcut = True
        # self.ln2.norm2.is_shortcut = True
        # self.ln2.ln.is_shortcut = True
        # self.ln2.bn.is_shortcut = True
        self.lif2 = LIFNode(act=False)

    def forward(self, x):
        # T B L D
        x = self.lif1(x)
        x = self.fc1(x)  # T B L D_hidden
        x = self.ln1(x)
        x = self.lif2(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x

    def print_fire_rate(self):
        print(f"MLP1 fire rate: {self.lif1.get_fire_rate()}")
        print(f"MLP2 fire rate: {self.lif2.get_fire_rate()}")


class block(nn.Module):
    def __init__(
        self,
        drop_dpr,
        dim,
        T,
        num_training_steps,
        heads=8,
        qkv_bias=False,
        seq_len=64,
        attn_type=3,
        norm_type=3,
        window_size=8,
    ):
        super().__init__()
        if attn_type == 1:
            self.attn = step_spiking_attention(
                dim=dim,
                T=T,
                heads=heads,
                seq_len=seq_len,
                num_training_steps=num_training_steps,
                norm_type=norm_type,
            )
            print("using step attention")
        elif attn_type == 2:
            self.attn = temporal_spiking_attention(
                dim=dim,
                T=T,
                heads=heads,
                seq_len=seq_len,
                num_training_steps=num_training_steps,
                norm_type=norm_type,

            )
            print("using temporal attention")
        elif attn_type == 3:
            self.attn = positional_spiking_attention(
                dim=dim,
                T=T,
                heads=heads,
                seq_len=seq_len,
                num_training_steps=num_training_steps,
                norm_type=norm_type,
                window_size=window_size,
            )
            print("using positional attention")
        else:
            throw_error("attn_type must be in [1,2,3]")
        self.mlp = mlp(
            T=T,
            in_features=dim,
            hidden_features=dim * 4,
            out_features=dim,
            num_training_steps=num_training_steps,
            norm_type=norm_type,
        )

    def print_fire_rate(self):
        self.attn.print_fire_rate()
        self.mlp.print_fire_rate()

    def forward(self, x, attention_mask=None, output_attentions=False):
        # B T L D
        x = x + self.attn(x, attention_mask=attention_mask)  # 残差连接
        x = x + self.mlp(x)
        return x

class new_spikformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.atan = surrogate_function
        self.T = config.T
        dim = config.n_embd
        heads = config.n_head
        seq_len = config.ctx_len
        attn_type = config.attn_type
        self.norm_type = config.norm_type
        num_training_steps = config.num_training_steps
        window_size = config.window_size
        self.drop_path_rate = 0.05  # 0.05效果更好
        # self.ln0 = Norm(
        #     dim, self.T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        # )
        self.blocks = nn.ModuleList(
            [
                block(
                    drop_dpr=self.drop_path_rate * float(idx) / config.n_layer,
                    dim=dim,
                    T=self.T,
                    heads=heads,
                    seq_len=seq_len,
                    attn_type=attn_type,
                    num_training_steps=num_training_steps,
                    norm_type=self.norm_type,
                    window_size=window_size,
                )
                for idx in range(config.n_layer)
            ]
        )
        self.last_ln = Norm(
            dim, self.T, step=int(num_training_steps * 0.10), norm_type=self.norm_type
        )
        # self.last_ln = LN([self.T, dim])
        # self.last_ln = BN(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 设置shortcut标识为了设置weight
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(m, "is_shortcut") and m.is_shortcut:
                nn.init.constant_(m.weight, thresh / (2**0.5))  # 我们最多只使用2个分支
            else:
                nn.init.constant_(m.weight, thresh)  # 0.5->init_threshold
            nn.init.constant_(m.bias, 0.0)

    def print_fire_rate(self):
        for i, block in enumerate(self.blocks):
            print(f"-----------Block {i+1} start------------")
            block.print_fire_rate()
            print(f"-----------Block {i+1} end------------")

    def forward(self, x, attention_mask=None, output_attentions=False):
        # B L D
        x = x.repeat(
            tuple([self.T] + torch.ones(len(x.size()), dtype=int).tolist())
        )  # T B L D,就是新增一个维度重复T倍，其他维度形状不变
        if output_attentions:
            representations = {}
            representations["attentions"] = []
            representations["hidden_states"] = []
        for i, decoder in enumerate(self.blocks):
            if output_attentions:
                attn, x = decoder(x, attention_mask=attention_mask, output_attentions=output_attentions)
                representations["attentions"].append(attn)
                representations["hidden_states"].append(x)
            else:
                x = decoder(x, attention_mask=attention_mask, output_attentions=output_attentions)  # T B L D
            # representations.append(self.transforms[i](x.mean(1)))  # B * L * D
            # last step,后续可以考虑只看看最后一层的输出,现在先不考虑那么多
            # representations.append(self.transforms[i](x[:,-1,:,:])) # B * L * D
        # T B L D
        x = self.last_ln(x)
        x = x.mean(0)  # B L D
        return x


from types import SimpleNamespace


class SpikeDecisionTransformer(nn.Module):
    def __init__(self, config):
        """
        @data_mode in [mdp, nonmdp]
        """
        super().__init__()
        T =  4
        attn_type = 2 # 2:tssa    3:pssa
        norm_type = 1 # 3:PTNorm 1:LN 2:BN
        window_size = 8
        print(f"norm_type: {norm_type}")
        self.T = T
        num_training_steps = config.num_training_steps
        print(f"Spiking Transformer using T: {T}")
        print(f"Spiking Transformer using norm_type: {norm_type}")
        print(f"Spiking Transformer using attn_type: {attn_type}")
        print(f"Spiking Transformer using window_size: {window_size}")
        print(f"Spiking Transformer using num_training_steps: {num_training_steps}")
        self.dropout = 0.1
        config = dict(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=config.n_embd,
            n_ctx=config.n_positions,  # decoder's max_len
            n_layer=config.n_layer,
            n_head=config.n_head,
            ctx_len=config.n_positions,
            T=self.T,
            attn_type=attn_type,
            num_training_steps=num_training_steps,
            norm_type=norm_type,
            window_size=window_size,
        )
        config = SimpleNamespace(**config)
        self.transformer = new_spikformer(config)

    def forward(
        self,
        srcs: torch.Tensor,
        attention_mask=None,
        output_attentions=False
    ):
        """
        @param srcs: (bz, seq_len, <=src_dim)
        """
        outputs = self.transformer(srcs, attention_mask=attention_mask, output_attentions=output_attentions)
        return outputs


class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])

class DecisionSpikeFormer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            remove_act_embs=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        print("Using DecisionTransformer")

        self.hidden_size = hidden_size
        token_num = 3 if not remove_act_embs else 2
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_positions=max_length,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = SpikeDecisionTransformer(config)

        self.remove_act_embs = remove_act_embs

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # self.embed_return = torch.nn.Linear(1, hidden_size)
        # self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        # self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed = nn.Linear(self.state_dim + self.act_dim + 1, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None, output_attentions=False):

        batch_size, seq_length = states.shape[0], states.shape[1]

        actions = torch.cat([torch.zeros_like(actions[:,0:1]), actions[:,:-1]], dim=1)
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # embed each modality with a different head
        # state_embeddings = self.embed_state(states)
        # action_embeddings = self.embed_action(actions)
        # returns_embeddings = self.embed_return(returns_to_go)

        # time_embeddings = self.embed_timestep(timesteps)


        # time embeddings are treated similar to positional embeddings
        # state_embeddings = state_embeddings + time_embeddings
        # action_embeddings = action_embeddings + time_embeddings
        # returns_embeddings = returns_embeddings + time_embeddings
        embeddings = self.embed(torch.cat([states, actions, returns_to_go], dim=-1))

        stacked_inputs = self.embed_ln(embeddings) # +time_embeddings


        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            stacked_inputs,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        x = transformer_outputs

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # or rewards (3)

        action_preds = self.predict_action(x)  # predict next action given state

        return None, action_preds, None

    # get_action并不需要attention_mask, 因为其没有填充
    def get_action(self, states, actions, returns_to_go, timesteps,**kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        state_preds, action_preds, return_preds = self.forward(
            states, actions, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1] # get_action方法只会在Inference中使用，此时的batch_size只能为1

    def get_actions(self, states, actions, returns_to_go, timesteps,**kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        assert self.training is False
        with torch.no_grad():
            state_preds, action_preds, return_preds = self.forward(
                states, actions, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0] # get_action方法只会在Inference中使用，此时的batch_size只能为1
