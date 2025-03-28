"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class Convolution(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.window_size = config.window_size
        hidden_size = config.n_embd
        self.conv_proj = config.conv_proj
        
        self.rtg_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        self.obs_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        self.act_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        
        if config.conv_proj:
            self.fc = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        window_size = self.window_size

        # pad the input tensor with zeros along the sequence dimension
        padded_tensor = torch.nn.functional.pad(x, (0, 0, window_size - 1, 0)).transpose(1, 2)
        
        rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
        obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
        act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]

        conv_tensor = torch.zeros((x.shape[0], x.shape[2], x.shape[1])).to('cuda')
        conv_tensor[:, :, ::3] = rtg_conv_tensor
        conv_tensor[:, :, 1::3] = obs_conv_tensor
        conv_tensor[:, :, 2::3] = act_conv_tensor
        conv_tensor = conv_tensor.transpose(1, 2)
        
        if self.conv_proj:
            conv_tensor = self.dropout(self.fc(conv_tensor))
       
        return conv_tensor
    

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, index):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
            
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y



class Pooling(nn.Module):
    def __init__(self, window_size):
        super(Pooling, self).__init__()
        self.W = window_size
        self.pooling = nn.AvgPool1d(kernel_size=window_size, stride=1, padding=0)
        print("Pooling window size: ", window_size)

    def forward(self, x): # Shape: (B, L, D)
        L = x.size(1)
        if L <= self.W:
            # print("x")
            # print(x)
            # print(x.shape)
            x_sum = torch.cumsum(x, dim=1)
            # print("x_sum")
            # print(x_sum)
            # print(x_sum.shape)
            weights = torch.arange(1, L+1).to(x.device).unsqueeze(0).unsqueeze(-1)
            # print("weights")
            # print(weights)
            # print(weights.shape)
            x_avg = x_sum / weights
            return x_avg
        x = F.pad(x, (0, 0, self.W-1, 0), mode='constant', value=0)  # Shape: (B, L + W - 1, D)
        x_transposed = x.transpose(1, 2)  # Shape: (B, D, L + W - 1)
        pooled = self.pooling(x_transposed).transpose(1, 2)  # Shape: (B, L, D)
        # edge case
        x_sum = torch.cumsum(x, dim=1)[:, self.W-1:2*self.W-2]
        weights = torch.arange(1, self.W).to(x.device).unsqueeze(0).unsqueeze(-1)
        x_avg = x_sum / weights
        pooled[:, :self.W-1, :] = x_avg
        return pooled

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, index):
        super().__init__()
        
        self.token_mixer = config.token_mixer
        self.n_layer = config.n_layer
        self.index = index
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if self.token_mixer == 'attn':
            self.attn = CausalSelfAttention(config, index)
        if self.token_mixer == 'conv':
            self.conv = Convolution(config, index)
        if self.token_mixer == 'conv-attn':
            if self.index < self.n_layer - 1:
                self.conv = Convolution(config, index)
            else:
                self.attn = CausalSelfAttention(config, index)
        if self.token_mixer == 'pool':
            self.pooling = Pooling(config.pooling_size)
            
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        if self.token_mixer == 'attn':
            x = x + self.attn(self.ln1(x))
        elif self.token_mixer == 'conv':
            x = x + self.conv(self.ln1(x))
        elif self.token_mixer == 'conv-attn':
            if self.index < self.n_layer - 1:
                x = x + self.conv(self.ln1(x))
            else:
                x = x + self.attn(self.ln1(x))
        elif self.token_mixer == 'pool':
            x = x + self.pooling(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ConvInterp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=3, padding=1):
        super(ConvInterp, self).__init__()
        print("Using ConvInterp")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, padding=padding)

    def forward(self, x):
        # x 的形状为 (B, 3L, D)
        x = x.permute(0, 2, 1)  # 改变形状为 (B, D, 3L)
        x = self.conv(x)  # 应用卷积，结果形状为 (B, D, L)
        x = x.permute(0, 2, 1)  # 恢复形状为 (B, L, D)
        return x


class ConvInterp2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(ConvInterp2, self).__init__()
        print("Using ConvInterp2")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, padding=padding)

    def forward(self, x):
        # x 的形状为 (B, 2L, D)
        x = x.permute(0, 2, 1)  # 改变形状为 (B, D, 2L)
        x = self.conv(x)  # 应用卷积，结果形状为 (B, D, L)
        x = x.permute(0, 2, 1)  # 恢复形状为 (B, L, D)
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config, index) for index in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


        self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        # self.cn = ConvInterp(config.n_embd, config.n_embd)
        self.cn = ConvInterp2(config.n_embd, config.n_embd)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        # R_t, s_t, a_t,R_{t+1} .......

        state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)

        if self.model_type == 'reward_conditioned_cnn':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            # token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            # token_embeddings[:,::3,:] = rtg_embeddings
            # token_embeddings[:,1::3,:] = state_embeddings
            # token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
            # token_embeddings = self.cn(token_embeddings)
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings
            token_embeddings[:,1::2,:] = state_embeddings
            token_embeddings = self.cn(token_embeddings)
        elif actions is not None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        # elif actions is None and self.model_type == 'reward_conditioned_cnn': # only happens at very first timestep of evaluation
        #     assert states.shape[1] == 1
        #     rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
        #     token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
        #     token_embeddings[:,::3,:] = rtg_embeddings # really just [:,0,:]
        #     token_embeddings[:,1::3,:] = state_embeddings # really just [:,1,:]
        #     token_embeddings[:,2::3,:] = torch.zeros_like(state_embeddings) # actually don't use in conv
        #     token_embeddings = self.cn(token_embeddings)
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            print(actions)
            print(actions.shape)
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if self.model_type == 'reward_conditioned_cnn':
            logits = logits
        elif actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss
