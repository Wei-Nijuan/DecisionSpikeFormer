import torch
from models.decision_transformer import DecisionTransformer
import numpy as np
import gym
import os
import random
import matplotlib.pyplot as plt
import math


from training.dt_trainer import DecisionTransformerTrainer

class CosineAnnealingWarmupLR:
    def __init__(self, optimizer, T_max, warmup_steps, eta_min=0.):
        self.optimizer = optimizer
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.last_epoch = -1
        # base_lrs是在init的时候就记录下来的，后面不会改变
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [(base_lr * (self.last_epoch + 1) / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self):
        self.last_epoch += 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device='cuda')
    return x

def get_env_info(env_name, dataset):
    if env_name == 'hopper':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 9000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [5000, 4000, 2500]
        scale = 1000.
    elif env_name == 'antmaze':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
        scale = 1.
    elif env_name == 'pen':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'hammer':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000, 3000]
        scale = 1000.
    elif env_name == 'door':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [2000, 1000, 500]
        scale = 100.
    elif env_name == 'relocate':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3000, 1000]
        scale = 1000.
    elif env_name == 'kitchen':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [500, 250]
        scale = 100.
    elif env_name == 'maze2d':
        if 'open' in dataset:
            dversion = 0
        else:
            dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [300, 200, 150,  100, 50, 20]
        scale = 10.
    else:
        raise NotImplementedError
    return env, max_ep_len, env_targets, scale
    # origin implementation in DC, 我们是v2版本，DC使用的是v3版本
    # if env_name == 'hopper':
    #     env = gym.make('Hopper-v3')
    #     max_ep_len = 1000
    #     env_targets = [3600, 7200, 36000, 72000]  # evaluation conditioning targets
    #     scale = 1000.  # normalization for rewards/returns
    # elif env_name == 'halfcheetah':
    #     env = gym.make('HalfCheetah-v3')
    #     max_ep_len = 1000
    #     env_targets = [12000, 24000, 120000, 240000]
    #     scale = 1000.
    # elif env_name == 'walker2d':
    #     env = gym.make('Walker2d-v3')
    #     max_ep_len = 1000
    #     env_targets = [5000, 10000, 50000, 100000]
    #     scale = 1000.
    # elif env_name == 'antmaze':
    #     import d4rl
    #     env = gym.make(f'{env_name}-{dataset}-v2')
    #     max_ep_len = 1000
    #     env_targets = [1.0, 10.0, 100.0, 1000.0, 100000.0] # successful trajectories have returns of 1, unsuccessful have returns of 0
    #     scale = 1.
    # else:
    #     raise NotImplementedError


def get_model_optimizer(variant, state_dim, act_dim, returns, scale, K, max_ep_len, device):
    T_max = variant['max_iters']*variant['num_steps_per_iter']
    if variant['model_type'] == 'dt':
        model = DecisionTransformer(
            env_name=variant['env'],
            dataset=variant['dataset'],
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'],
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout']
        )
    elif variant['model_type'] == 'pssa':
        from models.decision_spikeformer_pssa import DecisionSpikeFormer
        model = DecisionSpikeFormer(
            env_name=variant['env'],
            dataset=variant['dataset'],
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'],
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            drop_p=variant['dropout'],
            window_size=variant['conv_window_size'],
            activation_function=variant['activation_function'],
            resid_pdrop=variant['dropout'],
            pool_size=variant['pool_size'],
            num_training_steps=variant['max_iters']*variant['num_steps_per_iter'],
            warmup_ratio = variant['warmup_ratio']
        )
    elif variant['model_type'] == 'tssa':
        from models.decision_spikeformer_tssa import DecisionSpikeFormer
        model = DecisionSpikeFormer(
            env_name=variant['env'],
            dataset=variant['dataset'],
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'],
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            drop_p=variant['dropout'],
            window_size=variant['conv_window_size'],
            activation_function=variant['activation_function'],
            resid_pdrop=variant['dropout'],
            pool_size=variant['pool_size'],
            num_training_steps=variant['max_iters']*variant['num_steps_per_iter'],
            warmup_ratio = variant['warmup_ratio']
        )
    else:
        raise NotImplementedError
    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lambda steps: min((steps+1)/warmup_steps, 1)
    # )
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        T_max=T_max,
        warmup_steps=warmup_steps,
        eta_min=0
    )

    return model, optimizer, scheduler

def get_trainer(model_type, **kwargs):
    if model_type == 'dt':
        return DecisionTransformerTrainer(**kwargs)
    elif model_type == 'pssa' or model_type == 'tssa':
        from training.ds_trainer import DecisionSpikeFormerTrainer
        return DecisionSpikeFormerTrainer(**kwargs)
    else:
        raise NotImplementedError

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path