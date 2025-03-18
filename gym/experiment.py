import gym
import numpy as np
import torch
import wandb
from d4rl import infos

import argparse
import pickle
import random
import sys
from datetime import datetime
import os

from evaluation.evaluate_episodes import evaluate_episode_rtg, evaluate_episode_rtg_dv, evaluate_episode_rtg_reward, evaluate_episode_rtg_with_time
from utils import discount_cumsum,  get_env_info, get_model_optimizer, get_trainer


def save_checkpoint(state,filename):
    filename = filename
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def experiment(variant):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{env_name}-{dataset}'

    env, max_ep_len, env_targets, scale = get_env_info(env_name, dataset)


    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    if dataset == 'medium-expert': # 这种方法和直接加载medium-expert数据集的效果是等价的
        dataset_path = f'data-gym/{env_name}-expert-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        dataset_path = f'data-gym/{env_name}-medium-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories += pickle.load(f)
        random.shuffle(trajectories)
    elif env_name == 'pen' or env_name == 'door' or env_name == 'relocate' or env_name == 'hammer' or env_name == 'kitchen':
        dataset_path = f'data-gym/{env_name}-{dataset}-v0.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif env_name == 'maze2d':
        dataset_path = f'data-gym/{env_name}-{dataset}-v1.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        dataset_path = f'data-gym/{env_name}-{dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    if mode == 'delayed':
        print("-"*20)
        print('Using delayed reward setting')
        print("-"*20)
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')

    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    # 总共有2186个轨迹，p_sample是这2186个轨迹的采样概率,总和为1
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    prefix = variant['env'] + "_" + variant['dataset']
    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    

    def get_batch(batch_size=256, max_len=K):
        # Dynamically recompute p_sample if online training
        # 从所有轨迹中采样batch_size个轨迹, batch_inds是对应的索引
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True, # 可重复选择同一个轨迹
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1) # 随机选择起始位置

            # get sequences from dataset, 获取轨迹的一部分
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff, 这里使用-1来表示当前正在加入的轨迹
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]: # 补全rtg, 暂时不用care
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def get_batch_back_padding(batch_size=256, max_len=K):
        # Dynamically recompute p_sample if online training
        # 从所有轨迹中采样batch_size个轨迹, batch_inds是对应的索引
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True, # 可重复选择同一个轨迹
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            # si = random.randint(0, traj['rewards'].shape[0] - 1) # 随机选择起始位置
            # we just allow the last K
            while traj['rewards'].shape[0] <= max_len:
                batch_inds[i] = np.random.choice(
                    np.arange(num_trajectories),
                    size=1,
                    replace=True, # 可重复选择同一个轨迹
                    p=p_sample,  # reweights so we sample according to timesteps
                )[0]
                traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, max(0, traj['rewards'].shape[0] - max_len)) # 随机选择起始位置

            # get sequences from dataset, 获取轨迹的一部分
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff, 这里使用-1来表示当前正在加入的轨迹
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]: # 补全rtg, 暂时不用care
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # 我们不允许前向padding，但是我们依旧保留下面这段代码，因为其中有一些对state和rtg的操作
            assert max_len == tlen
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.ones((1, tlen)), np.zeros((1, max_len - tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask


    def eval_episodes(target):
        def fn(model):
            returns, lengths = [], []
            # model_times, other_times = [], []
            # 注意，为了实验的可复现性，我们保留了外层的随机数种子。
            num_seeds = 5 # 设置5个随机数种子，每个随机数种子都进行10次评估, 总共50次评估
            for _ in range(num_seeds):
                # set_seed(random.randint(0, 100000))
                env.seed(random.randint(0, 10000)+100)
                for _ in range(num_eval_episodes):
                    with torch.no_grad():
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device
                        )
                    returns.append(ret)
                    lengths.append(length)
                # model_times.append(model_time)
                # other_times.append(other_time)
            assert len(returns) == num_seeds * num_eval_episodes
            if env_name == 'pen' or env_name == 'door' or env_name == 'relocate' or env_name == 'hammer' or env_name == 'kitchen':
                reward_min = infos.REF_MIN_SCORE[f"{env_name}-{dataset}-v0"]
                reward_max = infos.REF_MAX_SCORE[f"{env_name}-{dataset}-v0"]
            elif env_name == 'maze2d':
                reward_min = infos.REF_MIN_SCORE[f"{env_name}-{dataset}-v1"]
                reward_max = infos.REF_MAX_SCORE[f"{env_name}-{dataset}-v1"]
            else:
                reward_min = infos.REF_MIN_SCORE[f"{env_name}-{dataset}-v2"]
                reward_max = infos.REF_MAX_SCORE[f"{env_name}-{dataset}-v2"]
            return {
                f'target_{target}_return_mean': np.mean(returns),
                f'target_{target}_return_std': np.std(returns),
                f'target_{target}_length_mean': np.mean(lengths),
                f'target_{target}_d4rl_score': (np.mean(returns) - reward_min) * 100 / (reward_max - reward_min),
            }
        return fn

    model, optimizer, scheduler = get_model_optimizer(variant, state_dim, act_dim, returns, scale, K, max_ep_len, device)



    loss_fn = lambda a_hat, a: torch.mean((a_hat - a)**2)

    trainer = get_trainer(
        model_type=variant['model_type'],
        model=model,
        batch_size=batch_size,
        get_batch=get_batch if variant['model_type'] != 'dv' else get_batch_back_padding,
        # get_batch=get_batch,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        eval_fns=[eval_episodes(tar) for tar in env_targets],
        total_steps = variant['max_iters']*variant['num_steps_per_iter'],
    )

    wandb_name = f"{variant['env']}-{variant['dataset']}"
    project_name = '[Decision ConvFormer] Gym'
    group_name = variant['model_type']
        
    if log_to_wandb:
        wandb.init(
            name=wandb_name,
            group=group_name,
            project=project_name,
            config=variant
        )
    # 获取当前时间
    exp_prefix = f"{variant['env']}_{variant['dataset']}_{variant['model_type']}"
    if variant['setting_name'] is not None:
        # exp_prefix = f"{variant['setting_name']}_{exp_prefix}"
        exp_prefix = os.path.join(variant['setting_name'], exp_prefix)
    # 2-stage training
    if variant['model_type'] == 'dv':
        if variant['load_tokenizer']:
            tokenizer_path = variant['tokenizer_path']
            tokenizer_state = torch.load(tokenizer_path, map_location=device)
            model.tokenizer.load_state_dict(tokenizer_state['tokenizer_state_dict'])
        else:
            # stage 0: train tokenizer, simple implementation
            for iter in range(variant['max_iters_token']):
                loss = trainer.train_token_iteration(num_steps=variant['num_steps_per_iter_token'], iter_num=iter+1)
                print(f'Iteration {iter+1}, Tokenizer loss: {loss}')
            # stop tokenizer
            trainer.stop_tokenizer()
            tokenizer_state = {
                'tokenizer_state_dict': model.tokenizer.state_dict(),
            }
            save_checkpoint(tokenizer_state, os.path.join(variant['save_path'], exp_prefix, 'tokenizer'))

    best_ret = -10000
    best_nor_ret = -1000
    best_iter = -1
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)
        ret = outputs['Best_return_mean']
        nor_ret = outputs['Best_normalized_score']
        if ret > best_ret:
            state = {
                'epoch': iter+1,
                'model_state_dict': model.state_dict(),
            }
            save_checkpoint(state, os.path.join(variant['save_path'], exp_prefix, 'epoch_{}.pth'.format(iter + 1)))
            best_ret = ret
            best_nor_ret = nor_ret
            best_iter = iter + 1
        print(f'Current best return mean is {best_ret}, normalized score is {best_nor_ret}, Iteration {best_iter}')
        print("="*80)
    print(f'The final best return mean is {best_ret}')
    print(f'The final best normalized return is {best_nor_ret}')
    print(f'The final best iteration is {best_iter}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse, no-reward-decay
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dc')  
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='gelu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10) #100
    parser.add_argument('--save_path', type=str, default='save')
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--max_iters_token', type=int, default=5) # 实测5个效果 > 10,2, 不加VQVAE也能到30分
    parser.add_argument('--num_steps_per_iter', type=int, default=1000) # 10000
    parser.add_argument('--num_steps_per_iter_token', type=int, default=1000) # 10000
    parser.add_argument('--remove_act_embs', action='store_true')

    parser.add_argument('--load_tokenizer', action='store_true')
    parser.add_argument('--tokenizer_path', type=str) # for loading pretrained tokenizer
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    # bias_window_size for freeFormer
    parser.add_argument('--bias_window_size', type=int, default=8) # bias_window_size, default=8, not use
    # full

    # convolution
    parser.add_argument('--conv_window_size', type=int, default=6)
    # setting name
    parser.add_argument('--setting_name', type=str, default=None)
    # pool size
    parser.add_argument('--pool_size', type=int, default=6)
    # encoder_have_cnn
    parser.add_argument('--encoder_type', type=str, default=None)
    parser.add_argument('--encoder_have_cnn', action='store_true')
    # spike former
    parser.add_argument('--warmup_ratio',  type=float, default=0.10)

    args = parser.parse_args()

    experiment(variant=vars(args))
