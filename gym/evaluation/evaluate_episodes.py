import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

def display_frames_to_video(frames, filename):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    # anim.save(path, writer="ffmpeg")
    anim.save(filename, writer="pillow")
    return anim
def visualize_conv(output, filename):
    import math
    import matplotlib.pyplot as plt
    output = output.cpu().numpy()
    # num_filters = output.shape[2]
    # seq_length = output.shape[1]
    # cols = 8
    # rows = math.ceil(num_filters / cols)
    # plt.figure(figsize=(cols*2, rows*2))
    # for i in range(num_filters):
    #     plt.subplot(rows, cols, i+1)
    #     plt.plot(output[0, :, i])
    #     plt.title(f'Filter {i+1}')
    #     plt.xlabel('Sequence Position')
    #     plt.ylabel('Activation')
    #     plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(filename)
    # output形状是(B, L, D), 在D维度上求均值
    mean_output = np.mean(output, axis=2)[0] # L,
    # 绘制散点图
    x = np.arange(mean_output.shape[0])
    plt.figure(figsize=(10, 5))
    plt.scatter(x, mean_output)
    plt.title("Mean of the output")
    plt.xlabel("Sequence Position")
    plt.ylabel("Activation")
    # plt.grid(True)
    plt.savefig(filename.replace(".png", "_mean.png"))

# 贪心策略
def evaluate_episode_rtg_reward(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):
    """注意action和reward的填充并不会影响states, states并没有填充"""
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    # 加上随机噪声
    batch_returns = target_return.to(dtype=torch.float32).reshape(1, -1, 1).repeat(32, 1, 1)
    batch_returns = batch_returns + torch.randn_like(batch_returns) * 0.1
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        batch_states = (states.to(dtype=torch.float32) - state_mean) / state_std
        batch_states = batch_states.reshape(1, -1, state_dim).repeat(32, 1, 1)
        batch_actions = actions.reshape(1, -1, act_dim).repeat(32, 1, 1)
        batch_timesteps = timesteps.to(dtype=torch.long).reshape(1, -1).repeat(32, 1)

        actions_pred, rewards_pred = model.get_action_with_reward(
            states=batch_states,
            actions=batch_actions,
            returns_to_go=batch_returns,
            timesteps=batch_timesteps,
        )
        # 将rewards_pred视作概率分布选择idx。后面可以尝试不使用概率分布，直接选择最大值
        # rewards_prob = torch.softmax(rewards_pred[:, -1, 0], dim=0)
        # idx = torch.multinomial(rewards_prob, 1)[0]  # 选择一个动作
        idx = torch.argmax(rewards_pred[:, -1, 0])
        action = actions_pred[idx, -1, :]
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = batch_returns[:,-1, 0] - (reward/scale)
        else:
            pred_return = batch_returns[:,-1, 0]
        batch_returns = torch.cat([batch_returns, pred_return.reshape(-1, 1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    """注意action和reward的填充并不会影响states, states并没有填充"""
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        
        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break
    
    return episode_return, episode_length


import time

def evaluate_episode_rtg_with_time(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0

    # 记录模型推理时间
    total_model_inference_time = 0.0
    # 记录其他操作时间
    total_other_operations_time = 0.0


    for t in range(max_ep_len):
        # 模型推理计时开始
        model_inference_start_time = time.perf_counter()

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # 计算推理时间并累加
        torch.cuda.synchronize()
        model_inference_time = time.perf_counter() - model_inference_start_time
        total_model_inference_time += model_inference_time

        # 其他操作的计时开始
        other_operations_start_time = time.perf_counter()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        # 计算其他操作的耗时
        total_other_operations_time += time.perf_counter() - other_operations_start_time

        if done:
            break
    # print(f"Total model inference time: {total_model_inference_time}")
    # print(f"Total other operations time: {total_other_operations_time}")
    return episode_return, episode_length, total_model_inference_time, total_other_operations_time


def evaluate_episode_rtg_dv(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    is_end = False
    for t in range(max_ep_len):
        # B L D
        actions = model.get_actions(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions = actions[0]
        for i in range(actions.shape[0]):
            action = actions[i]
            state, reward, done, _ = env.step(action.detach().cpu().numpy())
            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            if mode == 'normal':
                pred_return = target_return[0,-1] - (reward/scale)
            else:
                pred_return = target_return[0,-1]
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length+1)], dim=1)
            episode_return += reward
            episode_length += 1
            if done or episode_length >= max_ep_len:
                is_end=True
                break
        if is_end:
            break
    return episode_return, episode_length

def visualize_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):
    """注意action和reward的填充并不会影响states, states并没有填充"""
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    frames = []

    episode_return, episode_length = 0, 0
    action_space = env.action_space
    print("Action space:", action_space)
    print("State space:", env.observation_space)

    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        frames.append(env.render(mode="rgb_array"))
        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        print("-"*25)
        print(done)
        print(action)
        print(state)
        print("-"*25)

        if done:
            break

    return episode_return, episode_length, frames


def visualize_episode_cat_dt(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):
    """注意action和reward的填充并不会影响states, states并没有填充"""
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    frames = []

    episode_return, episode_length = 0, 0
    action_space = env.action_space
    print("Action space:", action_space)
    print("State space:", env.observation_space)
    max_length = model.max_length
    attn_entropy_list = []
    rewards_list = []
    step_list  = []



    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        # if t > 50:
        #     returns_to_go = target_return.to(dtype=torch.float32)
        #     states = states.reshape(1, -1, state_dim)
        #     actions = actions.reshape(1, -1, act_dim)
        #     returns_to_go = returns_to_go.reshape(1, -1, 1)
        #     timesteps = timesteps.reshape(1, -1)
        #
        #     states = states[:,-max_length:]
        #     actions = actions[:,-max_length:]
        #     returns_to_go = returns_to_go[:,-max_length:]
        #     timesteps = timesteps[:,-max_length:]
        #
        #     states = torch.cat(
        #         [torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
        #         dim=1).to(dtype=torch.float32)
        #     actions = torch.cat(
        #         [torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
        #         dim=1).to(dtype=torch.float32)
        #     returns_to_go = torch.cat(
        #         [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
        #         dim=1).to(dtype=torch.float32)
        #     timesteps = torch.cat(
        #         [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
        #     ).to(dtype=torch.long)
        #     batch_size, seq_length = states.shape[0], states.shape[1]
        #     time_embeddings = model.embed_timestep(timesteps)
        #     state_embeddings = model.embed_state(states) + time_embeddings
        #     returns_embeddings = model.embed_return(returns_to_go) + time_embeddings
        #     if not model.remove_act_embs:
        #         action_embeddings = model.embed_action(actions) + time_embeddings
        #     # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        #     # which works nice in an autoregressive sense since states predict actions
        #     if model.remove_act_embs:
        #         num_token_type = 2
        #         stacked_inputs = torch.stack(
        #             (returns_embeddings, state_embeddings), dim=1
        #         ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
        #     else:
        #         num_token_type = 3
        #         stacked_inputs = torch.stack(
        #             (returns_embeddings, state_embeddings, action_embeddings), dim=1
        #         ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
        #     outputs = model.cn(stacked_inputs)
        #     outputs = model.embed_ln(outputs)
        #     attention_mask = torch.ones((batch_size, int(seq_length)), dtype=torch.long, device=states.device)
        #
        #     transformer_outputs = model.transformer(
        #         inputs_embeds=outputs,
        #         attention_mask=attention_mask,
        #         output_attentions=True,
        #         output_hidden_states=True,
        #     )
        #     attentions = transformer_outputs['attentions']
        #     attentions_layer1 = attentions[0].squeeze(0).squeeze(0) # (seq_length, seq_length)
        #     attentions_layer2 = attentions[1].squeeze(0).squeeze(0)
        #     attentions_layer3 = attentions[2].squeeze(0).squeeze(0)
        #     hiddens = transformer_outputs['hidden_states']
        #     # hiddens_layer1 = hiddens[0].squeeze(0).squeeze(0).cpu().numpy() # (seq_length, hidden_size)
        #     hiddens_layer1 = hiddens[1].squeeze(0).squeeze(0).cpu().numpy()
        #     hiddens_layer2 = hiddens[2].squeeze(0).squeeze(0).cpu().numpy()
        #     hiddens_layer3 = transformer_outputs['last_hidden_state'].squeeze(0).squeeze(0).cpu().numpy()
        #     # 绘制注意力图
        #     def plot_attention(attentions, title):
        #         import matplotlib.pyplot as plt
        #         import seaborn as sns
        #         # sns.set_theme()
        #         plt.figure(figsize=(10, 10))
        #         sns.heatmap(attentions, cmap='viridis')
        #         plt.title(title)
        #         plt.show()
        #     plot_attention(attentions_layer1.cpu().numpy(), 'Layer 1')
        #     plot_attention(attentions_layer2.cpu().numpy(), 'Layer 2')
        #     plot_attention(attentions_layer3.cpu().numpy(), 'Layer 3')
        #
        #     stacked_inputs = stacked_inputs.cpu().numpy()
        #     outputs = outputs.cpu().numpy()
        #     import numpy as np
        #     import matplotlib.pyplot as plt
        #     from sklearn.manifold import TSNE
        #     import umap.umap_ as umap
        #     # Assuming input_data is (B, 3L, D) and output_data is (B, L, D)
        #     B, L, D =  outputs.shape
        #
        #     # Separate s, a, r from input_data
        #     r_input = stacked_inputs[:, 0::3, :].reshape(B * L, D)
        #     s_input = stacked_inputs[:, 1::3, :].reshape(B * L, D)
        #     a_input = stacked_inputs[:, 2::3, :].reshape(B * L, D)
        #
        #     # Reshape outputs
        #     output_reshaped =  outputs.reshape(B * L, D)
        #
        #     # Apply t-SNE
        #     tsne = TSNE(n_components=2, random_state=42, perplexity=2)
        #     s_embedded_input = tsne.fit_transform(s_input)
        #     a_embedded_input = tsne.fit_transform(a_input)
        #     r_embedded_input = tsne.fit_transform(r_input)
        #     output_embedded = tsne.fit_transform(output_reshaped)
        #     hidden1_embedded = tsne.fit_transform(hiddens_layer1)
        #     hidden2_embedded = tsne.fit_transform(hiddens_layer2)
        #     hidden3_embedded = tsne.fit_transform(hiddens_layer3)
        #     # hidden4_embedded = tsne.fit_transform(hiddens_layer4)
        #
        #     # # Apply UMAP
        #     # reducer = umap.UMAP(random_state=42)
        #     # s_embedded_input_umap = reducer.fit_transform(s_input)
        #     # a_embedded_input_umap = reducer.fit_transform(a_input)
        #     # r_embedded_input_umap = reducer.fit_transform(r_input)
        #     # output_embedded_umap = reducer.fit_transform(output_reshaped)
        #
        #     # Plotting
        #     def plot_embeddings(embedded, title):
        #         # 创建图形
        #         plt.figure(figsize=(10, 8))
        #         # 绘制散点图
        #         plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5, s=1000)
        #         # 为每个点添加标号
        #         for i, (x, y) in enumerate(embedded):
        #             plt.text(x, y, str(i + 1), fontsize=18, ha='center', va='center')  # 标号为 1, 2, 3, ...
        #         # 添加标题
        #         # plt.title(title, fontsize=18)
        #         # plt.tick_params(axis='both', which='major', labelsize=18)
        #         plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        #         # 显示图形
        #         plt.savefig(f"attn-reward/{title}.png")
        #     # plot_embeddings(s_embedded_input, 'States')
        #     # plot_embeddings(a_embedded_input, 'Actions')
        #     # plot_embeddings(r_embedded_input, "RTGs")
        #     # plot_embeddings(output_embedded, 'SAR Embeddings')
        #
        #     import matplotlib.pyplot as plt
        #
        #     def plot_embeddings_subplots(embedded_state, embedded_action, embedded_rtg, embedded_sar, title):
        #         # 创建一个包含 2 行 2 列子图的画布
        #         fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # 增大画布尺寸
        #
        #         # 绘制 State 嵌入
        #         axs[0, 0].scatter(embedded_state[:, 0], embedded_state[:, 1], alpha=0.8, s=300, color='blue')  # 增大点的大小
        #         for i, (x, y) in enumerate(embedded_state):
        #             axs[0, 0].text(x, y, str(i + 1), fontsize=13, ha='center', va='center',
        #                            color='white')  # 增大数字的字体大小
        #         axs[0, 0].set_title('States', fontsize=18)  # 增大标题字体
        #         axs[0, 0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        #
        #         # 绘制 Action 嵌入
        #         axs[0, 1].scatter(embedded_action[:, 0], embedded_action[:, 1], alpha=0.8, s=300, color='green')  # 增大点的大小
        #         for i, (x, y) in enumerate(embedded_action):
        #             axs[0, 1].text(x, y, str(i + 1), fontsize=13, ha='center', va='center',
        #                            color='white')  # 增大数字的字体大小
        #         axs[0, 1].set_title('Actions', fontsize=18)  # 增大标题字体
        #         axs[0, 1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        #
        #         # 绘制 RTG 嵌入
        #         axs[1, 0].scatter(embedded_rtg[:, 0], embedded_rtg[:, 1], alpha=0.8, s=300, color='red')  # 增大点的大小
        #         for i, (x, y) in enumerate(embedded_rtg):
        #             axs[1, 0].text(x, y, str(i + 1), fontsize=13, ha='center', va='center',
        #                            color='white')  # 增大数字的字体大小
        #         axs[1, 0].set_title('RTGs', fontsize=18)  # 增大标题字体
        #         axs[1, 0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        #
        #         # 绘制 SAR 嵌入
        #         axs[1, 1].scatter(embedded_sar[:, 0], embedded_sar[:, 1], alpha=0.8, s=300, color='purple')  # 增大点的大小
        #         for i, (x, y) in enumerate(embedded_sar):
        #             axs[1, 1].text(x, y, str(i + 1), fontsize=13, ha='center', va='center',
        #                            color='white')  # 增大数字的字体大小
        #         axs[1, 1].set_title('SAR Embeddings', fontsize=18)  # 增大标题字体
        #         axs[1, 1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        #
        #         # 调整子图之间的间距
        #         plt.tight_layout()
        #
        #         # 显示图形并保存，设置高分辨率
        #         plt.savefig(f"attn-reward/{title}_subplots.png", dpi=300)  # 增加分辨率
        #         plt.show()
        #
        #     # 假设 s_embedded_input, a_embedded_input, r_embedded_input, output_embedded 都是 2D 数组
        #     plot_embeddings_subplots(s_embedded_input, a_embedded_input, r_embedded_input, output_embedded, 'Combined Embeddings')
        #
        #
        #
        #
        #     # plot_embeddings(hidden1_embedded, 't-SNE of Hidden Layer 1')
        #     # plot_embeddings(hidden2_embedded, 't-SNE of Hidden Layer 2')
        #     # plot_embeddings(hidden3_embedded, 't-SNE of Hidden Layer 3')
        #     # plot_embeddings(hidden4_embedded, 't-SNE of Hidden Layer 4')
        #
        #     # continue
        #     action = model.get_action(
        #         states=(states.to(dtype=torch.float32) - state_mean) / state_std,
        #         actions=actions,
        #         returns_to_go=target_return.to(dtype=torch.float32),
        #         timesteps=timesteps,
        #     )
        #     actions[-1] = action
        #     action = action.detach().cpu().numpy()
        #
        #     frames.append(env.render(mode="rgb_array"))
        #     state, reward, done, _ = env.step(action)
        #     frames.append(env.render(mode="rgb_array"))
        #     # 保存frames, 只取最后的21个， 因为K=20
        #     frames = frames[-21:]
        #     filename = os.path.join("analyze", f"hopper-medium-dt-cnn.gif")
        #     # display_frames_to_video(frames, filename)
        #     print(f"reward:{reward}")
        #     print(f"return:{episode_return}")
        #
        #     # plot_embeddings(s_embedded_input_umap, 'UMAP of States (Before Convolution)')
        #     # plot_embeddings(a_embedded_input_umap, 'UMAP of Actions (Before Convolution)')
        #     # plot_embeddings(r_embedded_input_umap, 'UMAP of RTGs (Before Convolution)')
        #     # plot_embeddings(output_embedded_umap, 'UMAP of Output (After Convolution)')
        #
        #     # from sklearn.metrics import mutual_info_score
        #     # from sklearn.preprocessing import KBinsDiscretizer
        #     # est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        #     # s_input = est.fit_transform(s_input)
        #     # a_input = est.fit_transform(a_input)
        #     # r_input = est.fit_transform(r_input)
        #     # output_reshaped = est.fit_transform(output_reshaped)
        #     #
        #     # mi_sa = mutual_info_score(s_input.flatten(), a_input.flatten())
        #     # mi_sr = mutual_info_score(s_input.flatten(), r_input.flatten())
        #     # mi_ar = mutual_info_score(a_input.flatten(), r_input.flatten())
        #     # print("-"*25, "Before Convolution", "-"*25)
        #     # print(f'Mutual Information (s, a): {mi_sa}')
        #     # print(f'Mutual Information (s, r): {mi_sr}')
        #     # print(f'Mutual Information (a, r): {mi_ar}')
        #     # mi_oa = mutual_info_score(output_reshaped.flatten(), a_input.flatten())
        #     # mi_os = mutual_info_score(output_reshaped.flatten(), s_input.flatten())
        #     # mi_or = mutual_info_score(output_reshaped.flatten(), r_input.flatten())
        #     # print("-"*25, "After Convolution", "-"*25)
        #     # print(f'Mutual Information (o, a): {mi_oa}')
        #     # print(f'Mutual Information (o, s): {mi_os}')
        #     # print(f'Mutual Information (o, r): {mi_or}')
        #     exit(0)
        #     # stacked_inputs = model.cn(stacked_inputs)

        if t % 10 == 0:
            # save
            ori_states = states
            ori_actions = actions
            ori_returns = target_return
            ori_timesteps = timesteps

            returns_to_go = target_return.to(dtype=torch.float32)
            states = states.reshape(1, -1, state_dim)
            actions = actions.reshape(1, -1, act_dim)
            returns_to_go = returns_to_go.reshape(1, -1, 1)
            timesteps = timesteps.reshape(1, -1)

            states = states[:,-max_length:]
            actions = actions[:,-max_length:]
            returns_to_go = returns_to_go[:,-max_length:]
            timesteps = timesteps[:,-max_length:]

            states = torch.cat(
                [torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
            ).to(dtype=torch.long)
            batch_size, seq_length = states.shape[0], states.shape[1]
            # time_embeddings = model.embed_timestep(timesteps)
            # state_embeddings = model.embed_state(states) + time_embeddings
            # returns_embeddings = model.embed_return(returns_to_go) + time_embeddings
            # if not model.remove_act_embs:
            #     action_embeddings = model.embed_action(actions) + time_embeddings
            # # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
            # # which works nice in an autoregressive sense since states predict actions
            # if model.remove_act_embs:
            #     num_token_type = 2
            #     stacked_inputs = torch.stack(
            #         (returns_embeddings, state_embeddings), dim=1
            #     ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
            # else:
            #     num_token_type = 3
            #     stacked_inputs = torch.stack(
            #         (returns_embeddings, state_embeddings, action_embeddings), dim=1
            #     ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
            # outputs = model.cn(stacked_inputs)

            actions = torch.cat([torch.zeros_like(actions[:,0:1]), actions[:,:-1]], dim=1)
            embeddings = model.embed(torch.cat([states, actions, returns_to_go], dim=-1))

            time_embeddings = model.embed_timestep(timesteps)

            # time embeddings are treated similar to positional embeddings
            # state_embeddings = state_embeddings + time_embeddings
            # action_embeddings = action_embeddings + time_embeddings
            # returns_embeddings = returns_embeddings + time_embeddings
            outputs = embeddings + time_embeddings
            outputs = model.embed_ln(outputs)
            attention_mask = torch.ones((batch_size, int(seq_length)), dtype=torch.long, device=states.device)

            transformer_outputs = model.transformer(
                inputs_embeds=outputs,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )
            attentions = transformer_outputs['attentions']
            attentions_layer1 = attentions[0].squeeze(0).squeeze(0) # (seq_length, seq_length)
            # 计算熵
            torch.clip_(attentions_layer1, 1e-12, 1.0)
            entropy = -torch.sum(attentions_layer1 * torch.log(attentions_layer1), dim=1)
            attn_entropy_list.append(torch.mean(entropy).item())
            def plot_attention(attentions, title):
                import matplotlib.pyplot as plt
                import seaborn as sns
                # sns.set_theme()
                plt.figure(figsize=(10, 10))
                # sns.heatmap(attentions, cmap='viridis')
                # sns.heatmap(attentions, cmap='Blues', annot=False)
                sns.heatmap(attentions, cmap='Blues', annot=False, xticklabels=False,
                            yticklabels=False, cbar=False, square=True)
                plt.title(title)
                # plt.show()
                plt.savefig(f"attn2/{t}.png")
            # plot_attention(attentions_layer1.cpu().numpy(), 'Layer 1')
            print(t)
            # load
            states = ori_states
            actions = ori_actions
            target_return = ori_returns
            timesteps = ori_timesteps

        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # frames.append(env.render(mode="rgb_array"))
        img = env.render(mode="rgb_array")
        frames.append(img)
        if t % 10 == 0:
            filename = os.path.join("loco_cnndt", f"{t}.png")
            # 保存图片
            img = Image.fromarray(img)
            img.save(filename)
            # plt.show()
        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        # print("-"*25)
        # print(done)
        # print(action)
        # print(state)
        # print("-"*25)
        if t % 10 == 0:
            print(f"reward:{reward}")
            print(f"return:{episode_return}")
            rewards_list.append(reward)
            step_list.append(t)


        if done:
            break
    # 绘制图片 x:steps , 绘制 attention entropy和rewards两条线
    import matplotlib.pyplot as plt
    import seaborn as sns
    assert len(attn_entropy_list) == len(rewards_list)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=step_list, y=attn_entropy_list, label='Attention Entropy')
    sns.lineplot(x=step_list, y=rewards_list, label='Rewards')
    plt.title("Attention Entropy and Rewards")
    plt.savefig("attn-reward/dt-cat.png")
    print("attn_entropy_list:", attn_entropy_list)
    print("rewards_list:", rewards_list)
    print("step_list:", step_list)
    print("Average Entropy:", np.mean(attn_entropy_list))
    print("Average Reward:", np.mean(rewards_list))
    print("Return:", episode_return)

    return episode_return, episode_length, frames
def visualize_episode_cnn_dt(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):
    """注意action和reward的填充并不会影响states, states并没有填充"""
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    frames = []

    episode_return, episode_length = 0, 0
    action_space = env.action_space
    print("Action space:", action_space)
    print("State space:", env.observation_space)
    max_length = model.max_length
    attn_entropy_list = []
    rewards_list = []
    step_list  = []



    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        if t > 100:
            returns_to_go = target_return.to(dtype=torch.float32)
            states = states.reshape(1, -1, state_dim)
            actions = actions.reshape(1, -1, act_dim)
            returns_to_go = returns_to_go.reshape(1, -1, 1)
            timesteps = timesteps.reshape(1, -1)

            states = states[:,-max_length:]
            actions = actions[:,-max_length:]
            returns_to_go = returns_to_go[:,-max_length:]
            timesteps = timesteps[:,-max_length:]

            states = torch.cat(
                [torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
            ).to(dtype=torch.long)
            batch_size, seq_length = states.shape[0], states.shape[1]
            time_embeddings = model.embed_timestep(timesteps)
            state_embeddings = model.embed_state(states) + time_embeddings
            returns_embeddings = model.embed_return(returns_to_go) + time_embeddings
            if not model.remove_act_embs:
                action_embeddings = model.embed_action(actions) + time_embeddings
            # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
            # which works nice in an autoregressive sense since states predict actions
            if model.remove_act_embs:
                num_token_type = 2
                stacked_inputs = torch.stack(
                    (returns_embeddings, state_embeddings), dim=1
                ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
            else:
                num_token_type = 3
                stacked_inputs = torch.stack(
                    (returns_embeddings, state_embeddings, action_embeddings), dim=1
                ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
            outputs = model.cn(stacked_inputs)
            outputs = model.embed_ln(outputs)
            attention_mask = torch.ones((batch_size, int(seq_length)), dtype=torch.long, device=states.device)

            transformer_outputs = model.transformer(
                inputs_embeds=outputs,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )
            attentions = transformer_outputs['attentions']
            attentions_layer1 = attentions[0].squeeze(0).squeeze(0) # (seq_length, seq_length)
            attentions_layer2 = attentions[1].squeeze(0).squeeze(0)
            attentions_layer3 = attentions[2].squeeze(0).squeeze(0)
            hiddens = transformer_outputs['hidden_states']
            # hiddens_layer1 = hiddens[0].squeeze(0).squeeze(0).cpu().numpy() # (seq_length, hidden_size)
            hiddens_layer1 = hiddens[1].squeeze(0).squeeze(0).cpu().numpy()
            hiddens_layer2 = hiddens[2].squeeze(0).squeeze(0).cpu().numpy()
            hiddens_layer3 = transformer_outputs['last_hidden_state'].squeeze(0).squeeze(0).cpu().numpy()
            # 绘制注意力图
            def plot_attention(attentions, title):
                import matplotlib.pyplot as plt
                import seaborn as sns
                # sns.set_theme()
                plt.figure(figsize=(10, 10))
                sns.heatmap(attentions, cmap='viridis')
                plt.title(title)
                plt.show()
            plot_attention(attentions_layer1.cpu().numpy(), 'Layer 1')
            plot_attention(attentions_layer2.cpu().numpy(), 'Layer 2')
            plot_attention(attentions_layer3.cpu().numpy(), 'Layer 3')

            stacked_inputs = stacked_inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            import umap.umap_ as umap
            # Assuming input_data is (B, 3L, D) and output_data is (B, L, D)
            B, L, D =  outputs.shape

            # Separate s, a, r from input_data
            r_input = stacked_inputs[:, 0::3, :].reshape(B * L, D)
            s_input = stacked_inputs[:, 1::3, :].reshape(B * L, D)
            a_input = stacked_inputs[:, 2::3, :].reshape(B * L, D)
            s_input = s_input[ :20, :]
            a_input = a_input[ :20, :]
            r_input = r_input[ :20, :]

            # Reshape outputs
            output_reshaped =  outputs.reshape(B * L, D)
            output_reshaped = output_reshaped[ :10, :]

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=2)
            s_embedded_input = tsne.fit_transform(s_input)
            a_embedded_input = tsne.fit_transform(a_input)
            r_embedded_input = tsne.fit_transform(r_input)
            output_embedded = tsne.fit_transform(output_reshaped)
            hidden1_embedded = tsne.fit_transform(hiddens_layer1)
            hidden2_embedded = tsne.fit_transform(hiddens_layer2)
            hidden3_embedded = tsne.fit_transform(hiddens_layer3)
            # hidden4_embedded = tsne.fit_transform(hiddens_layer4)

            # # Apply UMAP
            # reducer = umap.UMAP(random_state=42)
            # s_embedded_input_umap = reducer.fit_transform(s_input)
            # a_embedded_input_umap = reducer.fit_transform(a_input)
            # r_embedded_input_umap = reducer.fit_transform(r_input)
            # output_embedded_umap = reducer.fit_transform(output_reshaped)

            # Plotting
            def plot_embeddings(embedded, title):
                # 创建图形
                plt.figure(figsize=(10, 8))
                # 绘制散点图
                plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5, s=1000)
                # 为每个点添加标号
                for i, (x, y) in enumerate(embedded):
                    plt.text(x, y, str(i + 1), fontsize=18, ha='center', va='center')  # 标号为 1, 2, 3, ...
                # 添加标题
                # plt.title(title, fontsize=18)
                # plt.tick_params(axis='both', which='major', labelsize=18)
                plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                # 显示图形
                plt.savefig(f"attn-reward/{title}.png")
            # plot_embeddings(s_embedded_input, 'States')
            # plot_embeddings(a_embedded_input, 'Actions')
            # plot_embeddings(r_embedded_input, "RTGs")
            # plot_embeddings(output_embedded, 'SAR Embeddings')

            import matplotlib.pyplot as plt

            def plot_embeddings_subplots(embedded_state, embedded_action, embedded_rtg, embedded_sar, title):
                # 创建一个包含 2 行 2 列子图的画布
                import matplotlib as mpl
                # 配置全局样式参数
                # mpl.rcParams['font.family'] = 'Times New Roman'  # 学术论文常用字体
                fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # 增大画布尺寸
                point_size = 800
                number_size = 18
                title_size = 20

                # 绘制 State 嵌入
                axs[0, 0].scatter(embedded_state[:, 0], embedded_state[:, 1], alpha=0.8, s=point_size, color='blue')  # 增大点的大小
                for i, (x, y) in enumerate(embedded_state):
                    axs[0, 0].text(x, y, str(i + 1), fontsize=number_size, ha='center', va='center',
                                   color='white')  # 增大数字的字体大小
                axs[0, 0].set_title('States', fontsize=title_size)  # 增大标题字体
                axs[0, 0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

                # 绘制 Action 嵌入
                axs[0, 1].scatter(embedded_action[:, 0], embedded_action[:, 1], alpha=0.8, s=point_size, color='green')  #
                # 增大点的大小
                for i, (x, y) in enumerate(embedded_action):
                    axs[0, 1].text(x, y, str(i + 1), fontsize=number_size, ha='center', va='center',
                                   color='white')  # 增大数字的字体大小
                axs[0, 1].set_title('Actions', fontsize=title_size)  # 增大标题字体
                axs[0, 1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

                # 绘制 RTG 嵌入
                axs[1, 0].scatter(embedded_rtg[:, 0], embedded_rtg[:, 1], alpha=0.8, s=point_size, color='red')  # 增大点的大小
                for i, (x, y) in enumerate(embedded_rtg):
                    axs[1, 0].text(x, y, str(i + 1), fontsize=number_size, ha='center', va='center',
                                   color='white')  # 增大数字的字体大小
                axs[1, 0].set_title('RTGs', fontsize=title_size)  # 增大标题字体
                axs[1, 0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

                # 绘制 SAR 嵌入
                axs[1, 1].scatter(embedded_sar[:, 0], embedded_sar[:, 1], alpha=0.8, s=point_size, color='purple')  # 增大点的大小
                for i, (x, y) in enumerate(embedded_sar):
                    axs[1, 1].text(x, y, str(i + 1), fontsize=number_size, ha='center', va='center',
                                   color='white')  # 增大数字的字体大小
                axs[1, 1].set_title('SAR Embeddings', fontsize=title_size)  # 增大标题字体
                axs[1, 1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

                # 调整子图之间的间距
                plt.tight_layout()

                # 显示图形并保存，设置高分辨率
                plt.savefig(f"attn-reward/{title}_subplots.png", dpi=300)  # 增加分辨率
                plt.show()
            def plot_embeddings_subplots2(embedded_state, embedded_action, embedded_rtg, embedded_sar, title):
                import matplotlib as mpl
                from matplotlib.lines import Line2D  # 新增线条绘制模块

                # 配置全局样式参数
                mpl.rcParams['font.family'] = 'Times New Roman'

                # 创建 1 行 4 列的子图布局
                fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 调整画布为宽屏格式
                plt.subplots_adjust(wspace=0.3)  # 增加子图水平间距

                point_size = 800
                number_size = 18
                title_size = 20

                # 绘制 State 嵌入（第1列）
                axs[0].scatter(embedded_state[:, 0], embedded_state[:, 1], alpha=0.8, s=point_size, color='blue')
                for i, (x, y) in enumerate(embedded_state):
                    axs[0].text(x, y, str(i + 1), fontsize=number_size, ha='center', va='center', color='white')
                axs[0].set_title('States', fontsize=title_size)
                # axs[0].text(0.5, 0.5, 'States', fontsize=title_size, ha='center', va='center', color='blue',
                #             transform=axs[0].transAxes)
                axs[0].tick_params(axis='both', which='both', bottom=False, labelbottom=False,
                                   left=False, labelleft=False)

                # 绘制 Action 嵌入（第2列）
                axs[1].scatter(embedded_action[:, 0], embedded_action[:, 1], alpha=0.8, s=point_size, color='green')
                for i, (x, y) in enumerate(embedded_action):
                    axs[1].text(x, y, str(i + 1), fontsize=number_size, ha='center', va='center', color='white')
                axs[1].set_title('Actions', fontsize=title_size)
                # axs[1].text(0.5, 0.5, 'Actions', fontsize=title_size, ha='center', va='center', color='green',
                #             transform=axs[1].transAxes)
                axs[1].tick_params(axis='both', which='both', bottom=False, labelbottom=False,
                                   left=False, labelleft=False)

                # 绘制 RTG 嵌入（第3列）
                axs[2].scatter(embedded_rtg[:, 0], embedded_rtg[:, 1], alpha=0.8, s=point_size, color='red')
                for i, (x, y) in enumerate(embedded_rtg):
                    axs[2].text(x, y, str(i + 1), fontsize=number_size, ha='center', va='center', color='white')
                axs[2].set_title('RTGs', fontsize=title_size)
                # axs[2].text(0.5, 0.5, 'RTGs', fontsize=title_size, ha='center', va='center', color='red',
                #             transform=axs[1].transAxes)
                axs[2].tick_params(axis='both', which='both', bottom=False, labelbottom=False,
                                   left=False, labelleft=False)

                # 绘制 SAR 嵌入（第4列）
                axs[3].scatter(embedded_sar[:, 0], embedded_sar[:, 1], alpha=0.8, s=point_size, color='purple')
                for i, (x, y) in enumerate(embedded_sar):
                    axs[3].text(x, y, str(i + 1), fontsize=number_size, ha='center', va='center', color='white')
                axs[3].set_title('SARs', fontsize=title_size)  # 修改标题保持一致性
                # axs[3].text(0.5, 0.5, 'Merged Token', fontsize=title_size, ha='center', va='center', color='purple',
                #             transform=axs[1].transAxes)
                axs[3].tick_params(axis='both', which='both', bottom=False, labelbottom=False,
                                   left=False, labelleft=False)

                # 在第3、4列之间添加虚线
                plt.tight_layout()
                pos3 = axs[2].get_position()
                pos4 = axs[3].get_position()
                line_x = (pos3.x1 + pos4.x0) / 2  # 计算中间位置
                line = Line2D([line_x, line_x], [pos3.y0, pos3.y1],
                              linestyle='--', linewidth=2,
                              color='gray', alpha=0.7,
                              transform=fig.transFigure)
                fig.add_artist(line)
                for spine in fig.get_axes()[0].spines.values():
                    spine.set_linewidth(2)  # 调整边框宽度

                for ax in axs:
                    for spine in ax.spines.values():
                        spine.set_linewidth(2)  # 设置每个子图的边框宽度

                plt.savefig(f"attn-reward/{title}_subplots.png", dpi=300, bbox_inches='tight')
                plt.show()

            # 假设 s_embedded_input, a_embedded_input, r_embedded_input, output_embedded 都是 2D 数组
            print("s_embedded_input=", s_embedded_input.tolist())
            print("a_embedded_input=", a_embedded_input.tolist())
            print("r_embedded_input=", r_embedded_input.tolist())
            print("output_embedded=", output_embedded.tolist())

            plot_embeddings_subplots2(s_embedded_input, a_embedded_input, r_embedded_input, output_embedded,
                                     'TokenMerger100-2')




# plot_embeddings(hidden1_embedded, 't-SNE of Hidden Layer 1')
            # plot_embeddings(hidden2_embedded, 't-SNE of Hidden Layer 2')
            # plot_embeddings(hidden3_embedded, 't-SNE of Hidden Layer 3')
            # plot_embeddings(hidden4_embedded, 't-SNE of Hidden Layer 4')

            # continue
            action = model.get_action(
                states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                actions=actions,
                returns_to_go=target_return.to(dtype=torch.float32),
                timesteps=timesteps,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            frames.append(env.render(mode="rgb_array"))
            state, reward, done, _ = env.step(action)
            frames.append(env.render(mode="rgb_array"))
            # 保存frames, 只取最后的21个， 因为K=20
            frames = frames[-21:]
            filename = os.path.join("analyze", f"hopper-medium-dt-cnn.gif")
            # display_frames_to_video(frames, filename)
            print(f"reward:{reward}")
            print(f"return:{episode_return}")

            # plot_embeddings(s_embedded_input_umap, 'UMAP of States (Before Convolution)')
            # plot_embeddings(a_embedded_input_umap, 'UMAP of Actions (Before Convolution)')
            # plot_embeddings(r_embedded_input_umap, 'UMAP of RTGs (Before Convolution)')
            # plot_embeddings(output_embedded_umap, 'UMAP of Output (After Convolution)')

            # from sklearn.metrics import mutual_info_score
            # from sklearn.preprocessing import KBinsDiscretizer
            # est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            # s_input = est.fit_transform(s_input)
            # a_input = est.fit_transform(a_input)
            # r_input = est.fit_transform(r_input)
            # output_reshaped = est.fit_transform(output_reshaped)
            #
            # mi_sa = mutual_info_score(s_input.flatten(), a_input.flatten())
            # mi_sr = mutual_info_score(s_input.flatten(), r_input.flatten())
            # mi_ar = mutual_info_score(a_input.flatten(), r_input.flatten())
            # print("-"*25, "Before Convolution", "-"*25)
            # print(f'Mutual Information (s, a): {mi_sa}')
            # print(f'Mutual Information (s, r): {mi_sr}')
            # print(f'Mutual Information (a, r): {mi_ar}')
            # mi_oa = mutual_info_score(output_reshaped.flatten(), a_input.flatten())
            # mi_os = mutual_info_score(output_reshaped.flatten(), s_input.flatten())
            # mi_or = mutual_info_score(output_reshaped.flatten(), r_input.flatten())
            # print("-"*25, "After Convolution", "-"*25)
            # print(f'Mutual Information (o, a): {mi_oa}')
            # print(f'Mutual Information (o, s): {mi_os}')
            # print(f'Mutual Information (o, r): {mi_or}')
            exit(0)
            # stacked_inputs = model.cn(stacked_inputs)

        # if t % 10 == 0:
        #     # save
        #     ori_states = states
        #     ori_actions = actions
        #     ori_returns = target_return
        #     ori_timesteps = timesteps
        #
        #     returns_to_go = target_return.to(dtype=torch.float32)
        #     states = states.reshape(1, -1, state_dim)
        #     actions = actions.reshape(1, -1, act_dim)
        #     returns_to_go = returns_to_go.reshape(1, -1, 1)
        #     timesteps = timesteps.reshape(1, -1)
        #
        #     states = states[:,-max_length:]
        #     actions = actions[:,-max_length:]
        #     returns_to_go = returns_to_go[:,-max_length:]
        #     timesteps = timesteps[:,-max_length:]
        #
        #     states = torch.cat(
        #         [torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
        #         dim=1).to(dtype=torch.float32)
        #     actions = torch.cat(
        #         [torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
        #         dim=1).to(dtype=torch.float32)
        #     returns_to_go = torch.cat(
        #         [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
        #         dim=1).to(dtype=torch.float32)
        #     timesteps = torch.cat(
        #         [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
        #     ).to(dtype=torch.long)
        #     batch_size, seq_length = states.shape[0], states.shape[1]
        #     time_embeddings = model.embed_timestep(timesteps)
        #     state_embeddings = model.embed_state(states) + time_embeddings
        #     returns_embeddings = model.embed_return(returns_to_go) + time_embeddings
        #     if not model.remove_act_embs:
        #         action_embeddings = model.embed_action(actions) + time_embeddings
        #     # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        #     # which works nice in an autoregressive sense since states predict actions
        #     if model.remove_act_embs:
        #         num_token_type = 2
        #         stacked_inputs = torch.stack(
        #             (returns_embeddings, state_embeddings), dim=1
        #         ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
        #     else:
        #         num_token_type = 3
        #         stacked_inputs = torch.stack(
        #             (returns_embeddings, state_embeddings, action_embeddings), dim=1
        #         ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
        #     outputs = model.cn(stacked_inputs)
        #     outputs = model.embed_ln(outputs)
        #     attention_mask = torch.ones((batch_size, int(seq_length)), dtype=torch.long, device=states.device)
        #
        #     transformer_outputs = model.transformer(
        #         inputs_embeds=outputs,
        #         attention_mask=attention_mask,
        #         output_attentions=True,
        #         output_hidden_states=True,
        #     )
        #     attentions = transformer_outputs['attentions']
        #     attentions_layer1 = attentions[0].squeeze(0).squeeze(0) # (seq_length, seq_length)
        #     # 计算熵
        #     torch.clip_(attentions_layer1, 1e-12, 1.0)
        #     entropy = -torch.sum(attentions_layer1 * torch.log(attentions_layer1), dim=1)
        #     attn_entropy_list.append(torch.mean(entropy).item())
        #     def plot_attention(attentions, title):
        #         import matplotlib.pyplot as plt
        #         import seaborn as sns
        #         # sns.set_theme()
        #         plt.figure(figsize=(10, 10))
        #         # sns.heatmap(attentions, cmap='viridis')
        #         # sns.heatmap(attentions, cmap='Blues', annot=False)
        #         sns.heatmap(attentions, cmap='Blues', annot=False, xticklabels=False,
        #                     yticklabels=False, cbar=False, square=True)
        #         plt.title(title)
        #         # plt.show()
        #         plt.savefig(f"attn2/{t}.png")
        #     plot_attention(attentions_layer1.cpu().numpy(), 'Layer 1')
        #     print(t)
        #     # load
        #     states = ori_states
        #     actions = ori_actions
        #     target_return = ori_returns
        #     timesteps = ori_timesteps

        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # frames.append(env.render(mode="rgb_array"))
        img = env.render(mode="rgb_array")
        frames.append(img)
        # if t % 10 == 0:
        #     filename = os.path.join("loco_cnndt", f"{t}.png")
        #     # 保存图片
        #     img = Image.fromarray(img)
        #     img.save(filename)
        #     # plt.show()
        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        # print("-"*25)
        # print(done)
        # print(action)
        # print(state)
        # print("-"*25)
        # if t % 10 == 0:
        #     print(f"reward:{reward}")
        #     print(f"return:{episode_return}")
        #     rewards_list.append(reward)
        #     step_list.append(t)


        if done:
            break
    # 绘制图片 x:steps , 绘制 attention entropy和rewards两条线
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # assert len(attn_entropy_list) == len(rewards_list)
    # plt.figure(figsize=(10, 5))
    # sns.lineplot(x=step_list, y=attn_entropy_list, label='Attention Entropy')
    # sns.lineplot(x=step_list, y=rewards_list, label='Rewards')
    # plt.title("Attention Entropy and Rewards")
    # plt.savefig("attn-reward/dt-cnn.png")
    import matplotlib.pyplot as plt
    import numpy as np

    # 创建数据（确保step_list, attn_entropy_list 和 rewards_list 变量已经定义）
    # 例如：step_list = np.arange(len(attn_entropy_list)) 这假设step_list是步骤索引。

    # 创建图形
    plt.figure(figsize=(10, 5))

    # 绘制堆叠折线图
    plt.stackplot(step_list, attn_entropy_list, rewards_list, labels=['Attention Entropy', 'Rewards'], alpha=0.6)

    # 设置标题和标签
    plt.title("Attention Entropy and Rewards", fontsize=16)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Values', fontsize=14)

    # 添加图例
    plt.legend(title='Metrics', loc='upper left', fontsize=12, title_fontsize=14)

    # 添加网格线
    plt.grid(True)

    # 保存图像
    plt.tight_layout()
    plt.savefig("attn-reward/dt-cnn-stacked.png")

    print("attn_entropy_list:", attn_entropy_list)
    print("rewards_list:", rewards_list)
    print("step_list:", step_list)
    print("Average Entropy:", np.mean(attn_entropy_list))
    print("Average Reward:", np.mean(rewards_list))
    print("Return:", episode_return)

    return episode_return, episode_length, frames


def visualize_episode_ds(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):
    """注意action和reward的填充并不会影响states, states并没有填充"""
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    frames = []

    episode_return, episode_length = 0, 0
    action_space = env.action_space
    print("Action space:", action_space)
    print("State space:", env.observation_space)
    max_length = model.max_length
    attn_entropy_list = []
    rewards_list = []
    step_list = []
    t_list = [97,  217, 413]

    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        if t in t_list :
            # save
            ori_states = states
            ori_actions = actions
            ori_returns = target_return
            ori_timesteps = timesteps

            returns_to_go = target_return.to(dtype=torch.float32)
            states = states.reshape(1, -1, state_dim)
            actions = actions.reshape(1, -1, act_dim)
            returns_to_go = returns_to_go.reshape(1, -1, 1)
            timesteps = timesteps.reshape(1, -1)

            states = states[:,-max_length:]
            actions = actions[:,-max_length:]
            returns_to_go = returns_to_go[:,-max_length:]
            timesteps = timesteps[:,-max_length:]
            attention_mask = torch.cat([torch.zeros(max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
            ).to(dtype=torch.long)
            batch_size, seq_length = states.shape[0], states.shape[1]

            actions = torch.cat([torch.zeros_like(actions[:,0:1]), actions[:,:-1]], dim=1)
            if attention_mask is None:
                # attention mask for GPT: 1 if can be attended to, 0 if not
                attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)
            # embed each modality with a different head
            # state_embeddings = self.embed_state(states)
            # action_embeddings = self.embed_action(actions)
            # returns_embeddings = self.embed_return(returns_to_go)
            # time embeddings are treated similar to positional embeddings
            # state_embeddings = state_embeddings + time_embeddings
            # action_embeddings = action_embeddings + time_embeddings
            # returns_embeddings = returns_embeddings + time_embeddings
            embeddings = model.embed(torch.cat([states, actions, returns_to_go], dim=-1))
            stacked_inputs = model.embed_ln(embeddings) # +time_embeddings
            # we feed in the input embeddings (not word indices as in NLP) to the model
            transformer_outputs = model.transformer(
                stacked_inputs,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            # transformer_outputs = model.transformer(
            #     inputs_embeds=inputs,
            #     attention_mask=attention_mask,
            #     output_attentions=True,
            #     output_hidden_states=True,
            # )
            attentions = transformer_outputs['attentions']
            print(attentions[0].shape)
            attentions_layer1 = attentions[0].squeeze(0).squeeze(0).squeeze(0) # (seq_length, seq_length)
            # print(attentions_layer1.shape)
            attentions_layer2 = attentions[1].squeeze(0).squeeze(0).squeeze(0)
            attentions_layer3 = attentions[2].squeeze(0).squeeze(0).squeeze(0)
            # 沿着最后一个维度进行Softmax操作
            attentions_layer1_norm = torch.nn.functional.softmax(attentions_layer1, dim=-1)
            attentions_layer2_norm = torch.nn.functional.softmax(attentions_layer2, dim=-1)
            attentions_layer3_norm = torch.nn.functional.softmax(attentions_layer3, dim=-1)
            hiddens = transformer_outputs['hidden_states']
            print(hiddens[0].shape)
            hiddens_layer1 = hiddens[0][0].squeeze(0).squeeze(0).cpu().numpy() # (seq_length, hidden_size)
            hiddens_layer2 = hiddens[0][1].squeeze(0).squeeze(0).cpu().numpy()
            hiddens_layer3 = hiddens[0][2].squeeze(0).squeeze(0).cpu().numpy()
            hiddens_layer4 = hiddens[0][3].squeeze(0).squeeze(0).cpu().numpy()
            # model.transformer.transformer.print_fire_rate()
            # hiddens_layer4 = transformer_outputs['last_hidden_state'].squeeze(0).squeeze(0).cpu().numpy()[1::3, :]
            # 绘制注意力图
            def plot_attention(attentions, title):
                import matplotlib.pyplot as plt
                import seaborn as sns
                # sns.set_theme()
                plt.figure(figsize=(10, 10))
                sns.heatmap(attentions, cmap="viridis", annot=False, xticklabels=False,yticklabels=False)
                # plt.title(title)
                # plt.show()
                plt.savefig(f"attn-spike/ds_{title}_{t}.png")
            # plot_attention(attentions_layer1.cpu().numpy(), 'Layer 1')
            # plot_attention(attentions_layer2.cpu().numpy(), 'Layer 2')
            # plot_attention(attentions_layer3.cpu().numpy(), 'Layer 3')
            # plot_attention(attentions_layer1_norm.cpu().numpy(), 'Layer1 Norm')
            # plot_attention(attentions_layer2_norm.cpu().numpy(), 'Layer2 Norm')
            # plot_attention(attentions_layer3_norm.cpu().numpy(), 'Layer3 Norm')
            # load
            states = ori_states
            actions = ori_actions
            target_return = ori_returns
            timesteps = ori_timesteps
            print(1532)
            # exit(0)

            # exit(0)

            stacked_inputs = stacked_inputs.cpu().numpy()
            inputs = stacked_inputs
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            # Assuming input_data is (B, 3L, D) and output_data is (B, L, D)
            # B, L, D =  state_embeddings.shape

            # Separate s, a, r from input_data
            # r_input = stacked_inputs[:, 0::3, :].reshape(B * L, D)
            # s_input = stacked_inputs[:, 1::3, :].reshape(B * L, D)
            # a_input = stacked_inputs[:, 2::3, :].reshape(B * L, D)

            # Reshape outputs
            B, L, D = inputs.shape
            output_reshaped =  inputs.reshape(B * L, D)
            # x = x.reshape(batch_size, seq_length, 3, model.hidden_size).permute(0, 2, 1, 3)

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=2)
            # s_embedded_input = tsne.fit_transform(s_input)
            # a_embedded_input = tsne.fit_transform(a_input)
            # r_embedded_input = tsne.fit_transform(r_input)
            print(1561)
            print(output_reshaped.shape)
            print(hiddens_layer1.shape)
            print(hiddens_layer2.shape)
            print(hiddens_layer3.shape)
            print(output_reshaped)
            # output_embedded = tsne.fit_transform(output_reshaped)
            print(1566)
            hidden1_embedded = tsne.fit_transform(hiddens_layer1)
            print(1568)
            hidden2_embedded = tsne.fit_transform(hiddens_layer2)
            print(1570)
            hidden3_embedded = tsne.fit_transform(hiddens_layer3)
            print(1569)
            hidden4_embedded = tsne.fit_transform(hiddens_layer4)
            # hidden4_embedded = tsne.fit_transform(hiddens_layer4)

            # # Apply UMAP
            # reducer = umap.UMAP(random_state=42)
            # s_embedded_input_umap = reducer.fit_transform(s_input)
            # a_embedded_input_umap = reducer.fit_transform(a_input)
            # r_embedded_input_umap = reducer.fit_transform(r_input)
            # output_embedded_umap = reducer.fit_transform(output_reshaped)

            # Plotting
            def plot_embeddings(embedded, title):
                # 创建图形
                plt.figure(figsize=(10, 8))
                # 绘制散点图
                plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.8, s=1000, color='blue')
                # 为每个点添加标号
                for i, (x, y) in enumerate(embedded):
                    plt.text(x, y, str(i + 1), fontsize=18, ha='center', va='center', color='white')  # 标号为
                    # 1, 2, 3, ...
                # 删除坐标轴及其刻度
                ax = plt.gca()
                ax.set_xticks([])  # 删除 x 轴刻度
                ax.set_yticks([])  # 删除 y 轴刻度
                ax.set_xticklabels([])  # 删除 x 轴刻度标签
                ax.set_yticklabels([])  # 删除 y 轴刻度标签
                # ax.spines['top'].set_visible(False)  # 隐藏顶部边框
                # ax.spines['bottom'].set_visible(False)  # 隐藏底部边框
                # ax.spines['left'].set_visible(False)  # 隐藏左侧边框
                # ax.spines['right'].set_visible(False)  # 隐藏右侧边框
                # 显示图形
                plt.savefig(f"attn_spike_embed/ds_{title}_{t}.png")
            # plot_embeddings(s_embedded_input, 't-SNE of States (Before Convolution)')
            # plot_embeddings(a_embedded_input, 't-SNE of Actions (Before Convolution)')
            # plot_embeddings(r_embedded_input, 't-SNE of RTGs (Before Convolution)')
            print(1591)
            # plot_embeddings(output_embedded, 't-SNE of Input')
            plot_embeddings(hidden1_embedded, 'TimeStep 1')
            plot_embeddings(hidden2_embedded, 'TimeStep 2')
            plot_embeddings(hidden3_embedded, 'TimeStep 3')
            plot_embeddings(hidden4_embedded, 'TimeStep 4')

            # continue
            # action = model.get_action(
            #     states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            #     actions=actions,
            #     returns_to_go=target_return.to(dtype=torch.float32),
            #     timesteps=timesteps,
            # )
            # actions[-1] = action
            # action = action.detach().cpu().numpy()
            #
            # frames.append(env.render(mode="rgb_array"))
            # state, reward, done, _ = env.step(action)
            # frames.append(env.render(mode="rgb_array"))
            # # 保存frames, 只取最后的21个， 因为K=20
            # frames = frames[-21:]
            # filename = os.path.join("analyze", f"hopper-medium-expert-dt.gif")
            # # display_frames_to_video(frames, filename)
            # print(f"reward:{reward}")
            # print(f"return:{episode_return}")
            #
            # plot_embeddings(s_embedded_input_umap, 'UMAP of States (Before Convolution)')
            # plot_embeddings(a_embedded_input_umap, 'UMAP of Actions (Before Convolution)')
            # plot_embeddings(r_embedded_input_umap, 'UMAP of RTGs (Before Convolution)')
            # plot_embeddings(output_embedded_umap, 'UMAP of Output (After Convolution)')
            #
            # from sklearn.metrics import mutual_info_score
            # from sklearn.preprocessing import KBinsDiscretizer
            # est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            # s_input = est.fit_transform(s_input)
            # a_input = est.fit_transform(a_input)
            # r_input = est.fit_transform(r_input)
            # output_reshaped = est.fit_transform(output_reshaped)
            #
            # mi_sa = mutual_info_score(s_input.flatten(), a_input.flatten())
            # mi_sr = mutual_info_score(s_input.flatten(), r_input.flatten())
            # mi_ar = mutual_info_score(a_input.flatten(), r_input.flatten())
            # print("-"*25, "Before Convolution", "-"*25)
            # print(f'Mutual Information (s, a): {mi_sa}')
            # print(f'Mutual Information (s, r): {mi_sr}')
            # print(f'Mutual Information (a, r): {mi_ar}')
            # mi_oa = mutual_info_score(output_reshaped.flatten(), a_input.flatten())
            # mi_os = mutual_info_score(output_reshaped.flatten(), s_input.flatten())
            # mi_or = mutual_info_score(output_reshaped.flatten(), r_input.flatten())
            # print("-"*25, "After Convolution", "-"*25)
            # print(f'Mutual Information (o, a): {mi_oa}')
            # print(f'Mutual Information (o, s): {mi_os}')
            # print(f'Mutual Information (o, r): {mi_or}')
            # stacked_inputs = model.cn(stacked_inputs)
            # exit(0)

        # if t%10==0:
        #     # save
        #     ori_states = states
        #     ori_actions = actions
        #     ori_returns = target_return
        #     ori_timesteps = timesteps
        #
        #     returns_to_go = target_return.to(dtype=torch.float32)
        #     states = states.reshape(1, -1, state_dim)
        #     actions = actions.reshape(1, -1, act_dim)
        #     returns_to_go = returns_to_go.reshape(1, -1, 1)
        #     timesteps = timesteps.reshape(1, -1)
        #
        #     states = states[:,-max_length:]
        #     actions = actions[:,-max_length:]
        #     returns_to_go = returns_to_go[:,-max_length:]
        #     timesteps = timesteps[:,-max_length:]
        #
        #     states = torch.cat(
        #         [torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
        #         dim=1).to(dtype=torch.float32)
        #     actions = torch.cat(
        #         [torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
        #         dim=1).to(dtype=torch.float32)
        #     returns_to_go = torch.cat(
        #         [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
        #         dim=1).to(dtype=torch.float32)
        #     timesteps = torch.cat(
        #         [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
        #     ).to(dtype=torch.long)
        #     batch_size, seq_length = states.shape[0], states.shape[1]
        #     time_embeddings = model.embed_timestep(timesteps)
        #     state_embeddings = model.embed_state(states) + time_embeddings
        #     returns_embeddings = model.embed_return(returns_to_go) + time_embeddings
        #     if not model.remove_act_embs:
        #         action_embeddings = model.embed_action(actions) + time_embeddings
        #     # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        #     # which works nice in an autoregressive sense since states predict actions
        #     if model.remove_act_embs:
        #         num_token_type = 2
        #         stacked_inputs = torch.stack(
        #             (returns_embeddings, state_embeddings), dim=1
        #         ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
        #     else:
        #         num_token_type = 3
        #         stacked_inputs = torch.stack(
        #             (returns_embeddings, state_embeddings, action_embeddings), dim=1
        #         ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
        #     # outputs = model.cn(stacked_inputs)
        #     inputs = stacked_inputs
        #     inputs = model.embed_ln(inputs)
        #     attention_mask = torch.ones((batch_size, int(seq_length)), dtype=torch.long, device=states.device)
        #     attention_masks = (attention_mask, attention_mask, attention_mask)
        #     stacked_attention_mask = torch.stack(
        #         attention_masks, dim=1
        #     ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        #     transformer_outputs = model.transformer(
        #         inputs_embeds=inputs,
        #         attention_mask=stacked_attention_mask,
        #         output_attentions=True,
        #         output_hidden_states=True,
        #     )
        #     attentions = transformer_outputs['attentions']
        #     attentions_layer1 = attentions[0].squeeze(0).squeeze(0) # (seq_length, seq_length)
        #     # 计算熵
        #     torch.clip_(attentions_layer1, 1e-12, 1.0)
        #     entropy = -torch.sum(attentions_layer1 * torch.log(attentions_layer1), dim=1)
        #     attn_entropy_list.append(torch.mean(entropy).item())
        #     # 绘制注意力图
        #     def plot_attention(attentions, title):
        #         import matplotlib.pyplot as plt
        #         import seaborn as sns
        #         # sns.set_theme()
        #         plt.figure(figsize=(10, 10))
        #         attentions=attentions[:20,:20]
        #         # sns.heatmap(attentions, cmap='viridis')
        #         sns.heatmap(attentions, cmap='Blues', annot=False, xticklabels=False,
        #                     yticklabels=False, cbar=False, square=True)
        #         # 移除坐标轴
        #         # ax.set(xlabel=None, ylabel=None)
        #         # ax.tick_params(left=False, bottom=False)
        #         plt.title(title)
        #         # plt.show()
        #         plt.savefig(f"attn/{t}_20.png")
        #     print(t)
        #     plot_attention(attentions_layer1.cpu().numpy(), f"Step {t}")
        #
        #     # load
        #     states = ori_states
        #     actions = ori_actions
        #     target_return = ori_returns
        #     timesteps = ori_timesteps


        print(1741)
        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # img = env.render(mode="rgb_array")
        # frames.append(img)
        # if t in t_list:
        #         filename = os.path.join("loco_dt_spike", f"{t}.png")
        #         # 保存图片
        #         img = Image.fromarray(img)
        #         img.save(filename)

        print(1759)
        state, reward, done, _ = env.step(action)
        if t > 30:
            print("-"*15)
            print(f"{t} reward:{reward}")
            print("-"*15)


        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        # if t % 10 == 0:
        #     print(f"reward:{reward}")
        #     print(f"return:{episode_return}")
        #     rewards_list.append(reward)
        #     step_list.append(t)
        # print("-"*25)
        # print(done)
        # print(action)
        # print(state)
        # print("-"*25)

        if done:
            break
    # 绘制图片
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # # 设置学术会议常见字体和字号
    # # 设置学术会议常见字体和字号
    # plt.rcParams.update({
    #     'font.family': 'serif',  # 设置字体为衬线字体 (Times New Roman)
    #     'font.serif': ['Times New Roman'],  # 确保使用 Times New Roman 字体
    #     'axes.labelsize': 12,  # 设置轴标签字体大小
    #     'xtick.labelsize': 10,  # 设置 X 轴刻度标签字体大小
    #     'ytick.labelsize': 10,  # 设置 Y 轴刻度标签字体大小
    #     'legend.fontsize': 10,  # 设置图例字体大小
    #     'figure.figsize': (12, 7),  # 设置图表大小
    #     'axes.titlesize': 14,  # 设置标题字体大小
    #     'axes.titlepad': 20,  # 设置标题与图表的间距
    # })
    # plt.figure(figsize=(10, 5))
    # sns.lineplot(x=step_list, y=attn_entropy_list, label='Attention Entropy')
    # sns.lineplot(x=step_list, y=rewards_list, label='Env Rewards')
    # # plt.title("Attention Entropy and Rewards")
    # plt.savefig("attn-reward/dt.png")
    # print("attn_entropy_list=", attn_entropy_list)
    # print("rewards_list=", rewards_list)
    # print("step_list=", step_list)
    # print("Average Entropy:", np.mean(attn_entropy_list))
    # print("Average Reward:", np.mean(rewards_list))
    # print("Return:", episode_return)
    #

    return episode_return, episode_length, frames

def visualize_episode_dt(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):
    """注意action和reward的填充并不会影响states, states并没有填充"""
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    frames = []

    episode_return, episode_length = 0, 0
    action_space = env.action_space
    print("Action space:", action_space)
    print("State space:", env.observation_space)
    max_length = model.max_length
    attn_entropy_list = []
    rewards_list = []
    step_list = []

    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        if t > 100:
            returns_to_go = target_return.to(dtype=torch.float32)
            states = states.reshape(1, -1, state_dim)
            actions = actions.reshape(1, -1, act_dim)
            returns_to_go = returns_to_go.reshape(1, -1, 1)
            timesteps = timesteps.reshape(1, -1)

            states = states[:,-max_length:]
            actions = actions[:,-max_length:]
            returns_to_go = returns_to_go[:,-max_length:]
            timesteps = timesteps[:,-max_length:]

            states = torch.cat(
                [torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
            ).to(dtype=torch.long)
            batch_size, seq_length = states.shape[0], states.shape[1]
            time_embeddings = model.embed_timestep(timesteps)
            state_embeddings = model.embed_state(states) + time_embeddings
            returns_embeddings = model.embed_return(returns_to_go) + time_embeddings
            if not model.remove_act_embs:
                action_embeddings = model.embed_action(actions) + time_embeddings
            # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
            # which works nice in an autoregressive sense since states predict actions
            if model.remove_act_embs:
                num_token_type = 2
                stacked_inputs = torch.stack(
                    (returns_embeddings, state_embeddings), dim=1
                ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
            else:
                num_token_type = 3
                stacked_inputs = torch.stack(
                    (returns_embeddings, state_embeddings, action_embeddings), dim=1
                ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
            # outputs = model.cn(stacked_inputs)
            inputs = stacked_inputs
            inputs = model.embed_ln(inputs)
            attention_mask = torch.ones((batch_size, int(seq_length)), dtype=torch.long, device=states.device)
            attention_masks = (attention_mask, attention_mask, attention_mask)
            stacked_attention_mask = torch.stack(
                attention_masks, dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
            transformer_outputs = model.transformer(
                inputs_embeds=inputs,
                attention_mask=stacked_attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )
            attentions = transformer_outputs['attentions']
            attentions_layer1 = attentions[0].squeeze(0).squeeze(0) # (seq_length, seq_length)
            attentions_layer2 = attentions[1].squeeze(0).squeeze(0)
            attentions_layer3 = attentions[2].squeeze(0).squeeze(0)
            hiddens = transformer_outputs['hidden_states']
            hiddens_layer1 = hiddens[0].squeeze(0).squeeze(0).cpu().numpy()[1::3, :] # (seq_length, hidden_size)
            hiddens_layer2 = hiddens[1].squeeze(0).squeeze(0).cpu().numpy()[1::3, :]
            hiddens_layer3 = hiddens[2].squeeze(0).squeeze(0).cpu().numpy()[1::3, :]
            # hiddens_layer4 = transformer_outputs['last_hidden_state'].squeeze(0).squeeze(0).cpu().numpy()[1::3, :]
            # 绘制注意力图
            def plot_attention(attentions, title):
                import matplotlib.pyplot as plt
                import seaborn as sns
                # sns.set_theme()
                plt.figure(figsize=(10, 10))
                sns.heatmap(attentions, cmap='viridis')
                plt.title(title)
                plt.show()
            plot_attention(attentions_layer1.cpu().numpy(), 'Layer 1')
            plot_attention(attentions_layer2.cpu().numpy(), 'Layer 2')
            plot_attention(attentions_layer3.cpu().numpy(), 'Layer 3')

            stacked_inputs = stacked_inputs.cpu().numpy()
            inputs = inputs.cpu().numpy()
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            import umap.umap_ as umap
            # Assuming input_data is (B, 3L, D) and output_data is (B, L, D)
            B, L, D =  state_embeddings.shape

            # Separate s, a, r from input_data
            r_input = stacked_inputs[:, 0::3, :].reshape(B * L, D)
            s_input = stacked_inputs[:, 1::3, :].reshape(B * L, D)
            a_input = stacked_inputs[:, 2::3, :].reshape(B * L, D)

            # Reshape outputs
            # output_reshaped =  inputs.reshape(B * L, D)
            # x = x.reshape(batch_size, seq_length, 3, model.hidden_size).permute(0, 2, 1, 3)

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=2)
            s_embedded_input = tsne.fit_transform(s_input)
            a_embedded_input = tsne.fit_transform(a_input)
            r_embedded_input = tsne.fit_transform(r_input)
            # output_embedded = tsne.fit_transform(output_reshaped)
            hidden1_embedded = tsne.fit_transform(hiddens_layer1)
            hidden2_embedded = tsne.fit_transform(hiddens_layer2)
            hidden3_embedded = tsne.fit_transform(hiddens_layer3)
            # hidden4_embedded = tsne.fit_transform(hiddens_layer4)

            # # Apply UMAP
            # reducer = umap.UMAP(random_state=42)
            # s_embedded_input_umap = reducer.fit_transform(s_input)
            # a_embedded_input_umap = reducer.fit_transform(a_input)
            # r_embedded_input_umap = reducer.fit_transform(r_input)
            # output_embedded_umap = reducer.fit_transform(output_reshaped)

            # Plotting
            def plot_embeddings(embedded, title):
                # 创建图形
                plt.figure(figsize=(10, 8))
                # 绘制散点图
                plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.8, s=1000, color='green')
                # 为每个点添加标号
                for i, (x, y) in enumerate(embedded):
                    plt.text(x, y, str(i + 1), fontsize=18, ha='center', va='center', color='white')  # 标号为 1, 2, 3, ...
                # 添加标题
                plt.title(title, fontsize=18)
                plt.tick_params(axis='both', which='major', labelsize=18)
                ax = plt.gca()
                ax.set_xticks([])  # 删除 x 轴刻度
                ax.set_yticks([])  # 删除 y 轴刻度
                ax.set_xticklabels([])  # 删除 x 轴刻度标签
                ax.set_yticklabels([])  # 删除 y 轴刻度标签
                # 显示图形
                plt.savefig(f"attn-reward/{title}.png")
            plot_embeddings(s_embedded_input, 't-SNE of States (Before Convolution)')
            plot_embeddings(a_embedded_input, 't-SNE of Actions (Before Convolution)')
            plot_embeddings(r_embedded_input, 't-SNE of RTGs (Before Convolution)')
            # plot_embeddings(output_embedded, 't-SNE of Output (After Convolution)')
            plot_embeddings(hidden1_embedded, 't-SNE of Hidden Layer 1')
            plot_embeddings(hidden2_embedded, 't-SNE of Hidden Layer 2')
            plot_embeddings(hidden3_embedded, 't-SNE of Hidden Layer 3')
            # plot_embeddings(hidden4_embedded, 't-SNE of Hidden Layer 4')

            # continue
            # action = model.get_action(
            #     states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            #     actions=actions,
            #     returns_to_go=target_return.to(dtype=torch.float32),
            #     timesteps=timesteps,
            # )
            # actions[-1] = action
            # action = action.detach().cpu().numpy()
            #
            # frames.append(env.render(mode="rgb_array"))
            # state, reward, done, _ = env.step(action)
            # frames.append(env.render(mode="rgb_array"))
            # # 保存frames, 只取最后的21个， 因为K=20
            # frames = frames[-21:]
            # filename = os.path.join("analyze", f"hopper-medium-expert-dt.gif")
            # # display_frames_to_video(frames, filename)
            # print(f"reward:{reward}")
            # print(f"return:{episode_return}")

            # plot_embeddings(s_embedded_input_umap, 'UMAP of States (Before Convolution)')
            # plot_embeddings(a_embedded_input_umap, 'UMAP of Actions (Before Convolution)')
            # plot_embeddings(r_embedded_input_umap, 'UMAP of RTGs (Before Convolution)')
            # plot_embeddings(output_embedded_umap, 'UMAP of Output (After Convolution)')

            # from sklearn.metrics import mutual_info_score
            # from sklearn.preprocessing import KBinsDiscretizer
            # est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            # s_input = est.fit_transform(s_input)
            # a_input = est.fit_transform(a_input)
            # r_input = est.fit_transform(r_input)
            # output_reshaped = est.fit_transform(output_reshaped)
            #
            # mi_sa = mutual_info_score(s_input.flatten(), a_input.flatten())
            # mi_sr = mutual_info_score(s_input.flatten(), r_input.flatten())
            # mi_ar = mutual_info_score(a_input.flatten(), r_input.flatten())
            # print("-"*25, "Before Convolution", "-"*25)
            # print(f'Mutual Information (s, a): {mi_sa}')
            # print(f'Mutual Information (s, r): {mi_sr}')
            # print(f'Mutual Information (a, r): {mi_ar}')
            # mi_oa = mutual_info_score(output_reshaped.flatten(), a_input.flatten())
            # mi_os = mutual_info_score(output_reshaped.flatten(), s_input.flatten())
            # mi_or = mutual_info_score(output_reshaped.flatten(), r_input.flatten())
            # print("-"*25, "After Convolution", "-"*25)
            # print(f'Mutual Information (o, a): {mi_oa}')
            # print(f'Mutual Information (o, s): {mi_os}')
            # print(f'Mutual Information (o, r): {mi_or}')
            exit(0)
            # stacked_inputs = model.cn(stacked_inputs)

        # if t%10==0:
        #     # save
        #     ori_states = states
        #     ori_actions = actions
        #     ori_returns = target_return
        #     ori_timesteps = timesteps
        #
        #     returns_to_go = target_return.to(dtype=torch.float32)
        #     states = states.reshape(1, -1, state_dim)
        #     actions = actions.reshape(1, -1, act_dim)
        #     returns_to_go = returns_to_go.reshape(1, -1, 1)
        #     timesteps = timesteps.reshape(1, -1)
        #
        #     states = states[:,-max_length:]
        #     actions = actions[:,-max_length:]
        #     returns_to_go = returns_to_go[:,-max_length:]
        #     timesteps = timesteps[:,-max_length:]
        #
        #     states = torch.cat(
        #         [torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
        #         dim=1).to(dtype=torch.float32)
        #     actions = torch.cat(
        #         [torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
        #         dim=1).to(dtype=torch.float32)
        #     returns_to_go = torch.cat(
        #         [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
        #         dim=1).to(dtype=torch.float32)
        #     timesteps = torch.cat(
        #         [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
        #     ).to(dtype=torch.long)
        #     batch_size, seq_length = states.shape[0], states.shape[1]
        #     time_embeddings = model.embed_timestep(timesteps)
        #     state_embeddings = model.embed_state(states) + time_embeddings
        #     returns_embeddings = model.embed_return(returns_to_go) + time_embeddings
        #     if not model.remove_act_embs:
        #         action_embeddings = model.embed_action(actions) + time_embeddings
        #     # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        #     # which works nice in an autoregressive sense since states predict actions
        #     if model.remove_act_embs:
        #         num_token_type = 2
        #         stacked_inputs = torch.stack(
        #             (returns_embeddings, state_embeddings), dim=1
        #         ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
        #     else:
        #         num_token_type = 3
        #         stacked_inputs = torch.stack(
        #             (returns_embeddings, state_embeddings, action_embeddings), dim=1
        #         ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
        #     # outputs = model.cn(stacked_inputs)
        #     inputs = stacked_inputs
        #     inputs = model.embed_ln(inputs)
        #     attention_mask = torch.ones((batch_size, int(seq_length)), dtype=torch.long, device=states.device)
        #     attention_masks = (attention_mask, attention_mask, attention_mask)
        #     stacked_attention_mask = torch.stack(
        #         attention_masks, dim=1
        #     ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        #     transformer_outputs = model.transformer(
        #         inputs_embeds=inputs,
        #         attention_mask=stacked_attention_mask,
        #         output_attentions=True,
        #         output_hidden_states=True,
        #     )
        #     attentions = transformer_outputs['attentions']
        #     attentions_layer1 = attentions[0].squeeze(0).squeeze(0) # (seq_length, seq_length)
        #     # 计算熵
        #     torch.clip_(attentions_layer1, 1e-12, 1.0)
        #     entropy = -torch.sum(attentions_layer1 * torch.log(attentions_layer1), dim=1)
        #     attn_entropy_list.append(torch.mean(entropy).item())
        #     # 绘制注意力图
        #     def plot_attention(attentions, title):
        #         import matplotlib.pyplot as plt
        #         import seaborn as sns
        #         # sns.set_theme()
        #         plt.figure(figsize=(10, 10))
        #         attentions=attentions[:20,:20]
        #         # sns.heatmap(attentions, cmap='viridis')
        #         sns.heatmap(attentions, cmap='Blues', annot=False, xticklabels=False,
        #                     yticklabels=False, cbar=False, square=True)
        #         # 移除坐标轴
        #         # ax.set(xlabel=None, ylabel=None)
        #         # ax.tick_params(left=False, bottom=False)
        #         plt.title(title)
        #         # plt.show()
        #         plt.savefig(f"attn/{t}_20.png")
        #     print(t)
        #     plot_attention(attentions_layer1.cpu().numpy(), f"Step {t}")
        #
        #     # load
        #     states = ori_states
        #     actions = ori_actions
        #     target_return = ori_returns
        #     timesteps = ori_timesteps


        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        img = env.render(mode="rgb_array")
        frames.append(img)
        # if t % 10 == 0:
        #         filename = os.path.join("loco_dt", f"{t}.png")
        #         # 保存图片
        #         img = Image.fromarray(img)
        #         img.save(filename)

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        # if t % 10 == 0:
        #     print(f"reward:{reward}")
        #     print(f"return:{episode_return}")
        #     rewards_list.append(reward)
        #     step_list.append(t)
        # print("-"*25)
        # print(done)
        # print(action)
        # print(state)
        # print("-"*25)

        if done:
            break
    # 绘制图片
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # # 设置学术会议常见字体和字号
    # # 设置学术会议常见字体和字号
    # plt.rcParams.update({
    #     'font.family': 'serif',  # 设置字体为衬线字体 (Times New Roman)
    #     'font.serif': ['Times New Roman'],  # 确保使用 Times New Roman 字体
    #     'axes.labelsize': 12,  # 设置轴标签字体大小
    #     'xtick.labelsize': 10,  # 设置 X 轴刻度标签字体大小
    #     'ytick.labelsize': 10,  # 设置 Y 轴刻度标签字体大小
    #     'legend.fontsize': 10,  # 设置图例字体大小
    #     'figure.figsize': (12, 7),  # 设置图表大小
    #     'axes.titlesize': 14,  # 设置标题字体大小
    #     'axes.titlepad': 20,  # 设置标题与图表的间距
    # })
    # plt.figure(figsize=(10, 5))
    # sns.lineplot(x=step_list, y=attn_entropy_list, label='Attention Entropy')
    # sns.lineplot(x=step_list, y=rewards_list, label='Env Rewards')
    # # plt.title("Attention Entropy and Rewards")
    # plt.savefig("attn-reward/dt.png")
    # print("attn_entropy_list=", attn_entropy_list)
    # print("rewards_list=", rewards_list)
    # print("step_list=", step_list)
    # print("Average Entropy:", np.mean(attn_entropy_list))
    # print("Average Reward:", np.mean(rewards_list))
    # print("Return:", episode_return)
    #

    return episode_return, episode_length, frames
def visualize_episode_cnn_dp(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):
    """注意action和reward的填充并不会影响states, states并没有填充"""
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    frames = []

    episode_return, episode_length = 0, 0
    action_space = env.action_space
    print("Action space:", action_space)
    print("State space:", env.observation_space)
    max_length = model.max_length

    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        if t > 50:
            returns_to_go = target_return.to(dtype=torch.float32)
            states = states.reshape(1, -1, state_dim)
            actions = actions.reshape(1, -1, act_dim)
            returns_to_go = returns_to_go.reshape(1, -1, 1)
            timesteps = timesteps.reshape(1, -1)

            states = states[:,-max_length:]
            actions = actions[:,-max_length:]
            returns_to_go = returns_to_go[:,-max_length:]
            timesteps = timesteps[:,-max_length:]

            states = torch.cat(
                [torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
            ).to(dtype=torch.long)
            batch_size, seq_length = states.shape[0], states.shape[1]
            time_embeddings = model.embed_timestep(timesteps)
            state_embeddings = model.embed_state(states) + time_embeddings
            returns_embeddings = model.embed_return(returns_to_go) + time_embeddings
            if not model.remove_act_embs:
                action_embeddings = model.embed_action(actions) + time_embeddings
            # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
            # which works nice in an autoregressive sense since states predict actions
            if model.remove_act_embs:
                num_token_type = 2
                stacked_inputs = torch.stack(
                    (returns_embeddings, state_embeddings), dim=1
                ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
            else:
                num_token_type = 3
                stacked_inputs = torch.stack(
                    (returns_embeddings, state_embeddings, action_embeddings), dim=1
                ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, model.hidden_size)
            inputs = model.cn(stacked_inputs)
            inputs = model.embed_ln(inputs)
            attention_mask = torch.ones((batch_size, int(seq_length)), dtype=torch.long, device=states.device)

            transformer_outputs, hiddens = model.transformer(
                inputs_embeds=inputs,
                output_hidden_states=True,
            )
            hiddens_layer1 = hiddens[0].squeeze(0).squeeze(0).cpu().numpy() # (seq_length, hidden_size)
            hiddens_layer2 = hiddens[1].squeeze(0).squeeze(0).cpu().numpy()
            hiddens_layer3 = hiddens[2].squeeze(0).squeeze(0).cpu().numpy()
            hiddens_layer4 = transformer_outputs.squeeze(0).squeeze(0).cpu().numpy()
            stacked_inputs = stacked_inputs.cpu().numpy()
            inputs = inputs.cpu().numpy()
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            import umap.umap_ as umap
            # Assuming input_data is (B, 3L, D) and output_data is (B, L, D)
            B, L, D =  inputs.shape

            # Separate s, a, r from input_data
            r_input = stacked_inputs[:, 0::3, :].reshape(B * L, D)
            s_input = stacked_inputs[:, 1::3, :].reshape(B * L, D)
            a_input = stacked_inputs[:, 2::3, :].reshape(B * L, D)

            # Reshape outputs
            output_reshaped =  inputs.reshape(B * L, D)

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=2)
            s_embedded_input = tsne.fit_transform(s_input)
            a_embedded_input = tsne.fit_transform(a_input)
            r_embedded_input = tsne.fit_transform(r_input)
            output_embedded = tsne.fit_transform(output_reshaped)
            hidden1_embedded = tsne.fit_transform(hiddens_layer1)
            hidden2_embedded = tsne.fit_transform(hiddens_layer2)
            hidden3_embedded = tsne.fit_transform(hiddens_layer3)
            hidden4_embedded = tsne.fit_transform(hiddens_layer4)

            # # Apply UMAP
            # reducer = umap.UMAP(random_state=42)
            # s_embedded_input_umap = reducer.fit_transform(s_input)
            # a_embedded_input_umap = reducer.fit_transform(a_input)
            # r_embedded_input_umap = reducer.fit_transform(r_input)
            # output_embedded_umap = reducer.fit_transform(output_reshaped)

            # Plotting
            def plot_embeddings(embedded, title):
                # 创建图形
                plt.figure(figsize=(10, 8))
                # 绘制散点图
                plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5, s=1000)
                # 为每个点添加标号
                for i, (x, y) in enumerate(embedded):
                    plt.text(x, y, str(i + 1), fontsize=18, ha='center', va='center')  # 标号为 1, 2, 3, ...
                # 添加标题
                plt.title(title, fontsize=18)
                plt.tick_params(axis='both', which='major', labelsize=18)
                # 显示图形
                # plt.show()
                plt.savefig(f"attn-reward/dp_{title}.png")
            plot_embeddings(s_embedded_input, 't-SNE of States (Before Convolution)')
            plot_embeddings(a_embedded_input, 't-SNE of Actions (Before Convolution)')
            plot_embeddings(r_embedded_input, 't-SNE of RTGs (Before Convolution)')
            plot_embeddings(output_embedded, 't-SNE of Output (After Convolution)')
            plot_embeddings(hidden1_embedded, 't-SNE of Hidden Layer 1')
            plot_embeddings(hidden2_embedded, 't-SNE of Hidden Layer 2')
            plot_embeddings(hidden3_embedded, 't-SNE of Hidden Layer 3')
            plot_embeddings(hidden4_embedded, 't-SNE of Hidden Layer 4')


            def plot_embeddings_subplots(embedded_state, embedded_action, embedded_rtg, embedded_sar, title):
                # 创建一个包含 2 行 2 列子图的画布
                fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # 增大画布尺寸

                # 绘制 State 嵌入
                axs[0, 0].scatter(embedded_state[:, 0], embedded_state[:, 1], alpha=0.8, s=300, color='blue')  # 增大点的大小
                for i, (x, y) in enumerate(embedded_state):
                    axs[0, 0].text(x, y, str(i + 1), fontsize=13, ha='center', va='center',
                                   color='white')  # 增大数字的字体大小
                axs[0, 0].set_title('SAR Embeddings', fontsize=18)  # 增大标题字体
                axs[0, 0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

                # 绘制 Action 嵌入
                axs[0, 1].scatter(embedded_action[:, 0], embedded_action[:, 1], alpha=0.8, s=300, color='green')  # 增大点的大小
                for i, (x, y) in enumerate(embedded_action):
                    axs[0, 1].text(x, y, str(i + 1), fontsize=13, ha='center', va='center',
                                   color='white')  # 增大数字的字体大小
                axs[0, 1].set_title('Pooling Layer 1', fontsize=18)  # 增大标题字体
                axs[0, 1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

                # 绘制 RTG 嵌入
                axs[1, 0].scatter(embedded_rtg[:, 0], embedded_rtg[:, 1], alpha=0.8, s=300, color='red')  # 增大点的大小
                for i, (x, y) in enumerate(embedded_rtg):
                    axs[1, 0].text(x, y, str(i + 1), fontsize=13, ha='center', va='center',
                                   color='white')  # 增大数字的字体大小
                axs[1, 0].set_title('Pooling Layer 2', fontsize=18)  # 增大标题字体
                axs[1, 0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

                # 绘制 SAR 嵌入
                axs[1, 1].scatter(embedded_sar[:, 0], embedded_sar[:, 1], alpha=0.8, s=300, color='purple')  # 增大点的大小
                for i, (x, y) in enumerate(embedded_sar):
                    axs[1, 1].text(x, y, str(i + 1), fontsize=13, ha='center', va='center',
                                   color='white')  # 增大数字的字体大小
                axs[1, 1].set_title('Pooling Layer 3', fontsize=18)  # 增大标题字体
                axs[1, 1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

                # 调整子图之间的间距
                plt.tight_layout()

                # 显示图形并保存，设置高分辨率
                plt.savefig(f"attn-reward/{title}_subplots.png", dpi=300)  # 增加分辨率
                plt.show()

            # 假设 s_embedded_input, a_embedded_input, r_embedded_input, output_embedded 都是 2D 数组
            plot_embeddings_subplots(output_embedded,hidden1_embedded , hidden2_embedded, hidden3_embedded,
                                     'Pooling Layers')



            # continue
            action = model.get_action(
                states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                actions=actions,
                returns_to_go=target_return.to(dtype=torch.float32),
                timesteps=timesteps,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            frames.append(env.render(mode="rgb_array"))
            state, reward, done, _ = env.step(action)
            frames.append(env.render(mode="rgb_array"))
            # 保存frames, 只取最后的21个， 因为K=20
            frames = frames[-21:]
            filename = os.path.join("analyze", f"hopper-medium-replay-dp-cnn.gif")
            display_frames_to_video(frames, filename)
            print(f"reward:{reward}")
            print(f"return:{episode_return}")

            # plot_embeddings(s_embedded_input_umap, 'UMAP of States (Before Convolution)')
            # plot_embeddings(a_embedded_input_umap, 'UMAP of Actions (Before Convolution)')
            # plot_embeddings(r_embedded_input_umap, 'UMAP of RTGs (Before Convolution)')
            # plot_embeddings(output_embedded_umap, 'UMAP of Output (After Convolution)')

            # from sklearn.metrics import mutual_info_score
            # from sklearn.preprocessing import KBinsDiscretizer
            # est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            # s_input = est.fit_transform(s_input)
            # a_input = est.fit_transform(a_input)
            # r_input = est.fit_transform(r_input)
            # output_reshaped = est.fit_transform(output_reshaped)
            #
            # mi_sa = mutual_info_score(s_input.flatten(), a_input.flatten())
            # mi_sr = mutual_info_score(s_input.flatten(), r_input.flatten())
            # mi_ar = mutual_info_score(a_input.flatten(), r_input.flatten())
            # print("-"*25, "Before Convolution", "-"*25)
            # print(f'Mutual Information (s, a): {mi_sa}')
            # print(f'Mutual Information (s, r): {mi_sr}')
            # print(f'Mutual Information (a, r): {mi_ar}')
            # mi_oa = mutual_info_score(output_reshaped.flatten(), a_input.flatten())
            # mi_os = mutual_info_score(output_reshaped.flatten(), s_input.flatten())
            # mi_or = mutual_info_score(output_reshaped.flatten(), r_input.flatten())
            # print("-"*25, "After Convolution", "-"*25)
            # print(f'Mutual Information (o, a): {mi_oa}')
            # print(f'Mutual Information (o, s): {mi_os}')
            # print(f'Mutual Information (o, r): {mi_or}')
            exit(0)
            # stacked_inputs = model.cn(stacked_inputs)

        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        frames.append(env.render(mode="rgb_array"))
        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        # print("-"*25)
        # print(done)
        # print(action)
        # print(state)
        # print("-"*25)

        if done:
            break

    return episode_return, episode_length, frames
