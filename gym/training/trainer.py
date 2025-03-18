import numpy as np
import torch

import time


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None,
                 total_steps=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()
        self.step = 0
        self.total_steps = total_steps

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        
        for i in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            if i % 100 == 0:
                print(f'Step {i}')
                print(f"train loss {train_loss}")
                # 打印学习率
                # print(f"lr {self.optimizer.param_groups[0]['lr']}")
                # for k, v in self.diagnostics.items():
                #     print(f'{k}: {v}')

        # 如果self包含gamma_step方法，则调用
        if hasattr(self, 'gamma_step'):
            self.gamma_step()
        logs['time/training'] = time.time() - train_start
        
        eval_start = time.time()
        self.model.eval()

        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/num_of_updates'] = iter_num * num_steps
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/lr'] = self.optimizer.param_groups[0]['lr']
        logs['time/evaluation'] = time.time() - eval_start
        # 打印学习率

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        # if print_logs:
        print('=' * 80)
        print(f'Iteration {iter_num}')
        best_ret = -10000
        best_nor_ret = -10000
        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret = max(best_ret, float(v))
            if 'd4rl_score' in k:
                best_nor_ret = max(best_nor_ret, float(v))
            print(f'{k}: {v}')
        logs['Best_return_mean'] = best_ret
        logs['Best_normalized_score'] = best_nor_ret
        return logs
