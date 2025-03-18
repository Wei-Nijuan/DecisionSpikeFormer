#!/bin/bash
model_type_list=("pssa")
env_list=("hopper" "walker2d" "halfcheetah")
dataset_list=("medium-replay" "medium-expert" "medium")
pool_size=6
cuda=0
lr=1e-4
max_iters=100
embed_dim=256

# noseed_gym_attn2_dt是local window size设置为2， 默认是设置为3的

for model_type in "${model_type_list[@]}";
do
  setting_name="gym_$model_type"
  log_dir="results/${setting_name}_lr${lr}_iters${max_iters}_dim${embed_dim}"
  echo $log_dir
  num_steps_per_iter=1000
  warmup_steps=$((max_iters*num_steps_per_iter/10))
  # 创建日志文件夹
  mkdir -p $log_dir
  for dataset in "${dataset_list[@]}";
  do
      for env in "${env_list[@]}";
      do
          embed_dim=$embed_dim
          echo $cuda
          echo $env-$dataset-$model_type
          nohup bash -c "CUDA_VISIBLE_DEVICES=$cuda python -u experiment.py \
            --env=$env \
            --dataset=$dataset \
            --model_type=$model_type \
            --batch_size=64 \
            --embed_dim=$embed_dim \
            --warmup_steps=$warmup_steps \
            --max_iters=$max_iters \
            --learning_rate=$lr \
            --setting_name=$setting_name \
            --num_steps_per_iter=$num_steps_per_iter \
            --num_eval_episodes=10 \
            --pool_size=$pool_size \
            > ./$log_dir/$env-$dataset-$model_type.log 2>&1" &
          cuda=$((cuda+1))
          if [ $cuda -eq 4 ]; then
              cuda=0
          fi
      done
  done
done
