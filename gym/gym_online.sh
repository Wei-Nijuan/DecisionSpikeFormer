#!/bin/bash
#model_type_list=("dm" "df" "dp" "dc" "dret")
model_type_list=("tssa" "pssa")
env_list=("hopper" "walker2d")
dataset_list=("medium-replay" "medium-expert" "medium")
pool_size=6
cuda=0
lr=5e-3
max_iters=20
embed_dim=128
seed=0

# noseed_gym_attn2_dt是local window size设置为2， 默认是设置为3的

for encoder_have_cnn in "${encoder_have_cnn_list[@]}";
do
    for model_type in "${model_type_list[@]}";
    do
        if [ $encoder_have_cnn == "true" ]; then
            setting_name="noseed_gym_cnn_$model_type"
            if [ $model_type == "dp" ]; then
                pool_size=2
                setting_name="noseed_gym_cnn2_$model_type"
            fi
            encoder_have_cnn_str="--encoder_have_cnn"
        elif [ $encoder_type == "cat" ]; then
            setting_name="noseed_gym_cat_$model_type"
            if [ $model_type == "dp" ]; then
                pool_size=2
                setting_name="noseed_gym_cat2_$model_type"
            fi
            encoder_have_cnn_str=""
        elif [ $encoder_type == "group" ]; then
          setting_name="noseed_gym_group_$model_type"
          if [ $model_type == "dp" ]; then
              pool_size=2
              setting_name="noseed_gym_group2_$model_type"
          fi
          encoder_have_cnn_str=""
        elif [ $encoder_type == "attn" ]; then
            setting_name="noseed_gym_attn_$model_type"
            if [ $model_type == "dp" ]; then
                pool_size=2
                setting_name="noseed_gym_attn2_$model_type"
            fi
            encoder_have_cnn_str=""
        elif [ $encoder_type == "pool" ]; then
            setting_name="noseed_gym_pool_$model_type"
            if [ $model_type == "dp" ]; then
                pool_size=2
                setting_name="noseed_gym_pool2_$model_type"
            fi
            encoder_have_cnn_str=""
        else
            setting_name="noseed_gym_$model_type"
            if [ $model_type == "dp" ]; then
                pool_size=6
                setting_name="noseed_gym_6_$model_type"
            fi
            encoder_have_cnn_str=""
        fi
        log_dir="results/seed$seed/${setting_name}_lr${lr}_iters${max_iters}_dim${embed_dim}"
        echo $log_dir

        num_steps_per_iter=1000
        warmup_steps=$((max_iters*num_steps_per_iter/10)) #{/10 or /5}
        # 创建日志文件夹
        mkdir -p $log_dir
        for env in "${env_list[@]}";
        do
            for dataset in "${dataset_list[@]}";
            do
                echo $cuda
                echo "$env"-"$dataset"-"$model_type"
                echo $encoder_have_cnn_str
                bash -c "CUDA_VISIBLE_DEVICES=$cuda python -u experiment.py \
                  --env=$env \
                  --dataset=$dataset \
                  --model_type=$model_type \
                  --batch_size=128 \
                  --embed_dim=$embed_dim \
                  --warmup_steps=$warmup_steps \
                  --max_iters=$max_iters \
                  --learning_rate=$lr \
                  --setting_name=$setting_name \
                  --num_steps_per_iter=$num_steps_per_iter \
                  --num_eval_episodes=10 \
                  --seed=$seed \
                  --pool_size=$pool_size \
                  --encoder_type=$encoder_type \
                  --n_layer=4 \
                  --K=100 \
                  $encoder_have_cnn_str \
                  | tee ./$log_dir/$env-$dataset-$model_type.log 2>&1" &
                cuda=$((cuda+1))
                if [ $cuda -eq 6 ]; then
                    cuda=0
                fi
            done
        done
    done
done

wait
echo "All jobs finished."