# Decision ConvFormer (DC)
token_mixer="pool"
encoder_have_cnn=True
batch_size=128
context_length=30
setting_name_suffix="conv2"

if [ "$encoder_have_cnn" = True ]; then
    setting_name="cnn_${token_mixer}_${setting_name_suffix}"
    pooling_size=2
    model_type="reward_conditioned_cnn"
else
    setting_name="${token_mixer}_${setting_name_suffix}"
    pooling_size=6
    model_type="reward_conditioned"
fi
echo "setting_name: $setting_name"

game_list=('Breakout')
random_list=(123)
cuda=0
for game in "${game_list[@]}"
do
    for seed in "${random_list[@]}"
    do
        log_dir="results/seed$seed/$setting_name"
        mkdir -p $log_dir
        nohup bash -c "CUDA_VISIBLE_DEVICES=$cuda python -u run_dt_atari.py --seed $seed --context_length $context_length \
        --game $game \
        --batch_size $batch_size \
        --model_type $model_type \
        --pooling_size $pooling_size \
        --token_mixer $token_mixer > ./$log_dir/$game.log" &
    done
    cuda=$((cuda+1))
    if [ $cuda -eq 8 ]; then
        cuda=0
    fi
done

#        --context_length 8 \
#        --conv_proj \

#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 8 --game 'Qbert' --batch_size 128 --token_mixer 'attn' --conv_proj --num_steps  50000
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 8 --game 'Pong' --batch_size 512 --token_mixer 'conv'
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 8 --game 'Seaquest' --batch_size 128 --token_mixer 'conv' --conv_proj
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 8 --game 'Asterix' --batch_size 128 --token_mixer 'conv'
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 8 --game 'Frostbite' --batch_size 128 --token_mixer 'conv' --conv_proj
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 8 --game 'Assault' --batch_size 128 --token_mixer 'conv' --conv_proj
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 8 --game 'Gopher' --batch_size 128 --token_mixer 'conv'
#done
#
#
## Decision Transformer (DT)
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --game 'Breakout' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --game 'Qbert' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 50 --game 'Pong' --batch_size 512
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --game 'Seaquest' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --game 'Asterix' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --game 'Frostbite' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --game 'Assault' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --game 'Gopher' --batch_size 128
#done
