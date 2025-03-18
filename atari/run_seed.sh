# Decision ConvFormer (DC)

# 位置参数
setting_name=$1
token_mixer=$2
model_type=$3
batch_size=$4
context_length=$5
game=$6
cuda=$7
pooling_size=$8
echo "setting_name: $setting_name"
echo "token_mixer: $token_mixer"
echo "model_type: $model_type"
echo "batch_size: $batch_size"
echo "context_length: $context_length"
echo "game: $game"
echo "cuda: $cuda"
echo "pooling_size: $pooling_size"
for seed in 123 231 312
do
    log_dir="results/seed$seed/$setting_name"
    echo "log_dir: $log_dir"
    sleep 5s
    mkdir -p $log_dir
    CUDA_VISIBLE_DEVICES=$cuda python -u run_dt_atari.py --seed $seed --context_length $context_length \
    --game $game \
    --batch_size $batch_size \
    --model_type $model_type \
    --pooling_size $pooling_size \
    --token_mixer $token_mixer > ./$log_dir/$game.log
done



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
