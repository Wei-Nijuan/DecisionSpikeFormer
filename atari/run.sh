# Decision ConvFormer (DC)
token_mixer="pool"
encoder_have_cnn=True
batch_size=128
context_length=30
setting_name="$token_mixer"


if [ "$encoder_have_cnn" = True ]; then
    setting_name="cnn_$token_mixer"
    pooling_size=2
    model_type="reward_conditioned_cnn"
else
    setting_name="${token_mixer}"
    pooling_size=6
    model_type="reward_conditioned"
fi
echo "setting_name: $setting_name"


#setting_name=$1
#token_mixer=$2
#model_type=$3
#batch_size=$4
#context_length=$5
#game=$6
#cuda=$7
#pooling_size=$8

chmod +x run_seed.sh
game_list=('Breakout' 'Qbert' 'Pong' 'Seaquest' 'Asterix' 'Frostbite' 'Assault' 'Gopher')
cuda=0
for game in "${game_list[@]}"
do
    nohup bash -c "./run_seed.sh $setting_name $token_mixer $model_type $batch_size $context_length $game $cuda $pooling_size" &
    cuda=$((cuda+1))
    if [ $cuda -eq 8 ]; then
        cuda=0
    fi
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
