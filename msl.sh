

if [ ! -d "./exp_logs" ]; then
    mkdir ./exp_logs
fi
if [ ! -d "./exp_logs/Check3" ]; then
    mkdir ./exp_logs/Check3
fi


if [ ! -d "./exp_logs/Check3/FIN" ]; then
    mkdir ./exp_logs/Check3/FIN
fi



for lr in 1e-2 
do 
    for h_channels in 64 #128 256 
    do 
        for hh_channels_f in 256
        do 
            for hh_channels_g in 512 #512 
            do
                for hh_channels_c in 256 #65 
                do
                    for missing_rate in 0 
                    do
                        CUDA_VISIBLE_DEVICES=3 python3 -u pad.py --seed 112 --data_path dataset/MSL --dataset MSL --win_size 30 --forecast_window 10 --step_size 30 --h_channels $h_channels  --lr $lr  --hh_channels_f $hh_channels_f --hh_channels_g $hh_channels_g --hh_channels_c $hh_channels_c --missing_rate $missing_rate --epoch 350 > ./exp_logs/Check3/FIN/MSL_301030_missing_rate_{$missing_rate}_lr_{$lr}_dims_{$h_channels}_{$hh_channels_f}_{$hh_channels_g}_{$hh_channels_c}.csv
                    done
                done
            done
        done
    done
done
