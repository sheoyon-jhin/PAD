

if [ ! -d "./exp_logs" ]; then
    mkdir ./exp_logs
fi

if [ ! -d "./exp_logs/SWAT" ]; then
    mkdir ./exp_logs/SWAT
fi




for lr in 1e-2
do 
    for h_channels in 60 30 80
    do 
        for hh_channels_f in 90 80 70 
        do 
            for hh_channels_g in 60 30 40 20 
            do
                for hh_channels_c in 20 10 60 70
                do
                    for missing_rate in 0
                    do
                        for forecast_window in 10 
                        do
                            CUDA_VISIBLE_DEVICES=0 python3 -u ../pad2.py --seed 112  --missing_rate $missing_rate --data_path dataset/SWAT --dataset SWAT --win_size 30 --forecast_window $forecast_window --step_size 30 --h_channels $h_channels  --lr $lr  --hh_channels_f $hh_channels_f --hh_channels_g $hh_channels_g --hh_channels_c $hh_channels_c --epoch 350 > ../exp_logs/SWAT/SWAT_lr_{$lr}_{$missing_rate}_dims_{$h_channels}_{$hh_channels_f}_{$hh_channels_g}_{$hh_channels_c}.csv
                        done
                    done
                done
            done
        done
    done
done

