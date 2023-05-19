import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='PAD')
    parser.add_argument('--seed', type=int, default=112,help='Seed - Test your luck!')   
    
    parser.add_argument('--model', type=str, default='ncde',help='Model Name')
    parser.add_argument('--h_channels', type=int, default=64,help='Hidden Channels') 
    parser.add_argument('--hh_channels_f', type=int, default=128,help='Hidden Channels_f') 
    parser.add_argument('--hh_channels_g', type=int, default=64,help='Hidden Channels_g') 
    parser.add_argument('--hh_channels_c', type=int, default=64,help='Hidden Channels_c') 

    
    parser.add_argument('--lr', type=float, default=0.01,help='Learning Rate') 
    parser.add_argument('--weight_decay', type=float, default=0.0001,help='Weight Decay') 
    parser.add_argument('--epoch', type=int, default=350,help='Epoch') 
    parser.add_argument('--solver_method', type=str, default='rk4',help='ODE Solver Methods') 
    
    parser.add_argument("--data_path", type=str, default="dataset/SWAT")
    parser.add_argument("--dataset",type=str,default='SWAT')
    parser.add_argument("--batch_size", type=int, default=256)
    #forecasting
    parser.add_argument('--win_size',type=int, default = 30,help='look_window')
    parser.add_argument('--forecast_window',type=int,default=10,help='forecast window')
    parser.add_argument('--step_size',type=int,default=30,help='stride window')
    
    #random missing
    parser.add_argument('--interpolation',type=str,default='cubic',help='Interpolation Method')
    parser.add_argument('--missing_rate',type=float,default=0,help='Missing rates')
    
    
    
    # parser.add_argyment('--')
    args = parser.parse_args()
    return args