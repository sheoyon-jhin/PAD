import os
import os.path as osp
import time
import random
import numpy as np
import torch
from parse2 import parse_arguments
import sklearn.model_selection
import torchdiffeq
from sklearn import metrics 
import pathlib

from data_factory.data_loader import get_loader_segment
from sklearn.metrics import precision_recall_fscore_support

import datasets
import torchcde
import warnings
warnings.filterwarnings("ignore")

CUBICS = ['natural_cubic','cubic']
torch.backends.cudnn.benchmark = True

def delete_get_index(target1,target2,target3,target4,index):
    # import pdb ; pdb.set_trace()
    count = 0 
    for id,ind in enumerate(index):
        ind = ind-count 
        target1 = torch.cat([target1[:ind],target1[ind+1:]])
        target2 = torch.cat([target2[:ind],target2[ind+1:]])
        target3 = torch.cat([target3[:ind],target3[ind+1:]])
        target4 = torch.cat([target4[:ind],target4[ind+1:]])
        count +=1
    return target1,target2,target3,target4

def get_dataset(args):
    if args.dataset =='MSL':
        loc,input_size = datasets.msl.get_data(args.win_size,args.forecast_window,args.step_size,args.missing_rate,learning_method='self-supervised')
    elif args.dataset == 'SMD':
        loc,input_size = datasets.smd.get_data(args.win_size,args.forecast_window,args.step_size,args.missing_rate,learning_method='self-supervised')
    elif args.dataset =='SMAP':
        loc,input_size = datasets.smap.get_data(args.win_size,args.forecast_window,args.step_size,args.missing_rate,learning_method='self-supervised')
    elif args.dataset =='PSM':
        loc,input_size = datasets.psm.get_data(args.win_size,args.forecast_window,args.step_size,args.missing_rate)
    elif args.dataset =='SWAT':
        loc,input_size = datasets.swat.get_data(args.win_size,args.forecast_window,args.step_size,args.missing_rate)
    elif args.dataset =='SWAT_SAMSUNG':
        loc,input_size = datasets.swat_samsung.get_data(args.win_size,args.forecast_window,args.step_size,args.missing_rate)
    
    elif args.dataset =='WADI':
        loc,input_size = datasets.wadi.get_data(args.win_size,args.forecast_window,args.step_size,args.missing_rate)
    elif args.dataset =='SMAP2':
        loc,input_size = datasets.smap2.get_data(args.win_size,args.forecast_window,args.step_size,args.missing_rate,learning_method='self-supervised')
    
    print(input_size)
    here = pathlib.Path(__file__).resolve().parent
    base_base_loc = here / 'datasets/processed_data'
    if args.interpolation=='natural_cubic':
        coeff_loc = loc / ('NaturalCoeffs')
    else:    
        coeff_loc = loc / ('Coeffs')
    times        = torch.load(str(loc)+'/times.pt')
   
    train_X = torch.load(str(loc) +'/train_seq_data.pt')
    train_y  = torch.load(str(loc) +'/train_y_data.pt')
    # import pdb ; pdb.set_trace()
    train_next_y = torch.load(str(loc)+'/train_forecast_y.pt')
    train_next_forecast = torch.load(str(loc)+'/train_forecast_seq.pt')
    
    # GET RID OF DRIFT PART IN TRAIN SET 
    # print(train_X.shape,train_y.shape,train_next_y.shape,train_next_forecast.shape)
    
    # get_info = ((train_y.sum(dim=1)>0).to(train_y.dtype) != (train_next_y.sum(dim=1)>0).to(train_next_y.dtype)).to(train_y.dtype)
    # get_index = get_info.nonzero()
    # train_X,train_y,train_next_forecast,train_next_y = delete_get_index(train_X,train_y,train_next_forecast,train_next_y,get_index)
    # # import pdb ; pdb.set_trace()
    # print(train_X.shape,train_y.shape,train_next_y.shape,train_next_forecast.shape)
    # check_info = ((train_y.sum(dim=1)>0).to(train_y.dtype) != (train_next_y.sum(dim=1)>0).to(train_next_y.dtype)).to(train_y.dtype)
    # check_index = check_info.nonzero()
    # print(check_index)
    
    val_X =torch.load(str(loc) +'/val_seq_data.pt')
    val_y = torch.load(str(loc) +'/val_y_data.pt')
    val_next_y = torch.load(str(loc)+'/val_forecast_y.pt')
    val_next_forecast = torch.load(str(loc)+'/val_forecast_seq.pt')

    test_X =torch.load(str(loc) +'/test_seq_data.pt')
    test_y = torch.load(str(loc) +'/test_y_data.pt')
    test_next_y = torch.load(str(loc)+'/test_forecast_y.pt')
    test_next_forecast = torch.load(str(loc)+'/test_forecast_seq.pt')
    if os.path.exists(coeff_loc):
        
        pass
    else:
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(coeff_loc):
            os.mkdir(coeff_loc)
        if not os.path.exists(coeff_loc):
            os.mkdir(coeff_loc)
        if args.interpolation =='natural_cubic':
            print("Start extrapolation!")
            train_coeffs = torchcde.natural_cubic_coeffs(train_X)
            
            torch.save(train_coeffs,str(coeff_loc)+'/train_coeffs.pt')
            train_next_coeffs = torchcde.natural_cubic_coeffs(train_next_forecast)
            torch.save(train_next_coeffs,str(coeff_loc)+'/train_next_coeffs.pt')
            print("finish extrapolation Train coeff")
            val_coeffs = torchcde.natural_cubic_coeffs(val_X)
            
            
            torch.save(val_coeffs,str(coeff_loc)+'/val_coeffs.pt')
            print("finish extrapolation Val coeff")
            test_coeffs = torchcde.natural_cubic_coeffs(test_X)
            torch.save(test_coeffs,str(coeff_loc)+'/test_coeffs.pt')
            
            print("finish extrapolation Test coeff")
            print("success!")
        else:
            print("Start extrapolation!")
            train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)
            
            torch.save(train_coeffs,str(coeff_loc)+'/train_coeffs.pt')
            train_next_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_next_forecast)
            torch.save(train_next_coeffs,str(coeff_loc)+'/train_next_coeffs.pt')
            print("finish extrapolation Train coeff")
            val_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(val_X)
            
            
            torch.save(val_coeffs,str(coeff_loc)+'/val_coeffs.pt')
            print("finish extrapolation Val coeff")
            test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
            torch.save(test_coeffs,str(coeff_loc)+'/test_coeffs.pt')
            print("finish extrapolation Test coeff")
            print("success!")
    
    train_coeffs = torch.load(str(coeff_loc)+'/train_coeffs.pt')
    train_next_coeffs = torch.load(str(coeff_loc)+'/train_next_coeffs.pt')
    val_coeffs = torch.load(str(coeff_loc)+'/val_coeffs.pt')
    test_coeffs = torch.load(str(coeff_loc)+'/test_coeffs.pt')
    
    # import pdb ; pdb.set_trace()
    
    train_coeffs=train_coeffs.to(device)
    train_next_coeffs = train_next_coeffs.to(device)
    val_coeffs=val_coeffs.to(device)
    test_coeffs=test_coeffs.to(device)
    train_y = train_y.to(device)
    val_y = val_y.to(device)
    test_y = test_y.to(device)
    train_next_y = train_next_y.to(device)
    val_next_y = val_next_y.to(device)
    test_next_y = test_next_y.to(device)
    
        
    # load dataset
    data_path = args.data_path
    dataset = args.dataset
    # import pdb ; pdb.set_trace()
    train_dataset = torch.utils.data.TensorDataset(train_X,train_coeffs,train_next_coeffs, train_y,train_next_y,train_next_forecast)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataset = torch.utils.data.TensorDataset(val_X,val_coeffs, val_y,val_next_y,val_next_forecast)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_y.shape[0])    
    test_dataset = torch.utils.data.TensorDataset(test_X,test_coeffs, test_y,test_next_y,test_next_forecast)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_y.shape[0])
    return input_size, train_dataloader, val_dataloader,test_dataloader


class ODEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(ODEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 60)
        self.linear2 = torch.nn.Linear(60, hidden_channels)

    def forward(self, t, z):
        z = self.linear1(z)
        z = z.tanh()
        z = self.linear2(z)
        z = z.tanh()
        return z
class CDEFunc_f(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,hidden_hidden_channels):
        super(CDEFunc_f, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # f / g 
        self.linear0 = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear1 = torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        self.linear3 = torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        self.linear4 = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)
        
        
        # FOR MSL, SMD 
        # self.linear1 = torch.nn.Linear(hidden_channels, 128)
        # self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)


    def forward(self, t, z):
        
        z = self.linear0(z)
        z = z.relu()
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.relu()
        z = self.linear3(z)
        z = z.relu()
        z = self.linear4(z)
        # z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)

        return z 
class CDEFunc_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,hidden_hidden_channels):
        super(CDEFunc_g, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # f / g 
        self.linear0 = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear1 = torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        self.linear3 = torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        self.linear4 = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)
        
        
        # FOR MSL, SMD 
        # self.linear1 = torch.nn.Linear(hidden_channels, 128)
        # self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)


    def forward(self, t, z):
        
        z = self.linear0(z)
        z = z.relu()
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.relu()
        z = self.linear3(z)
        z = z.relu()
        z = self.linear4(z)
        # z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)

        return z 
    
class CDEFunc_c(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,hidden_hidden_channels):
        super(CDEFunc_c, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # f / g 
        self.linear1 = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)
        
        
        # FOR MSL, SMD 
        # self.linear1 = torch.nn.Linear(hidden_channels, 128)
        # self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)


    def forward(self, t, z):
        
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)

        return z 
    
class NeuralDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,hidden_hiddens, output_channels,forecast_window,device, interpolation="cubic"):
        super(NeuralDE, self).__init__()
        if args.model=='ncde':
            
            hidden_hidden_f,hidden_hidden_g,hidden_hidden_c = hidden_hiddens 
            
            self.func_f = CDEFunc_f(input_channels, hidden_channels,hidden_hidden_f)
            self.func_g = CDEFunc_g(input_channels, hidden_channels,hidden_hidden_g)
            self.func_c = CDEFunc_c(input_channels, hidden_channels,hidden_hidden_c)
            self.readout = torch.nn.Linear(hidden_channels, output_channels)
            self.readout2 = torch.nn.Linear(hidden_channels, output_channels)
            self.forecast = torch.nn.Linear(hidden_channels,input_channels)
            self.reconstruct = torch.nn.Linear(hidden_channels,input_channels)
            self.interpolation = interpolation
            # self.readout2= torch.nn.Linear(hidden_channels,)
        if args.model =='node':
            
            self.func = ODEFunc(input_channels, hidden_channels)
            self.readout = torch.nn.Linear(hidden_channels, output_channels)
            self.readout2 = torch.nn.Linear(60, output_channels)
            
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.interpolation = interpolation
        self.input_channels = input_channels
        self.forecast_window = forecast_window
        # self.win_size = 
        self.device=device
        
    def forward(self, _coeffs,mode,adjoint=True,**kwargs):
        # import pdb ; pdb.set_trace()
        if mode =='train':
            coeffs,next_coeffs = _coeffs
        else:
            coeffs = _coeffs
        if self.interpolation in CUBICS:
            if mode =='train':
                X = torchcde.CubicSpline(coeffs)
                next_X = torchcde.CubicSpline(next_coeffs)
            else:
                X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            if mode =='train':
                X = torchcde.LinearInterpolation(coeffs)
                next_X = torchcde.LinearInterpolation(next_coeffs)
            else:
                X = torchcde.LinearInterpolation(coeffs)
                   
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")
    
        batch_dims = coeffs.shape[:-2]
        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        
        # X0 = X.evaluate(X.interval[0])
        # z0 = self.initial(X0)
        times = torch.arange(X.interval[-1].item()+1).to(coeffs.device)
        X0 = X.evaluate(times)
        # X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        z0 = z0.sum(dim=1)
        if mode =='train':
            # next_X0 = next_X.evaluate(next_X.interval[0])
            # next_z0 = self.initial(next_X0)
            next_times = torch.arange(next_X.interval[-1].item()+1).to(coeffs.device)
            next_X0 = next_X.evaluate(next_times)
            next_z0 = self.initial(next_X0)
            next_z0 = next_z0.sum(dim=1)
        
        if args.model =='ncde':
            
            times = torch.arange(X.interval[-1].item()+1).to(z0.device)
            # z_T = torchcde.cdeint(X=X,z0=z0,func=(self.func_f,self.func_c),t=X.interval)
            z_T = torchcde.cdeint(X=X,z0=z0,func=(self.func_f,self.func_c),t=times)
            if mode =='train':
                next_z_T = torchcde.cdeint(X=next_X,z0=next_z0,func=(self.func_f,self.func_c),t=times)
            h_T= torchcde.cdeint(X=X, z0=z0,func=(self.func_g,self.func_c),t= times)
            
            
            pred_y = z_T[:,-1,:]
            pred_reconstruct = self.reconstruct(z_T)
            # sigmoid = torch.nn.Sigmoid()
            # pred_reconstruct = sigmoid(pred_reconstruct)
            pred_y = self.readout(pred_y)
            pred_y = pred_y.squeeze(-1)
            look_window = times.shape[0]
            if mode =='train':
                next_pred_y = next_z_T[:,look_window-self.forecast_window:,:]
                next_pred_reconstruct = self.forecast(next_pred_y)
                next_pred_y = self.readout2(next_pred_y)
                next_pred_y_gt = next_pred_y.squeeze(-1)
                next_pred_y_gt = next_pred_y_gt[:,-1]
            
            ## Forecasting part  
            forecast_hidden =  h_T[:,look_window-self.forecast_window:,:]
            pred_next_forecast = self.forecast(forecast_hidden)
            pred_next_y = self.readout2(forecast_hidden)
            pred_next_y = pred_next_y.squeeze(-1)
            
            
            pred_next_y = pred_next_y[:,-1]
            
        if args.model =='node':
            if 'atol' not in kwargs:
                kwargs['atol'] = 1e-6
            if 'rtol' not in kwargs:
                kwargs['rtol'] = 1e-4
            if 'method' not in kwargs:
                kwargs['method'] = 'rk4'
            if kwargs['method'] == 'rk4':
                if 'options' not in kwargs:
                    kwargs['options'] = {}
                options = kwargs['options']
                if 'step_size' not in options and 'grid_constructor' not in options:
                    time_diffs = 1.0
                    options['step_size'] = time_diffs
            # import pdb ;pdb.set_trace()
            odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
            times = torch.arange(coeffs.shape[1]).to(coeffs.device)
            times = times.float()
            # import pdb ; pdb.set_trace()
            t = (times-times.min())/(times.max()-times.min())
            z_T = odeint(func=self.func, y0=z0, t=t, **kwargs)
            
            
        
            z_T = z_T.permute(1,0,2)
            
            
            pred_y = self.readout(z_T).squeeze(-1)
            pred_y = self.readout2(pred_y)
            # import pdb ;pdb.set_trace()
            sigmoid = torch.nn.Sigmoid()
            pred_y = sigmoid(pred_y)
        if mode =='train':
            return pred_y,pred_reconstruct,pred_next_y,pred_next_forecast,next_pred_y_gt,next_pred_reconstruct
        else:
            return pred_y,pred_reconstruct,pred_next_y,pred_next_forecast
            



def train(args,optimizer,forecast_loss,train_dataloader,val_dataloader,test_dataloader):
    for epoch in range(args.epoch):
        model.train()
        full_pred_y=torch.Tensor().to(device)
        full_pred_next_y = torch.Tensor().to(device)
        full_true_y=torch.Tensor().to(device)
        full_true_next_y=torch.Tensor().to(device)
        
        pred_latent_y = torch.Tensor().to(device)
        pred_latent_next_y = torch.Tensor().to(device)
        start_time= time.time()
        train_loss_ = 0 
        mse_loss_ = 0 
        for batch in train_dataloader:
            # import pdb ; pdb.set_trace()
            present,batch_coeffs,batch_next_coeffs, batch_y,next_y ,next_forecast= batch 
            coeffs = (batch_coeffs,batch_next_coeffs)
            pred_y,reconstruct,pred_next_y,forecast_y ,next_gt,next_gt_reconstruct= model(coeffs,mode='train')
            pred_y = pred_y.squeeze(-1)
            pred_next_y = pred_next_y.squeeze(-1)
            # import pdb ; pdb.set_trace()
            if args.missing_rate >0:
                
                X = torchcde.CubicSpline(batch_coeffs)
                times = torch.arange(X.interval[-1].item()+1)
                present = X.evaluate(times)
                
                next_X= torchcde.CubicSpline(batch_next_coeffs)
                next_times = torch.arange(next_X.interval[-1].item()+1)
                next_forecast = next_X.evaluate(next_times)
            # import pdb ; pdb.set_trace()
            fore_loss = forecast_loss(forecast_y.cpu(),next_forecast.cpu()) #256,20,55
            present_score  = pred_y # (present.cuda() - reconstruct).abs().mean(dim=[1,2]) + pred_y
            next_gt_score = next_gt # (next_forecast.cuda() - next_gt_reconstruct).abs().mean(dim=[1,2]) + next_gt
            precursor_score = pred_next_y
            
            binary_prediction = (present_score>0).to(batch_y.dtype)
            next_binary_prediction = (precursor_score>0).to(batch_y.dtype)
            next_gt = (next_gt_score>0).to(batch_y.dtype)
            
            
            # binary_prediction = (pred_y<0.02).to(batch_y.dtype)
            
            
            batch_y = (batch_y.sum(dim=1)>0).to(batch_y.dtype)
            next_y = (next_y.sum(dim=1)>0).to(batch_y.dtype)
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            # Knowledge distillation
            next_loss = torch.nn.functional.binary_cross_entropy_with_logits(next_gt, next_binary_prediction)
            
            
            full_pred_y=torch.cat([full_pred_y,binary_prediction])
            full_pred_next_y=torch.cat([full_pred_next_y,next_binary_prediction])
            
            full_true_y=torch.cat([full_true_y,batch_y])
            full_true_next_y=torch.cat([full_true_next_y,next_y])
            
            pred_latent_y = torch.cat([pred_latent_y,present_score])
            pred_latent_next_y = torch.cat([pred_latent_next_y,precursor_score])
            
            neg_loss = (pred_y * batch_y ).sum()/ batch_y.sum()
            neg_next_loss = (pred_next_y * next_y ).sum()/ next_y.sum()
            
            pos_y = (batch_y<=0).to(batch_y.dtype)
            pos_next_y = (next_y<=0).to(batch_y.dtype)
            pos_loss = (pred_y * pos_y ).sum()/ pos_y.sum()
            pos_next_loss = (pred_next_y * pos_next_y ).sum()/ pos_next_y.sum()
            
            full_loss = next_loss + loss
            train_loss_ += full_loss
            mse_loss_ += fore_loss
            
            
            full_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        preds = pred_latent_y.squeeze(-1).detach().cpu().numpy()
        next_preds = pred_latent_next_y.squeeze(-1).detach().cpu().numpy()
        
        y = full_true_y.squeeze(-1).detach().cpu().numpy()
        next_y = full_true_next_y.squeeze(-1).detach().cpu().numpy()

        train_loss = train_loss_ / len(train_dataloader)
        mse_loss = mse_loss_ / len(train_dataloader)
        
        fpr,tpr,thresholds = metrics.roc_curve(y,preds,pos_label=1)
        next_fpr,next_tpr,next_thresholds = metrics.roc_curve(next_y,next_preds,pos_label=1)
            
            
        Full_precision, Full_recall, Full_f_score, _     = precision_recall_fscore_support(full_true_y.cpu(), full_pred_y.cpu(),average='weighted')
        Next_precision, Next_recall, Next_f_score, _     = precision_recall_fscore_support(full_true_next_y.cpu(), full_pred_next_y.cpu(),average='weighted')
        
        
        # print('Epoch: {}   Train loss:      {:.4f}  Train forecasting loss :      {:.4f}    Time :{:.4f}'.format(epoch, train_loss,mse_loss,(time.time()-start_time)))
        
        print('Epoch: {}   Train loss:      {:.4f}  Train Pr :      {:.4f}       Train Re :      {:.4f}   Train F1:       {:.4f}      Time :{:.4f}'.format(epoch, loss.item(),Full_precision,Full_recall,Full_f_score,(time.time()-start_time)))
        print('[Precursor]                          Train Next Pr : {:.4f}       Train Next Re : {:.4f}   Train Next F1:  {:.4f} '.format(Next_precision,Next_recall,Next_f_score))
        print('[Forecasting]                        Train MSE :     {:.4f} \n'.format(mse_loss))
        evaluate(args,'val',val_dataloader,epoch)
        evaluate(args,'test',test_dataloader,epoch)
        
        
        
def evaluate(args,mode,eval_dataloader,epoch):
    
    model.eval()
    full_pred_y=torch.Tensor().to(device)
    full_pred_next_y = torch.Tensor().to(device)
    full_true_y=torch.Tensor().to(device)
    full_true_next_y=torch.Tensor().to(device)
    
    pred_latent_y = torch.Tensor().to(device)
    pred_latent_next_y = torch.Tensor().to(device)
    best_auroc=0
    start_time= time.time()
    eval_loss_ = 0
    eval_mse_loss_ =0 
    for batch in eval_dataloader:
        start = time.time()

        present,batch_coeffs, batch_y,next_y,next_forecast = batch
        
        pred_y,reconstruct,pred_next_y,forecast_y = model(batch_coeffs,mode=mode)
        pred_y = pred_y.squeeze(-1)
        pred_next_y = pred_next_y.squeeze(-1)
        
        if args.missing_rate >0:
                
            X = torchcde.CubicSpline(batch_coeffs)
            times = torch.arange(X.interval[-1].item()+1).to(batch_coeffs.device)
            present = X.evaluate(times)
            
        
        fore_loss = forecast_loss(forecast_y.cpu(),next_forecast) #256,20,55
        
        
        present_score  = (present.cuda() - reconstruct).mean(dim=[1,2]) + pred_y
        precursor_score =  pred_next_y
        
        binary_prediction = (present_score>0).to(batch_y.dtype)
        next_binary_prediction = (precursor_score>0).to(batch_y.dtype)
        
        
        
        batch_y = (batch_y.sum(dim=1)>0).to(batch_y.dtype) # 481/614
        next_y = (next_y.sum(dim=1)>0).to(next_y.dtype) # 490/614 614개중 490개 맞춤
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(binary_prediction.cuda(), batch_y)
        next_loss = torch.nn.functional.binary_cross_entropy_with_logits(next_binary_prediction.cuda(), next_y)
        
        full_pred_y=torch.cat([full_pred_y,binary_prediction])
        full_pred_next_y=torch.cat([full_pred_next_y,next_binary_prediction])
        
        full_true_y=torch.cat([full_true_y,batch_y])
        full_true_next_y=torch.cat([full_true_next_y,next_y])
        
        pred_latent_y = torch.cat([pred_latent_y,present_score])
        pred_latent_next_y = torch.cat([pred_latent_next_y,precursor_score])
        # import pdb;pdb.set_trace()
        
        
        full_loss = loss+next_loss 
        eval_loss_ += full_loss
        eval_mse_loss_ += fore_loss
        
    
    
    preds = pred_latent_y.squeeze(-1).detach().cpu().numpy()
    next_preds = pred_latent_next_y.squeeze(-1).detach().cpu().numpy()
    
    y = full_true_y.squeeze(-1).detach().cpu().numpy()
    next_y = full_true_next_y.squeeze(-1).detach().cpu().numpy()
   
    
    
    fpr,tpr,thresholds = metrics.roc_curve(y,preds,pos_label=1)
    next_fpr,next_tpr,next_thresholds = metrics.roc_curve(next_y,next_preds,pos_label=1)
    Full_precision, Full_recall, Full_f_score, _     = precision_recall_fscore_support(full_true_y.cpu(), full_pred_y.cpu(),average='weighted')
    Next_precision, Next_recall, Next_f_score, _     = precision_recall_fscore_support(full_true_next_y.cpu(), full_pred_next_y.cpu(),average='weighted')
    
    eval_loss = eval_loss_ / len(eval_dataloader)
    eval_mse_loss = eval_mse_loss_ / len(eval_dataloader)
    if mode =='val':
        print('Epoch: {}   Validation loss: {:.4f} , Validation Pr :      {:.4f}       Validation Re :      {:.4f}  Validation F1: {:.4f} Time :{:.4f}'.format(epoch, eval_loss.item(),Full_precision,Full_recall,Full_f_score,(time.time()-start_time)))
        print('[Precursor]                           Validation Next Pr : {:.4f}       Validation Next Re : {:.4f}  Val Next F1:   {:.4f} '.format(Next_precision,Next_recall,Next_f_score))
        print('[Forecasting]                         Validation MSE :     {:.4f} \n'.format(eval_mse_loss))
    else:
        print('Epoch: {}   Test loss:       {:.4f} , Test Pr :      {:.4f}       Test Re :      {:.4f}    Test F1:       {:.4f}  Time :{:.4f}'.format(epoch, eval_loss.item(),Full_precision,Full_recall,Full_f_score,(time.time()-start_time)))
        print('[Precursor]                           Test Next Pr : {:.4f}       Test Next Re : {:.4f}    Test Next F1:  {:.4f}  '.format(Next_precision,Next_recall,Next_f_score))
        print('[Forecasting]                         Test MSE :     {:.4f} \n'.format(eval_mse_loss))
        
        
        print("---------------------------------------------------------------------------------")
        


        
if __name__ == "__main__":
    
    args = parse_arguments()
    
    manual_seed = args.seed
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    
    print(args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    input_size, train_dataloader,val_dataloader,test_dataloader = get_dataset(args)
    
    
    # build model
    hidden_hiddens = (args.hh_channels_f,args.hh_channels_g,args.hh_channels_c)
    model = NeuralDE(input_channels=input_size, hidden_channels=args.h_channels, hidden_hiddens=hidden_hiddens,output_channels=1,forecast_window=args.forecast_window,device=device)
    model=model.to(device)
    print("======= >>> Model Info <<< =======")
    print(model)
    print("==================================")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    forecast_loss = torch.nn.MSELoss()
    
    
    train(args,optimizer,forecast_loss,train_dataloader,val_dataloader,test_dataloader)
    
    
