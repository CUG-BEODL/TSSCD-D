# -*- coding: utf-8 -*-

import torch
from dataloader.ucr2018 import *
from model.models import *
from model.model_backbone import TSSCD,TSSCD_D
from model.model_backbone import TCN
from model.model_backbone import LSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(x_train, y_train, x_val, y_val, x_test, y_test,opt,data_ds):

    batch_size = opt.batch_size  
    tot_epochs = opt.epochs  

    ckpt_dir = opt.ckpt_dir    

    if opt.model_name == 'TSSCD':
        backbone = TSSCD().cuda()
    elif opt.model_name == 'TSSCD-D':
        backbone = TSSCD_D().cuda()
    elif opt.model_name == 'TCN':
        backbone = TCN().cuda()
    elif opt.model_name == 'LSTM':
        backbone = LSTM().cuda()
        
    augment_type = opt.aug_type

    model = Model(backbone).cuda()
    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=augment_type)
    val_set = UCR2018(data=x_val, targets=y_val, transform=None)
    test_set = UCR2018(data=x_test, targets=y_test, transform=None)
    train_loader_label = torch.utils.data.DataLoader(train_set_labeled, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    torch.save(model.state_dict(), '{}/model_init.pth'.format(ckpt_dir))
    print("Start training")

    best_val_acc, test_acc, best_epoch,   test_oa, test_aa, test_kappa, test_S_PA, test_S_UA, test_S_F1, test_T_PA, test_T_UA, test_T_F1= model.train(tot_epochs=tot_epochs, 
                                     train_loader_label=train_loader_label,                                    
                                     val_loader=val_loader,
                                     test_loader=test_loader,
                                     opt=opt,data_ds = data_ds)

    return best_val_acc, test_acc, best_epoch, test_oa, test_aa, test_kappa, test_S_PA, test_S_UA, test_S_F1, test_T_PA, test_T_UA, test_T_F1



