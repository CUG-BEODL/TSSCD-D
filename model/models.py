# -*- coding: utf-8 -*-

import torch
from optim.pytorchtools import EarlyStopping
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils.metrics import Evaluator, SpatialChangeDetectScore, TemporalChangeDetectScore
from utils.utils import *
import datetime

class Diceloss(nn.Module):
    def __init__(self, smooth=1.):
        super(Diceloss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=0).sum(dim=0)
        loss = (1 - ((2. * intersection + self.smooth) / (
                pred.sum(dim=0).sum(dim=0) + target.sum(dim=0).sum(dim=0) + self.smooth)))
        return loss.mean()
class Model(torch.nn.Module):

  def __init__(self, backbone):
    super(Model, self).__init__()
    self.backbone = backbone

  def train(self, tot_epochs,train_loader_label,val_loader, test_loader, opt,data_ds):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.to(device)
    patience = opt.patience
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/model_best_{}_{}.pth'.format(opt.ckpt_dir, data_ds,dt_string))

    optimizer = torch.optim.Adam([{'params': self.backbone.parameters()}], lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 5, gamma = 0.7 ) 
    loss_fn = nn.CrossEntropyLoss()  # classification loss function
    loss_ch_noch = Diceloss()  # changed loss function
    best_acc=0

    for epoch in range(tot_epochs):
        
        acc_epoch= torch.tensor(data=0, dtype=torch.float, device=device)
        loss_epoch_label= torch.tensor(data=0, dtype=torch.float, device=device)
        val_loss_epoch = torch.tensor(data=0, dtype=torch.float, device=device)
        self.backbone.train()
        train_tqdm = tqdm(enumerate(train_loader_label), total=len(train_loader_label), 
                        desc=f'Epoch {epoch+1} [Train]')
        
        for i, data_labeled in train_tqdm:
            optimizer.zero_grad()

            x, target = data_labeled
            x, target = x.cuda(), target.cuda()
            target = target.squeeze(-1)

            output = self.backbone(x.permute(0,2,1))  # (batch, num_classes, time_steps)

            pre_label = torch.argmax(input=output, dim=1)
        
            pre_No_change = pre_label.max(dim=1).values == pre_label.min(dim=1).values
            label_No_change = target.max(dim=1).values == target.min(dim=1).values

            loss_label = loss_fn(output, target.squeeze(-1))

            loss_change = loss_ch_noch(pre_No_change, label_No_change)

            loss_label = loss_label + loss_change
            loss_label.backward()
            optimizer.step()
            loss_epoch_label += loss_label.item()

            correct = (pre_label == target).sum().item()
            accuracy = 100.0 * correct / target.numel()
            acc_epoch += accuracy
            
            # 更新进度条
            train_tqdm.set_postfix({
                'loss': loss_epoch_label/(i+1),
                'acc': acc_epoch/(i+1)
            })
        train_tqdm.close()
        acc_epoch /= len(train_loader_label)
        loss_epoch_label /= len(train_loader_label)

        acc_vals = []
        acc_tests = []
      
        self.backbone.eval()
        scheduler.step()

        with torch.no_grad():
            val_tqdm = tqdm(enumerate(val_loader), total=len(val_loader), 
                   desc=f'Epoch {epoch+1} [Val]')
    
            for i, (x, target) in val_tqdm:
          
                x = x.cuda()
                target = target.cuda()
                target = target.squeeze(-1)
                output = self.backbone(x.permute(0,2,1)).detach()
                
                pre_label = torch.argmax(input=output, dim=1)
                pre_No_change = pre_label.max(dim=1).values == pre_label.min(dim=1).values
                label_No_change = target.max(dim=1).values == target.min(dim=1).values
                val_loss = loss_fn(output, target)
                val_loss_change = loss_ch_noch(pre_No_change, label_No_change)
                val_loss = val_loss + val_loss_change
                val_loss_epoch += val_loss

                correct = 0
                accuracy = 0

                correct = (pre_label == target).sum().item()
                accuracy = 100.0 * correct / target.numel()
                acc_vals.append(accuracy)

                val_tqdm.set_postfix({'val_acc': np.mean(acc_vals)})                   
            val_tqdm.close()
            val_acc = sum(acc_vals) / len(acc_vals)
            val_loss_epoch = val_loss_epoch / len(val_loader)
            #val_change_point_acc = sum(change_point_acc_vals) / len(change_point_acc_vals)
            print("val_loss_epoch [{}]: ".format(epoch),val_loss_epoch)
          

        # ====================== 主流程 ======================
        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            acc_tests = []
            with torch.no_grad():
                evaluator = Evaluator(opt.nb_class)
                spatialscore = SpatialChangeDetectScore()
                temporalscore = TemporalChangeDetectScore(error_rate=11)
                
                test_tqdm = tqdm(enumerate(test_loader), total=len(test_loader), 
                    desc=f'Epoch {epoch+1} [Test]')

                for i, (x, target) in test_tqdm:
                    x, target = x.cuda(), target.cuda()
                    output = self.backbone(x.permute(0,2,1)).detach()
                    
                    prediction = output.argmax(dim=1)
                    
                    target = target.squeeze(dim=2)
                    evaluator.add_batch(target.cpu().numpy(), prediction.cpu().numpy())
                    correct = (prediction == target).sum().item()

                    accuracy = (100.0 * correct / target.numel())
                    acc_tests.append(accuracy)

                    predList = prediction.cpu().numpy()
                    labelList = target.cpu().numpy()
                
                    for pred, label in zip(predList, labelList):
                        predata, prechangepoints, pretypes = FilteringSeries(pred, method='Majority')
                        labdata, labchangepoints, labtypes = FilteringSeries(label)
                        spatialscore.addValue(labchangepoints, prechangepoints)
                        temporalscore.addValue(labchangepoints, prechangepoints)
                    test_tqdm.set_postfix({
                        'acc': accuracy
                    })
                test_tqdm.close()
            
            test_acc = np.mean(acc_tests)
            Acc = evaluator.Pixel_Accuracy()
            Acc_class, Acc_mean = evaluator.Pixel_Accuracy_Class()
            F1 = evaluator.F1()
            Kappa = evaluator.Kappa()
    
            
            test_oa = round(Acc, 4)*100
            test_aa = round(Acc_mean, 4)*100
            F1 = evaluator.F1()
            test_f1 = round(F1, 4)*100
            Kappa = evaluator.Kappa()
            test_kappa = round(Kappa, 4)*100

            spatialscore.getScore()
            spatial_f1 = spatialscore.spatial_f1
            test_s_pa = round(spatialscore.spatial_pa, 4)*100
            test_s_ua = round(spatialscore.spatial_ua, 4)*100
            test_s_f1 = round(spatial_f1, 4)*100
            
            temporalscore.getScore()
            temporal_f1 = temporalscore.temporal_f1
            test_t_pa = round(temporalscore.temporal_pa, 4)*100
            test_t_ua = round(temporalscore.temporal_ua, 4)*100
            test_t_f1 = round(temporal_f1, 4)*100

            confusion_matrix = evaluator.confusion_matrix  
            num_classes = confusion_matrix.shape[0]

            producer_accuracy = np.zeros(num_classes)
            user_accuracy = np.zeros(num_classes)

            for i in range(num_classes):
                producer_accuracy[i] = confusion_matrix[i, i] / (np.sum(confusion_matrix[i, :]) + 1e-6) 
                user_accuracy[i] = confusion_matrix[i, i] / (np.sum(confusion_matrix[:, i]) + 1e-6)

            print("\nClass-wise Producer Accuracy (PA) and User Accuracy (UA):")
            for i in range(num_classes):
                print(f"Class {i}: PA = {producer_accuracy[i]:.4f}, UA = {user_accuracy[i]:.4f}")

            print("\nTable: Evaluation of classification accuracy")
            print("Class\tProducer Accuracy (PA)\tUser Accuracy (UA)")
            for i in range(num_classes):
                print(f"{i}\t{producer_accuracy[i]:.4f} ± {np.std(producer_accuracy):.4f}\t{user_accuracy[i]:.4f} ± {np.std(user_accuracy):.4f}")


            with open(f"{opt.log_dir}/classification_accuracy.txt", "w") as f:
                f.write("Class\tProducer Accuracy (PA)\tUser Accuracy (UA)\n")
                for i in range(num_classes):
                    f.write(f"{i}\t{producer_accuracy[i]:.4f} ± {np.std(producer_accuracy):.4f}\t{user_accuracy[i]:.4f} ± {np.std(user_accuracy):.4f}\n")

            overall_accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
            print(f"\nOverall Accuracy (OA): {overall_accuracy:.4f} ± {np.std(overall_accuracy):.4f}")



        print("test_acc:", test_acc)            
        best_val_acc = best_acc
        print('[Test-{}] Val ACC: {:.2f}%, Best val Acc: {:.2f}% in Epoch {}, OA: {:.2f}%, AA: {:.2f}%, Kappa: {:.2f}%,S-PA: {:.2f}%, S-UA: {:.2f}%, S-F1: {:.2f}%, T-PA: {:.2f}%, T-UA: {:.2f}%, T-F1: {:.2f}%'.format(\
            epoch, val_acc, best_val_acc, best_epoch, test_oa, test_aa, test_kappa, test_s_pa, test_s_ua, test_s_f1, test_t_pa, test_t_ua, test_t_f1))
            
    
        early_stopping(val_loss_epoch, self.backbone)
        if early_stopping.early_stop:
          print("Early stopping")
          break

        if (epoch+1)%opt.save_freq==0:
          print("[INFO] save model at epoch {}!".format(epoch))
          torch.save(self.backbone.state_dict(), '{}/model_{}.pth'.format(opt.ckpt_dir, epoch))

        print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%'.format(epoch + 1, opt.model_name, opt.dataset_name, loss_epoch_label, acc_epoch))


    return best_val_acc ,test_acc,  best_epoch, test_oa, test_aa, test_kappa, test_s_pa, test_s_ua, test_s_f1, test_t_pa, test_t_ua, test_t_f1

