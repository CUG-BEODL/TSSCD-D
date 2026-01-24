# -*- coding: utf-8 -*-

import datetime
from optim.pretrain import *
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--feature_size', type=int, default=192, help='feature_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--patience', type=int, default=20, help='training patience')
    parser.add_argument('--aug_type', type=str, default='Filp', help='Augmentation type')
    parser.add_argument('--class_type', type=str, default='3C', help='Classification type')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dataset_name', type=str, default='Beibuwan', help='dataset')
    parser.add_argument('--nb_class', type=int, default=11, help='class number')
    parser.add_argument('--ucr_path', type=str, default='./datasets', help='Data root for dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/', help='Data path for checkpoint.')
    parser.add_argument('--log_dir', type=str, default='./results/', help='Data path for log.')
    parser.add_argument('--model_name', type=str, default='TSSCD', choices=['TSSCD','TCN','LSTM','TSSCD-D'], help='choose methods')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    import os
    import numpy as np
    opt = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    Seeds = [2022,2023,2024,2025,2026]  
    Runs = range(0, 2, 1)

    opt.log_dir = './results/{}/{}'.format(opt.dataset_name, opt.model_name)
    os.makedirs(opt.log_dir, exist_ok=True)

    with open("{}/train_detail.log".format(opt.log_dir), 'a+') as file2print_detail_train:
        print(datetime.datetime.now(), file=file2print_detail_train)
        print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tVal_Acc_label\tEpoch_max", file=file2print_detail_train)
        file2print_detail_train.flush()

    with open("{}/test.log".format(opt.log_dir), 'a+') as file2print:
        print(datetime.datetime.now(), file=file2print)
        print("Dataset\tAcc_mean\tAcc_std\tEpoch_max", file=file2print)
        file2print.flush()

    with open("{}/test_detail.log".format(opt.log_dir), 'a+') as file2print_detail:
        print(datetime.datetime.now(), file=file2print_detail)
        print("Dataset\tTrain\tlabel_ratio\tTest\tDimension\tClass\tSeed\tAcc_max\tEpoch_max\ttest_oa\ttest_aa\ttest_kappa\ttest_S_PA\ttest_S_UA\ttest_S_F1\ttest_T_PA\ttest_T_UA\ttest_T_F1", file=file2print_detail)
        file2print_detail.flush()

    best_acc_seed, ACCs_seed, MAX_EPOCHs_seed = {}, {}, {}
    test_oa_seed, test_aa_seed, test_kappa_seed = {}, {}, {}
    test_S_PA_seed, test_S_UA_seed, test_S_F1_seed = {}, {}, {}
    test_T_PA_seed, test_T_UA_seed, test_T_F1_seed = {}, {}, {}

    for seed in Seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        opt.ckpt_dir = './ckpt/{}/{}/{}/{}'.format(opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),str(seed))
        os.makedirs(opt.ckpt_dir, exist_ok=True)

        print('[INFO] Running at:', opt.dataset_name)
        x_train, y_train, x_val, y_val, x_test, y_test,data_ds = load_ucr2018(opt.ucr_path, opt.dataset_name, opt.nb_class)

        best_acc_run, ACCs_run, MAX_EPOCHs_run = {}, {},{}
        test_oa_run, test_aa_run, test_kappa_run = {}, {}, {}
        test_S_PA_run, test_S_UA_run, test_S_F1_run = {}, {}, {}
        test_T_PA_run, test_T_UA_run, test_T_F1_run = {}, {}, {}

        for run in Runs:
            best_val_acc, acc_test, epoch_max, test_oa, test_aa, test_kappa, test_S_PA, test_S_UA, test_S_F1, test_T_PA, test_T_UA, test_T_F1 = train(
                x_train, y_train, x_val, y_val, x_test, y_test, opt,data_ds)
            print("test_oa: {}, test_aa: {}, test_kappa: {}, test_S_PA: {}, test_S_UA: {}, test_S_F1: {}, test_T_PA: {}, test_T_UA: {}, test_T_F1: {}".format(
                test_oa, test_aa, test_kappa, test_S_PA, test_S_UA, test_S_F1, test_T_PA, test_T_UA, test_T_F1))
            best_acc_run[run] = best_val_acc
            ACCs_run[run] = acc_test
            MAX_EPOCHs_run[run] = epoch_max
            test_oa_run[run] = test_oa
            test_aa_run[run] = test_aa
            test_kappa_run[run] = test_kappa
            test_S_PA_run[run] = test_S_PA
            test_S_UA_run[run] = test_S_UA
            test_S_F1_run[run] = test_S_F1
            test_T_PA_run[run] = test_T_PA
            test_T_UA_run[run] = test_T_UA
            test_T_F1_run[run] = test_T_F1

        best_acc_seed[seed] = round(np.mean(list(best_acc_run.values())),2)
        ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
        MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))
        test_oa_seed[seed] = round(np.mean(list(test_oa_run.values())), 2)
        test_aa_seed[seed] = round(np.mean(list(test_aa_run.values())), 2)
        test_kappa_seed[seed] = round(np.mean(list(test_kappa_run.values())), 2)
        test_S_PA_seed[seed] = round(np.mean(list(test_S_PA_run.values())), 2)
        test_S_UA_seed[seed] = round(np.mean(list(test_S_UA_run.values())), 2)
        test_S_F1_seed[seed] = round(np.mean(list(test_S_F1_run.values())), 2)
        test_T_PA_seed[seed] = round(np.mean(list(test_T_PA_run.values())), 2)
        test_T_UA_seed[seed] = round(np.mean(list(test_T_UA_run.values())), 2)
        test_T_F1_seed[seed] = round(np.mean(list(test_T_F1_run.values())), 2)

        with open("{}/train_detail.log".format(opt.log_dir), 'a+') as file2print_detail_train:
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
                    seed, best_acc_seed[seed],MAX_EPOCHs_seed[seed]),
                    file=file2print_detail_train)
            file2print_detail_train.flush()

        with open("{}/test_detail.log".format(opt.log_dir), 'a+') as file2print_detail:
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                opt.dataset_name, x_train.shape[0], opt.label_ratio, x_test.shape[0], x_train.shape[1], opt.nb_class,
                seed, ACCs_seed[seed], MAX_EPOCHs_seed[seed], test_oa_seed[seed], test_aa_seed[seed], test_kappa_seed[seed],
                test_S_PA_seed[seed], test_S_UA_seed[seed], test_S_F1_seed[seed], test_T_PA_seed[seed], test_T_UA_seed[seed], test_T_F1_seed[seed]),
                file=file2print_detail)
            file2print_detail.flush()

    ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
    ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
    MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))

    with open("{}/test.log".format(opt.log_dir), 'a+') as file2print:
        print("{}\t{}\t{}\t{}".format(
            opt.dataset_name, ACCs_seed_mean, ACCs_seed_std, MAX_EPOCHs_seed_max),
            file=file2print)
        file2print.flush()