#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import collections
import torch
import numpy as np
import transformers
from os.path import join

import data.finetuning_er_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.binding as module_arch
from trainer.bert_finetuning_er_trainer import BERTERTrainer as Trainer
from parse_config import ConfigParser

import pandas as pd
import multiprocessing as mp
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, recall_score

# 设置 spawn 启动方法
mp.set_start_method('spawn', force=True)

def test(model, data_loader, tcr_tokenizer, antigen_tokenizer, hla_tokenizer, device, config):
    model.eval()
    result_dict = {
        'TCR_Beta': [],
        'TCR_Alpha': [],
        'Antigen': [],
        'HLA': [],
        'y_pred': [],
        'Label': []  # 使用 Label 列来存储标签
    }
    with torch.no_grad():
        for batch_idx, (tcr_a_tokenized, tcr_b_tokenized, antigen_tokenized, hla_tokenized, target) in enumerate(data_loader):
            tcr_a_tokenized = {k: v.to(device) for k, v in tcr_a_tokenized.items()}
            tcr_b_tokenized = {k: v.to(device) for k, v in tcr_b_tokenized.items()}
            antigen_tokenized = {k: v.to(device) for k, v in antigen_tokenized.items()}
            hla_tokenized = {k: v.to(device) for k, v in hla_tokenized.items()}

            output = model(tcr_a_tokenized, tcr_b_tokenized, antigen_tokenized, hla_tokenized)

            y_pred = torch.sigmoid(output)
            y_pred = y_pred.cpu().detach().numpy()
            result_dict['y_pred'].append(y_pred)

            # 解码序列
            tcr_beta = tcr_tokenizer.batch_decode(tcr_a_tokenized['input_ids'], skip_special_tokens=True)
            tcr_beta = [s.replace(" ", "") for s in tcr_beta]
            tcr_alpha = tcr_tokenizer.batch_decode(tcr_b_tokenized['input_ids'], skip_special_tokens=True)
            tcr_alpha = [s.replace(" ", "") for s in tcr_alpha]
            antigen = antigen_tokenizer.batch_decode(antigen_tokenized['input_ids'], skip_special_tokens=True)
            antigen = [s.replace(" ", "") for s in antigen]
            hla = hla_tokenizer.batch_decode(hla_tokenized['input_ids'], skip_special_tokens=True)
            hla = [s.replace(" ", "") for s in hla]

            # 获取 Label（标签）
            label_values = target.cpu().detach().numpy().flatten()  # 转换为 numpy 数组
            result_dict['Label'].append(label_values)

            result_dict['TCR_Beta'].append(tcr_beta)
            result_dict['TCR_Alpha'].append(tcr_alpha)
            result_dict['Antigen'].append(antigen)
            result_dict['HLA'].append(hla)

    y_pred = np.concatenate(result_dict['y_pred'])
    label_values = np.concatenate(result_dict['Label'])  # 连接 Label 数据

    test_df = pd.DataFrame({
        'TCR_Beta': [v for l in result_dict['TCR_Beta'] for v in l],
        'TCR_Alpha': [v for l in result_dict['TCR_Alpha'] for v in l],
        'Antigen': [v for l in result_dict['Antigen'] for v in l],
        'HLA': [v for l in result_dict['HLA'] for v in l],
        'y_pred': list(y_pred.flatten()),
        'Label': list(label_values)
    })
    output_path = join(config.log_dir, 'Generalization_test_result.csv')
    test_df.to_csv(output_path, index=False)

    # 计算评价指标（参考metric.py中的分类指标）
    metrics_info = ""
    if len(np.unique(label_values)) == 2:  # 确保有正负样本
        auroc = roc_auc_score(label_values, y_pred)
        precision, recall, _ = precision_recall_curve(label_values, y_pred)
        auprc = auc(recall, precision)
        acc = accuracy_score(label_values, y_pred.round())
        rec = recall_score(label_values, y_pred.round())
        metrics_info = f"AUROC: {auroc:.4f}\nAUPRC: {auprc:.4f}\nAccuracy: {acc:.4f}\nRecall: {rec:.4f}\n"
    else:
        metrics_info = "Warning: Only one class in labels, cannot compute metrics.\n"

    # 保存指标到 info.log
    log_path = join(config.log_dir, 'info.log')
    with open(log_path, 'w') as f:
        f.write(metrics_info)

    return test_df

def main(config):
    logger = config.get_logger('generalization')

    # fix random seeds for reproducibility
    seed = config['data_loader']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    # 更新数据文件路径为零样本泛化数据集
    config['data_loader']['args']['data_dir'] = "../unseen_covid_test_data_with_HLA_and_seq.csv"
    data_loader = config.init_obj('data_loader', module_data)

    tcr_tokenizer = data_loader.get_tcr_tokenizer()
    antigen_tokenizer = data_loader.get_antigen_tokenizer()
    hla_tokenizer = data_loader.get_hla_tokenizer()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)

    logger.info('Loading checkpoint from {}'.format(config['discriminator_resume']))
    checkpoint = torch.load(config['discriminator_resume'], map_location="cuda")
    state_dict = checkpoint['state_dict']
    # 改了：model.load_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to("cuda")

    # Test
    logger = config.get_logger('test')
    test_df = test(model=model, data_loader=data_loader, tcr_tokenizer=tcr_tokenizer,
                   antigen_tokenizer=antigen_tokenizer, hla_tokenizer=hla_tokenizer, device="cuda", config=config)
    logger.info(f"Test results saved to {join(config.log_dir, 'Generalization_test_result.csv')}")
    logger.info(f"Metrics saved to {join(config.log_dir, 'info.log')}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='generalization.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                      help='local rank for nGPUs training')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)