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


try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


def compute_macro_auc01(y_true, y_pred, antigens):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    antigens = np.array(antigens)

    unique_antigens = np.unique(antigens)
    auc_scores = []

    for ag in unique_antigens:
        mask = (antigens == ag)
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]


        if len(np.unique(sub_true)) < 2:
            continue

        try:
            p_auc = roc_auc_score(sub_true, sub_pred, max_fpr=0.1)
            auc_scores.append(p_auc)
        except ValueError:
            continue

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)


def test(model, data_loader, tcr_tokenizer, antigen_tokenizer, hla_tokenizer, device, config):
    model.eval()
    result_dict = {
        'TCR_Beta': [],
        'TCR_Alpha': [],
        'Antigen': [],
        'HLA': [],
        'y_pred': [],
        'Label': []
    }
    with torch.no_grad():
        for batch_idx, (tcr_a_tokenized, tcr_b_tokenized, antigen_tokenized, hla_tokenized, target) in enumerate(
                data_loader):
            tcr_a_tokenized = {k: v.to(device) for k, v in tcr_a_tokenized.items()}
            tcr_b_tokenized = {k: v.to(device) for k, v in tcr_b_tokenized.items()}
            antigen_tokenized = {k: v.to(device) for k, v in antigen_tokenized.items()}
            hla_tokenized = {k: v.to(device) for k, v in hla_tokenized.items()}

            output = model(tcr_a_tokenized, tcr_b_tokenized, antigen_tokenized, hla_tokenized)

            y_pred = torch.sigmoid(output)
            y_pred = y_pred.cpu().detach().numpy()
            result_dict['y_pred'].append(y_pred)

            tcr_beta = tcr_tokenizer.batch_decode(tcr_a_tokenized['input_ids'], skip_special_tokens=True)
            tcr_beta = [s.replace(" ", "") for s in tcr_beta]
            tcr_alpha = tcr_tokenizer.batch_decode(tcr_b_tokenized['input_ids'], skip_special_tokens=True)
            tcr_alpha = [s.replace(" ", "") for s in tcr_alpha]
            antigen = antigen_tokenizer.batch_decode(antigen_tokenized['input_ids'], skip_special_tokens=True)
            antigen = [s.replace(" ", "") for s in antigen]
            hla = hla_tokenizer.batch_decode(hla_tokenized['input_ids'], skip_special_tokens=True)
            hla = [s.replace(" ", "") for s in hla]

            label_values = target.cpu().detach().numpy().flatten()
            result_dict['Label'].append(label_values)

            result_dict['TCR_Beta'].append(tcr_beta)
            result_dict['TCR_Alpha'].append(tcr_alpha)
            result_dict['Antigen'].append(antigen)
            result_dict['HLA'].append(hla)

    y_pred_all = np.concatenate(result_dict['y_pred']).flatten()
    label_values_all = np.concatenate(result_dict['Label']).flatten()

    tcr_beta_all = [v for l in result_dict['TCR_Beta'] for v in l]
    tcr_alpha_all = [v for l in result_dict['TCR_Alpha'] for v in l]
    antigen_all = [v for l in result_dict['Antigen'] for v in l]
    hla_all = [v for l in result_dict['HLA'] for v in l]

    test_df = pd.DataFrame({
        'TCR_Beta': tcr_beta_all,
        'TCR_Alpha': tcr_alpha_all,
        'Antigen': antigen_all,
        'HLA': hla_all,
        'y_pred': list(y_pred_all),
        'Label': list(label_values_all)
    })

    output_path = join(config.log_dir, 'Generalization_result.csv')
    test_df.to_csv(output_path, index=False)

    metrics_info = ""
    if len(np.unique(label_values_all)) == 2:

        auroc = roc_auc_score(label_values_all, y_pred_all)
        precision, recall, _ = precision_recall_curve(label_values_all, y_pred_all)
        auprc = auc(recall, precision)
        acc = accuracy_score(label_values_all, y_pred_all.round())
        rec = recall_score(label_values_all, y_pred_all.round())

        macro_auc01 = compute_macro_auc01(label_values_all, y_pred_all, antigen_all)

        metrics_info = (
            f"AUROC: {auroc:.4f}\n"
            f"AUPRC: {auprc:.4f}\n"
            f"Macro AUC0.1: {macro_auc01:.4f}\n"
            f"Accuracy: {acc:.4f}\n"
            f"Recall: {rec:.4f}\n"
        )

        print(metrics_info)
    else:
        metrics_info = "Warning: Only one class in labels, cannot compute metrics.\n"
        print(metrics_info)

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

    config['data_loader']['args']['logger'] = logger

    config['data_loader']['args']['data_dir'] = "../unseen.csv"
    data_loader = config.init_obj('data_loader', module_data)

    tcr_tokenizer = data_loader.get_tcr_tokenizer()
    antigen_tokenizer = data_loader.get_antigen_tokenizer()
    hla_tokenizer = data_loader.get_hla_tokenizer()

    model = config.init_obj('arch', module_arch)

    logger.info('Loading checkpoint from {}'.format(config['discriminator_resume']))

    checkpoint = torch.load(config['discriminator_resume'], map_location="cuda")
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=False)
    model.to("cuda")

    logger = config.get_logger('test')
    test_df = test(model=model, data_loader=data_loader, tcr_tokenizer=tcr_tokenizer,
                   antigen_tokenizer=antigen_tokenizer, hla_tokenizer=hla_tokenizer, device="cuda", config=config)
    logger.info(f"Test results saved to {join(config.log_dir, 'Generalization_result.csv')}")
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