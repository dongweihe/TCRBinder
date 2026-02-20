#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import collections
import torch
import numpy as np
import transformers
from os.path import join
from torch.utils.data import DataLoader
import data.finetuning_er_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.binding as module_arch
from trainer.bert_finetuning_er_trainer import BERTERTrainer as Trainer
from parse_config import ConfigParser
import multiprocessing as mp

def main(config):
    logger = config.get_logger('train')
    # fix random seeds for reproducibility
    seed = config['data_loader']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_dataset(valid=True)
    test_data_loader = data_loader.split_dataset(valid=False, test=True)
    logger.info('Number of pairs in train: {}, valid: {}, and test: {}.'.format(
        data_loader.sampler.__len__(),
        valid_data_loader.sampler.__len__(),
        test_data_loader.sampler.__len__()
    ))
    tcr_tokenizer = data_loader.get_tcr_tokenizer()
    antigen_tokenizer = data_loader.get_antigen_tokenizer()
    hla_tokenizer = data_loader.get_hla_tokenizer()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.BetaModel.to(device)
    model.AlphaModel.to(device)
    model.AntigenModel.to(device)
    model.HLAModel.to(device)
    BERT_params = list(map(id, model.BetaModel.parameters())) + list(map(id, model.AlphaModel.parameters())) + list(map(id, model.AntigenModel.parameters())) + list(map(id, model.HLAModel.parameters()))
    base_params = filter(lambda p: id(p) not in BERT_params, model.parameters())

    BERT_lr = config["optimizer"]["args"]["BERT_lr"]
    lr = config["optimizer"]["args"]["lr"]
    weight_decay = config["optimizer"]["args"]["weight_decay"]
    optimizer = torch.optim.AdamW(
        [{'params': base_params},
         {'params': model.BetaModel.parameters(), 'lr': BERT_lr},
         {'params': model.AlphaModel.parameters(), 'lr': BERT_lr},
         {'params': model.AntigenModel.parameters(), 'lr': BERT_lr},
         {'params': model.HLAModel.parameters(), 'lr': BERT_lr}], lr=lr, weight_decay=weight_decay)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()

    """Test."""
    logger = config.get_logger('test')
    resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # save three models
    beta_bert_save_dir = join(config.save_dir, 'betamodel')
    alpha_bert_save_dir = join(config.save_dir, 'alphamodel')
    antigen_bert_save_dir = join(config.save_dir, 'antigenmodel')
    hla_bert_save_dir = join(config.save_dir, 'hlamodel')
    logger.info(f'Saving four berts to {beta_bert_save_dir}, {alpha_bert_save_dir}, {antigen_bert_save_dir}, {hla_bert_save_dir}')
    os.makedirs(beta_bert_save_dir, exist_ok=True)
    model.BetaModel.save_pretrained(beta_bert_save_dir)
    os.makedirs(alpha_bert_save_dir, exist_ok=True)
    model.AlphaModel.save_pretrained(alpha_bert_save_dir)
    os.makedirs(antigen_bert_save_dir, exist_ok=True)
    model.AntigenModel.save_pretrained(antigen_bert_save_dir)
    os.makedirs(hla_bert_save_dir, exist_ok=True)
    model.HLAModel.save_pretrained(hla_bert_save_dir)

    test_metrics, test_output = trainer.test(tcr_tokenizer=tcr_tokenizer,
                                             antigen_tokenizer=antigen_tokenizer,
                                             hla_tokenizer=hla_tokenizer)
    logger.info(test_output)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
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