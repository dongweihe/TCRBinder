# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join, exists
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_prepare_tokenizer import get_tokenizer
from base_data_loader import BaseDataLoader
from data_prepare_utility import is_valid_aaseq
from transformers import AutoTokenizer


class TCR_Antigen_Dataset_AbDab(BaseDataLoader):
    def __init__(self, logger,
                 seed,
                 batch_size,
                 validation_split,
                 test_split,
                 num_workers,
                 data_dir,
                 tcr_vocab_dir,
                 tcr_tokenizer_dir,
                 tokenizer_name='common',
                 # receptor_tokenizer_name='common',
                 token_length_list='2,3',
                 # receptor_token_length_list='2,3',
                 antigen_seq_name='antigen',
                 beta_seq_name='TCR_Beta',
                 alpha_seq_name='TCR_Alpha',
                 hla_seq_name='HLA',
                 label_name='Label',
                 # receptor_seq_name='beta',
                 test_tcrs=100,
                 # neg_ratio=1.0,
                 shuffle=True,
                 antigen_max_len=None,
                 hla_max_len=None,
                 beta_max_len=None,
                 alpha_max_len=None, ):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.beta_seq_name = beta_seq_name
        self.alpha_seq_name = alpha_seq_name
        self.antigen_seq_name = antigen_seq_name
        self.hla_seq_name = hla_seq_name
        self.label_name = label_name

        self.test_tcrs = test_tcrs
        self.shuffle = shuffle
        self.beta_max_len = beta_max_len
        self.alpha_max_len = alpha_max_len
        self.antigen_max_len = antigen_max_len
        self.hla_max_len = hla_max_len

        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()

        self.BetaTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                            add_hyphen=False,
                                            logger=self.logger,
                                            vocab_dir=tcr_vocab_dir,
                                            token_length_list=token_length_list)
        self.beta_tokenizer = self.BetaTokenizer.get_bert_tokenizer(
            max_len=self.beta_max_len,
            tokenizer_dir=tcr_tokenizer_dir)

        self.AlphaTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                            add_hyphen=False,
                                            logger=self.logger,
                                            vocab_dir=tcr_vocab_dir,
                                            token_length_list=token_length_list)
        self.alpha_tokenizer = self.AlphaTokenizer.get_bert_tokenizer(
            max_len=self.alpha_max_len,
            tokenizer_dir=tcr_tokenizer_dir)

        self.AntigenTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=tcr_vocab_dir,
                                              token_length_list=token_length_list)
        self.antigen_tokenizer = self.AntigenTokenizer.get_bert_tokenizer(
            max_len=self.antigen_max_len,
            tokenizer_dir=tcr_tokenizer_dir)

        self.HLATokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                          add_hyphen=False,
                                          logger=self.logger,
                                          vocab_dir=tcr_vocab_dir,
                                          token_length_list=token_length_list)
        self.hla_tokenizer = self.HLATokenizer.get_bert_tokenizer(
            max_len=self.hla_max_len,
            tokenizer_dir=tcr_tokenizer_dir)

        esm_dir = '/data/coding/TCRBinder-main/Code/esm2/esm2_150m/'

        self.antigen_tokenizer = AutoTokenizer.from_pretrained(esm_dir, cache_dir="../esm2/esm2_150m/",
                                                               max_len=self.antigen_max_len)

        self.hla_tokenizer = AutoTokenizer.from_pretrained(esm_dir, cache_dir="../esm2/esm2_150m/",
                                                           max_len=self.hla_max_len)

        dataset = self._get_dataset(pair_df=self.pair_df)
        super().__init__(dataset, batch_size, seed, shuffle, validation_split, test_split,
                         num_workers)

    def get_beta_tokenizer(self):
        return self.beta_tokenizer

    def get_alpha_tokenizer(self):
        return self.alpha_tokenizer

    def get_tcr_tokenizer(self):
        return self.beta_tokenizer

    def get_antigen_tokenizer(self):
        return self.antigen_tokenizer

    def get_hla_tokenizer(self):
        return self.hla_tokenizer

    def get_test_dataloader(self):
        return self.test_dataloader

    def _get_dataset(self, pair_df):
        abag_dataset = AbAGDataset_CovAbDab(
            beta_seqs=list(pair_df[self.beta_seq_name]),
            alpha_seqs=list(pair_df[self.alpha_seq_name]),
            antigen_seqs=list(pair_df[self.antigen_seq_name]),
            hla_seqs=list(pair_df[self.hla_seq_name]),
            labels=list(pair_df[self.label_name]),
            tcr_split_fun=self.BetaTokenizer.split,
            antigen_split_fun=self.AntigenTokenizer.split,
            tcr_tokenizer=self.beta_tokenizer,
            antigen_tokenizer=self.antigen_tokenizer,
            hla_tokenizer=self.hla_tokenizer,
            tcr_max_len=self.beta_max_len,
            antigen_max_len=self.antigen_max_len,
            hla_max_len=self.hla_max_len,
            logger=self.logger
        )
        return abag_dataset

    def _split_dataset(self):
        # if exists(join(self.neg_pair_save_dir, 'unseen_tcrs-seed-'+str(self.seed)+'.csv')):
        #     test_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'unseen_tcrs-seed-'+str(self.seed)+'.csv'))
        #     self.logger.info(f'Loading created unseen tcrs for test with shape {test_pair_df.shape}')

        tcr_list = list(set(self.pair_df['tcr']))
        selected_tcr_index_list = self.rng.integers(len(tcr_list), size=self.test_tcrs)
        self.logger.info(f'Select {self.test_tcrs} from {len(tcr_list)} tcr')
        selected_tcrs = [tcr_list[i] for i in selected_tcr_index_list]
        test_pair_df = self.pair_df[self.pair_df['tcr'].isin(selected_tcrs)]
        # test_pair_df.to_csv(join(self.neg_pair_save_dir, 'unseen_tcrs-seed-'+str(self.seed)+'.csv'), index=False)

        selected_tcrs = list(set(test_pair_df['tcr']))
        train_valid_pair_df = self.pair_df[~self.pair_df['tcr'].isin(selected_tcrs)]

        self.logger.info(
            f'{len(train_valid_pair_df)} pairs for train and valid and {len(test_pair_df)} pairs for test.')

        return train_valid_pair_df, test_pair_df

    def _create_pair(self):
        pair_df = pd.read_csv(self.data_dir)

        if self.shuffle:
            pair_df = pair_df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Shuffling dataset")
        self.logger.info(f"There are {len(pair_df)} samples")
        # pair_df = pair_df.sample(n=3680, random_state=42).reset_index(drop=True)
        return pair_df

    def _load_seq_pairs(self):
        self.logger.info(f'Loading from {self.using_dataset}...')
        self.logger.info(f'Loading {self.tcr_seq_name} and {self.receptor_seq_name}')
        column_map_dict = {'alpha': 'cdr3a', 'beta': 'cdr3b', 'tcr': 'tcr'}
        keep_columns = [column_map_dict[c] for c in [self.tcr_seq_name, self.receptor_seq_name]]

        df_list = []
        for dataset in self.using_dataset:
            df = pd.read_csv(join(self.data_dir, dataset, 'full.csv'))
            df = df[keep_columns]
            df = df[(df[keep_columns[0]].map(is_valid_aaseq)) & (df[keep_columns[1]].map(is_valid_aaseq))]
            self.logger.info(f'Loading {len(df)} pairs from {dataset}')
            df_list.append(df[keep_columns])
        df = pd.concat(df_list)
        self.logger.info(f'Current data shape {df.shape}')
        df_filter = df.dropna()
        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After dropping na and duplicates, current data shape {df_filter.shape}')

        column_rename_dict = {column_map_dict[c]: c for c in [self.tcr_seq_name, self.receptor_seq_name]}
        df_filter.rename(columns=column_rename_dict, inplace=True)

        df_filter['label'] = [1] * len(df_filter)
        df_filter.to_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'), index=False)

        return df_filter


class AbAGDataset_CovAbDab(Dataset):
    def __init__(self, beta_seqs, alpha_seqs, antigen_seqs, hla_seqs, labels, tcr_split_fun, antigen_split_fun,
                 tcr_tokenizer, antigen_tokenizer, hla_tokenizer, tcr_max_len, antigen_max_len, hla_max_len,
                 logger):
        self.beta_seqs = beta_seqs
        self.alpha_seqs = alpha_seqs
        self.antigen_seqs = antigen_seqs
        self.hla_seqs = hla_seqs
        self.labels = labels
        self.tcr_split_fun = tcr_split_fun
        self.antigen_split_fun = antigen_split_fun
        self.tcr_tokenizer = tcr_tokenizer
        self.antigen_tokenizer = antigen_tokenizer
        self.hla_tokenizer = hla_tokenizer
        self.tcr_max_len = tcr_max_len
        self.antigen_max_len = antigen_max_len
        self.hla_max_len = hla_max_len
        self.logger = logger
        self._has_logged_example = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        beta, alpha, antigen, hla = self.beta_seqs[i], self.alpha_seqs[i], self.antigen_seqs[i], self.hla_seqs[i]
        label = self.labels[i]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        beta_tensor = self.tcr_tokenizer(self._insert_whitespace(self.tcr_split_fun(beta)),
                                               padding="max_length",
                                               max_length=self.tcr_max_len,
                                               truncation=True,
                                               return_tensors="pt")
        alpha_tensor = self.tcr_tokenizer(self._insert_whitespace(self.tcr_split_fun(alpha)),
                                               padding="max_length",
                                               max_length=self.tcr_max_len,
                                               truncation=True,
                                               return_tensors="pt")
        antigen_tensor = self.antigen_tokenizer(antigen,
                                                padding="max_length",
                                                max_length=self.antigen_max_len,
                                                truncation=True,
                                                return_tensors="pt")
        hla_tensor = self.hla_tokenizer(hla,
                                        padding="max_length",
                                        max_length=self.hla_max_len,
                                        truncation=True,
                                        return_tensors="pt")

        label_tensor = torch.FloatTensor(np.atleast_1d(label)).to(device)

        beta_tensor = {k: v.to(device) for k, v in beta_tensor.items()}
        alpha_tensor = {k: v.to(device) for k, v in alpha_tensor.items()}
        antigen_tensor = {k: v.to(device) for k, v in antigen_tensor.items()}
        hla_tensor = {k: v.to(device) for k, v in hla_tensor.items()}

        beta_tensor = {k: torch.squeeze(v) for k, v in beta_tensor.items()}
        alpha_tensor = {k: torch.squeeze(v) for k, v in alpha_tensor.items()}
        antigen_tensor = {k: torch.squeeze(v) for k, v in antigen_tensor.items()}
        hla_tensor = {k: torch.squeeze(v) for k, v in hla_tensor.items()}

        return beta_tensor, alpha_tensor, antigen_tensor, hla_tensor, label_tensor

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)


class ABAGDataset_4_input(Dataset):
    def __init__(self, tcr_a_seqs,
                 tcr_b_seqs,
                 tcr_c_seqs,
                 receptor_seqs,
                 labels,
                 tcr_split_fun,
                 receptor_split_fun,
                 tcr_tokenizer,
                 receptor_tokenizer,
                 tcr_max_len,
                 receptor_max_len,
                 logger):
        self.tcr_a_seqs = tcr_a_seqs
        self.tcr_b_seqs = tcr_b_seqs
        self.tcr_c_seqs = tcr_c_seqs
        self.receptor_seqs = receptor_seqs
        self.labels = labels
        self.tcr_split_fun = tcr_split_fun
        self.receptor_split_fun = receptor_split_fun
        self.tcr_tokenizer = tcr_tokenizer
        self.receptor_tokenizer = receptor_tokenizer
        self.tcr_max_len = tcr_max_len
        self.receptor_max_len = receptor_max_len
        self.logger = logger
        self._has_logged_example = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        tcr_a, tcr_b, tcr_c, receptor = self.tcr_a_seqs[i], self.tcr_b_seqs[i], \
                                                       self.tcr_c_seqs[i], self.receptor_seqs[i]
        label = self.labels[i]
        tcr_a_tensor = self.tcr_tokenizer(self._insert_whitespace(self.tcr_split_fun(tcr_a)),
                                                    padding="max_length",
                                                    max_length=self.tcr_max_len,
                                                    truncation=True,
                                                    return_tensors="pt",
                                                    )
        tcr_b_tensor = self.tcr_tokenizer(self._insert_whitespace(self.tcr_split_fun(tcr_b)),
                                                    padding="max_length",
                                                    max_length=self.tcr_max_len,
                                                    truncation=True,
                                                    return_tensors="pt")

        tcr_c_tensor = self.tcr_tokenizer(self._insert_whitespace(self.tcr_split_fun(tcr_c)),
                                                    padding="max_length",
                                                    max_length=self.tcr_max_len,
                                                    truncation=True,
                                                    return_tensors="pt")

        receptor_tensor = self.receptor_tokenizer(self._insert_whitespace(self.receptor_split_fun(receptor)),
                                                  padding="max_length",
                                                  max_length=self.receptor_max_len,
                                                  truncation=True,
                                                  return_tensors="pt")

        label_tensor = torch.FloatTensor(np.atleast_1d(label))
        tcr_a_tensor = {k: torch.squeeze(v) for k, v in tcr_a_tensor.items()}
        tcr_b_tensor = {k: torch.squeeze(v) for k, v in tcr_b_tensor.items()}
        tcr_c_tensor = {k: torch.squeeze(v) for k, v in tcr_c_tensor.items()}
        receptor_tensor = {k: torch.squeeze(v) for k, v in receptor_tensor.items()}

        # if not self._has_logged_example:
        #     self.logger.info(f"Example of tokenized tcr: {tcr} -> {tcr_tensor}")
        #     self.logger.info(f"Example of tokenized receptor: {receptor} -> {receptor_tensor}")
        #     self.logger.info(f"Example of label: {label} -> {label_tensor}")
        #     self._has_logged_example = True

        return tcr_a_tensor, tcr_b_tensor, tcr_c_tensor, receptor_tensor, label_tensor

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)


class TCR_Antigen_Dataset_4_input(BaseDataLoader):
    def __init__(self, logger,
                 seed,
                 batch_size,
                 validation_split,
                 test_split,
                 num_workers,
                 data_dir,
                 tcr_vocab_dir,
                 tcr_tokenizer_dir,
                 tcr_tokenizer_name='common',
                 tcr_token_length_list='2,3',
                 tcr_seq_a_name="cdr1",
                 tcr_seq_b_name="cdr2",
                 tcr_seq_c_name="cdr3",
                 receptor_seq_name="beta",
                 test_tcrs=100,
                 shuffle=True,
                 cdr_max_len=None,
                 ab_max_len=None):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir

        self.tcr_seq_a_name = tcr_seq_a_name
        self.tcr_seq_b_name = tcr_seq_b_name

        self.tcr_seq_c_name = tcr_seq_c_name
        self.receptor_seq_name = receptor_seq_name

        self.test_tcrs = test_tcrs
        # self.neg_ratio = neg_ratio
        self.shuffle = shuffle
        self.cdr_max_len = cdr_max_len
        self.ab_max_len = ab_max_len
        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()
        # train_valid_pair_df, test_pair_df = self._split_dataset()

        self.logger.info(f'Creating {tcr_seq_a_name} tokenizer...')
        self.TCRTokenizer_a = get_tokenizer(tokenizer_name=tcr_tokenizer_name,
                                                 add_hyphen=False,
                                                 logger=self.logger,
                                                 vocab_dir=tcr_vocab_dir,
                                                 token_length_list=tcr_token_length_list)
        self.tcr_tokenizer_a = self.TCRTokenizer_a.get_bert_tokenizer(
            max_len=cdr_max_len,
            tokenizer_dir=tcr_tokenizer_dir)

        # self.logger.info(f'Creating {tcr_seq_name} tokenizer...')
        self.TCRTokenizer_b = get_tokenizer(tokenizer_name=tcr_tokenizer_name,
                                                 add_hyphen=False,
                                                 logger=self.logger,
                                                 vocab_dir=tcr_vocab_dir,
                                                 token_length_list=tcr_token_length_list)
        self.tcr_tokenizer_b = self.TCRTokenizer_b.get_bert_tokenizer(
            max_len=self.cdr_max_len,
            tokenizer_dir=tcr_tokenizer_dir)

        self.TCRTokenizer_c = get_tokenizer(tokenizer_name=tcr_tokenizer_name,
                                                 add_hyphen=False,
                                                 logger=self.logger,
                                                 vocab_dir=tcr_vocab_dir,
                                                 token_length_list=tcr_token_length_list)
        self.tcr_tokenizer_c = self.TCRTokenizer_c.get_bert_tokenizer(
            max_len=self.cdr_max_len,
            tokenizer_dir=tcr_tokenizer_dir)

        # self.logger.info(f'Creating {receptor_seq_name} tokenizer...')
        self.antigen_tokenizer = get_tokenizer(tokenizer_name=tcr_tokenizer_name,
                                               add_hyphen=False,
                                               logger=self.logger,
                                               vocab_dir=tcr_vocab_dir,
                                               token_length_list=tcr_token_length_list)
        self.receptor_tokenizer = self.antigen_tokenizer.get_bert_tokenizer(
            max_len=self.ab_max_len,
            tokenizer_dir=tcr_tokenizer_dir)

        dataset = self._get_dataset(pair_df=self.pair_df)
        super().__init__(dataset, batch_size, seed, shuffle, validation_split, test_split,
                         num_workers)

    def get_tcr_tokenizer(self):
        return self.tcr_tokenizer_a

    def get_antigen_tokenizer(self):
        return self.receptor_tokenizer

    def get_test_dataloader(self):
        return self.test_dataloader

    def _get_dataset(self, pair_df):
        er_dataset = ABAGDataset_4_input(tcr_a_seqs=list(pair_df[self.tcr_seq_a_name]),
                                         tcr_b_seqs=list(pair_df[self.tcr_seq_b_name]),
                                         tcr_c_seqs=list(pair_df[self.tcr_seq_c_name]),
                                         receptor_seqs=list(pair_df[self.receptor_seq_name]),
                                         labels=list(pair_df['affinity']),  # ll_cdr
                                         tcr_split_fun=self.TCRTokenizer_a.split,
                                         receptor_split_fun=self.antigen_tokenizer.split,
                                         tcr_tokenizer=self.tcr_tokenizer_a,
                                         receptor_tokenizer=self.receptor_tokenizer,
                                         tcr_max_len=self.cdr_max_len,
                                         receptor_max_len=self.ab_max_len,
                                         logger=self.logger)
        return er_dataset

    def _split_dataset(self):
        # if exists(join(self.neg_pair_save_dir, 'unseen_tcrs-seed-'+str(self.seed)+'.csv')):
        #     test_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'unseen_tcrs-seed-'+str(self.seed)+'.csv'))
        #     self.logger.info(f'Loading created unseen tcrs for test with shape {test_pair_df.shape}')

        tcr_list = list(set(self.pair_df['tcr']))
        selected_tcr_index_list = self.rng.integers(len(tcr_list), size=self.test_tcrs)
        self.logger.info(f'Select {self.test_tcrs} from {len(tcr_list)} tcr')
        selected_tcrs = [tcr_list[i] for i in selected_tcr_index_list]
        test_pair_df = self.pair_df[self.pair_df['tcr'].isin(selected_tcrs)]
        # test_pair_df.to_csv(join(self.neg_pair_save_dir, 'unseen_tcrs-seed-'+str(self.seed)+'.csv'), index=False)

        selected_tcrs = list(set(test_pair_df['tcr']))
        train_valid_pair_df = self.pair_df[~self.pair_df['tcr'].isin(selected_tcrs)]

        self.logger.info(
            f'{len(train_valid_pair_df)} pairs for train and valid and {len(test_pair_df)} pairs for test.')

        return train_valid_pair_df, test_pair_df

    def _create_pair(self):
        pair_df = pd.read_csv(self.data_dir)

        if self.shuffle:
            pair_df = pair_df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Shuffling dataset")
        self.logger.info(f"There are {len(pair_df)} samples")

        return pair_df

    def _load_seq_pairs(self):
        self.logger.info(f'Loading from {self.using_dataset}...')
        self.logger.info(f'Loading {self.tcr_seq_name} and {self.receptor_seq_name}')
        column_map_dict = {'alpha': 'cdr3a', 'beta': 'cdr3b', 'tcr': 'tcr'}
        keep_columns = [column_map_dict[c] for c in [self.tcr_seq_name, self.receptor_seq_name]]

        df_list = []
        for dataset in self.using_dataset:
            df = pd.read_csv(join(self.data_dir, dataset, 'full.csv'))
            df = df[keep_columns]
            df = df[(df[keep_columns[0]].map(is_valid_aaseq)) & (df[keep_columns[1]].map(is_valid_aaseq))]
            self.logger.info(f'Loading {len(df)} pairs from {dataset}')
            df_list.append(df[keep_columns])
        df = pd.concat(df_list)
        self.logger.info(f'Current data shape {df.shape}')
        df_filter = df.dropna()
        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After dropping na and duplicates, current data shape {df_filter.shape}')

        column_rename_dict = {column_map_dict[c]: c for c in [self.tcr_seq_name, self.receptor_seq_name]}
        df_filter.rename(columns=column_rename_dict, inplace=True)

        df_filter['label'] = [1] * len(df_filter)
        df_filter.to_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'), index=False)

        return df_filter





