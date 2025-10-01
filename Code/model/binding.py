# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel, RoFormerModel


# beta, aphla, MHC, antigen

class BERTBinding_AbDab_cnn(nn.Module):
    def __init__(self, beta_dir, alpha_dir, antigen_dir, hla_dir, emb_dim=256):
        super().__init__()
        self.BetaModel = AutoModel.from_pretrained(beta_dir, output_hidden_states=True, return_dict=True)
        self.AlphaModel = AutoModel.from_pretrained(alpha_dir, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(antigen_dir, output_hidden_states=True, return_dict=True,
                                                      cache_dir="../esm2")
        self.HLAModel = AutoModel.from_pretrained(hla_dir, output_hidden_states=True, return_dict=True,
                                                  cache_dir="../esm2")

        self.cnn1 = MF_CNN(in_channel=120)
        self.cnn2 = MF_CNN(in_channel=120)
        self.cnn3 = MF_CNN(in_channel=12, hidden_size=76)  # 56)
        self.cnn4 = MF_CNN(in_channel=34, hidden_size=76)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 4, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, beta, alpha, antigen, hla):
        beta_encoded = self.BetaModel(**beta).last_hidden_state
        alpha_encoded = self.AlphaModel(**alpha).last_hidden_state
        antigen_encoded = self.AntigenModel(**antigen).last_hidden_state
        hla_encoded = self.HLAModel(**hla).last_hidden_state

        beta_cls = self.cnn1(beta_encoded)
        alpha_cls = self.cnn2(alpha_encoded)
        antigen_cls = self.cnn3(antigen_encoded)
        hla_cls = self.cnn4(hla_encoded)

        concated_encoded = torch.concat((beta_cls, alpha_cls, hla_cls, antigen_cls), dim=1)

        output = self.binding_predict(concated_encoded)

        return output


class BERTBinding_biomap_cnn(nn.Module):
    def __init__(self, beta_dir, alpha_dir, antigen_dir, emb_dim=256):
        super().__init__()
        self.BetaModel = AutoModel.from_pretrained(beta_dir, output_hidden_states=True, return_dict=True)
        self.AlphaModel = AutoModel.from_pretrained(alpha_dir, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(antigen_dir, output_hidden_states=True, return_dict=True,
                                                      cache_dir="../esm2")

        self.cnn1 = MF_CNN(in_channel=170)
        self.cnn2 = MF_CNN(in_channel=170)
        self.cnn3 = MF_CNN(in_channel=512, hidden_size=76)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 3, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, beta, alpha, antigen):
        beta_encoded = self.BetaModel(**beta).last_hidden_state
        alpha_encoded = self.AlphaModel(**alpha).last_hidden_state
        antigen_encoded = self.AntigenModel(**antigen).last_hidden_state

        beta_cls = self.cnn1(beta_encoded)
        alpha_cls = self.cnn2(alpha_encoded)
        antigen_cls = self.cnn3(antigen_encoded)

        concated_encoded = torch.concat((beta_cls, alpha_cls, antigen_cls), dim=1)

        output = self.binding_predict(concated_encoded)

        return output


class BERTBinding_4_input_cnn(nn.Module):
    def __init__(self, PretrainModel_dir, emb_dim):
        super().__init__()
        self.CDRModel_a = AutoModel.from_pretrained(PretrainModel_dir)
        self.CDRModel_b = AutoModel.from_pretrained(PretrainModel_dir)
        self.CDRModel_c = AutoModel.from_pretrained(PretrainModel_dir)
        self.ABModel = AutoModel.from_pretrained(PretrainModel_dir, cache_dir="../esm2")

        self.cnn1 = MF_CNN(in_channel=18)
        self.cnn2 = MF_CNN(in_channel=18)
        self.cnn3 = MF_CNN(in_channel=18)
        self.cnn4 = MF_CNN(in_channel=120)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 4, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, tcr_a, tcr_b, tcr_c, receptor):
        tcr_a_encoded = self.CDRModel_a(**tcr_a).last_hidden_state
        tcr_b_encoded = self.CDRModel_b(**tcr_b).last_hidden_state
        tcr_c_encoded = self.CDRModel_c(**tcr_c).last_hidden_state
        receptor_encoded = self.ABModel(**receptor).last_hidden_state

        tcr_a_cls = self.cnn1(tcr_a_encoded)
        tcr_b_cls = self.cnn2(tcr_b_encoded)
        tcr_c_cls = self.cnn3(tcr_c_encoded)
        receptor_cls = self.cnn4(receptor_encoded)

        concated_encoded = torch.concat((tcr_a_cls, tcr_b_cls, tcr_c_cls, receptor_cls), dim=1)

        output = self.binding_predict(concated_encoded)

        return output


class MF_CNN(nn.Module):
    def __init__(self, in_channel=118, emb_size=20, hidden_size=92):  # 189):
        super(MF_CNN, self).__init__()

        # self.emb = nn.Embedding(emb_size,128)  # 20*128
        self.conv1 = cnn_liu(in_channel=in_channel, hidden_channel=64)  # 118*64
        self.conv2 = cnn_liu(in_channel=64, hidden_channel=32)  # 64*32

        self.conv3 = cnn_liu(in_channel=32, hidden_channel=32)

        self.fc1 = nn.Linear(32 * hidden_size, 128)  # 32*29*512
        self.fc2 = nn.Linear(128, 128)

        self.fc3 = nn.Linear(128, 128)

    def forward(self, x):
        # x = x
        # x = self.emb(x)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = x.view(x.shape[0], -1)

        x = nn.ReLU()(self.fc1(x))
        sk = x
        x = self.fc2(x)

        x = self.fc3(x)
        return x + sk


class cnn_liu(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=2, out_channel=2):
        super(cnn_liu, self).__init__()

        self.cnn = nn.Conv1d(in_channel, hidden_channel, kernel_size=5, stride=1)  # bs * 64*60
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)  # bs * 32*30

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.max_pool(x)
        x = self.relu(x)
        return x