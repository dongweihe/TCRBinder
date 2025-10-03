# TCRBinder: unified pre-trained language model with paired-chain coherence for predicting T-cell receptor binding specificity
<p align="center">
<img src="https://github.com/dongweihe/TCRBinder/blob/main/TCRBinder.png" align="middle" height="100%" width="100%" />
</p >
Deciphering how human T cells recognise peptide–HLA (pHLA) complexes underpins next-generation vaccines and personalised immunotherapies, yet extreme sequence diversity and paired chains interdependence still hamper reliable in silico prediction of T-cell receptor (TCR) specificity. To overcome these hurdles we built TCRBinder, a paired chain aware deep model with a multi-branch encoder that routes each molecular component through dedicated transformer-based modules to capture contextual signals in both HLA pseudo-sequences and antigenic peptides and simultaneously processes the TCR $\alpha$ and $\beta$ chains. By doing so, chain coherence is enforced to emulate peptide–HLA–TCR (PHT) interactions and expose residue-level contact motifs. Across PHT and peptide–TCR (pTCR) benchmarks, the model delivered state-of-the-art performance (AUC-ROC = 0.944, AUPR = 0.855 for the PHT task) and remained superior on multiple independent datasets. Our results closely monitored the dynamics of clonal expansion and, in a large SARS-CoV-2 repertoire containing wholly unseen peptides, improved the AUC-ROC by up to 17.4\% over the leading alternatives. Moreover, TCRBinder provided mechanistic insight by pinpointing contact hot-spots and quantifying residue contributions to binding probability. These capabilities position TCRBinder as a versatile tool for rational antigen discovery, immunotherapy stratification, and neoantigen vaccine design.

# The environment of TCRBinder
```
numpy==1.22.4
pandas==1.4.3
scikit_learn==1.2.2
scipy==1.7.3
torch==1.12.1
tqdm==4.64.1
transformers==4.24.0
```

# Installation Guide
Clone this Github repo and set up a new conda environment. It normally takes about 10 minutes to install on a normal desktop computer.
```
# create a new conda environment
$ conda create --name TCRBinder python=3.9
$ conda activate TCRBinder

# install requried python dependencies
$ pip install -r requirements.txt

# clone the source code of TCRBinder
$ git clone https://github.com/dongweihe/TCRBinder.git
$ cd TCRBinder
```

# Dataset description
The raw interaction data were compiled from publicly accessible resources, including VDJdb (https://vdjdb.cdr3.net/), IEDB (https://www.iedb.org/), McPAS-TCR (http://friedmanlab.weizmann.ac.il/McPAS-TCR/), OTS (https://opig.stats.ox.ac.uk/webapps/ots), and the 10x Genomics datasets (https://www.10xgenomics.com/datasets).

# Training and Usage Guide

We separately pre-train two RoFormer models on TCR α chain and β chain sequences. Then, utilize pHLA-TCR (PHT) binding data to fine-tune these models, finally forming the TCRBinder model.

## Pre-training RoFormer Models
You need to download the files “ProcessedData/CDR3_Alpha.csv” and “ProcessedData/CDR3_Beta.csv” from Zenodo (https://doi.org/10.5281/zenodo.14282419), and place them into the “/ProcessedData” folder.
The commands for pre-training two RoFormer models on TCR α chain and β chain sequences are:
```
python pretrain_main.py --config ./config/common/pretrain_alpha.json
python pretrain_main.py --config ./config/common/pretrain_beta.json
```
After training completes, the pre-trained Roformer models will be saved in `../Result_alpha/checkpoints/Pretrain/XXXX_XXXXXX` and `../Result_beta/checkpoints/Pretrain/XXXX_XXXXXX` folders, where `XXXX_XXXXXX` is the training timestamp.

## Fine-tuning TCRBinder
Since we use ESM2 model parameters as the antigen model, you need to download the ESM2 model parameters from Hugging Face (https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main) and place them into the `/Code/esm2/esm2_150m` directory.
You can use our provided simple example dataset (“Sample.csv”) to run our model. The training command for TCRBinder is:
```
python finetuning_main.py --config ./config/common/finetuning.json
```
Before running the task, please timely replace the file paths for `"tcr_tokenizer_dir"`, `"beta_dir"`, and `"alpha_dir"` in the `finetuning.json` configuration file.

## Evaluation on External Datasets
After fine-tuning, you can evaluate external datasets. Before evaluation, please copy the absolute path of `model_best.pth` from the `../Result_PHT/checkpoints/` directory to the `"discriminator_resume"` field in the `generalization.json` file. Then, replace `beta_dir` with `../Result_beta/checkpoints/Pretrain/XXXX_XXXXXX/`, replace `"alpha_dir"` with `../Result_alpha/checkpoints/Pretrain/XXXX_XXXXXX/`, and update the corresponding `"tcr_tokenizer_dir"`, `"alpha_dir"`, and `"beta_dir"`.

The generalization command for TCRBinder is:
```
python generalization.py --config ./config/common/generalization.json
```

# Acknowledgments

If you have any questions, please contact us via email:

[Weihe Dong](mailto:WeiheDong@stu.hit.edu.cn)