# TCRBinder: unified pre-trained language model with paired-chain coherence for predicting T-cell receptor binding specificity
<p align="center">
<img src="https://github.com/dongweihe/TCRBinder/blob/main/TCRBinder.png" align="middle" height="80%" width="80%" />
</p >
Deciphering how human T cells recognise peptide–HLA (pHLA) complexes underpins next-generation vaccines and personalised immunotherapies, yet extreme sequence diversity and paired chains interdependence still hamper reliable in silico prediction of T-cell receptor (TCR) specificity. To overcome these hurdles we built TCRBinder, a paired chain aware deep model with a multi-branch encoder that routes each molecular component through a dedicated transformer, which allows ESM2 to capture contextual signals in both HLA pseudo-sequences and antigenic peptides, while a rotary-embedding RoFormer simultaneously processes the TCR $\alpha$ and $\beta$ chains so that cross-chain coherence is enforced to emulate Peptide–HLA–TCR crosstalk and expose residue level contact motifs. Across peptide–HLA–TCR and peptide–TCR benchmarks the model delivered state-of-the-art accuracy (AUC-ROC = 0.95, AUPR = 0.86 for the ternary task) and remained superior on multiple independent and external datasets. Its affinity scores tightly tracked clonal expansion dynamics and, on a large SARS-CoV-2 repertoire containing wholly unseen peptides, improved predictive precision by more than ten percentage points over leading alternatives. Moreover, TCRBinder’s outputs mirrored clinical response to immune checkpoint blockade in two patient cohorts and provided mechanistic insight by pinpointing contact hot-spots and quantifying residue contributions to binding probability. These capabilities position TCRBinder as a versatile tool for rational antigen discovery, immunotherapy stratification and neoantigen vaccine design.

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

# clone the source code of THLAnet
$ git clone https://github.com/
$ cd THLAnet
```

# Dataset description
The raw interaction data were compiled from publicly accessible resources, including VDJdb (https://vdjdb.cdr3.net/), IEDB (https://www.iedb.org/), McPAS-TCR (http://friedmanlab.weizmann.ac.il/McPAS-TCR/), OTS (https://opig.stats.ox.ac.uk/webapps/ots), and the 10x Genomics datasets (https://www.10xgenomics.com/datasets).

By default, you can run our model using the immunogenicity dataset with:
```
python finetuning_er_main.py --config ./config/common/TCRBinder.json

```

# Acknowledgments

If you have any questions, please contact us via email:

[Weihe Dong](mail to:WeiheDong@stu.hit.edu.cn)
