# Fibertools: training supervised and semi-supervised CNN models for m6A detection from Fiber-seq HiFi reads. 

This repository is for training supervised and semi-supervised convolutional neural network for m6A detection from Fiber-seq reads.


## Install
```
python -m pip install git+https://github.com/mrvollger/m6a-calling
```
## Reinstall
```
pip uninstall m6a-calling && pip install git+https://github.com/mrvollger/m6a-calling
```
## Prepare data
Download training and validation data for all three chemistries from Zenodo[TODO: add link] and set up requisite folders for saving models and results with the following instructions: 
```bash
mkdir -p paper_v1/data
cd paper_v1/data

wget TODO: add link
gzip -d TODO: add file name

mkdir -p paper_v1/models
mkdir -p paper_v1/results
mkdir -p paper_v1/figures
```
Ensure that the `sup_train_data`, `sup_val_data`, `semi_train_data` and `semi_val_data` variables in the `config.yml` file are pointing to the correct folder.

## Train supervised CNN model

You can run all three versions of the CNN model with the following commands: 

```bash
m6a_supervised_cnn --config_file paper_v1/config.yml --train_chem train_2_2_chemistry

m6a_supervised_cnn --config_file paper_v1/config.yml --train_chem train_3_2_chemistry

m6a_supervised_cnn --config_file paper_v1/config.yml --train_chem train_revio_chemistry
```

All required resources are defined in the `config.yml` file. See configuration section for more details on the resources. 

## Train semi-supervised CNN model

To run the semi-supervised CNN model, run the following commands: 

```bash
m6a_semi_supervised_cnn --config_file paper_v1/config.yml --train_chem train_2_2_chemistry

m6a_semi_supervised_cnn --config_file paper_v1/config.yml --train_chem train_3_2_chemistry

m6a_semi_supervised_cnn --config_file paper_v1/config.yml --train_chem train_revio_chemistry
```

## Inference on semi-supervised CNN model

To generate precision at different CNN scores for all chemistries, run the following commands:

```bash
m6a_semi_supervised_cnn_predict --config_file paper_v1/config.yml --train_chem train_2_2_chemistry

m6a_semi_supervised_cnn_predict --config_file paper_v1/config.yml --train_chem train_3_2_chemistry

m6a_semi_supervised_cnn_predict --config_file paper_v1/config.yml --train_chem train_revio_chemistry
```

## Make training data
```bash
m6adata \
  --threads 20 - \ 
  --hifi fiberseq.bam \ # input fiberseq bam file
  -o output.ml.npz \ # training data for a ML model
  --train \ # must be included
  -s 0.03 \ # sample just 3% of the data
  --is_u16 \ # has u16 kinetics values 
  -m 244 \ # min ML value to include in training
  --ec 6 \ # minimum CCS coverage to use in training
```

## Train new Fiber-seq chemistry models
To train semi-supervised CNN model for a new chemistry, you first need to generate a config file entry. The configuration file is divided into three sections, one section for each new chemistry. For training a model for a new chemistry, simply add the required resources in the `config.yml` file under a new header. Then run the following commands: 

```bash
m6a_supervised_cnn --config_file paper_v1/config.yml --train_chem <train_new_chemistry>

m6a_semi_supervised_cnn --config_file paper_v1/config.yml --train_chem <train_new_chemistry>

m6a_semi_supervised_cnn_predict --config_file paper_v1/config.yml --train_chem <train_new_chemistry>
```

Here is an example resources section: 

```
[train_new_chemistry]
fdr=0.05
device=cuda
sup_lr=0.0001
semi_lr=0.0001
input_size=6
input_length=15
semi_batch_size=32
semi_num_workers=2
sup_batch_size=32
sup_num_workers=8
train_sample=1
train_sample_fraction=0.4
val_sample=1
val_sample_fraction=0.25
min_pos_proportion=0.0001
add_m6a_percent=1.0
total_m6a_percent=70.0
supervised_train_epochs=30
semi_supervised_train_epochs=20
sup_train_data=paper_v1/data/PS00109_2.npz
sup_val_data=paper_v1/data/PS00109_3_val.npz
semi_train_data=paper_v1/data/PS00109_2_semi_train.npz
semi_val_data=paper_v1/data/PS00109_3_semi_val.npz
best_supervised_model_name=paper_v1/models/m6A_2_2_supervised_cnn.best_tmp.torch.pickle
final_supervised_model_name=paper_v1/models/m6A_2_2_supervised_cnn.final_tmp.torch
pretrain_model_name=paper_v1/models/m6A_2_2_supervised_cnn.best.torch.pickle
best_semi_supervised_model_name=paper_v1/models/m6A_2_2_semi_supervised_cnn.best.torch_tmp.pickle
final_semi_supervised_model_name=paper_v1/models/m6A_2_2_semi_supervised_cnn.final.torch_tmp.pickle
save_pos=paper_v1/results/semi_train_2_2_chemistry.npz
score_ap_json=paper_v1/results/semi_2_2_score_precision.json
score_ap_table=paper_v1/results/semi_2_2_score_precision.tsv
```

And here are the explanations for each variable:

```
fdr: acceptable false discovery rate for training the semi-supervised model
device: training device for models, cpu or cuda
input_size: number of input channels, currently 4 nucleotides, IPD and PW. 
input_length: length of every input, currently 15 nucleotides. 
semi_batch_size: Batch size for training semi-supervised CNN model. 
semi_num_workers: Number of worker threads for preprocessing data for semi-supervised CNN model. 
sup_batch_size: Batch size for training supervised CNN model. 
sup_num_workers: Number of worker threads for preprocessing data for supervised CNN model.
train_sample: Whether to subsample training data, 0-> false, 1 -> True
train_sample_fraction: Fraction of training data to subsample. 
val_sample: Whether to subsample validation data, 0-> false, 1 -> True
val_sample_fraction: Fraction of validation data to subsample. 
min_pos_proportion: minimum proportion of m6A calls above this threshold, ensures a good initialization for the semi-supervised training. 
add_m6a_percent: minimum proportion of addition m6A calls an iteration of semi-supervised model should find to continue training 
total_m6a_percent: Minimum proportion of total m6A calls the semi-supervised model needs to stop training.
supervised_train_epochs: number of training epochs for the supervised approach
semi_supervised_train_epochs: number of training epochs for the semi-supervised approach
sup_train_data: path to supervised training data
sup_val_data: path to supervised validation data
semi_train_data: path to semi-supervised training data
semi_val_data:  path to semi-supervised validation data
best_supervised_model_name: path and name for best supervised model
final_supervised_model_name: path and name for final supervised model
best_semi_supervised_model_name: path and name for best semi-supervised model
final_semi_supervised_model_name: path and name for final semi-supervised model
save_pos: path to save validation AUPR and # positives identified in validation set for semi-supervised approach
score_ap_json: json for cnn score to u8 precision conversion
score_ap_table: table with cnn scores, precision and u8 precision
```






