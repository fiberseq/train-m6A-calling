# Fibertools: training supervised and semi-supervised CNN models for m6A detection from Fiber-seq HiFi reads. 

This repository is for training supervised and semi-supervised convolutional neural network for m6A detection from Fiber-seq reads.


# install
```
python -m pip install git+https://github.com/mrvollger/m6a-calling
```
# reinstall
```
pip uninstall m6a-calling && pip install git+https://github.com/mrvollger/m6a-calling
```
# Prepare data
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

## Training supervised CNN model

You can run all three versions of the CNN model with the following commands: 

```bash
m6a_supervised_cnn --config_file paper_v1/config.yml --train_chem train_2_2_chemistry

m6a_supervised_cnn --config_file paper_v1/config.yml --train_chem train_3_2_chemistry

m6a_supervised_cnn --config_file paper_v1/config.yml --train_chem train_revio_chemistry
```

All required resources are defined in the `config.yml` file. See configuration section for more details on the resources. 

## Training semi-supervised CNN model

To run the semi-supervised CNN model, run the following commands: 

```bash
m6a_semi_supervised_cnn --config_file paper_v1/config.yml --train_chem train_2_2_chemistry

m6a_semi_supervised_cnn --config_file paper_v1/config.yml --train_chem train_3_2_chemistry

m6a_semi_supervised_cnn --config_file paper_v1/config.yml --train_chem train_revio_chemistry
```

## Inference on semi-supervised CNN model

To generate precision at different CNN scores for all chemistries, run the following commands:

```bash
m6a_semi_supervised_cnn_predict paper_v1/config.yml --train_chem train_2_2_chemistry

m6a_semi_supervised_cnn_predict paper_v1/config.yml --train_chem train_3_2_chemistry

m6a_semi_supervised_cnn_predict paper_v1/config.yml --train_chem train_revio_chemistry
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
