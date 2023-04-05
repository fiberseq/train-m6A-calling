# m6A-calling

# install
```
python -m pip install git+https://github.com/mrvollger/m6a-calling
```
# reinstall
```
pip uninstall m6a-calling && pip install git+https://github.com/mrvollger/m6a-calling
```


# training the first model 
```
python m6a_calling/m6a_calling.py --train_data m6A_train.npz --device cpu --model_save_path  model.model 
```

# make training data
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
