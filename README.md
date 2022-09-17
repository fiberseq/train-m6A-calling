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