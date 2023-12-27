# Precipitation forecasting bench

The precipitation forecasting aims to generate future time frames by learning from the historical obsevation.

## Quick walkthrough for implementation and training

1. Add your model file to /models directory
```
.py
```
2. Write your own config file in /options 
```
.py
```
3. Run 
```
python train.py --config .py
```


## Directory explained

* In the root directory,
  - `train.py`: Entry point to the training. Runs trainer.py.
  - `utils.py`:
