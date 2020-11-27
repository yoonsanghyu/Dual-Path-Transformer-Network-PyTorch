# Dual-Path-Transformer-Network-PyTorch
Unofficial implementation of Dual-Path Transformer Network for speech separation(Interspeech 2020)

## Data pre-processing

-- in-dir: It means your WSJ2mix dataset directory. (It has tr/cv/tt folders)
-- out-dir: It saves json files(file information) (recommand to use 'data' directory like me.)

```bash
$ python preprocess.py --in-dir /data/min --out-dir data --sample-rate 8000
```

## Training

```bash
$ python train.py
```

If you choose --out-dir option, you have to set --train_dir '{your_directory}/tr' --valid_dir '{your_directory}/cv' 

```bash
$ python train.py --tr-
```

## Reference

https://github.com/kaituoxu/Conv-TasNet

https://github.com/ujscjj/DPTNet
