# Dual-Path-Transformer-Network-PyTorch
Unofficial implementation of Dual-Path Transformer Network for speech separation(Interspeech 2020)

## Plan
- [x] Data pre-processing
- [x] Training
- [ ] Inference

## Data pre-processing

-- in-dir: It means your WSJ2mix dataset directory. (It has tr/cv/tt folders)

-- out-dir: It saves json files(file information) (recommand not to change)

```bash
$ python preprocess.py --in-dir /data/min --out-dir data --sample-rate 8000
```

## Training

```bash
$ python train.py
```

If you change --out-dir option, you have to set --train_dir '{your_directory}/tr' --valid_dir '{your_directory}/cv' 

```bash
$ python train.py --train_dir 'data/tr' --valid_dir 'data/cv'
```

## Reference

https://github.com/kaituoxu/Conv-TasNet

https://github.com/ujscjj/DPTNet
