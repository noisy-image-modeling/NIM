# NIM

Noisy Image Modeling anonymous repository.

## Setup Environment

```zsh
conda env create -n <env name>
conda activate <env name>
echo "PYTHONPATH=`pwd`" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# we use wandb as default logger
wandb login
```

## NIM Pre-training

Downloading and extracting BTCV, AMOS22 and Abdomen1K-CT images to `umei/datasets/<dataset name>/origin`.

First run pre-processing:
```zsh
python scripts/snim/preprocess.py
```

Then run pre-training:
```zsh
python scripts/snim/main-omega.py <pre-training config path> do_train=true train_cache_num=<set cache according to your RAM size>
```

## Fine-tuning on BTCV

Run training:
```zsh
python scripts/btcv/main-omega.py `fine-tuning config path` do_train=true fold_ids="[0]" backbone.ckpt_path=<pre-trained checkpoint path> backbone.key_prefix=encoder. data_ratio=<training data ratio>
```

Run evaluation:
```zsh
python scripts/btcv/main-omega.py `fine-tuning config path` do_eval=true fold_ids="[0]" backbone.ckpt_path=<pre-trained path> backbone.key_prefix=encoder. data_ratio=<training data ratio> ckpt_path=<fine-tuned checkpoint path> sw_overlap=0.5 do_tta=false
```
