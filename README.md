# MIRA (PyTorch)
Official PyTorch implementation of \
[Unsupervised Visual Representation Learning via Mutual Information Regularized Assignment](https://arxiv.org/abs/2211.02284)  \
Dong Hoon Lee, Sungik Choi, Hyunwoo Kim, Sae-Young Chung \
NeurIPS 2022

## Requirements
- `torch==1.11.0`
- `torchvision==0.12.0`
- `nvidia-dali-cuda110==1.12.0` (optional)
- `tqdm`
- `wandb`

We also include `environment.yaml` for [conda](https://conda.io) environment. 

## Pretraining
```
python -m torch.distributed.launch --nproc_per_node=16 train.py --data-dir $DATA_DIR --entity $WANDB_ENTITY --project $WANDB_PROJECT 
```

## Evaluation
### Linear evaluation on ImageNet
```
python -m torch.distributed.launch --nproc_per_node=16 eval_linear.py $MODEL --data-dir $DATA_DIR --entity $WANDB_ENTITY --project $WANDB_PROJECT
```

<!-- ### Semi-supervised evaluation on ImageNet
```
python -m torch.distributed.launch --nproc_per_node=16 eval_semi.py $MODEL --data-dir $DATA_DIR --entity $WANDB_ENTITY --project $WANDB_PROJECT
```

### *k*-NN evaluation on ImageNet
```
python -m torch.distributed.launch --nproc_per_node=16 eval_knn.py $MODEL --label-perc $LABEL_PERC --data-dir $DATA_DIR
```
-->

## Pretrained weights on ImageNet-1k
| method | arch | batch-size | epochs | multi-crop | linear eval | download | script |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MIRA | ResNet-50 | 4096 | 800 | 2x224 | 74.1 | [mira-800ep.pth.tar](https://www.dropbox.com/s/m5nepxvp4dkh8bo/mira-800ep.pth.tar?dl=0) | [script](https://www.dropbox.com/s/u5kgrwnj4gfaww8/mira_800ep_pretrain.sh?dl=0) |
| MIRA | ResNet-50 | 4096 | 400 | 2x224 + 6x96 | 75.6 | [mira-mc-400ep.pth.tar](https://www.dropbox.com/s/vygli69xu6x4vlo/mira-mc-400ep.pth.tar?dl=0) | [script](https://www.dropbox.com/s/y02dqc8j9rpe2d0/mira_mc_400ep_pretrain.sh?dl=0) |

## Acknowledgement
Our implementation uses code from the following repositories: [DINO](https://github.com/facebookresearch/dino), [SwAV](https://github.com/facebookresearch/swav) [MoCo-v3](https://github.com/facebookresearch/moco-v3), [VISSL](https://github.com/facebookresearch/vissl), and [solo-learn](https://github.com/vturrisi/solo-learn)

## Citation
If you find our work useful, please consider citing it:
```
@inproceedings{lee2022mira,
    title={Unsupervised Visual Representation Learning via Mutual Information Regularized Assignment},
    author={Lee, Dong Hoon and Choi, Sungik and Kim, Hyunwoo and Chung, Sae-Young},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```