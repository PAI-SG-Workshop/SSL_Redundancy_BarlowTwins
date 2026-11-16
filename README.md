# SSL_Redundancy_BarlowTwins

**Barlow Twins self-supervised pre-training of ResNet-18 on FashionLetters36 - PAI-SG SSL**

Part of the *Physical AI* reference series of the **Statera Working Group**. One algorithm (redundancy reduction) of the
SSL comparison in [SSL_Wave1_FL36](https://github.com/Statera-Workshop/SSL_Wave1_FL36); semi-supervised follow-up in [SemiSL_Wave2_FL36](https://github.com/Statera-Workshop/SemiSL_Wave2_FL36).

## Method

Projector 512-2048(BN)-2048(BN)-2048, loss on the batch-standardised cross-correlation: on-diagonal (1 - c)^2 + 0.0051 x off-diagonal c^2, x 0.1; LARS lr 0.3, wd 1e-4, asymmetric augmentation.

**Common setup** (identical for the six SSL repositories; full protocol and the cross-method comparison:
[SSL_Wave1_FL36](https://github.com/Statera-Workshop/SSL_Wave1_FL36)). ResNet-18 with the ImageNet stem (= the CenterNet ResNet-18; `torchvision` keys = `CenterNet.net.*`),
pre-trained on the 4,096 FashionLetters36 train images without labels or boxes. Two 96 x 96 views per sample, each
a RandomResizedCrop of one 128-256 px region of a 608 x 608 image; no horizontal flip (letters are not
mirror-invariant); grayscale input x / 255 as 3 channels, no mean / std; batch 256, warmup 10 epochs then cosine,
fp32, seed 777; 1 epoch = 32,768 samples. Recipes from the solo-learn CIFAR configs.

**Evaluation.** Representation: Crop36 (26,716 / 6,699 object crops, 36 classes), kNN (k 20) and linear probe
after re-estimating BN statistics on the train crops (labels unused). Detection: CenterNet on FL36 (36 classes),
the series protocol of [OD_Image_1-Stage_CenterNet](https://github.com/Statera-Workshop/OD_Image_1-Stage_CenterNet) v1.1 (Adam 1e-4, effective
batch 8, PreciseBN, last epoch, test mAP50); only the backbone initialisation differs.
From-scratch CenterNet reference: 30 epochs 0.611 / 0.609 (public baseline 0.644), 90 epochs 0.862 / 0.859.

## Results (FashionLetters36)

| Backbone | Crop36 kNN (BN re-est.) | Linear probe top-1 |
| --- | --- | --- |
| random init (reference) | 0.642 | 0.605 |
| 50 epochs | 0.867 | 0.896 |
| 200 epochs | 0.886 | 0.907 |
| 800 epochs | 0.893 | 0.910 |

**CenterNet, 100 % labels, 30 epochs (test mAP50):** 50 epochs, SSL seeds 777 / 778 / 779: 0.744 / 0.755 / 0.765; **200 epochs: 0.822 / 0.803 / 0.814 (mean 0.813)**; 800 epochs: 0.814.

### Label efficiency (CenterNet mAP50, step-matched labeled subsets)

| Init | Budget | 1 % | 5 % | 10 % | 25 % | 100 % |
| --- | --- | --- | --- | --- | --- | --- |
| from scratch | 30 ep | 0.139 | 0.463 | 0.564 | 0.583 | 0.611 |
| from scratch | 90 ep | 0.131 | 0.605 | 0.725 / 0.720 | 0.814 | 0.862 / 0.859 |
| Barlow 200 | 30 ep | 0.319 | 0.676 | 0.771 | 0.806 | 0.813 (3 seeds) |
| Barlow 200 | 90 ep | 0.330 | 0.711 | 0.770 / 0.774 / 0.775 (3 SSL seeds) | 0.831 | 0.867 / 0.872 |

**Best and steadiest SSL backbone of the comparison.** At 30 fine-tuning epochs it adds +0.20 over from-scratch; after 90 epochs the gain falls with the number of labels: +0.20 (1 %), +0.11 (5 %), +0.05 (10 %, outside the seed spread), +0.02 (25 %), +0.01 (100 %). SSL saturates at ~200 epochs (800 epochs: same detection). Barlow 800 at 90 fine-tuning epochs: 10 % 0.775, 100 % 0.884. With semi-supervised CenterNet on top (10 % labels, 90 epochs) it reaches 0.821 - see [SemiSL_Detection_UnbiasedTeacher](https://github.com/Statera-Workshop/SemiSL_Detection_UnbiasedTeacher).

## Weights

Release `v1.0` (torchvision ResNet-18 keys, `fc` removed, SHA-256 in `SHA256SUMS.txt`):

- `ssl_barlow_e50_backbone.pth`
- `ssl_barlow_e50_s778_backbone.pth`
- `ssl_barlow_e50_s779_backbone.pth`
- `ssl_barlow_e200_backbone.pth`
- `ssl_barlow_e200_s778_backbone.pth`
- `ssl_barlow_e200_s779_backbone.pth`
- `ssl_barlow_e800_backbone.pth`

```python
import torch, torchvision
m = torchvision.models.resnet18(weights=None); m.fc = torch.nn.Identity()
m.load_state_dict(torch.load("ssl_barlow_e800_backbone.pth", weights_only=True))   # strict
# input: grayscale x / 255 repeated to 3 channels, no mean / std
```

## Run

```bash
export PHYSICAL_AI_ROOT=~/SSL_Wave1        # FL36 generated with OD_Image_1-Stage_CenterNet (data_processing_fl36.py)
python code/prep_ssl_data.py               # unlabeled image array + Crop36 (RESULT OK)
python code/ssl_train_ext.py --method barlow --epochs 200 --workers 8 --gate-margin -1 --knn-every 25
python code/final_eval.py                  # kNN / probe with re-estimated BN
# CenterNet: copy code/train_image_centernet_ssl.py and code/train_image_centernet_lab.py into OD_Image_1-Stage_CenterNet
python train_image_centernet_ssl.py --pretrained $PHYSICAL_AI_ROOT/runs/ssl_barlow/backbone_e200.pth
```

`code/` holds the shared Wave 1 code (all SSL methods live in `ssl_train.py` / `ssl_train_ext.py`; this repository's
runs use `--method barlow`). `results/` holds this method's training logs, its CenterNet runs and the
from-scratch baselines. Third-party code: [THIRD_PARTY.md](THIRD_PARTY.md) (solo-learn MIT notice in `licenses/`).
