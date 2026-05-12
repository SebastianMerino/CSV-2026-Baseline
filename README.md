# Carotid Plaque Segmentation and Vulnerability Classification

Two-stage semi-supervised learning framework for carotid ultrasound analysis. Prepared for the Machine Learning in Medical Applications Course.

## Requirements

```bash
pip install torch torchvision numpy scipy scikit-learn tensorboard h5py pillow
```
## Data Preparation
First, generate the 5-fold cross-validation splits:

```bash
python split_train_valid_test.py
```
This will create the following structure:

```bash
data/
├── fold_0/
│   ├── train_labeled.json
│   ├── train_unlabeled.json
│   ├── valid.json
│   └── test.json
├── fold_1/
│   └── ...
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
└── fold_4/
    └── ...
```
## Training
Replace X with the fold number (0-4).

Stage 1: Semi-Supervised Segmentation
```bash
python train_new_arq.py \
    --model UNetV2 \
    --train_epochs 100 \
    --batch_size 12 \
    --save_path ./checkpoints_fold_X \
    --train-labeled-json ./data/fold_X/train_labeled.json \
    --valid-labeled-json ./data/fold_X/valid.json \
    --train-unlabeled-json ./data/fold_X/train_unlabeled.json
```

## Stage 2: Post-Segmentation Classification
```bash
python train_post.py \
    --seg_checkpoint ./checkpoints_fold_X/best_seg.pth \
    --save_path ./checkpoints_fold_X_post \
    --model UNetV2 \
    --classifier full \
    --train_epochs 100 \
    --batch_size 12 \
    --lr 0.001 \
    --loss_type focal \
    --gpu 0 \
    --train-labeled-json ./data/fold_X/train_labeled.json \
    --valid-labeled-json ./data/fold_X/valid.json
```

## Inference
```bash
python inference_post.py \
    --eval_json ./data/fold_X/test.json \
    --seg_checkpoint ./checkpoints_fold_X/best_seg.pth \
    --cls_checkpoint ./checkpoints_fold_X_post/full/best_opt.pth \
    --model UNetV2 \
    --classifier full \
    --save_preds \
    --save_grid \
    --gpu 0
```

## Training All Folds
To train all 5 folds sequentially:
```bash
for fold in 0 1 2 3 4; do
    echo "===== Training Fold $fold ====="
    
    # Stage 1
    python train_new_arq.py \
        --model UNetV2 \
        --train_epochs 100 \
        --batch_size 12 \
        --save_path ./checkpoints_fold_${fold} \
        --train-labeled-json ./data/fold_${fold}/train_labeled.json \
        --valid-labeled-json ./data/fold_${fold}/valid.json \
        --train-unlabeled-json ./data/fold_${fold}/train_unlabeled.json
    
    # Stage 2
    python train_post.py \
        --seg_checkpoint ./checkpoints_fold_${fold}/best_seg.pth \
        --save_path ./checkpoints_fold_${fold}_post \
        --model UNetV2 \
        --classifier full \
        --train_epochs 100 \
        --batch_size 12 \
        --lr 0.001 \
        --loss_type focal \
        --gpu 0 \
        --train-labeled-json ./data/fold_${fold}/train_labeled.json \
        --valid-labeled-json ./data/fold_${fold}/valid.json
    
    # Inference
    python inference_post.py \
        --eval_json ./data/fold_${fold}/test.json \
        --seg_checkpoint ./checkpoints_fold_${fold}/best_seg.pth \
        --cls_checkpoint ./checkpoints_fold_${fold}_post/full/best_opt.pth \
        --model UNetV2 \
        --classifier full \
        --save_preds \
        --save_grid \
        --gpu 0
done
```

## Output Structure
```bash
checkpoints_fold_X/
├── best_seg.pth          # Best segmentation model
├── best_cls.pth          # Best classification model (Stage 1)
├── latest.pth            # Latest checkpoint
└── tensorboard/          # Training logs

checkpoints_fold_X_post/
└── full/
    ├── best.pth          # Best classifier (F1@0.5)
    ├── best_opt.pth      # Best classifier (optimal threshold)
    ├── latest.pth        # Latest checkpoint
    └── tensorboard/      # Training logs
```