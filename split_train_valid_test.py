#!/usr/bin/env python3
"""
split_train_valid_test.py

Generate train/valid/test JSON splits for semi-supervised CSV 2026 challenge.
Modified for 5-fold cross-validation.

Creates for each fold:
  - train_labeled.json: Labeled training samples
  - train_unlabeled.json: Unlabeled training samples  
  - valid.json: Validation samples (for training monitoring)
  - test.json: Test samples (for final evaluation with inference_post.py)

All splits are stratified by class.
"""

import os
import json
import h5py
import random
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate train/valid/test JSON splits for CSV 2026 challenge (5-fold CV)"
    )
    parser.add_argument("--root", type=str, default="./data", help="Dataset root path")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    return parser.parse_args()


def read_class_label(label_h5_path):
    """Read class label from HDF5 file."""
    try:
        with h5py.File(label_h5_path, 'r') as hf:
            cls_raw = hf['cls'][()]
            try:
                return int(cls_raw)
            except Exception:
                if hasattr(cls_raw, "tolist"):
                    return int(cls_raw.tolist()[0])
                else:
                    return int(cls_raw[0])
    except Exception:
        return 0


def create_eval_format(entries):
    """
    Convert entries to evaluation format with explicit cls_gt.
    """
    eval_list = []
    for entry in entries:
        cls_val = read_class_label(entry['label'])
        eval_list.append({
            'image': entry['image'],
            'label': entry['label'],
            'cls_gt': cls_val,
        })
    return eval_list


def split_into_folds(data_list, n_folds):
    """Split a list into n_folds approximately equal parts using round-robin."""
    folds = [[] for _ in range(n_folds)]
    for i, item in enumerate(data_list):
        folds[i % n_folds].append(item)
    return folds


def pct(count, total):
    """Calculate percentage."""
    return (count / total * 100) if total > 0 else 0.0


if __name__ == "__main__":
    args = get_args()

    dataset_root_path = args.root
    n_folds = args.n_folds
    images_dir_path = os.path.join(dataset_root_path, 'train', 'images')
    labels_dir_path = os.path.join(dataset_root_path, 'train', 'labels')

    # Collect filenames
    all_image_filenames = [name for name in os.listdir(images_dir_path) if name.endswith('.h5')]
    all_labeled_filenames = [name.replace('_label', '') for name in os.listdir(labels_dir_path) if name.endswith('.h5')]
    all_unlabeled_filenames = [name for name in all_image_filenames if name not in all_labeled_filenames]

    # Set seed for reproducibility
    random.seed(args.seed)

    # Build labeled entries grouped by class
    class0_list = []
    class1_list = []

    for filename in all_labeled_filenames:
        image_h5_path = os.path.abspath(os.path.join(images_dir_path, filename))
        label_h5_path = os.path.abspath(os.path.join(labels_dir_path, filename.replace('.h5', '_label.h5')))
        
        cls_val = read_class_label(label_h5_path)
        
        entry = {
            'image': image_h5_path,
            'label': label_h5_path,
        }
        
        if cls_val == 0:
            class0_list.append(entry)
        else:
            class1_list.append(entry)

    # Shuffle before splitting
    random.shuffle(class0_list)
    random.shuffle(class1_list)

    original_class0 = len(class0_list)
    original_class1 = len(class1_list)

    # ─────────────────────────────────────────────────────────────────────────
    # Build unlabeled training set (same for all folds)
    # ─────────────────────────────────────────────────────────────────────────
    train_unlabeled_list = []
    for filename in all_unlabeled_filenames:
        image_h5_path = os.path.abspath(os.path.join(images_dir_path, filename))
        train_unlabeled_list.append({
            'image': image_h5_path,
            'label': None
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Create stratified folds: split each class into n_folds parts
    # ─────────────────────────────────────────────────────────────────────────
    class0_folds = split_into_folds(class0_list, n_folds)
    class1_folds = split_into_folds(class1_list, n_folds)

    # ─────────────────────────────────────────────────────────────────────────
    # Generate each fold
    # ─────────────────────────────────────────────────────────────────────────
    print("")
    print("=" * 60)
    print(f"Generating {n_folds}-Fold Cross-Validation Splits")
    print("=" * 60)
    print("")
    print(f"Original labeled samples: {original_class0 + original_class1}")
    print(f"  - class 0 (stable):     {original_class0}")
    print(f"  - class 1 (vulnerable): {original_class1}")
    print(f"Unlabeled samples: {len(train_unlabeled_list)}")
    print("")

    for fold_idx in range(n_folds):
        # Create fold directory
        fold_dir = os.path.join(dataset_root_path, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)

        # ─────────────────────────────────────────────────────────────────────
        # TEST set: fold_idx-th portion of each class
        # ─────────────────────────────────────────────────────────────────────
        test_class0 = class0_folds[fold_idx]
        test_class1 = class1_folds[fold_idx]
        test_list = test_class0 + test_class1

        # ─────────────────────────────────────────────────────────────────────
        # VALIDATION set: (fold_idx + 1) % n_folds portion of each class
        # ─────────────────────────────────────────────────────────────────────
        valid_fold_idx = (fold_idx + 1) % n_folds
        valid_class0 = class0_folds[valid_fold_idx]
        valid_class1 = class1_folds[valid_fold_idx]
        valid_list = valid_class0 + valid_class1

        # ─────────────────────────────────────────────────────────────────────
        # TRAINING set: all other folds
        # ─────────────────────────────────────────────────────────────────────
        train_class0 = []
        train_class1 = []
        for i in range(n_folds):
            if i != fold_idx and i != valid_fold_idx:
                train_class0.extend(class0_folds[i])
                train_class1.extend(class1_folds[i])
        train_labeled_list = train_class0 + train_class1
        random.shuffle(train_labeled_list)

        # ─────────────────────────────────────────────────────────────────────
        # Create evaluation-format JSONs (with cls_gt for inference_post.py)
        # ─────────────────────────────────────────────────────────────────────
        test_eval_list = create_eval_format(test_list)
        valid_eval_list = create_eval_format(valid_list)

        # ─────────────────────────────────────────────────────────────────────
        # Save JSON files for this fold
        # ─────────────────────────────────────────────────────────────────────
        output_files = {
            'train_labeled.json': train_labeled_list,
            'train_unlabeled.json': train_unlabeled_list,
            'valid.json': valid_list,
            'test.json': test_eval_list,
        }

        for filename, data in output_files.items():
            filepath = os.path.join(fold_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)

        # ─────────────────────────────────────────────────────────────────────
        # Calculate and print statistics for this fold
        # ─────────────────────────────────────────────────────────────────────
        train_total = len(train_labeled_list)
        train_c0 = len(train_class0)
        train_c1 = len(train_class1)

        valid_total = len(valid_list)
        valid_c0 = len(valid_class0)
        valid_c1 = len(valid_class1)

        test_total = len(test_list)
        test_c0 = len(test_class0)
        test_c1 = len(test_class1)

        print("-" * 60)
        print(f"FOLD {fold_idx} (output: {fold_dir})")
        print("-" * 60)
        print("")
        print("TRAINING SET:")
        print(f"  Labeled samples: {train_total}")
        print(f"    - class 0 (stable):     {train_c0} ({pct(train_c0, train_total):.1f}%)")
        print(f"    - class 1 (vulnerable): {train_c1} ({pct(train_c1, train_total):.1f}%)")
        print(f"  Unlabeled samples: {len(train_unlabeled_list)}")
        print("")
        print("VALIDATION SET (for training monitoring):")
        print(f"  Total samples: {valid_total}")
        print(f"    - class 0 (stable):     {valid_c0} ({pct(valid_c0, valid_total):.1f}%)")
        print(f"    - class 1 (vulnerable): {valid_c1} ({pct(valid_c1, valid_total):.1f}%)")
        print("")
        print("TEST SET (for final evaluation):")
        print(f"  Total samples: {test_total}")
        print(f"    - class 0 (stable):     {test_c0} ({pct(test_c0, test_total):.1f}%)")
        print(f"    - class 1 (vulnerable): {test_c1} ({pct(test_c1, test_total):.1f}%)")
