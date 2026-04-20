#!/usr/bin/env python3
"""
split_train_valid_test.py

Generate train/valid/test JSON splits for semi-supervised CSV 2026 challenge.

Creates:
  - train_labeled.json: Labeled training samples
  - train_unlabeled.json: Unlabeled training samples  
  - valid.json: Validation samples (for training monitoring)
  - test.json: Test samples (for final evaluation with inference_post.py)

All splits are balanced by class when possible.
"""

import os
import json
import h5py
import random
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate train/valid/test JSON splits for CSV 2026 challenge"
    )
    parser.add_argument("--root", type=str, default="./data", help="Dataset root path")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--val_size", type=int, default=30,
                        help="Number of samples for validation (balanced 1:1)")
    parser.add_argument("--test_size", type=int, default=30,
                        help="Number of samples for test (balanced 1:1)")
    return parser.parse_args()


def read_class_label(label_h5_path):
    """Read class label from h5 file."""
    try:
        with h5py.File(label_h5_path, 'r') as hf:
            cls_raw = hf['cls'][()]
            try:
                cls_val = int(cls_raw)
            except Exception:
                if hasattr(cls_raw, "tolist"):
                    cls_val = int(cls_raw.tolist()[0])
                else:
                    cls_val = int(cls_raw[0])
        return cls_val
    except Exception:
        return 0


def balanced_sample(class0_list, class1_list, total_size, seed=None):
    """
    Sample balanced subset from two class lists.
    
    Returns:
        sampled: List of sampled entries
        remaining0: Remaining class 0 entries
        remaining1: Remaining class 1 entries
    """
    if seed is not None:
        random.seed(seed)
    
    per_class = total_size // 2
    
    # Limit to available samples
    avail0 = len(class0_list)
    avail1 = len(class1_list)
    per_class = min(per_class, avail0, avail1)
    
    if per_class == 0:
        return [], class0_list.copy(), class1_list.copy()
    
    # Sample
    sampled0 = random.sample(class0_list, per_class)
    sampled1 = random.sample(class1_list, per_class)
    
    # Get remaining
    sampled0_set = set(e['image'] for e in sampled0)
    sampled1_set = set(e['image'] for e in sampled1)
    
    remaining0 = [e for e in class0_list if e['image'] not in sampled0_set]
    remaining1 = [e for e in class1_list if e['image'] not in sampled1_set]
    
    return sampled0 + sampled1, remaining0, remaining1


def create_eval_format(entries, labels_dir_path=None):
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


if __name__ == "__main__":
    args = get_args()

    dataset_root_path = args.root
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
    # Step 1: Sample TEST set first (held out for final evaluation)
    # ─────────────────────────────────────────────────────────────────────────
    test_list, class0_remaining, class1_remaining = balanced_sample(
        class0_list, class1_list, args.test_size, seed=args.seed
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Sample VALIDATION set from remaining
    # ─────────────────────────────────────────────────────────────────────────
    valid_list, class0_remaining, class1_remaining = balanced_sample(
        class0_remaining, class1_remaining, args.val_size, seed=args.seed + 1
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Remaining labeled samples go to TRAIN
    # ─────────────────────────────────────────────────────────────────────────
    train_labeled_list = class0_remaining + class1_remaining
    random.shuffle(train_labeled_list)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Build unlabeled training set
    # ─────────────────────────────────────────────────────────────────────────
    train_unlabeled_list = []
    for filename in all_unlabeled_filenames:
        image_h5_path = os.path.abspath(os.path.join(images_dir_path, filename))
        train_unlabeled_list.append({
            'image': image_h5_path,
            'label': None
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Create evaluation-format JSONs (with cls_gt for inference_post.py)
    # ─────────────────────────────────────────────────────────────────────────
    test_eval_list = create_eval_format(test_list)
    valid_eval_list = create_eval_format(valid_list)  # Optional: if you want to eval on valid too

    # ─────────────────────────────────────────────────────────────────────────
    # Save JSON files
    # ─────────────────────────────────────────────────────────────────────────
    output_files = {
        'train_labeled.json': train_labeled_list,
        'train_unlabeled.json': train_unlabeled_list,
        'valid.json': valid_list,           # For training script (CSVSemiDataset)
        'test.json': test_eval_list,        # For inference_post.py (with cls_gt)
    }

    for filename, data in output_files.items():
        filepath = os.path.join(dataset_root_path, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    # ─────────────────────────────────────────────────────────────────────────
    # Calculate statistics
    # ─────────────────────────────────────────────────────────────────────────
    train_class0 = len(class0_remaining)
    train_class1 = len(class1_remaining)
    train_total = train_class0 + train_class1

    valid_class0 = sum(1 for e in valid_list if read_class_label(e['label']) == 0)
    valid_class1 = len(valid_list) - valid_class0

    test_class0 = sum(1 for e in test_list if read_class_label(e['label']) == 0)
    test_class1 = len(test_list) - test_class0

    def pct(count, total):
        return (count / total * 100) if total > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Print summary
    # ─────────────────────────────────────────────────────────────────────────
    print("")
    print("=" * 60)
    print("Dataset Split Summary")
    print("=" * 60)
    print("")
    print(f"Original labeled samples: {original_class0 + original_class1}")
    print(f"  - class 0 (stable):     {original_class0}")
    print(f"  - class 1 (vulnerable): {original_class1}")
    print("")
    print("-" * 60)
    print("")
    print("TRAINING SET:")
    print(f"  Labeled samples: {train_total}")
    print(f"    - class 0 (stable):     {train_class0} ({pct(train_class0, train_total):.1f}%)")
    print(f"    - class 1 (vulnerable): {train_class1} ({pct(train_class1, train_total):.1f}%)")
    print(f"  Unlabeled samples: {len(train_unlabeled_list)}")
    print("")
    print("VALIDATION SET (for training monitoring):")
    print(f"  Total samples: {len(valid_list)}")
    print(f"    - class 0 (stable):     {valid_class0} ({pct(valid_class0, len(valid_list)):.1f}%)")
    print(f"    - class 1 (vulnerable): {valid_class1} ({pct(valid_class1, len(valid_list)):.1f}%)")
    print("")
    print("TEST SET (for final evaluation):")
    print(f"  Total samples: {len(test_list)}")
    print(f"    - class 0 (stable):     {test_class0} ({pct(test_class0, len(test_list)):.1f}%)")
    print(f"    - class 1 (vulnerable): {test_class1} ({pct(test_class1, len(test_list)):.1f}%)")
    print("")
    print("-" * 60)
    print("")
    print("Output files:")
    for filename in output_files.keys():
        filepath = os.path.join(dataset_root_path, filename)
        print(f"  - {filepath}")
    print("")
    print("Usage:")
    print("  Training:   python train.py --train-labeled-json ./data/train_labeled.json \\")
    print("                              --train-unlabeled-json ./data/train_unlabeled.json \\")
    print("                              --valid-labeled-json ./data/valid.json")
    print("")
    print("  Evaluation: python inference_post.py --eval_json ./data/test.json \\")
    print("                                       --seg_checkpoint ./checkpoints/best.pth \\")
    print("                                       --cls_checkpoint ./checkpoints_post/best_opt.pth")
    print("=" * 60)