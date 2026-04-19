# CSV 2026 Challenge 

## 📦 Dataset Description

This repository provides the training dataset for the **CSV 2026:Carotid Plaque Segmentation and Vulnerability Assessment in Ultrasound**.
The challenge aims to promote the development of robust algorithms for automated plaque segmentation and vulnerability classification.

The dataset is collected from **Multiple Ultrasound Systems and Multiple Hospital Centers**, introducing substantial variability in imaging appearance and clinical practice.
At the current stage, **only the training set is released**, following a semi-supervised learning setting.

🔒 **All released ultrasound images have undergone strict de-identification and anonymization procedures** prior to public release.
The dataset **does not contain any visible personal identifiers**, and no information that could be used to identify or infer the identity of individual subjects is included.

---

## 🗂️ 1. Dataset Structure

The training set contains **1000 cases** in total and is organized as follows:

```
train/
├── images/
│   ├── 0000.h5
│   ├── 0001.h5
│   ├── ...
│   └── 0999.h5
└── labels/
    ├── 0000_label.h5
    ├── 0001_label.h5
    ├── ...
    └── 0199_label.h5
```

### 1.1 Image Data (`train/images/`)

- Contains **1000 `.h5` files**
- File naming: `0000.h5` to `0999.h5`.
- Each `.h5` file includes two ultrasound views:
  - `long_img`: longitudinal (long-axis) carotid ultrasound image
  - `trans_img`: transverse (short-axis) carotid ultrasound image


### 1.2 Label Data (`train/labels/`)

- Contains annotations for **the first 200 cases only**, consistent with the semi-supervised setting.
- File naming: `0000_label.h5` to `0199_label.h5`.
- Each label file includes:
  - `long_mask`: segmentation mask for the longitudinal view
  - `trans_mask`: segmentation mask for the transverse view
  - `cls`: case-level classification label
    - `0`: Low Risk
    - `1`: High Risk  

- Risk stratification follows the **2024 Plaque-RADS guideline**: **Saba L, Cau R, Murgia A, et al. Carotid plaque-RADS: a novel stroke risk classification system[J]. Cardiovascular Imaging, 2024, 17(1): 62-75.** 

Mapping rules:
- **RADS = 2** → Low Risk
- **RADS = 3–4** → High Risk

Cases **0000–0199** are provided as **labeled data**.
Cases **0200–0999** are provided as **unlabeled data**.

- Entire dataset (training, validation, testing):  
  **Low Risk : High Risk ≈ 4 : 1**
- Labeled training subset only:  
  **Low Risk : High Risk = 1 : 1**

The balanced labeled subset ensures sufficient high-risk cases for supervised learning.

---


## 🧠 2. Proportion of Labeled Data

In earlier announcements, it was stated that **10% of the training data** would be annotated.
After further review, the proportion of labeled data was increased to **20% (200 cases)**.

This decision was made because:
- The dataset spans **Multiple Ultrasound Systems and Multiple Hospital Centers**.
- A 10% labeled subset does not sufficiently cover key variations across devices and institutions.
- Increasing the labeled proportion improves representation while preserving the semi-supervised nature of the challenge.

---
## 🛠️ 3. Image Preprocessing

All images and corresponding segmentation masks have undergone a unified preprocessing pipeline:

- The **longer side** of each image is resized to **512 pixels** while preserving the aspect ratio.
- The resized image is **center-aligned** along the shorter dimension.
- **Zero padding** is applied symmetrically to obtain a fixed-size image.

This process is applied consistently to both images and masks, ensuring pixel-wise alignment.

---

## 📜 4. Watermark and Data Usage Policy

- All ultrasound images contain a **text watermark in the lower-left corner**.
- The watermark does **not interfere with algorithm development or evaluation**.
- **Removing or altering the watermark is not permitted** without authorization.

If watermark-free images are required for specific research purposes, please contact: 📧**Email:** csv2026_challenge@163.com

Please clearly describe the intended purpose and scope of usage.
After approval, de-identified images without watermarks may be provided.

---

## 📝 5. Notes

- This dataset is intended **solely for research and challenge participation**.
- Any use beyond the challenge scope requires prior approval from the organizers.
- Details regarding validation data, test data, and evaluation metrics will be released in later stages.

---

© CSV 2026 Challenge Organizing Committee
