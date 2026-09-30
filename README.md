# Breast Ultrasound Classification: Transfer-Learning Feature Extraction with Grad-CAM & LIME

Benchmarks six ImageNet-pretrained CNNs as **frozen feature extractors** for classifying breast ultrasound images as **benign** or **malignant** (BUSI dataset). Features are classified with Logistic Regression under stratified 5-fold cross-validation, and the pipeline includes explainability tooling (feature-map visualization, Grad-CAM, LIME).

## Overview

1. Load and resize images to 224x224 RGB.
2. Extract global-average-pooled features from each pretrained backbone (no fine-tuning).
3. Standardize features and train Logistic Regression per fold.
4. Report F1, ROC-AUC, MCC and Cohen's Kappa (mean, std, 95% normal-approximation CI across folds).
5. Generate interpretability visualizations for a sample benign and malignant image.

## Models

| Model | Feature dim |
|---|---|
| VGG16 | 512 |
| ResNet50 | 2048 |
| MobileNetV2 | 1280 |
| InceptionResNetV2 | 1536 |
| NASNetMobile | 1056 |
| Xception | 2048 |

## Dataset

[Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset), `Dataset_BUSI_with_GT`.

Expected structure:

```
Dataset_BUSI_with_GT/
├── benign/
├── malignant/
└── normal/      # ignored by this script
```

Only `benign` (label 0) and `malignant` (label 1) are used.

## Requirements

- Python 3.9+
- tensorflow / keras
- numpy, opencv-python, pillow, matplotlib
- scikit-learn, scikit-image
- lime

```bash
pip install tensorflow numpy opencv-python pillow matplotlib scikit-learn scikit-image lime
```

## Usage

1. Set `data_path` in the `__main__` block to your local dataset folder.
2. Run:

```bash
python main.py
```

The script will extract features, run 5-fold CV for every model, then attempt feature-map, Grad-CAM and LIME visualizations.

## Results (5-fold CV, mean with 95% CI)

| Model | F1 | AUC | MCC | Kappa |
|---|---|---|---|---|
| VGG16 | 0.8659 [0.8515, 0.8803] | 0.9615 [0.9478, 0.9752] | 0.8026 [0.7817, 0.8235] | 0.8025 [0.7815, 0.8235] |
| ResNet50 | **0.8835** [0.8601, 0.9068] | **0.9709** [0.9593, 0.9824] | **0.8323** [0.7998, 0.8648] | **0.8311** [0.7978, 0.8643] |
| MobileNetV2 | 0.8654 [0.8363, 0.8944] | 0.9654 [0.9524, 0.9785] | 0.8057 [0.7659, 0.8454] | 0.8046 [0.7639, 0.8452] |
| InceptionResNetV2 | 0.8766 [0.8413, 0.9119] | 0.9636 [0.9502, 0.9770] | 0.8186 [0.7663, 0.8709] | 0.8182 [0.7656, 0.8708] |
| NASNetMobile | 0.8272 [0.8053, 0.8491] | 0.9408 [0.9324, 0.9492] | 0.7471 [0.7162, 0.7781] | 0.7468 [0.7157, 0.7780] |
| Xception | 0.8766 [0.8438, 0.9093] | 0.9692 [0.9585, 0.9799] | 0.8217 [0.7756, 0.8678] | 0.8207 [0.7741, 0.8674] |

ResNet50 scored best on all four metrics; NASNetMobile scored lowest.

> **Caution:** these numbers were produced with the data issue described below (mask images included). Re-run after fixing before reporting them.

## Known Issues & Fixes

**1. Segmentation masks are loaded as images (affects results).**
BUSI stores `*_mask.png` files in the same class folders. The loader picks them up, giving 1312 "images" (891 benign / 421 malignant) instead of the ~647 real ones. Masks of the same scan can land in different folds (data leakage) and are trivially separable, which inflates metrics. Fix in `load_images_pil`:

```python
if not file.lower().endswith(('.png', '.jpg', '.jpeg')) or '_mask' in file.lower():
    continue
```

**2. "No convolutional layers found" (feature maps and Grad-CAM).**
`layer.output_shape` is not available in Keras 3. Use `layer.output.shape` instead:

```python
def is_conv_layer(layer):
    try:
        return len(layer.output.shape) == 4
    except Exception:
        return False
```

**3. InceptionResNetV2 / Xception `include_top=True` error.**
These models require 299x299 input when loading full ImageNet weights with the classifier head. Resize to 299x299 for Grad-CAM/LIME with these two models.

**4. Grad-CAM/LIME use ImageNet classifiers, not the trained benign/malignant classifier.**
The heatmaps currently explain the top *ImageNet* class, which is not meaningful for tumor classification. For clinically relevant explanations, fine-tune a model (or attach a binary head) and run Grad-CAM/LIME on that.

## Project Structure

```
.
├── main.py      # full pipeline: loading, feature extraction, CV, XAI
└── README.md
```

## Notes

- Random seed is fixed (`42`) for shuffling and CV splits.
- Confidence intervals use a normal approximation over 5 folds, so treat them as rough estimates.
- Feature extraction on CPU is slow for VGG16 (~4.5 min for the dataset); a GPU is recommended.

## License

Add your license here (e.g., MIT). Dataset usage is subject to BUSI's own terms.
