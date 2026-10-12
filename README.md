# QuIIL-DeepClue · AIMS-TBI '26

Official PyTorch implementation of Team **QuIIL-DeepClue** for the **AIMS-TBI '26 Challenge** (MICCAI 2026 Workshop).

📄 **Paper:** [Rebalancing Lesion Exposure: Instance-Aware Sampling for msTBI Lesion Segmentation](https://papers.miccai.org/miccai-2026-sat/AIMS_TBI_004.html)

<p align="center">
  <img src="assets/main_fig.jpg" width="80%" alt="Overview of the instance-aware sampling pipeline">
</p>
<p align="center">
  <img src="assets/distribution.jpg" width="80%" alt="Lesion exposure distribution">
</p>
<p align="center">
  <img src="assets/res.png" width="80%" alt="Results">
</p>

## Contents

- [Challenge overview](#challenge-overview)
- [Method](#method)
- [Installation](#installation)
- [Usage](#usage)
  - [0. Environment variables](#0-environment-variables)
  - [1. Dataset preparation](#1-dataset-preparation)
  - [2. Planning and preprocessing](#2-planning-and-preprocessing)
  - [3. Training](#3-training)
  - [4. Inference](#4-inference)
  - [5. Evaluation](#5-evaluation)
- [Acknowledgements](#acknowledgements)

## Challenge overview

[AIMS-TBI '26](https://aims-tbi26.grand-challenge.org/) (**A**utomated **I**dentification of **M**oderate-to-**S**evere **TBI** Lesions) targets the detection and segmentation of lesions in T1-weighted MRI from patients with moderate-to-severe traumatic brain injury (msTBI). Target lesions include contusions, hemorrhage, hematoma, encephalomalacia, gliosis, white matter lesions, and surgical drainage tracts.

| | |
| --- | --- |
| **Input** | Native-space T1w MRI, plus tabular demographic and clinical data |
| **Output** | Binary TBI lesion segmentation mask |

## Method

msTBI lesions vary widely in number and size, so standard foreground sampling over-exposes large lesions and rarely shows small ones to the network. We rebalance this with **instance-aware lesion sampling**: lesion instance locations are stored during preprocessing, and training patches are drawn at the instance level so that every lesion receives comparable exposure.

| Component | Setting |
| --- | --- |
| Framework | nnU-Net v2 (modified) |
| Architecture | Residual Encoder U-Net (ResEnc M) |
| Configuration | `3d_fullres` |
| Patch size | 128 × 128 × 128 |
| Batch size | 8 |
| Sampling | Instance-aware lesion sampling |
| Loss | 0.3 · CE + 0.4 · Dice + 0.3 · Focal |
| Inference | 5-fold ensemble |

| Identifier | Name |
| --- | --- |
| Trainer | `nnUNetTrainerInstanceAwareSampling_03CE04Dice03FocalLoss` |
| Plans | `nnUNetResEncUNetMpatch128_InstanceSampling_Batch8_Plans` |
| Preprocessor | `wInstanceLocationPreprocessor` |

## Installation

Create a Python environment with the nnU-Net v2 dependencies (PyTorch etc.), then install the modified nnU-Net bundled in this repository in editable mode:

```bash
git clone <this-repo-url>
cd AIMSTBI26_QuIIL_DeepClue
pip install -e nnUNet
```

> **Note:** Use the bundled `nnUNet` rather than the upstream package — the custom trainer, sampler, and preprocessor live there.

## Usage

### 0. Environment variables

```bash
export nnUNet_raw="DATA/nnUNet_raw"
export nnUNet_preprocessed="DATA/nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"

# Avoid CPU oversubscription in data-loading workers
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### 1. Dataset preparation

Arrange the challenge data under `$nnUNet_raw` in the nnU-Net v2 raw format:

```
DATA/nnUNet_raw/Dataset<DATASET_ID>_AIMSTBI/
├── dataset.json
├── imagesTr/
│   ├── <case_id>_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── <case_id>.nii.gz
    └── ...
```

### 2. Planning and preprocessing

**(a) Fingerprint and baseline plans.** Extract the dataset fingerprint and generate the ResEnc M plans:

```bash
nnUNetv2_extract_fingerprint -d <DATASET_ID>
nnUNetv2_plan_experiment -d <DATASET_ID> -pl nnUNetPlannerResEncM
```

**(b) Create the custom plans.** Copy `nnUNetResEncUNetMPlans.json` (in `$nnUNet_preprocessed/Dataset<DATASET_ID>_AIMSTBI/`) to `nnUNetResEncUNetMpatch128_InstanceSampling_Batch8_Plans.json` and set the following fields:

| Field | Value |
| --- | --- |
| `plans_name` | `nnUNetResEncUNetMpatch128_InstanceSampling_Batch8_Plans` |
| `configurations.3d_fullres.data_identifier` | `nnUNet_wInstanceLocationPlans_3d_fullres` |
| `configurations.3d_fullres.preprocessor_name` | `wInstanceLocationPreprocessor` |
| `configurations.3d_fullres.batch_size` | `8` |
| `configurations.3d_fullres.patch_size` | `[128, 128, 128]` |

<details>
<summary>Or apply the edits with a script</summary>

```bash
python - <<'EOF'
import json, os
d = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset<DATASET_ID>_AIMSTBI")
name = "nnUNetResEncUNetMpatch128_InstanceSampling_Batch8_Plans"
with open(os.path.join(d, "nnUNetResEncUNetMPlans.json")) as f:
    plans = json.load(f)
plans["plans_name"] = name
cfg = plans["configurations"]["3d_fullres"]
cfg["data_identifier"] = "nnUNet_wInstanceLocationPlans_3d_fullres"
cfg["preprocessor_name"] = "wInstanceLocationPreprocessor"
cfg["batch_size"] = 8
cfg["patch_size"] = [128, 128, 128]
with open(os.path.join(d, f"{name}.json"), "w") as f:
    json.dump(plans, f, indent=4)
EOF
```

</details>

The custom preprocessor stores lesion instance locations alongside the preprocessed volumes; the instance-aware sampler reads them during training.

**(c) Preprocess** with the custom plans:

```bash
nnUNetv2_preprocess -d <DATASET_ID> \
  -plans_name nnUNetResEncUNetMpatch128_InstanceSampling_Batch8_Plans \
  -c 3d_fullres
```

### 3. Training

Train all five folds:

```bash
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train <DATASET_ID> 3d_fullres $FOLD \
    -p nnUNetResEncUNetMpatch128_InstanceSampling_Batch8_Plans \
    -tr nnUNetTrainerInstanceAwareSampling_03CE04Dice03FocalLoss
done
```

To log to Weights & Biases, prefix the command with `nnUNet_wandb_enabled=1`:

```bash
nnUNet_wandb_enabled=1 nnUNetv2_train <DATASET_ID> 3d_fullres 0 \
  -p nnUNetResEncUNetMpatch128_InstanceSampling_Batch8_Plans \
  -tr nnUNetTrainerInstanceAwareSampling_03CE04Dice03FocalLoss
```

### 4. Inference

**5-fold ensemble (final submission):**

```bash
nnUNetv2_predict \
  -d <DATASET_ID> \
  -i <input_images_dir> \
  -o <ensemble_pred_dir> \
  -f 0 1 2 3 4 \
  -c 3d_fullres \
  -p nnUNetResEncUNetMpatch128_InstanceSampling_Batch8_Plans \
  -tr nnUNetTrainerInstanceAwareSampling_03CE04Dice03FocalLoss
```

**Single fold** (e.g. for per-fold analysis; `--save_probabilities` also writes softmax outputs):

```bash
nnUNetv2_predict \
  -d <DATASET_ID> \
  -i <input_images_dir> \
  -o <fold_pred_dir> \
  -f <FOLD> \
  -c 3d_fullres \
  -p nnUNetResEncUNetMpatch128_InstanceSampling_Batch8_Plans \
  -tr nnUNetTrainerInstanceAwareSampling_03CE04Dice03FocalLoss \
  --save_probabilities
```

### 5. Evaluation

Compute case-level Dice, HD95, and ASSD against the ground truth:

```bash
python test.py \
  --pred_dir <pred_dir> \
  --gt_dir DATA/nnUNet_raw/Dataset<DATASET_ID>_AIMSTBI/labelsTr
```

## Acknowledgements

This work builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) by MIC-DKFZ. Data are provided by the [AIMS-TBI challenge](https://github.com/adionicas/AIMS-TBI-challenge) organizers through the ENIGMA Consortium TBI Working Group.

## Citations

``` 
@InProceedings{NohSeu_Rebalancing_MICCAISAT2026,
  author = {Noh, Seungha AND Yu, Cheolhee AND Yang, Hyun AND Kwak, Jin Tae},
  title = {Rebalancing Lesion Exposure: Instance-Aware Sampling for msTBI Lesion Segmentation},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Workshops and Challenges},
  year = {2026},
  publisher = {Springer Nature Switzerland},
  volume = {LNCS 17254},
  month = {pending},
  page = {pending}
}

```