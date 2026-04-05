# CV Fruit Classification — Project K

This repository contains the deliverables for **Project K** of the **Computer Vision in AI** course.  
The project investigates **image classification** for three fruit classes — **Apple**, **Banana**, and **Lemon** — using images derived from the **Open Images Dataset V7** and implemented with **TensorFlow / Keras**.

| Item | Detail |
|------|--------|
| **Task** | Multi-class image classification |
| **Classes** | Apple · Banana · Lemon |
| **Dataset source** | Open Images Dataset V7 (detection task, converted to classification patches) |
| **Primary frameworks** | TensorFlow / Keras |
| **Backbone families** | VGG-16, VGG-19 |
| **Dataset split** | 75% train / 25% test |
| **Main deliverable** | `project_k.ipynb` |

---

## Project objective

The objective of this project is to compare several convolutional-network configurations for fruit classification under a reproducible experimental setup.  
The work includes:

- dataset preparation and exploratory analysis,
- a baseline VGG-16 model trained from scratch,
- transfer learning with ImageNet-pretrained VGG-16,
- a data-augmentation experiment,
- a custom VGG-19 architecture with an added bottleneck block,
- final evaluation with own images, activation maps, confusion matrices, and a consolidated comparison table.

---

## Main deliverable

The complete submission is contained in:

- **`project_k.ipynb`** — single end-to-end notebook containing data exploration, all experiments, training/evaluation logic, visualizations, and final analysis.

The notebook is designed so that the project can be reproduced from one main file, provided that the required environment and dataset are available.

---

## Experimental scope

The notebook covers the following milestones.

| Milestone | Title | Description |
|-----------|-------|-------------|
| **C1** | Foundation | Repository setup, reproducibility preparation, dataset pipeline, and exploratory data analysis |
| **C2** | Baseline | VGG-16 trained from scratch, including training and evaluation |
| **C3** | Transfer learning | ImageNet-pretrained VGG-16 and comparison against the baseline over the first 10 epochs |
| **C4** | Augmentation | Retraining with `RandomRotation`, `RandomTranslation`, and `RandomCrop` |
| **C5** | Architecture | Custom VGG-19 configuration with a bottleneck block after `block4_conv4` |
| **C6** | Final evaluation | Own-image predictions, activation maps, confusion matrices, infrastructure summary, and final comparison |

### C5 architecture specification

The custom architecture experiment follows the project specification by rebuilding the network from **VGG-19** after `block4_conv4`, then adding:

1. a **bottleneck layer** with `padding="same"`,
2. a **1×1 convolution** with **1024 filters**, `padding="valid"`, stride 1, followed by **LeakyReLU**,
3. a **3×3 convolution** with **1024 filters**, `padding="same"`, stride 1, followed by **ReLU**,
4. frozen convolutional layers in **conv3 and earlier**,
5. a classification head using **Flatten + fully connected layers + Softmax**.

---

## Required analysis covered in the notebook

The final notebook includes the analyses required by the project brief:

- dataset exploration and visual inspection,
- class distribution and basic image statistics,
- comparison of training, validation, and test behaviour,
- test-set evaluation for all experiments,
- confusion matrices and the most frequently confused class pairs,
- number of trainable / total parameters,
- inference-time measurement,
- own-image evaluation,
- activation maps (Grad-CAM),
- final experiment comparison table.

---

## Repository structure

```text
cv-fruit-classification/
├── project_k.ipynb                  # main project notebook
├── README.md                        # project overview and usage instructions
├── QUICKSTART.txt                   # short setup guide for teammates
├── requirements.txt                 # main dependency list
├── requirements.lock.txt            # pinned dependency snapshot
├── .gitignore
├── scripts/
│   ├── prepare_dataset_oiv7.py      # dataset preparation script
│   └── verify_tf.py                 # TensorFlow / GPU verification script
├── reports/                         # generated plots, CSV summaries, and result artefacts
├── own_images/                      # optional folder for external test images
├── src/                             # optional helper modules
└── data/                            # generated dataset (git-ignored)
    ├── train/
    │   ├── Apple/
    │   ├── Banana/
    │   └── Lemon/
    └── test/
        ├── Apple/
        ├── Banana/
        └── Lemon/
```

---

## Setup and execution

### 1. Clone the repository

```bash
git clone <repository-url>
cd cv-fruit-classification
```

### 2. Create and activate a virtual environment

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If needed, a locked snapshot of the installed environment can be regenerated with:

```bash
pip freeze > requirements.lock.txt
```

### 4. Verify TensorFlow availability

```bash
python scripts/verify_tf.py
```

### 5. Prepare the dataset

```bash
python scripts/prepare_dataset_oiv7.py --out data --max-samples 3000 --seed 42
```

This prepares the classification dataset in:

- `data/train/{Apple,Banana,Lemon}/`
- `data/test/{Apple,Banana,Lemon}/`

### 6. Run the notebook

```bash
jupyter notebook project_k.ipynb
```

Then execute:

- **Kernel → Restart & Run All**

---

## Reproducibility

The repository is organised to support reproducible execution:

- deterministic dataset preparation with fixed seed,
- fixed random seeds in model training,
- explicit dependency lists,
- single-notebook execution for the full project workflow,
- generated reports saved to the `reports/` directory,
- dataset, virtual environment, and large local artefacts excluded from version control.

---

## Team

| Member | Branch |
|--------|--------|
| **Dorin-Emilian Avram** | `dev/avd` |
| **Vadim Steshkov** | `dev/vad` |
| **Angelo Ottendorfer** | `dev/ang` |

---

## Notes

- The `data/`, `venv/`, and model artefacts are not intended for version control.
- The `own_images/` folder is used for qualitative evaluation in the final milestone and may remain empty in the repository except for a placeholder file.
- Final plots and CSV summaries are generated automatically by the notebook and stored in `reports/`.
