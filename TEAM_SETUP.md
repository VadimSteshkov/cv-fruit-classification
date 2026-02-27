# 🍎🍌🍋 TEAM SETUP GUIDE  
CV Fruit Classification Project

This document explains how to properly set up the project locally.

---

## 1. Clone the repository

git clone https://github.com/VadimSteshkov/cv-fruit-classification.git
cd cv-fruit-classification

---

## 2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

Mac / Linux:

python3 -m venv venv
source venv/bin/activate

After activation, you should see (venv) in your terminal.

---

## 3. Install project dependencies

pip install -r requirements.txt

---

## 4. Download the dataset (Open Images V6)

Download training images:

oidv6 downloader --dataset data/raw --type_data train --classes Apple Banana Lemon

Optional validation set:

oidv6 downloader --dataset data/raw --type_data validation --classes Apple Banana Lemon

Note: Download may take some time.

---

## 5. Verify PyTorch installation

python src/test_torch.py

Expected output example:

Torch version: 2.x.x
CUDA available: False

If CUDA is False, training will run on CPU.

---

## Expected Project Structure

```text
cv-fruit-classification/
│
├── data/
│   ├── raw/
│   ├── train/
│   │   ├── Apple/
│   │   ├── Banana/
│   │   └── Lemon/
│   └── test/
│       ├── Apple/
│       ├── Banana/
│       └── Lemon/
│
├── src/
├── models/
├── notebooks/
├── requirements.txt
└── TEAM_SETUP.md
```
---

## Important rules

- Do NOT push the data/ folder
- Do NOT push the venv/ folder
- Do NOT push model weights (.pth, .pt)
- Always activate the virtual environment before working
- Always pull latest changes before starting work

---

## If something does not work

1. Make sure Python version is 3.10–3.11
2. Recreate the virtual environment
3. Reinstall requirements
4. Ensure venv is activated