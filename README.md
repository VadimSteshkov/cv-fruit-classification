# CV Fruit Classification

Computer Vision project: Image classification of Apple, Banana and Lemon using Open Images Dataset and VGG16.

## Project Structure

data/
    raw/        # downloaded dataset
    train/      # 75% split
    test/       # 25% split
src/            # training scripts
models/         # saved models

## Setup

1. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Download dataset:
   oidv6 downloader --dataset data\raw --type_data train --classes Apple Banana Lemon --limit 1000 --yes

4. Split dataset:
   python src\01_split_train_test.py

## Model

- Base architecture: VGG16
- Experiments:
  - Training from scratch
  - Transfer learning
  - Data augmentation
  - Confusion matrix evaluation