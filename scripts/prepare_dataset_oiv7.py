#!/usr/bin/env python3
"""
Project K — Dataset preparation (v2 — no FiftyOne / no MongoDB)
================================================================
Download Open Images V7 bounding-box annotations, fetch the source
images from AWS S3, crop each detection into a classification patch,
and export a 75/25 train/test directory tree ready for Keras.

Usage
-----
	python scripts/prepare_dataset_oiv7.py --out data --max-samples 3000 --seed 42

Output
------
	data/
	  dataset_info.json          # metadata for reproducibility
	  train/ Apple|Banana|Lemon/
	  test/  Apple|Banana|Lemon/

No FiftyOne or MongoDB required — only uses public CSV metadata from
Google Cloud Storage and downloads images from AWS S3.
"""

import argparse
import csv
import io
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
#  Open Images V7 — public URLs
# ═══════════════════════════════════════════════════════════════

_GCS = "https://storage.googleapis.com/openimages"

CLASS_DESC_URL = f"{_GCS}/v7/oidv7-class-descriptions-boxable.csv"

# Bounding-box annotations (validation + test are small; train is ~2.4 GB)
BBOX_URLS = {
	"validation": f"{_GCS}/v5/validation-annotations-bbox.csv",
	"test":       f"{_GCS}/v5/test-annotations-bbox.csv",
	# v7 train URL returns 403; bbox annotations are identical across v5/v6/v7
	"train":      f"{_GCS}/v6/oidv6-train-annotations-bbox.csv",
}

# AWS S3 mirror for actual image files
IMG_URL_TEMPLATE = "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"

# Target classes (display names used for folder structure)
TARGET_CLASSES = ["Apple", "Banana", "Lemon"]

# OIV7 uses different names for some classes — map them to our names
_OIV7_NAME_MAP = {
	"Apple": "Apple",
	"Banana": "Banana",
	"Lemon (plant)": "Lemon",
}


# ═══════════════════════════════════════════════════════════════
#  Helpers — downloading
# ═══════════════════════════════════════════════════════════════

def _download_file(url, dest, desc=None):
	"""Download *url* to *dest* with a progress bar.  Skips if cached."""
	if os.path.isfile(dest):
		print(f"  [cached] {dest}")
		return
	os.makedirs(os.path.dirname(dest), exist_ok=True)
	desc = desc or os.path.basename(dest)
	print(f"  Downloading {desc} ...")

	# Get file size for progress bar
	req = urllib.request.Request(url, method="HEAD")
	try:
		resp = urllib.request.urlopen(req, timeout=30)
		total = int(resp.headers.get("Content-Length", 0))
	except Exception:
		total = 0

	with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
		def _reporthook(block_num, block_size, _total_size):
			pbar.update(block_size)
		urllib.request.urlretrieve(url, dest, reporthook=_reporthook)


def _download_image(image_id, split, dest_dir, retries=3):
	"""Download a single image from AWS S3.  Returns path or None."""
	url = IMG_URL_TEMPLATE.format(split=split, image_id=image_id)
	dest = os.path.join(dest_dir, f"{image_id}.jpg")
	if os.path.isfile(dest):
		return dest
	for attempt in range(retries):
		try:
			urllib.request.urlretrieve(url, dest)
			return dest
		except (urllib.error.HTTPError, urllib.error.URLError, OSError):
			if attempt < retries - 1:
				time.sleep(0.5 * (attempt + 1))
	return None


# ═══════════════════════════════════════════════════════════════
#  Step 1 — Resolve class names → Machine IDs (MIDs)
# ═══════════════════════════════════════════════════════════════

def load_class_mids(cache_dir):
	"""Return {class_name: mid} for TARGET_CLASSES."""
	csv_path = os.path.join(cache_dir, "class-descriptions-boxable.csv")
	_download_file(CLASS_DESC_URL, csv_path, "class descriptions")

	name_to_mid = {}
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.reader(f)
		for row in reader:
			if len(row) >= 2:
				mid, name = row[0].strip(), row[1].strip()
				if name in _OIV7_NAME_MAP:
					display_name = _OIV7_NAME_MAP[name]
					name_to_mid[display_name] = mid
	# Verify all found
	missing = set(TARGET_CLASSES) - set(name_to_mid.keys())
	if missing:
		raise RuntimeError(f"Could not find MIDs for: {missing}")
	print(f"  Class MIDs: {name_to_mid}")
	return name_to_mid


# ═══════════════════════════════════════════════════════════════
#  Step 2 — Download & filter bbox annotations
# ═══════════════════════════════════════════════════════════════

def load_annotations(cache_dir, target_mids, max_source_images, seed):
	"""
	Download annotation CSVs and filter for our classes.

	Strategy: start with validation + test (small files, ~80 MB total).
	If that yields enough source images, skip the huge train CSV (2.4 GB).

	Returns:
		annotations : dict  {image_id: [(class_name, split, xmin, xmax, ymin, ymax), ...]}
	"""
	mid_to_name = {v: k for k, v in target_mids.items()}
	target_mid_set = set(target_mids.values())

	# Collect all matching annotations
	all_annots = defaultdict(list)  # image_id → [(class, split, bbox...)]

	# Try validation + test first (small downloads)
	for split in ["validation", "test"]:
		csv_path = os.path.join(cache_dir, f"{split}-annotations-bbox.csv")
		_download_file(BBOX_URLS[split], csv_path, f"{split} bbox annotations")

		count = 0
		with open(csv_path, "r", encoding="utf-8") as f:
			reader = csv.reader(f)
			header = next(reader)
			# Find column indices
			idx = {name: i for i, name in enumerate(header)}
			for row in reader:
				label = row[idx["LabelName"]]
				if label in target_mid_set:
					image_id = row[idx["ImageID"]]
					xmin = float(row[idx["XMin"]])
					xmax = float(row[idx["XMax"]])
					ymin = float(row[idx["YMin"]])
					ymax = float(row[idx["YMax"]])
					cls_name = mid_to_name[label]
					all_annots[image_id].append((cls_name, split, xmin, xmax, ymin, ymax))
					count += 1
		print(f"  {split}: {count} bounding boxes for our classes")

	# Check if we have enough
	unique_images = len(all_annots)
	print(f"  Total source images from validation+test: {unique_images}")

	if unique_images < max_source_images:
		print(f"  Need more images — downloading train annotations (large file, ~2.4 GB) ...")
		csv_path = os.path.join(cache_dir, "train-annotations-bbox.csv")

		if os.path.isfile(csv_path):
			print(f"  [cached] {csv_path}")
			# Read from cached file
			count = 0
			with open(csv_path, "r", encoding="utf-8") as f:
				reader = csv.reader(f)
				header = next(reader)
				idx = {name: i for i, name in enumerate(header)}
				for row in reader:
					label = row[idx["LabelName"]]
					if label in target_mid_set:
						image_id = row[idx["ImageID"]]
						xmin = float(row[idx["XMin"]])
						xmax = float(row[idx["XMax"]])
						ymin = float(row[idx["YMin"]])
						ymax = float(row[idx["YMax"]])
						cls_name = mid_to_name[label]
						all_annots[image_id].append((cls_name, "train", xmin, xmax, ymin, ymax))
						count += 1
			print(f"  train: {count} bounding boxes for our classes")
		else:
			# Stream the large CSV — download to file first for caching
			_download_file(BBOX_URLS["train"], csv_path, "train bbox annotations (~2.4 GB)")
			count = 0
			with open(csv_path, "r", encoding="utf-8") as f:
				reader = csv.reader(f)
				header = next(reader)
				idx = {name: i for i, name in enumerate(header)}
				for row in reader:
					label = row[idx["LabelName"]]
					if label in target_mid_set:
						image_id = row[idx["ImageID"]]
						xmin = float(row[idx["XMin"]])
						xmax = float(row[idx["XMax"]])
						ymin = float(row[idx["YMin"]])
						ymax = float(row[idx["YMax"]])
						cls_name = mid_to_name[label]
						all_annots[image_id].append((cls_name, "train", xmin, xmax, ymin, ymax))
						count += 1
			print(f"  train: {count} bounding boxes for our classes")

	# Sample source images
	all_image_ids = list(all_annots.keys())
	rng = random.Random(seed)
	rng.shuffle(all_image_ids)

	if len(all_image_ids) > max_source_images:
		all_image_ids = all_image_ids[:max_source_images]
		print(f"  Sampled {max_source_images} source images (seed={seed})")
	else:
		print(f"  Using all {len(all_image_ids)} source images")

	# Filter annotations to sampled images only
	filtered = {img_id: all_annots[img_id] for img_id in all_image_ids}
	total_boxes = sum(len(v) for v in filtered.values())
	print(f"  Total bounding boxes to crop: {total_boxes}")

	return filtered


# ═══════════════════════════════════════════════════════════════
#  Step 3 — Download images & crop bounding boxes
# ═══════════════════════════════════════════════════════════════

def download_and_crop(annotations, cache_dir, workers=8):
	"""
	Download source images and crop each bounding box.

	Returns:
		patches : list of (class_name, PIL.Image)
	"""
	img_cache = os.path.join(cache_dir, "images")
	os.makedirs(img_cache, exist_ok=True)

	patches = []
	failed = 0

	# Group by split for correct download URL
	image_splits = {}  # image_id → split
	for img_id, annots in annotations.items():
		# All annots for an image share the same split
		image_splits[img_id] = annots[0][1]

	# Download images in parallel
	print(f"\n  Downloading {len(annotations)} source images ...")
	image_paths = {}

	with ThreadPoolExecutor(max_workers=workers) as pool:
		futures = {
			pool.submit(_download_image, img_id, split, img_cache): img_id
			for img_id, split in image_splits.items()
		}
		for future in tqdm(as_completed(futures), total=len(futures), desc="  Images"):
			img_id = futures[future]
			path = future.result()
			if path:
				image_paths[img_id] = path
			else:
				failed += 1

	if failed:
		print(f"  Warning: {failed} images could not be downloaded (404 / removed from Flickr)")

	# Crop bounding boxes
	print(f"\n  Cropping bounding boxes ...")
	for img_id, annots in tqdm(annotations.items(), desc="  Cropping"):
		if img_id not in image_paths:
			continue
		try:
			img = Image.open(image_paths[img_id]).convert("RGB")
		except Exception:
			continue
		w, h = img.size
		for cls_name, _split, xmin, xmax, ymin, ymax in annots:
			# Convert normalised coords to pixels
			left = int(xmin * w)
			upper = int(ymin * h)
			right = int(xmax * w)
			lower = int(ymax * h)
			# Sanity check
			if right <= left or lower <= upper:
				continue
			crop = img.crop((left, upper, right, lower))
			# Skip tiny crops
			if crop.width < 10 or crop.height < 10:
				continue
			patches.append((cls_name, crop))

	return patches


# ═══════════════════════════════════════════════════════════════
#  Step 4 — Split & export
# ═══════════════════════════════════════════════════════════════

def export_patches(patches, out_dir, test_frac, seed):
	"""Shuffle, split, and save patches to directory tree."""
	rng = random.Random(seed)
	rng.shuffle(patches)

	split_idx = int(len(patches) * (1 - test_frac))
	train_patches = patches[:split_idx]
	test_patches = patches[split_idx:]

	# Create directories
	for split_name in ["train", "test"]:
		for cls in TARGET_CLASSES:
			os.makedirs(os.path.join(out_dir, split_name, cls), exist_ok=True)

	# Counters per class
	counters = defaultdict(int)

	def _save(patch_list, split_name):
		for cls_name, crop in tqdm(patch_list, desc=f"  Saving {split_name}"):
			counters[(split_name, cls_name)] += 1
			idx = counters[(split_name, cls_name)]
			fname = f"{cls_name.lower()}_{idx:05d}.jpg"
			dest = os.path.join(out_dir, split_name, cls_name, fname)
			crop.save(dest, "JPEG", quality=95)

	_save(train_patches, "train")
	_save(test_patches, "test")

	return train_patches, test_patches, counters


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
	parser = argparse.ArgumentParser(
		description="Download OIV7 fruit patches (Apple/Banana/Lemon)"
	)
	parser.add_argument("--out", default="data",
		help="Output directory (default: data)")
	parser.add_argument("--max-samples", type=int, default=3000,
		help="Max source images to download (default: 3000)")
	parser.add_argument("--seed", type=int, default=42,
		help="Random seed for split & sampling (default: 42)")
	parser.add_argument("--test-frac", type=float, default=0.25,
		help="Fraction for test split (default: 0.25)")
	parser.add_argument("--workers", type=int, default=8,
		help="Parallel image download threads (default: 8)")
	args = parser.parse_args()

	out_dir = args.out
	cache_dir = os.path.join(out_dir, ".cache")
	os.makedirs(cache_dir, exist_ok=True)

	print("=" * 60)
	print("  Project K — OIV7 Dataset Preparation (v2)")
	print("=" * 60)

	# ── Step 1: Resolve class MIDs ───────────────────────────
	print("\n[1/4] Resolving class names → MIDs ...")
	name_to_mid = load_class_mids(cache_dir)

	# ── Step 2: Load & filter annotations ────────────────────
	print("\n[2/4] Loading bounding-box annotations ...")
	annotations = load_annotations(cache_dir, name_to_mid, args.max_samples, args.seed)

	# ── Step 3: Download images & crop ───────────────────────
	print("\n[3/4] Downloading images and cropping patches ...")
	patches = download_and_crop(annotations, cache_dir, workers=args.workers)
	print(f"\n  Total patches: {len(patches)}")

	# Count per class
	from collections import Counter
	cls_counts = Counter(cls for cls, _ in patches)
	for cls in TARGET_CLASSES:
		print(f"    {cls:>8s}: {cls_counts.get(cls, 0)}")

	# ── Step 4: Split & export ───────────────────────────────
	print(f"\n[4/4] Splitting {args.test_frac:.0%} test and exporting ...")

	# Clean existing output (but not cache)
	for split_name in ["train", "test"]:
		split_dir = os.path.join(out_dir, split_name)
		if os.path.isdir(split_dir):
			shutil.rmtree(split_dir)

	train_patches, test_patches, counters = export_patches(
		patches, out_dir, args.test_frac, args.seed
	)

	# ── Save metadata ────────────────────────────────────────
	train_counts = {cls: counters.get(("train", cls), 0) for cls in TARGET_CLASSES}
	test_counts = {cls: counters.get(("test", cls), 0) for cls in TARGET_CLASSES}

	meta = {
		"source": "open-images-v7 (detections → patches, direct download)",
		"created_at": datetime.now().isoformat(timespec="seconds"),
		"seed": args.seed,
		"max_source_images": args.max_samples,
		"test_frac": args.test_frac,
		"num_patches_total": len(patches),
		"num_train": len(train_patches),
		"num_test": len(test_patches),
		"train_counts": train_counts,
		"test_counts": test_counts,
		"classes": TARGET_CLASSES,
	}
	info_path = os.path.join(out_dir, "dataset_info.json")
	with open(info_path, "w") as f:
		json.dump(meta, f, indent=2)

	# ── Summary ──────────────────────────────────────────────
	print()
	print("=" * 60)
	print("  Done. Dataset is ready.")
	print("=" * 60)
	print(f"  Output dir  : {out_dir}/")
	print(f"  Total patches: {len(patches)}")
	print(f"  Train        : {len(train_patches)}")
	print(f"  Test         : {len(test_patches)}")
	print(f"  Split        : {1 - args.test_frac:.0%} / {args.test_frac:.0%}")
	print()
	print("  Per-class breakdown:")
	print(f"  {'Class':>10s}  {'Train':>6s}  {'Test':>6s}  {'Total':>6s}")
	print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}")
	for cls in TARGET_CLASSES:
		tr = train_counts[cls]
		te = test_counts[cls]
		print(f"  {cls:>10s}  {tr:>6d}  {te:>6d}  {tr+te:>6d}")
	print()
	print(f"  Metadata saved to: {info_path}")
	print(f"  Cache dir (reusable): {cache_dir}/")
	print()


if __name__ == "__main__":
	main()
