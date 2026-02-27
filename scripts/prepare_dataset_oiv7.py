#!/usr/bin/env python3
"""
Project K — Dataset preparation
================================
Download Open Images V7 detection labels via FiftyOne, crop each annotated
object into a classification patch, and export a 75/25 train/test directory
tree ready for Keras ImageDataGenerator / image_dataset_from_directory.

Usage
-----
    python scripts/prepare_dataset_oiv7.py --out data --max-samples 3000 --seed 42

Output
------
    data/
      dataset_info.json          # metadata for reproducibility
      train/ Apple|Banana|Lemon/
      test/  Apple|Banana|Lemon/
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import fiftyone as fo
import fiftyone.zoo as foz


# ── Project K required classes ───────────────────────────────────────────────
CLASSES = ["Apple", "Banana", "Lemon"]
CLS_FIELD = "classification"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_empty_dir(path: str) -> None:
	"""Remove *path* if it exists and recreate it empty."""
	if os.path.isdir(path):
		shutil.rmtree(path)
	os.makedirs(path, exist_ok=True)


def _safe_delete_dataset(name: str) -> None:
	"""Delete a FiftyOne dataset by *name* if it exists (ignore errors)."""
	if fo.dataset_exists(name):
		try:
			fo.delete_dataset(name)
		except Exception as exc:
			print(f"[WARN] Could not delete dataset '{name}': {exc}")


def _detect_detections_field(dataset: fo.Dataset) -> str:
	"""Auto-detect the Detections field in *dataset* schema."""
	schema = dataset.get_field_schema()

	for fname, field in schema.items():
		try:
			if (
				isinstance(field, fo.EmbeddedDocumentField)
				and field.document_type == fo.Detections
			):
				return fname
		except Exception:
			pass

	# Common fallback name used by the OIV7 zoo dataset
	if "ground_truth" in schema:
		return "ground_truth"

	raise ValueError(
		"Could not auto-detect detections field. "
		f"Available fields: {', '.join(sorted(schema.keys()))}"
	)


def _materialise_view(view, name: str):
	"""
	Convert a DatasetView / PatchesView into a persistent FiftyOne Dataset.
	Tries multiple strategies to stay compatible across FiftyOne versions.
	"""
	_safe_delete_dataset(name)

	# Strategy 1: to_dataset (FiftyOne >= 0.23)
	if hasattr(view, "to_dataset"):
		try:
			return view.to_dataset(name=name)
		except Exception as exc:
			print(f"[WARN] view.to_dataset failed: {exc}")

	# Strategy 2: clone
	if hasattr(view, "clone"):
		try:
			return view.clone(name=name)
		except TypeError:
			return view.clone(name)

	# Strategy 3: from_view (older versions)
	if hasattr(fo.Dataset, "from_view"):
		try:
			return fo.Dataset.from_view(view, name=name)
		except Exception as exc:
			print(f"[WARN] fo.Dataset.from_view failed: {exc}")

	raise RuntimeError(
		"Could not materialise patches view into a Dataset — "
		"check your FiftyOne version."
	)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Project K dataset prep: "
			"Open Images V7 detections → patches → classification directory tree"
		),
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--out", type=str, default="data",
		help="Root output directory (creates train/ and test/ inside)",
	)
	parser.add_argument(
		"--dataset-name", type=str, default="cvai_project_k_oiv7",
		help="FiftyOne dataset name (used for caching)",
	)
	parser.add_argument(
		"--max-samples", type=int, default=3000,
		help="Max images to fetch from OIV7 (more images → more patches)",
	)
	parser.add_argument(
		"--seed", type=int, default=42,
		help="Random seed for shuffle and split",
	)
	parser.add_argument(
		"--test-frac", type=float, default=0.25,
		help="Fraction of patches for the test set (Project K requires 0.25)",
	)
	parser.add_argument(
		"--det-field", type=str, default="",
		help="Detections field name (auto-detect if empty)",
	)
	parser.add_argument(
		"--keep-existing-output", action="store_true",
		help="Do not wipe existing train/test directories before export",
	)
	return parser.parse_args()


# ── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
	args = parse_args()

	if not (0.0 < args.test_frac < 1.0):
		raise ValueError("--test-frac must be in (0, 1)")

	out_train = os.path.join(args.out, "train")
	out_test = os.path.join(args.out, "test")

	# Prepare output directories
	if args.keep_existing_output:
		os.makedirs(out_train, exist_ok=True)
		os.makedirs(out_test, exist_ok=True)
	else:
		_ensure_empty_dir(out_train)
		_ensure_empty_dir(out_test)

	# Clean up previous FiftyOne datasets to ensure idempotent re-runs
	_safe_delete_dataset(args.dataset_name)
	patches_ds_name = f"{args.dataset_name}_patches"
	_safe_delete_dataset(patches_ds_name)

	# ── Step 1: Download OIV7 detection samples via FiftyOne Zoo ─────────
	print("=" * 60)
	print("Project K — Dataset Preparation")
	print("=" * 60)
	print(f"  Classes      : {CLASSES}")
	print(f"  Max samples  : {args.max_samples}")
	print(f"  Seed         : {args.seed}")
	print(f"  Split        : train {1.0 - args.test_frac:.0%} / test {args.test_frac:.0%}")
	print(f"  Output       : {os.path.abspath(args.out)}")
	print()

	ds = foz.load_zoo_dataset(
		"open-images-v7",
		split="train",
		label_types=["detections"],
		classes=CLASSES,
		max_samples=args.max_samples,
		dataset_name=args.dataset_name,
	)
	print(f"[INFO] Base dataset loaded: {ds.name}  ({len(ds)} images)")

	# ── Step 2: Detect the field that holds bounding-box annotations ─────
	det_field = args.det_field.strip() or _detect_detections_field(ds)
	print(f"[INFO] Using detections field: '{det_field}'")

	# ── Step 3: Convert detections → patches (one patch = one crop) ──────
	patches_view = ds.to_patches(det_field)
	patches = _materialise_view(patches_view, patches_ds_name)
	print(f"[INFO] Patches dataset created: {patches.name}  ({len(patches)} patches)")

	# ── Step 4: Build a classification label from each patch's Detection ─
	labels = []
	for sample in patches.iter_samples(progress=True):
		# Use getattr instead of .get() for compatibility with FiftyOne >= 1.x
		det = getattr(sample, det_field, None)
		lbl = det.label if det is not None else None
		labels.append(fo.Classification(label=lbl))

	schema = patches.get_field_schema()
	if CLS_FIELD not in schema:
		patches.add_sample_field(
			CLS_FIELD,
			fo.EmbeddedDocumentField,
			embedded_doc_type=fo.Classification,
		)
	patches.set_values(CLS_FIELD, labels)

	# ── Step 5: Shuffle + 75/25 split ────────────────────────────────────
	patches_shuf = patches.shuffle(seed=args.seed)
	n_total = len(patches_shuf)
	n_test = int(round(n_total * args.test_frac))
	n_test = max(1, min(n_test, n_total - 1))

	test_view = patches_shuf.take(n_test)
	train_view = patches_shuf.skip(n_test)
	print(f"[INFO] Split: train={len(train_view)}, test={len(test_view)}")

	# ── Step 6: Export to ImageClassificationDirectoryTree ────────────────
	print(f"[INFO] Exporting train → {out_train}")
	train_view.export(
		export_dir=out_train,
		dataset_type=fo.types.ImageClassificationDirectoryTree,
		label_field=CLS_FIELD,
		overwrite=True,
	)

	print(f"[INFO] Exporting test  → {out_test}")
	test_view.export(
		export_dir=out_test,
		dataset_type=fo.types.ImageClassificationDirectoryTree,
		label_field=CLS_FIELD,
		overwrite=True,
	)

	# ── Step 7: Save dataset metadata for reproducibility ────────────────
	class_counts = patches.count_values(f"{CLS_FIELD}.label")
	info = {
		"created_at": datetime.now().isoformat(timespec="seconds"),
		"source": "open-images-v7 (detections → patches)",
		"classes": CLASSES,
		"max_samples": args.max_samples,
		"seed": args.seed,
		"test_frac": args.test_frac,
		"detections_field": det_field,
		"classification_field": CLS_FIELD,
		"num_patches_total": n_total,
		"num_train": len(train_view),
		"num_test": len(test_view),
		"class_counts_total": class_counts,
		"output": {
			"train_dir": os.path.abspath(out_train),
			"test_dir": os.path.abspath(out_test),
		},
	}

	info_path = os.path.join(args.out, "dataset_info.json")
	with open(info_path, "w", encoding="utf-8") as fout:
		json.dump(info, fout, indent=2, ensure_ascii=False)

	print(f"[INFO] Dataset info saved → {info_path}")
	print()
	print("=" * 60)
	print("Done. Dataset is ready.")
	print("=" * 60)


if __name__ == "__main__":
	main()
