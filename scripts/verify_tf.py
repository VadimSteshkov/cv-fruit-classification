#!/usr/bin/env python3
"""Quick sanity check: print TensorFlow version and GPU availability."""

import tensorflow as tf


def main() -> None:
	print(f"TensorFlow version : {tf.__version__}")
	print(f"Keras version      : {tf.keras.__version__}")

	gpus = tf.config.list_physical_devices("GPU")
	print(f"GPUs detected      : {len(gpus)}")

	if gpus:
		for g in gpus:
			print(f"  • {g}")
	else:
		print("No GPU found — training will run on CPU.")
		print("For GPU support install the matching CUDA / cuDNN toolkit.")


if __name__ == "__main__":
	main()
