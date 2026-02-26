# SECAD
SECAD is a deep learning framework for **serous effusion cytology image analysis** proposed in our research work.

> Note: This repository is a minimal public snapshot. Some paths in the provided test scripts are placeholders and must be set to your local dataset/checkpoint locations.

## Requirements

- Python >= 3.8 (recommended)
- PyTorch >= 1.10 (CUDA recommended)

## Evaluation scripts

### `tests/test.py` (WSI-level evaluation)

This script evaluates a **WSI-level classifier** that operates on precomputed feature maps, which were stored in pickle files.

How to use:

- Open `tests/test.py` and set:
  - `data_csv_path` — path to your CSV with GT
  - `pkl_root` — directory containing feature-map `.pkl` files
  - `model_path` — path to the trained WSI-level checkpoint
  - `csv_path` — output CSV file path
- Optionally adjust `CUDA_VISIBLE_DEVICES` and `DataLoader` workers.

Run:

```bash
python tests/test.py
```

### `tests/test_patch.py` (Patch-level evaluation)

This script evaluates the **patch-level model** on cropped patches, using annotation metadata.

How to use:

- Open `tests/test_patch.py` and set:
  - `val_data` — path to your patch image directory
  - `xml_root` — path to XML annotations
  - checkpoint path passed to `torch.load(...)`
  - optionally `CUDA_VISIBLE_DEVICES`
- Ensure the checkpoint’s architecture matches `model_name` and `img_size`.

Run:

```bash
python tests/test_patch.py
```

## Training scripts

### `train_baseline2.py` (Patch-Level Training)

This script trains a patch-level image classifier using image patches and optional XML annotations.

Data Required:
- Patch image folder
- XML annotation folder

How to use:

Set paths in training script:
   - train_data
   - train_xml
   - val_data
   - val_xml
   - checkpoint_dir

Run:

```bash
python train_patch.py
```

### `train_wsi.py` (Slide-Level Training)

This script trains a slide-level classifier using pre-extracted feature maps (.pkl files).

Data Required:
- CSV file (slide id, label, feature reference)
- Feature `.pkl` files directory

How to use:

Set paths in training script:
   - train_csv
   - val_csv
   - pkl_root
   - checkpoint_dir

Run:

```bash
python train_wsi.py
```

## Notes

- All scripts assume a CUDA environment and explicitly set `CUDA_VISIBLE_DEVICES`.
- Paths in the scripts are placeholders or author-specific directories; update them to match your environment.

## License

MIT License (see `LICENSE`).
