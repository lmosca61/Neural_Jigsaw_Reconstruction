# Neural Jigsaw Reconstruction

Reconstructing a **96 × 96 RGB image** from nine shuffled, eroded patches using a **shared CNN encoder, Transformer attention, and a residual U-Net**.

The model learns to arrange visual information and reconstruct missing borders through differentiable soft routing, under a budget of six million trainable parameters.

**Stack:** Python · TensorFlow · Keras · NumPy · Matplotlib

## Task

Each source image is divided into a 3 × 3 grid of 32 × 32 cells. Removing a two-pixel border from every cell produces nine 28 × 28 patches, which are then shuffled.

The network receives these patches and predicts the original image. Approximately **23.4% of the source pixels are missing** from the input, so the task requires both spatial reasoning and reconstruction.

| | Shape |
|---|---|
| Input | `(batch, 9, 28, 28, 3)` |
| Output | `(batch, 96, 96, 3)` |
| Pixel range | `[0, 1]` |

## Architecture

| Component | Role |
|---|---|
| Shared CNN encoder | Encodes each patch into a 192-dimensional token using convolutions, max pooling, and a dense projection. |
| Transformer encoder | Three self-attention blocks with four heads model relationships between patches. |
| Learned grid queries | Nine learned queries cross-attend to contextual tokens to represent output grid cells. |
| Neural router | Computes soft cell-to-patch assignments from grid slots and original patch tokens. |
| Differentiable assembler | Blends patches with attention weights and assembles a zero-padded 96 × 96 canvas. |
| Residual U-Net | Predicts corrections to the assembled canvas; the final output is clipped to `[0, 1]`. |

**Trainable parameters: 3,739,059**, as recorded in the original model summary. The model was trained from scratch without a pretrained backbone.

Entropy regularization encourages concentrated routing weights, while a column-sum penalty encourages balanced patch use. These are soft constraints: they do not guarantee an exact one-to-one permutation.

## Dataset

The experiment uses the **100,000-image unlabeled split of STL-10**, partitioned by index:

| Partition | Images | Index range |
|---|---:|---|
| Training | 80,000 | `0:80000` |
| Validation | 10,000 | `80000:90000` |
| Test | 10,000 | `90000:100000` |

This is a custom reconstruction holdout, **not the official STL-10 classification benchmark**.

The notebook uses a read-only memory map for the binary images and normalizes images within each batch, avoiding a full float32 copy of the dataset in RAM.

Dataset information: [STL-10](https://cs.stanford.edu/~acoates/stl10/).

## Results and evaluation

### Test-set reconstruction

Results from the corrected inference-only evaluation, reported on **10,000 test images** with RGB values normalized to `[0, 1]`:

| Metric | Value |
|---|---:|
| Mean per-image MAE | **0.03875** |
| Standard deviation of per-image MAE | **0.04802** |
| MAE reduction relative to the mean-patch baseline | **78.75%** |

The evaluator computes one MAE per image, then reports the mean and population standard deviation across all test images. Every image receives equal weight, including those in the final partial batch. The standard deviation describes variation between images; it is not a confidence interval.

The mean-patch baseline averages the nine patches, tiles the average into an 84 × 84 canvas, and resizes it to 96 × 96. Baseline and model evaluation reset the same NumPy seed and use the same non-shuffled test generator, so their sequential evaluation loops receive the same puzzles. Relative MAE reduction is calculated as `100 × (baseline MAE − model MAE) / baseline MAE`.

## Run the notebook

Open [`UNet_reconstruction.ipynb`](UNet_reconstruction.ipynb) in Google Colab. A GPU runtime is recommended for full test-set inference.

1. Upload the existing checkpoint as `best_jigsaw.weights.h5` to the runtime working directory, or let the notebook attempt its configured Google Drive download.
2. Keep **`TRAIN_MODEL = False`**.
3. Run the notebook cells in order. The notebook loads the dataset, computes the baseline, builds the model, loads the trained weights, and evaluates reconstruction quality.
4. Confirm that both evaluation summaries report **10,000 test images**.
5. Save the notebook with the updated evaluation outputs and qualitative examples.

The training block is skipped when `TRAIN_MODEL = False`. Historical training curves are embedded as an image and remain visible without rerunning training or restoring a `history` object.

If checkpoint loading fails, supply the existing trained checkpoint before continuing to evaluation.

### Checkpoint

The notebook contains the original [Google Drive checkpoint link](https://drive.google.com/file/d/1i44PssbOpJ6xczLpGgykk5GyxYXNWO8W/view) and supports downloading it with `gdown`.

Availability of that link and identity with the original best local checkpoint have not been independently verified during the notebook cleanup. No weights are embedded in the notebook.

### Local environment

For execution outside Colab, install the required packages in an isolated environment:

```bash
pip install tensorflow keras numpy matplotlib gdown jupyterlab
jupyter lab UNet_reconstruction.ipynb
```

The implementation uses Keras 3 APIs with TensorFlow. Exact dependency versions from the original training run were not recorded; this is an unpinned setup, not a validated environment lockfile.

## Original training setup

| Setting | Value |
|---|---|
| Batch size | 32 |
| Maximum epochs | 60 |
| Optimizer | AdamW |
| Peak learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Schedule | Three-epoch linear warmup followed by cosine decay |
| Reconstruction objective | Pixel MAE + `0.15 ×` Sobel-gradient MAE |
| Routing regularization | Entropy weight `0.01`; column penalty weight `0.5` |
| Checkpoint selection | Lowest validation MAE |
| Early stopping | Patience 10, restoring best weights |

The training objective includes gradient and routing terms, so total loss is distinct from reconstruction MAE.

## Limitations

- Soft routing may blend or reuse patches; reconstruction MAE does not measure exact permutation recovery.
- The training curves show a train/validation gap. Checkpoint selection does not eliminate overfitting.
- No ablation study isolates the contribution of the Transformer, routing regularizers, edge loss, or U-Net.
- Results concern this custom STL-10 holdout; performance on other image distributions has not been established.
- Full reproduction requires the trained checkpoint and dataset. The original run did not retain a complete dependency lockfile or per-image evaluation errors.
