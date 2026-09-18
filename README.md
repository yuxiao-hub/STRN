# STRN: 3D Human Motion Prediction

This repository implements STRN for 3D human motion prediction. The code supports Human3.6M and CMU-Mocap, including training, checkpoint evaluation, and Walking-action visualization.

Run all commands from the project root.

## Requirements

- Python 3.8 or later
- PyTorch 2.0 or later with a CUDA-compatible build
- NumPy, SciPy, h5py, pandas, matplotlib, seaborn, six, and Pillow

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

For GPU use, install a PyTorch build compatible with the CUDA version on the server before installing the remaining packages if necessary.

## Datasets

### Human3.6M

Download the exponential-map version from the [Human3.6M website](http://vision.imar.ro/human3.6m/description.php) or the commonly used archive:

<http://www.cs.stanford.edu/people/ashesh/h3.6m.zip>

After extracting and converting the data to the text format used by this project, place it under `datasets/h3.6m/`.

### CMU-Mocap

Download CMU-Mocap from the [official website](http://mocap.cs.cmu.edu/). The text-format preprocessing used by this project can also be obtained from the [ConvSeq2Seq data repository](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics).

Place the processed files under `datasets/cmu/`.

## Directory Structure

```text
.
├── main_h36m_3d.py             # Human3.6M training and overall evaluation
├── main_h36m_3d_eval.py        # Human3.6M action evaluation
├── main_cmu_3d.py              # CMU training
├── main_cmu_3d_eval.py         # CMU evaluation with per-sample outputs
├── visualize_h36m_walking_ours.py
├── visualize_h36m_walking_gt.py
├── model/                       # STRN model definitions
├── utils/                       # Data loading and preprocessing
├── datasets/
│   ├── h3.6m/
│   └── cmu/
├── checkpoint/                  # Checkpoints and evaluation results
├── outputs/                     # Generated figures and visualizations
├── requirements.txt
└── README.md
```

The expected dataset layout is:

```text
datasets/
├── h3.6m/
│   ├── S1/
│   ├── S5/
│   ├── S6/
│   ├── S7/
│   ├── S8/
│   ├── S9/
│   └── S11/
└── cmu/
    ├── train/
    │   ├── basketball/
    │   ├── basketball_signal/
    │   ├── directing_traffic/
    │   ├── jumping/
    │   ├── running/
    │   ├── soccer/
    │   ├── walking/
    │   └── washwindow/
    └── test/
        ├── basketball/
        ├── basketball_signal/
        ├── directing_traffic/
        ├── jumping/
        ├── running/
        ├── soccer/
        ├── walking/
        └── washwindow/
```

For Human3.6M, each subject directory should contain files such as:

```text
datasets/h3.6m/S5/walking_1.txt
datasets/h3.6m/S5/walking_2.txt
```

For CMU, each action directory should contain numbered text files such as:

```text
datasets/cmu/train/walking/walking_1.txt
datasets/cmu/test/walking/walking_1.txt
```

## Training

### Human3.6M

The standard setting uses 50 observed frames and predicts the next 10 frames. The training split contains S1, S6, S7, S8, and S9; S11 is used for validation and S5 for testing.

```bash
python main_h36m_3d.py \
  --ckpt checkpoint/repro_h36m \
  --in_features 66 \
  --num_stage 14 \
  --d_model 256 \
  --kernel_size 10 \
  --input_n 50 \
  --output_n 10 \
  --dct_n 20 \
  --skip_rate 1 \
  --lr_now 0.0005 \
  --epoch 50 \
  --batch_size 32 \
  --test_batch_size 32
```

The checkpoint is saved under:

```text
checkpoint/repro_h36m/main_h36m_3d_in50_out10_ks10_dctn20/
```

### CMU-Mocap

```bash
python main_cmu_3d.py \
  --ckpt checkpoint/repro_cmu \
  --in_features 75 \
  --num_stage 12 \
  --d_model 256 \
  --kernel_size 10 \
  --input_n 50 \
  --output_n 10 \
  --dct_n 20 \
  --skip_rate 1 \
  --lr_now 0.0005 \
  --epoch 50 \
  --batch_size 32 \
  --test_batch_size 32
```

The checkpoint is saved under:

```text
checkpoint/repro_cmu/main_cmu_3d_in50_out10_ks10_dctn20/
```

## Evaluation

### Human3.6M

The following command evaluates the existing 2024 STRN checkpoint. `--ckpt` must point to the inner experiment directory that contains `ckpt_best.pth.tar`.

```bash
python main_h36m_3d.py \
  --is_eval \
  --ckpt checkpoint/main_h36m_3d_in50_out10_ks10_dctn20/main_h36m_3d_in50_out10_ks10_dctn20 \
  --in_features 66 \
  --num_stage 14 \
  --d_model 256 \
  --kernel_size 10 \
  --input_n 50 \
  --output_n 10 \
  --dct_n 20 \
  --test_batch_size 32
```

Important outputs:

```text
checkpoint/main_h36m_3d_in50_out10_ks10_dctn20/main_h36m_3d_in50_out10_ks10_dctn20/test_sample_errors.npy
checkpoint/main_h36m_3d_in50_out10_ks10_dctn20/main_h36m_3d_in50_out10_ks10_dctn20/strn_h36m_sample_ids.csv
checkpoint/main_h36m_3d_in50_out10_ks10_dctn20/main_h36m_3d_in50_out10_ks10_dctn20/test_sample_ids.csv
```

`test_sample_errors.npy` contains one scalar MPJPE for each test window. The ID CSV files use the same row order as the error array.

### CMU-Mocap

Use the dedicated evaluation entry point for the recovered 2024 CMU checkpoint:

```bash
python main_cmu_3d_eval.py \
  --ckpt checkpoint/main_cmu_3d_in50_out10_ks10_dctn20 \
  --output_dir checkpoint/main_cmu_3d_in50_out10_ks10_dctn20 \
  --test_batch_size 32 \
  --device cuda
```

Important outputs include:

```text
checkpoint/main_cmu_3d_in50_out10_ks10_dctn20/test_pre_action_sample_errors.npy
checkpoint/main_cmu_3d_in50_out10_ks10_dctn20/strn_cmu_sample_errors.npy
checkpoint/main_cmu_3d_in50_out10_ks10_dctn20/strn_cmu_frame_errors.npy
checkpoint/main_cmu_3d_in50_out10_ks10_dctn20/strn_cmu_sample_ids.csv
```

The CMU evaluator also writes per-action error arrays and sample ID CSV files for the eight test actions.

## Walking Visualization

The following command visualizes a Human3.6M Walking sequence using the formal STRN checkpoint:

```bash
python visualize_h36m_walking_ours.py \
  --model-path checkpoint/main_h36m_3d_in50_out10_ks10_dctn20/main_h36m_3d_in50_out10_ks10_dctn20/ckpt_best.pth.tar \
  --subject S5 \
  --action walking \
  --subaction 1 \
  --sample-idx 0 \
  --input-n 50 \
  --output-n 25 \
  --output-step 10 \
  --in-features 66 \
  --kernel-size 10 \
  --d-model 256 \
  --num-stage 14 \
  --dct-n 20 \
  --device cuda:0 \
  --output outputs/h36m_prediction/walking_hisrep_style.png
```

The visualization script also writes a text metadata file with the same base name. To generate a ground-truth-only reference image:

```bash
python visualize_h36m_walking_gt.py \
  --subject S5 \
  --action walking \
  --subaction 1 \
  --sample-idx 0 \
  --input-n 50 \
  --output-n 25 \
  --output outputs/h36m_prediction/walking_hisrep_style_gt.png
```

