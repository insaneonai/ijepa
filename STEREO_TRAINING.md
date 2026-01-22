# Stereo I-JEPA Training Setup

## Quick Start

### 1. Update Config File

Edit `configs/flyingthings3d_stereo_naive.yaml`:

```yaml
data:
  root_path: /path/to/FlyingThings3D # ⚠️ CHANGE THIS
  batch_size: 32 # Adjust for your GPU (32 for 12GB, 64 for 24GB)

logging:
  folder: ./logs/stereo_naive_ft3d # ⚠️ CHANGE THIS
```

### 2. Ensure Dataset is Downloaded

Download FlyingThings3D from:
https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

Expected structure:

```
FlyingThings3D/
├── frames_cleanpass/
│   ├── TRAIN/
│   │   ├── A/
│   │   │   ├── left/
│   │   │   └── right/
│   │   ├── B/
│   │   └── C/
│   └── TEST/
└── disparity/
```

### 3. Train the Model

#### Option A: Using helper script

```bash
python train_stereo.py
```

#### Option B: Direct command

```bash
python main.py --fname configs/flyingthings3d_stereo_naive.yaml --devices cuda:0
```

### 4. Monitor Training

Logs will be saved to the folder specified in config:

```
logs/stereo_naive_ft3d/
├── stereo_jepa_r0.csv         # Training metrics
├── stereo_jepa-latest.pth.tar # Latest checkpoint
└── stereo_jepa-ep50.pth.tar   # Checkpoint at epoch 50
```

## Configuration Guide

### Key Parameters to Adjust

**Batch Size** (based on GPU memory):

- 12GB GPU (RTX 3080/3060): `batch_size: 16-32`
- 24GB GPU (RTX 3090/4090/A5000): `batch_size: 64-128`
- 48GB GPU (A6000/A100): `batch_size: 128-256`

**Model Size**:

- `vit_tiny`: Fastest, good for debugging (192 dim)
- `vit_small`: Good balance (384 dim)
- `vit_base`: Standard choice (768 dim) ✅ **Recommended**
- `vit_large`: High capacity (1024 dim)

**Training Duration**:

- Quick experiment: `epochs: 20`
- Baseline: `epochs: 50` ✅ **Recommended**
- Full training: `epochs: 100-300`

**Mixed Precision** (for Ampere+ GPUs):

```yaml
meta:
  use_bfloat16: true # Speeds up training ~2x
```

## What This Setup Does

**Architecture**: Cross-View Naive Stereo I-JEPA

- **Context Encoder**: Processes LEFT image with spatial masking
- **Target Encoder**: Processes RIGHT image (full, no masking)
- **Predictor**: Maps context features → target features
- **Loss**: Smooth L1 between predicted and actual target features

**Key Differences from Original I-JEPA**:

1. Uses stereo pairs instead of single images
2. Left image for context, right image for target
3. Tests if stereo geometry helps learn 3D-aware representations

## Expected Training Behavior

**First few epochs**:

- Loss: ~1.0-3.0 (depends on initialization)
- Should decrease steadily

**After 10-20 epochs**:

- Loss: ~0.3-0.8
- Model starting to learn stereo correspondence

**After 50 epochs**:

- Loss: ~0.1-0.3
- Ready for evaluation

**Warning Signs**:

- ❌ Loss stays > 2.0 after 10 epochs → Check data loading
- ❌ Loss = NaN → Lower learning rate
- ❌ Loss oscillates wildly → Check batch size / learning rate

## Next Steps After Training

1. **Evaluate depth estimation** (linear probe)
2. **Test cross-view retrieval**
3. **Analyze attention patterns**
4. **Compare to monocular baseline**

See research plan for full evaluation protocol!

## Troubleshooting

### "CUDA out of memory"

→ Reduce `batch_size` in config

### "Dataset not found"

→ Check `root_path` in config points to FlyingThings3D directory

### "Slow data loading"

→ Increase `num_workers` (but not more than CPU cores)

### Training too slow

→ Set `use_bfloat16: true` (if you have Ampere+ GPU)
→ Reduce image resolution: `crop_size: 192` or `crop_size: 160`

## Hardware Recommendations

**Minimum**:

- GPU: GTX 1080 Ti / RTX 2080 (11GB VRAM)
- RAM: 16GB
- Storage: ~30GB for dataset

**Recommended**:

- GPU: RTX 3090 / A5000 (24GB VRAM)
- RAM: 32GB
- Storage: SSD with ~50GB free

**Training Time Estimates** (50 epochs, batch_size=32, vit_base):

- RTX 3090: ~12-16 hours
- RTX 4090: ~8-10 hours
- A100: ~6-8 hours
