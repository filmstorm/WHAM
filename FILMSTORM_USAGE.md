# WHAM Usage Guide - Filmstorm Custom Setup

## Overview
This setup includes WHAM (World-grounded Humans with Accurate Motion) with custom modifications for:
- GPU memory optimization
- BVH export for animation workflows

## Installation Complete
All dependencies and models are installed. The environment is ready to use.

## Running WHAM Demo

### Basic Usage
```bash
# Activate environment
conda activate wham

# Process a video and visualize
CUDA_VISIBLE_DEVICES=1 python demo.py --video examples/drone_video.mp4 --visualize

# Process and save results
CUDA_VISIBLE_DEVICES=1 python demo.py --video examples/drone_video.mp4 --save_pkl --visualize
```

### Output Structure
```
output/demo/[video_name]/
├── wham_output.pkl       # SMPL pose and translation (world-space)
├── tracking_results.pth  # Detection and tracking data
├── slam_results.pth      # SLAM/camera motion data
└── overlay.mp4           # Visualization video
```

## Converting to BVH

### Basic BVH Export
```bash
# Convert WHAM output to BVH animation
python wham_to_bvh.py -i output/demo/happywalk/wham_output.pkl -o motion.bvh

# Specify FPS and subject
python wham_to_bvh.py -i output/demo/video/wham_output.pkl -o motion.bvh --fps 30 --subject 0
```

### BVH Features
- **World-grounded motion**: Character moves through scene space (not in-place)
- **Proper orientation**: Character faces walking direction with correct pelvis rotation
- **Standard skeleton**: SMPL 24-joint hierarchy compatible with animation tools
- **Scale**: Exported in centimeters for visibility

### BVH Coordinate System
- **Y-up**: Vertical axis
- **X-axis**: Primary movement direction
- **Z-axis**: Lateral movement
- **Rotation order**: ZXY (Euler angles in degrees)

## Custom Modifications

### 1. GPU Memory Optimization (`demo.py:89`)
Added cleanup after preprocessing to free GPU memory:
```python
# Free GPU memory after preprocessing
del detector
del extractor
if slam is not None:
    del slam
torch.cuda.empty_cache()
logger.info('Freed GPU memory after preprocessing')
```

This prevents OOM errors when running on consumer GPUs (8GB VRAM tested).

### 2. BVH Converter (`wham_to_bvh.py`)
Complete SMPL to BVH converter with:
- Direct coordinate passthrough (WHAM already uses correct Y-up world-space)
- Preserved pelvis rotation (no destructive transforms)
- Proper root motion with scene traversal
- Compatible with Blender, MotionBuilder, Maya, etc.

## Example Workflow

### Full Pipeline
```bash
# 1. Activate environment
conda activate wham

# 2. Process video
CUDA_VISIBLE_DEVICES=1 python demo.py \
  --video examples/drone_video.mp4 \
  --save_pkl \
  --visualize

# 3. Convert to BVH
python wham_to_bvh.py \
  -i output/demo/drone_video/wham_output.pkl \
  -o output/demo/drone_video/motion.bvh \
  --fps 30

# 4. Import motion.bvh into your animation software
```

### Multi-Person Videos
If video contains multiple people, the converter auto-selects the longest sequence. To specify a subject:
```bash
python wham_to_bvh.py -i wham_output.pkl -o motion.bvh --subject 0
```

## GPU Selection
The system has multiple GPUs. Use GPU 1 for WHAM (GPU 0 typically used by desktop):
```bash
CUDA_VISIBLE_DEVICES=1 python demo.py [options]
```

## Troubleshooting

### Out of Memory
- Ensure using GPU 1: `CUDA_VISIBLE_DEVICES=1`
- Check GPU usage: `nvidia-smi`
- The demo.py cleanup should handle this automatically

### Character Orientation Issues
The current converter preserves WHAM's exact rotations. If character appears rotated:
- Check the pelvis Y-rotation in BVH should match WHAM output (~70-80° for forward walk)
- Verify your animation software's coordinate system matches Y-up

### DPVO Compilation Issues
Already fixed with `-U__SIZEOF_INT128__` flag in `third-party/DPVO/setup.py`

## File Locations

### Key Scripts
- `demo.py` - Main WHAM processing (with GPU cleanup)
- `wham_to_bvh.py` - BVH converter
- `wham_to_bvh_backup.py` - Original backup version

### Models & Checkpoints
```
checkpoints/
├── wham_vit_bedlam_w_3dpw.pth.tar  # Main WHAM model
├── hmr2a.ckpt                       # HMR2 model
├── dpvo.pth                         # SLAM model
├── yolov8x.pt                       # Detector
└── vitpose-h-multi-coco.pth        # Pose estimator
```

### SMPL Models
```
_DATA/data/smpl/
├── SMPL_NEUTRAL.pkl
├── SMPL_MALE.pkl
└── SMPL_FEMALE.pkl
```

## Credits
- Original WHAM: https://github.com/yohanshin/WHAM/
- Custom modifications: Filmstorm (GPU optimization, BVH export)
- Reference BVH converter: HybrIK project
