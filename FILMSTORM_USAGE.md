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

### Single Person BVH Export
```bash
# Convert WHAM output to BVH animation (single person)
python wham_to_bvh.py -i output/demo/happywalk/wham_output.pkl -o motion.bvh

# Specify FPS and subject
python wham_to_bvh.py -i output/demo/video/wham_output.pkl -o motion.bvh --fps 30 --subject 0
```

### Multi-Person Scene Export

**Unified Scene BVH (RECOMMENDED)** - All characters in one file:
```bash
# Export all people to a single unified BVH file
python wham_to_bvh_scene.py -i output/demo/video/wham_output.pkl -o scene.bvh

# Custom minimum frames threshold (default: 30)
python wham_to_bvh_scene.py -i wham_output.pkl -o scene.bvh --min-frames 50
```

**Separate BVH Files** - Each character gets own file:
```bash
# Export all detected people as separate synchronized BVH files
python wham_to_bvh_multiperson.py -i output/demo/video/wham_output.pkl -o output/demo/video/scene

# Export without frame synchronization (each person has independent timeline)
python wham_to_bvh_multiperson.py -i wham_output.pkl -o scene --no-sync
```

### Single-Person BVH Features
- **World-grounded motion**: Character moves through scene space (not in-place)
- **Proper orientation**: Character faces walking direction with correct pelvis rotation
- **Standard skeleton**: SMPL 24-joint hierarchy compatible with animation tools
- **Scale**: Exported in centimeters for visibility

### Multi-Person Scene Features

**Unified Scene BVH** (wham_to_bvh_scene.py):
- ✅ **Single BVH file** - All characters in one file for easy import
- ✅ **Unified world-space** - Characters positioned correctly relative to each other
- ✅ **Frame synchronization** - All characters on same timeline
- ✅ **Hierarchical structure** - Each character as separate branch under Scene root
- ✅ **T-pose padding** - Characters appear/disappear smoothly

The unified exporter creates:
```
scene.bvh                  # Single file with all characters
```

Hierarchy structure:
```
ROOT Scene
  JOINT Character_00       # Person 0
    JOINT Character_00_Pelvis
      ... (full skeleton)
  JOINT Character_01       # Person 1
    JOINT Character_01_Pelvis
      ... (full skeleton)
  ...
```

**Separate BVH Files** (wham_to_bvh_multiperson.py):
- **Individual files** - Each person gets own BVH file
- **Unified world-space** - All positioned correctly relative to each other
- **Frame synchronization** - Optional (--no-sync to disable)
- **Scene metadata** - JSON file with character info

The separate exporter creates:
```
scene/
├── person_00.bvh          # Each detected person
├── person_01.bvh
├── person_02.bvh
├── ...
└── scene_metadata.json    # Scene info and character list
```

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

**Single person export** - Auto-selects longest sequence, or specify subject:
```bash
python wham_to_bvh.py -i wham_output.pkl -o motion.bvh --subject 0
```

**Multi-person scene export**:

Option 1 - Unified BVH (RECOMMENDED):
```bash
python wham_to_bvh_scene.py -i wham_output.pkl -o scene.bvh
# Import single scene.bvh file with all characters
```

Option 2 - Separate BVH files:
```bash
python wham_to_bvh_multiperson.py -i wham_output.pkl -o scene/
# Import all person_*.bvh files together
```

The separate files option includes metadata showing each person's position:
```json
{
  "fps": 30.0,
  "scene_frames": 356,
  "people": [
    {
      "subject_id": 0,
      "bvh_file": "person_00.bvh",
      "frames": 356,
      "frame_range": [0, 355],
      "center_position_cm": [824.3, -20.4, -37.7]
    },
    ...
  ]
}
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

**Fix details:**
The DPVO CUDA extensions failed to compile due to conda sysroot headers containing `__int128` types incompatible with CUDA nvcc. Fixed by adding `-U__SIZEOF_INT128__` to all nvcc `extra_compile_args` in the setup.py:

```python
# In third-party/DPVO/setup.py
ext_modules=[
    CUDAExtension('cuda_corr',
        sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
        extra_compile_args={
            'cxx':  ['-O3'],
            'nvcc': ['-O3', '-U__SIZEOF_INT128__'],  # Added this flag
        }),
    # ... same for cuda_ba and lietorch_backends
]
```

This fix is in the DPVO submodule and wasn't committed to the main repo.

## File Locations

### Key Scripts
- `demo.py` - Main WHAM processing (with GPU cleanup)
- `wham_to_bvh.py` - Single-person BVH converter
- `wham_to_bvh_scene.py` - Unified scene BVH converter (all characters in one file)
- `wham_to_bvh_multiperson.py` - Multi-person BVH converter (separate files)
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
