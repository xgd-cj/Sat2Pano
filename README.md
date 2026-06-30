# Sat2Density

Sat2Density is a two-stage satellite-to-panorama generation pipeline.

Sat2Pano tackles the task of Satellite-to-Street View Synthesis, aiming to generate photorealistic ground-level panoramas and videos from given satellite imagery and camera trajectories.
To address the trade-off dilemma between geometric consistency and texture fidelity, a novel coarse-to-fine two-stage generation framework is proposed.
Specifically, this paper first proposes a Geometry-Aware Coarse Synthesis module.
By utilizing explicit density prediction and color residual refinement, it establishes an accurate physical skeleton of the scene from the satellite image and renders coarse geometric cues. 
Subsequently, PanoDiT (Panorama Diffusion Transformer) is proposed to overcome the bottleneck of texture generation. 
PanoDiT leverages the geometric cues obtained from the coarse generation stage as Strong Spatial Guidance and incorporates a sky reference image for Global Illumination Control within the latent space to collaboratively refine and enhance the coarse results.
Experimental results demonstrate that Sat2Pano not only maintains cross-view 3D geometric consistency but also effectively injects photorealistic texture details, outperforming existing state-of-the-art methods in terms of generation quality and realism.

## Code Layout

```text
model_s1.py              Stage 1 density/color-volume model and renderer
model_s2.py              Stage 2 SatDiT model
dataset_s1.py            Stage 1 dataset loader
dataset_s2.py            Stage 2 dataset loader reference
train_stage1.py          Stage 1 training
train_stage2.py          Stage 2 training, random-sky version
infer_stage1.py          Stage 1 image inference and moving-frame generation
infer_stage2.py          Stage 2 frame/image refinement
metrics.py               SSIM/PSNR/LPIPS/FID/KID evaluation
utils/                   Dataset and visualization helpers
```



## Environment

Python 3.9+ and CUDA GPUs are recommended. Stage 2 training requires CUDA.

```bash
conda create -n sat2density python=3.10 -y
conda activate sat2density
pip install -r requirements.txt
```

Install PyTorch with the CUDA version that matches your machine if the generic install does not match your driver.

The code downloads Hugging Face models for Stage 2:

- `stabilityai/sd-vae-ft-mse`
- `facebook/DiT-XL-2-256` scheduler

If Hugging Face access is slow, set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Data Format

Stage 1 expects:

```text
data/CVUSA_train/
  train/
    sat_images/
    pano_images/
    sky_masks/
  val/
    sat_images/
    pano_images/
    sky_masks/
```

Stage 2 expects:

```text
data/CVACT_stage2train/
  init_proj/
  opacity/
  depth/
  ground_truth/
  sky/
```



## Train Stage 1

```bash
python train_stage1.py \
  --data_root data/CVUSA_train \
  --checkpoint_dir checkpoints/stage1 \
  --log_dir logs/stage1 \
  --output_dir outputs/stage1_train \
  --epochs 150 \
  --batch_size 2 \
  --n_gpus 1
```

Resume:

```bash
python train_stage1.py \
  --data_root data/CVUSA_train \
  --resume checkpoints/stage1/latest_model.pth \
  --checkpoint_dir checkpoints/stage1
```

## Train Stage 2

This is the random-sky version.

```bash
python train_stage2.py \
  --train_root data/CVACT_stage2train \
  --save_dir checkpoints/stage2 \
  --epochs 300 \
  --batch_size 8 \
  --n_gpus 1
```

## Single Image Inference

Stage 1 only:

```bash
python infer_stage1.py \
  --checkpoint checkpoints/stage1/best_model.pth \
  --sam_checkpoint sam_vit_l_0b3195.pth \
  --input examples/satellite.png \
  --output_dir outputs/stage1_inference
```

This writes:

```text
*_rgb.png
*_opacity.png
*_depth.png
*_density.npy
*_density.ply
```

Stage 2 refinement for Stage 1 outputs:

```bash
python infer_stage2.py \
  --input_dir outputs/stage1_inference \
  --sky_dir data/CVUSA_train/train/sky_masks \
  --checkpoint checkpoints/stage2/latest_dit.pth \
  --output_dir outputs/stage2_inference \
  --single_image \
  --steps 1000
```

## Continuous Frame Generation

Use Stage 1 moving mode to draw a path on the satellite image and render one frame per sampled path point:

```bash
python infer_stage1.py \
  --checkpoint checkpoints/stage1/best_model.pth \
  --input examples/0000177.jpg \
  --output_dir outputs/video/S1 \
  --move
```

OpenCV will show the satellite image.

- Left click: add a path point
- Right click: undo the last point
- Enter or Space: confirm and start rendering
- `D`: use the default straight path

The output uses sequence names:

```text
outputs/video/S1/0000177_000_rgb.png
outputs/video/S1/0000177_000_depth.png
outputs/video/S1/0000177_000_opacity.png
outputs/video/S1/0000177_001_rgb.png
...
```

Then refine the generated sequence with Stage 2:

```bash
python infer_stage2.py \
  --input_dir outputs/video/S1 \
  --sky_dir data/CVUSA_train/train/sky_masks \
  --checkpoint checkpoints/stage2/latest_dit.pth \
  --output_dir outputs/video/S2 \
  --steps 1000
```

This writes:

```text
outputs/video/S2/0000177_000_gen.png
outputs/video/S2/0000177_001_gen.png
...
```

To create an MP4:

```bash
ffmpeg -framerate 24 -i outputs/video/S2/0000177_%03d_gen.png \
  -c:v libx264 -pix_fmt yuv420p outputs/video/0000177_stage2.mp4
```

## Utilities

Colorize Stage 1 depth frames:

```bash
python utils/colorize_depth.py \
  --input_dir outputs/video/S1 \
  --output_dir outputs/video/S1_depth_color \
  --colormap magma
```

Evaluate generated panoramas:

```bash
python metrics.py \
  --gen_dir outputs/stage2_inference \
  --gt_dir data/ground_truth \
  --device cuda
```


