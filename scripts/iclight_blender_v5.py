#!/usr/bin/env python3
"""
Run IC-Light relighting on v5 dynamic Blender scene renders.

IC-Light (fbc model) takes foreground RGBA + background image and relights
the foreground to match the background illumination. We run it on rendered
frames from the DC-only harmonized PLY to see if it produces better targets
than the whitebox/PCT-Net harmonizers.

Usage:
    conda activate sa4d
    python scripts/iclight_blender_v5.py
"""
import os
import sys
import math
import glob

import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from tqdm import tqdm
from torch.hub import download_url_to_file

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICLIGHT_ROOT = "/home/ubuntu/IC-Light"

# Paths
RENDER_DIR = os.path.join(SA4D_ROOT, "results_v5/dynamic/difix_on_dc_only/dc_only_video_renders")
MASK_PT = os.path.join(SA4D_ROOT, "output/v5/dynamic_A/segment_results/bd_mask.pt")
OUTPUT_DIR = os.path.join(SA4D_ROOT, "results_v5/dynamic/iclight_relight")

# IC-Light model
SD15_NAME = "stablediffusionapi/realistic-vision-v51"
ICLIGHT_MODEL = os.path.join(ICLIGHT_ROOT, "models/iclight_sd15_fbc.safetensors")
ICLIGHT_URL = "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors"


def setup_iclight():
    """Load IC-Light fbc model."""
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
    from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
    from diffusers.models.attention_processor import AttnProcessor2_0
    from transformers import CLIPTextModel, CLIPTokenizer

    print("Loading SD1.5 components...")
    tokenizer = CLIPTokenizer.from_pretrained(SD15_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD15_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(SD15_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(SD15_NAME, subfolder="unet")

    # Modify UNet input conv for 12 channels (4 latent + 4 fg + 4 bg)
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels,
                                       unet.conv_in.kernel_size,
                                       unet.conv_in.stride,
                                       unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    # Hook forward
    unet_original_forward = unet.forward
    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
    unet.forward = hooked_unet_forward

    # Download and load IC-Light weights
    os.makedirs(os.path.dirname(ICLIGHT_MODEL), exist_ok=True)
    if not os.path.exists(ICLIGHT_MODEL):
        print(f"Downloading IC-Light weights...")
        download_url_to_file(url=ICLIGHT_URL, dst=ICLIGHT_MODEL)

    print("Loading IC-Light weights...")
    sd_offset = sf.load_file(ICLIGHT_MODEL)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged

    device = torch.device('cuda')
    text_encoder = text_encoder.to(device=device, dtype=torch.float16)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    unet = unet.to(device=device, dtype=torch.float16)

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, steps_offset=1)

    t2i_pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler, safety_checker=None,
        requires_safety_checker=False, feature_extractor=None,
        image_encoder=None)

    print("IC-Light ready!")
    return t2i_pipe, vae, tokenizer, text_encoder, device


def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h


def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def encode_prompt_pair(tokenizer, text_encoder, device, positive, negative):
    max_length = tokenizer.model_max_length
    c_ids = tokenizer(positive, truncation=True, max_length=max_length,
                      padding="max_length", return_tensors="pt").input_ids.to(device)
    uc_ids = tokenizer(negative, truncation=True, max_length=max_length,
                       padding="max_length", return_tensors="pt").input_ids.to(device)
    c = text_encoder(c_ids).last_hidden_state
    uc = text_encoder(uc_ids).last_hidden_state
    return c, uc


@torch.inference_mode()
def relight_frame(t2i_pipe, vae, tokenizer, text_encoder, device,
                  fg_rgba, bg_rgb, prompt="", steps=25, cfg=2.0, seed=42):
    """
    Relight foreground to match background lighting.
    fg_rgba: [H, W, 4] uint8 (RGBA, alpha = foreground mask)
    bg_rgb: [H, W, 3] uint8
    Returns: [H, W, 3] uint8 relit image
    """
    H, W = 512, 512  # IC-Light works at 512x512

    # Prepare foreground: premultiplied alpha on grey background
    fg = fg_rgba[:, :, :3].astype(np.float32)
    alpha = fg_rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg_premult = fg * alpha + 127.0 * (1.0 - alpha)
    fg_premult = fg_premult.clip(0, 255).astype(np.uint8)

    # Resize
    fg_resized = np.array(Image.fromarray(fg_premult).resize((W, H), Image.LANCZOS))
    bg_resized = np.array(Image.fromarray(bg_rgb).resize((W, H), Image.LANCZOS))

    # Encode to latents
    concat_conds = numpy2pytorch([fg_resized, bg_resized]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(tokenizer, text_encoder, device,
                                         prompt, "dark, shadow, lowres, bad")

    rng = torch.Generator(device=device).manual_seed(seed)
    latents = t2i_pipe(
        prompt_embeds=conds, negative_prompt_embeds=unconds,
        width=W, height=H, num_inference_steps=steps,
        num_images_per_prompt=1, generator=rng,
        output_type='latent', guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    result = pytorch2numpy(pixels)[0]
    return result


def create_fg_rgba(composite_img, mask_2d):
    """Create RGBA foreground from composite + mask."""
    rgba = np.zeros((*composite_img.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = composite_img
    rgba[:, :, 3] = (mask_2d * 255).astype(np.uint8)
    return rgba


def create_bg_only(composite_img, mask_2d):
    """Create background by inpainting masked region with surrounding color."""
    bg = composite_img.copy()
    # Simple: just use the composite as bg (IC-Light uses it for lighting reference)
    return bg


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get rendered frames
    render_files = sorted(glob.glob(os.path.join(RENDER_DIR, "*.png")))
    if not render_files:
        print(f"No renders found in {RENDER_DIR}")
        return
    print(f"Found {len(render_files)} rendered frames")

    # We need a 2D mask of the breakdancer for each frame.
    # For simplicity, render the mask from the 4DGS model for a few frames,
    # or use a rough spatial mask. Let's use a simple approach:
    # render a couple frames and create masks by comparing to Scene B.
    # Actually, the simplest: just use a fixed bounding box crop of the breakdancer region.
    #
    # Better approach: render the scene with and without the breakdancer mask to get a 2D mask.
    # But for quick testing, let's use IC-Light on the full frame with a rough mask.

    # Load IC-Light
    t2i_pipe, vae, tokenizer, text_encoder, device = setup_iclight()

    # For each frame, we need FG (breakdancer RGBA) and BG (scene without breakdancer).
    # Approximate: use the Scene B render as background reference (same camera, correct lighting).
    scene_b_dir = os.path.join(SA4D_ROOT, "results_v5/dynamic/scene_B")
    scene_b_video = os.path.join(scene_b_dir, "video_rgb.mp4")

    # Extract Scene B frames
    import subprocess
    scene_b_frames_dir = "/tmp/scene_b_frames_icl"
    os.makedirs(scene_b_frames_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", scene_b_video,
                    f"{scene_b_frames_dir}/frame_%05d.png"],
                   capture_output=True)
    scene_b_files = sorted(glob.glob(f"{scene_b_frames_dir}/*.png"))
    print(f"Scene B frames: {len(scene_b_files)}")

    n = min(len(render_files), len(scene_b_files))

    # For the foreground mask, we'll threshold the difference between
    # composite (with dark breakdancer) and a rough background estimate.
    # Simpler: use a fixed region. The breakdancer occupies roughly
    # the left-center portion of the frame. Let's just make a rough rect mask.
    # Actually, for IC-Light the FG input is the object on a neutral background.
    # The BG input tells it what lighting to use.
    # So FG = breakdancer extracted, BG = the full scene (or Scene B).

    # Let's use Scene A original renders as the composite, and Scene B as the
    # lighting reference background.
    orig_dir = os.path.join(SA4D_ROOT, "results_v5/dynamic/scene_A")
    orig_video = os.path.join(orig_dir, "video_rgb.mp4")
    orig_frames_dir = "/tmp/orig_frames_icl"
    os.makedirs(orig_frames_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", orig_video,
                    f"{orig_frames_dir}/frame_%05d.png"],
                   capture_output=True)
    orig_files = sorted(glob.glob(f"{orig_frames_dir}/*.png"))

    # Create a rough breakdancer mask by thresholding darkness
    # The breakdancer in Scene A is darker/redder than the bright white snow
    results = []
    for i in tqdm(range(n), desc="IC-Light relight"):
        composite = np.array(Image.open(orig_files[i]))
        scene_b = np.array(Image.open(scene_b_files[i]))

        # Create mask: pixels that differ significantly between A and B
        # are likely the breakdancer (different lighting)
        diff = np.abs(composite.astype(float) - scene_b.astype(float)).mean(axis=2)
        mask = (diff > 15).astype(np.float32)

        # Dilate mask slightly
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=3).astype(np.float32)

        # Create RGBA foreground
        fg_rgba = create_fg_rgba(composite, mask)

        # Use Scene B as the background (lighting reference)
        bg = scene_b

        # Run IC-Light
        relit = relight_frame(t2i_pipe, vae, tokenizer, text_encoder, device,
                              fg_rgba, bg, prompt="outdoor snowy scene, bright daylight",
                              steps=25, cfg=2.0, seed=42 + i)

        # Resize back to original size
        orig_h, orig_w = composite.shape[:2]
        relit_full = np.array(Image.fromarray(relit).resize((orig_w, orig_h), Image.LANCZOS))

        # Composite: blend relit foreground into Scene A background
        mask_3d = np.stack([mask] * 3, axis=2)
        final = (relit_full * mask_3d + composite * (1 - mask_3d)).clip(0, 255).astype(np.uint8)

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"frame_{i:05d}.png")
        Image.fromarray(final).save(out_path)

        # Also save comparison strip: original | IC-Light | Scene B GT
        strip = np.concatenate([composite, final, scene_b], axis=1)
        Image.fromarray(strip).save(os.path.join(OUTPUT_DIR, f"compare_{i:05d}.png"))

        results.append(final)

    # Build video
    import imageio
    imageio.mimwrite(os.path.join(OUTPUT_DIR, "video_iclight.mp4"),
                     results, fps=30, quality=8, macro_block_size=1)

    # Compute metrics vs Scene B
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    print("\nMetrics vs Scene B ground truth:")
    for label, src_files in [("Original A", orig_files),
                              ("IC-Light", [os.path.join(OUTPUT_DIR, f"frame_{i:05d}.png") for i in range(n)])]:
        psnrs, ssims_list = [], []
        for i in range(n):
            a = np.array(Image.open(src_files[i]))
            b = np.array(Image.open(scene_b_files[i]))
            if a.shape != b.shape:
                b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.BICUBIC))
            psnrs.append(psnr(b, a, data_range=255))
            ssims_list.append(ssim(b, a, channel_axis=2, data_range=255))
        print(f"  {label:20s}  PSNR={np.mean(psnrs):.3f}  SSIM={np.mean(ssims_list):.4f}")

    print("Done!")


if __name__ == "__main__":
    main()
