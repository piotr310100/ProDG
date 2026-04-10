import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxPipeline
from omegaconf import DictConfig
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from data import create_indexed_dataloader
from experiments.epic_generative_learnable import (
    VariationalPromptBank,
    differentiable_flux_generate,
    get_heatmap,
    set_seeds,
)
from matrix import create_matrix
from models import create_backbone_model, create_modified_head
from prototypes import compute_activation_bbox, pixelwise_multiply, topk_active_channels


def explain_predictions(config: DictConfig):
    set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = config.output_path
    output_dir = os.path.join(checkpoint_dir, config.visualization.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Backbone and Model Bundle...")
    model_bundle = create_backbone_model(
        config.model.name,
        device,
        config.model.custom_weights_path,
        config.model.num_classes,
    )
    backbone = model_bundle.backbone.to(torch.float32).eval().to(device)
    base_model = model_bundle.base_model.to(torch.float32).eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    m_3d, s_3d = mean.squeeze().view(3, 1, 1).cpu(), std.squeeze().view(3, 1, 1).cpu()

    print(f"Loading Generative Model: {config.generative_model.model_id}")
    pipe = FluxPipeline.from_pretrained(
        config.generative_model.model_id, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    print(f"Loading trained weights from: {checkpoint_dir}")

    U = create_matrix(config.matrix.type, model_bundle.num_channels, device).to(
        torch.float32
    )
    u_path = os.path.join(checkpoint_dir, f"{config.matrix.type}.pt")
    if not os.path.exists(u_path):
        raise FileNotFoundError(f"Could not find matrix weights at {u_path}")
    U.load_state_dict(torch.load(u_path, map_location=device))
    U.eval()

    prompt_bank = VariationalPromptBank(model_bundle.num_channels, pipe, device)
    prompt_path = os.path.join(checkpoint_dir, "learned_prompts.pt")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Could not find learned prompts at {prompt_path}")
    prompt_bank.load_state_dict(torch.load(prompt_path, map_location=device))
    prompt_bank.eval()

    mod_head = create_modified_head(base_model, config.model.name, U=U)

    val_loader = create_indexed_dataloader(
        config.dataset.datapath_val,
        model_bundle.transform,
        batch_size=1,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        shuffle=config.dataloader.shuffle,
    )

    print(
        f"Starting Concept Explanation for {config.visualization.num_images} images..."
    )

    with torch.no_grad():
        for i, (img_v, _) in enumerate(
            tqdm(val_loader, total=config.visualization.num_images)
        ):
            if i >= config.visualization.num_images:
                break

            img_v_cuda = img_v.to(device)
            img_v_norm = (img_v_cuda - mean) / std

            v_feats_raw = backbone(img_v_norm)
            v_feats_rot = pixelwise_multiply(
                v_feats_raw.to(torch.float32), U().to(torch.float32)
            )

            # Fetch heatmap values for the entire original image across all channels
            hm_orig = get_heatmap(
                backbone, mod_head, img_v_norm, batch_size=1, device=device
            )

            top_ch = topk_active_channels(
                backbone,
                mod_head,
                img_v_cuda[0],
                k=config.visualization.k_top,
                device=device,
            )

            # Setup figures (Original figure and Overlay figure)
            fig, axes = plt.subplots(
                config.visualization.k_top,
                6,
                figsize=(24, 4 * config.visualization.k_top),
            )
            fig_overlay, axes_overlay = plt.subplots(
                config.visualization.k_top,
                6,
                figsize=(24, 4 * config.visualization.k_top),
            )

            for row, ch in enumerate(top_ch):
                orig_img_tensor = (img_v_cuda[0].cpu() * s_3d + m_3d).clip(0, 1)
                orig_pil = to_pil_image(orig_img_tensor)

                x1, y1, x2, y2 = compute_activation_bbox(
                    v_feats_rot[0, ch].unsqueeze(0),
                    (orig_img_tensor.shape[1], orig_img_tensor.shape[2]),
                )

                draw = ImageDraw.Draw(orig_pil)
                draw.rectangle(
                    [x1.item(), y1.item(), x2.item() - 1, y2.item() - 1],
                    outline=(255, 255, 0),
                    width=3,
                )

                axes[row, 0].imshow(np.array(orig_pil))
                axes[row, 0].set_ylabel(f"Channel {ch}", fontsize=14, fontweight="bold")
                axes[row, 0].set_xticks([])
                axes[row, 0].set_yticks([])

                # Generate the overlay map for the specific channel dynamically normalized
                hm_c = hm_orig[0, ch]
                hm_c_norm = hm_c / (np.max(hm_c) + 1e-8)

                axes_overlay[row, 0].imshow(np.array(orig_pil))
                axes_overlay[row, 0].imshow(hm_c_norm, cmap="jet", alpha=0.6)
                axes_overlay[row, 0].set_ylabel(
                    f"Channel {ch}", fontsize=14, fontweight="bold"
                )
                axes_overlay[row, 0].set_xticks([])
                axes_overlay[row, 0].set_yticks([])

                ch_tensor = torch.tensor([ch] * 5, device=device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pe, ppe, _, _ = prompt_bank(ch_tensor)
                    ex_imgs_t = differentiable_flux_generate(
                        pipe,
                        pe,
                        ppe,
                        num_steps=config.generative_model.gen_steps,
                        guidance_scale=config.generative_model.guidance_scale,
                        device=device,
                        seed=config.seed,
                    )

                for col in range(5):
                    proto_img_t = ex_imgs_t[col : col + 1]
                    proto_in = F.interpolate(
                        proto_img_t.float(), (224, 224), mode="bilinear"
                    )

                    proto_in_norm = (proto_in - mean) / std
                    proto_feats = backbone(proto_in_norm)
                    proto_rot = pixelwise_multiply(
                        proto_feats.to(torch.float32), U().to(torch.float32)
                    )

                    # Fetch the entire heatmap for the specific prototype visualization
                    hm_proto = get_heatmap(
                        backbone, mod_head, proto_in_norm, batch_size=1, device=device
                    )

                    px1, py1, px2, py2 = compute_activation_bbox(
                        proto_rot[0, ch].unsqueeze(0),
                        (proto_img_t.shape[2], proto_img_t.shape[3]),
                    )

                    proto_pil = to_pil_image(proto_img_t[0].float().cpu())
                    p_draw = ImageDraw.Draw(proto_pil)
                    p_draw.rectangle(
                        [px1.item(), py1.item(), px2.item() - 1, py2.item() - 1],
                        outline=(255, 255, 0),
                        width=3,
                    )

                    axes[row, col + 1].imshow(np.array(proto_pil))
                    axes[row, col + 1].axis("off")

                    # Overlay for prototypes
                    hm_p_c = hm_proto[0, ch]
                    hm_p_c_tensor = torch.from_numpy(hm_p_c).unsqueeze(0).unsqueeze(0)
                    hm_p_c_resized = (
                        F.interpolate(
                            hm_p_c_tensor,
                            size=(proto_img_t.shape[2], proto_img_t.shape[3]),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze()
                        .numpy()
                    )

                    hm_p_c_norm = hm_p_c_resized / (np.max(hm_p_c_resized) + 1e-8)

                    axes_overlay[row, col + 1].imshow(np.array(proto_pil))
                    axes_overlay[row, col + 1].imshow(
                        hm_p_c_norm, cmap="jet", alpha=0.6
                    )
                    axes_overlay[row, col + 1].axis("off")

            # Save the bounding box grid
            fig.tight_layout()
            save_path = os.path.join(output_dir, f"explanation_{i:03d}.jpg")
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

            # Save the heatmap overlay grid
            fig_overlay.tight_layout()
            save_path_overlay = os.path.join(
                output_dir, f"explanation_{i:03d}_overlay.jpg"
            )
            fig_overlay.savefig(save_path_overlay, bbox_inches="tight")
            plt.close(fig_overlay)

    print(f"Finished. Explanations saved in {os.path.abspath(output_dir)}")
