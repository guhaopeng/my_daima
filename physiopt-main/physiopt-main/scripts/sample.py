import os
import sys
import shutil
from argparse import ArgumentParser

sys.path.append("..")

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
import trimesh
import torch
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.modules import sparse as sp
from trellis.modules.sparse.basic import save_slat_conds

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, default="A chair looking like a avocado.")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--rf", action="store_true")
    parser.add_argument("--no_slat", action="store_true")
    parser.add_argument("--out_folder", type=str, default="out/tmp")
    parser.add_argument("--n_samples", type=int, default=1)
    args = parser.parse_args()

    image_mode = args.image is not None

    with torch.no_grad():

        # Load a pipeline from a model folder or a Hugging Face model hub.
        if image_mode:
            pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-image-large"
            )
        else:
            pipeline = TrellisTextTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-text-xlarge"
            )
        pipeline.cuda()

        # Prepare formats to return
        formats = ["gaussian"]
        if args.rf:
            formats.append("radiance_field")
        if args.mesh:
            formats.append("mesh")
        if not args.no_slat:
            formats.append("slat")

        # Run the pipeline
        outputs = pipeline.run(
            Image.open(args.image) if image_mode else args.text,
            seed=args.seed,
            # Optional parameters
            # sparse_structure_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 7.5,
            # },
            # slat_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 7.5,
            # },
            formats=formats,
            num_samples=args.n_samples,
        )
        # outputs is a dictionary containing generated 3D assets in different formats:
        # - outputs['gaussian']: a list of 3D Gaussians
        # - outputs['radiance_field']: a list of radiance fields
        # - outputs['mesh']: a list of meshes

        os.makedirs(args.out_folder, exist_ok=True)

        # If it's an image, copy it. If it's a text, write it.
        if image_mode:
            _, ext_image = os.path.splitext(args.image)
            shutil.copy(args.image, os.path.join(args.out_folder, "image" + ext_image))
        else:
            pass

        for i_sample in range(args.n_samples):

            # Render/export visualization of the outputs
            video = render_utils.render_video(outputs["gaussian"][i_sample])["color"]
            imageio.mimsave(
                os.path.join(args.out_folder, f"sample_gs_{i_sample:02d}.mp4"),
                video,
                fps=30,
            )
            if args.rf:
                video = render_utils.render_video(outputs["radiance_field"][i_sample])[
                    "color"
                ]
                imageio.mimsave(
                    os.path.join(args.out_folder, f"sample_rf_{i_sample:02d}.mp4"),
                    video,
                    fps=30,
                )
            if args.mesh:
                video = render_utils.render_video(outputs["mesh"][i_sample])["normal"]
                imageio.mimsave(
                    os.path.join(args.out_folder, f"sample_mesh_{i_sample:02d}.mp4"),
                    video,
                    fps=30,
                )

                mesh = trimesh.Trimesh(
                    vertices=outputs["mesh"][i_sample].vertices.cpu().numpy(),
                    faces=outputs["mesh"][i_sample].faces.cpu().numpy(),
                )
                mesh.export(os.path.join(args.out_folder, f"sample_{i_sample:02d}.obj"))

                # GLB files can be extracted from the outputs
                try:
                    glb = postprocessing_utils.to_glb(
                        outputs["gaussian"][i_sample],
                        outputs["mesh"][i_sample],
                        # Optional parameters
                        simplify=0.95,  # Ratio of triangles to remove in the simplification process
                        texture_size=1024,  # Size of the texture used for the GLB
                        y_up=False,
                    )
                    glb.export(
                        os.path.join(args.out_folder, f"sample_{i_sample:02d}.glb")
                    )
                except Exception as e:
                    print("Failed to export GLB file!")

            # Save Gaussians as PLY files
            outputs["gaussian"][i_sample].save_ply(
                os.path.join(args.out_folder, f"sample_{i_sample:02d}.ply")
            )

            if not args.no_slat:
                slat: sp.SparseTensor = outputs["slat"][i_sample]
                cond: torch.Tensor = outputs["cond"]
                neg_cond: torch.Tensor = outputs["neg_cond"]
                z_s: torch.Tensor = outputs["z_s"][i_sample]
                save_slat_conds(
                    os.path.join(args.out_folder, f"slat_{i_sample:02d}.pt"),
                    slat,
                    cond,
                    neg_cond,
                    z_s,
                )
