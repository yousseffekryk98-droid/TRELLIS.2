#!/usr/bin/env python3
"""Export a single image-to-3D Ninjago fighter mesh as a game-ready GLB.

Requires the TRELLIS.2 dependencies, NVIDIA CUDA GPU and model weights.
Produces an unrigged textured mesh; the game supplies gameplay collision.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Licensed transparent or isolated character reference")
    parser.add_argument("--character-id", required=True, help="Exact fighter id, e.g. kai-tournament")
    parser.add_argument("--output-dir", type=Path, required=True, help="Game's public/assets/models/fighters/trellis2 directory")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline-type", choices=("512", "1024", "1024_cascade", "1536_cascade"), default="512")
    parser.add_argument("--faces", type=int, default=45000, help="Target game mesh face count")
    parser.add_argument("--texture-size", type=int, default=1024, help="Export atlas width")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import re
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.character_id):
        raise SystemExit("Character ID must be lowercase letters/numbers separated by hyphens.")
    if not args.image.is_file():
        raise SystemExit(f"Reference image missing: {args.image}")
    if args.faces < 1000 or args.texture_size < 256:
        raise SystemExit("Use --faces >= 1000 and --texture-size >= 256")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("TRELLIS.2 generation needs NVIDIA CUDA; run this on a supported Linux GPU workstation.")

    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    print(f"Loading TRELLIS.2 pipeline: {args.model}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    with Image.open(args.image) as raw:
        image = raw.convert("RGBA")
    print(f"Generating {args.character_id} using {args.pipeline_type}...")
    with torch.inference_mode():
        mesh = pipeline.run(image, seed=args.seed, pipeline_type=args.pipeline_type)[0]

    # The final GLB is optimized for Three.js/Capacitor rather than
    # keeping the multi-million face high-resolution research output.
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.faces,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.character_id}.glb"
    glb.export(str(destination), extension_webp=True)
    print(f"Created {destination} ({destination.stat().st_size:,} bytes)")
    print("Review the silhouette, orientation, materials and license. This mesh is not rigged.")
    print("Then run npm run sync:trellis in the game repository.")


if __name__ == "__main__":
    main()
