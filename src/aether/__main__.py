import argparse
import sys
from pathlib import Path
from aether.config import load_config
from aether.pipeline import NebulaPipeline
from aether.io.dataset import generate_synthetic_nebula, download_and_extract
from aether.io.image import save_image

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aether",
        description="Aether: 3D Nebula Reconstruction and Visualization from 2D Images",
    )
    subparsers = parser.add_subparsers(dest="command")

    vis_parser = subparsers.add_parser("visualize", help="Reconstruct and display nebula in 3D")
    vis_parser.add_argument("--image", type=str, help="Path to nebula image")
    vis_parser.add_argument("--sample", action="store_true", help="Use a generated synthetic nebula")
    vis_parser.add_argument("--rep", choices=["point_cloud", "mesh", "volume"], default=None)
    vis_parser.add_argument("--model", choices=["hybrid", "astronomical", "midas_small", "dpt_hybrid", "dpt_large"], default=None)
    vis_parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    vis_parser.add_argument("--scale", type=float, default=None)
    vis_parser.add_argument("--config", type=str, default=None)
    vis_parser.add_argument("--spectral", action="store_true", help="Enable spectral emission line decomposition")
    vis_parser.add_argument("--spectral-mode", choices=["natural_rgb", "hubble_sho", "hoo_bicolor"], default=None)
    vis_parser.add_argument("--deproject", action="store_true", help="Enable true 3D morphological deprojection")
    vis_parser.add_argument("--morphology", choices=["bipolar", "toroidal", "ellipsoidal"], default=None)
    vis_parser.add_argument("--inclination", type=float, default=None)
    vis_parser.add_argument("--pa", type=float, default=None)
    vis_parser.add_argument("--cinematic", type=str, default=None)
    vis_parser.add_argument("--cinematic-mode", choices=["orbit", "fly_through"], default="orbit")
    vis_parser.add_argument("--export", type=str, default=None)
    vis_parser.add_argument("--screenshot", type=str, default=None)
    vis_parser.add_argument("--headless", action="store_true", help="Off-screen rendering mode")

    sample_parser = subparsers.add_parser("generate-sample", help="Generate synthetic nebula test image")
    sample_parser.add_argument("--output", type=str, default="data/samples/sample_nebula.png")
    sample_parser.add_argument("--width", type=int, default=512)
    sample_parser.add_argument("--height", type=int, default=512)
    sample_parser.add_argument("--seed", type=int, default=42)

    export_parser = subparsers.add_parser("export", help="Export 3D geometry to file")
    export_parser.add_argument("--image", type=str, required=True)
    export_parser.add_argument("--output", type=str, required=True)
    export_parser.add_argument("--rep", choices=["point_cloud", "mesh", "volume"], default="point_cloud")
    export_parser.add_argument("--model", choices=["hybrid", "astronomical", "midas_small", "dpt_hybrid", "dpt_large"], default="hybrid")
    export_parser.add_argument("--spectral", action="store_true", help="Enable spectral emission line decomposition")
    export_parser.add_argument("--spectral-mode", choices=["natural_rgb", "hubble_sho", "hoo_bicolor"], default=None)
    export_parser.add_argument("--deproject", action="store_true", help="Enable true 3D morphological deprojection")
    export_parser.add_argument("--morphology", choices=["bipolar", "toroidal", "ellipsoidal"], default=None)
    export_parser.add_argument("--inclination", type=float, default=None)
    export_parser.add_argument("--pa", type=float, default=None)
    export_parser.add_argument("--scale", type=float, default=None)
    export_parser.add_argument("--config", type=str, default=None)

    dl_parser = subparsers.add_parser("download", help="Download dataset")
    dl_group = dl_parser.add_mutually_exclusive_group(required=True)
    dl_group.add_argument("--url", type=str)
    dl_group.add_argument("--kagglehub", type=str)
    dl_parser.add_argument("--dest", type=str, default="data/dataset")

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "generate-sample":
        arr = generate_synthetic_nebula(width=args.width, height=args.height, seed=args.seed)
        save_image(args.output, arr)
        print(f"Sample nebula written to {args.output}")
        return

    if args.command == "download":
        download_and_extract(args.url, args.kagglehub, Path(args.dest))
        return

    if args.command in ("visualize", "export"):
        config = load_config(args.config)

        if getattr(args, "device", None):
            config.device = args.device
        if getattr(args, "model", None):
            config.depth.model_type = args.model
        if getattr(args, "rep", None):
            config.geometry.representation = args.rep
        if getattr(args, "scale", None):
            config.geometry.point_cloud.depth_scale = args.scale
            config.geometry.mesh.depth_scale = args.scale
        if getattr(args, "spectral", False):
            config.spectral.enabled = True
        if getattr(args, "spectral_mode", None):
            config.spectral.mode = args.spectral_mode
        if getattr(args, "deproject", False):
            config.morphology.enabled = True
        if getattr(args, "morphology", None):
            config.morphology.model_type = args.morphology
        if getattr(args, "inclination", None) is not None:
            config.morphology.inclination_deg = args.inclination
        if getattr(args, "pa", None) is not None:
            config.morphology.position_angle_deg = args.pa

        pipeline = NebulaPipeline(config)

        if getattr(args, "sample", False):
            image_data = generate_synthetic_nebula()
        else:
            if not getattr(args, "image", None):
                sample_path = Path("data/samples/sample_nebula.png")
                if not sample_path.exists():
                    arr = generate_synthetic_nebula()
                    save_image(sample_path, arr)
                image_data = str(sample_path)
            else:
                image_data = args.image

        if args.command == "export":
            result = pipeline.process(
                image_data,
                representation=args.rep,
                spectral=config.spectral.enabled,
                deproject=config.morphology.enabled,
            )
            pipeline.export(result, args.output)
            print(f"Exported {args.rep} to {args.output}")
            return

        is_headless = args.headless or bool(args.screenshot) or bool(getattr(args, "cinematic", None))
        pipeline.visualizer.off_screen = is_headless

        result = pipeline.process(
            image_data,
            representation=args.rep,
            spectral=config.spectral.enabled,
            deproject=config.morphology.enabled,
        )
        
        if args.export:
            pipeline.export(result, args.export)
            print(f"Exported geometry to {args.export}")

        if getattr(args, "cinematic", None):
            pipeline.render_cinematic(result, args.cinematic, mode=args.cinematic_mode)
            print(f"Cinematic animation saved to {args.cinematic}")
            return

        pipeline.render(
            result,
            show=not is_headless,
            save_path=args.screenshot,
        )
        if args.screenshot:
            print(f"Screenshot saved to {args.screenshot}")

if __name__ == "__main__":
    main()