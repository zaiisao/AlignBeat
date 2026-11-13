import argparse
from pathlib import Path

import torch


def main(args):
    if Path(args.output_path).exists():
        print(f"Output path {args.output_path} already exists. Exiting.")
        return

    checkpoint = torch.load(args.input_path, map_location="cpu")

    # Keep model and datamodule metadata; drop optimizer and trainer state.
    checkpoint = {
        k: v
        for k, v in checkpoint.items()
        if k
        in [
            "state_dict",
            "datamodule_hyper_parameters",
            "hyper_parameters",
            "pytorch-lightning_version",
        ]
    }

    # Remove the local data path, which may not exist on another machine.
    if "data_dir" in checkpoint["datamodule_hyper_parameters"]:
        del checkpoint["datamodule_hyper_parameters"]["data_dir"]

    torch.save(checkpoint, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()

    main(args)
