import argparse
import datetime
import json
from pathlib import Path

from helpers import save_useful_info
from train import train
import torch


def train_zeggs(config, device=torch.device("cuda:0")):
    train_options = config["train_opt"]
    network_options = config["net_opt"]
    paths = config["paths"]

    base_path = Path(paths["base_path"])
    path_processed_data = base_path / paths["path_processed_data"] / "processed_data.npz"
    path_data_definition = base_path / paths["path_processed_data"] / "data_definition.json"

    # Output directory
    if paths["output_dir"] is None:
        output_dir = (base_path / "outputs") / datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_dir.mkdir(exist_ok=True, parents=True)
        paths["output_dir"] = str(output_dir)
    else:
        output_dir = Path(paths["output_dir"])

    # Path to models
    if paths["models_dir"] is None and not train_options["resume"]:
        models_dir = output_dir / "saved_models"
        models_dir.mkdir(exist_ok=True)
        paths["models_dir"] = str(models_dir)
    else:
        models_dir = Path(paths["models_dir"])

    # Log directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    config["paths"] = paths
    with open(output_dir / 'options.json', 'w') as fp:
        json.dump(config, fp, indent=4)

    save_useful_info(output_dir)

    train(
        models_dir=models_dir,
        logs_dir=logs_dir,
        path_processed_data=path_processed_data,
        path_data_definition=path_data_definition,
        train_options=train_options,
        network_options=network_options,
        device=device
    )


if __name__ == "__main__":
    """
    python main.py -o "../configs/configs_v1.json" -n "zeggs_v1"
    """
    # Setting parser
    parser = argparse.ArgumentParser(description="Train ZEGGS Network.")

    # Hparams
    parser.add_argument("-o", "--options", type=str, help="Options filename")
    parser.add_argument('-n', '--name', type=str, help="Name", required=False)
    parser.add_argument("-gpu", "--gpu", type=str, default="cuda:0", required=True, help="GPU")

    args = parser.parse_args()

    with open(args.options, "r") as f:
        options = json.load(f)
    if args.name:
        options["name"] = args.name

    device = torch.device(args.gpu)

    train_zeggs(config=options, device=device)
