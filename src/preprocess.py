import pandas as pd
from nlp.lang import translate_process_data
from tqdm import tqdm
from utils import read_config, get_config_section
import argparse
import translators as ts
from pathlib import Path

tqdm.pandas()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess job classification dataset"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="artifacts/processed/",
        help="Path to the output file",
    )
    return parser.parse_args()


def verify_config(cfg, args):
    if "dataset" not in cfg:
        raise ValueError("Missing dataset section in the configuration file")
    if "source" not in cfg["dataset"]:
        raise ValueError(
            "Missing source file in the dataset section of the configuration file"
        )
    if not Path(args.output).parent.exists():
        raise ValueError("Output directory does not exist")
    Path(args.output).mkdir(exist_ok=True)


def main():
    args = parse_args()
    cfg = read_config(args.config)
    dataset_cfg = get_config_section(cfg, "dataset")
    verify_config(cfg, args)
    df_dataset = pd.read_csv(dataset_cfg["source"])
    df_dataset = translate_process_data(df_dataset)

    output_dir = Path(args.output)
    df_dataset.to_csv(output_dir / "wi_dataset_processed_en.csv", index=False)
    out_columns = ["id", "title", "lang", "description", "final_description"]
    df_dataset[out_columns].to_csv(
        output_dir / "wi_dataset_processed_multilingual.csv", index=False
    )


if __name__ == "__main__":
    main()
