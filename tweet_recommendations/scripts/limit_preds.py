import argparse
from pathlib import Path

import pandas as pd
import tqdm


def limit_from_frame(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    preds = frame.pred.tolist()
    new_preds = [row[:limit] for row in preds]
    frame.pred = new_preds
    return frame


def process_single(path_to_file: str, limit: int) -> None:
    frame = pd.read_pickle(path_to_file)
    frame = limit_from_frame(frame, limit)
    frame.to_pickle(path_to_file)


def limit_tags(path_to_file: str, limit: int, batch: bool) -> None:
    if batch:
        all_pickles = list(Path(path_to_file).rglob("*.pkl"))
        for pickle in tqdm.tqdm(all_pickles):
            if pickle.name in ["train_preds.pkl", "test_preds.pkl"]:
                process_single(pickle.as_posix(), limit)
    else:
        process_single(path_to_file, limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script that reduces number of predicted tags to save"
                    "memory."
    )

    parser.add_argument(
        "-f", "--file", help="Path of the file/folder with predictions"
    )
    parser.add_argument(
        "-l",
        "--limit",
        help="Number of maximum tags to store",
        default=200,
        type=int,
    )

    parser.add_argument(
        "-b",
        "--batch",
        help="Batch process all `train_preds` and `test_preds` files",
        action="store_true",
        default=True,
    )

    args = parser.parse_args()

    limit_tags(args.file, args.limit, args.batch)


if __name__ == "__main__":
    main()
