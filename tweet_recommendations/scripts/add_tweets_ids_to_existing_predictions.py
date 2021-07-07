"""Adds an id column for each prediction row in the experiments' data.

The script is a leftover after unfinished script of the experimentation runner.
Even the data rows weren't shuffled, it was still ambiguous for others
how each row corresponds to the original dataset.

Example usage:
$ python tweet_recommendations/scripts/add_tweets_ids_to_existing_predictoins.py \
> --input experiments/<experiment_name> \
> --dataset data/processed/<dataset_name_ex_udi> \
> --split_type (retweet_count|followers)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_to_frame(
    experiments_frame: pd.DataFrame, dataset: pd.DataFrame
) -> pd.DataFrame:
    experiments_frame["tweet_id"] = dataset["id"].to_numpy(dtype=np.int64)
    return experiments_frame


def add_id_column(
    experiment_folder: Path, dataset_path: Path, split_type: str
):
    train_preds_path = experiment_folder / "train_preds.pkl"
    test_preds_path = experiment_folder / "test_preds.pkl"

    train_val_test_splits = pd.read_pickle(
        dataset_path / "train_val_test_splits.pkl"
    )
    tweets = pd.read_pickle(dataset_path / "tweets.pkl")
    train_val_test_splits = train_val_test_splits[split_type]

    # validation split is left for the hyperparameter optimization
    train_tweets_ids = set(train_val_test_splits["train_user_ids"])
    test_tweets_ids = set(train_val_test_splits["test_user_ids"])

    train_tweets = tweets[tweets["user_id"].isin(train_tweets_ids)]
    test_tweets = tweets[tweets["user_id"].isin(test_tweets_ids)]

    add_to_frame(pd.read_pickle(train_preds_path), train_tweets).to_pickle(
        train_preds_path
    )
    add_to_frame(pd.read_pickle(test_preds_path), test_tweets).to_pickle(
        test_preds_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "The script adds tweet ids to each row in the predictions table "
            "to be sure that the mapping is not ambiuguous."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="input_folder",
        help="Path to the experiment's folder",
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset_path",
        help="Path the experiment path",
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--split",
        dest="split_type",
        help="Name of the split used in the original experiment",
    )

    args = parser.parse_args()

    add_id_column(args.input_folder, args.dataset_path, args.split_type)


if __name__ == "__main__":
    main()
