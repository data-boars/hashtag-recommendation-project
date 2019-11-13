import argparse
import importlib
import json
import numpy as np
import time
import typing as t
import logging
from pathlib import Path

import pandas as pd
import yaml

from tweet_recommendations.method import Method

logger = logging.getLogger("Predictions")
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

file_hdlr = logging.FileHandler("log.log")
file_hdlr.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(file_hdlr)
logger.setLevel(logging.INFO)

KwargsType = t.Optional[t.Dict[str, t.Any]]
PREDICTIONS_LIMIT = 200


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Script for creating predictions of different methods for "
            "a particular dataset."
        )
    )
    parser.add_argument(
        "-n", "--name", type=str, required=True, help="Name of the method"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset path to use during predictions",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory which will contain predictions of the model",
    )
    parser.add_argument(
        "-c",
        "--class",
        dest="class_module",
        type=str,
        required=True,
        help="Class to be evaulated",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        required=False,
        help="Config of the method",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="retweet_count",
        help=(
            "Name of factor to use during stratified split, "
            "either 'retweet_count' or 'followers'"
        ),
    )

    args = parser.parse_args()
    return args


def load_model_from_path(
    class_module_path: str, **kwargs: KwargsType
) -> Method:
    module, a_class = class_module_path.rsplit(".", maxsplit=1)
    model = getattr(importlib.import_module(module), a_class)(**kwargs)
    return model


def extract_ground_truth_from_frame(
    frame: pd.DataFrame
) -> t.List[t.List[str]]:
    hashtags = frame["hashtags"]
    return [[tag["text"] for tag in a_row] for a_row in hashtags]


def time_exectuion(
    method: t.Callable[[t.Any], t.Any],
    *args: t.Sequence[t.Any],
    **kwargs: KwargsType
) -> t.Tuple[float, t.Any]:
    start = time.time()
    result = method(*args, **kwargs)
    end = time.time()
    return end - start, result


def make_predictions(
    name: str,
    dataset_path: str,
    output_dir: str,
    class_module_path: str,
    split_name: str,
    **method_kwargs: KwargsType
):
    output_dir = Path(output_dir)
    dataset_path = Path(dataset_path)

    assert output_dir.exists()
    assert dataset_path.exists()

    output_model_path = output_dir / name
    output_model_path.mkdir(exist_ok=True)

    model = load_model_from_path(class_module_path, **method_kwargs)

    train_val_test_splits = pd.read_pickle(
        dataset_path / "train_val_test_splits.pkl"
    )
    tweets = pd.read_pickle(dataset_path / "tweets.pkl")
    train_val_test_splits = train_val_test_splits[split_name]

    # validation split is left for the hyperparameter optimization
    train_tweets_ids = set(train_val_test_splits["train_user_ids"])
    test_tweets_ids = set(train_val_test_splits["test_user_ids"])

    train_tweets = tweets[tweets["user_id"].isin(train_tweets_ids)]
    test_tweets = tweets[tweets["user_id"].isin(test_tweets_ids)]

    if "lemmas" in train_tweets.columns:
        train_data = train_tweets["lemmas"].tolist()
        test_data = test_tweets["lemmas"].tolist()
    else:
        train_data = train_tweets["text"].tolist()
        test_data = test_tweets["text"].tolist()

    logger.info("Fitting time ...")
    fit_time, _ = time_exectuion(model.fit, x=train_tweets)

    logger.info("Predicting on training data ...")
    predict_train_time, train_predictions = time_exectuion(
        model.transform, x=train_data
    )

    logger.info("Predicting on test data ...")
    predict_test_time, test_predictions = time_exectuion(
        model.transform, x=test_data
    )

    if isinstance(train_predictions, np.ndarray):
        train_predictions = train_predictions[:, :PREDICTIONS_LIMIT].tolist()
        test_predictions = test_predictions[:, :PREDICTIONS_LIMIT].tolist()

    train_y_pred_true_frame = pd.DataFrame(
        data={
            "pred": train_predictions,
            "true": extract_ground_truth_from_frame(train_tweets),
            "tweet_id": train_tweets["id"],
        }
    )

    test_y_pred_true_frame = pd.DataFrame(
        data={
            "pred": test_predictions,
            "true": extract_ground_truth_from_frame(test_tweets),
            "tweet_id": test_tweets["id"],
        }
    )

    train_y_pred_true_frame.to_pickle(output_model_path / "train_preds.pkl")
    test_y_pred_true_frame.to_pickle(output_model_path / "test_preds.pkl")

    (output_model_path / "timings.json").write_text(
        json.dumps(
            {
                "fit": fit_time,
                "predicting_on_train": predict_train_time,
                "predicting_on_test": predict_test_time,
            },
            indent=2,
        )
    )


def main():
    args = get_args()

    config_path = args.config

    if config_path is not None and len(config_path) > 0:
        with open(config_path) as f:
            kwargs = yaml.safe_load(f)
    else:
        kwargs = {}

    make_predictions(
        args.name,
        args.dataset,
        args.output,
        args.class_module,
        args.split,
        **kwargs
    )


if __name__ == "__main__":
    main()
