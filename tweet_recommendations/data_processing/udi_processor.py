import argparse
import os
import sys
from collections import deque
from functools import partial
from multiprocessing import Pool, Queue
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import pandas as pd
from tqdm import tqdm

MERGED_FILENAME = "UDI-TwitterCrawl-Aug2012.csv"
TWEET_SEPARATOR = "***"
VALUE_SEPARATOR = ":"
MULTILINE_FIELDS = ["Text", "Origin"]
SINGLE_FIELDS = [
    "Type",
    "URL",
    "ID",
    "Time",
    "RetCount",
    "Favorite",
    "MentionedEntities",
    "Hashtags",
]


def udi_parse_dataset(
    dataset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    verbose: bool = False,
    minimal_hashtags: int = 0,
    parallelism: int = 2,
    merge: bool = False,
) -> None:
    """
    Parse whole dataset in UDI format, save results as structured DataFrames

    :param dataset_dir: directory containing data files
    :param output_dir: directory designated for processed files
    :param verbose: enables console outputs, defaults to False
    :param minimal_hashtags: minimal number of hashtags tweet has to have to be saved
    :param parallelism: number of concurrent jobs
    :param merge: save single DataFrame (merge everything)
    """
    parallelism = max(1, parallelism)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    dataset_filenames = os.listdir(dataset_dir)
    queue = Queue() if merge else None

    with Pool(parallelism, _init_parse_and_save, (queue,)) as pool:
        job = pool.imap(
            partial(
                _parse_and_save,
                input_dir=dataset_dir,
                output_dir=output_dir,
                minimal_hashtags=minimal_hashtags,
            ),
            dataset_filenames,
        )
        if verbose:
            job = tqdm(job, total=len(dataset_filenames))

        # Consuming an iterator by feeding it into 0-length deque
        # https://docs.python.org/3/library/itertools.html#itertools-recipes
        deque(job, maxlen=0)

    if merge:
        _merge_and_save(queue, output_dir, verbose)


def _merge_and_save(queue: Queue, output_dir: Path, verbose: bool = False):
    df = pd.DataFrame()

    if verbose:
        print("Merging dataframes...")
    while not queue.empty():
        df = pd.concat((df, queue.get()), ignore_index=True)

    if verbose:
        print("Saving merged dataframe")
    output_path = output_dir / MERGED_FILENAME
    df.to_csv(output_path)


def _parse_and_save(
    input_filename: str,
    input_dir: Path,
    output_dir: Optional[Path],
    minimal_hashtags: int = 0,
) -> None:
    """
    Parse and save single file to CSV or store in a queue for merging.
    This is an entrypoint for concurrent subprocesses.

    :param input_filename: name of processed file
    :param input_dir: directory containing data files
    :param output_dir: directory containing output files
    :param minimal_hashtags: minimal hashtags for a single tweet to be saved
    """
    # `queue` will be initialized when Pool is created
    queue = _parse_and_save.queue

    absolute_input_path = input_dir / input_filename
    output_filename = Path(f"{input_filename}.csv")
    absolute_output_path = output_dir / output_filename

    try:
        # Parse only if not already parsed, or in merge mode
        if queue or not absolute_output_path.exists():
            single_dataframe = _parse_file(
                absolute_input_path, minimal_hashtags=minimal_hashtags
            )

        if queue:
            queue.put(single_dataframe)
        elif not single_dataframe.empty:
            single_dataframe.to_csv(absolute_output_path, index=False)
    except KeyboardInterrupt:
        raise
    except Exception:
        print(absolute_output_path)
        print(sys.exc_info())


def _init_parse_and_save(queue: Optional[Queue]) -> None:
    """
    Prepare `_parse_and_save` function to use `multiprocessing.Queue`.
    This function should not be called manually, it is an initializer
    for `multiprocessing.Pool`

    :param queue: shared queue
    """
    _parse_and_save.queue = queue


def _parse_file(filename: str, minimal_hashtags: int = 0) -> pd.DataFrame:
    """
    Parse single file from UDI Dataset and return structured DataFrame

    :param filename: path to parsed file
    :param minimal_hashtags: minimal hashtags for tweet to be saved
    :return: structured DataFrame
    """
    with open(filename, "r", encoding="utf-8") as tweets_file:
        file_lines = tweets_file.readlines()

    tweets = []
    current_tweet = {}
    current_multiline_field = None
    for line in file_lines:
        if line.strip() == TWEET_SEPARATOR:
            if not current_tweet:
                current_tweet = {}
            else:
                if minimal_hashtags > 0 and _has_hashtags(
                    current_tweet, minimal_hashtags
                ):
                    tweets.append(current_tweet)
                current_tweet = {}
        else:
            splitted_line = line.split(VALUE_SEPARATOR)
            line_beginning = splitted_line[0]
            if line_beginning in MULTILINE_FIELDS:
                # Handling fields that can span multiple lines
                key, text = _parse_single_line(splitted_line)
                current_tweet[key] = text
                current_multiline_field = key
            elif line_beginning in SINGLE_FIELDS:
                # Handling fields known to be only single-line
                key, text = _parse_single_line(splitted_line)
                current_tweet[key] = text
            else:
                # Handling continued multiline text
                current_tweet[current_multiline_field] = (
                    current_tweet[current_multiline_field] + "\n" + line
                )

    return pd.DataFrame(tweets)


def _parse_single_line(splitted_line: str) -> Tuple[str, str]:
    """
    Parse single-lined fields

    :param splitted_line: single line of text in format `Key: value`
    :return: parsed line as tuple (key, value)
    """
    key = splitted_line[0]

    splitted_values = splitted_line[1:]
    value = VALUE_SEPARATOR.join(splitted_values).strip()
    value = _parse_value(key, value)

    return key, value


def _parse_value(key: str, value: str) -> Union[List, bool, int, str]:
    """
    Parses values for single-lined fields.

    Specifically:
    - build lists of hashtags and mentions from raw text
    - convert "Favorite" field into proper boolean
    - convert retweet count to number
    Everything else is left as-is.

    :param key: key of current field
    :param value: value assigned to the field
    :return: parsed value, according to above rules
    """
    if key in ("Hashtags", "MentionedEntities"):
        parsed_value = value.strip()
        if not parsed_value:
            parsed_value = []
        else:
            parsed_value = parsed_value.split(" ")
    elif key == "Favorite":
        parsed_value = bool(value)
    elif key == "RetCount":
        parsed_value = int(value)
    else:
        parsed_value = value
    return parsed_value


def _has_hashtags(tweet: Dict, minimal_hashtags: int = 1) -> bool:
    return "Hashtags" in tweet and len(tweet["Hashtags"]) >= minimal_hashtags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse UDI Dataset and save results as pickled DataFrames."
    )
    parser.add_argument("input_path", type=str, help="Dataset directory")
    parser.add_argument(
        "output_path", type=str, help="Target directory for processed dataset"
    )
    parser.add_argument(
        "--min-hashtags",
        type=int,
        default=0,
        help="Minimal number of hashtags for tweet to be saved (discards all others)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--parallelism",
        "-p",
        type=int,
        default=2,
        help="Number of concurrent jobs for dataset parsing",
    )
    parser.add_argument(
        "--merge",
        "-m",
        action="store_true",
        help="Merge all DataFrames into one, instead of saving one per single user",
    )

    args = parser.parse_args()

    udi_parse_dataset(
        args.input_path,
        args.output_path,
        verbose=args.verbose,
        minimal_hashtags=args.min_hashtags,
        parallelism=args.parallelism,
        merge=args.merge,
    )
