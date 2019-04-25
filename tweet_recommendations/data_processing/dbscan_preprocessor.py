import argparse
from pathlib import Path

import dask.dataframe as dd
import dask.multiprocessing
import spacy

dask.config.set(scheduler="processes")

SPACY_MODEL_NAME = "en_core_web_md"
SPACY_DISABLED_PIPELINE = ["parser", "ner"]
SPACY_STEM_ONLY_DISABLED_PIPELINE = ["parser", "ner", "tagger"]
SPACE = " "


def preprocess_dataset(
    dataset_path: str,
    target_path: str,
    verbose: bool = False,
    column: str = "Text",
    stem_only: bool = False,
) -> None:
    """
    Preprocess dataset:
    - lemmatize tokens in tweets
    - remove out-of-vocabulary words

    :param dataset_path: Path to dataset directory with *.csv files
    :param target_path: Target directory for preprocessed dataset
    :param verbose: show `dask`'s progress bar
    :param column: column name containing text in CSV file
    """
    if stem_only:
        disabled_components = SPACY_STEM_ONLY_DISABLED_PIPELINE
    else:
        disabled_components = SPACY_DISABLED_PIPELINE

    if verbose:
        print(
            f"Loading model {SPACY_MODEL_NAME}, disabled components: {disabled_components}"
        )

    nlp = spacy.load(SPACY_MODEL_NAME, disable=disabled_components)
    dataset_path = Path(dataset_path)

    def clear_tweet(row):
        raw_text = str(row[column])
        analyzed_text = nlp(raw_text)
        processed_tokens = []

        for token in analyzed_text:
            if not token.is_oov:
                processed_tokens.append(token.lemma_)

        row[column] = SPACE.join(processed_tokens)
        return row

    if verbose:
        print(f"Reading input file from {dataset_path}")

    dataset = dd.read_csv(dataset_path)

    if verbose:
        from dask.diagnostics import ProgressBar

        ProgressBar().register()
        print("Beginning preprocessing")

    dataset = dataset.apply(clear_tweet, axis=1).compute()

    if verbose:
        print(f"Saving preprocessed dataset as {target_path}")

    dataset.to_csv(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse UDI Dataset and save results as pickled DataFrames."
    )
    parser.add_argument("input_path", type=str, help="Dataset directory")
    parser.add_argument("output_path", type=str, help="Target directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--column",
        "-c",
        type=str,
        help=("Column containing text in CSV file, Default: Text (as in UDI dataset)"),
        default="Text",
    )
    parser.add_argument(
        "--stem-only",
        "-s",
        action="store_true",
        help=(
            "Disables tagging module, resulting"
            "in stemming instead of lemmatization process"
        ),
    )

    args = parser.parse_args()

    preprocess_dataset(
        args.input_path, args.output_path, args.verbose, args.column, args.stem_only
    )
