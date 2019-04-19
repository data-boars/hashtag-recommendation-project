import argparse
from pathlib import Path

import dask.dataframe as dd
import spacy

SPACY_MODEL_NAME = "en_core_web_md"
SPACE = " "


def preprocess_dataset(
    dataset_path: str, target_path: str, verbose: bool = False
) -> None:
    """
    Preprocess dataset:
    - lemmatize tokens in tweets
    - remove out-of-vocabulary words

    :param dataset_path: Path to dataset directory with *.csv files
    :param target_path: Target directory for preprocessed dataset
    :param verbose: show `dask`'s progress bar
    """
    nlp = spacy.load(SPACY_MODEL_NAME)
    dataset_path = Path(dataset_path) / "*.csv"

    def clear_tweet(row):
        raw_text = str(row["Text"])
        analyzed_text = nlp(raw_text)
        processed_tokens = []

        for token in analyzed_text:
            if not token.is_oov:
                processed_tokens.append(token.lemma_)

        row["Text"] = SPACE.join(processed_tokens)
        return row

    if verbose:
        from dask.diagnostics import ProgressBar

        ProgressBar().register()

    dataset = dd.read_csv(dataset_path)
    dataset = dataset.apply(clear_tweet, axis=1).compute()
    dataset.to_csv(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse UDI Dataset and save results as pickled DataFrames."
    )
    parser.add_argument("input_path", type=str, help="Dataset directory")
    parser.add_argument("output_path", type=str, help="Target directory")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    preprocess_dataset(args.input_path, args.output_path, args.verbose)
