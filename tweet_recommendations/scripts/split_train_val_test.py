import argparse

import sys
sys.path.insert(0, '')

import logging
import pandas as pd
import numpy as np
import pickle as pkl
from tweet_recommendations.data_processing.split_train_test import get_train_test_users



DEFAULT_TEST_PERCENTAGE = 0.2
DEFAULT_VAL_PERCENTAGE = 0.2

#VAL_TRAIN_RATIO = VAL_PERCENTAGE / (1 - TEST_PERCENTAGE)

DEFAULT_SEED = "0xCAFFE"


def main():
    parser = argparse.ArgumentParser("Unify data to easy-to-use and coherent form.\n"
                                     "Example execution: \n"
                                     "    $ python split_train_val_test.py "
                                     "--source_tweets './data/our/tweets.pkl' "
                                     "--source_users './data/our/users.pkl' "
                                     "--output_split_path './data/our/train_val_test_split.pkl' "
                                     "-t 0.2 -v 0.2 --seed '0xCAFFE'"
                                     "--verbose \n")
    parser.add_argument("--source_tweets", help="Path to source .pkl file containing tweets DataFrame.", required=True)
    parser.add_argument("--source_users", help="Path to source .pkl file containing users DataFrame.", required=True)
    parser.add_argument("--output_split_path", required=True,
                        help="Path for an output .pkl file with dict with splitting result.")
    parser.add_argument("-t", "--test_percentage", type=float,
                        help="Set test dataset percentage [0-1].", default=DEFAULT_TEST_PERCENTAGE)
    parser.add_argument("-v", "--val_percentage", type=float,
                        help="Set val dataset percentage [0-1].", default=DEFAULT_VAL_PERCENTAGE)
    parser.add_argument("--seed", help="Set random seed - string that can be `eval`-ed in python", default=DEFAULT_SEED)
    parser.add_argument("--verbose", help="Increase output verbosity", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    TEST_PERCENTAGE = args.test_percentage
    VAL_PERCENTAGE = args.val_percentage
    VAL_TRAIN_RATIO = VAL_PERCENTAGE / (1 - TEST_PERCENTAGE)
    SEED = eval(args.seed)

    logging.info("Reading source files...")
    tweets = pd.read_pickle(args.source_tweets)
    users = pd.read_pickle(args.source_users)

    logging.info("Source data successfully loaded.")
    logging.info("Starting splitting process")

    splits = {}
    for splitter in ['retweet_count', 'followers']:
        train_val_ids, test_ids = get_train_test_users(tweets, users,
                                                       split_on=splitter,
                                                       test_size=TEST_PERCENTAGE,
                                                       random_state=SEED)
        twdf = tweets[tweets['user_id'].isin(train_val_ids)]
        train_ids, val_ids = get_train_test_users(twdf, users,
                                                  split_on=splitter,
                                                  test_size=VAL_TRAIN_RATIO,
                                                  random_state=SEED)

        splits[splitter] = {"train_user_ids": train_ids.tolist(),
                            "val_user_ids": val_ids.tolist(),
                            "test_user_ids": test_ids.tolist()}

    logging.info("Splitting done.")

    logging.info("Saving...")

    with open(args.output_split_path, "wb") as f:
        pkl.dump(splits, f)

    logging.info("Done!")


if __name__ == '__main__':
    main()
