import argparse
import logging
import pandas as pd


def main():
    parser = argparse.ArgumentParser("Unify data to easy-to-use and coherent form.\n"
                                     "Example execution: \n"
                                     "    $ python unify_data.py "
                                     "--source_path './data/municipal_elections_base_dataset.p' "
                                     "--output_users_path './data/our/users.pkl' "
                                     "--output_tweets_path './data/our/tweets.pkl' "
                                     "--verbose \n")
    parser.add_argument("--source_path", help="Path to source .pkl file containing dict with users and tweets DataFrames")
    parser.add_argument("--output_users_path", help="Path for an output .pkl file containing users DF.")
    parser.add_argument("--output_tweets_path", help="Path for an output .pkl file containing tweets DF.")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading source file...")

    data = pd.read_pickle(args.source_path)
    users = data["users_data"]
    tweets = data["tweets_data"]

    logging.info("Source data successfully loaded.")
    logging.info("Starting processing tweets.")

    tweets["user_id"] = tweets["user_id"].astype(int)
    tweets["hashtags_list"] = tweets["hashtags"].apply(lambda htags: [x["text"] for x in htags])

    logging.info("Tweets have been processed.")
    logging.info("Starting processing users.")

    users = users[users["user_id"].str.isdigit()].copy()  # filtering out some corrupted rows
    users["user_id"] = users["user_id"].astype(int)
    users["followers"] = users["followers"].astype(float)

    logging.info("Users have been processed.")
    logging.info("Saving...")

    users.to_pickle(args.output_users_path)
    tweets.to_pickle(args.output_tweets_path)

    logging.info("Done!")


if __name__ == '__main__':
    main()
