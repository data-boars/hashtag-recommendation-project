import argparse

import pandas as pd

from tweet_recommendations.other_methods.dbscan_based_method import DBScanBasedEstimator

parser = argparse.ArgumentParser()
parser.add_argument("w2v_path")
args = parser.parse_args()

tweets_with_lemmas = pd.read_pickle("data/source_data/original_tweets_with_lemmas.p")

dbs = DBScanBasedEstimator(args.w2v_path, verbose=True)
dbs.fit(tweets_with_lemmas, minimal_hashtag_occurence=10)

print("Predicting ... ")
print(dbs.transform([["wieźć", "szwagier", "na", "wybory"]]))
print(dbs.transform(["Wiozę szwagra na wybory"]))
print(dbs.transform([
    "Co by tu dzisiaj zrobić ...",
    "Wiozę szwagra na wybory"
]))
print(dbs.transform([
    ["co", "by", "tu", "dzisiaj", "zrobić", "..."],
    ["wieźć", "szwagier", "na", "wybory"]
]))
