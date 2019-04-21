import argparse

import numpy as np
import pandas as pd

from tweet_recommendations.our_method import OurMethod

parser = argparse.ArgumentParser()
parser.add_argument("w2v_path")
args = parser.parse_args()

tweets_with_lemmas = pd.read_pickle("data/source_data/original_tweets_with_lemmas.p")

our = OurMethod(0.0, path_to_keyedvectors_model=args.w2v_path, popularity_measure='pagerank', verbose=True)
our.fit(tweets_with_lemmas, min_hashtag_count=10)

print("Predicting ... ")
print(our.transform([["wieźć", "szwagier", "na", "wybory"]]))
print(our.transform(["Wiozę szwagra na wybory"]))
print(our.transform([
    "Co by tu dzisiaj zrobić ...",
    "Wiozę szwagra na wybory"
]))
print(our.transform([
    ["co", "by", "tu", "dzisiaj", "zrobić", "..."],
    ["wieźć", "szwagier", "na", "wybory"]
]))
