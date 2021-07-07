import argparse

import pandas as pd

from tweet_recommendations.other_methods import Tweet2VecFeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
args = parser.parse_args()

tweets_with_lemmas = pd.read_pickle(
    "data/source_data/original_tweets_with_lemmas.p"
)

model = Tweet2VecFeatureExtractor(
    model_path=args.model_path, verbose=True, last_epoch=-1
)

print("Predicting ... ")
print(model.transform([["wieźć", "szwagier", "na", "wybory"]])[0][:10])
print(model.transform(["Wiozę szwagra na wybory"])[0][:10])
print(
    [
        res[:10]
        for res in model.transform(
            ["Co by tu dzisiaj zrobić ...", "Wiozę szwagra na wybory"]
        )
    ]
)
print(
    [
        res[:10]
        for res in model.transform(
            [
                ["co", "by", "tu", "dzisiaj", "zrobić", "..."],
                ["wieźć", "szwagier", "na", "wybory"],
            ]
        )
    ]
)
