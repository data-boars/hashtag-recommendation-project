import pickle as pkl
import time

import numpy as np

from tweet_recommendations.other_methods.lemmatizer import (
    TaggerToygerLemmatizer
)
from tweet_recommendations.other_methods.tweet2vec_method import Tweet2Vec

lemmatizer = TaggerToygerLemmatizer("data/processed/lemmas.p")
the_graph = Tweet2Vec(10, "models/init", True, -1)

with open("data/source_data/original_tweets.p", "rb") as f:
    original_tweets = pkl.load(f)

tweets_with_lemmas = lemmatizer.fit_transform(original_tweets, verbose=True)
the_graph.fit(tweets_with_lemmas)

times = []
for _ in range(10):
    now = time.time()
    print(
        the_graph.transform(
            (
                ["co", "by", "tu", "dzisiaj", "zrobić"],
                ["wieźć", "szwagier", "na", "wybory"],
                ["co", "by", "tu", "dzisiaj", "zrobić"],
            )
        )[:, :10]
    )
    times.append(time.time() - now)

print(np.mean(times), np.std(times))

times = []
for _ in range(10):
    now = time.time()
    print(
        the_graph.transform(
            (
                "Co by tu dzisiaj zrobić ...",
                "Wiozę szwagra na wybory",
                "Co by tu dzisiaj zrobić ...",
            )
        )[:, :10]
    )
    times.append(time.time() - now)
print(np.mean(times), np.std(times))
