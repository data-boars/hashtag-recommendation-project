import pickle as pkl
import time
import numpy as np

from tweet_recommendations.other_methods.graph_summarization_for_hashtag_recommendation import GraphSummarizationMethod
from tweet_recommendations.other_methods.lemmatizer import TaggerToygerLemmatizer

lemmatizer = TaggerToygerLemmatizer("data/processed/lemmas.p")
the_graph = GraphSummarizationMethod(max_iterations=100, minimal_random_walk_change_difference_value=1e-4, verbose=True,
                                     damping_factor=0.8)

with open("data/source_data/original_tweets.p", "rb") as f:
    original_tweets = pkl.load(f)

tweets_with_lemmas = lemmatizer.fit_transform(original_tweets, verbose=True)
the_graph.fit(tweets_with_lemmas, minimal_hashtag_occurence=3)
# print(the_graph.transform(("Wiozę szwagra na wybory",))[0][:10])
# print(the_graph.transform(("Co by tu dzisiaj zrobić ...",))[0][:10])

times = []
for _ in range(10):
    now = time.time()
    print(the_graph.transform((
        ["co", "by", "tu", "dzisiaj", "zrobić"],
        ["wieźć", "szwagier", "na", "wybory"],
        ["co", "by", "tu", "dzisiaj", "zrobić"],
    ))[:, :10])
    times.append(time.time() - now)

print(np.mean(times), np.std(times))

times = []
for _ in range(10):
    now = time.time()
    print(the_graph.transform((
        "Co by tu dzisiaj zrobić ...",
        "Wiozę szwagra na wybory",
        "Co by tu dzisiaj zrobić ...",
    ))[:, :10])
    times.append(time.time() - now)
print(np.mean(times), np.std(times))
