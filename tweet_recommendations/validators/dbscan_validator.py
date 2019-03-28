import pickle as pkl

from tweet_recommendations.other_methods.lemmatizer import TaggerToygerLemmatizer
from tweet_recommendations.other_methods.dbscan_based_method import DBScanBasedEstimator
from tweet_recommendations.embeddings.models import FastText
from functools import partial

fast_text = partial(FastText(), fasttext_model=None)

lemmatizer = TaggerToygerLemmatizer("data/processed/lemmas.p")
the_graph = DBScanBasedEstimator()

with open("data/source_data/original_tweets.p", "rb") as f:
    original_tweets = pkl.load(f)

tweets_with_lemmas = lemmatizer.fit_transform(original_tweets, verbose=True)
the_graph.fit(tweets_with_lemmas, minimal_hashtag_occurence=3)
print(the_graph.transform("Wiozę szwagra na wybory")[:10])
