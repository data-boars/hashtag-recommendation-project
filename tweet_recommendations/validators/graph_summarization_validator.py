import pickle as pkl

from tweet_recommendations.other_methods.graph_summarization_for_hashtag_recommendation import GraphSummarizationMethod
from tweet_recommendations.other_methods.lemmatizer import TaggerToygerLemmatizer

lemmatizer = TaggerToygerLemmatizer("data/processed/lemmas.p")
the_graph = GraphSummarizationMethod()

with open("data/source_data/original_tweets.p", "rb") as f:
    original_tweets = pkl.load(f)

tweets_with_lemmas = lemmatizer.fit_transform(original_tweets, verbose=True)
the_graph.fit(tweets_with_lemmas, verbose=True, damping_factor=0.7, iterations=10, minimal_hashtag_occurence=3)
print(the_graph.transform("Wiozę szwagra na wybory")[:10])
