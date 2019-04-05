import pandas as pd
import numpy as np

from tweet_recommendations.embeddings.sif_embedding import SIFEmbedding
from tweet_recommendations.other_methods.dbscan_based_method import DBScanBasedEstimator

tweets_with_lemmas = pd.read_pickle("data/source_data/original_tweets_with_lemmas.p")
sif = SIFEmbedding("path_to/kgr10.plain.skipgram.dim100.neg10.vec")
tweet_embeddings = sif.fit_transform(tweets_with_lemmas, None)

sif_embeddings_as_array = [tweet_embeddings[i, :] for i in range(tweet_embeddings.shape[0])]

tweets_with_lemmas["embedding"] = np.nan
tweets_with_lemmas.embedding.astype(object)
tweets_with_lemmas["embedding"] = pd.Series(sif_embeddings_as_array)

dbs = DBScanBasedEstimator()
dbs.fit(tweets_with_lemmas, minimal_hashtag_occurence=10)

print(dbs.transform(np.array(tweets_with_lemmas[:5]["embedding"].to_list())))