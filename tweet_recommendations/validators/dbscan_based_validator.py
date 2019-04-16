import pandas as pd

from tweet_recommendations.embeddings.sif_embedding import SIFEmbedding
from tweet_recommendations.other_methods.dbscan_based_method import DBScanBasedEstimator

tweets_with_lemmas = pd.read_pickle("path/to/tweet_data")
sif = SIFEmbedding("path/to/kgr10.plain.skipgram.dim100.neg10.vec")

dbs = DBScanBasedEstimator(sif, verbose=True)
dbs.fit(tweets_with_lemmas, minimal_hashtag_occurence=10)

print(dbs.transform(["Wiozę", "szwagra", "na" "wybory"]))
print(dbs.transform("Co by tu dzisiaj zrobić ..."))
