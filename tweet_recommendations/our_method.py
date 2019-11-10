import enum
from typing import Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import tqdm
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

from tweet_recommendations.method import Method
from tweet_recommendations.utils.clients import get_wcrft2_results_for_text


class PopularityMeasure(enum.Enum):
    PageRank = "pagerank"
    MeanRetweets = "mean_retweets"


class OurMethod(Method):
    def __init__(
        self,
        popularity_to_similarity_ratio: float,
        popularity_measure: PopularityMeasure = "pagerank",
        path_to_keyedvectors_model: Optional[str] = None,
        embedding_name: Optional[str] = "embedding",
        personalised_pagerank: bool = True,
        weighted_pagerank: bool = True,
        verbose: bool = False,
    ):
        """
        Our proposed hashtag recommendation method.

        :param popularity_to_similarity_ratio: This ratio decides which is
                more important: semantic similarity vs hashtag popularity.
                0 means that only similarity matters,
                1 means that only popularity matters,
                0.5 means that both of the above should matter the same.
        :param popularity_measure: Popularity measure to be used.
                Available measures are: `pagerank` and `mean_retweets`.
        :param path_to_keyedvectors_model: Optional argument. Path to `gensim`
                model converted by script `convert_embedding_model_to_mmap.py`.
                It can be either word2vec or fasttext, `gensim` handles both.
                When path isn't set, `embedding_name` argument is required.
                Also `fit` and `transform` won't accept text input and will
                have to contain column named the same as `embedding_name`.
                You will have to provide precomputed embeddings.
        :param embedding_name: Optional, name of used embedding method.
                Required when `path_to_keyedvectors_model` is not provided.
                Input DataFrame in `fit` method will have to contain column
                named the same as value provided to this argument.
                If `path_to_keyedvectors_model` is provided `embedding_name`
                is ignored and is just used as a name.
        :param personalised_pagerank: Whether to use tweet retweets and hashtag
                mean retweets as personalisation vector in PageRank algorithm.
                Default is `True`.
        :param weighted_pagerank: Whether to use tweet-hashtag embedding
                similarity as weights in PageRank algorithm. Default is `True`.
        :param verbose: Whether method should be verbose.
        """

        assert (
            0 <= popularity_to_similarity_ratio <= 1
        ), "Ratio should be bound between 0 and 1"
        self.ratio = popularity_to_similarity_ratio
        self.popularity_measure = PopularityMeasure(popularity_measure).value
        self.embedding_name = embedding_name

        self._weighted_pagerank = weighted_pagerank
        self._personalised_pagerank = personalised_pagerank

        self.embedding_model = None
        if path_to_keyedvectors_model is not None:
            self.embedding_model = KeyedVectors.load(
                path_to_keyedvectors_model, mmap="r"
            )
        self.verbose = verbose

        self._G: nx.Graph = None
        self._hashtags_df: pd.DataFrame = None

    def fit(
        self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params
    ):

        min_hashtag_count = fit_params.get("min_hashtag_count", 10)

        if self.embedding_model is None:
            assert (
                self.embedding_name in x.columns
            ), "When no embedding model provided, input DataFrame should " f"contain column named same as `{self.embedding_name}`."
        else:
            assert (
                "lemmas" in x.columns
            ), "When embedding model is provided input DataFrame should " 'contain column "lemmas".'

        x = x[x["hashtags"].str.len() > 0]
        if self.verbose:
            print("Removing tweets with too rare hashtags.")
        x = self.drop_tweets_with_hashtags_that_occurred_less_than(
            x, min_hashtag_count
        )

        if isinstance(x["hashtags"].iloc[0][0], dict):
            x["hashtags"] = x["hashtags"].apply(
                lambda t: [item["text"] for item in t]
            )

        if self.embedding_model is not None:
            x[self.embedding_name] = self.embed_lemmas(x["lemmas"])
        else:
            assert (
                len(x[self.embedding_name].iloc[0].shape) == 1
            ), f"`{self.embedding_name}` column should contain single vectors."

        if self.verbose:
            print("Dropping tweets without proper words.")
            count_before = len(x)
        x = x.dropna(subset=[self.embedding_name])
        if self.verbose:
            print(f"Dropped: {count_before - len(x)} rows.")
        if self.verbose:
            print("Building Graph")
        self._G = self._build_base_graph(x)

        if self.verbose:
            print("Aggregating hashtag embeddings and mean retweets.")
        self._calculate_hashtag_embeddings_and_mean_retweets()

        if self.verbose:
            print("Calculating edge weights.")
        self._calculate_edge_weights()

        if self.verbose:
            print("Calculating PageRank.")
        self._calculate_pagerank()

        hashtags = [
            {"hashtag": node, **self._G.nodes[node]}
            for node in self._G.nodes
            if self._G.nodes[node]["node_type"] == "hashtag"
        ]
        self._hashtags_df = pd.DataFrame(hashtags)

    def embed_lemmas(
        self, texts_lemmas: Union[Sequence[Sequence[str]], pd.Series]
    ):
        if not isinstance(texts_lemmas, pd.Series):
            texts_lemmas = pd.Series(texts_lemmas)

        def _embed_text(lemmas):
            vectors = []
            for word in lemmas:
                try:
                    vectors.append(self.embedding_model.word_vec(word.lower()))
                except KeyError:
                    continue
            if len(vectors) > 0:
                return np.asarray(vectors).mean(axis=0)
            else:
                return None

        if self.verbose:
            print("Embedding lemmas")
            tqdm.tqdm.pandas()
            return texts_lemmas.progress_apply(_embed_text)
        else:
            return texts_lemmas.apply(_embed_text)

    def transform(
        self,
        x: Union[Tuple[Tuple[str, ...]], Tuple[str, ...], Tuple[np.ndarray]],
        **transform_params,
    ):

        x = list(x)
        # tokenize and lemmatize text
        if isinstance(x[0], str):
            x = [get_wcrft2_results_for_text(txt) for txt in x]

        # turn tokens to embeddings
        if isinstance(x[0], list):
            x = list(self.embed_lemmas(x))

        # calculate mean tweet embeddings
        if len(x[0].shape) > 1:
            x = [emb.mean(axis=0) for emb in x]

        hashtag_emb = np.asarray(
            self._hashtags_df[self.embedding_name].tolist()
        )

        if self.verbose:
            print("Calculating similarities")
        sim = self.embedding_similarity(np.asarray(x), hashtag_emb)
        sim = self.normalise(sim)

        pop = self._hashtags_df[self.popularity_measure].to_numpy()
        pop = self.normalise(pop.reshape(1, -1))

        sim_pop = (1 - self.ratio) * sim + (self.ratio * pop)
        if self.verbose:
            print("Calculating ranking.")
        rankings = np.argsort(-sim_pop, axis=1)
        result = self._hashtags_df["hashtag"].to_numpy()[rankings].tolist()
        return result

    @staticmethod
    def normalise(array: np.ndarray):
        array = array - array.min(axis=1).reshape(-1, 1)
        array = array / (array.max(axis=1).reshape(-1, 1) + 1e-18)
        return array

    @staticmethod
    def embedding_similarity(x: np.ndarray, y: np.ndarray):
        """
        Computes angular similarity based on cosine similarity.
        https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
        Angular similarity is bound to [0, 1] and angular distance is a formal distance metric.
        """

        # clipping to avoid computational errors
        similarity = np.clip(cosine_similarity(x, y), -1, 1)
        ang_dist = np.arccos(similarity) / np.pi
        ang_sim = 1 - ang_dist
        return ang_sim

    def _build_base_graph(self, tweets: pd.DataFrame):
        """
        Builds base tweets-hasthtags graph structure.
        :param tweets: pd.DataFrame containing tweets & featured hashtags.
        :return: nx.Graph of tweets & hashtags.
        """

        g = nx.Graph()

        def add_row_to_graph(row, graph, embedding_name):
            if len(row["hashtags"]) > 0:
                graph.add_node(row["id"])
                graph.nodes[row["id"]]["node_type"] = "tweet"
                graph.nodes[row["id"]]["retweets"] = row["retweet_count"]
                graph.nodes[row["id"]][embedding_name] = row[embedding_name]

                for hashtag in row.hashtags:
                    graph.add_node(hashtag)
                    graph.nodes[hashtag]["node_type"] = "hashtag"
                    graph.add_edge(row["id"], hashtag)

        if self.verbose:
            tqdm.tqdm.pandas()
            tweets.progress_apply(
                lambda r: add_row_to_graph(r, g, self.embedding_name), axis=1
            )
        else:
            tweets.apply(
                lambda r: add_row_to_graph(r, g, self.embedding_name), axis=1
            )
        return g

    def _calculate_hashtag_embeddings_and_mean_retweets(self):

        for node in tqdm.tqdm(
            self._G.nodes, total=len(self._G.nodes), disable=not self.verbose
        ):
            if self._G.nodes[node]["node_type"] == "hashtag":
                embeddings, retweets_counts = [], []
                for neighbour in self._G.neighbors(node):
                    tweet = self._G.nodes[neighbour]
                    embeddings.append(tweet[self.embedding_name])
                    retweets_counts.append(tweet["retweets"])

                # hashtag embeddings
                embeddings = np.asarray(embeddings)
                self._G.nodes[node][self.embedding_name] = embeddings.mean(0)
                # mean retweets
                retweets_counts = np.asarray(retweets_counts)
                self._G.nodes[node]["mean_retweets"] = retweets_counts.mean(0)

    def _calculate_pagerank(self):
        edge_weight_key = "similarity" if self._weighted_pagerank else None

        if self._personalised_pagerank:
            pers_vector = dict()
            for node, attributes in self._G.nodes.data():
                if attributes["node_type"] == "hashtag":
                    pers_vector[node] = attributes["mean_retweets"] + 1
                else:
                    pers_vector[node] = attributes["retweets"] + 1
        else:
            pers_vector = None

        graph_pagerank = nx.pagerank(
            self._G, weight=edge_weight_key, personalization=pers_vector
        )
        nx.set_node_attributes(self._G, graph_pagerank, "pagerank")

    def _calculate_edge_weights(self):
        for (node_from, node_to, edge_features) in tqdm.tqdm(
            self._G.edges(data=True),
            total=len(self._G.edges),
            disable=not self.verbose,
        ):
            emb_from = self._G.nodes[node_from][self.embedding_name]
            emb_to = self._G.nodes[node_to][self.embedding_name]

            sim = self.embedding_similarity(
                emb_from.reshape(1, -1), emb_to.reshape(1, -1)
            ).item()

            edge_features["distance"] = 1 - sim
            edge_features["similarity"] = sim
