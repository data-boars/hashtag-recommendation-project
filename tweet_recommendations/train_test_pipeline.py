from typing import Any, List

from tweet_recommendations.data_processing.data_loader import (
    get_cleared_dataset_without_specific_content,
    load_dataset_as_dataframe_with_given_fields,
    load_toyger_data,
    save_dataset_as_pickle,
    tokenize_tweet_content_to_types,
)
from tweet_recommendations.embeddings.fast_text import get_fasttext_tweets_embeddings
from tweet_recommendations.embeddings.word2vec import get_w2v_tweets_embeddings
from tweet_recommendations.rank_pipeline import get_hashtag_rank_for_given_tweet_text
from tweet_recommendations.utils.constants import TOYGER_DATA_FILE_PATH
from tweet_recommendations.utils.metrics import mean_average_precision_at_k

use_w2v_embedding = True


def evaluate_results(
    tweets: List[str], expected_rank: List[Any], top_k_precision: int
) -> float:
    predicted_ranks = []
    for tweet in tweets:
        predicted_ranks.append(get_hashtag_rank_for_given_tweet_text(tweet))
    return mean_average_precision_at_k(expected_rank, predicted_ranks, top_k_precision)


if __name__ == "__main__":
    raw_data = load_dataset_as_dataframe_with_given_fields()

    tokenized_tweet_content = tokenize_tweet_content_to_types(
        raw_data, ["hashtag", "emoji", "smiley"]
    )

    cleared_dataset = get_cleared_dataset_without_specific_content(
        tokenized_tweet_content, ["url", "mention", "reserved_words", "number"]
    )

    save_dataset_as_pickle(cleared_dataset, "sample_cleared_dataset_name.pkl")

    # since there are huge problem with memory leaks in toyger, we get processed data from pickle
    lemmatized_tweet_content = load_toyger_data(TOYGER_DATA_FILE_PATH)

    embeddings = (
        get_w2v_tweets_embeddings(lemmatized_tweet_content)
        if use_w2v_embedding
        else get_fasttext_tweets_embeddings(lemmatized_tweet_content)
    )

    # hashtag_rank = perform_some_graph_rank_magic(embeddings)

    # mAP = evaluate_results(test_tweets, expected_test_tweets_rank, top_k_precision=5)
    # print("mAP Test results: {}".format(mAP))
