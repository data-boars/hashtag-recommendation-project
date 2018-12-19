from tweet_recommendations.embeddings.fast_text import get_fasttext_tweet_embedding
from tweet_recommendations.embeddings.word2vec import get_w2v_tweet_embedding
from tweet_recommendations.utils.clients import get_wcrft2_results_for_text

use_w2v_embedding = True


def get_hashtag_rank_for_given_tweet_text(tweet: str):
    tagged_tweet_content = get_wcrft2_results_for_text(tweet)

    embeddings = (
        get_w2v_tweet_embedding(tagged_tweet_content)
        if use_w2v_embedding
        else get_fasttext_tweet_embedding(tagged_tweet_content)
    )

    hashtag_rank = []
    # hashtag_rank = perform_some_graph_rank_magic(embeddings)

    return hashtag_rank