from tweet_recommendations.embeddings.fast_text import get_fasttext_tweet_embedding
from tweet_recommendations.embeddings.word2vec import get_w2v_tweet_embedding

sample_tweet_content = (
    "Przykładowa treść tweeta do której mają być zasugerowane hasztagi."
)
use_w2v_embedding = True


def get_hashtag_rank_for_given_tweet_text(tweet: str):
    # sample_tweet_content = some_method_to_tag_and_lemmatze_tweet_content(tweet)
    dummy_lemmatized_tweet_content = ["Przykładowa", "treść", "tweeta", "itp"]

    embeddings = (
        get_w2v_tweet_embedding(dummy_lemmatized_tweet_content)
        if use_w2v_embedding
        else get_fasttext_tweet_embedding(dummy_lemmatized_tweet_content)
    )

    hashtag_rank = []
    # hashtag_rank = perform_some_graph_rank_magic(embeddings)

    return hashtag_rank


if __name__ == "__main__":
    hashtag_rank = get_hashtag_rank_for_given_tweet_text(sample_tweet_content)
    print(hashtag_rank)
