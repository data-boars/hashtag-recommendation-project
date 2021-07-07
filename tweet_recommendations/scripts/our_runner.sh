#!/usr/bin/env bash

echo "=== DBScan - Our ===" \
    && ./tweet_recommendations/scripts/predictions_dbscan_based_method_our_dataset.sh \
    && echo "=== Graph - Our ===" \
    && ./tweet_recommendations/scripts/predictions_graph_based_method_our_dataset.sh \
    && echo "=== Our - Our ===" \
    && ./tweet_recommendations/scripts/predictions_our_method_w2v_our_dataset.sh \
    && echo "=== Tweet2Vec - Our ===" \
    && ./tweet_recommendations/scripts/predictions_tweet2vec_method_our_dataset.sh \
    && echo "=== Our Tweet2Vec - Our ===" \
    && ./tweet_recommendations/scripts/predictions_our_method_tweet2vec_our_dataset.sh
