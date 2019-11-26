#!/usr/bin/env bash

echo "=== DBScan - UDI ===" \
    && ./tweet_recommendations/scripts/predictions_dbscan_based_method_udi_dataset.sh \
    && echo "=== Graph - UDI ===" \
    && ./tweet_recommendations/scripts/predictions_graph_based_method_udi_dataset.sh \
    && echo "=== Our - UDI ===" \
    && ./tweet_recommendations/scripts/predictions_our_method_w2v_udi_dataset.sh \
    && echo "=== Tweet2Vec - UDI ===" \
    && ./tweet_recommendations/scripts/predictions_tweet2vec_method_udi_dataset.sh