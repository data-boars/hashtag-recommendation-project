#!/usr/bin/env bash

./tweet_recommendations/scripts/predictions_dbscan_based_method_our_dataset.sh \
    && ./tweet_recommendations/scripts/predictions_graph_based_method_our_dataset.sh \
    && ./tweet_recommendations/scripts/predictions_our_method_our_dataset.sh \
    && ./tweet_recommendations/scripts/predictions_tweet2vec_method_our_dataset.sh