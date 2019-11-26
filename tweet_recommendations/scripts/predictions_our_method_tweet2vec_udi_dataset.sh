#!/usr/bin/env bash

python -m tweet_recommendations.predictions_runner \
    --name our_tweet2vec_udi \
    --dataset data/processed/udi \
    --output experiments \
    --class tweet_recommendations.our_method.OurMethodTweet2Vec \
    --config tweet_recommendations/configs/our_method_tweet2vec_our_dataset_predictions.yml \
    --split followers