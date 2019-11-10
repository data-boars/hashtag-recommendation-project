#!/usr/bin/env bash

python -m tweet_recommendations.predictions_runner \
    --name our \
    --dataset data/processed/our \
    --output experiments \
    --class tweet_recommendations.our_method.OurMethod \
    --config tweet_recommendations/configs/our_method_predictions.yml \
    --split retweet_count