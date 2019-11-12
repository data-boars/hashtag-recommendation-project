#!/usr/bin/env bash

python -m tweet_recommendations.predictions_runner \
    --name our_our \
    --dataset data/processed/our \
    --output experiments \
    --class tweet_recommendations.our_method.OurMethod \
    --config tweet_recommendations/configs/our_method_our_dataset_predictions.yml \
    --split followers