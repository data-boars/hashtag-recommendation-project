#!/usr/bin/env bash

python -m tweet_recommendations.predictions_runner \
    --name dbscan_based_udi \
    --dataset data/processed/udi \
    --output experiments \
    --class tweet_recommendations.other_methods.DBScanBasedMethod \
    --config tweet_recommendations/configs/dbscan_based_method_our_dataset_predictions.yml \
    --split retweet_count