#!/usr/bin/env bash

python -m tweet_recommendations.predictions_runner \
    --name our_w2v_udi \
    --dataset data/processed/udi \
    --output experiments \
    --class tweet_recommendations.our_method.OurMethodW2V \
    --config tweet_recommendations/configs/our_method_w2v_our_dataset_predictions.yml \
    --split followers