#!/usr/bin/env bash

python -m tweet_recommendations.predictions_runner \
    --name graph_based_our \
    --dataset data/processed/our \
    --output experiments \
    --class tweet_recommendations.other_methods.GraphSummarizationMethod \
    --config tweet_recommendations/configs/graph_based_method_our_dataset_predictions.yml \
    --split followers