#!/usr/bin/env bash

python -m tweet_recommendations.predictions_runner \
    --name tweet2vec_udi \
    --dataset data/processed/udi \
    --output experiments \
    --class tweet_recommendations.other_methods.Tweet2Vec \
    --config tweet_recommendations/configs/tweet2vec_method_our_dataset_predictions.yml \
    --split followers