#!/usr/bin/env bash

./predictions_dbscan_based_method_udi_dataset.sh \
    && ./predictions_graph_based_method_udi_dataset.sh \
    && ./predictions_our_method_udi_dataset.sh \
    && ./predictions_tweet2vec_method_udi_dataset.sh