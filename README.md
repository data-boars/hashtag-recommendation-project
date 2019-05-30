# Anonymous submission
## Anonymous authors

---

## Overview

This repository contains supplemental material for 
the article [REDACTED], submitted to [REDACTED]. It contains implementations of introduced method, as well as baseline methods used as reference. All of them are compatible with scikit-learn API. We also share information on how to reconstruct datasets used for experiments.

## Repository structure 

```python
├── data  # used datasets
├── docs
├── notebooks
├── reports
└── tweet_recommendations # our method
    ├── data_processing # preprocessing scripts
    ├── embeddings # word/tweets embeddings scripts
    ├── other_methods # baselines for our method
    ├── scripts
    ├── utils # various utilities
    └── validators
```

The implementation of our method can be found in file `tweet_recommendations/our_method.py`. 

## Usage

Usage instructions can be found in the documentation of our method. The method requires hyperparameter μ to be specified during initialization of `OurMethod` object. Because we use word embeddings to represent semantic relationships, path to GenSim KeyedVector with chosen embedding model can be specified, so that tweet embeddings can be computed automatically. Alternatively,
if precomputed tweet embeddings are available, they can be used during `fit`.

## Datasets

According to Twitter's Developer Policy [[online]](https://developer.twitter.com/en/developer-terms/agreement-and-policy.html#c-respect-users-control-and-privacy), section
I.C.2:

> If Twitter Content is deleted, gains protected status, or is otherwise suspended, withheld, modified, or removed from the Twitter Service (including removal of location information), you will make all reasonable efforts to delete or modify such Twitter Content (as applicable) as soon as reasonably possible, and in any case within 24 hours after a request to do so by Twitter or by a Twitter user with regard to their Twitter Content, unless otherwise prohibited by applicable law or regulation, and with the express written permission of Twitter.

We are not allowed to share the exact datasets that were used for our research -- we are unable make sure all of the tweets should be still available. Instead, we share IDs of tweets from the dataset, so that they can be reconstructed using Twitter API. This way ensures compliance with Twitter's policies, as the API won't return non-public tweets. 

The dataset files are available in `data/` directory,
where every file contains a tweet ID in each line.

For word embeddings, we've used CLARIN-PL FastText models [[online]]( http://hdl.handle.net/11321/606).