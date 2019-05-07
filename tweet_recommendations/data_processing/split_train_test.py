import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_to_train_test_by_user(tweets_df: pd.DataFrame):
    assert "username" in tweets_df

    users = tweets_df.groupby("username").agg({"retweet_count": "mean"})
    unpopular_users = users[users["retweet_count"] == 0]
    popular_users = users[users["retweet_count"] > 0]

    popular_users["deciles"] = pd.qcut(popular_users.retweet_count, 8,
                                       duplicates='drop').cat.codes

    tr_users, te_users = train_test_split(popular_users.index,
                                          test_size=0.3,
                                          stratify=popular_users["deciles"])

    tr_users = list(unpopular_users.index) + list(tr_users)
    train = tweets_df[tweets_df["username"].isin(tr_users)]
    test = tweets_df[tweets_df["username"].isin(te_users)]
    return train, test
