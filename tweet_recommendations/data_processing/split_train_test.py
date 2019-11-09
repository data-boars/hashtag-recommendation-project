import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test_by_user(tweets_df: pd.DataFrame, users_df: pd.DataFrame, split_on: str, test_size: str = 0.3):
    tr_users, te_users = get_train_test_users(tweets_df, users_df, split_on, test_size)
    train = tweets_df[tweets_df["user_id"].isin(tr_users)]
    test = tweets_df[tweets_df["user_id"].isin(set(te_users))]
    return train, test


def get_train_test_users(tweets_df: pd.DataFrame, users_df: pd.DataFrame, split_on: str, test_size: str = 0.3):
    if split_on == "retweet_count":
        users = tweets_df.groupby("user_id").agg({"retweet_count": "mean"}).reset_index()
        users = users.rename(columns={"retweet_count": "split_variable"})
    elif split_on == "followers":
        users = users_df[["user_id", "followers"]].rename(columns={"followers": "split_variable"})
        users = users[users["user_id"].isin(tweets_df["user_id"])]
    else:
        raise Exception()

    unpopular_users = users[users["split_variable"] == 0].copy()
    unpopular_users['decile'] = 0
    popular_users = users[users["split_variable"] > 0].copy()

    popular_users["decile"] = pd.qcut(popular_users["split_variable"], 10,
                                      duplicates='drop').cat.codes + 1

    users = pd.concat([unpopular_users, popular_users])
    tr_users, te_users = train_test_split(users["user_id"],
                                          test_size=test_size,
                                          stratify=users["decile"])
    return tr_users, te_users
