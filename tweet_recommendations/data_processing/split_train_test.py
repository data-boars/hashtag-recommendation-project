import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_user(tweets_df: pd.DataFrame):

    assert "user" in tweets_df

    users = tweets_df.groupby("user").agg({"retweet_count": "mean"})
    users["log_retweet_count"] = users["retweet_count"].apply(
        lambda x: int(np.round(np.log(x + 1) / np.log(8), 0))
    )

    train_users, val_test_users = train_test_split(
        users.index, test_size=0.3, stratify=users["log_retweet_count"]
    )
    val_test_users = users[users.index.isin(val_test_users)]
    test_users, val_users = train_test_split(
        val_test_users.index,
        test_size=0.5,
        stratify=val_test_users["log_retweet_count"],
    )

    return {
        "train": tweets_df[tweets_df["user"].isin(train_users)],
        "val": tweets_df[tweets_df["user"].isin(val_users)],
        "test": tweets_df[tweets_df["user"].isin(test_users)],
    }
