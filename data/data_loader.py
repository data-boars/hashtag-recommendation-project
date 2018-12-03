import pandas as pd
import preprocessor as p

from tweet_recommendations.utils.constants import EXTERNAL_DATA_PATH, OUTPUT_DATA_PATH


def load_dataset_as_dataframe_with_given_fields(
    fields=["hashtags", "text", "retweet_count", "id"], filename="original_tweets.p"
):
    tweets_df = pd.read_pickle(EXTERNAL_DATA_PATH.format(filename=filename))
    dataset = pd.DataFrame(tweets_df)[fields]
    return dataset


def save_dataset_as_pickle(dataset, filename="output_dataset.pkl"):
    dataset.to_pickle(OUTPUT_DATA_PATH.format(filename=filename))


def extract_hashtags(list_of_dicts):
    output_hashtags = []
    for hash_dict in list_of_dicts:
        output_hashtags.append(hash_dict["text"])
    return list(set(output_hashtags))


def tokenize_tweet_content_to_types(dataset, tokenize_type_list):
    """Tokenize all tweets with with defined contents i.e 'something #DataScience' with 'something $HASHTAG$' """
    tuple_to_unpack = get_filter_obejcts_as_tuple(tokenize_type_list)
    p.set_options(*tuple_to_unpack)
    dataset["text"] = dataset["text"].apply(lambda txt: p.tokenize(txt))
    return dataset


def get_cleared_dataset_without_specific_content(dataset, clearing_type_list):
    dataset["hashtags"] = dataset["hashtags"].apply(lambda txt: extract_hashtags(txt))
    tuple_to_unpack = get_filter_obejcts_as_tuple(clearing_type_list)
    p.set_options(*tuple_to_unpack)
    dataset["text"] = dataset["text"].apply(lambda txt: p.clean(txt))
    return dataset


def get_filter_obejcts_as_tuple(text_content_types):
    lower_case_types = [string_type.lower() for string_type in text_content_types]
    filter_types = []
    if "url" in lower_case_types:
        filter_types.append(p.OPT.URL)
    if "mention" in lower_case_types:
        filter_types.append(p.OPT.MENTION)
    if "hashtag" in lower_case_types:
        filter_types.append(p.OPT.HASHTAG)
    if "reserved_words" in lower_case_types:
        filter_types.append(p.OPT.RESERVED)
    if "emoji" in lower_case_types:
        filter_types.append(p.OPT.EMOJI)
    if "smiley" in lower_case_types:
        filter_types.append(p.OPT.SMILEY)
    if "number" in lower_case_types:
        filter_types.append(p.OPT.NUMBER)
    tuple_to_unpack = tuple(filter_types)
    return tuple_to_unpack
