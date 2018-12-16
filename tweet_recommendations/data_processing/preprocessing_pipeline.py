from tweet_recommendations.data_processing.data_loader import (
    get_cleared_dataset_without_specific_content,
    load_dataset_as_dataframe_with_given_fields,
    tokenize_tweet_content_to_types,
    save_dataset_as_pickle,
)

raw_dataset = load_dataset_as_dataframe_with_given_fields()
cleared_dataset_with_extracted_hashtags = get_cleared_dataset_without_specific_content(
    raw_dataset, ["url", "emoji", "mention", "smiley", "number"]
)
tokenized_dataset = tokenize_tweet_content_to_types(
    cleared_dataset_with_extracted_hashtags,
    ["url", "emoji", "mention", "smiley", "number", "hashtag"],
)
save_dataset_as_pickle(tokenized_dataset)
