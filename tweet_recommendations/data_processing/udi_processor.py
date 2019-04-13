import pandas as pd
from typing import Tuple

TWEET_SEPARATOR = "***"
VALUE_SEPARATOR = ":"
MULTILINE_FIELDS = ["Text", "Origin"]
SINGLE_FIELDS = ["Type", "URL", "ID", "Time", "Retcount", "Favorite", "MentionedEntities", "Hashtags"]

def udi_parse_file(filename: str) -> pd.DataFrame:
    """
    Parse single file from UDI Dataset and returns structured DataFrame
    """
    with open(filename, "r", encoding="utf-8") as user_file:
        file_lines = user_file.readlines()

    tweets = []
    current_tweet = None
    current_multiline_field = None
    for line in file_lines:
        if line.startswith(TWEET_SEPARATOR):
            if current_tweet is None:
                current_tweet = {}
            else:
                tweets.append(current_tweet)
                current_tweet = None
        else:
            splitted_line = line.split(VALUE_SEPARATOR)
            line_beginning = splitted_line[0]
            if line_beginning in MULTILINE_FIELDS:
                key, text = _parse_single_line(splitted_line)
                current_tweet[key] = text
                current_multiline_field = key
            elif line_beginning in SINGLE_FIELDS:
                key, text = _parse_single_line(splitted_line)
                current_tweet[key] = text
            else:
                current_tweet[current_multiline_field] = current_tweet[current_multiline_field] + "\n" + line

    return pd.DataFrame(tweets)

def _parse_single_line(splitted_line: str) -> Tuple[str, str]:
    key = splitted_line[0]

    splitted_values = splitted_line[1:]
    value = VALUE_SEPARATOR.join(splitted_values).strip()
    value = _parse_value(key, value)
    
    return key, value

def _parse_value(key, value):
    if key in ("Hashtags", "MentionedEntities"):
        parsed_value = value.split(" ")
    elif key == "Favorite":
        parsed_value = bool(value)
    elif key == "Retcount":
        parsed_value = int(value)
    else:
        parsed_value = value
    return parsed_value