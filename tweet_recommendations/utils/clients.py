from typing import List

import requests
from bs4 import BeautifulSoup

WCRFT2_ENDPOINT = "http://ws.clarin-pl.eu/nlprest2/base/process/"
WCRFT2_JSON_TEMPLATE = {"lpmn": "any2txt|wcrft2", "text": "", "user": "sample@mail.com"}


def parse_wcrft_response(xml_response: str) -> List[str]:
    xml_soup = BeautifulSoup(xml_response, features="xml")
    all_tags = xml_soup.find_all("base")
    all_words = [tag.text for tag in all_tags]

    return all_words


def get_wcrft2_results_for_text(tweet_to_process: str) -> List[str]:
    lemmatized_tweet_words_as_list = []
    request_body = WCRFT2_JSON_TEMPLATE
    request_body["text"] = tweet_to_process

    r = requests.post(
        WCRFT2_ENDPOINT, json=request_body, headers={"Content-Type": "application/json"}
    )
    status_code = r.status_code

    if status_code == 200:
        r.encoding = "ISO-8859-1"
        xml_response = r.content
        lemmatized_tweet_words_as_list = parse_wcrft_response(xml_response)
    else:
        print(
            "Something went wrong with clarin service - status code: {}".format(
                status_code
            )
        )
    return lemmatized_tweet_words_as_list
