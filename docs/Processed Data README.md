# Data description
Two files were produced in total which are described below.

## File `lemmatised.pkl`
Contains list of tuples in on of the following formats:
```python
correct_data = [tweet_id, tweet_content_including_emojis_list_of_strings, 'ok']
```
```python
error_data = [file_name, error_message_string, 'not ok']
```
Error samples are left just in case for debugging or getting which tweets were erroneous.
Dealing with those is presented below.

**Disclaimers**:
- all capital letters are left as they were (it's easy to just `lower()` them all)
- tweet content includes words, punctuation, emojis, unicode hyphens etc.
- it's not guaranteed that lemmas are either correct or appropriate (toyger-tagger works ... bad sometimes)
- there are 20 tweets in total which were empty (and they caused errors in processing)


## File `tagged.zip`
Contains all texts from `original_tweets.p` provided earlier processed by `tagger-toyger`. 
Each processed text as saved in separate file in format `<tweet_id>.tagget.txt`. Processed
files contains data provided by `tagger-toyger`, including POS tag, either word is old fashioned or not,
possible disambiguations, index in the analysed sentenced.

## Snippets
Load processed and lemmatised data:
```python
import pickle
with open('<your_path>/lemmatised.pkl', 'rb') as f:
    data = pickle.load(f)
```

Filtering:
```python
ok_data = [x for x in data if x[-1] == 'ok']
not_ok_data = [x for x in data if [-1] == 'not ok']
```