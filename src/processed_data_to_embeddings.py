import tqdm
import pandas as pd
import fastText
import numpy as np
import pickle as pkl
from collections import defaultdict


def process(embedding_model_path: str, input_path: str, output_path: str):
    embedding_model = fastText.load_model(embedding_model_path)
    with open(input_path, 'rb') as f:
        tagged_files = pkl.load(f)
    output = defaultdict(list)
    for tweet, seq, is_ok in tqdm.tqdm(tagged_files):
        if is_ok == 'ok':
            all_embeddings = []
            for word in seq:
                all_embeddings.append(embedding_model.get_word_vector(word))
            output['tweet_id'].append(tweet)
            output['embeddings'].append(np.mean(all_embeddings, axis=0))
    frame = pd.DataFrame(output)
    frame.to_pickle(output_path)
    del embedding_model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Processing tagged files to embeddings format")
    parser.add_argument("--embedding_model_path")
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")

    args = parser.parse_args()
    process(args.embedding_model_path, args.input_path, args.output_path)
