import argparse

import numpy as np
from gensim.models import FastText, KeyedVectors


def convert_w2v(model_path: str, output_path: str):
    print("Loading ...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    print("Saving ...")
    model.save(output_path)
    print("Sanity check ...", end=" ")
    saved_model = KeyedVectors.load(output_path, mmap="r")
    np.testing.assert_allclose(saved_model["okej"], model["okej"])
    print("\u2713")  # tick mark


def convert_fasttext(model_path: str, output_path: str):
    print("Loading ...")
    model = FastText.load_fasttext_format(model_path)
    print("Saving ...")
    model.wv.save(output_path)
    print("Sanity check ...", end=" ")
    saved_model = KeyedVectors.load(output_path, mmap="r")
    np.testing.assert_allclose(saved_model["okej"], model.wv["okej"])
    print("\u2713")  # tick mark


def main():
    parser = argparse.ArgumentParser("Converted of normally saved as a dict w2v model to mmaped file, so the model"
                                     "can be loaded almost immediately")
    parser.add_argument("model_path", help="Path to model saved as *.vec model for w2v or *.bin for fasttext")
    parser.add_argument("output_path",
                        help="Path for an output file without an extension. "
                             "Two or three files will be saved, depending on the selected model type: "
                             "one main mmaped file with `vectors` extension and the other as a reference file. "
                             "For FastText there comes also file with ngrams."
                             "Reference file should be loaded then in `load` method")
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Wanted to convert model is a fasttext model")
    parser.add_argument("--w2v", action="store_true", default=False, help="Wanted to convert model is a w2v model")
    args = parser.parse_args()

    if args.fast:
        convert_fasttext(args.model_path, args.output_path)
    elif args.w2v:
        convert_w2v(args.model_path, args.output_path)
    else:
        raise ValueError("Either --fast or --w2v must be present in args.")


if __name__ == '__main__':
    main()
