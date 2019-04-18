import argparse

import numpy as np
from gensim.models import KeyedVectors


def convert(model_path: str, output_path: str):
    print("Loading ...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    print("Saving ...")
    model.save(output_path)
    print("Sanity check ...")
    saved_model = KeyedVectors.load(output_path, mmap="r")
    np.testing.assert_allclose(saved_model["okej"], model["okej"])
    print("\u2713")  # tick mark


def main():
    parser = argparse.ArgumentParser("Converted of normally saved as a dict w2v model to mmaped file, so the model"
                                     "can be loaded almost immediately")
    parser.add_argument("model_path", help="Path to model saved as *.vec model")
    parser.add_argument("output_path",
                        help="Path for an output file without an extension. Two files will be saved: "
                             "one main mmaped file with `vectors` extension and the other as a reference file. "
                             "Reference file should be loaded then in `load` method")
    args = parser.parse_args()
    convert(args.model_path, args.output_path)


if __name__ == '__main__':
    main()
