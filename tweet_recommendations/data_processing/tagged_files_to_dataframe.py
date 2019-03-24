import argparse
import codecs
import multiprocessing as mp
from pathlib import Path
from typing import *

import dask.bag as db
import pandas as pd
from dask.diagnostics import ProgressBar

parser = argparse.ArgumentParser(
    description="Converts tagged files by tagger toyger to dataframe with an id and lemmas")
parser.add_argument("--dir", help="Str path to zip file")
parser.add_argument("--out", help="Output file name with `p` extension")

args = parser.parse_args()


def parse_path(path: Path) -> Dict[str, List[str]]:
    tweet_id = path.name.split(".")[0]
    with codecs.open(path.as_posix(), encoding='utf-8') as f:
        file_content = f.read().strip().replace("'", "").replace('"', "")
    try:
        frame = pd.read_csv(pd.compat.StringIO(file_content), quotechar="'", sep="\t", header=None)
        frame = frame[frame[4] != "interp"]
        frame = frame.drop_duplicates(subset=[1])
        frame[3] = frame[3].map(lambda x: str(x).lower().split(':')[0])
        return {
            "id_str": tweet_id,
            "lemmas": frame[3].tolist()
        }
    except:
        print(file_content)
        print(tweet_id)
        return {
            "id_str": "",
            "lemmas": []
        }


def main():
    paths = list(Path(args.dir).rglob("*.txt"))
    pbar = ProgressBar()
    pbar.register()
    a_bag = db.from_sequence(paths, npartitions=mp.cpu_count())
    a_bag = a_bag.map(lambda a_path: parse_path(a_path))
    frame_data = a_bag.compute()
    pbar.unregister()

    frame = pd.DataFrame(frame_data)
    frame.to_pickle(args.out)


if __name__ == '__main__':
    main()
