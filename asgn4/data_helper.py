
import re

import pandas as pd



def get_reviews(fpath):
    """
    :param fpath: file path to dataset
    :return:
    """
    print("get reviews")
    print("fpath")
    print(fpath)
    if "test_examples.tsv" in fpath:
        print("testt is true")
        with open(fpath, "r") as fin:
            df = pd.read_csv(fpath, sep="\t")
            pos = df.loc[df.sentiment=="pos"]
            neg = df.loc[df.sentiment=="neg"]
            #lines = [line.strip().rstrip() for line in fin.read().split("\n")]
        return pos["text"], neg["text"]
        #return lines

    else:
        df = pd.read_csv(fpath, sep="\t")
        pos = df.loc[df.sentiment=="pos"]
        neg = df.loc[df.sentiment=="neg"]
        return pos["text"], neg["text"]





def test_main():
    pass



if __name__ == "__main__":
    test_main()
