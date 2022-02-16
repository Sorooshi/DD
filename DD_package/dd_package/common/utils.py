import os
import sys
import pickle
from pathlib import Path


def save_a_dict(a_dict, name, save_path, ):
    with open(os.path.join(save_path, name+".pickle"), "wb") as fp:
        pickle.dump(a_dict, fp)
    return None


def load_a_dict(name, save_path, ):
    with open(os.path.join(save_path, name + ".pickle"), "rb") as fp:
        a_dict = pickle.load(fp)
    return a_dict
