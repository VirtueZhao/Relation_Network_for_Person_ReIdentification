import json


def read_json(fpath):
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj
