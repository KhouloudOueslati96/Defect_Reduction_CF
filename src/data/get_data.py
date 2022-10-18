import os
import sys
from glob import glob

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def get_project(dir):
    return sorted(glob(os.path.abspath(os.path.join(dir, "*.csv"))), key=lambda x: x.split("/")[-1])


def get_all_projects():
    all = dict()
    dirs = glob(os.path.join(root, "data/*/"))
    for dir in dirs:
        all.update({dir.split('/')[-2]: get_project(dir)})
    return all


if __name__ == "__main__":
    data = get_all_projects()
