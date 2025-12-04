
import os
import numpy as np
import pandas as pd

path_stock_dayK = "./data/stock_dayK/"
path_index_dayK = "./data/index_dayK/"
path_inst = "./data/stock_institution/"

def load_files(path=path_stock_dayK, left_range=0.3, seed=1919840, shuffle=True):
    path_dayK_files = [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    path_dayK_files.sort()
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(path_dayK_files)

    split_pt = int(len(path_dayK_files) * left_range)

    left = path_dayK_files[:split_pt]
    right = path_dayK_files[split_pt:]

    return left, right

def chromosome_extend(ch, length):
    """
    Extend the chromosome to the given length by padding 0.
    """
    ch = np.array(ch)
    assert len(ch) > 0, "chromosome is empty"
    if len(ch) >= length:
        return ch
    if len(ch) < length:
        # padding 0
        ch = np.concatenate((ch, np.zeros(length - len(ch))))
    return ch

def chromosomes_extend(chromosomes, length):
    """
    Extend the chromosomes to the given length by padding 0.
    """
    for i in range(len(chromosomes)):
        chromosomes[i] = chromosome_extend(chromosomes[i], length)
    return chromosomes

def load_taiex(path=path_index_dayK):
    """
    Load TAIEX data from the given path.
    """
    taiex_data = pd.read_csv(path + "TAIEX.csv")
    return taiex_data

def load_tpex(path=path_index_dayK):
    """
    Load TPEX data from the given path.
    """
    tpex_data = pd.read_csv(path + "TPEx.csv")
    return tpex_data

