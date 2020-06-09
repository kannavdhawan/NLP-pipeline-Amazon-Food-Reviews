import numpy as np
import os 
import sys
from N_grams_generation import data_conversion
from N_grams_generation import n_grams
def main(data_path):
    lists=data_conversion(data_path)
    N_grams_generation(lists)

if __name__ == '__main__':
    main(sys.argv[1])
#calls