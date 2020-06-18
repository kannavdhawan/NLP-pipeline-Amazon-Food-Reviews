from gensim_w import load_data,save_model,load_model
import sys
import os

def main(data_path):
    formatted_dataset=load_data(data_path)
    model_path=save_model(formatted_dataset)
    load_model(model_path)

if __name__=='__main__':
    main(os.sys.argv[1])
