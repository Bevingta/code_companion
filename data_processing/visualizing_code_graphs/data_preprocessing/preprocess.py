###* FULL DATA PREPROCESSING *###

import os
from train_word2vec import train_w2v
from gensim.models import Word2Vec
import sys

# making sure always in the right directory, no matter where you run the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Location of the current file
W2V_DIR = os.path.join(BASE_DIR, "..", "word2vec_code.model") # need to make sure this path is correct, based on where it is placed
print(W2V_DIR)
DATABASE_DIR = os.path.join(BASE_DIR, "..", "final_data", "all_train_data_new.json")
CODE_GRAPHS_DIR = os.path.join(BASE_DIR, "code_graph_handling")
sys.path.append(CODE_GRAPHS_DIR)
from regex_converter_main import create_single_graph # type: ignore

word2vec_is_pretrained = os.path.exists(W2V_DIR)


def preprocess():
    ##* STEP 1: LOAD/CREATE WORD2VEC MODEL *###
    # if we need to train word2vec
    if word2vec_is_pretrained is False:
        path_to_dataset = "./final_data/all_train_data_new.json"
        train_w2v(path_to_dataset)
    
    word2vec_model = Word2Vec.load("word2vec_code.model")

    ##* STEP 2: LOAD GRAPHS
    OUTPUT_DIR = os.path.join(BASE_DIR, "code_graphs")
    create_single_graph(DATABASE_DIR, 4, OUTPUT_DIR)

if __name__ == "__main__":
    preprocess()