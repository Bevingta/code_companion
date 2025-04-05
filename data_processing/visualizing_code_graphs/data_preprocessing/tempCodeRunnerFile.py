
# make sure you are checking in the right directory
path_to_pretrained_w2v = "./word2vec_code.model"
word2vec_is_pretrained = os.path.exists(path_to_pretrained_w2v)

def preprocess():
    ##* STEP 1: LOAD/CREATE WORD2VEC MODEL *###
    # if we need to train word2vec
    if word2vec_is_pretrained is False:
        path_to_dataset = "./final_data/all_train_data_new.json"
        train_w2v(path_to_dataset)
    
    word2vec_model = Word2Vec.load("word2vec_code.model")

    ##* STEP 2: LOAD GRAPHS
    create_single_graph("./final_data/all_train_data_new.json", 0)

if __name__ == "__main__":
    preprocess()