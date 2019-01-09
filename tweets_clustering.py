import text_preprocessing as preprocess
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import optics
import matplotlib.pyplot as plt
import time

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
  
    with open(gloveFile, encoding="utf8" ) as f:
       content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model


if __name__ == "__main__":
    '''
    file = "Ressources/glove.6B.50d.txt"
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_input_file=file, word2vec_output_file="Ressources/gensim_glove_vectors.txt")
    '''

    # Loading model for vectorization
    #file = "Ressources/glove.6B.50d.txt"
    #model = loadGloveModel(file)
    '''
    glove_model_path = "Ressources/gensim_glove_vectors.txt"
    

    # Loading model for vectorization
    print(">> Loading glove model from: " + glove_model_path)
    glove_model = KeyedVectors.load_word2vec_format(glove_model_path, binary=False)
    '''
   
    tweets_path = "Datasets/Health-News-Tweets/"

    # Extracting and preprocessing tweets
    print(">> Extracting tweets from: " + tweets_path)
    start = time.time()
    tweets = []
    tweets_avg = []
    file = open(tweets_path + "bbchealth.txt", "r") 
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        splitted = line.split('|')
        try:
            tweets.append(preprocess.preprocess(splitted[2]))
        except Exception as e:
            print("Error: " + str(e))
    file.close()
    print(">> " + str(time.time() - start) + "s execution time.")

    print(">> Loading Word2Vec on extracted tweets..")
    start = time.time()
    model = Word2Vec(tweets, min_count=1)
    data = model[model.wv.vocab]
    print(">> " + str(time.time() - start) + "s execution time.")

    #TODO: Maybe run similarity, or something esle before calling OPTICS
    
    print(">> Running OPTICS..")
    start = time.time()
    optics.template_clustering(data[453180:], 150, 50)
    print(">> " + str(time.time() - start) + "s execution time.")
     

    




    



