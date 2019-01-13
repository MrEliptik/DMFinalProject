import optics
import text_preprocessing
import pickle
import os.path

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import numpy as np

from sklearn.cluster import DBSCAN


def readSynopsis(path):
    synopsis = []
   
    file = open(path, "r") 
    lines = file.readlines()

    print(" " + str(len(lines)) + " synopsis")

    for line in lines:
        line = line.strip()
        if(line != "" and "BREAKS HERE" not in line):
            synopsis.append(line)
    return synopsis

def readTitles(path):
    titles = []
   
    file = open(path, "r") 
    lines = file.readlines()

    print(" " + str(len(lines)) + " titles")

    for line in lines:
        line = line.strip()
        if(line != ""):
            titles.append(line)
    return titles

def savePickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def loadPickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def vectorize(data):
    #define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                    min_df=0.2, stop_words='english',
                                    use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

    return tfidf_vectorizer.fit_transform(data).toarray() #fit the vectorizer to synopses

if __name__ == "__main__":
    synopsis_path   = "Datasets/imdb_top100/synopsis.txt"
    titles_path     = "Datasets/imdb_top100/titles.txt"
    pickle_path     = "Datasets/imdb_top100/top100_synopsys.pkl"

    # Never been run, need to extract value and save as pickle for
    # later faster access
    if(not os.path.isfile(pickle_path)):
        print(">> Reading synopsys: " + synopsis_path)
        synopsis = readSynopsis(synopsis_path)

        print(">> Reading titles: " + titles_path)
        titles = readTitles(titles_path)

        '''
        print(">> Processing synopsys")
        processed = []
        for element in synopsis:
            processed.append(text_preprocessing.preprocess(element))
        '''

        print(">> Pickling synopsys: " + pickle_path)
        savePickle(synopsis, pickle_path)
    else:
        print(">> Reading titles: " + titles_path)
        titles = readTitles(titles_path)

        print(">> Loading pickled synopsys: " + pickle_path)
        processed = loadPickle(pickle_path)

        print(">> Vectorize data")
        vectorized = vectorize(processed)
        
        clust = optics.cluster(vectorized, 5)

        labelIDs = np.unique(clust.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        print("[INFO] # unique faces: {}".format(numUniqueFaces))

        '''
        from sklearn.cluster import KMeans

        num_clusters = 5
        km = KMeans(n_clusters=num_clusters)
        km.fit(vectorized)

        clusters = km.labels_.tolist()
        '''

    print()