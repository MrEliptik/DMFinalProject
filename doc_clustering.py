from gensim.test.utils import common_texts
import gensim.models as g
import optics
import text_preprocessing
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

def readVocab(path):
    vocab = []
   
    file = open(path, "r") 
    lines = file.readlines()

    print(" " + str(len(lines)) + " words in the vocab")

    for line in lines:
        line = line.strip()
        vocab.append(line)

    return vocab

def readDocs(path, vocab):
    docs = []
    currentDoc = "1"
    oldDoc = currentDoc
    words = []

    file = open(path, "r") 
    lines = file.readlines()

    nb_documents    = lines[0].strip()
    nb_words        = lines[1].strip()
    nnz             = lines[2].strip()

    print(" " + str(nb_documents) + " documents ; " + str(nb_words) + " words ; " + str(nnz) + " non zero bag of words")
    
    for line in lines[3:]:
        line = line.strip()
        components = line.split(' ')
        currentDoc = components[0]
        # Change document
        if(oldDoc != currentDoc):
            # Store the previous doc
            docs.append(words)

            # Update index
            oldDoc = currentDoc
            words = []
        else:
            # Construct the words for the current doc
            # -1 because words are indexed from 1 in the vocab file
            # but stored starting from 0 in the array
            words.append(vocab[int(components[1]) - 1])        

    return docs

def readDocsAsTfidfMatrix(path, vocab):
    file = open(path, "r") 
    lines = file.readlines()

    nb_documents    = lines[0].strip()
    nb_words        = lines[1].strip()
    nnz             = lines[2].strip()

    matrix = np.empty(shape=(int(nb_documents), int(nb_words)))

    print(" " + str(nb_documents) + " documents ; " + str(nb_words) + " words ; " + str(nnz) + " non zero bag of words")
    
    for line in lines[3:]:
        line = line.strip()
        # components[0] = currentDoc ; components[1] = word ; components[2] = count
        components = line.split(' ')
        doc = int(components[0])
        word = int(components[1])
        count = int(components[2])

        # -1 because the file index start at 1, but we store starting from 0
        matrix[doc-1][word-1] = count
        
    return matrix

def vectorizeDocs(docs):
    vectors = []

    model = "Ressources/doc2vec.bin"

    # Inference hyper-parameters
    start_alpha=0.01
    infer_epoch=1000

    # Load model
    print(" " + "loading model: " + model)
    d2v = g.Doc2Vec.load(model)

    print(" " + "infering vector for each doc")
    for doc in docs:
        vectors.append(d2v.infer_vector(doc))


    return vectors

def vectorizeMatrix(matrix):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

    return tfidf_vectorizer.fit_transform(matrix) 

if __name__ == "__main__":
    vocab_path = "Datasets/kos/vocab.kos.txt"
    doc_path = "Datasets/kos/docword.kos.txt"

    print(">> Reading vocab file: " + vocab_path)
    # Read and store vocab as array
    vocab = readVocab(vocab_path)

    print(">> Reading documents file: " + doc_path)
    # Read docword.nytimes.txt 
    matrix = readDocsAsTfidfMatrix(doc_path, vocab)

    print(">> Vectorizing documents")
    #vectorized = vectorizeDocs(matrix)
    #vectorized = vectorizeMatrix(matrix)
    
    print(">> Running OPTICS")
    optics.cluster(matrix, min_samples=10, n_jobs=4)

    print()
        