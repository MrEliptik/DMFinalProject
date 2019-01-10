from sklearn.feature_extraction.text import TfidfVectorizer
import optics

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

def vectorizeDocs(docs):

    def dummy_fun(doc):
        return doc

    vectors = []

    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None) 

    print(' Fitting..')
    tfidf.fit(docs)

    print(' Transforming into vectors..')
    for doc in docs:
        # doc is wrapped in a list for transformation
        # important to be seen as one doc with multiple words
        vectors.append(tfidf.transform([doc]).data)

    return vectors


if __name__ == "__main__":
    vocab_path = "Datasets/kos/vocab.kos.txt"
    doc_path = "Datasets/kos/docword.kos.txt"

    print(">> Reading vocab file: " + vocab_path)
    # Read and store vocab as array
    vocab = readVocab(vocab_path)

    print(">> Reading documents file: " + doc_path)
    # Read docword.nytimes.txt 
    docs = readDocs(doc_path, vocab)

    print(">> Vectorizing documents")
    vectorized = vectorizeDocs(docs)
    
    optics.template_clustering(vectorized, 5, 10)

    print()
        