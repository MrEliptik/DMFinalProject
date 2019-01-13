import text_preprocessing
import optics

def readComments(path):
    comments = []
   
    file = open(path, "r") 
    lines = file.readlines()

    print(" " + str(len(lines)) + " comments in this file")

    for line in lines:
        line = line.strip()
        line.split("    ")
        comments.append(line[1])

    return comments

if __name__ == "__main__":
    path = "Datasets/OpinRankDataset/hotels/beijing/china_beijing_aloft_beijing_haidian"

    print(">> Reading vocab file: " + path)
    comments = readComments(path)

    print(">> Vectorizing documents")
    #vectorized = vectorizeDocs(matrix)
    #vectorized = vectorizeMatrix(matrix)
    
    '''
    print(">> Running OPTICS")
    optics.cluster(matrix, 20, 5)

    dbscan = DBSCAN(n_jobs=4)
    clt = dbscan.fit(matrix)
    print(clt.labels_)


    print()
    '''