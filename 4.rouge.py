import os, numpy as np
from nltk.tokenize import word_tokenize as wt
from string import punctuation as punct

path1 = './4.summarized/'
path2 = './summary/'

def bigram(path):
    result = []
    for file in os.listdir(path):
        f = open(path+file).read()
        f = f.splitlines()[2:]

        temp = []
        for row in f:
            row = wt(row.lower())
            row = [i for i in row if i not in punct]

            for i in range(len(row)-1):
                temp.append((row[i]+' '+row[i+1]))
        result.append(temp)
    return result

resl = bigram(path1)
sums = bigram(path2)

f = open('RECALL.txt','w')
rerata = []
for i in range(20):
    N = len(sums[i])
    I = len(set(resl[i])&set(sums[i]))
    R = (I/N)*100
    rerata.append(R)
    mystr = 'Recall Summary Dokumen ke-'+str(i+1)+': '+str(round(R,2))+'%'
    f.write(mystr+'\n')
    print(mystr)
print('\nRERATA RECALL:',str(round(np.average(rerata),2))+'%')
f.close()
