import numpy as np, math, os

alldata = []
for file in os.listdir('1.clean'):
    f = open('./1.clean/' + file).read().splitlines()
    temp = []
    for i in f[1:]:
        a, b = i.split(' | ')
        temp.append(b)
    alldata.append(temp)

scores = []
for file in os.listdir('3.scored'):
    f = open('./3.scored/' + file).read().splitlines()
    temp = []
    for i in f[2:]:
        a, b, c = i.split(' | ')
        try:
            temp.append(float(b))
        except:
            temp.append(0)
    scores.append(temp)

for enu, data in enumerate(alldata):

    vocab = []
    for i in data:
        for j in i.split():
            if j not in vocab:
                vocab.append(j)

    N = len(data)
    V = len(vocab)

    tf = []
    for i in data:
        i = i.split()
        temp = []
        for j in vocab:
            temp.append(i.count(j) / len(i))
        tf.append(temp)

    df = []
    for i in vocab:
        count = 0
        for j in data:
            j = j.split()
            if i in j:
                count += 1
        df.append(count)

    idf = []
    for i in df:
        idf.append(math.log10(len(data) / i))

    tfidf = np.array([np.array(i) * np.array(idf) for i in tf])

    a, b, Vt = np.linalg.svd(tfidf.T)

    persen = 50

    pick = int((persen / 100) * len(data))

    indexs = []
    for i in range(len(Vt)):
        index = np.argmax(Vt[i])
        while index in indexs:
            Vt[i][index] = min(Vt[i]) - 1
            index = np.argmax(Vt[i])

        indexs.append(index)
        if len(indexs) == pick:
            break
