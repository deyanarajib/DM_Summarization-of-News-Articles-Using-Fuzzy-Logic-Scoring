import os, time
import skfuzzy as fuzz, numpy as np
from matplotlib import pyplot as plt

X = np.arange(0,1.0001,0.0001)
X = np.asarray([np.round(i,4) for i in X])

FS1 = ['unimportant','average','important']
FS2 = ['L','M','H']

lab1 = dict([(i,j) for j,i in enumerate(FS1)])
lab2 = dict([(i,j) for i,j in enumerate(FS2)])

#L = fuzz.trapmf(X,[-0.1,0,0.2,0.4])
#M = fuzz.trapmf(X,[0.2,0.4,0.6,0.8])
#H = fuzz.trapmf(X,[0.6,0.8,1,1.1])

F1L = fuzz.gaussmf(X,0.07,0.033)
F1M = fuzz.gaussmf(X,0.17,0.02)
F1H = fuzz.gaussmf(X,0.6,0.15)

F2L = fuzz.gaussmf(X,0.15,0.07)
F2M = fuzz.gaussmf(X,0.45,0.08)
F2H = fuzz.gaussmf(X,0.8,0.07)

F3L = fuzz.gaussmf(X,0.15,0.06)
F3M = fuzz.gaussmf(X,0.5,0.075)
F3H = fuzz.gaussmf(X,0.82,0.055)

F4L = fuzz.gaussmf(X,0.15,0.07)
F4M = fuzz.gaussmf(X,0.5,0.06)
F4H = fuzz.gaussmf(X,0.82,0.055)

F5L = fuzz.gaussmf(X,0.1,0.033)
F5M = fuzz.gaussmf(X,0.82,0.07)
F5H = fuzz.gaussmf(X,0.4,0.1)

F6L = fuzz.gaussmf(X,0.05,0.029)
F6M = fuzz.gaussmf(X,0.15,0.02)
F6H = fuzz.gaussmf(X,0.6,0.14)

F7L = fuzz.gaussmf(X,0.17,0.07)
F7M = fuzz.gaussmf(X,0.5,0.065)
F7H = fuzz.gaussmf(X,0.82,0.08)

F8L = fuzz.gaussmf(X,0.025,0.014)
F8M = fuzz.gaussmf(X,0.08,0.01)
F8H = fuzz.gaussmf(X,0.55,0.145)

#L = fuzz.trimf(X,[-0.0001,0.2,0.4])
#M = fuzz.trimf(X,[0.3,0.5,0.7])
#H = fuzz.trimf(X,[0.6,0.8,1.0001])

#plt.plot(X,L,X,M,X,H)
#plt.ylim([-0.1,1.1])
#plt.show()

NS = open('Ns.txt').read().split()
NS = np.int32(NS)

Unimportant = fuzz.trimf(X,[-0.0001,0.2,0.4])
Average     = fuzz.trimf(X,[0.3,0.5,0.7])
Important   = fuzz.trimf(X,[0.6,0.8,1.0001])

tri = dict([('unimportant',[0,  0.2,0.4]),
            ('average',    [0.3,0.5,0.7]),
            ('important',  [0.6,0.8,1])])

frules = [i.strip().split() for i in open('frules.txt').readlines()]
frules = [(i[:-1],i[-1].lower()) for i in frules]

indeks = dict([(j,i) for i,j in enumerate(X)])

path_inp1 = './2.feature/'
path_inp2 = './0.dataset raw/'
path_out1 = './3.scored/'
path_out2 = './4.summarized/'

for enum,file in enumerate(os.listdir(path_inp1)):
    start = time.time()
    print(file,end=' ')

    f = open(path_inp1+file).read()

    data,label = [],[]
    for row in f.splitlines():
        temp = []
        label.append(row.split()[0])
        for col in row.split()[1:]:
            temp.append(float(col))
        data.append(temp)
    data = np.asarray(data)

    score = []
    for x in data:

        FS,FV = ['','',''],[0,0,0]

        for R,fs in frules:
            a,m,b = tri[fs]

            c = []
            for j,r in enumerate(R):
                val = np.round(x[j],4)
                c.append(eval('F'+str(j+1)+r)[indeks[val]])
            minc = min(c)
        
            if FS[lab1[fs]] == '':
                FS[lab1[fs]] = fs
                FV[lab1[fs]] = minc
            else:
                if minc > FV[lab1[fs]]:
                    FV[lab1[fs]] = minc
        
        #print(FV)

        U = [i if i <= FV[0] else FV[0] for i in Unimportant]
        A = [i if i <= FV[1] else FV[1] for i in Average]
        I = [i if i <= FV[2] else FV[2] for i in Important]
    
        mfx = np.max(np.vstack((U,A,I)),0)
        #plt.plot(X,mfx)
        #plt.xlim([-0.1,1.1])
        #plt.show()
        score.append(fuzz.defuzz(X,mfx,'centroid'))

    N = NS[enum]
    N_best = [label[i].split('_')[1] for i in np.argsort(score)[::-1][:N]]
    N_best = sorted(N_best)

    numb = file.split('.')[0]
    f_in   = open(path_inp2+str(int(numb))+'.txt').read()
    f_out1 = open(path_out1+file,'w')
    f_out2 = open(path_out2+file,'w')

    v = 0
    for i,sent in enumerate(f_in.splitlines()):
        numb = '0'*(2-len(str(i-1)))+str(i-1)
        strg = 'sent_'+numb
        if i <= 1:
            f_out1.write(sent+'\n')
            f_out2.write(sent+'\n')
            continue
        
        if strg in label:
            vals = str(np.round(score[v],5))
            v += 1
        else:
            vals = '-'

        if numb in N_best:
            f_out2.write(sent+'\n')
        
        vals = vals+' '*(8-len(vals))
        sent = sent[:100]+'...' if len(sent) > 100 else sent
        f_out1.write(strg+' | '+vals+'| '+sent+'\n')
    f_out1.close()
    f_out2.close()
    
    print('time:',time.time()-start)
