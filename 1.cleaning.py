import re, os, numpy as np
from collections import Counter
from string import punctuation as punct
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()
factory = StopWordRemoverFactory()
stop_words = factory.get_stop_words()

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def punct_except(p):
    return ''.join(set(punct)-set(p))

def remove_punc(match):
    word = match.group(0)
    neww = word[1:-1]
    for p in '-/':
        neww = neww.replace(p,' ')
    return ' '+neww+' '

def remove_dash(match):
    word = match.group(0)
    return word[1:]

folder = './0.dataset raw/'

docs = []
flat = []
dstm = []
for file in os.listdir(folder):
    
    raw = open(folder+file).read()
    raw = raw.replace('\n',' \n ').lower()

    for p in '”“':
        raw = raw.replace(p,'"')

    for p in '’‘':
        raw = raw.replace(p,"'")

    for p in '—–':
        raw = raw.replace(p,'-')

    for p in punct_except("'-_"):
        raw = raw.replace(p,' ')

    raw = re.sub("'",'',raw)
    raw = re.sub('-\W|\W-',' ',raw)
    
    raw = re.sub('[ ][ ]+',' ',raw)
    raw = re.sub('\W\d+[/-]\d+\W',remove_punc,raw)
    raw = re.sub('[ ][ ]+',' ',raw)

    raw = re.sub('_comma_','.',raw)
    raw = re.sub('_per_','/',raw)

    raw = re.sub('-[km]u\W|-nya\W|-lah\W',remove_dash,raw)
    raw = re.sub('[ ][ ]+',' ',raw)

    while raw.endswith(('\n',' ')):
        raw = raw[:-1]

    tempdoc = []
    for no,sent in enumerate(raw.splitlines()):
        if no == 0:
            no = 'title   | '
        elif no == 1:
            no = ''
        else:
            no = '0'*(2-len(str(no-1)))+str(no-1)
            no = 'sent_'+no+' | '

        tempsent = []
        for word in word_tokenize(sent):
            
            if word in stop_words:
                continue
            
            flat.append(word)
            tempsent.append(word)
            
        tempdoc.append(no+' '.join(tempsent))
    docs.append('\n'.join(tempdoc))
 
vocab = sorted(set(flat))
f = open('vocabulary.txt','w')
f.write(' '.join(vocab))
f.close()

stem = open('dict_stem.txt').read()
stem = dict([i.split(' >> ') for i in stem.splitlines()])

V = len(vocab)
f = open('dict_stem.txt','w')
for i in vocab:
    try:
        f.write(i+' >> '+stem[i]+'\n')
    except:
        f.write(i+' >> '+stemmer.stem(i)+'\n')
f.close()

word_edit = open('word_edit.txt').read()
word_edit = dict([i.split(' >> ') for i in word_edit.splitlines()])

f = open('word_edit.txt','w')
for i in vocab:
    try:
        f.write(i+' >> '+word_edit[i]+'\n')
    except:
        print(i,'berhasil ditambahkan ke word_edit')
        word_edit[i] = i
        f.write(i+' >> '+i+'\n')
f.close()

stem_edit = open('stem_edit.txt').read()
stem_edit = dict([i.split(' >> ') for i in stem_edit.splitlines()])

f = open('stem_edit.txt','w')
for i in vocab:
    for token in word_edit[i].split('_'):
        if token in stop_words: continue
        try:
            f.write(token+' >> '+stem_edit[token]+'\n')
        except:
            print(token,'berhasil ditambahkan ke stem_edit')
            stemtok = stemmer.stem(token)
            stem_edit[token] = stemtok
            f.write(token+' >> '+stemtok+'\n')
f.close()

prop_edit = open('propernoun_edit.txt').read()
prop_edit = dict([i.split(' >> ') for i in prop_edit.splitlines()])

f = open('propernoun_edit.txt','w')
for i in vocab:
    try:
        f.write(i+' >> '+prop_edit[i]+'\n')
    except:
        print(i,'berhasil ditambahkan ke prop_edit')
        prop_edit[i] = '0'
        f.write(i+' >> 0\n')
f.close()

for i,doc in enumerate(docs):
    temp2 = []
    for sent in doc.splitlines():
        try: labs,sent = sent.split(' | ')
        except: continue
        temp1 = []
        for word in sent.split():
            for word in word_edit[word].split('_'):
                if word in stop_words: continue
                word = stem_edit[word]
                temp1.append(word)
        temp2.append(labs+' | '+' '.join(temp1))
    data = '\n'.join(temp2)

    name = '0'*(3-len(str(i+1)))+str(i+1)+'.txt'
    f = open('./1.clean/'+name,'w')
    f.write(data)
    f.close()
    
