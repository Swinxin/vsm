# -*- coding: utf-8 -*-
"""
@author: Administrator
"""
import math
import numpy as np
from collections import Counter
corpus = ['Julie loves me more than Linda loves me',
'Jane likes me more than Julie loves me',
'He likes basketball more than baseball']

'''
for doc in corpus:
    tf = Counter()
    for word in doc.split():
        tf[word] += 1
    print tf.items() #向量的长度只是单个文本长度
'''        
def buildVocabuary(corpus):
    """
    利用set建立词典
    """
    lexicon = set()
    for doc in corpus:
        for word in doc.split():
            lexicon.add(word)
    return lexicon
def tf(term,document):
    """
    term frequence
    """
    return freq(term,document)

def freq(term,document):
    return document.split().count(term)

def tfMatrix(corpus = corpus,normalize = False):
    vocabuary = buildVocabuary(corpus)
    print "size of vocabuary is",len(vocabuary)
    termMatrix = []
    for doc in corpus:
        tf_vector = [tf(word,doc) for word in vocabuary]
        if normalize:
            tf_vector = l2_normalizer(tf_vector)
        termMatrix.append(tf_vector)
    return termMatrix
def l2_normalizer(vec):
    """
    对向量标准化
    """
    denom = sum([v**2 for v in vec])
    return [v / math.sqrt(denom) for v in vec]
def wordNumInDoc(word,corpus):
    """
    出现word 的文档数
    """
    count = 0
    for doc in corpus:
        if freq(word,doc) > 0:
            count += 1
    return count
    
def idf(word,corpus):
    docNum = len(corpus)
    df = wordNumInDoc(word,corpus)
    #语料中文档的总数 ÷ 包含该词语的文档数
    return np.log(docNum / df+1)#?????????
def idfMatrix(corpus = corpus):
    voc = buildVocabuary(corpus)
    idfVec = [idf(word,corpus) for word in voc]
    return idfVec
    
def tfifd(corpus = corpus):
    tfMat = tfMatrix(corpus,normalize=True)
    idfMat = idfMatrix(corpus)
    n = len(idfMat)
    idf = np.zeros((n,n))
    np.fill_diagonal(idf,idfMat) #将idf填充到 zero矩阵的对角线上
    return np.dot(np.mat(tfMat),idf) # 点乘
if __name__ == "__main__":
    print tfifd()