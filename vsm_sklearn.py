# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:41:10 2016

@author: Administrator
"""

corpus = ['Julie loves me more than Linda loves me',
'Jane likes me more than Julie loves me',
'He likes basketball more than baseball']

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(min_df=1)
term_freq_matrix = count_vectorizer.fit_transform(corpus)
print "Vocabulary:", count_vectorizer.vocabulary_

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(term_freq_matrix)
tf_idf_matrix = tfidf.transform(term_freq_matrix)
print tf_idf_matrix.todense()
print "-------------"
#使用tfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer  
tfidf_vectorizer = TfidfVectorizer(min_df = 1)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

print tfidf_matrix.todense()