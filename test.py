# import libararies
import numpy as np
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
import os
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout, Bidirectional, GRU, Convolution1D, Reshape
from keras.optimizers import Adam

import numpy as np
from random import randint
import os
import json
import settings
import pickle
import nltk.data
from pyvi import ViTokenizer
from sklearn.svm import LinearSVC
from gensim import corpora, matutils
from sklearn.metrics import classification_report
from keras.models import load_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import KeyedVectors


def feature_extract(type_extract, X_data, y_data, X_test, Y_test):
	svd = ""
	extract = ""
	if type_extract == "count_vector":
	    count_vect = CountVectorizer(analyzer='word', token_pattern=r'/w{1,}')
	    count_vect.fit(X_data)
	    # transform the training and validation data using count vectorizer object
	    X_data_count = count_vect.transform(X_data)
	    X_test_count = count_vect.transform(X_test)
	    X_data = X_data_count
	    X_test = X_test_count


	if type_extract == "word_level_tf_idf":
	    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
	    tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
	    X_data_tfidf =  tfidf_vect.transform(X_data)
	    # assume that we don't have test set before
	    X_test_tfidf =  tfidf_vect.transform(X_test)
	    svd = TruncatedSVD(n_components=300, random_state=42)
	    svd.fit(X_data_tfidf)

	    X_data_tfidf_svd = svd.transform(X_data_tfidf)
	    X_test_tfidf_svd = svd.transform(X_test_tfidf)

	    X_data = X_data_tfidf_svd
	    X_test = X_test_tfidf_svd
	    svd_extract = svd
	    extract = tfidf_vect


	if type_extract == "n_gram_level_tf_idf":
	    # ngram level - we choose max number of words equal to 30000 except all words (100k+ words)
	    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
	    tfidf_vect_ngram.fit(X_data)
	    X_data_tfidf_ngram =  tfidf_vect_ngram.transform(X_data)
	    # assume that we don't have test set before
	    X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

	    svd_ngram = TruncatedSVD(n_components=300, random_state=42)
	    svd_ngram.fit(X_data_tfidf_ngram)

	    X_data_tfidf_ngram_svd = svd_ngram.transform(X_data_tfidf_ngram)
	    X_test_tfidf_ngram_svd = svd_ngram.transform(X_test_tfidf_ngram)

	    X_data = X_data_tfidf_ngram_svd
	    X_test = X_test_tfidf_ngram_svd
	    svd_extract = svd_ngram
	    extract = tfidf_vect_ngram


	if type_extract == "character_level_tf_idf":
	    # ngram-char level - we choose max number of words equal to 30000 except all words (100k+ words)
	    tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 3))
	    tfidf_vect_ngram_char.fit(X_data)
	    X_data_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_data)
	    # assume that we don't have test set before
	    X_test_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_test)

	    svd_ngram_char = TruncatedSVD(n_components=300, random_state=42)
	    svd_ngram_char.fit(X_data_tfidf_ngram_char)

	    X_data_tfidf_ngram_char_svd = svd_ngram_char.transform(X_data_tfidf_ngram_char)
	    X_test_tfidf_ngram_char_svd = svd_ngram_char.transform(X_test_tfidf_ngram_char)

	    X_data = X_data_tfidf_ngram_char_svd
	    X_test = X_test_tfidf_ngram_char_svd
	    svd_extract = svd_ngram_char
	    extract = tfidf_vect_ngram_char


	if type_extract == "word_embedding":
	    word2vec_model_path = ""
	    w2v = KeyedVectors.load_word2vec_format(word2vec_model_path)
	    vocab = w2v.wv.vocab
	    wv = w2v.wv

	    def get_word2vec_data(X):
	        word2vec_data = []
	        for x in X:
	            sentence = []
	            for word in x.split(" "):
	                if word in vocab:
	                    sentence.append(wv[word])

	            word2vec_data.append(sentence)

	        return word2vec_data
	    
	    X_data_w2v = get_word2vec_data(X_data)
	    X_test_w2v = get_word2vec_data(X_test)

	    X_data = X_data_w2v
	    X_test = X_test_w2v

	return X_data, X_test, svd, extract


def load_data(path_X_data, path_y_data, path_X_test, path_y_test):
    X_data = pickle.load(open(path_X_data, 'rb'))
    y_data = pickle.load(open(path_y_data, 'rb'))

    X_test = pickle.load(open(path_X_test, 'rb'))
    y_test = pickle.load(open(path_y_test, 'rb'))

    return X_data, y_data, X_test, y_test


class NLP(object):
    def __init__(self, text = None):
        self.text = text
        self.__set_stopwords()

    def __set_stopwords(self):
        self.stopwords = FileReader(settings.STOP_WORDS).read_stopwords()


    def split_words(self):
        try:
            return [x.strip(settings.SPECIAL_CHARACTER).lower() for x in self.text.split()]
        except TypeError:
            return []

    def get_words_feature(self):
        split_words = self.split_words()
        return [word for word in split_words if word.encode('utf-8') not in self.stopwords]
    
class FileReader(object):
    def __init__(self, filePath, encoder = None):
        self.filePath = filePath
        self.encoder = encoder if encoder != None else 'utf-16'

    def read_stopwords(self):
        with open(self.filePath, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        return stopwords


def encode_class(y_data, y_test):
    encoder = preprocessing.LabelEncoder()
    y_data_n = encoder.fit_transform(y_data)
    y_test_n = encoder.fit_transform(y_test)

    return y_data_n, y_test_n, encoder.classes_


if __name__ == "__main__":
    X_data, y_data, X_test, y_test = load_data("D:/topic_classification/X_data1.pkl", "D:/topic_classification/y_data1.pkl", "D:/topic_classification/X_test1.pkl", "D:/topic_classification/y_test1.pkl")
    X_data, X_test, svd, extract = feature_extract("word_level_tf_idf", X_data, y_data, X_test, y_test)
    y_data_n, y_test_n, class_name = encode_class(y_data, y_test)

    data = []
    file = open("D:/topic_classification/test.txt", "rb")
    query_str = ""
    for line in file:
        query_str = line
        
    words = NLP(query_str).get_words_feature()
    words = ' '.join(words)
    words = gensim.utils.simple_preprocess(words)
    words = ' '.join(words)
    words = ViTokenizer.tokenize(words)

    data.append(words)
    data = extract.transform(data)
    data = svd.transform(data)
    classifier = load_model("D:/topic_classification/trained_dnn_model_700_epoch_val_acc_0.85_test_acc_0.8446875.h5")

    class_predict = classifier.predict(data)
    print(class_predict)
    print(class_name)

