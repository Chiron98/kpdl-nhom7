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


def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)

                X.append(lines)
                y.append(path)
    return X, y


def save_data(X_data, y_data, path_x, path_y):
    pickle.dump(X_data, open(path_x, 'wb'))
    pickle.dump(y_data, open(path_y, 'wb'))


def load_data(path_X_data, path_y_data, path_X_test, path_y_test):
    X_data = pickle.load(open(path_X_data, 'rb'))
    y_data = pickle.load(open(path_y_data, 'rb'))

    X_test = pickle.load(open(path_X_test, 'rb'))
    y_test = pickle.load(open(path_y_test, 'rb'))

    return X_data, y_data, X_test, y_test


def feature_extract(type_extract, X_data, y_data, X_test, Y_test):
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
    
    return X_data, X_test


def encode_class(y_data, y_test):
    encoder = preprocessing.LabelEncoder()
    y_data_n = encoder.fit_transform(y_data)
    y_test_n = encoder.fit_transform(y_test)

    return y_data_n, y_test_n


def create_rcnn_model():
    input_layer = Input(shape=(300,))
    
    layer = Reshape((10, 30))(input_layer)
    layer = Bidirectional(GRU(128, activation='relu', return_sequences=True))(layer)    
    layer = Convolution1D(100, 3, activation="relu")(layer)
    layer = Flatten()(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    
    output_layer = Dense(23, activation='softmax')(layer)
    
    classifier = Model(input_layer, output_layer)
    classifier.summary()
    classifier.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return classifier


def create_dnn_model():
    input_layer = Input(shape=(300,))
    layer = Dense(1024, activation='relu')(input_layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    output_layer = Dense(23, activation='softmax')(layer)
    
    classifier = Model(input_layer, output_layer)
    classifier.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return classifier


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


def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    
    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
        
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)
    
        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        
    print("Validation accuracy: ", accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", accuracy_score(test_predictions, y_test))

    return classifier, accuracy_score(val_predictions, y_val), accuracy_score(test_predictions, y_test)


if __name__ == "__main__":
    # file = open("/home/tuananh/tuananh/topic_classification/data/topic_detection_train.v1.0.txt", "r")
    # X_data = []
    # y_data = []
    # X_test = []
    # y_test = []
    # number = 0
    # for line in file:
    #     if number < 12800:
    #         words = NLP(line).get_words_feature()
    #         y_data.append(words[0])
    #         words = ' '.join(words)
    #         words = gensim.utils.simple_preprocess(words)
    #         words = ' '.join(words)
    #         words = ViTokenizer.tokenize(words)
    #         X_data.append(words)
    #         number += 1
    #         print(words)
    #     else:
    #         words = NLP(line).get_words_feature()
    #         y_test.append(words[0])
    #         words = ' '.join(words)
    #         words = gensim.utils.simple_preprocess(words)
    #         words = ' '.join(words)
    #         words = ViTokenizer.tokenize(words)
    #         X_test.append(words)
    #         number += 1
    #         print(words)
    

    # save_data(X_data, y_data, "data/X_data1.pkl", "data/y_data1.pkl")
    # save_data(X_test, y_test, "data/X_test1.pkl", "data/y_test1.pkl")

    X_data, y_data, X_test, y_test = load_data("D:/topic_classification/X_data1.pkl", "D:/topic_classification/y_data1.pkl", "D:/topic_classification/X_test1.pkl", "D:/topic_classification/y_test1.pkl")
    X_data, X_test = feature_extract("word_level_tf_idf", X_data, y_data, X_test, y_test)
    y_data_n, y_test_n = encode_class(y_data, y_test)
    
    # classifier = create_dnn_model()
    classifier = load_model("D:/topic_classification/trained_dnn_model_700_epoch_val_acc_0.85_test_acc_0.8446875.h5")
    classifier, val_acc, test_acc = train_model(classifier=classifier, X_data=X_data, y_data=y_data_n, X_test=X_test, y_test=y_test_n, is_neuralnet=True, n_epochs=1000)

    classifier.save('weights/trained_dnn_model_{}_epoch_val_acc_{}_test_acc_{}.h5'.format(str(3000), str(val_acc), str(test_acc)))

