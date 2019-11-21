import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'VNTC/Data/10Topics/Ver1.1/Train_Full/')
DATA_TEST_PATH = os.path.join(DIR_PATH, 'VNTC/Data/10Topics/Ver1.1/Test_Full/')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data_train.json')
DATA_TEST_JSON = os.path.join(DIR_PATH, 'data_test.json')
STOP_WORDS = os.path.join(DIR_PATH, 'stopwords-nlp-vi.txt')
SPECIAL_CHARACTER = '0123456789%@$►.,=+-!;/()*"&^:#|\n\t\''
DICTIONARY_PATH = 'dictionary.txt'