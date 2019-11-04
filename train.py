from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.utils import class_weight, compute_sample_weight
from optparse import OptionParser

import xml.etree.ElementTree as ET
import numpy as np
import re
import glob
import random
import os
import pickle

def Transform(sequences, index):
    label_sequences = []
    for categorical_sequence in sequences:
        label_sequence = []
        for categorical in categorical_sequence:
            label_sequence.append(index[np.argmax(categorical)])
        label_sequences.append(label_sequence)
    return label_sequences

def main(options):
    OUTPUT_DIR = options.output_directory
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
       
    try:
        FILENAMES = glob.glob(options.input_file)
    except:
        FILENAMES = options.input_file

    CHECKPOINT_NAME = options.checkpoint_name
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, '{}.h5'.format(CHECKPOINT_NAME))
    VECTORIZER_NAME = options.vectorizer_name
    VECTORIZER_PATH = os.path.join(OUTPUT_DIR, '{}.pickle'.format(VECTORIZER_NAME))
    MAX_SEQUENCE_LENGTH = options.sequence_length
    NUM_FILES = options.file_number
    NUM_EPOCH = options.epochs
    BATCH_SIZES = options.batch_size
    MAX_NUM_WORDS = 20000
    TRAINING_SPLIT = 0.8

    print("Data preprocessing...")
    paragraphs = []
    for filename in FILENAMES[:NUM_FILES]:
        tree = ET.parse(filename)
        root = tree.getroot()
        for wordElement in root.iter('post'):
            text = wordElement.text.lower()
            text = text.strip()
            text = re.sub(r"what's", "what is ", text)
            text = re.sub(r"\'s", " is", text)
            text = re.sub(r"\'ve", " have ", text)
            text = re.sub(r"can't", "cannot ", text)
            text = re.sub(r"n't", " not ", text)
            text = re.sub(r"i'm", "i am ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            text = re.sub(r'[.]+', ".", text)
            text = re.sub(r'[?]+', "?", text)
            text = re.sub(r'[!]+', ".", text)
            text = re.sub(r'[:]+', ",", text)
            text = re.sub(r'[;]+', ",", text)
            text = re.sub(r'[^.,\?a-zA-Z ]', '', text)
            paragraphs.append(text)
    
    words_labels_join = []
    for paragraph in paragraphs:
        words = paragraph.split()
        for word in words:
            if re.match("^[a-z]+$", word):
                words_labels_join.append((word, 0))
            if re.match("^[a-z]+,$", word):
                words_labels_join.append((word[:-1], 1))
            if re.match("^[a-z]+\.$", word):
                words_labels_join.append((word[:-1], 2))
            if re.match("^[a-z]+\?$", word):
                words_labels_join.append((word[:-1] , 3))
    print("Total number of words: %d" % len(words_labels_join))

    words_labels_chunk = [words_labels_join[i:i + MAX_SEQUENCE_LENGTH] for i in range(0, len(words_labels_join), MAX_SEQUENCE_LENGTH)]
    words_labels_chunk = words_labels_chunk[:-1]
    random.shuffle(words_labels_chunk)
    features = [[x[0] for x in sublist] for sublist in words_labels_chunk]
    labels = [[x[1] for x in sublist] for sublist in words_labels_chunk]

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token=1)
    tokenizer.fit_on_texts(features)
    features_numeric = tokenizer.texts_to_sequences(features)

    with open(VECTORIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Splitting Data
    length = len(features_numeric)
    train_feature = features_numeric[:int(length*0.8)]
    test_feature = features_numeric[int(length*0.8):]
    train_label = labels[:int(length*0.8)]
    test_label = labels[int(length*0.8):]

    train_label = to_categorical(np.asarray(train_label))
    test_label = to_categorical(np.asarray(test_label))

    label_index = {0:0, 1:1, 2:2, 3:3}
    
    print("Building model...")
    blstm_model = Sequential()
    blstm_model.add(InputLayer(input_shape=(MAX_SEQUENCE_LENGTH, )))
    blstm_model.add(Embedding(MAX_NUM_WORDS, 64))
    blstm_model.add(Bidirectional(LSTM(64, return_sequences=True)))
    blstm_model.add(Bidirectional(LSTM(64, return_sequences=True)))
    #blstm_model.add(Dropout(0.2))
    blstm_model.add(TimeDistributed(Dense(4)))
    blstm_model.add(Activation('softmax'))
    blstm_model.compile(loss='categorical_crossentropy',
                optimizer=Adam(0.001),
                metrics=['accuracy'],
                sample_weight_mode='temporal')
    #blstm_model.summary()

    print("Training...")
    blstm_model.fit(np.array(train_feature), 
          train_label, 
          batch_size=BATCH_SIZES, 
          epochs=NUM_EPOCH,
          validation_split=0.2)
    blstm_model.save(CHECKPOINT_PATH) 

    print("Testing...")
    blstm_y_pred = blstm_model.predict(np.array(test_feature), batch_size=BATCH_SIZES, verbose=1)
    text_label_trans = Transform(test_label, label_index)
    blstm_y_pred_trans = Transform(blstm_y_pred, label_index)
    print(classification_report(np.array(text_label_trans).flatten(), np.array(blstm_y_pred_trans).flatten(), labels=[0, 1, 2, 3]))

if __name__ == "__main__":
    usage = "usage: %prog [-s infile] [option]"
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--input_file", dest="input_file", default="input/*.xml", help="input file (xml)", type="string")
    parser.add_option("-d", "--output_directory", dest="output_directory", default="output", help="output directory name", type="string")
    parser.add_option("-o", "--checkpoint_name", dest="checkpoint_name", default="blstm", help="model check point file name", type="string")
    parser.add_option("-v", "--vectorizer_name", dest="vectorizer_name", default="blstm", help="vectorizer pickle file name", type="string")
    parser.add_option("-s", "--sequence_length", dest="sequence_length", default=50, help="sequence length for punctuating", type="int")
    parser.add_option("-f", "--file_number", dest="file_number", default=350, help="trained files", type="int")
    parser.add_option("-b", "--batch_size", dest="batch_size", default=128, help="batch size", type="int")
    parser.add_option("-e", "--epochs", dest="epochs", default=25, help="epochs", type="int")
    (options, args) = parser.parse_args()
    main(options)