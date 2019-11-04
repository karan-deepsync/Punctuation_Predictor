from train import Transform
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from optparse import OptionParser

import numpy as np
import pickle

def main(options): 
    BLSTM_CHECKPOINT_NAME = options.model_path
    TOKENIZER_PATH = options.vectorizer_path
    MAX_SEQUENCE_LENGTH = options.sequence_length
    INPUT = options.input
    blstm_model = load_model(BLSTM_CHECKPOINT_NAME)

    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    label_index = {0:0, 1:1, 2:2, 3:3}
    if not INPUT:
        str_input = 'this is a string of text with no punctuation this is a new sentence' 
        #str_input = 'halloween is officially behind us which means for the next two months or so it is going to be all christmas all the time but before you get sick of the overplayed music and the excessive gift buying why not take advantage and celebrate a little canadas wonderland is launching a brand new winter festival at the end of this month and it just might be the perfect place to geek out and enjoy some holiday cheer wonderland announced the new festival in the summer of 2018 and they revealed the official launch date about a month ago but as new details about the festival continue to emerge it becomes more and more clear that it is sure to be a can not miss event this holiday season'
    else:
        str_input = options.input
        
    str_split = str_input.split()
    str_chunk = [str_split[i:i + MAX_SEQUENCE_LENGTH] for i in range(0, len(str_split), MAX_SEQUENCE_LENGTH)]
    str_numeric = np.array(tokenizer.texts_to_sequences(str_chunk))
    str_pad = pad_sequences(str_numeric, MAX_SEQUENCE_LENGTH, padding='post')
    blstm_str_pred = blstm_model.predict(str_pad, batch_size=64, verbose=1)
    blstm_str_trans = Transform(blstm_str_pred, label_index)

    result = []
    for row, chunk in enumerate(str_chunk):
        for col, word in enumerate(chunk):
            if blstm_str_trans[row][col] == 0:
                result.append(word)
            if blstm_str_trans[row][col] == 1:
                result.append(word)
                result.append('<comma>')
            if blstm_str_trans[row][col] == 2:
                result.append(word)
                result.append('<period>')
            if blstm_str_trans[row][col] == 3:
                result.append(word)
                result.append('<question_mark>')
    print(' '.join(result))



if __name__ == "__main__":
    usage = "usage: %prog [-s infile] [option]"
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--input", dest="input", default=None, help="input string", type="string")
    parser.add_option("-m", "--model_path", dest="model_path", default="output/blstm.h5", help="trained model checkpoint path", type="string")
    parser.add_option("-v", "--vectorizer_path", dest="vectorizer_path", default="output/blstm.pickle", help="trained vectorizer path", type="string")
    parser.add_option("-s", "--sequence_length", dest="sequence_length", default=50, help="sequence length for punctuating", type="int")

    (options, args) = parser.parse_args()
    main(options)