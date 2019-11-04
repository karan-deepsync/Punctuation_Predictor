# Punctuation_Predictor
This punctuation predictor trained bi-directional LSTMs to learn how to automatically punctuate a sentence. The set of operation it learns include: comma, period and question mark.

## Dataset
The Blog Authorship Corpus - http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm

## Performance
punctuation   | precision  | recall | f1-score
--------------|-------|-------| --------
None          | 0.93  |  0.95 |  0.94
Comma         | 0.26  |  0.26 |  0.26
Period        | 0.41  |  0.33 |  0.36
Question mark | 0.14  |  0.11 |  0.12

## Example Run
Requirements:
- Python 3.x
- Numpy
- Keras

### Training
`tar zxvf input.tar.gz` to unzip the data
Training is done on blog data (xml file) stored in `input`.
- Parameters

`--output_directory`: output directory name, _default = "output"_

`--checkpoint_name`: _default = "blstm"_

`--vectorizer_name`: _default = "blstm"_

`--sequence_length`: sequence length for punctuating, _default=50_

`--file_number`: trained file number, _default = 350_

`--batch_size`: _default = 128_

`--epochs`: _default = 25_

`python train.py `

### Testing
- Parameters

`--input`: input string __only consist of lower case alphabet__

`--model_path`: _default="output/blstm.h5"_

`--vectorizer_path`: _default="output/blstm.pickle"_

`--sequence_length`: __should be the same as training sequence length__, _default=50_

`predictor.py --input "this is a string of text with no punctuation this is a new sentence"`

## Model Setup
I began with was a single uni-direction LSTM but it got confused with comma and period. After changing to bi-direction LSTM, the performance of period is much better.

## Future Work
- Hidden layer initialization - In most task, the neural network generate good results when starting with a zero initial state

- Chunking data in different way - I chunk the article in a fix size so a single sentance could be seperated in two chunks. It might be harder to predict the punctuation of the last word.

- Try different pretrained embedding model (Glove, Wang2vec etc) to capture semantic relatedness 

- Try different models (CNN, CRF)

- Self-attention 
