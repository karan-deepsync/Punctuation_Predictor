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
`--input`: input string
`--model_path`: _default="output/blstm.h5"_
`--vectorizer_path`: _default="output/blstm.pickle"_
`--sequence_length`: __should be the same as training sequence length__, _default=50_

`predictor.py --input "this is a string of text with no punctuation this is a new sentence"`


