import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

DATA_FILE = './data1.txt'

#grab data
input_texts = []
target_texts = []
with open(DATA_FILE, 'r') as file:
    for line in file.readlines():
        in_chunk, out_chunk = line.split('\t')
        input_texts.append(in_chunk.replace("\n", ""))
        target_texts.append(out_chunk.replace("\n", ""))

input_chars_list = set()
target_chars_list = set()

for chunk in input_texts:
    for char in chunk:
        input_chars_list.add(char)
        
for chunk in target_texts:
    for char in chunk:
        target_chars_list.add(char)

print(target_chars_list)
input_chars_list = sorted(list(input_chars_list))
target_chars_list = sorted(list(target_chars_list))
num_encoder_tokens = len(input_chars_list)
num_decoder_tokens = len(target_chars_list)
max_encoder_seq_length = max([len(chunk) for chunk in input_texts])
max_decoder_seq_length = max([len(chunk) for chunk in target_texts])

    
print(len(input_texts), max_encoder_seq_length, num_encoder_tokens)
print(len(target_texts), max_decoder_seq_length, num_decoder_tokens)

input_char_to_i_dict = dict([(char, i) for i, char in enumerate(input_chars_list)])
target_char_to_i_dict = dict([(char, i) for i, char in enumerate(target_chars_list)])
print(target_char_to_i_dict)
input_i_to_char_dict = dict([(i, char) for i, char in enumerate(input_chars_list)])
target_i_to_char_dict = dict([(i, char) for i, char in enumerate(target_chars_list)])


encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype=np.dtype('?'))
decoder_input_data = np.zeros((len(input_texts), num_decoder_tokens), dtype=np.dtype('?'))
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.dtype('?'))

for chunk_i, (input_chunk, target_chunk) in enumerate(zip(input_texts, target_texts)):
    for char_i, char in enumerate(input_chunk):
        encoder_input_data[chunk_i, char_i, input_char_to_i_dict[char]] = 1
    try:
        target_char = target_chunk[0]
    except:
        target_char = " " 
    decoder_input_data[chunk_i, target_char_to_i_dict[target_char]] = 1

    
"""
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab

# prepare the dataset of input to output pairs encoded as integers

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
"""

X = encoder_input_data
y = decoder_input_data

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# load the network weights
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
try:
    model.fit(X, y, epochs=500, batch_size=64, callbacks=callbacks_list)
except KeyboardInterrupt:
    pass

def into_vector(input_str):
    user_input_vector = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
    for i, char in enumerate(input_str):
        user_input_vector[0, i, input_char_to_i_dict[char]] = 1
    return user_input_vector

def decode_sequence(input_seq, length=50):
    current_seq = input_seq
    current_output = ""
    for i in range(length):
        user_input_vector = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
        for i, char in enumerate(current_seq):
            user_input_vector[0, i, input_char_to_i_dict[char]] = 1
            
        #input_vector = into_vector(current_seq)
        #print(input_vector.shape)
        pred_softmax = model.predict(user_input_vector, verbose=False, steps=1)
        pred_char = target_i_to_char_dict[np.argmax(pred_softmax)]
        current_output += pred_char
        #print(current_output)
        current_seq = current_seq[1:] + pred_char
    return current_output


while(True):
    user_input = input("Justin: ")
    print("Roy: " + decode_sequence(user_input))

"""

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "\nDone."
"""
