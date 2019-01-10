from keras.models import Model, load_model
from keras.layers import Dense, Input, LSTM
from keras import optimizers
import numpy as np
import time, os, re
import sys

    
if len(sys.argv) == 2:
    DATA_FILE = sys.argv[1]
elif len(sys.argv) == 1:
    DATA_FILE = './data2.txt'
else:
    sys.argv[10000] #shortcut way to raise an exception lol

#grab data
input_texts = []
target_texts = []
with open(DATA_FILE, 'r') as file:
    for line in file.readlines():
        in_chunk, out_chunk = line.split('\t')
        input_texts.append(in_chunk)
        target_texts.append(out_chunk)


#Set up neural network
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256
LEARNING_RATE = 0.001


#input_texts = justin_chunks[:len(justin_chunks)//3]
#target_texts = roy_chunks[:len(justin_chunks)//3]

#print("Lengths: {0}, {1}".format(len(justin_chunks)//3,len(roy_chunks)//3))
#using \n as start char and \t as stop char (REVERSED NOW)
for seq in range(len(target_texts)):
    target_texts[seq] = "\t" + target_texts[seq] + "\n"

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

data_chunks = 2
    
print(len(input_texts), max_encoder_seq_length, num_encoder_tokens)
print(len(target_texts), max_decoder_seq_length, num_decoder_tokens)



input_char_to_i_dict = dict([(char, i) for i, char in enumerate(input_chars_list)])
target_char_to_i_dict = dict([(char, i) for i, char in enumerate(target_chars_list)])
print(target_char_to_i_dict)
input_i_to_char_dict = dict([(i, char) for i, char in enumerate(input_chars_list)])
target_i_to_char_dict = dict([(i, char) for i, char in enumerate(target_chars_list)])

file = open('hehe.txt', 'w')
file.write(str(target_char_to_i_dict.keys()))

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype=np.dtype('?'))
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.dtype('?'))
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.dtype('?'))

for chunk_i, (input_chunk, target_chunk) in enumerate(zip(input_texts, target_texts)):
    for char_i, char in enumerate(input_chunk):
        encoder_input_data[chunk_i, char_i, input_char_to_i_dict[char]] = 1
    for char_i, char in enumerate(target_chunk):
        decoder_input_data[chunk_i, char_i, target_char_to_i_dict[char]] = 1
        if char_i > 0:
            decoder_target_data[chunk_i, char_i - 1, target_char_to_i_dict[char]] = 1

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(LATENT_DIM, return_state=True)
encoder_ouputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
optimizer = optimizers.RMSprop(lr=LEARNING_RATE)
training_model.compile(optimizer=optimizer, loss='categorical_crossentropy')
try:
    training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=BATCH_SIZE, epochs = EPOCHS, validation_split=0.3)
except KeyboardInterrupt:
    pass


training_model.save('seq2seq_weights/'+str(time.ctime()).replace(' ','_') + 'TRAINING_MODEL.h5')

#defining encoder/decoder models 
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

encoder_model.save('seq2seq_weights/'+str(time.ctime()).replace(' ','_') + '_ENCODER_MODEL.h5')
decoder_model.save('seq2seq_weights/'+str(time.ctime()).replace(' ','_') + '_DECODER_MODEL.h5')

def decode_sequence(input_seq):
    print(input_seq.shape)
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1,num_decoder_tokens))
    target_seq[0,0,target_char_to_i_dict['\t']] = 1
    stop_condition = False
    output_string = ''
    while not stop_condition:
        output_tokens, state_h, state_c = decoder_model.predict([target_seq] + states_value)
        output_tokens_index = np.argmax(output_tokens[0,-1,:])
        output_char = target_i_to_char_dict[output_tokens_index]
        #print(output_char)
        if output_char == '\n' or len(output_string) > max_decoder_seq_length:
            stop_condition = True
        
        output_string += output_char
        target_seq = np.zeros((1,1,num_decoder_tokens))
        target_seq[0,0,output_tokens_index] = 1
        state_values = [state_h, state_c]
    return output_string


while(True):
    user_input = input("Justin: ")
    user_input_vector = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for i, char in enumerate(user_input):
        user_input_vector[0,i,input_char_to_i_dict[char]] = 1
    print("Roy: " + decode_sequence(user_input_vector))


        
        
        
        
