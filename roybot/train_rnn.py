from keras.models import Model, load_model
from keras.layers import Dense, Input, LSTM
from keras import optimizers
import numpy as np
import time, os, re
import sys
import csv
import argparse

def import_params(params_file = 'params/params.csv'):
    params_list = []
    with open(params_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            params_list.append({})
            current_params = params_list[-1]
            current_params["epochs"] = int(row["epochs"])
            current_params["batch_size"] = int(row["batch_size"])
            current_params["latent_dim"] = int(row["latent_dim"])
            current_params["learning_rate"] = float(row["learning_rate"])
    return params_list

#def build_data(data_file):
input_texts = []
target_texts = []
with open('data/data3_5char.txt', 'r') as file:
    for line in file.readlines():
        try:
            in_chunk, out_chunk = line.split('\t')
        except Exception as e:
            print(e)
        input_texts.append(in_chunk)
        target_texts.append(out_chunk)
    #return input_texts, target_texts


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

print(len(input_texts), max_encoder_seq_length, num_encoder_tokens)
print(len(target_texts), max_decoder_seq_length, num_decoder_tokens)

input_char_to_i_dict = dict([(char, i) for i, char in enumerate(input_chars_list)])
target_char_to_i_dict = dict([(char, i) for i, char in enumerate(target_chars_list)])
print(target_char_to_i_dict)
input_i_to_char_dict = dict([(i, char) for i, char in enumerate(input_chars_list)])
target_i_to_char_dict = dict([(i, char) for i, char in enumerate(target_chars_list)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype=np.dtype('float32'))
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.dtype('float32'))
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.dtype('float32'))

for chunk_i, (input_chunk, target_chunk) in enumerate(zip(input_texts, target_texts)):
    for char_i, char in enumerate(input_chunk):
        encoder_input_data[chunk_i, char_i, input_char_to_i_dict[char]] = 1
    for char_i, char in enumerate(target_chunk):
        decoder_input_data[chunk_i, char_i, target_char_to_i_dict[char]] = 1
        if char_i > 0:
            decoder_target_data[chunk_i, char_i - 1, target_char_to_i_dict[char]] = 1

def rnn(input_texts, target_texts, params_dict, predict=False):
    BATCH_SIZE = params_dict['batch_size']
    EPOCHS = params_dict['epochs']
    LATENT_DIM = params_dict['latent_dim']
    LEARNING_RATE = params_dict['learning_rate']

    

    if not predict:
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder_lstm = LSTM(LATENT_DIM, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        #optimizer = optimizers.RMSprop()
        training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        try:
            training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=BATCH_SIZE, epochs = EPOCHS, validation_split=0.2)
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
        
        return training_model, encoder_model, decoder_model, encoder_input_data

    else:
        return encoder_input_data
    
def decode_sequence(input_seq, encoder_model, decoder_model):
    print(input_seq.shape)
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    #print(states_value)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        #print(output_tokens, h, c)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence




def decode(input_str):
    #char_dict = dict((char,i) for i, char in enumerate(set(input_str.lower())))
    input_seq = np.zeros((1,max_encoder_seq_length , num_encoder_tokens), dtype='float32')
    for i, char in enumerate(input_str):
        input_seq[0, i, input_token_index[char]] = 1
    return decode_sequence(input_seq)

"""

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
    user_input = input("Justin: ").lower()
    user_input_vector = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for i, char in enumerate(user_input):
        user_input_vector[0,i,input_char_to_i_dict[char]] = 1
    print("Roy: " + decode_sequence(user_input_vector))
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs=2, help="[data_file] [params_file] : (1) path to file containing training data, path to file contain with two tab-separated columns, the first being input and second being output, (2) path to params CSV")
    parser.add_argument('--predict', nargs=2, help="saved weights files")
    args = parser.parse_args()
    if args.train:
        params_list = import_params(args.train[1])
        #input_texts, output_texts = build_data(args.train[0])
        #training_model, encoder_model, decoder_model, encoder_input_data = rnn(input_texts, output_texts, params_list[0], True)
    if args.predict:
        encoder_model_file, decoder_model_file = args.predict
        encoder_input_data = rnn(input_texts, target_texts, params_list[0], True)
        encoder_model = load_model(encoder_model_file)
        decoder_model = load_model(decoder_model_file)

    #if not args.predict or args.train:
    #    print("Too few arguments")
    #    return 0
        
    for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

if __name__ == '__main__':
    main()
