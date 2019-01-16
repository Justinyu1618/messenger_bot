from keras.models import Model, load_model
from keras.layers import Dense, Input, LSTM
from keras import optimizers
import numpy as np
import time, os, re
import sys
import csv
import argparse

global input_chars_list, target_chars_list
global num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length
global input_char_to_i_dict, target_char_to_i_dict, input_i_to_char_dict, target_i_to_char_dict
global encoder_input_data, decoder_input_data, decoder_target_data
global input_texts, target_texts

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

def build_data(data_file):
    global input_texts, target_texts
    input_texts = []
    target_texts = []
    with open(data_file, 'r') as file:
        for line in file.readlines():
            try:
                in_chunk, out_chunk = line.split('\t')
                input_texts.append(in_chunk)
                target_texts.append(out_chunk)
            except Exception as e:
                print(e)
            
    return input_texts, target_texts

def process_data(input_texts, target_texts):
    global input_chars_list, target_chars_list
    global num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length
    global input_char_to_i_dict, target_char_to_i_dict, input_i_to_char_dict, target_i_to_char_dict
    global encoder_input_data, decoder_input_data, decoder_target_data
    
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
                

def rnn(params_dict, predict=False):
    BATCH_SIZE = params_dict['batch_size']
    EPOCHS = params_dict['epochs']
    LATENT_DIM = params_dict['latent_dim']
    LEARNING_RATE = params_dict['learning_rate']

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

    #defining encoder/decoder models 
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(LATENT_DIM,))
    decoder_state_input_c = Input(shape=(LATENT_DIM,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
    
    return training_model, encoder_model, decoder_model, encoder_input_data

def decode_sequence(input_seq, encoder_model, decoder_model):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_char_to_i_dict['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_i_to_char_dict[sampled_token_index]
        decoded_sentence += sampled_char
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    return decoded_sentence
"""

def decode_sequence(input_seq,encoder_model, decoder_model):
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
"""

def decode(input_str):
    #char_dict = dict((char,i) for i, char in enumerate(set(input_str.lower())))
    input_seq = np.zeros((1,max_encoder_seq_length , num_encoder_tokens), dtype='float32')
    for i, char in enumerate(input_str):
        input_seq[0, i, input_token_index[char]] = 1
    return decode_sequence(input_seq)


def sample_test(num_tests, encoder_model, decoder_model, save=None, save_dir=None):
    try:
        file = open(f'{save_dir}/log_training_samples', 'w')
    except Exception:
        pass
    for seq_index in range(num_tests):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model)
        if save:
            file.write('-\n')
            file.write(f'Input sentence: {input_texts[seq_index]}\n')
            file.write(f'Decoded sentence: {decoded_sentence}\n')
        else:
            print('-')
            print('Input sentence:', input_texts[seq_index])
            print('Decoded sentence:', decoded_sentence)

def launch_cli(encoder_model, decoder_model):
    while(True):
        user_input = input("Justin: ").lower()
        user_input_vector = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        for i, char in enumerate(user_input):
            user_input_vector[0,i,input_char_to_i_dict[char]] = 1
        print("Roy: " + decode_sequence(user_input_vector, encoder_model, decoder_model))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs=3, help="[data_file] [params_file] [save_dir]: (1) path to file containing training data, path to file contain with two tab-separated columns, the first being input and second being output, (2) path to params CSV (3) Directory to save models")
    parser.add_argument('--predict', nargs=1, help="saved weights files dir")
    args = parser.parse_args()
    if args.train:
        params_list = import_params(args.train[1])
        input_texts, target_texts = build_data(args.train[0])
        process_data(input_texts, target_texts)
        for run in range(len(params_list)):
            training_model, encoder_model, decoder_model, encoder_input_data = rnn(params_list[run], True)
            save_dir = args.train[2]
            i = sorted([int(f.name) for f in os.scandir(save_dir) if f.is_dir()])
            print(i)
            if i:
                i = str(int(i[-1]) + 1)
            else:
                i = '0'
            os.makedirs(f'{save_dir}/{i}')
            training_model.save(f'{save_dir}/{i}/training.h5')
            encoder_model.save(f'{save_dir}/{i}/encoder.h5')
            decoder_model.save(f'{save_dir}/{i}/decoder.h5')
            with open(f'{save_dir}/{i}/data_pointer', 'w') as data_file:
                data_file.write(args.train[0])
            with open(f'{save_dir}/log.csv', 'a') as csv_file:
                params_dict = params_list[run]
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow([i,
                                 args.train[0],
                                 params_dict['epochs'],
                                 params_dict['batch_size'],
                                 params_dict['latent_dim'],
                                 params_dict['learning_rate']])
            sample_test(10, encoder_model, decoder_model)
            sample_test(200, encoder_model, decoder_model, save=True, save_dir=f'{save_dir}/{i}')
            print(f'Saving to directory {save_dir}/{i}')
        
    elif args.predict:
        file_dir = args.predict[0]
        training_model = load_model(file_dir + '/training.h5')
        encoder_model = load_model(file_dir + '/encoder.h5')
        decoder_model = load_model(file_dir + '/decoder.h5')
        data_file = open(file_dir + '/data_pointer').readline()
        input_texts, target_texts = build_data(data_file)
        process_data(input_texts, target_texts)
        sample_test(20, encoder_model, decoder_model)
        launch_cli(encoder_model, decoder_model)

    else:
        print("Too few arguments")
        return 0
        
    
if __name__ == '__main__':
    main()
