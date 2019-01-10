from bs4 import BeautifulSoup
from keras.models import Model, load_model
from keras.layers import Dense, Input, LSTM
from keras import optimizers
import numpy as np
import time, os, re

MAX_SEQ_LEN = 40

html_file_path = './roy_messages_2017-2018.html'
justin_color = '#0084ff'
roy_color = '#f1f0f0'

roy_soup = BeautifulSoup(open(html_file_path),'html.parser')

all_dialogue_list = roy_soup.find_all('div', style=re.compile("border-radius: 13px"))

roy_chunks = []
justin_chunks = []


def only_ascii(string):
    return re.sub(r'[^\x00-\x7f]',r'', string)

def shorten(string, seq_len, end=True):
    if len(string) > seq_len:
        if end:
            return string[len(string)-seq_len:]
        else:
            return string[:seq_len]
    return string

for i in range(len(all_dialogue_list) - 1):
    curr_line = all_dialogue_list[i]
    next_line = all_dialogue_list[i+1]
    is_justin_curr = re.search(justin_color, curr_line['style'])
    is_roy_next = re.search(roy_color, next_line['style'])

    if is_justin_curr and is_roy_next:
        justin_chunks.append(shorten(curr_line.text, MAX_SEQ_LEN, True).lower().replace('\n',''))
        roy_chunks.append(shorten(next_line.text, MAX_SEQ_LEN, False).lower().replace('\n',''))


#justin_chunks = justin_chunks[:-1]  #this is because Justin has 1 more message than Roy


    

with open('data2.txt', 'w') as file:
    for i in range(len(roy_chunks)):
        file.write(justin_chunks[i] + '\t' + roy_chunks[i] + '\n')
        
        
        
