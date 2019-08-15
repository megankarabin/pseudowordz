import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
import flask
from jinja2 import Template, Environment, FileSystemLoader
from io import open
import glob
import os
import string
import torch
import torch.nn as nn
import unicodedata

#===============================================================================
                ### Functions for reading data into the model ###

def findFiles(path): return glob.glob(path)

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split(' ')
    return [unicodeToAscii(line) for line in lines]


category_lines = {}
all_categories = []
for filename in findFiles('data/words/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor
#===============================================================================
                    ### Defining the model class ###

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

#===============================================================================
                        ### Loading the model ###

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = 'model/model.pth'
rnn = load_checkpoint(model)

#===============================================================================
                ### Functions for getting random letters ###

letters = string.ascii_letters[:26]
letters = [i for i in letters if i != 'x' and i != 'x']

def random_letter():
    random_letter = np.random.choice(letters)
    return random_letter

def random_letters(number=3):
    random_letters = np.random.choice(letters, number)
    return random_letters

#===============================================================================
                        ### Sampling Functions ###

max_length = 20

def sample(category, start_letter):
    with torch.no_grad():
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


def multiword_generation(category='English', number=3):
    words = []
    for i in range(0, 100):
        while len(words) < number:
            word = sample(category, random_letter())
            if word not in words:
                words.append(word)
            else:
                break
    return words


def get_random_word():
    info = []

    random_words_url = 'https://randomword.com/'
    response = requests.get(random_words_url)

    soup = BeautifulSoup(response.content, 'lxml')
    random_word = soup.find('div', {'id': 'random_word'})
    definition = soup.find('div', {'id': 'random_word_definition'})

    real = random_word.text
    definition = definition.text.strip()

    info.append(real)
    info.append(definition)

    return info

#===============================================================================
    ### Functions for randomly selecting real/fake words for the game ###

def select_real_or_fake():
    flip = np.random.choice([0,1])
    if flip == 0:
        word = get_random_word()
        return word, 'real'
    else:
        word = sample('English', random_letter())
        return [word], 'fake'

# keep status_dict global so that it can store all words heretofore displayed
status_dict = {}
status_dict


def is_fake(word):
    if word[1] == 'fake':
        status_dict[word[0][0]] = ['fake']
    elif word[1] == 'real':
        status_dict[word[0][0]] = ['real', word[0][1]]
    return status_dict


def game_output(word_info):
    word = word_info[0][0]
    if status_dict[word][0] == 'real':
        definition = word_info[0][1]
        return word, definition.capitalize()
    else:
        return word


def message_output(response, word):
    if response == status_dict[word][0]:
        message = 'Correct!'
    else:
        message = 'Incorrect.'
    return message


def get_score(counter):
    score = correct_count/counter
    if score <= 0.5:
        end_msg = 'You lose!'
    if score > 0.5:
        end_msg = 'You win!'
    return score, end_msg
#===============================================================================
                                ### App ###

counter = 0
correct_count = 0

app = flask.Flask(__name__)


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/getwords')
def getwords():
    return render_template('getwords.html')


@app.route('/result', methods=['POST','GET'])
def result():
    if flask.request.method == 'POST':
        result = flask.request.form

        data_dict = {
            'language': result['language'],
            'num_words': int(result['num_words'])
        }

        pseudowords = multiword_generation(data_dict['language'], data_dict['num_words'])

    return render_template('result.html', data=data_dict, words=pseudowords)


@app.route('/play', methods=['POST', 'GET'])
def play():

    if flask.request.method == 'GET':

        global counter
        counter = 0
        counter += 1

        global correct_count
        correct_count = 0

        word = select_real_or_fake() # first, get the word
        is_fake(word) # next, save it to the status dict with its status (and def if applicable)

        return render_template('play.html', word=word, counter=counter, correct_count=correct_count)

    if flask.request.method == 'POST':

        input = flask.request.form
        counter += 1
        word = select_real_or_fake()
        is_fake(word)

        return render_template('play.html', word=word, counter=counter, correct_count=correct_count)


@app.route('/response', methods=['POST', 'GET'])
def response():

    if flask.request.method == 'POST':

        result = flask.request.form

        data_dict = {
            'counter': int(result['counter']),
            'correct_count': int(result['correct_count']),
            'user_response': result['input'],
            'word': result['word']
        }

        response = data_dict['user_response']
        my_word = data_dict['word']
        message = message_output(response, my_word)

        if message == 'Correct!':
            global correct_count
            correct_count += 1

        if data_dict['counter'] == 10:
            score, end_msg = get_score(data_dict['counter'])
            data_dict['score'] = score
            data_dict['end_msg'] = end_msg

        return render_template('response.html', data=data_dict, message=message, word=my_word, status_dict=status_dict, correct_count=correct_count)


if __name__ == '__main__':

    HOST = '127.0.0.1'

    app.run(debug=True, port=6001)
