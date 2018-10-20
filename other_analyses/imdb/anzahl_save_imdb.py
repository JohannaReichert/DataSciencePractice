import pandas as pd
import json
import numpy as np
from time import sleep
from collections import Counter

with open("data/imdb/train_data.csv", "r") as file:
    train_data = [list(map(int,line.split(','))) for line in file]

with open("data/imdb/test_data.csv", "r") as file:
    test_data = [list(map(int,line.split(','))) for line in file]

train_labels = pd.read_csv("data/imdb/train_labels.csv", header=None)
test_labels = pd.read_csv("data/imdb/test_labels.csv", header=None)

with open('data/imdb/word_index.json', "r") as f_word_index:
    word_index = json.loads(f_word_index.read())

with open('data/imdb/reverse_word_index.json', "r") as f_reverse_word_index:
    reverse_word_index_alt = json.loads(f_reverse_word_index.read())
#konvertiert einfach nur den key von string zu int
reverse_word_index = {int(key):reverse_word_index_alt[key] for key in reverse_word_index_alt}
# niedrigste nr von train_data = 1. ist so eine art startindex (am anfang von jedem review)
# 2 und 3 sind wahrscheinlich ersetzungen für wörter, die nicht im wörterbuch sind)
# 4 ist dann die erste richtige zahl (entspricht 1 = 'the' im word index)
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[1]])
print(train_data)

def occurence_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        occ_dict = dict(Counter(sequence))
        results[i, list(occ_dict.keys())] = list(occ_dict.values()) # set specific indices of results[i] to 1s
    return results
test = occurence_sequences(train_data[1:10])
# Our vectorized training data
x_train = occurence_sequences(train_data)

sleep(7)
# Our vectorized test data
x_test = occurence_sequences(test_data)
print(x_test[0])
sleep(7)


# Our vectorized labels

#y_train = np.asarray(train_labels).astype('float32')
#y_test = np.asarray(test_labels).astype('float32')
#print(y_train[0])

#pd.DataFrame(x_train).to_csv("../data/imdb/x_train_occ.csv", index=False)
#pd.DataFrame(x_test).to_csv("../data/imdb/x_test_occ.csv", index=False)


#pd.DataFrame(y_train).to_csv("../data/imdb/y_train.csv", index=False)
#pd.DataFrame(y_test).to_csv("../data/imdb/y_test.csv", index=False)
