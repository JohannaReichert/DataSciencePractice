import pandas as pd
import json
import numpy as np
from time import sleep

with open("../data/imdb/train_data.csv", "r") as file:
    train_data = [list(map(int,line.split(','))) for line in file]

with open("../data/imdb/test_data.csv", "r") as file:
    test_data = [list(map(int,line.split(','))) for line in file]

train_labels = pd.read_csv("../data/imdb/train_labels.csv", header=None)
test_labels = pd.read_csv("../data/imdb/test_labels.csv", header=None)

with open('../data/imdb/word_index.json', "r") as f_word_index:
    word_index = json.loads(f_word_index.read())

with open('../data/imdb/reverse_word_index.json', "r") as f_reverse_word_index:
    reverse_word_index = json.loads(f_reverse_word_index.read())
#reverse_word_index fängt bei 1 an, mit 'the'
reverse_word_index = {int(key):reverse_word_index[key] for key in reverse_word_index}

# train_data müsste um 3 verschoben werden, um mit word_index übereinzustimmen??
# also train_data -3. d.h. die Werte 1,2,3 können wir gar nicht "entschlüsseln"
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[1]])
print(train_data)

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
print(x_train[1])
sleep(7)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
print(x_test[1])
sleep(7)
# Our vectorized labels
sleep(7)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(y_train[0])

#pd.DataFrame(x_train).to_csv("../data/imdb/x_train_vect.csv", index=False)
#pd.DataFrame(x_test).to_csv("../data/imdb/x_test_vect.csv", index=False)
#pd.DataFrame(y_train).to_csv("../data/imdb/y_train.csv", index=False)
#pd.DataFrame(y_test).to_csv("../data/imdb/y_test.csv", index=False)