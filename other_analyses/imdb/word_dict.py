import json

""" für die Auswahl (10000 häufigste) wurden die mit value <=10000 gewählt"""
with open('../data/imdb/word_index.json', "r") as f_word_index:
    word_index = json.loads(f_word_index.read())
# getting only the 10000 most common words (which are included in our train data)
word_index = {key:value for key,value in word_index.items() if int(value)<= 10000}
"""with open('../data/imdb/reverse_word_index.json', "r") as f_reverse_word_index:
    reverse_word_index = json.loads(f_reverse_word_index.read())"""

# removing the short (probably unimportant) words
short_words = {key:value for key,value in word_index.items() if len(key)<4}

new_word_index = word_index.copy()
for k in short_words:
    new_word_index.pop(k,None)

print(len(new_word_index)) # 9380

