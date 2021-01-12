# read in the extracted text file      
with open('text8') as f:
    text = f.read()

# print out the first 100 characters
#print(text[:100])


import utils

# get list of words

words = utils.preprocess(text)
# It converts any punctuation into tokens, so a period is changed to <PERIOD>. In this data set, there aren't any periods, but it will help in other NLP problems.
# It removes all words that show up five or fewer times in the dataset. This will greatly reduce issues due to noise in the data and improve the quality of the vector representations.
# It returns a list of words in the text
#print(words[:30])

# print some stats about this word data
#print("Total words in text: {}".format(len(words)))
#print("Unique words: {}".format(len(set(words)))) # `set` removes any duplicate words


# Next, I'm creating two dictionaries to convert words to integers and back again (integers to words). This is again done with a function in the utils.py file. create_lookup_tables takes in a list of words in a text and returns two dictionaries.

# The integers are assigned in descending frequency order, so the most frequent word ("the") is given the integer 0 and the next most frequent is 1, and so on
vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]

#print(int_words[:30])



