import numpy as np
from data_utils import pad_sequences, to_piece, to_bags, getDataset, minibatches, load_vocab, get_processing_word

def getdata():
    vocab_words      = load_vocab('../data/processed_data/words.txt')
    vocab_relations  = load_vocab('../data/processed_data/relation.txt')
    processing_word  = get_processing_word(vocab_words, UNK = "<UNK>")
    processing_relation = get_processing_word(vocab_relations, UNK='NA')

    train = getDataset('../data/processed_data/train.txt', processing_word, processing_relation)
    data = minibatches(train, 50)
    return data


data = getdata()
test_data = next(data)
word_ids, pos1_ids, pos2_ids, pos, relations = test_data
word_bags, pos1_bags, pos2_bags, pos_bags, y_bags, num_bags = to_bags(test_data)
