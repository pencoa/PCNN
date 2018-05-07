from model.config import Config
from model.data_utils import getDataset, get_vocabs, UNK, get_polyglot_vocab, write_vocab, \
                        load_vocab, export_trimmed_polyglot_vectors, get_processing_word

from model.data_utils import process_wordvectors, process_relation2id

def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train, dev
    and test) and extract the vocabularies in terms of words, tags. Having built
    the vocabularies it writes them in a file. The writing of vocabulary in a
    file assigns an id (the line #) to each word. It then extract the relevant
    polyglot vectors and stores them in a np array such that the i-th entry
    corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word()

    # Generators
    dev   = getDataset(config.filename_dev, processing_word)
    test  = getDataset(config.filename_test, processing_word)
    train = getDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_poly = get_polyglot_vocab(config.filename_polyglot)

    # Get common vocab
    vocab = vocab_words & vocab_poly
    vocab.add(UNK)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim Polygloe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_polyglot_vectors(vocab, config.filename_polyglot, \
                            config.filename_trimmed, config.dim_word)

    # Process pre trained word vectors
    process_wordvectors(config.filename_wordvectors, config.filename_words, \
                        config.filename_embeddings)

    # Process relation2id
    process_relation2id(config.filename_relation_origin, config.filename_relation)

    # Process train and test datasets
    check_entity_in_sentence(config.filename_train_origin, config.filename_train, \
                            config.filename_train_wrong)
    check_entity_in_sentence(config.filename_test_origin, config.filename_test, \
                            config.filename_test_wrong)


if __name__ == '__main__':
    main()
