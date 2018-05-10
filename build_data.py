from model.config import Config
from model.data_utils import process_wordvectors, process_relation2id, check_entity_in_sentence

def main():
    """Procedure to build data

    You MUST RUN this procedure to preprocess datasets. 


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config
    config = Config(load=False)

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
