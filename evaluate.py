from model.data_utils import getDataset, pos_constrain
from model.pcnn_model import PCNNModel
from model.config import Config


def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of PCNNModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input sentence> Steve_Jobs is one co-founder of Apple_Inc.""")

    while True:
        sentence = input("input sentence> ")
        entity1  = input("input entity1> ")
        entity2  = input("input entity2> ")

        sequence = sentence.split()

        if words_raw == ["exit"]:
            break

        if entity1 not in sentence or entity2 not in sentence:
            print("entity not found in sentence.")
            break

        ent1     = sentence.index(entity1)
        ent2     = sentence.index(entity2)
        words, pos1_ids, pos2_ids = [], [], []
        for idx, word in enumerate(sequence):
            words.append(word)
            pos1 = pos_constrain(idx - ent1)
            pos1_ids.append(pos1)
            pos2 = pos_constrain(idx - ent2)
            pos2_ids.append(pos2)

        pos = [ent1, ent2, len(sequence)-1]
        pos.sort()

        preds = model.predict(words, pos1_ids, pos2_ids, pos)
        print("{}, {}, {}".format(entity1, preds, entity2))


def main():
    # create instance of config
    config = Config()

    # build model
    model = PCNNModel(config)
    model.build()
    model.restore_session(config.restore_model)

    # create dataset
    test  = getDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    interactive_shell(model)


if __name__ == "__main__":
    main()
