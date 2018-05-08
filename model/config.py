import os

from .general_utils import get_logger
from .data_utils import get_trimmed_polyglot_vectors, load_vocab, \
        get_processing_word

class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed
        vectors)

        """
        # 1. vocabulary
        self.vocab_words     = load_vocab(self.filename_words)
        self.vocab_relations = load_vocab(self.filename_relation)

        self.nwords     = len(self.vocab_words)
        self.nrelations = len(self.vocab_relations)
        self.nposition  = 500

        # 2. get processing functions that map str -> id
        self.processing_word     = get_processing_word(self.vocab_words, UNK = "<UNK>")
        self.processing_relation = get_processing_word(self.vocab_relations, UNK='NA')

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_polyglot_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "./results/test/"
    dir_model  = dir_output + "model.weights/" # directory to save models
    path_log   = dir_output + "log.txt"
    restore_model = "./results/test/model.weights/early_best.ckpt"

    # embeddings
    dim_word = 50
    dim_pos  = 5
    dim = dim_word + 2*dim_pos

    # convolution
    window_size  = 3
    feature_maps = 230

    filename_train_origin = "./data/origin_data/train.txt"
    filename_train = "./data/processed_data/train.txt"
    filename_train_wrong = "./data/processed_data/wrong_parse_train.txt"

    filename_test_origin = "./data/origin_data/test.txt"
    filename_test = "./data/processed_data/test.txt"
    filename_test_wrong = "./data/processed_data/wrong_parse_test.txt"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "./data/processed_data/words.txt"
    filename_embeddings = ".data/processed_data/vectors.npz"

    filename_relation_origin = "./data/origin_data/relation2id.txt"
    filename_relation = "./data/processed_data/relation.txt"

    # word vectors file
    filename_wordvectors = "./data/origin_data/vec.txt"

    use_pretrained = True

    max_iter = None # if not None, max number of examples in Dataset

    # training
    train_word_embeddings = False
    train_pos_embeddings = True
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 50
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3
    early_stop       = True
    max_train_step   = 100000
