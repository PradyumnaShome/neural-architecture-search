import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

import logging
import constants

def get_model_word_count(arch, vocabulary):
    """
    Given a vocabulary (list of various neural net layers), return a vector of counts in a neural architecture description.
    """
    vectorizer = CountVectorizer(vocabulary=vocabulary)

    arch = arch.replace('leaky_relu', 'leaky')
    logging.debug(arch)

    # Pad vocabulary items with spaces
    for layer in vocabulary:
        arch = arch.replace(layer, f" {layer} " )

    # Convert text to list
    arch_list = [arch]

    # Counts for each word in the vocabulary 
    word_counts = vectorizer.fit_transform(arch_list).toarray()

    return word_counts

def get_all_arch(train_data):
    corpus_word_counts = []
    global_word_counts = [0 * len(constants.KNOWN_LAYERS)]
    arch_list = train_data["arch_and_hp"]

    for arch in arch_list:
        word_counts = get_model_word_count(arch=arch, vocabulary=constants.KNOWN_LAYERS)

        global_word_counts.append(word_counts)
        logging.debug(f"Word Counts: {word_counts}")

        corpus_word_counts = [x + y for x, y in zip(word_counts, corpus_word_counts)]

    logging.info(f"Vocabulary: {constants.KNOWN_LAYERS}")
    logging.info(f"Corpus Word Counts: {corpus_word_counts}")
    logging.info(f"Global Word Counts: {global_word_counts}")
    return corpus_word_counts, global_word_counts


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(relativeCreated)1d-%(threadName)s-%(message)s')

    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    arch = get_all_arch(train_data)
    pass


if __name__ == "__main__":
    main()
