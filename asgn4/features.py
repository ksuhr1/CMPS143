
import nltk
import re
import word_category_counter
import data_helper
import os, sys
from word2vec_extractor import Word2vecExtractor

DATA_DIR = "data"
LIWC_DIR = "liwc"

word_category_counter.load_dictionary(LIWC_DIR)
STOPWORDS = nltk.corpus.stopwords.words('english')

w2vecmodel = "data/glove-w2v.txt"
w2v = None


def normalize(token, should_normalize=True):
    """
    This function performs text normalization.

    If should_normalize is False then we return the original token unchanged.
    Otherwise, we return a normalized version of the token, or None.

    For some tokens (like stopwords) we might not want to keep the token. In
    this case we return None.

    :param token: str: the word to normalize
    :param should_normalize: bool
    :return: None or str
    """
    if not should_normalize:
        normalized_token = token

    else:

        ###     YOUR CODE GOES HERE
        normalized_token = None
        token = token.lower()
        if token not in STOPWORDS and re.search(r'\w', token):
            normalized_token = token

    return normalized_token



def get_words_tags(text, should_normalize=True):
    """
    This function performs part of speech tagging and extracts the words
    from the review text.

    You need to :
        - tokenize the text into sentences
        - word tokenize each sentence
        - part of speech tag the words of each sentence

    Return a list containing all the words of the review and another list
    containing all the part-of-speech tags for those words.

    :param text:
    :param should_normalize:
    :return:
    """
    words = []
    tags = []

    # tokenization for each sentence

    ###     YOUR CODE GOES HERE
    for sent in nltk.sent_tokenize(text):
        for word, pos in nltk.pos_tag(nltk.word_tokenize(sent)):
            checked_word = normalize(word, should_normalize)
            if checked_word is None:
                continue
            words.append(checked_word)
            tags.append(pos)

    return words, tags


def bin(count):
    """
    Results in bins of  0, 1, 2, 3 >=
    :param count: [int] the bin label
    :return:
    """
    # Just a wild guess on the cutoff
    return count if count < 2 else 3


def get_ngram_features(tokens):
    """
    This function creates the unigram and bigram features as described in
    the assignment3 handout.

    :param tokens:
    :return: feature_vectors: a dictionary values for each ngram feature
    """
    feature_vectors = {}

    ###     YOUR CODE GOES HERE

    uni_fdist = nltk.FreqDist(tokens)
    for token, freq in uni_fdist.items():
        #feature_vectors["UNI_{0}".format(token)] = bin(freq)
        feature_vectors["UNI_{0}".format(token)] = freq
    bi_fdist = nltk.FreqDist(nltk.bigrams(tokens))
    for (b1, b2), freq in bi_fdist.items():
        feature_vectors["BIGRAM_{0}_{1}".format(b1, b2)] = freq
        #feature_vectors["BIGRAM_{0}_{1}".format(b1, b2)] = bin(freq)

    return feature_vectors


def get_pos_features(tags):
    """
    This function creates the unigram and bigram part-of-speech features
    as described in the assignment3 handout.

    :param tags: list of POS tags
    :return: feature_vectors: a dictionary values for each ngram-pos feature
    """
    feature_vectors = {}

    ###     YOUR CODE GOES HERE

    uni_fdist = nltk.FreqDist(tags)
    for tag, freq in uni_fdist.items():
        #val = bin(freq)
        val = freq
        feature_vectors["UNI_{0}".format(tag)] = val

    bi_fdist = nltk.FreqDist(nltk.bigrams(tags))
    for (b1, b2), freq in bi_fdist.items():
        feature_vectors["BIGPOS_{0}_{1}".format(b1, b2)] = freq
        #feature_vectors["BIGPOS_{0}_{1}".format(b1, b2)] = bin(freq)

    return feature_vectors

def get_word_embedding_features(text):
    global w2v
    if w2v is None:
        print("loading word vectors ...", w2vecmodel)
        w2v = Word2vecExtractor(w2vecmodel)
    feature_dict = w2v.get_doc2vec_feature_dict(text)
    return feature_dict



def bin_liwc(count):
    if count < 3:
        bin = 1
    elif count < 10:
        bin = 2
    else:
        bin = 3
    return bin


liwc_vals = []
def get_liwc_features(words):
    """
    Adds a simple LIWC derived feature

    :param words:
    :return:
    """
    feature_vectors = {}
    text = " ".join(words)
    liwc_scores = word_category_counter.score_text(text, raw_counts=True)

    # All possible keys to the scores start on line 269
    # of the word_category_counter.py script
    liwc_categories = word_category_counter.Dictionary._liwc_categories
    for long_name, _, _, _, _ in liwc_categories:
        val = int(liwc_scores[long_name])
       # feature_vectors["LIWC:{}".format(long_name.replace(" ", "-"))] = bin_liwc(val)
        feature_vectors["LIWC:{}".format(long_name.replace(" ", "-"))] = val
    return feature_vectors


FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "only_liwc"}

#FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "only_liwc",
#                "word_embedding"}


def get_features_category_tuples(category_text_dict, feature_set):
    """

    You will might want to update the code here for the competition part.

    :param category_text_dict:
    :param feature_set:
    :return:
    """
    features_category_tuples = []
    texts = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)

    for category in category_text_dict:
        for text in category_text_dict[category]:

            words, tags = get_words_tags(text)
            feature_vectors = {}

            if feature_set == "word_pos_features":
                feature_vectors.update(get_ngram_features(words))
                feature_vectors.update(get_pos_features(tags))

            elif feature_set == "word_features":
                feature_vectors.update(get_ngram_features(words))

            elif feature_set == "word_pos_liwc_features":
                feature_vectors.update(get_ngram_features(words))
                feature_vectors.update(get_pos_features(tags))
                feature_vectors.update(get_liwc_features(words))
            elif feature_set == "only_liwc":
                feature_vectors.update(get_liwc_features(words))


            features_category_tuples.append((feature_vectors, category))
            texts.append(text)

    return features_category_tuples, texts



def get_features_from_texts(all_texts, feature_set):
    features_category_tuples = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set,
                                                                                                 FEATURE_SETS)
    for text in all_texts:

        words, tags = get_words_tags(text)
        feature_vectors = {}

        if feature_set == "word_pos_features":
            feature_vectors.update(get_ngram_features(words))
            feature_vectors.update(get_pos_features(tags))

        elif feature_set == "word_features":
            feature_vectors.update(get_ngram_features(words))

        elif feature_set == "word_pos_liwc_features":
            feature_vectors.update(get_ngram_features(words))
            feature_vectors.update(get_pos_features(tags))
            feature_vectors.update(get_liwc_features(words))
        elif feature_set == "only_liwc":
            feature_vectors.update(get_liwc_features(words))

        features_category_tuples.append(feature_vectors)

    return features_category_tuples





def write_features_category(features_category_tuples, outfile_name):
    """
    Save the feature values to file.

    :param features_category_tuples:
    :param outfile_name:
    :return:
    """
    with open(outfile_name, "w", encoding="utf-8") as fout:
        for (features, category) in features_category_tuples:
            fout.write("{0:<10s}\t{1}\n".format(category, features))


def features_stub():
    datafile = "restaurant-training.data"

    raw_data = data_helper.read_file(os.path.join(DATA_DIR, datafile))
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}
    feature_set = "word_features"

    features_category_tuples, texts = get_features_category_tuples(category_texts, feature_set)

    #raise NotImplemented
    #filename = "???"
    filename = outfile_name
    write_features_category(features_category_tuples, filename)



if __name__ == "__main__":
    print("hello world!")
    features_stub()

