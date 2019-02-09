

import nltk
import re
import word_category_counter
import data_helper
import os, sys
from nltk.util import ngrams

DATA_DIR = "data"
LIWC_DIR = "liwc"

word_category_counter.load_dictionary(LIWC_DIR)


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
        stopwords = nltk.corpus.stopwords.words('english')
        if token not in stopwords and re.search(r'\w',token):
            normalized_token = token.lower()
        else:
            normalized_token = None

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
    sentences = nltk.sent_tokenize(text)
    ###     YOUR CODE GOES HERE

    for sent in sentences:
        sent_words = nltk.word_tokenize(sent)
        word_pos_tuples = nltk.pos_tag(sent_words)
        for word,tag in word_pos_tuples:        
            norm = normalize(word, should_normalize)
            if norm is not None:
                words.append(norm)
                tags.append(tag)
            else:
                continue 
    return words, tags


def get_ngram_features(tokens):
    """
    This function creates the unigram and bigram features as described in
    the assignment3 handout.

    :param tokens:
    :return: feature_vectors: a dictionary values for each ngram feature
    """
    feature_vectors = {}

    ###     YOUR CODE GOES HERE
    
    #unigrams
    uni_dist = nltk.FreqDist(tokens)
    feats = {}
    for word, freq in uni_dist.items():
        bin_val = bin(freq)
        feats["UNI_{}".format(word)] = bin_val
    feature_vectors.update(feats)
    
    #bigrams
    bigrams = nltk.bigrams(tokens)
    bigram_tuple = []
    for i in bigrams:
        bigram_tuple.append(i)

    bi_dist = nltk.FreqDist(bigram_tuple)
    for word, freq in bi_dist.items():
        bin_val = bin(freq)
        feats["BIGRAM_{}_{}".format(word[0], word[1])] = bin_val
    feature_vectors.update(feats)

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
   
    feats = {}
    uni_dist = nltk.FreqDist(tags)
    for word, freq in uni_dist.items():
        bin_val = bin(freq)
        feats["UNI_{}".format(word)] = bin_val
    feature_vectors.update(feats)

    bigrams = nltk.bigrams(tags)
    bigram_tuple = []
    for i in bigrams:
        bigram_tuple.append(i)

    bi_dist = nltk.FreqDist(bigram_tuple)
    for word, freq in bi_dist.items():
        bin_val = bin(freq)
        feats["BIGRAM_{}_{}".format(word[0], word[1])] = bin_val
    feature_vectors.update(feats)

    return feature_vectors


def bin(count):
    """
    Results in bins of  0, 1, 2, 3 >=
    :param count: [int] the bin label
    :return:
    """
    the_bin = None
    ###     YOUR CODE GOES HERE
    if count < 2:
        the_bin = count
    else:
        the_bin = 3
    
    return the_bin


def get_liwc_features(words):
    """
    Adds a simple LIWC derived feature

    :param words:
    :return:
    """

    # TODO: binning
    
    feature_vectors = {}
    text = " ".join(words)
    liwc_scores = word_category_counter.score_text(text)
    
    feats = {}
    for word, freq in liwc_scores.items():
        bin_val = bin(freq)
        feats["LIWC_{}".format(word)] = bin_val
    feature_vectors.update(feats)
    
    # All possible keys to the scores start on line 269
    # of the word_category_counter.py script
    negative_score = liwc_scores["Negative Emotion"]
    positive_score = liwc_scores["Positive Emotion"]
    feature_vectors["Negative Emotion"] = negative_score
    feature_vectors["Positive Emotion"] = positive_score

    if positive_score > negative_score:
        feature_vectors["liwc:positive"] = 1
    else:
        feature_vectors["liwc:negative"] = 1

    return feature_vectors


FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features"}

def get_features_category_tuples(category_text_dict, feature_set):
    """

    You will might want to update the code here for the competition part.

    :param category_text_dict:
    :param feature_set:
    :return:
    """
    features_category_tuples = []
    all_texts = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)

    for category in category_text_dict:
        for text in category_text_dict[category]:
            words, tags = get_words_tags(text)
            feature_vectors = {}
            ###     YOUR CODE GOES HERE
            if feature_set is 'word_features':
                feats = get_ngram_features(words)                                                  
                feature_vectors.update(feats)
            
            elif feature_set is 'word_pos_features':
                feats = get_ngram_features(words)
                feature_vectors.update(feats)
                
                tagfeats = get_pos_features(tags)
                feature_vectors.update(tagfeats)            
            elif feature_set is 'word_pos_liwc_features':
                feats = get_ngram_features(words)
                feature_vectors.update(feats)
    
                tagfeats = get_pos_features(tags)
                feature_vectors.update(tagfeats)
                
                liwcfeat = get_liwc_features(words)
                feature_vectors.update(liwcfeat)

            else:
                raise Exception("Invalid feature set {}".format(feature_set))

            features_category_tuples.append((feature_vectors, category))
            all_texts.append(text)

    return features_category_tuples, all_texts


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


if __name__ == "__main__":



    pass



