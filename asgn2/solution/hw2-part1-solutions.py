import argparse, re, nltk

# https://docs.python.org/3/howto/regex.html
# https://docs.python.org/3/library/re.html
# https://www.debuggex.com/


def get_words(pos_sent):
    """
    Given a part-of-speech tagged sentence, return a sentence
    including each token separated by whitespace.

    As an interim step you need to fill word_list with the
    words of the sentence.

    :param pos_sent: [string] The POS tagged stentence
    :return:
    """

    # add the words of the sentence to this list in sequential order.
    word_list = []

    # Your code goes here

    # Write a regular expression that matches only the
    # words of each word/pos-tag in the sentence.
    pat = r'(\S+)/\S+'
    word_list = re.findall(pat, pos_sent)

    # END OF YOUR CODE
    retval = " ".join(word_list) if len(word_list) > 0 else None
    return retval


def get_pos_tags(pos_sent):
    return set(re.findall(r'\S+/(\S+)', pos_sent))


def get_noun_phrase(pos_sent):
    """
    Find all simple noun phrases in pos_sent.

    A simple noun phrase is a single optional determiner followed by zero
    or more adjectives ending in one or more nouns.

    This function should return a list of noun phrases without tags.

    :param pos_sent: [string]
    :return: noun_phrases: [list]
    """
    noun_phrases = []

    # Your code goes here

    # Penn Tagset
    # Optional DT: (\S+/DT\s*)?
    # Adjetive can be JJ,JJR,JJS: (\S+/JJ\w?\s*)*
    # Noun can be NN,NNS,NNP,NNPS: (\S+/NN\w*\s*)+
    # ?P<> named group
    # ?: non-capturing group
    # See https://docs.python.org/3/howto/regex.html#non-capturing-and-named-groups
    pattern = re.compile(r'(?P<NP>(?:\S+/DT\s*)?(?:\S+/JJ\w?\s*)*(?:\S+/NN\w*\s*)+)')
    for m in pattern.finditer(pos_sent):
        noun_phrases.append(get_words(m.group('NP')))


    # END OF YOUR CODE

    return noun_phrases


def read_stories(fname):
    stories = []
    with open(fname, 'r') as pos_file:
        story = []
        for line in pos_file:
            if line.strip():
                story.append(line)
            else:
                stories.append("".join(story))
                story = []
    return stories



def most_freq_noun_phrase(pos_sent_fname, verbose=True):
    """

    :param pos_sent_fname:
    :return:
    """
    story_phrases = {}
    story_id = 1
    for story in read_stories(pos_sent_fname):
        most_common = []
        # your code starts here

        noun_phrases = []
        for line in story.split("\n"):
            # If line is not empty.
            np = [n.lower() for n in get_noun_phrase(line)]
            # print(str(np) + "\n")
            noun_phrases += np


        np_freq = nltk.FreqDist(noun_phrases)
        most_common = np_freq.most_common(3)

        # end your code
        if verbose:
            print("The most freq NP in document[" + str(story_id) + "]: " + str(most_common))
        story_phrases[story_id] = most_common
        story_id += 1

    return story_phrases


def test_get_words():
    """
    Tests get_words().
    Do not modify this function.
    :return:
    """
    print("\nTesting get_words() ...")
    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT ' \
               'animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print(pos_sent)
    retval = str(get_words(pos_sent))
    print("retval:", retval)

    gold = "All animals are equal , but some animals are more equal than others ."
    assert retval == gold, "test Fail:\n {} != {}".format(retval, gold)

    print("Pass")


def test_get_noun_phrases():
    """
    Tests get_noun_phrases().
    Do not modify this function.
    :return:
    """
    print("\nTesting get_noun_phrases() ...")

    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT ' \
               'animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print("input:", pos_sent)
    retval = str(get_noun_phrase(pos_sent))
    print("retval:", retval)

    gold = "['All animals', 'some animals', 'others']"
    assert retval == gold, "test Fail:\n {} != {}".format(retval, gold)

    print("Pass")


def test_most_freq_noun_phrase(infile="fables-pos.txt"):
    """
    Tests get_noun_phrases().
    Do not modify this function.
    :return:
    """
    print("\nTesting most_freq_noun_phrase() ...")

    import os
    if os.path.exists(infile):
        noun_phrase = most_freq_noun_phrase(infile, False)
        gold = "[('the donkey', 6), ('the mule', 3), ('load', 2)]"
        retval = str(noun_phrase[7])

        print("gold:\t", gold)
        print("retval:\t", retval)

        assert retval == gold, "test Fail:\n {} != {}".format(noun_phrase[7], gold)
        print("Pass")
    else:
        print("Test fail: path does not exist;", infile)


def run_tests():
    test_get_words()
    test_get_noun_phrases()
    test_most_freq_noun_phrase()


if __name__ == '__main__':

    # comment this out if you dont want to run the tests
    run_tests()

    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('-i', dest="pos_sent_fname", default="fables-pos.txt",  help='File name that contant the POS.')

    args = parser.parse_args()
    pos_sent_fname = args.pos_sent_fname

    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print(pos_sent)
    print(str(get_words(pos_sent)))
    print(str(get_noun_phrase(pos_sent)))

    most_freq_noun_phrase(pos_sent_fname)

