import re, nltk, argparse, sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

def get_score(review):
    return int(re.search(r'Overall = ([1-5])', review).group(1))

def get_text(review):
    return re.search(r'Text = "(.*)"', review).group(1)

def read_reviews(file_name):
    """
    Dont change this function.

    :param file_name:
    :return:
    """
    file = open(file_name, "rb")
    raw_data = file.read().decode("latin1")
    file.close()

    positive_texts = []
    negative_texts = []
    first_sent = None
    for review in re.split(r'\.\n', raw_data):
        overall_score = get_score(review)
        review_text = get_text(review)
        if overall_score > 3:
            positive_texts.append(review_text)
        elif overall_score < 3:
            negative_texts.append(review_text)
        if first_sent == None:
            sent = nltk.sent_tokenize(review_text)
            if (len(sent) > 0):
                first_sent = sent[0]
    return positive_texts, negative_texts, first_sent


########################################################################
## Dont change the code above here
######################################################################

def normalize_data(value_texts):
    text = [word.lower() for word in value_texts]
    word_tokens = []
    for review in text:
        sentences = nltk.sent_tokenize(review)
        for sent in sentences:
            sent_words = nltk.word_tokenize(sent)
            for word in sent_words:
                word_tokens.append(word)

    stop_words = set(stopwords.words('english'))
    clean_words = [word for word in word_tokens if not word in stop_words]
    regex = re.compile(r"[\w]+")
    filtered = [i for i in clean_words if regex.search(i)]
    return filtered

def bigram_data(category, word_tokens):
    file_name = category+"-bigram-freq.txt"
    
    bigrams = ngrams(word_tokens, 2)
    bigram_list = []

    #add bigrams to bigram_list
    for i in bigrams:
        bigram_list.append(i)
    
    freqlist = []
    cflist = nltk.ConditionalFreqDist(bigram_list)
    with open(file_name,"a") as outfile:
        #for condition in cflist:
        for condition in cflist:
            mostfreq = cflist[condition].most_common()
            for word,count in mostfreq:
                phrase = condition+" "+word+" "+str(count)
                freqlist.append(phrase)
        
        freqlist = sorted(freqlist,key=lambda x: int(re.search(r"\d+$",x).group()))
        for seq in reversed(freqlist):
            outfile.write(seq)
            outfile.write("\n")

def unigram_data(category, word_tokens):
    file_name = category+"-unigram-freq.txt"

    word_fdist = nltk.FreqDist(word_tokens)
    freq = word_fdist.most_common()
    with open(file_name,"a") as outfile:
        for term,count in freq:
            outfile.write(term+" "+str(count))
            outfile.write("\n")

def process_reviews(file_name):
    positive_texts, negative_texts, first_sent = read_reviews(file_name)
    pos = 'positive'
    neg = 'negative'
    # There are 150 positive reviews and 150 negative reviews.
    #print(len(positive_texts))
    #print(len(negative_texts))

    # Your code goes here
    #normalize the data
    posdata = normalize_data(positive_texts)
    negdata = normalize_data(negative_texts)
    
    #unigrams
    unigram_data(pos, posdata)
    unigram_data(neg, negdata)
    
    #bigrams
    bigram_data(pos, posdata)
    bigram_data(neg, negdata)

    #text collocations
    
    postext = nltk.Text(posdata)
    negtext = nltk.Text(negdata)
    print("----------Positive reviews collocations----------")
    postext.collocations()
    print("\n")
    print("----------Negative reviews collocations----------")
    negtext.collocations()



# Write to File, this function is just for reference, because the encoding matters.
def write_file(file_name, data):
    file = open(file_name, 'w', encoding="utf-8")    # or you can say encoding="latin1"
    file.write(data)
    file.close()


def write_unigram_freq(category, unigrams):
    """
    A function to write the unigrams and their frequencies to file.

    :param category: [string]
    :param unigrams: list of (word, frequency) tuples
    :return:
    """
    uni_file = open("{0}-unigram-freq-n.txt".format(category), 'w', encoding="utf-8")
    for word, count in unigrams:
        uni_file.write("{0:<20s}{1:<d}\n".format(word, count))
    uni_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('-f', dest="fname", default="restaurant-training.data",  help='File name.')
    args = parser.parse_args()
    fname = args.fname

    process_reviews(fname)
