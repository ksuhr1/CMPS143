import re, nltk, argparse

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


def process_reviews(file_name):
    positive_texts, negative_texts, first_sent = read_reviews(file_name)

    # There are 150 positive reviews and 150 negative reviews. Vocabularies 3035 vs 2520.
    # print(len(positive_texts))
    # print(len(negative_texts))
    # Your code goes here
    # Procedure 3.2.1
    analyze_reviews("positive", positive_texts, first_sent)
    analyze_reviews("negative", negative_texts, first_sent)

def analyze_reviews(category, reviews, first_sent):
    # Procedure 3.2.1
    normalized_reviews = normalize_data(reviews)

    # Procedure 3.2.2
    uni_freq = nltk.FreqDist(normalized_reviews)
    unigrams = uni_freq.most_common()
    write_unigram_freq(category, unigrams)
    uni_fd = uni_freq

    # Procedure 3.2.3
    bi_fd, bi_cfd = write_bigram_freq(category, normalized_reviews)

    # Question 3.3
    answers_to_questions(category, uni_fd, bi_fd, bi_cfd, first_sent, normalized_reviews)

# Procedure 3.2.1
def normalize_data(texts):
    stopwords = nltk.corpus.stopwords.words('english')
    data = [w.lower() for w in nltk.word_tokenize(u" ".join(texts))
               if w.lower() not in stopwords         # Remove stopwords
               and re.search(r'\w', w)]    # Remove words that do not have a word character.
    #print(len(data))
    # 6554 in positive, 9381 in negative
    return data

# Procedure 3.2.2
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


# Procedure 3.2.3
def write_bigram_freq(category, reviews):
    bi_file = open("{0}-bigram-freq-n.txt".format(category), 'w', encoding="utf-8")
    review_bigrams = nltk.bigrams(reviews)
    bi_fd = nltk.FreqDist(u"{} {}".format(b1, b2) for b1, b2 in review_bigrams)
    for bi, count in bi_fd.most_common():
        b1 = bi.split()[0]
        b2 = bi.split()[1]
        bi_file.write("{0:<20s}{1:<20s}{2:<d}\n".format(b1, b2, count))
    bi_file.close()
    bi_cfd = nltk.ConditionalFreqDist(nltk.bigrams(reviews))
    return bi_fd, bi_cfd

def answers_to_questions(category, uni_freq, bi_fd, bi_cfd, first_sent, reviews):
    print("Question 3.3.1")
    print("The 15 most probable words for category '{0}' are:".format(category))
    for word, count in uni_freq.most_common(15):
        print(u"{0:<20s} {1:>10}/{2} = {3:.4f}".format(word, count, uni_freq.N(), uni_freq.freq(word)))
    print()
    print("The 10 most probable bigrams for category '{0}' are:".format(category))
    for bi, count in bi_fd.most_common(10):
        print(u"{0:<20s} {1:>10}/{2} = {3:.4f}".format(bi, count, bi_fd.N(), bi_fd.freq(bi)))
    print()

    print("Question 3.3.2")
    print("Collocations of category:", category)
    nltk.Text(reviews).collocations()
    print()

    print("Question 3.3.3")
    print("The first sentence is:", first_sent)
    bigrams = [word for bi in nltk.bigrams(normalize_data(nltk.word_tokenize(first_sent))) for word in bi]
    print("The bigrams after normalization:", bigrams)
    P1 = uni_freq.freq(bigrams[0])
    print("P({0}) = {1}/{2} = {3:.4f}".format(bigrams[0], uni_freq[bigrams[0]], uni_freq.N(), P1))
    for i in range(1, len(bigrams)):
        P2 = bi_cfd[bigrams[i-1]].freq(bigrams[i])  # Note that the first argument is condition.
        print("P({0}|{1}) = {2:.4f}".format(bigrams[i], bigrams[i-1], P2))
        P1 *= P2
    print("By multiplying the above probabilities, P(first sentence) = {0:.4f} in the '{1}' domain.".format(P1,category))
    if category == "negative":
        print("Don't surprise if you get 0 probability in the negative domain, because the P(restaurant|excellent) is 0 or very close to 0. ")
    print()

    print("Question 3.3.4")
    print("P(an AND excellent AND restaurant) = P(an) * P(excellent | an) * P(restaurant | an AND excellent)")
    print("This model makes a Markov assumption of order 2. A 4-gram model would make a Markov assumption of order 3")
    print()

    if category == "positive":
        print("Question 3.3.5")
        #print("P(mashed OR potatoes) = P(mashed) + P(potatoes) - P(mashed AND potatoes)")
        P_m = uni_freq.freq('mashed')
        P_p = uni_freq.freq('potatoes')
        
        print(("P(mashed OR potatoes) = P(mashed) + P(potatoes) = {0:.4f}".format(P_m + P_p)))
        	
        print(("P(mashed) = {0}/{1} = {2:.4f}".format(uni_freq['mashed'], uni_freq.N(), P_m)))
        print(("P(potatoes) = {0}/{1} = {2:.4f}".format(uni_freq['potatoes'], uni_freq.N(), P_p)))
        """
        print("As for P(mashed AND potatoes), there are two ways to get this result:")
        print("\t Way (1) Gain the P(mashed AND potatoes) from the bigrams frequency:")
        print(("\t\t P(mashed AND potatoes) = {0}/{1} = {2:.4f}".format(bi_fd["mashed potatoes"], bi_fd.N(), bi_fd.freq("mashed potatoes"))))

        print("\t Way (2) Gain the P(mashed AND potatoes) from the bigrams conditional probability:")
        # One way to calculate P(potatoes | mashed) if we don't use the cfd function.
        #P2 = bi_fd.freq("mashed potatoes")/ uni_freq.freq('mashed')
        #print(("\t\t P(potatoes | mashed) = P(mashed AND potatoes) / P(mashed) = {0:.4f}/{1:.4f} = {2:.4f}".format(bi_fd.freq("mashed potatoes"), uni_freq.freq('mashed'), P2)))

        # Another way to get the P(potatoes | mashed) by nltk.ConditionalFreqDist, should be the same as above.
        P2 = bi_cfd['mashed'].freq('potatoes')  # Note that the first element is condition.
        print(("\t\t P(potatoes | mashed) = {0:.4f}".format(P2)))
        P3 = P_m * P2
        # You may check this P(mashed AND potatoes) with the previous one to see if it is correct.
        print(("\t\t P(mashed AND potatoes) = P(mashed)P(potatoes | mashed) = {0:.4f} * {1:.4f} = {2:.4f}".format(P_m, P2, P3)))
        print("And thus,")
        print(("P(mashed OR potatoes) = P(mashed) + P(potatoes) - P(mashed AND potatoes) = {0:.4f}".format(P_m + P_p - P3)))
        """        
        print()

    print("Question 3.3.6")
    print("You will get a 0 probability of an unseen sentence, if you encounter a word that is not in your frequency tables")
    print()

    print("Question 3.3.7")
    print("A higher-order n-gram model is better than bi-gram or tri-gram because it captures more context. However, it also often leads to data sparsity.")
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('-f', dest="fname", default="restaurant-training.data",  help='File name.')
    args = parser.parse_args()
    fname = args.fname

    process_reviews(fname)
