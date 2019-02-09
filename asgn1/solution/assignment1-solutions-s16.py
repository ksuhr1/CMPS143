#!/usr/bin/env python

import nltk, zipfile, argparse, sys

###############################################################################
## Utility Functions ##########################################################
###############################################################################
# This method takes the path to a zip archive.
# It first creates a ZipFile object.
# Using a list comprehension it creates a list where each element contains
# the raw text of the fable file.
# We iterate over each named file in the archive:
#     for fn in zip_archive.namelist()
# For each file that ends with '.txt' we open the file in read only
# mode:
#     zip_archive.open(fn, 'rU')
# Finally, we read the raw contents of the file:
#     zip_archive.open(fn, 'rU').read()
def unzip_corpus(input_file):
    zip_archive = zipfile.ZipFile(input_file)
    try:
        contents = [zip_archive.open(fn, 'rU').read().decode('utf-8')
                for fn in zip_archive.namelist() if fn.endswith(".txt")]
    except ValueError as e:
        contents = [zip_archive.open(fn, 'r').read().decode('utf-8')
                for fn in zip_archive.namelist() if fn.endswith(".txt")]
    return contents

###############################################################################
## Stub Functions #############################################################
###############################################################################
def process_corpus(corpus_name):

    input_file = corpus_name + ".zip"
    corpus_contents = unzip_corpus(input_file)

    # [list of [list of [list of (word, pos) tuples in a sentence] of sentences] of documents] 
    doc_sent_tuples = []
    tags = []
    words = []
    pos_word_tuples = []
    # split the sentences in each document
    for raw_text in corpus_contents:
        sentences = nltk.sent_tokenize(raw_text)
        sent_tuples = []
        
        # tokenization for each sentence
        for sent in sentences:
            # [list of tokens in a sentence]
            sent_words = nltk.word_tokenize(sent)
            # [list of (word, pos) tuples in a sentence]
            word_pos_tuples = nltk.pos_tag(sent_words)
            # [list of [list of (pos, word) tuples in a sentence] of sentences]
            sent_tuples.append(word_pos_tuples)
            
            for tup in word_pos_tuples:
                word = tup[0]
                pos = tup[1]
                tags.append(pos)
                # lowercase word for word
                words.append(word.lower())#.encode('utf-8')
                # append tuple of (pos, word), lowercase word for word
                pos_word_tuples.append((pos, word.lower()))#.encode('utf-8')         
                
        # [list of [list of [list of (pos, word) tuples in a sentence] of sentences] of documents] 
        doc_sent_tuples.append(sent_tuples)
        
    word_fdist = nltk.FreqDist(words)
    pos_fdist = nltk.FreqDist(tags)
    cfdist = nltk.ConditionalFreqDist(pos_word_tuples)

    # output corpus_pos.txt
    write_tags(corpus_name, doc_sent_tuples)
    # output corpus_word_freq.txt
    write_word_freq(word_fdist, corpus_name)    
    # output corpus_pos_word_freq.txt
    write_pos_cfd(cfdist, corpus_name, set(words))
    
    # stdout
    print("1. Corpus name: "+corpus_name)
    print("") 
    #print("There are", len(corpus_contents), "documents in the corpus")
    print("2. Number of tokens: {0}".format(len(words)))
    print("") 
    print("3. Vocabulary size: {0}".format(word_fdist.B()))
    print("") 
    print("4. Most frequent POS: {0} with freq: {1}".format(pos_fdist.max(), pos_fdist[pos_fdist.max()]))
    print("") 
    print("5. Most frequent words in POS and their similar words:")
    text = nltk.Text(words)
    #print(len(text))
    for pos in ["NN", "VBD", "JJ", "RB"]:
        word = cfdist[pos].max()
        #print(cfdist[pos].most_common(5))
        print("- most similar words to '{0}' in POS '{1}' are:".format(word, pos))
        text.similar(word)
        print("")      
    print("6. Collocations: ")
    text.collocations()

def write_tags(corpus_name, doc_sent_tuples):
    fh = open("{0}-pos.txt".format(corpus_name), "w")
    for document in doc_sent_tuples:
        for sentence in document:
            fh.write(u" ".join(u"{0}/{1}".format(*t) for t in sentence))
            fh.write('\n')
        fh.write('\n')
    fh.close()

def write_word_freq(fdist, corpus_name):
    fh = open("{0}-word-freq.txt".format(corpus_name), "w")
    #for word, freq in fdist.iteritems():
        #fh.write(u"{0:30s} {1:5d}\n".format(word, freq).encode('utf-8'))
    for sample in sorted(fdist, key=fdist.get, reverse=True):
    #for sample in fdist:
        fh.write(u"{0:30s} {1:5d}\n".format(sample, fdist[sample]))
    fh.close()

def write_pos_cfd(cfdist, corpus_name, vocabulary):
    fh = open("{0}-pos-word-freq.txt".format(corpus_name), "w")
    fh.write("POS\t")
    fh.write(u"\t".join(vocabulary))
    fh.write("\n")
    for condition in cfdist.conditions():
        fdist = cfdist[condition]
        fh.write("{0:s}\t".format(condition))
        for word in vocabulary:
            count = fdist.get(word, 0)
            fh.write(str(count) + "\t")
        fh.write("\n")
    fh.close()
    
###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 1')
    parser.add_argument('--corpus', required=True, dest="corpus", metavar='NAME',  help='Which corpus to process {fables, blogs}')

    args = parser.parse_args()
    
    corpus_name = args.corpus
    
    if corpus_name == "fables" or "blogs":
        process_corpus(corpus_name)
    else:
        print("Unknown corpus name: {0}".format(corpus_name))
        
