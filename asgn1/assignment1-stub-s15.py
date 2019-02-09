#!/usr/bin/env python

import nltk, zipfile, argparse, pprint, sys

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
    # Your code goes here
    file_name = corpus_name+"-pos.txt"
    freq_file = corpus_name+"-word-freq.txt"
    cond_file = corpus_name+"-pos-word-freq.txt"
    #write name of corpous to stdout
    totalcount = 0
    vocabsize = 0
    poslist = []
    taglist = []
    wordslist = []
    unqiuelist = []
    wordstr = []
    tuplearr = []
    normaltext = []
    beginarr = []
    corpusarr = []
    with open(file_name,'a') as f:
        for doc in corpus_contents:
            #delimit the sentences for each document in the corpus
            sent = nltk.sent_tokenize(doc)
            #part-of-speech tagger to each tokenize sent
            #tokenize the words of each sentence of each doc
            words = [nltk.word_tokenize(item) for item in sent]
            for word in words:
                corpusarr.append(word)
                #lowercase words
                flat_words = [term.lower() for term in word]
                lowerfreq = nltk.FreqDist(flat_words)
                vocabsize += lowerfreq.B() 
                
                #make array of tokenized words of corpus  
                for i in word:
                    normaltext.append(i)
                #make an array of lowercase tokenized words of corpus
                for i in flat_words:
                    wordstr.append(i)

                #count total words in corpus
                freq = nltk.FreqDist(word)
                totalcount += freq.N()
                    
                #pos tagging
                poslist = nltk.pos_tag(word)
                #reverse tuple of pos tagging
                for item in poslist:
                    tlist = tuple(reversed(item))
                    tuplearr.append(tlist)
                    beginarr.append(item)
                #make second value of tuple lowercase
                newtuple = [(pos,word.lower()) for pos,word in tuplearr]
#                print(newtuple)
                #most freq part of speech
                pos_counter = nltk.FreqDist(pos for (word, pos) in poslist)
                for word,tag in poslist:
                    wordslist.append(word)
                    taglist.append(tag)
                for val in poslist:
                    combined = nltk.tuple2str(val)
                    f.write(combined)
                    f.write(" ")
                f.write("\n")
            f.write("\n")
        #gett frequency of unique words
        uniquefreq = nltk.FreqDist(wordstr)
        uniquedict = uniquefreq.most_common(15000)

        with open(freq_file,'a') as r:
            for k,j in uniquedict:
                r.write(k+", "+ str(j))
                r.write("\n")
        tagfreq = nltk.FreqDist(taglist)
        winner = tagfreq.most_common(50)
    print("1. Corpus name:", corpus_name)
    print("2. Total words in corpus", totalcount)
    print("3. Vocabulary size of the corpus", uniquefreq.B())
    print("4. The most frequent part-of-speech tag is", winner[0][0], "with frequency", winner[0][1])
    condfileoutput = open(cond_file,'a')
    sys.stdout = condfileoutput
    cflist = nltk.ConditionalFreqDist(newtuple)
    cflist.tabulate()
    sys.stdout = sys.__stdout__
    print(cflist)

    noun = cflist['NN'].most_common(1)
    noun1 = noun[0][0]
    print(normaltext)
    text = nltk.Text(normaltext)
    
    print("5. The most frequent word in the POS(NN) is:", noun1,"and its similar words are:")
    text.similar(noun1)
    vbd = cflist['VBD'].most_common(1)
    vbd1 = vbd[0][0]
    print("5. The most frequent word in the POS(VBD) is:",vbd1,"and its similar words are:")
    text.similar(vbd1)
    jj = cflist['JJ'].most_common(1)
    jj1 = jj[0][0]
    print("5. The most frequent word in the POS(JJ) is:",jj1,"and its similar words are:")
    text.similar(jj1)
    rb = cflist['RB'].most_common(1)
    rb1 = rb[0][0]
    print("5. The most frequent word in the POS(RB) is:",rb1,"and its similar words are:")
    text.similar(rb1)
    print("6. Collocations:")
    text.collocations()

    pass

###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 1')
    parser.add_argument('--corpus', required=True, dest="corpus", metavar='NAME',
                        help='Which corpus to process {fables, blogs}')

    args = parser.parse_args()
    
    corpus_name = args.corpus
    
    if corpus_name == "fables" or "blogs":
        process_corpus(corpus_name)
    else:
        print("Unknown corpus name: {0}".format(corpus_name))
        
