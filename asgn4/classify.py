
import re, nltk, pickle, argparse
import os, sys
import data_helper
from features import get_features_category_tuples, get_features_from_texts

from sklearn import svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import sklearn, numpy
import pandas as pd
import random
from sklearn import tree
from nltk.classify import SklearnClassifier

DATA_DIR = "data"

random.seed(10)


MODEL_DIR = "models/"
OUTPUT_DIR = "output/"
FEATURES_DIR = "features/"

# This will be used for feature selection later on
# Initialize it to None so that we use all the features the first time around
#selected_features = None


def write_features_category(features_category_tuples, output_file_name):
    print("write_features_category")
    output_file = open("{}-features.txt".format(output_file_name), "w", encoding="utf-8")
    for (features, category) in features_category_tuples:
        output_file.write("{0:<10s}\t{1}\n".format(category, features))
    output_file.close()


def get_classifier(classifier_fname):
    print("get classifier")
    classifier_file = open(classifier_fname, 'rb')
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier


def save_classifier(classifier, classifier_fname):
    print("save classifier")
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()
    info_file = open(classifier_fname.split(".")[0] + '-informative-features.txt', 'w', encoding="utf-8")
    for feature, n in classifier.most_informative_features(100):
        info_file.write("{0}\n".format(feature))
    info_file.close()


def evaluate(classifier, features_category_tuples, reference_text, data_set_name=None):
    print("evaluating")
    # test on the data
    accuracy = nltk.classify.accuracy(classifier, features_category_tuples)
    print("{0:6s} {1:8.5f}".format("Accuracy", accuracy))

    accuracy_results_file = open("{}_results.txt".format(data_set_name), 'w', encoding='utf-8')
    accuracy_results_file.write('Results of {}:\n\n'.format(data_set_name))
    accuracy_results_file.write("{0:10s} {1:8.5f}\n\n".format("Accuracy", accuracy))

    features_only = []
    reference_labels = []
    for feature_vectors, category in features_category_tuples:
        features_only.append(feature_vectors)
        reference_labels.append(category)

    predicted_labels = classifier.classify_many(features_only)

    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)
    print(confusion_matrix)

    accuracy_results_file.write(str(confusion_matrix))
    accuracy_results_file.write('\n\n')
    accuracy_results_file.close()

    predict_results_file = open("{}_output.txt".format(data_set_name), 'w', encoding='utf-8')
    for reference, predicted, text in zip(
            reference_labels,
            predicted_labels,
            reference_text
    ):
        if reference != predicted:
            predict_results_file.write("{0} {1}\n{2}\n\n".format(reference, predicted, text))
    predict_results_file.close()

    return accuracy, confusion_matrix



def build_features(data_file, feat_name, save_feats=None):
    # read text data
    positive_texts, negative_texts = data_helper.get_reviews(os.path.join(DATA_DIR, data_file))

    category_texts = {"positive": positive_texts, "negative": negative_texts}

    # build features
    features_category_tuples, texts = get_features_category_tuples(category_texts, feat_name)

    # save features to file
    if save_feats is not None:
        write_features_category(features_category_tuples, save_feats)
    
    return features_category_tuples, texts



def train_model(datafile, feature_set, split_name, save_model=None, save_feats=None, binning=False):

    features_data, texts = build_features(datafile, feature_set)

    #train on the training data of word_features
    classifier = nltk.classify.NaiveBayesClassifier.train(features_data)

    if save_model is not None:
        save_classifier(classifier, save_model)
    return classifier


#helper function to train model on scikit-learn classifiers
def train_scikit_model(best_features, feature_set, split_name, classifier_name):

    #train on the training data of word_features
    
    #find which classifier model to use
    if classifier_name == "nb":
        cls = nltk.classify.NaiveBayesClassifier.train(best_features)
    elif classifier_name == "nb_sk":
        cls = SklearnClassifier(BernoulliNB()).train(best_features)
    elif classifier_name == "dt":
        cls = nltk.classify.DecisionTreeClassifier.train(best_features)
    elif classifier_name == "dt_sk":
        cls = SklearnClassifier(tree.DecisionTreeClassifier()).train(best_features)
    elif classifier_name == "svm_sk" or classifier_name == "svm":
        cls = SklearnClassifier(svm.SVC())
    else:
        assert False, "unknown classifier name:{}; known names: nb, dt, svm, nb_sk, dt_sk, svm_sk".format(classifier_name)
    return cls

def train_scikit_eval(best_features, eval_file, feature_set, classifier_name, results=False):

    # train the model
    split_name = "train"
    
    #model = train_model(train_file, feature_set, split_name)
    model = train_scikit_model(best_features, feature_set, split_name, classifier_name)

    #model.show_most_informative_features(20)

    # evaluate the model
    if model is None:
        model = get_classifier(classifier_fname)

    if eval_file == "dev_examples.tsv":
        eval_name="dev-"
    elif eval_file == "test_examples.tsv":
        eval_name="test-"
    else:
        eval_name="unknown-"

    features_data, texts = build_features(eval_file, feature_set)
    accuracy, cm = evaluate(model, features_data, texts, data_set_name=eval_name+"eval-{}".format(feature_set))

    results_file = open(results,'a')
    sys.stdout = results_file
    print(classifier_name)
    print("The accuracy of {} is: {}".format(eval_file, accuracy))
    print("Confusion Matrix:")
    print(str(cm))
    sys.stdout = sys.__stdout__

    return accuracy



def train_eval(train_file, eval_file, feature_set, results=False):

    # train the model
    split_name = "train"
    
    model = train_model(train_file, feature_set, split_name)

    model.show_most_informative_features(20)

    # evaluate the model
    if model is None:
        model = get_classifier(classifier_fname)

    if eval_file == "dev_examples.tsv":
        eval_name="dev-"
    elif eval_file == "test_examples.tsv":
        eval_name="test-"
    else:
        eval_name="unknown-"

    features_data, texts = build_features(eval_file, feature_set)
    accuracy, cm = evaluate(model, features_data, texts, data_set_name=eval_name+"eval-{}".format(feature_set))
    
    results_file = open(results,'a')
    sys.stdout = results_file
    print("\nThe accuracy of {} is: {}".format(eval_file, accuracy))
    print("Confusion Matrix:")
    print(str(cm))
    sys.stdout = sys.__stdout__

    return accuracy

def filter_features(feature_info, selected_features):
    feature_category_tuple = []
    for feats,cat in feature_info:
        feature_vector = {}
        for fname in feats:
            if selected_features == None or fname in selected_features:
                count = feats.get(fname)
                feature_vector[fname] = count
        if len(feature_vector) != 0:
            feature_category_tuple.append((feature_vector, cat))

    return feature_category_tuple

def feature_selection():
    global selected_features 

    feature_set = "word_features"
    train_data = "train_examples.tsv"
    develop_data = "dev_examples.tsv"
    test_data = "test_examples.tsv"
    print("In the process of feature selection for feature set: {}".format(feature_set))

    #get feature vectors for train file
    train_features, texts = build_features(train_data, feature_set)
    classifier = nltk.NaiveBayesClassifier.train(train_features)

    ## Feature selection

    #look at best features of training set
    best = (0.0, 0)
    best_features = classifier.most_informative_features(10000)
    #loops through 10,000 best features
    for i in [2**i for i in range(5, 15)]:
        selected_features = set([fname for fname, value in best_features[:i]])
        train_features, texts = build_features(train_data, feature_set)
        dev_features, texts = build_features(develop_data, feature_set)
        test_features, texts = build_features(test_data, feature_set)

        train_tuples = filter_features(train_features, selected_features)
        dev_tuples = filter_features(dev_features, selected_features)
    
        #returns the acccuracy of that subset of features on the development data
        #classifier = nltk.NaiveBayesClassifier.train(train_features)

        classifier = nltk.NaiveBayesClassifier.train(train_tuples)
        accuracy = nltk.classify.accuracy(classifier, dev_tuples)
        print("accuracy on dev data on subset {} of features".format(i))
        print("{0:6d} {1:8.5f}".format(i, accuracy))
    
         
        #finds absolute best features of all the subsets
        if accuracy > best[0]:
            best = (accuracy, i)

     # Now train on the best features
    selected_features = set([fname for fname, value in best_features[:best[1]]])
    train_features, texts = build_features(train_data, feature_set)
    dev_features, texts = build_features(develop_data, feature_set)
    test_features, test_texts = build_features(test_data, feature_set)

    train_tuples = filter_features(train_features, selected_features)
    test_tuples = filter_features(test_features, selected_features)

    classifier = nltk.NaiveBayesClassifier.train(train_tuples)
    accuracy = nltk.classify.accuracy(classifier, test_tuples)
    print("accuracy on test data with best features")
    print("{0:6s} {1:8.5f}".format("Test", accuracy))
    evaluate(classifier, test_tuples, test_texts, "best_features")
    
    print("Returning best features from subset {} with an accuracy of {}".format(i, accuracy))
    return feature_set, train_tuples


def predict_main(review_file, pred_file):
    # train the model
    train_file = "train_examples.tsv"
    split_name = "train"
    results = []

    feat_set = ["word_features", "word_pos_features",
                "word_pos_liwc_features",
                "only_liwc"
    ]
    
    classifier_set = [ 'nb_sk','dt_sk']

    feature_set, best_feats = feature_selection()
    scikit_file = "scikit_results.tsv"    

    for cname in classifier_set:    
        print("\n" + "-"*50)
        print("  * training the model with {} classifier".format(cname))
        model = train_scikit_model(best_feats, feature_set, split_name, cname)

        accur = train_scikit_eval(best_feats, review_file, feature_set, cname, results=scikit_file)
        print("onto the next...")
    for feat_set in feat_set:
        print("\n" + "-"*50)
        print("  * training the model with {}".format(feat_set))
        model = train_model(train_file, feat_set, split_name)
           
        acc = train_eval(train_file, review_file, feat_set, results=pred_file)

        results.append({
            "features": feat_set,
            "accuracy": acc,
            "review_file": review_file
        })

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index = False)
    #go on to feature selection

    """
    print("  * calculating test set predictions and saving to {}".format(pred_file))
    test = data_helper.get_reviews(review_file)
    feats = get_features_from_texts(test, feature_set)

    with open(pred_file, "w") as fout:
        for feat in feats:
            pred = model.classify(feat)
            fout.write("{}\n".format(pred))
    """



"""
def train_main():

    train_data = "train_examples.tsv"
    eval_data = "dev_examples.tsv"
    results = []

    feat_set = [
        "word_features", "word_pos_features",
        "word_pos_liwc_features",
        "only_liwc"
    ]
    for feat_set in feat_set:
        print("\n" + "-"*50)
        print("Training with {} \n".format(feat_set))
        acc = train_eval(train_data, eval_data, feat_set)

        results.append({
            "features": feat_set,
            "accuracy": acc,
        })

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
"""

if __name__ == "__main__":
    """
    (a) The file with the reviews in it to be classified.
    (b) The second should be the name of a file to write predictions to. When saving predictions, each
    predicted label should be on a separate line in the output file, in the same order as the input file.
    This file should be the output of a function called evaluate. The evaluate function should also
    calculate the accuracy and confusion matrix if it is supplied with the example labels.
    """

    "data/test.txt"
    parser = argparse.ArgumentParser(description='Assignment 3')
    parser.add_argument('-r', dest="reviews", default="dev_examples.tsv", required=False,
                        help='The file with the reviews in it to be classified.')
    parser.add_argument('-p', dest="pred_file", default="predictions.txt", required=False,
                        help='The file to write predictions to.')

    args = parser.parse_args()
    eval_data = args.reviews
    pred_data = args.pred_file
    

    #if args.reviews is not None:
        # An example way of calling the script to make predictions
        # python3 classify.py -r data/test.txt -p test-preds.txt
    predict_main(args.reviews, args.pred_file)
    
    #else:
    #    train_main()





