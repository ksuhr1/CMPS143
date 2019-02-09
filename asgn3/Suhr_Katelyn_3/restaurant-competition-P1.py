
import re, nltk, pickle, argparse
import os
import data_helper
import sys
from features import get_features_category_tuples


DATA_DIR = "data"


def write_features_category(features_category_tuples, output_file_name):
    output_file = open("{}-features.txt".format(output_file_name), "w", encoding="utf-8")
    for (features, category) in features_category_tuples:
        output_file.write("{0:<10s}\t{1}\n".format(category, features))
    output_file.close()


def get_classifier(classifier_fname):
    classifier_file = open(classifier_fname, 'rb')
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier


def save_classifier(classifier, classifier_fname):
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()
    info_file = open(classifier_fname.split(".")[0] + '-informative-features.txt', 'w', encoding="utf-8")
    for feature, n in classifier.most_informative_features(100):
        info_file.write("{0}\n".format(feature))
    info_file.close()


def evaluate(classifier, features_category_tuples, reference_text, data_set_name=None, prediction=None):
    print("Evaluating model")

    accuracy = nltk.classify.accuracy(classifier, features_category_tuples)
    print("{0:6s} {1:8.5f}".format("Accuracy", accuracy))

    features_only = [example[0] for example in features_category_tuples]
    reference_labels = [example[1] for example in features_category_tuples]
    predicted_labels = classifier.classify_many(features_only)
    with open(prediction,'a') as f:
        for label in predicted_labels:
            f.write(label)
            f.write("\n")
        f.close()

    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)
    #print(confusion_matrix)

    return accuracy, confusion_matrix


def build_features(data_file, feat_name, save_feats=None):
    # read text data

    if data_file == "test.txt":
        test_texts = data_helper.get_reviews(os.path.join(DATA_DIR, data_file))
        category_texts = {"test data": test_texts}
    else:        
        positive_texts, negative_texts = data_helper.get_reviews(os.path.join(DATA_DIR, data_file))

        category_texts = {"positive": positive_texts, "negative": negative_texts}
    
    # build features
    features_category_tuples, texts = get_features_category_tuples(category_texts, feat_name)

    # save features to file
    if save_feats is not None:
        write_features_category(features_category_tuples, save_feats)
    
    return features_category_tuples, texts



def train_model(datafile, feature_set, save_model=None):
    
    file_name = feature_set + save_model

    features_data, texts = build_features(datafile, feature_set, save_feats=file_name)
    ###     YOUR CODE GOES HERE
    # TODO: train your model here
    
    #Train on the training data
    classifier = nltk.classify.NaiveBayesClassifier.train(features_data)
    
    if save_model is not None:
        save_classifier(classifier, file_name)
    return classifier


def train_eval(train_file, feature_set, eval_file=None, pred_file=None):
    
    if train_file == "train_examples.tsv":
        save_data = "-training"
    elif train_file == "dev_examples.tsv":
        save_data = "-development"
    elif train_file == "test.txt":
        save_data = "-testing"
    else:
        save_data = "-unknown"
    
    model = train_model(train_file, feature_set, save_model=save_data)
   
    #model.show_most_informative_features(20)
    # save the model
    if model is None:
        #model = get_classifier(classifier_fname)
        model = save_classifier(model, classifier_name)

 
    # evaluate the model
    if eval_file is not None:
        #builds features on development set
        if eval_file == "test.txt":
            dev_name = feature_set+"-development"
            print("Building dev features for "+feature_set)
            dev_features, devtexts = build_features("dev_examples.tsv", feature_set, save_feats=dev_name)
            file_name = feature_set+"-testing"
            print("Building test features for "+feature_set)
            features_data, texts = build_features(eval_file, feature_set, save_feats=file_name)
        elif eval_file == "dev_examples.tsv":
            test_name = feature_set+"-testing"
            print("Building test features for "+feature_set)
            test_features, testext = build_features("test.txt", feature_set, save_feats=test_name)
            print("Building dev features for "+feature_set)
            file_name = feature_set+"-development"
            features_data, texts = build_features(eval_file, feature_set, save_feats=file_name)
        else:
            file_name="-unknown_data_set"
            features_data, texts=build_features(eval_file,feature_set, save_feats=file_name)


        accuracy, cm = evaluate(model, features_data, texts, data_set_name=None,prediction=pred_file)


        allresults = False
        if feature_set == "word_features":
            outfile = "output-ngrams.txt"
            allresults = True
        elif feature_set == "word_pos_features":
            outfile = "output-pos.txt"
            allresults = True
        elif feature_set == "word_pos_liwc_features":
            outfile = "output-liwc.txt"
            allresults = True
        #include best classifier
        else:
            outfile = ""

        #bestoutfile = "output-best.txt"

        #direct output to all-results.txt for word_features and word_pos_features
        results_path = "all-results.txt"
        mode = 'a' if os.path.exists(results_path) else 'w'
        if allresults is True:
            resultout = open(results_path, mode)
            sys.stdout = resultout 
            print(feature_set)  
            print("The accuracy of {} is: {}".format(eval_file, accuracy))
            print("Confusion Matrix:")
            print(str(cm))
            sys.stdout = sys.__stdout__

        #evaluate function file on dev data for different features
        file_results = open(outfile,'w')
        sys.stdout = file_results
        print("The accuracy of {} is: {}".format(eval_file, accuracy))
        print("Confusion Matrix:")
        print(str(cm))
        #redirect sys stdout
        sys.stdout = sys.__stdout__
        
        if feature_set == "word_features":
            #best results
            best_results = open("output-best.txt",'w')
            sys.stdout = best_results
            print("The accuracy of {} is: {}".format(eval_file, accuracy))
            print("Confusion Matrix:")
            print(str(cm))
            sys.stdout = sys.__stdout__
    else:
        accuracy = None

    return accuracy


def main():


    # add the necessary arguments to the argument parser
    parser = argparse.ArgumentParser(description='Assignment 3')

    parser.add_argument('-e', dest="eval_fname", default="dev_examples.tsv",
                        help='File name of the testing data.')

    parser.add_argument('-o', dest="pred_fname", default="restaurant-competition-model-P1-predictions.txt",
                        help="File name of predictions for the best classifier")

    args = parser.parse_args()
    
    train_data = "train_examples.tsv"
    eval_data = args.eval_fname
    pred_data = args.pred_fname    

    for feat_set in ["word_features", "word_pos_features", "word_pos_liwc_features"]:
        print("\nTraining with {}".format(feat_set))
        acc = train_eval(train_data, feat_set, eval_file=eval_data, pred_file=pred_data)


if __name__ == "__main__":
    main()


