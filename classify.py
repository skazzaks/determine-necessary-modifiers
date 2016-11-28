"""Builds a classifier based on modifier records and then classifies
unseen records using it."""
import argparse
from classification_record import ModifierRecord
import csv
import logging
import numpy as np
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
log = logging.getLogger('classify')
TRAINING_DATA_PERC = 80


def extract_data_records(data_file):
    """Extracts the records from the datafile.
    data_file - A data file with all the modifier records
    returns - A list of ModifierRecords
    """
    data_records = []
    with open(data_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='|')
        for row in csvreader:
            data_records.append(ModifierRecord(row))

    return data_records


def extract_features(modifier_records):
    """
    Extracts the features we care about from the given list of records

    modifier_records - The list of ModifierRecords
    returns - list of features
    """
    feature_records = []
    targets = []
    for r in modifier_records:
        feature_dict, target = r.get_features_and_target()
        feature_records.append(feature_dict)
        targets.append(target)

    return feature_records, targets


def build_classifier(data_records):
    """builds a classifier from the data_records
    data_records - The list of records that should be used to
    build the classifier

    returns: the built classifier and the vectorizer used
    """

    # Make a classifier
    classifier = svm.SVC(gamma=0.01, C=100.)

    # Get features and targets from the dataset
    feature_list, targets = extract_features(data_records)

    # Fit the data
    vec = DictVectorizer()
    x_data = vec.fit_transform(feature_list).toarray()
    classifier.fit(x_data, targets)

    return classifier, vec


def classify(classifier, vectorizer, data_records):
    """Classifies data_records based on the classifier.

    classifier - the classifier that will classify the data
    vectorizer - the vectorizer that converts the features to a vector
    data_records - the data to be classified
    """
    feature_records, targets = extract_features(data_records)
    x_data = vectorizer.transform(feature_records).toarray()

    print(classifier.predict(x_data))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
                         description='Builds a classifier for the given data \
                         and then classifies unseen data based upon it.')
    argparser.add_argument('data_file', help='The data that is to be ' +
                           'classified.')
    args = argparser.parse_args()

    # Get the data records from the file
    data_records = extract_data_records(args.data_file)

    # Split the dataset into training and test
    training_end_ind = int(len(data_records) * (TRAINING_DATA_PERC / 100.0))
    log.debug(training_end_ind)
    training_data = data_records[:training_end_ind]
    test_data = data_records[training_end_ind:]

    log.debug(len(training_data))
    log.debug(training_data[-1].record_data)
    log.debug(len(test_data))
    log.debug(test_data[0].record_data)

    # Build a classifier from the data records
    log.info("Train classifier")
    classifier, vectorizer = build_classifier(data_records)

    # Classify unseen records
    log.info("Classify unseen records")
    classify(classifier, vectorizer, test_data)
