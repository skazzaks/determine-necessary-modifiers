"""Builds a classifier based on modifier records and then classifies
unseen records using it."""
import argparse
from classification_record import AnnotatedModifierRecord
import csv
import logging
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
log = logging.getLogger('classify')
TRAINING_DATA_PERC = 80

# Cheating to use a global - just want to store the headers
csv_record_headers = ''


def extract_data_records(data_file):
    """Extracts the records fro m the datafile.
    data_file - A data file with all the modifier records
    returns - A list of ModifierRecords
    """
    data_records = []
    with open(data_file, 'r') as csvfile:
        global csv_record_headers
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        headers = next(csvreader)
        csv_record_headers = '|'.join(headers)
        log.debug(headers)
        last_id = None
        last_record = None
        for row in csvreader:
            current_id = row[0]

            if current_id == last_id:
                last_record.add_annotation_data(row, headers)
            else:
                current_record = AnnotatedModifierRecord(row, headers)
                data_records.append(current_record)

            last_id = current_id
            last_record = current_record
            log.debug(last_record)

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
    classifier = svm.SVC(kernel='linear', gamma=0.01, C=100.)

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

    # Get the list of predictions
    predictions = classifier.predict(x_data)

    log.debug('Predictions: ' + str(len(predictions)) + ' | Targets: '
              + str(len(x_data)))

    return targets, predictions


def filter_out_non_helpful_records(data_records):
    """Filters out records we don't want to train on. Right now this only
    includes records where there was no consensus in the annotation"""
    for r in data_records:
        if r.get_cruciality() is None:
            continue
        # We only want records that got exactly 3 votes
        if len(r.cruciality) != 3:
            continue
        else:
            yield r


def analyze_results(data_records):
    """Does a quick analysis of the different types of results acquired"""
    # Let's calculate what percentage of records were had which level of
    # agreement
    total_count = 0
    #
    totals_by_agreement = {}
    totals_by_type = {}

    for r in data_records:
        # We only want the cases with 3 records
        if len(r.cruciality) != 3:
            continue

        # Update the total count
        total_count += 1

        # Build up the agreement map
        key = ' '.join(sorted(r.cruciality))

        if key not in totals_by_agreement.keys():
            totals_by_agreement[key] = []

        totals_by_agreement[key].append(r)

        # Build up type map
        cruciality = r.get_cruciality()
        if cruciality not in totals_by_type.keys():
            totals_by_type[cruciality] = []

        totals_by_type[cruciality].append(r)

    with open('data_analysis.txt', 'w') as o:
        o.write('Total Record Count: ' + str(total_count) + '\n\n')
        o.write('Agreement types and counts\n\n')
        total_count = float(total_count)
        for k, v in totals_by_agreement.items():
            perc = float(len(v)) / total_count
            o.write(str(k) + ':\t' + str(len(v)) + '\t' + str(perc) + '\n')

        o.write('\nCruciality Counts\n\n')

        for k, v in totals_by_type.items():
            perc = float(len(v)) / total_count
            o.write(str(k) + ':\t' + str(len(v)) + '\t' + str(perc) + '\n')


def evaluate(targets, predictions):
    right_count = 0
    total_count = 0

    for i in range(0, len(targets)):
        total_count += 1

        if targets[i] == predictions[i]:
            right_count += 1

    right_perc = right_count / total_count
    log.info('Right: ' + str(right_count) + '/' + str(total_count) +
             ' Perc: ' + str(right_perc) + '%')


def write_out_ext_spreadsheet(test_data, targets, predictions):
    """This writes out a new spreadsheet that is complemented with the new
    information we have learned: the prediction of cruciality."""
    with open('modifiers_with_predictions.csv', 'w') as o:
        o.write(csv_record_headers +
                '| aggregate_cruciality|'
                + 'predicted_cruciality\n')

        # Write out each record
        for i, r in enumerate(test_data):
            o.write(r.original_line)
            o.write('| ' + str(r.get_cruciality()))
            o.write('| ' + predictions[i])
            o.write('\n')

    with open('stats_incorrect_predictions.txt', 'w') as o:
        wrong_map = {}
        for i, r in enumerate(test_data):
            pred = predictions[i]
            act = targets[i]

            if pred == act:
                continue

            # We got an error, let's see which type
            key = pred + '-> ' + act

            if key not in wrong_map.keys():
                wrong_map[key] = 0

            wrong_map[key] += 1

        for k, v in wrong_map.items():
            o.write(k + '\t' + str(v) + '\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
                         description='Builds a classifier for the given data \
                         and then classifies unseen data based upon it.')
    argparser.add_argument('data_file', help='The data that is to be ' +
                           'classified.')
    args = argparser.parse_args()

    # Get the data records from the file
    data_records = extract_data_records(args.data_file)

    # Now that we have all the records, let's do some analysis on them.
    log.info('Analyzing Results')
    analyze_results(data_records)

    # Split the dataset into training and test
    training_end_ind = int(len(data_records) * (TRAINING_DATA_PERC / 100.0))
    log.debug(training_end_ind)
    training_data = data_records[:training_end_ind]
    test_data = data_records[training_end_ind:]

    # Filter out some records we don't care about
    training_data = list(filter_out_non_helpful_records(training_data))

    log.debug('Training data: ' + str(len(training_data)))
    log.debug('Test data: ' + str(len(test_data)))

    # Build a classifier from the data records
    log.info("Train classifier")
    classifier, vectorizer = build_classifier(training_data)

    # Classify unseen records
    log.info("Classify unseen records")
    targets, predictions = classify(classifier, vectorizer, test_data)

    # Evaluation
    log.info('Evaluation results')
    evaluate(targets, predictions)

    # Write out the records to file
    log.info('Writing out new expanded spreadsheet')
    write_out_ext_spreadsheet(test_data, targets, predictions)
