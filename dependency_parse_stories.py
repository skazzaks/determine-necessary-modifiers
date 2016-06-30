import nltk
from nltk.parse.stanford import StanfordDependencyParser
import sys
import os.path
import time
import datetime
import argparse
import dill
import os
from helpers import Story
from multiprocessing import Process

EXCLUDED_POS = ['NNP', 'NN', 'VBP', 'VBD', '.', 'CC', 'DT', 'NNS', 'IN', 'PRP',
                'TO', 'VB', 'PRP$', 'VBG', ',', 'VBN', 'VBZ', 'WP']
WORDS = ['not', '\'nt']
REMOVABLE_TYPES = ['advmod', 'amod']
DELIMITER = '|'
TOKENS_WITHOUT_SPACE = ["'", ".", "!", ":", ",", "n't"]


class ROC_Stories_Reader():
    def __init__(self, filename):
        self.f = open(filename, 'r')

    # gets the next line of the data record in the format
    # [storyid, title, 1, 2, 3, 4, 5]
    def get_next_record(self):
        before = self.f.tell()
        l = self.f.readline()
        after = self.f.tell()
        if before == after:
            self.f.close()
            return False

        final = l.split('/')
        return "ROC", final


def parse_sentence(parser, sent):
    """Parses the target sentence and returns it"""

    # parse the sentence
    return list(parser.raw_parse(sent))

def do_loop(proc, parser, start_line, num_to_proc, output_dir, process_num):

    for i in range(start_line - 1):
        record_type, l = proc.get_next_record()

    i = start_line
    processed_count = 0
    start_time = time.time()
    while True:
        record_type, l = proc.get_next_record()
        end_time = time.time()
        full_time = end_time - start_time
        print("Story: " + str(i) + '[' + str(processed_count + 1) +
              '|'+ str(num_to_proc) + "] Time: "
                            + str(datetime.timedelta(seconds=full_time)))

        if l:
            original_sentences = []
            dparsed_sentences = []
            original_title = l[1]
            story_text = ''
            for line_num, sent in enumerate(l[2:]):
                if line_num is not 0:
                    story_text += ' '

                original_sentences.append(sent)
                story_text += sent

            story_text = story_text.replace('\n', '').replace('\r', '')
            dparsed_title = parse_sentence(parser, l[1])

            for line_num, sent in enumerate(l[2:]):
                p = parse_sentence(parser, sent)
                dparsed_sentences.append(p)

            story = Story(l[0], story_text, 'ROC', original_sentences,
                          original_title, dparsed_sentences, dparsed_title)
            # dump the record out to the file system
            dill.dump(story, open(output_dir +
                                  '/story_' + str(i) + '.dat', 'wb'))
            i += 1
            processed_count += 1

        else:
            break

        if processed_count >= num_to_proc:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serialize stories to file')
    input_file = sys.argv[1]
    start_record = sys.argv[2]
    number_of_records = sys.argv[3]
    output_directory = sys.argv[4]

    # Get the number of lines in a file
    with open(input_file) as f:
        for i, l in enumerate(f):
            pass
    total = i + 1

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    start_line = 0

    # get a ref to the Stanford DepParser
    parser = StanfordDependencyParser(
        model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

    proc = ROC_Stories_Reader(input_file)

    p = Process(target=do_loop,
                args=(proc, parser, int(start_record),
                        int(number_of_records),
                        output_directory, i))
    p.start()


