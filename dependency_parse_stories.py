import glob
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
        self.filename = filename
    def get_total_count(self):
        total = 0

        # Get the number of lines in a file
        with open(self.filename) as f:
            for i, l in enumerate(f):
                pass
        total = i + 1
        return total

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
        return "ROC", final[0], final[1], final[2:]


class microtext_Reader():
    def __init__(self, path_to_files):
        self.filelist = list(glob.glob(path_to_files + "/*.txt"))
        self.current_index = -1

    def get_total_count(self):
        return len(self.filelist)

    def get_next_record(self):
        self.current_index += 1

        if self.current_index >= len(self.filelist):
            return False

        thefile = self.filelist[self.current_index]
        with open(thefile, 'r') as f:
            sentences = nltk.sent_tokenize(f.read())

        return "MICRO", thefile, "[no title]", sentences


def parse_sentence(parser, sent):
    """Parses the target sentence and returns it"""

    # parse the sentence
    return list(parser.raw_parse(sent))

def do_loop(proc, parser, start_line, num_to_proc, output_dir, process_num):

    for i in range(start_line - 1):
        proc.get_next_record()

    i = start_line
    processed_count = 0
    start_time = time.time()
    while True:
        record_type, id, title, sentences  = proc.get_next_record()
        end_time = time.time()
        full_time = end_time - start_time
        print("Story: " + str(i) + '[' + str(processed_count + 1) +
              '|'+ str(num_to_proc) + "] Time: "
                            + str(datetime.timedelta(seconds=full_time)))

        if sentences:
            original_sentences = []
            dparsed_sentences = []
            story_text = ''
            for line_num, sent in enumerate(sentences):
                if line_num is not 0:
                    story_text += ' '

                original_sentences.append(sent)
                story_text += sent

            story_text = story_text.replace('\n', '').replace('\r', '')
            dparsed_title = parse_sentence(parser, title)

            for line_num, sent in enumerate(sentences):
                p = parse_sentence(parser, sent)
                dparsed_sentences.append(p)
                print(line_num)
                print(sentences)

            story = Story(id, story_text, record_type, original_sentences,
                          title, dparsed_sentences, dparsed_title)
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
    parser.add_argument('input_file', help='The input file or directory, ' +
                        'depending on the file type.')
    parser.add_argument('output_directory',
                        help='The directory for the output')
    parser.add_argument('file_type', help='ROC or MICRO')
    args = parser.parse_args()

    proc = None
    if args.file_type == 'ROC':
        proc = ROC_Stories_Reader(args.input_file)
    else:
        proc = microtext_Reader(args.input_file)

    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)
    start_line = 0

    # get a ref to the Stanford DepParser
    parser = StanfordDependencyParser(
        model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

    p = Process(target=do_loop,
                args=(proc, parser, 1,
                      proc.get_total_count(),
                      args.output_directory, 1))
    p.start()
