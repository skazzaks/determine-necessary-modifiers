from nltk import word_tokenize
import os.path
import os
import time
import datetime
import dill
import argparse
from multiprocessing import Process

EXCLUDED_POS = ['NNP', 'NN', 'VBP', 'VBD', '.', 'CC', 'DT', 'NNS', 'IN', 'PRP',
                'TO', 'VB', 'PRP$', 'VBG', ',', 'VBN', 'VBZ', 'WP']
WORDS = ['not', '\'nt']
REMOVABLE_TYPES = ['advmod', # includes quantmod
                   'amod',
                   'appos',
                   'compound',
                   'nmod:npmod',
                   'nummod', # Same ate [3] sheep [number and num
                   #'quantmod', # included in 'advmod'
                   'nmod:poss', # Bill's clothes ['poss']
                   'det:predet', # All the boys are here ['predet']
                   'case', # prepositions
                   'nmod:tmod',
                   #'vmod' #     didn't exist, under xcomp, but too general
                   ]
DELIMITER = '|'
TOKENS_WITHOUT_SPACE = ["'", ".", "!", ":", ",", "n't"]


def get_statistics_for_sentence(prev_stats, interesting_records):
    prev_stats['count'] += len(interesting_records)
    prev_stats['sentences'] += 1

    for r in interesting_records:
        r_t = r[1]
        r_pos = r[2][1]

        ts = prev_stats['types']

        if r_t not in ts:
            ts[r_t] = 0

        ts[r_t] += 1

        tpos = prev_stats['pos']

        if r_pos not in tpos:
            tpos[r_pos] = 0

        tpos[r_pos] += 1


def handle_sentence(story, story_filename, sentence_num, output_file, stats):
    """Handle the given sentence and output the results to output_file."""
    # parse the sentence
    # p = [list(parse.triples()) for parse in parser.raw_parse(sent)]
    # now it is just a matter of iterating over our items and removing them
    # one by one
    # first, get just the set we care about
    count = 0

    # get the first dependency tree
    dsent = story.dparsed_sentences[sentence_num - 1]
    sent = story.original_sentences[sentence_num - 1]

    # get the triples of the dep graph
    p = [list(parse.triples()) for parse in dsent]

    interesting_records = list(x for x in p[0] if x[1] in REMOVABLE_TYPES)
    count = len(interesting_records)

    # Get statistics based on the sentence
    results = get_statistics_for_sentence(stats, interesting_records)

    # if the count is greater than 0, we have something to remove, so let's go!
    if count == 0:
        return results

    t_sent = word_tokenize(sent)
    # INTERESTING SENTENCE FOUND - let's write it out to the file

    # Remove the words full mesh - first 1, then 2, ...X
    for x in range(0, count):
        words_to_remove = x + 1

        for y in range(0, count - x):
            words_removed = 0
            output_file.write(story.source + DELIMITER +
                              story.storyid + DELIMITER +
                              story_filename + DELIMITER + story.full_story +
                              DELIMITER + story.original_title +
                              DELIMITER + str(sentence_num) + DELIMITER)

            # Go though each word of the sentence and see if it should be
            # removed.
            full_sentence_with_words_to_remove = ''
            for ind, w in enumerate(t_sent):
                match_found = False
                for r in interesting_records[y: y + words_to_remove]:
                # r is like [(('overweight', 'VBN'), 'advmod', ('as', 'RB'))]
                    r_w = r[2][0]
                    if r_w.upper() == w.upper():
                        words_removed += 1
                        match_found = True
                        break
                if match_found:
                    full_sentence_with_words_to_remove += ' [' + w + ']'
                    continue
                else:
                    if ind == 0 or w in TOKENS_WITHOUT_SPACE:
                        output_file.write(w)
                        full_sentence_with_words_to_remove += w
                    else:
                        output_file.write(" " + w)
                        full_sentence_with_words_to_remove += ' ' + w


            output_file.write(DELIMITER + full_sentence_with_words_to_remove)
            # Output the interesting words
            output_file.write(DELIMITER +
                              str([r[2][0] for r in
                               interesting_records[y:y+words_to_remove]]))

            output_file.write(DELIMITER +
                              str(interesting_records[y:y+words_to_remove]))

            # Output the full sentence
            output_file.write(DELIMITER + sent.replace('\r','').replace('\n', ''))
            if words_removed > count:
                output_file.write(DELIMITER + "Too many words removed!")

            output_file.write('\r\n')

def output_stats(output_file, stats):
    with open(output_file, 'w') as f:
        f.write('Phenomena per sentence: ' +
                str(stats['count'] / stats['sentences']) + '\r\n')

        f.write('Phenomena per story: ' +
                str(stats['count'] / stats['stories']) + '\r\n')


        f.write('Dependency Types:\r\n')

        for k,v in stats['types'].items():
            f.write('\t'  + k + ": " + str(v) + '\r\n')


        f.write('POS:\r\n')

        for k,v in stats['pos'].items():
            f.write('\t' + k + ': ' + str(v) + '\r\n')

def finish_up(stats):
    output_stats('./stats', stats)


def do_loop(file_list, process_num, output_file):
    prev_stats = {}
    prev_stats['count'] = 0
    prev_stats['types'] = {}
    prev_stats['pos'] = {}
    prev_stats['sentences'] = 0
    prev_stats['stories'] = 0

    stories_processed = 0


    with(open(output_file, 'w')) as o:
        # go through each file and process the information we care about
        # including which words could be removed
        start_time = time.time()
        for f in file_list:
            # deserialize the story
            story = dill.load(open(f, 'rb'))

            prev_stats['stories'] += 1
            for line_num, sent in enumerate(story.dparsed_sentences):
                handle_sentence(story, f, line_num, o, prev_stats)

            stories_processed += 1
            if stories_processed % 500 == 0:
                end_time = time.time()
                full_time = end_time - start_time
                print("Proc " + str(process_num) + ": " +
                    str(stories_processed) + " Time: "
                    + str(datetime.timedelta(seconds=full_time)))


        finish_up(prev_stats)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Determine the set of \
                                        potentially removable modifiers for the\
                                        sentences in the provided input \
                                        directory')
    argparser.add_argument('input_directory',
                           help='The path to the input files.')
    argparser.add_argument('procs',
                           help='The number of processors to use.')
    argparser.add_argument('output_directory',help='The output path.')
    args = argparser.parse_args()


    proc_count = int(args.procs)

    # Get the number of lines in a file
    file_list = [args.input_directory + '/' +
                 f for f in os.listdir(args.input_directory)]
    total = len(file_list)

    print('Files to process: ' + str(total))

    interval = int(int(total) / int(proc_count))
    print(str(proc_count) + " processes, each processing " + str(interval) +
          'records')

    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)
    for i in range(0, proc_count):

        range_start = (i * interval)

        if i == (proc_count - 1):
            range_end = total - 1
        else:
            range_end = ((i + 1) * interval) - 1

        print('[' + str(range_start) + ':' + str(range_end) + ']')
        # make a process for each grouping
        p = Process(target=do_loop,
                    args=(file_list[range_start:range_end], i,
                          args.output_directory + '/sentences_'+
                          str(i) + '.csv'))
        p.start()
