from nltk import word_tokenize
import os.path
import os
import time
import datetime
import dill
import argparse
from multiprocessing import Process
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

EXCLUDED_POS = ['NNP', 'NN', 'VBP', 'VBD', '.', 'CC', 'DT', 'NNS', 'IN', 'PRP',
                'TO', 'VB', 'PRP$', 'VBG', ',', 'VBN', 'VBZ', 'WP']
WORDS = ['not', '\'nt']
REMOVABLE_TYPES = ['advmod', # includes quantmod
                   'amod',
                   # 'appos', too few
                   'compound',
                   # 'nmod:npmod', too few
                   # 'nummod', # Same ate [3] sheep [number and num
                   # usually needed
                   #'quantmod', # included in 'advmod'
                   #'nmod:poss', # Bill's clothes ['poss'] - lots of u
                   # ungrammatical results
                   # 'det:predet', # All the boys are here ['predet'] too few
                   # 'case', # prepositions - too many and most were needed
                   # TODO DSF - mark the whole prepositional phrase maybe?
                   # but lots of mistakes
                   'nmod:tmod', # aren't too many, but interesting, also aren't
                   # many types (lots of 'one day') - i think we could learn
                   # here
                   #'vmod' #     didn't exist, under xcomp, but too general
                   ]
DELIMITER = '|'
TOKENS_WITHOUT_SPACE = ["'", ".", "!", ":", ",", "n't"]


def filter_out_mods(mods):
    for m in mods:
        mod = m[2]
        pos = mod[1]
        modtype = m[1]
        word = mod[0]

        if modtype == 'nmod:poss' and pos == 'PRP$':
            continue

        if word.lower() == 'when':
            continue

        yield m

def is_node_relevant(n):
    if n['rel'] not in REMOVABLE_TYPES:
        return False
    if n['word'].lower() == 'when':
        return False

    return True

def plot_frequencies(data, ax, legend_label):
    # now sort tuples by count
    data = sorted(data, key=lambda tup: tup[1], reverse=True)

    # tuples (pair,count)
    x = [t[0] for t in data]
    y = [t[1] for t in data]

    x_pos = np.arange(len(x))

    ax.plot(x_pos, y, label=legend_label)
    ax.set_yscale('log')
    plt.xlim([0, len(x)])
    plt.ylim([.5, y[0]])


def get_statistics_for_sentence(prev_stats, interesting_records):
    prev_stats['count'] += len(interesting_records)
    prev_stats['sentences'] += 1

    for r in interesting_records:
        # stanford type
        r_t = r[1]
        r_pos = r[2][1]

        ts = prev_stats['types']

        if r_t not in ts:
            ts[r_t] = {}
            ts[r_t]['count'] = 0
            ts[r_t]['pos'] = {}

        ts[r_t]['count'] += 1

        if r_pos not in ts[r_t]['pos']:
            ts[r_t]['pos'][r_pos] = 0

        ts[r_t]['pos'][r_pos] += 1

        # add the head, modifier, and modifier-head counts
        head = r[0][0].lower()
        mod = r[2][0].lower()

        m = prev_stats['modifiers']
        if mod not in m:
            m[mod] = {}

        if head not in m[mod]:
            m[mod][head] = 0

        m[mod][head] += 1

        h = prev_stats['heads']

        if head not in h:
            h[head] = 0

        h[head] += 1


def remove_dependent_nodes(nodes, i, remove_list):
    last_node_ind = -1
    total_skipped = 0

    for j, n in nodes.items():
        next_index = j - last_node_ind

        if next_index is not 1:
            total_skipped += (next_index - 1)
        last_node_ind = j

        if j == 0:
            continue
        if int(n['head']) == i:
            remove_list.append(j - total_skipped)
            remove_dependent_nodes(nodes, j, remove_list)


def write_line(story, nodes, trigger_node, words_to_remove, output_file,
               story_filename, sentence_num):
    words_to_remove = sorted(set(words_to_remove))

    output_file.write(story.source + DELIMITER +
                            story.storyid + DELIMITER +
                            story_filename + DELIMITER + story.full_story +
                            DELIMITER + story.original_title +
                            DELIMITER + str(sentence_num))

    # Go though each word of the sentence and see if it should be
    # removed.
    '''
    compiled_sentence = ''
    for i, n in nodes.items():
        print(n)
        if i == 0:
            continue

        if i != 1 and n['word'] not in TOKENS_WITHOUT_SPACE:
            compiled_sentence += ' '

        # The word should be removed
        if i in words_to_remove:
            if (i - 1) not in words_to_remove:
                compiled_sentence += '['

        compiled_sentence += n['word']

        # Do we need to close the parenth?
        if i in words_to_remove:
            if i + 1 not in words_to_remove:
                compiled_sentence += ']'

    output_file.write(DELIMITER + compiled_sentence)
    '''
    # Output the full sentencei
    full_sentence = story.original_sentences[sentence_num] \
                      .replace('\r','').replace('\n', '')
    output_file.write(DELIMITER + full_sentence)


    # Output the interesting words
    output_file.write(DELIMITER + trigger_node['rel'])
    s = ''

    for i in full_sentence.split()\
                        [words_to_remove[0] - 1: words_to_remove[-1]]:
        s += i + ' '

    output_file.write(DELIMITER + s)
    output_file.write(DELIMITER + nodes[int(trigger_node['head'])]['word'])

    # words to remove
    output_file.write(DELIMITER + str(words_to_remove[0]))
    output_file.write(DELIMITER + str(words_to_remove[-1]))

    # head word index
    output_file.write(DELIMITER + str(trigger_node['head']))
    output_file.write('\r\n')

def handle_sentence(story, story_filename, sentence_num, output_file, stats):
    """Handle the given sentence and output the results to output_file."""
    # parse the sentence
    # now it is just a matter of iterating over our items and removing them
    # one by one
    # first, get just the set we care about
    count = 0

    # get the first dependency tree
    dsent = story.dparsed_sentences[sentence_num]
    sent = story.original_sentences[sentence_num]

    # get the triples of the dep graph
    p = [list(parse.triples()) for parse in dsent]

    # Triple format
    # [(head, POS), dep-type, (mod, POS)]
    # [(('walk', 'VB'), 'advmod', ('now', 'RB'))]
    # this is always a list, but we always just want the first element
    dsent = dsent[0]

    interesting_records = list(x for x in p[0] if x[1] in REMOVABLE_TYPES)
    # apply some filtering of the records
    interesting_records = list(filter_out_mods(interesting_records))

    count = len(interesting_records)
    # Get statistics based on the sentence
    results = get_statistics_for_sentence(stats, interesting_records)

    # if the count is greater than 0, we have something to remove, so let's go!
    if count == 0:
        return results

    # INTERESTING SENTENCE FOUND - let's write it out to the file
    words_with_apos = 0
    words_seen = -1

    for i, n in dsent.nodes.items():
        words_seen += 1
        if i == 0:
            continue

        if "'" in n['word']:
            words_with_apos += 1

        words_to_remove = []
        # check to see if the node has any of the phenomena we care about
        if is_node_relevant(n):
            # This is the occurrence we should consider,
            # so do the work
            words_to_remove.append(words_seen - words_with_apos)

            # also remove any dependent nodes
            remove_dependent_nodes(dsent.nodes, i, words_to_remove)

            # if n['word'] == 'hair':
            #     print(sent)
            #     print(words_to_remove)
            #
            write_line(story, dsent.nodes, n, words_to_remove, output_file,
                       story_filename, sentence_num)


def output_stats(output_path, stats):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    with open(output_path + 'general', 'w') as f:
        f.write('Phenomena per sentence: ' +
                str(stats['count'] / stats['sentences']) + '\r\n')

        f.write('Phenomena per story: ' +
                str(stats['count'] / stats['stories']) + '\r\n')


        f.write('Dependency Types:\r\n')

        for k,v in stats['types'].items():
            f.write('\t'  + str(k) + ": " + str(v['count']) + '\r\n')

        f.write('POS:\r\n')

        for k,v in stats['types'].items():
            f.write('\t' + k + ':\r\n')

            for k,v in v['pos'].items():
                f.write('\t\t' + k + ': ' + str(v) + '\r\n')

    fig, ax = plt.subplots()

    with open(output_path + 'heads_and_modifiers.csv', 'w') as f:
        # try to bring everything together and sort it

        # header
        f.write('modifier' + DELIMITER + 'head' + DELIMITER + 'count' +
                DELIMITER + 'total mod count' + DELIMITER +
                'perc with this head\r\n')
        all_items = []

        for k,v in stats['modifiers'].items():
            total = sum(v.values())
            for k2, v2 in v.items():
                # figure out the total for this modifier and the perc
                f.write(k + DELIMITER + k2 + DELIMITER +
                        str(v2) + DELIMITER + str(total) +
                        DELIMITER + str(v2/total)+'\r\n')
                all_items.append((k + k2, v2))

        start_time = time.time()

        plot_frequencies(all_items, ax, 'Modifier/Head')

        end_time = time.time()
        full_time = end_time - start_time
        print('Time at the end: ' + str(datetime.timedelta(seconds=full_time)))



    with open(output_path + 'heads.csv', 'w') as f:
        all_heads = []
        # header
        f.write('head' + DELIMITER + 'count\r\n')

        for k,v in stats['heads'].items():
            f.write(k + DELIMITER + str(v) + '\r\n')
            all_heads.append((k, v))

        plot_frequencies(all_heads, ax, 'Head')

    with open(output_path + 'modifiers.csv', 'w') as f:
        all_mods = []
        # header
        f.write('modifier' + DELIMITER + 'count\r\n')

        for k,v in stats['modifiers'].items():
            s = sum(v.values())
            f.write(k + DELIMITER + str(s) + '\r\n')
            all_mods.append((k, s))

        plot_frequencies(all_mods, ax, 'Modifier')


    plt.legend()
    plt.savefig(output_path + '/freq_dist.png')


def finish_up(output_dir, stats):
    output_stats(output_dir + '/stats/', stats)


def do_loop(file_list, process_num, output_dir):
    prev_stats = {}
    prev_stats['count'] = 0 # total number of phenomena
    prev_stats['types'] = {} # types of phenomena
    prev_stats['sentences'] = 0 # sentence count
    prev_stats['stories'] = 0 # story count
    prev_stats['modifiers'] = {} # modifiers, their count, and their heads
                                 # {'modifier': {'head': count}}
    prev_stats['heads'] = {} # {'head': count}
    stories_processed = 0

    output_file = output_dir + '/sentences_' + str(i) + '.csv'
    with(open(output_file, 'w')) as o:
        # Write a header
        o.write('Source' + DELIMITER +
                          'StoryID' + DELIMITER +
                          'File' + DELIMITER +
                  'Full Story' + DELIMITER +
                          'Title' + DELIMITER +
                          'Sentence Number' + DELIMITER +
                          'Original Sentence' + DELIMITER +
                          'Modifier Type' + DELIMITER +
                          'Modifier' + DELIMITER +
                          'Head' + DELIMITER +
                          'Removed Words Start Index' + DELIMITER +
                          'Removed Words End Index' + DELIMITER +
                          'Head Word Index'+
                          '\r\n')

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


        finish_up(output_dir, prev_stats)


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
          ' records')

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
                          args.output_directory))
        p.start()
