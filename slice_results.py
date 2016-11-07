"""Slice Results
This file takes in the big list of results that we got from our initial filters
and further narrows the list down, now that we have some aggregate information
to filter on.
"""
import logging
import csv
import argparse
import math
from collections import OrderedDict

# The maximimum number of records to get per modifier
MAX_PER_MODIFIER = 15
MAX_PER_FILTER = 6000
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
log = logging.getLogger('slice')


def filter_get_specific_modifiers(freq_list_num, freq_list_mod, final_mods,
                                  count_to_get, min_instance_count):
    """Takes in a list of modifiers that we definitely want to include and
    finds them in our big list, adjusting the remaining count accordingly.
    """
    mod_list_to_include = ['good']  # TODO
    total_used = 0

    for m in mod_list_to_include:
        count = min(freq_list_mod[m], MAX_PER_MODIFIER)
        log.debug(m + " " + str(freq_list_mod[m]))
        # We can get all of them
        if count_to_get > count:
            final_mods[m] = count
            count_to_get -= count
            total_used += count
        else:
            final_mods[m] = count_to_get
            total_used += count_to_get
            count_to_get = 0

        if count_to_get == 0:
            break

    return total_used


def filter_get_slices(freq_list_num, freq_list_mod, final_mods,
                      count_to_get, min_instance_count):
    """Slices the data based on modifier count frequency and pulls modifiers
    according to weighted values
    Slicing PROPORTIONS
    [1, 1, 1] = 1/3 from the least frequent, median, and most frequent
    [1, 2, 3] = least freq = 1/6, middle = 1/3, most = 1/2
    """
    PROPORTIONS = [1, 1, 1, 1, 1]
    counts_per_group = get_counts_based_on_proportions(PROPORTIONS,
                                                       count_to_get)

    log.debug(counts_per_group)
    total_used = get_keys_to_use(freq_list_num, counts_per_group, final_mods,
                                 min_instance_count)

    return total_used


def get_frequency_list_from_file(the_file):
    """Takes in a file of the format word|frequency and returns
    two ordered dictionary: 1 sorted by frequency with frequency as the key
    and one with modifier as the key
    """
    freq_list_num = {}
    freq_list_mod = {}

    with open(the_file, 'r') as f:
        c = csv.reader(f, delimiter='|', quotechar='\x07')
        next(c)
        for r in c:
            key = int(r[1])
            if key not in freq_list_num.keys():
                freq_list_num[key] = []
            freq_list_num[key].append(r[0])

            freq_list_mod[r[0]] = int(r[1])

    return OrderedDict(sorted(freq_list_num.items())), freq_list_mod


def get_counts_based_on_proportions(proportions, goal_count):
    total = sum(proportions)
    section_counts = []
    remaining = goal_count

    for i, p in enumerate(proportions):
        if i == len(proportions):
            amount_for_this_rec = remaining
        else:
            perc = float(p) / float(total)
            amount_for_this_rec = int(goal_count * perc)
            section_counts.append(amount_for_this_rec)
        remaining -= amount_for_this_rec

    # We might have a little left over - just tack it on to the end - it is not
    # a big deal
    section_counts[-1] += remaining

    return section_counts


def process_slice(starting_key, keys_asc, freq_dict, count_to_get,
                  final_mod_list):
    """Starting at the starting_key, go through the data and pull out keys
    for the final set until our count is satisfied.
    """
    log.debug('Starting Index: ' + str(starting_key))
    total_used = 0
    starting_key = int(starting_key)

    for key in keys_asc[starting_key:]:
        mods = freq_dict[key]
        log.debug('Count with length' + str(key) + ': ' + str(len(mods)))
        for mod in mods:
            if mod in final_mod_list.keys():
                continue
            count = min(count_to_get, MAX_PER_MODIFIER, key)
            # log.debug('count: ' + str(count))
            final_mod_list[mod] = count
            count_to_get -= count
            total_used += count
            if count_to_get == 0:
                break

        if count_to_get == 0:
            break

    log.debug('Total Used: ' + str(total_used))
    return total_used


def get_keys_to_use(freq_dict, counts_per_slice, final_mod_list,
                    min_instance_count):
    """
    Gets the keys that we should use for the final list of records
    freq_dict = the dictionary of the form {freq_num: [list of mod]}
    counts_per_group = the number of records we should grab from each grouping
    """
    freq_keys_asc = sorted(freq_dict.keys())
    freq_keys_desc = sorted(freq_dict.keys(), reverse=True)
    slices = len(counts_per_slice)
    total_used = 0

    # Make sure we don't process any modifiers whose frequency is below
    # our threshold
    freq_keys_asc = list([x for x in freq_keys_asc
                         if x >= min_instance_count])

    # Start with the last section, which always takes the last element
    total_used += process_slice(0, freq_keys_desc, freq_dict,
                                counts_per_slice[-1], final_mod_list)

    # Now, do the beginning slice counting up
    total_used += process_slice(0, freq_keys_asc, freq_dict,
                                counts_per_slice[0], final_mod_list)

    # Finally, process the middle slices
    distance_between_slices = len(freq_keys_asc) / slices

    for i, s in enumerate(counts_per_slice[1: -1]):
        index = i + 1
        # The starting index is the slice number (plus one, since we are
        # skipping 0, times the distance between slices)
        starting_index = math.ceil(index * distance_between_slices)

        amount = process_slice(starting_index, freq_keys_asc, freq_dict,
                               counts_per_slice[index], final_mod_list)

        log.debug('Amount for index ' + str(index) + ': ' + str(amount))
        total_used += amount

    log.debug('Final Amount: ' + str(total_used))
    return total_used


def buffer_final_list(freq_list_num, final_mods, amount_to_add):
    log.debug(freq_list_num)
    log.debug(amount_to_add)

    for k, v in freq_list_num.items():
        for w in v:
            if w not in final_mods.keys():
                amount = min(k, amount_to_add)
                final_mods[w] = amount
                amount_to_add -= amount

                if amount_to_add == 0:
                    return


def get_list_of_modifiers_to_include(freq_list_num, freq_list_mod,
                                     goal_count,
                                     min_instance_count, filters):
    remaining_count = goal_count

    # The final list of modifiers that we will include
    # {'modifier':count_to_use} {'good':400}
    final_mods = {}

    # Call each function, appending to the final list of modifiers to include
    for f in filters:
        if remaining_count > 0:
            amount_to_get = min(remaining_count, MAX_PER_FILTER)

            amount_processed = f(freq_list_num, freq_list_mod, final_mods,
                                 amount_to_get, min_instance_count)

            remaining_count -= amount_processed

    # If we still have records left over, just buffer the list until we get
    # to our target count
    if remaining_count != 0:
        buffer_final_list(freq_list_num, final_mods, remaining_count)

    return final_mods


def get_total_line_count(the_file):
    with open(the_file, 'r') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def do_the_work(data_file, freq_file, output_dir, goal_count,
                min_instance_count, filters):
    """
    Reads in all of our data and filters out records based on our criteria
    """
    total_line_count = get_total_line_count(data_file)
    freq_list_num, freq_list_mod = get_frequency_list_from_file(freq_file)

    modifiers = get_list_of_modifiers_to_include(freq_list_num, freq_list_mod,
                                                 goal_count,
                                                 min_instance_count,
                                                 filters)

    # Just some log checks to make sure we are getting the right amounts
    log.debug(modifiers)
    log.debug(sum(modifiers.values()))
    log.debug(sum(v for k, v in modifiers.items() if v == 1))
    log.debug(sum(v for k, v in modifiers.items() if k == 'good'))

    # Sanity check
    if sum(modifiers.values()) != goal_count:
        print("ERROR! We didn't get enough records!")
        exit()

    # Now that we have the modifiers, get the list
    create_list_of_results(data_file, output_dir, modifiers)


def create_list_of_results(data_file, output_dir, final_mod_list):
    """Given the original data file, the directory to output too, and the
    list of modifiers we should pull out, grabs out the results.
    """
    COLUMN_MODIFIER = 8

    log.debug(final_mod_list)
    with open(data_file, 'r') as f:
        with open(output_dir + '/final_modifier_list.csv', 'w') as o:
            c = csv.reader(f, delimiter='|', quotechar='\x07')

            count = 0
            in_count = 0
            total = 0
            DELIMITER = '|'
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
                    'Preremove Sentence Part' + DELIMITER +
                    'To Remove Part' + DELIMITER +
                    'Postremove Sentence Part' + DELIMITER +
                    'Removed Words Start Index' + DELIMITER +
                    'Removed Words End Index' + DELIMITER +
                    'Head Word Index' +
                    '\r\n')

            for r in c:
                total += 1
                modifier = r[COLUMN_MODIFIER]

                if modifier == 'good':
                    count += 1
                if modifier in final_mod_list.keys():
                    in_count += 1
                    final_mod_list[modifier] -= 1
                    if final_mod_list[modifier] == 0:
                        final_mod_list.pop(modifier, None)

                    # log.debug('found modifier = ' + modifier)
                    o.write('|'.join(r) + '\r\n')

    log.debug(final_mod_list)
    log.debug(sum(final_mod_list.values()))
    log.debug('good count = ' + str(count))
    # log.debug('good used = ' + str(final_mod_list['good']))
    log.debug('in count = ' + str(in_count))
    log.debug('total = ' + str(total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slices our resulting data'
                                     'a more managable list.')
    parser.add_argument('input_file', help='The input csv file containing the'
                        'results.')
    parser.add_argument('input_mod_freq_file', help='The file containing the'
                        ' list of modifiers and their frequencies.')
    parser.add_argument('output_directory', help='The directory for the '
                        'output')
    parser.add_argument('goal_record_count', type=int,
                        help='The total amount of records'
                        ' that we should have at the end.')
    parser.add_argument('min_instance_count', type=int,
                        help='The minimum number of instances to consider\
                         for a modifier.')
    args = parser.parse_args()

    filters = [filter_get_specific_modifiers, filter_get_slices]

    do_the_work(args.input_file, args.input_mod_freq_file,
                args.output_directory, args.goal_record_count,
                args.min_instance_count, filters)
