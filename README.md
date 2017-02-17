# determine-necessary-modifiers
# Author: Devon Fritz
This project allows one to create and analyze a dataset of linguistic modifiers
in terms of their cruciality within the context of a story.

This document contains the instructions to recreate the experiment, both with
the two datasets used here as well as with arbitrary datasets. Please report
any bugs to devon.s.fritz@gmail.com

Description of files:
- classification_record.py - This module contains a class that encapsulates
    a modified record, annotated or not. It can be then used by the classifier.
- classify.py - A script that both builds up classifier from annotated
    training data as well as classifies an unseen part of the data as test
    data, comparing the clasification results with the gold standard.
- dependency_parse_stories.py - This script dependency parses provided texts
    and serializes objects including the parses to disk so that they can be
    used by other scripts without having to parse the data each time, an
    expensive process.
- detect_removable_constituents.py - Uses the parsed objects above to determine
    which modifiers should be considered for removal. It builds a csv
    of candidates and includes supplementary information, such as POS and
    position in the sentence.
- slice_results.py - Takes the csv output by detect_removable_constituents.py
    and slices it down to an inputted size according to different criteria.
- helpers.py - A module with a helper class, Story.

Usage:

1) Parse all of the data and store it to disk.
   - Run dependency_parse_stories.py on the dataset of your choice: ROCStories or
     ARGMicroTexts (not provided, but free online).
     Example: python dependency_parse_stories.py [path-to-ROC-data-file] [output-directory] [Type [ARG|ROC]]

2) Now that we have objects that have abstracted away the details of the corpus source,
we can work on figuring out which modifiers are removable.
    - Run detect_removable_constituents.py
      Example: python detect_removable_constituents.py [path-to-cached-parse-objects] [number of processors - just do 1] [output-directory]

3) Now, filter down the list of results as required. This will grab different slices of the data.
    - Run slice_results.py
      Example: python slice_results.py [path-to-spreadsheet from the last step] [path-to-mod-freq-file generated in the stats from the last step] [output-directory] [goal-record-count] [min-instance-per-modifier-count]

4) Now that the data is the size we went, it can be annotated manually in a column called "cruciality".

5) Build a classifier with annotated data. Using the annotated data you created, or the csv provided here, build a classifier and classify the data.
    - Run classify.py
      Example: python classify.py [data-file-to-classify]
