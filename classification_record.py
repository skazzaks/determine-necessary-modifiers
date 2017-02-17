"""Contains the classes we need for classification."""
import dill
import logging

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
log = logging.getLogger('classification_record')


class ModifierRecord():
    """Represents a record in the dataset that has modifier information"""
    def __init__(self, row_data, headers):
        self.data_map = {}
        self.COL_MODIFIER = 'modifier'
        self.COL_MODTYPE = 'modifier_type'
        self.COL_HEAD = 'head'
        self.COL_MODIFIER_START_INDEX = 'removed_words_start_index'
        self.COL_HEAD_INDEX = 'head_word_index'
        self.COL_ORIGINAL_SENTENCE = 'original_sentence'
        self.COL_FILE_TO_DPARSED_STORY = 'file'
        self.COL_SENTENCE_NUM = 'sentence_number'
        self.store_values(row_data, headers)
        self.original_line = '|'.join(row_data)
        # WARNING! This relies on the parsed stories being in the location
        # in the csv file. You might have to do some work to get this feature
        # to work.
        self.story = dill.load(open(
                         self.data_map[
                            self.COL_FILE_TO_DPARSED_STORY], 'rb'))

    def store_values(self, row_data, headers):
        for i, header in enumerate(headers):
            self.data_map[header] = row_data[i]

    def __str__(self):
        return self.data_map


class AnnotatedModifierRecord(ModifierRecord):
    """Represents one record that has modifier information that has
    been annotated"""
    def __init__(self, row_data, headers):
        self.COL_TRUST = '_trust'
        self.COL_CRUCIALITY = 'cruciality'
        self.trust = []
        self.cruciality = []
        super().__init__(row_data, headers)
        self.add_annotation_data(row_data, headers)

    def __str__(self):
        norm = super().__str__()
        return str(norm) + '\r\n\r\n' + str(self.trust) + '\r\n\r\n'\
            + str(self.cruciality)

    def add_annotation_data(self, row_data, headers):
        """Adds a annotation data to the current modifier. This is mainly
        because the full spreadsheet has multiple records per result."""
        self.trust.append(float(self.get_value_for_header(
                              row_data, headers, self.COL_TRUST)))

        self.cruciality.append(self.get_value_for_header(
                               row_data, headers, self.COL_CRUCIALITY))

    def get_value_for_header(self, row_data, headers, header):
        """Finds the value corresponding to the header"""
        for i, h in enumerate(headers):
            if header == h:
                return row_data[i]

    def get_cruciality(self):
        """Returns the cruciality for the current record. This is one of three
        values: ungrammatical, crucial, or notcrucial. For reach record, there
        are three assessments.

        Right now, we take the most frequent value. In the case that none have
        a majority, we should probably reject the record.
        """
        result_map = {}
        for v in self.cruciality:
            if v not in result_map.keys():
                result_map[v] = 0

            result_map[v] += 1

        # Now, let's see if any record has 2 or 3
        for k, v in result_map.items():
            if v >= 2:
                return k
        # If we didn't find a majority opinion, let's return 'None'
        return None

    def get_depth_feature(self):
        """Gets the 'depth' of the modifier in the dependency tree."""
        sent_num = int(self.data_map[self.COL_SENTENCE_NUM])
        mod = self.data_map[self.COL_MODIFIER]
        dsent = self.story.dparsed_sentences[sent_num][0]
        key = None

        for k in dsent.nodes:
            if dsent.nodes[k]['word'] == mod:
                key = k
                break

        depth = self.determine_depth(key, dsent.nodes, 0)

        return depth

    def determine_depth(self, key_current_node, nodes, current_depth):
        """Recursive method that determines the depth of the current node
        in the tree"""
        head_key = nodes[key_current_node]['head']
        if head_key is None:
            return current_depth
        else:
            return self.determine_depth(head_key, nodes, current_depth + 1)

    def add_word_embedding_features(self, embedding_model, word, features,
                                    column):
        """Appends word embedding features using the passed in trained
        model

        embedding_model - the gensim.Word2Vec model
        word - The word for which we should get the features
        features - The current list of features
        """
        word = word.lower()

        try:
            dims = embedding_model[word]
        except KeyError:
            # If we encounter and OOV word, just make it a basic stop word
            dims = embedding_model['the']

        # For each dimension in our resulting vector, make a new feature
        for i, d in enumerate(dims):
            features[column + 'emb_word_'+str(i)] = d

    def get_features_and_target(self, embedding_model):
        """Returns the features and the target for the current record"""
        features = {}
        # features['modifier'] = self.data_map[self.COL_MODIFIER]
        features['modifier_type'] = self.data_map[self.COL_MODTYPE]
        # features['head'] = self.data_map[self.COL_HEAD]
        self.add_word_embedding_features(embedding_model,
                                         self.data_map[self.COL_MODIFIER],
                                         features,
                                         self.COL_MODIFIER)
        self.add_word_embedding_features(embedding_model,
                                         self.data_map[self.COL_HEAD],
                                         features,
                                         self.COL_HEAD)
        sent = self.data_map[self.COL_ORIGINAL_SENTENCE]
        word_count = len(sent.split())
        features['word_count'] = word_count
        # features['modifier_start_index'] = self.data_map[
        #                                   self.COL_MODIFIER_START_INDEX]
        # features['head_index'] = self.data_map[self.COL_HEAD_INDEX]
        # The depth of the modifier in the dependency parse tree - helps
        # a percentage point.
        # WARNING! Relies upon dparsed stories cached on the harddrive at the
        # location specified in the final csv file.
        features['mod_depth'] = self.get_depth_feature()

        log.info(features)
        target = self.data_map[self.COL_CRUCIALITY]

        return features, target
