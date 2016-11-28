"""Contains a class that represents one result record that is to be
classified"""
import random


class ModifierRecord():

    def __init__(self, record_data_map):
        self.record_data = record_data_map
        self.ORIG_SENT_COL = 6
        self.MOD_TYPE_COL = 7
        self.MODIFIER_COL = 8
        self.HEAD_COL = 9
        self.FULL_MOD_COL = 11
        self.FULL_STORY_COL = 3
        self.SENT_NUM_COL = 5
        self.TARGET_COL = 12

    def get_features_and_target(self):
        """Extracts the features from this data_record

        returns - A dictionary with features and values
        """
        feature_dict = {}

        feature_dict['MOD_TYPE'] = self.record_data[self.MOD_TYPE_COL]
        feature_dict['MOD'] = self.record_data[self.MODIFIER_COL]
        feature_dict['HEAD'] = self.record_data[self.HEAD_COL]
        feature_dict['SENT_NUM'] = self.record_data[self.SENT_NUM_COL]

        # TODO DSF - this has to be the real value
        target = random.randrange(3)

        return feature_dict, target
