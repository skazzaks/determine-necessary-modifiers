"""Contains the classes we need for classification."""


class ModifierRecord():
    """Represents a record in the dataset that has modifier information"""
    def __init__(self, row_data, headers):
        self.data_map = {}
        self.COL_MODIFIER = 'modifier'
        self.COL_MODTYPE = 'modifier_type'
        self.COL_HEAD = 'head'
        self.store_values(row_data, headers)

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
                return v
        # If we didn't find a majority opinion, let's return 'None'
        return None

    def get_features_and_target(self):
        """Returns the features and the target for the current record"""
        features = {}
        features['modifier'] = self.data_map[self.COL_MODIFIER]
        features['modifier_type'] = self.data_map[self.COL_MODTYPE]
        features['head'] = self.data_map[self.COL_HEAD]

        target = self.data_map[self.COL_CRUCIALITY]

        return features, target
