class TagMap(object):
    """This class is used to map the I2B2 tag types to our proposed tag types"""

    def __init__(self):
        """Initialize a mapping between the proposed tag types and the I2B2 tag types"""
        self._tag_map = {'PATIENT': ['PATIENT'], 'STAFF': ['DOCTOR', 'USERNAME'], 'AGE': ['AGE'], 'DATE': ['DATE'],
                         'PHONE': ['PHONE', 'FAX'], 'EMAIL': ['EMAIL'],
                         'ID': ['SSN', 'HEALTHPLAN', 'ACCOUNT',
                                'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM', 'MEDICALRECORD'],
                         'HOSP': ['HOSPITAL', 'DEPARTMENT', 'ROOM'], 'PATORG': ['ORGANIZATION'],
                         'LOC': ['STREET', 'CITY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER'],
                         'OTHERPHI': ['OTHERPHI', 'URL']}

    def get_proposed_tags(self):
        """
        Return a mapping between the I2B2 type and the corresponding proposed tag types
        Returns:
                map_tag (dict): A mapping where the key is the I2B2 type and the value is the proposed tag type
        """
        map_tag = dict()
        for proposed_tag, i2b2_tags in self._tag_map.items():
            for i2b2_tag in i2b2_tags:
                map_tag[i2b2_tag] = proposed_tag

        return map_tag
