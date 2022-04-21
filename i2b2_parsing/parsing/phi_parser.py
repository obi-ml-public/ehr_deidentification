from pathlib import Path
from tarfile import TarFile
from typing import List, Dict, Union

from bs4 import BeautifulSoup


class PHIParser(object):
    """
    This class is used to read the the XML files present
    in the TAR file and gather the text, spans and any metadata
    information present in the i2b2 note.
    """

    def __init__(self, tar_info: List[Dict[str, Union[TarFile, list]]]):
        """
        Initialize a dictionary that contains a mapping between
        the TAR object (TAR file) and its members (XML files)
        Args:
            tar_info (dict):mapping between the TAR object (TAR file) 
                            and its members (XML files)
        """
        self._tar_info = tar_info

    def parse(self):
        """
        Parse XML data using BeautifulSoup. Use the XML parser
        provided by BeautifulSoup to extract the relevant tags.
        Returns:
            (iterator): Yields a dictionary that contains the text of the note,
                        PHI span information and metadata of the note
        """
        for info in self._tar_info:
            tar_obj = info['tar_obj']
            for compressed_file in info['members']:
                file_obj = tar_obj.extractfile(compressed_file)
                patient, record = Path(compressed_file).stem.split('-')
                bs_data = BeautifulSoup(file_obj.read(), 'xml')
                yield {'text': bs_data.find('TEXT').text,
                       'spans': [tag.attrs for tags in bs_data.find_all('TAGS') for tag in tags.find_all()],
                       'meta': {'note_id': patient + '-' + record, 'patient': patient, 'record': record,
                                'institute': 'i2b2'}}
