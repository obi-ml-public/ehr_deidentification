# Parse the I2B2 XML data and store it in a JSONL format
import json
import tarfile
from argparse import ArgumentParser

from parsing import PHIParser, TARMembers
from tags import TagMap


def main(tar_file):
    # Read the xml files stored in the TAR file
    # Store the TAR object and it's members (the XML files)
    # Now that we have the TAR object and the names of the
    # XML files stored in this TAR file, we can read and parse the
    # data directly from the tar file - no need to uncompress the file.
    tar_obj = tarfile.open(tar_file, 'r')
    members = TARMembers.get_members(tar_obj=tar_obj, suffix='.xml')
    tar_info = [{'tar_obj': tar_obj, 'members': members}]
    # Parse all the XML files present in the TAR file
    # Convert the I2B2 tags to the BWH tags and store the
    # parsed data in the JSONL format. This object has a parse
    # function that can be called - which will go through each XML file
    # present in the mapping between the TAR object and it's members (XML files)
    # in the dictionary (tar_info). It goes through each xml file in the TAR file
    # and extracts the text, spans and metadata ans stores it in a dictionary
    # and yields an iterable that goes over the list of dictionaries
    phi_parser = PHIParser(tar_info)
    # Get the mapping between i2b2 tagset and the mgb proposed tagset
    # The mapping is a dictionary where the keys it the i2b2 tag and
    # the value is the mgb tag
    proposed_tags = TagMap().get_proposed_tags()
    # Use the phi_parser object
    # This yields an iterable that iterates through each XML file
    # The iterator parses the XML file and yields a dictionary
    # that contains the note text, spans (which are the phi spans) and some
    # patient metadata (patient number and record).
    # Each dictionary is then written in the jsonl format to the output file
    # that is passed as a CLI argument. Before writing we make sure to convert
    # the i2b2 tag to the mgb tag.
    for phi_record in phi_parser.parse():
        new_spans = list()
        for span in phi_record['spans']:
            if proposed_tags.get(span['TYPE'], False):
                span['label'] = proposed_tags.get(span['TYPE'])
                new_spans.append(span)
        phi_record['spans'] = new_spans
        yield phi_record


if __name__ == "__main__":
    # The following code sets up the arguments to be passed via CLI or via a JSON file
    cli_parser = ArgumentParser(description='configuration arguments provided at run time from the CLI')
    cli_parser.add_argument('--tar_file', type=str, default=None, help='the TAR file that contains the i2b2 data')
    cli_parser.add_argument('--output_file', type=str, help='the location where you want to store the jsonl i2b2 data')

    args = cli_parser.parse_args()
    # Get the tar file in json format
    i2b2_dataset = main(args.tar_file)
    # Store the jsonl data
    with open(args.output_file, 'w') as file:
        for i2b2_data in i2b2_dataset:
            file.write(json.dumps(i2b2_data) + '\n')
