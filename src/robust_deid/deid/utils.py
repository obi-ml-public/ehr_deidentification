def remove():
    return {'PATIENT': '',
            'STAFF': '',
            'AGE': '',
            'DATE': '',
            'PHONE': '',
            'MRN': '',
            'ID': '',
            'EMAIL': '',
            'PATORG': '',
            'LOC': '',
            'HOSP': '',
            'OTHERPHI': ''}


def replace_tag_type():
    return {'PATIENT': 'PATIENT',
            'STAFF': 'STAFF',
            'AGE': 'AGE',
            'DATE': 'DATE',
            'PHONE': 'PHONE',
            'MRN': 'MRN',
            'ID': 'ID',
            'EMAIL': 'EMAIL',
            'PATORG': 'PATORG',
            'LOC': 'LOCATION',
            'HOSP': 'HOSPITAL',
            'OTHERPHI': 'OTHERPHI'}


def replace_informative():
    return {'PATIENT': '<<PATIENT:{}>>',
            'STAFF': '<<STAFF:{}>>',
            'AGE': '<<AGE:{}>>',
            'DATE': '<<DATE:{}>>',
            'PHONE': '<<PHONE:{}>>',
            'MRN': '<<MRN:{}>>',
            'ID': '<<ID:{}>>',
            'EMAIL': '<<EMAIL:{}>>',
            'PATORG': '<<PATORG:{}>>',
            'LOC': '<<LOCATION:{}>>',
            'HOSP': '<<HOSPITAL:{}>>',
            'OTHERPHI': '<<OTHERPHI:{}>>'}
