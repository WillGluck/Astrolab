import os
import sys
import csv
sys.path.insert(0, '../')

from  astrolab_image_processor import AstrolabImageProcessor

import re

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    return sorted(l, key=alphanum_key)

#Make sure this script use the processed images.
included_cols = [8]
images_source = '/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/SDDS/images/'
images = sort_nicely(os.listdir(images_source))

with open('/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/SDDS/zoo2MainSpecz.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    with open('/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/SDDS/labels.csv', 'w') as labelfile:
        spamwriter = csv.writer(labelfile)

        for row in spamreader:

            if spamreader.line_num -1 >= 5001:
                break
            if spamreader.line_num - 1 >= 2 :
                content = list(row[i] for i in included_cols)
                if content:
                    if str(spamreader.line_num - 1) + '.jpg' in images:
                        if content[0].startswith('E'):
                            spamwriter.writerow([spamreader.line_num - 1, 0])
                        elif content[0].startswith('S'):
                            spamwriter.writerow([spamreader.line_num - 1, 1])
                        else:
                            spamwriter.writerow([spamreader.line_num - 1, 2])
