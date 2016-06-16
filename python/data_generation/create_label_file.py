import sys
sys.path.insert(0, '../')

import os
import csv
from util import Util


from  astrolab_image_processor import AstrolabImageProcessor

#Make sure this script use the processed images.
included_cols = [8]
images_source = '/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/SDDS/images/'
images = Util.sort_nicely(os.listdir(images_source))

with open('/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/SDDS/zoo2MainSpecz.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    with open('/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/SDDS/labels.csv', 'w') as labelfile:
        spamwriter = csv.writer(labelfile)

        for row in spamreader:

            if spamreader.line_num -1 >= 5500:
                break
            if spamreader.line_num >= 1 :
                content = list(row[i] for i in included_cols)
                if content:
                    if str(spamreader.line_num) + '.jpg' in images:
                        print(content[0])
                        if content[0].startswith('E'):
                            spamwriter.writerow([spamreader.line_num - 1, 0])
                        else:
                            spamwriter.writerow([spamreader.line_num - 1, 1])
