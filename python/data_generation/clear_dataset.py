import sys
sys.path.insert(0, '../')

import shutil
import os
import csv
from util import Util

included_cols = [8]
images_source = '/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/DATASET/test/images/raw/'
images = Util.sort_nicely(os.listdir(images_source))

with open('/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/SDDS/zoo2MainSpecz.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if spamreader.line_num >= 1006 and spamreader.line_num <= 2006:
            content = list(row[i] for i in included_cols)
            if content:
                if str(spamreader.line_num - 1) + '.jpg' in images:
                    if not (content[0].startswith('E') or content[0].startswith('S')):
                        shutil.move(
                            '/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/DATASET/test/images/raw/' + str(spamreader.line_num) + '.jpg',
                            '/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/DATASET/test/images/trash/'
                        )
