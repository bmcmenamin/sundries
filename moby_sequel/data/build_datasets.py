"""Divide the clean data into train/test files"""

import os
from glob import iglob

CLEAN_DIR = os.path.join(os.path.curdir, 'clean')
OUT_DIR = os.path.curdir

train_data = open(os.path.join(OUT_DIR, 'train_data.txt'), 'wt')
test_data = open(os.path.join(OUT_DIR, 'test_data.txt'), 'wt')

for fname in iglob(os.path.join(CLEAN_DIR, '*.txt')):
    with open(fname, 'rt') as file:
        for line_num, line_text in enumerate(file):
            if (line_num % 10) == 0:
                test_data.write(line_text)
            else:
                train_data.write(line_text)

    train_data.flush()
    test_data.flush()

train_data.close()
test_data.close()