from typing import Type, Union
import warnings
import datetime
import unittest
import logging
import os
import pandas as pd
import numpy as np


from mri_surv.statistics.other_dementia import reversion_stats

log_file = os.path.join(os.path.abspath("./logs"),
                                        "cohort.log")
with open(log_file, 'w') as fi:
    fi.write(f'Logs {datetime.datetime.now()}')
hdlr = logging.FileHandler(log_file)
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)
logger.setLevel('INFO')

class TestReversions(unittest.TestCase):
    def test_reversions(self):
        reversion_stats.main()
        df_reverted = pd.read_csv('/metadata/data_processed/reverted_rids.csv', dtype={'RID': str})
        self.assertTrue(False)