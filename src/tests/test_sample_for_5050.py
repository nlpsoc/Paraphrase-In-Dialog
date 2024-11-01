"""
    You don't want this

    legacy testing file
"""
from unittest import TestCase

from paraphrase.utility.PC_utility import get_sampled_5050_pc_ids, get_pilot_pc_labels
from sample_for_5050 import get_pc_paraphrase_statistics, \
    get_pc_non_paraphrase_statistics


class Test(TestCase):
    def test_updated_labels_pc_statistics(self):
        df_sampled_pc = get_sampled_5050_pc_ids()
        result_dict = get_pilot_pc_labels()

        get_pc_paraphrase_statistics(result_dict, df_sampled_pc)
        get_pc_non_paraphrase_statistics(result_dict, df_sampled_pc)
