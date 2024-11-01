"""
    test stat printing
"""
from unittest import TestCase

import pandas as pd

from paraphrase.utility.PC_utility import print_pc_stats
from paraphrase.lead_pc import get_pc_first_batch_label_dict, load_pc_second_batch


class Test(TestCase):
    def test_print_stats(self):
        # pilot_df, _ = get_pilot_label_dict()
        # print("Pilot stats:")
        # print_pc_stats(pilot_df)

        # Paraphrase candidate STATS
        #   load paraphrase candidates annotations
        first_pc_anns_df, _ = get_pc_first_batch_label_dict()
        second_pc_anns_df, _ = load_pc_second_batch()
        all_pc = pd.concat([first_pc_anns_df, second_pc_anns_df])
        #   print stats
        print_pc_stats(all_pc)
