from unittest import TestCase

from paraphrase import interview_data
from paraphrase.annotation_data import get_individual_human_anns
from paraphrase.utility.annotation_df import get_unique_q_ids


class Test(TestCase):
    def test_get_all_crowd_annotations(self):
        train_anns = get_individual_human_anns(question_ids="TRAIN")
        train_anns.to_csv("train.tsv", sep="\t", index=False)
        dev_anns = get_individual_human_anns(question_ids="DEV")
        dev_anns.to_csv("dev.tsv", sep="\t", index=False)
        test_anns = get_individual_human_anns(question_ids="TEST")
        test_anns.to_csv("test.tsv", sep="\t", index=False)


