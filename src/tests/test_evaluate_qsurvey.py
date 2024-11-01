"""
    You don't need this

    legacy testing file
"""
import math
from unittest import TestCase
from paraphrase.utility.evaluate_qsurvey import FIRST_100_IDs, get_freq, calc_str_agreements_cat, \
    calc_classification_agreement, \
    get_highlight_choices, calc_HighlightScores, \
    calc_overlaps
from paraphrase.annotation_data import get_txt_highlight
from paraphrase.utility.annotation_df import merge_transform, transform_hl_report, AllAnnotationCategories
import paraphrase.utility.evaluate_qsurvey as evaluate_qsurvey
from pandas.util.testing import assert_frame_equal
import pandas as pd
from paraphrase.interview_data import MediaSumProcessor


class Test(TestCase):

    def setUp(self) -> None:
        self.first_100_ids = FIRST_100_IDs
        self.tsv_path = "paraphrase/utility/fixtures/annotations/Dummy_Qualtrics-Annotations.tsv"
        self.tsv_test = "fixtures/annotations/Dummy_Transformed-Results.tsv"
        # self.tsv_60 = "fixtures/annotations/22-01-16_Sample_41-100_January+16,+2023_04.12_numeric.tsv"
        # self.tsv_11 = "fixtures/annotations/22-01-12_Sample_21-40_January+18,+2023_03.00.tsv"
        # self.tsv_11_chunk = "fixtures/annotations/22-01-12_Chunked-Sample_1-20_January+18,+2023_04.08.tsv"
        self.tsv_50 = "fixtures/annotations/22-01-30_Sample_152-202_max-5_February+6,+2023_07.39.tsv"

        self.sample_21 = self.first_100_ids[20]
        self.sample_41 = self.first_100_ids[40]

    def test_transform_report(self):
        pd_report = pd.read_csv(self.tsv_path, sep="\t", encoding="utf-16")
        pd_transformed = transform_hl_report(pd_report, False)  # , start_nbr=21
        pd_test = pd.read_csv(self.tsv_test, sep="\t", encoding="utf-16")
        # self.assertTrue(pd_transformed.equals(pd_test))
        assert_frame_equal(pd_transformed, pd_test)

    def test_calc_agreement(self):
        pd_report = pd.read_csv(self.tsv_path, sep="\t", encoding="utf-16")
        pd_transformed = transform_hl_report(pd_report, False)
        result_dict = calc_classification_agreement(pd_transformed, include_history=False)
        self.assertGreater(result_dict[AllAnnotationCategories.Elaboration_Question], 0)

    def test_get_frequency(self):
        pd_transformed = merge_transform([self.tsv_path], False)
        result_dict = get_freq(pd_transformed, False)

        self.assertEqual(result_dict[AllAnnotationCategories.Referred_to], 1)
        self.assertEqual(result_dict[AllAnnotationCategories.Rephrase], 0)
        self.assertEqual(result_dict[AllAnnotationCategories.Interpretation], 0)
        self.assertEqual(result_dict[AllAnnotationCategories.Elaboration_Information], 0)
        self.assertEqual(result_dict[AllAnnotationCategories.Elaboration_Question], 1)

    def test_overlap(self):
        pd_transformed = merge_transform([self.tsv_path], False)
        result_dict = calc_overlaps(pd_transformed)
        self.assertEqual(result_dict[AllAnnotationCategories.Referred_to], 0)
        self.assertTrue(math.isnan(result_dict[AllAnnotationCategories.Rephrase]))
        self.assertTrue(math.isnan(result_dict[AllAnnotationCategories.Interpretation]))
        self.assertTrue(math.isnan(result_dict[AllAnnotationCategories.Elaboration_Information]))
        self.assertEqual(result_dict[AllAnnotationCategories.Elaboration_Question], 1)

    def test_50_sample(self):
        pd_report = merge_transform([self.tsv_50], False)
        result_dict = calc_overlaps(pd_report)
        print(result_dict)

        result_dict = get_freq(pd_report, False)
        print(result_dict)
        result_dict = calc_classification_agreement(pd_report)
        print(result_dict)

        result_dict = calc_HighlightScores(pd_report)
        print(result_dict)

    def test_get_examples(self):
        # pd_report = merge_transform([self.tsv_11], [21])
        # pd_report = merge_transform([self.tsv_60], [41])
        pd_report = merge_transform([self.tsv_50], False)
        interview = MediaSumProcessor()
        # evaluate_qsurvey.print_disagreement_examples(AnnotationCategories.Rephrase, pd_report, interview)
        # evaluate_qsurvey.print_disagreement_examples(AnnotationCategories.Elaboration_Question, pd_report, interview)
        evaluate_qsurvey.print_agreement_examples(AllAnnotationCategories.Interpretation, pd_report, interview)

        # evaluate_qsurvey.print_example_cat([AnnotationCategories.Referred_to, AnnotationCategories.Rephrase,
        #                                     AnnotationCategories.Interpretation,
        #                                     AnnotationCategories.Elaboration_Information,
        #                                     AnnotationCategories.Elaboration_Question], pd_report, interview)
        # evaluate_qsurvey.print_example_cat([AnnotationCategories.Elaboration_Question], pd_report, interview)

        # evaluate_qsurvey.print_agreement_examples(AnnotationCategories.Interpretation, pd_report, interview)
        # evaluate_qsurvey.print_disagreement_examples(AnnotationCategories.Rephrase, pd_report, interview)
        #
        # pd_report = merge_transform([self.tsv_11_chunk], [1])
        # interview = MediaSumProcessor()
        # evaluate_qsurvey.print_agreement_examples(AnnotationCategories.Interpretation, pd_report, interview,
        #                                           chunked=True)
        # evaluate_qsurvey.print_disagreement_examples(AnnotationCategories.Rephrase, pd_report, interview,
        #                                              chunked=True)

    def test_get_highligth_choices(self):
        q_id = "CNN-254960-3"
        pd_report = pd.read_csv(self.tsv_path, sep="\t", encoding="utf-16")
        pd_report = transform_hl_report(pd_report, False)
        category = AllAnnotationCategories.Elaboration_Question
        choices = get_highlight_choices(q_id, pd_report, category, "Annotator1")
        self.assertEqual([10, 11, 12, 13, 14, 15, 16, 17], choices)

    def test_get_highlights(self):
        interview = MediaSumProcessor()
        q_id = "CNN-254960-3"
        pd_report = pd.read_csv(self.tsv_path, sep="\t", encoding="utf-16")
        pd_report = transform_hl_report(pd_report, False)
        category = AllAnnotationCategories.Elaboration_Question
        choices = get_highlight_choices(q_id, pd_report, category, "Annotator1")
        highlighted = get_txt_highlight(q_id, [choices, choices], interview)[0]
        self.assertEqual("What do you think the punishment will be?", highlighted)
