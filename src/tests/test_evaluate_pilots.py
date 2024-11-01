"""
    You don't want to look at this file

    legacy testing file, several of these tests won't run anymore, keeping for documentation purposes
"""
import statistics
from unittest import TestCase

import numpy as np
import numpy.testing
import random
import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt

import paraphrase.utility.annotation_df as annotation_df
import paraphrase.annotation_data as annotation_data
import paraphrase.lead_pc as screening_utility
from paraphrase.set_id_consts import PAID_ANNOTATORS_SORTED_BY_START_TIME, PAID_ANNOTATORS_SORTED_BY_START_TIME_18
from paraphrase.utility.annotation_df import AnnotationColumns, get_vote_shares, predict
from paraphrase.annotation_data import get_PARA_RAND_crowd_annotations, ANNOTATIONS_5050_20_PATH
import paraphrase.utility.evaluate_qsurvey as evaluate_qsurvey
from paraphrase import interview_data
import paraphrase.utility.stats as statistics_util
from paraphrase.utility.PC_utility import get_sampled_5050_pc_ids, get_pilot_pc_labels, get_no_yes_ids_singlepc, \
    get_error_ids, \
    get_context_ids, get_ambiguous_ids, get_pc_paraphrase_ids, get_non_paraphrase_ids, \
    get_non_repetitions_singlepc, get_highlex_ids_singlepc, create_tsv_ids_to_check_singlepc, \
    get_diff_nonpara_ids_singlepc, get_related_nonpara_ids_singlepc, get_random_q_ids
from paraphrase.lead_pc import load_pc_second_batch
from evaluate_pilots import eval_demographics
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from paraphrase.utility.annotator_allocation_util import _get_early_stopping_ids, _get_entropy_stopping_ids, \
    _get_conv_predictions
from paraphrase.interview_data import MediaSumProcessor


class Test(TestCase):

    def setUp(self) -> None:
        # paths to 100 annotations
        self.pc100_annotations_paths = ANNOTATIONS_5050_20_PATH

    def test_save_df(self):
        # USE: this was called to create "100_annotations.tsv" in
        #   "active-listening/result/Annotations/Paraphrase Annotations/5050"
        highlight_df = annotation_df.merge_transform(self.pc100_annotations_paths)
        print(annotation_df.get_unique_q_ids(highlight_df))
        highlight_df.to_csv("utility/fixtures/output/100_annotations.tsv", sep="\t", index=False)

    def test_several_PC_studies(self):
        PC_1 = "utility/fixtures/annotations/23-04-25_Paraphrase_candidate_1-30_April+26,+2023_01.08.tsv"
        PC_31 = "utility/fixtures/annotations/23-04-25_Paraphrase_candidate_31-60_April+26,+2023_01.10.tsv"
        PC_61 = "utility/fixtures/annotations/23-04-25_Paraphrase_candidate_61-90_April+26,+2023_01.12.tsv"
        an_paths = [PC_1, PC_31, PC_61]

        # Test that it finds 90 unique pairs
        pd_report = annotation_df.merge_transform(an_paths, paraphrase_candidate=True)
        result_dict = evaluate_qsurvey.get_freq(pd_report, paraphrase_candidate=True)
        print(result_dict)
        self.assertEqual(90, result_dict["# Annotated Paraphrase Pairs"])

        # Test that it finds cases out of 90 unique pairs
        result_dict = evaluate_qsurvey.calc_pc_eval_crowd(pd_report)
        print(f"\t{result_dict}")
        # evaluate_annotations.main([PC_1, PC_31], paraphrase_candidate=True)

        result_dict = evaluate_qsurvey.calc_classification_agreement(pd_report, paraphrase_candidate=True)
        print(f"\t{result_dict}")

    def test_20_PC_studies(self):
        PC_91 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate_91-110_May+3,+2023_02.37.tsv"
        PC_111 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate_111-130_May+3,+2023_02.37.tsv"
        PC_131 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate_131-150_May+3,+2023_02.37.tsv"
        PC_151 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate_151-170_May+3,+2023_02.37.tsv"
        PC_171 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate_171-190_May+3,+2023_02.36.tsv"
        an_paths = [PC_91, PC_111, PC_131, PC_151, PC_171]

        pd_report = annotation_df.merge_transform(an_paths, paraphrase_candidate=True)
        result_dict = evaluate_qsurvey.get_freq(pd_report, paraphrase_candidate=True)
        print(result_dict)
        result_dict = evaluate_qsurvey.calc_pc_eval_crowd(pd_report)
        print(f"\t{result_dict}")

        result_dict = evaluate_qsurvey.calc_classification_agreement(pd_report, paraphrase_candidate=True)
        print(f"\t{result_dict}")

    def test_w_CC_PC(self):
        """"""
        pc_191 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate-w-CC_191-206_May+8,+2023_06.39.tsv"
        pc_207 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate-w-CC_207-222_May+9,+2023_02.31.tsv"
        pc_223 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate-w-CC_223-238_May+9,+2023_02.29.tsv"
        pc_239 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate-w-CC_239-254_May+9,+2023_02.40.tsv"
        pc_255 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate-w-CC_255-270_May+9,+2023_02.33.tsv"
        pc_271 = "utility/fixtures/annotations/23-05-02_Paraphrase-Candidate-w-CC_271-286_May+9,+2023_02.32.tsv"
        pc_287 = "utility/fixtures/annotations/23-05-09_Paraphrase-Candidate-w-CC_287-304_May+10,+2023_01.44.tsv"
        pc_305 = "utility/fixtures/annotations/23-05-09_Paraphrase-Candidate-w-CC_305-322_May+10,+2023_01.44.tsv"
        pc_323 = "utility/fixtures/annotations/23-05-09_Paraphrase-Candidate-w-CC_323-340_May+10,+2023_01.45.tsv"
        pc_341 = "utility/fixtures/annotations/23-05-09_Paraphrase-Candidate-w-CC_341-358_May+10,+2023_01.45.tsv"
        gold = "utility/fixtures/annotations/23-05-11_Paraphrase-Candidate_Recall_50_May+11,+2023_05.51.tsv"
        # pd_report = annotation_df.merge_transform([pc_191], paraphrase_candidate=True)
        # self.assertEqual(len(pd_report), 66)  # 16*2*2 (comment field + label) + 2 (#annotators)

        pd_report = annotation_df.merge_transform(
            [pc_191, pc_207, pc_223, pc_239, pc_255, pc_271, pc_287, pc_305,
             pc_323, pc_341], paraphrase_candidate=True,
            manual_admission=["605380e706df2f26164b0ba2"])
        gold_report = annotation_df.merge_transform([gold], paraphrase_candidate=True, manual_removal=[])
        gold_binary_dict = annotation_df.get_annotation_dict(gold_report, paraphrase_candidate=True)
        gold_pc_ids = [q_id for sel, q_id in zip(gold_binary_dict["binaries"][0], gold_binary_dict["q_ids"]) if sel]
        print(f"GOLD -- Share of PCs {len(gold_pc_ids) / len(gold_binary_dict['q_ids'])}")

        # pd_report = evaluate_qsurvey.merge_transform([pc_191, pc_207, pc_223, pc_239, pc_255, pc_271], paraphrase_candidate=True)
        # print(f"\t{pd_report}")

        result_dict = evaluate_qsurvey.calc_pc_eval_crowd(
            pd_report[pd_report[AnnotationColumns.Question_ID].isin(gold_binary_dict["q_ids"])])
        print(f"\t{result_dict}")
        cs_pc_ids = result_dict["Resulting Paraphrase Candidates"]
        cohen = statistics_util.cohen_kappa_agreement(
            [1 if q_id in gold_pc_ids else 0 for q_id in gold_binary_dict['q_ids']],
            [1 if q_id in cs_pc_ids else 0 for q_id in gold_binary_dict['q_ids']])
        print(f"Cohen's Kappa Agreement: {cohen}")

        cs_pc_ids = set(cs_pc_ids).intersection(set(gold_binary_dict["q_ids"]))
        print(f"CS -- Share of PCs on same set: {len(cs_pc_ids) / len(set(gold_binary_dict['q_ids']))}")
        print(f"RECALL -- Share of gold PCs that are also selected by crowd: "
              f"{len(cs_pc_ids.intersection(set(gold_pc_ids))) / len(gold_pc_ids)} in absolute numbers "
              f" {len(cs_pc_ids.intersection(set(gold_pc_ids)))} out of possible {len(gold_pc_ids)} "
              f"are selected as PCs by both ")
        print(f"PRECISION -- Share of selected crowd PCs that are gold PCs:"
              f" {len(cs_pc_ids.intersection(set(gold_pc_ids))) / len(cs_pc_ids)} in absolute numbers"
              f" {len(cs_pc_ids.intersection(set(gold_pc_ids)))} out of {len(cs_pc_ids)} are correctly selected")
        # interview = MediaSumProcessor()
        for pc_id in gold_pc_ids:
            if pc_id not in cs_pc_ids:
                print(pc_id)
        print("---------")
        for pc_id in cs_pc_ids:
            if pc_id not in gold_pc_ids:
                print(pc_id)

        result_dict = evaluate_qsurvey.calc_classification_agreement(pd_report, paraphrase_candidate=True)
        print(f"\t{result_dict}")

    def test_lead_author_first_batch_PC(self):
        # first pass annotations (! does not include updated versions)
        # pc_359 = "utility/fixtures/annotations/23-05-11_Paraphrase-Candidate_Recall_359-408_May+17,+2023_02.18.tsv"
        # pc_409 = "utility/fixtures/annotations/23-05-11_Paraphrase-Candidate_Recall_409-458_May+17,+2023_03.48.tsv"
        # pc_459 = "utility/fixtures/annotations/23-05-11_Paraphrase-Candidate_Recall_459-508_May+17,+2023_03.48.tsv"
        # pc_509 = "utility/fixtures/annotations/23-05-15_Paraphrase-Candidate_Recall_509-558_May+17,+2023_03.48.tsv"
        # pc_559 = "utility/fixtures/annotations/23-05-15_Paraphrase-Candidate_Recall_559-608_May+17,+2023_03.49.tsv"
        # pc_609 = "utility/fixtures/annotations/23-05-15_Paraphrase-Candidate_Recall_609-658_May+17,+2023_03.49.tsv"
        # pc_659 = "utility/fixtures/annotations/23-05-15_Paraphrase-Candidate_Recall_659-708_May+17,+2023_03.49.tsv"
        # pc_709 = "utility/fixtures/annotations/23-05-16_Paraphrase-Candidate_Recall_709-758_May+17,+2023_03.50.tsv"
        # pc_759 = "utility/fixtures/annotations/23-05-16_Paraphrase-Candidate_Recall_759-808_May+17,+2023_03.50.tsv"
        # pc_809 = "utility/fixtures/annotations/23-05-16_Paraphrase-Candidate_Recall_809-858_May+17,+2023_03.50.tsv"
        # pc_859 = "utility/fixtures/annotations/23-05-16_Paraphrase-Candidate_Recall_859-908_May+17,+2023_03.50.tsv"
        # pd_report = evaluate_qsurvey.merge_transform([pc_359, pc_409, pc_459, pc_509, pc_559, pc_609, pc_659,
        #                                               pc_709, pc_759, pc_809, pc_859],
        #                                              paraphrase_candidate=True)
        # annotations with new scheme
        pc_959 = "utility/fixtures/annotations/23-05-16_Paraphrase-Candidate_Recall_959-1008_June+9,+2023_01.14.tsv"
        pc_1009 = "utility/fixtures/annotations/23-05-16_Paraphrase-Candidate_Recall_1009-1058_June+9,+2023_01.14.tsv"
        pc_1059 = "utility/fixtures/annotations/23-06-06_Paraphrase-Candidate_Recall_1059-1108_June+9,+2023_01.15.tsv"
        pc_1109 = "utility/fixtures/annotations/23-06-06_Paraphrase-Candidate_Recall_1109-1158_June+9,+2023_01.15.tsv"
        pd_report = annotation_df.merge_transform([pc_959, pc_1009, pc_1059, pc_1109],
                                                  paraphrase_candidate=True)
        pd_report.to_csv("959-1158_annotations.tsv", sep="\t", index=False)

        result_dict = screening_utility.count_stats_pc_second_batch(pd_report)
        print(f"\t{result_dict}")

        interview = MediaSumProcessor("../../data/MediaSum/news_dialogue.json")
        labels = [screening_utility.AnnotationLabels.repetition,
                  screening_utility.AnnotationLabels.context,
                  screening_utility.AnnotationLabels.conclusion,
                  screening_utility.AnnotationLabels.it_reference,
                  screening_utility.AnnotationLabels.inspired_reference,
                  "Rest",
                  "Filtered Out"]
        for keyword in labels:
            print(f'printing 2 examples for {keyword}')
            if "Rest" not in keyword:
                q_ids = random.sample(result_dict[keyword], 2)
            else:
                q_ids = result_dict[keyword]
            for q_id in q_ids:
                ie_utt, ir_utt, summary = interview.get_qdict_from_qid(q_id)
                print(f"Q-ID: {q_id}")
                print(f"Guest:\n\t{ie_utt}")
                print(f"Host:\n\t{ir_utt}")
                print("-----------")
            # else:
            #     for q_id in result_dict[keyword]:
            #         ie_utt, ir_utt, summary = interview.get_gh_pair_from_id(q_id)
            #         print(f"Q-ID: {q_id}")
            #         print(f"Guest:\n\t{ie_utt}")
            #         print(f"Host:\n\t{ir_utt}")
            #         print("-----------")

    def test_lead_author_second_batch_PCs(self):
        """
            after the first batch lead author annotations,
            this evaluates the following PC annotations by lead author
                some contain original PC setup
                rest (from 1409) were annotated using the hl scheme

                with only comments
                CLEAR-NON-PARAPHRASE vs. CLEAR-PARAPHRASE vs. AMBIGUOUS (DIFFICULT, CONTEXT)
                MOSTLY-REPETITION vs. INTERESTING

        :return:
        """
        df_sorted, hl_ids = load_pc_second_batch()

        no_ids, yes_ids = get_no_yes_ids_singlepc(df_sorted, print_stats=True)

        error_ids = get_error_ids(df_sorted, print_stats=True)
        context_ids = get_context_ids(df_sorted, print_stats=True)
        ambigous_ids = get_ambiguous_ids(df_sorted)

        paraphrase_ids = get_pc_paraphrase_ids(df_sorted, print_stats=True)
        non_paraphrase_ids = get_non_paraphrase_ids(df_sorted, print_stats=True)

        repetition_ids, nr_para_ids = get_non_repetitions_singlepc(df_sorted, print_stats=True)

        high_lex_sim = get_highlex_ids_singlepc(df_sorted, print_stats=True)
        diff_non_paraphrase_ids = get_diff_nonpara_ids_singlepc(df_sorted, print_stats=True)
        rel_n_para_ids = get_related_nonpara_ids_singlepc(df_sorted, print_stats=True)

        interview = interview_data.MediaSumProcessor()
        create_tsv_ids_to_check_singlepc(ambigous_ids, df_sorted, interview)

    def test_lead_author_updated_PC(self):
        an_path_1 = "../result/Annotations/359-908_updated-PCs.tsv"
        df1 = pd.read_csv(an_path_1, sep="\t")
        an_path_2 = "../result/Annotations/959-1158_annotations.tsv"
        df2 = pd.read_csv(an_path_2, sep="\t")
        # Concatenate vertically
        concatenated = pd.concat([df1, df2], ignore_index=True)
        result_dict = screening_utility.count_stats_pc_second_batch(concatenated)
        print(f"\t{result_dict}")

        interview = MediaSumProcessor("../../data/MediaSum/news_dialogue.json")
        labels = ["Paraphrases",
                  screening_utility.AnnotationLabels.negated,
                  screening_utility.AnnotationLabels.context,
                  screening_utility.AnnotationLabels.conclusion,
                  screening_utility.AnnotationLabels.difficult,
                  screening_utility.AnnotationLabels.high_lex_sim,
                  screening_utility.AnnotationLabels.perspective_shift,
                  screening_utility.AnnotationLabels.pragmatic]

        for keyword in labels:
            q_ids = result_dict[keyword]
            print(f'printing 2 examples for {keyword}')
            print_ids = random.sample(result_dict[keyword], 2)
            for q_id in q_ids:
                ie_utt, ir_utt, summary = interview.get_qdict_from_qid(q_id)
                comment = concatenated[(concatenated["QID"] == q_id) & (concatenated["Category"] == "Comment")][
                    "Highlighted"].values.tolist()[0]
                if keyword in dir(screening_utility.AnnotationLabels):
                    self.assertIn(keyword, comment)
                if q_id in print_ids:
                    print(f"Q-ID: {q_id}")
                    print(f"Guest:\n\t{ie_utt}")
                    print(f"Host:\n\t{ir_utt}")
                    print("-----------")

    def test_transform_separate_binaries_to_combined_nan_annotations(self):
        q_ids = [['CNN-380447-5', 'NPR-43338-14', 'NPR-10765-6', 'NPR-10765-14', 'NPR-10765-17', 'NPR-10765-19',
                  'NPR-10765-21', 'NPR-18176-5', 'NPR-18176-7', 'NPR-18176-10', 'NPR-18176-14', 'NPR-18176-16',
                  'NPR-45032-3', 'NPR-45032-5', 'NPR-45032-7', 'NPR-45032-9', 'CNN-248640-3', 'CNN-248640-5',
                  'NPR-12973-3', 'NPR-12973-8', 'NPR-12973-10', 'NPR-12973-12', 'NPR-12973-14', 'NPR-1580-3',
                  'NPR-12237-3', 'NPR-12237-5', 'NPR-12237-8', 'NPR-27943-6', 'CNN-407066-3', 'CNN-407066-5'],
                 ['NPR-7797-5', 'NPR-7797-8', 'CNN-191341-5', 'NPR-14597-4', 'NPR-14597-8', 'CNN-185107-3',
                  'NPR-29642-3', 'NPR-29642-6', 'NPR-29642-9', 'NPR-8808-5', 'NPR-8808-7', 'NPR-8808-10', 'NPR-8808-12',
                  'NPR-8808-14', 'CNN-34183-3', 'CNN-34183-5', 'CNN-34183-7', 'CNN-34183-9', 'NPR-44706-11',
                  'CNN-188022-3', 'CNN-188022-5', 'CNN-250097-6', 'CNN-250097-8', 'CNN-250097-10', 'CNN-250097-12',
                  'CNN-250097-14', 'CNN-114979-3', 'CNN-114979-5', 'CNN-114979-9', 'CNN-11668-3']]
        annotations = [[1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
                       [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]]

        result, _ = annotation_df._transform_separate_binaries_to_combined_nan_annotations(annotations, q_ids)
        nan = float('nan')
        # self.assertEqual(nan, nan)
        numpy.testing.assert_equal(result[0],
                                   [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
                                    0, 1] + [nan] * 30)
        numpy.testing.assert_equal(result[1],
                                   [nan] * 30 + [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
                                                 1, 0, 1, 0, 0, 0])
        # self.assertEqual(result[0], [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1] + [nan]*30)
        # self.assertEqual(result[1], [nan]*30 + [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0])

    def test_null_model(self):
        import statistics

        random.seed = 42
        n_items = 16  # 20
        chance_1 = 0.5  # or: 0.44, i.e., chance to select as possible paraphrase

        pc_values = []
        cohen_values = []
        cohen_mv_values = []
        max_mv_cohen = []
        accuracy = []
        contained_zeroes = []
        annotations = []
        agg = numpy.array([[1, 1]])  # first 0, then 1
        individual_pc_shares = []
        for i in range(10000):
            an_1 = random.choices([0, 1], k=n_items, weights=[round((1 - chance_1) * 100), round(chance_1 * 100)])
            an_2 = random.choices([0, 1], k=n_items, weights=[round((1 - chance_1) * 100), round(chance_1 * 100)])
            annotations.append(an_1)
            annotations.append(an_2)
            evaluate_qsurvey._calc_pc_eval_scores(an_1, an_2, pc_values, cohen_values, cohen_mv_values, max_mv_cohen,
                                                  accuracy, contained_zeroes, individual_pc_shares)
            aggregate = aggregate_raters(numpy.array([an_1, an_2]).transpose())
            zero_index = numpy.where(aggregate[1] == 0)[0]
            if zero_index != 0:
                aggregate[0][:, [1, 0]] = aggregate[0][:, [0, 1]]
            agg = numpy.r_[agg, aggregate[0]]
            # mv = [1 if 1 in elems else 0 for elems in zip(an_1, an_2)]
            # agreement = [1 if a == b else 0 for a, b in zip(an_1, an_2)]
            # consumed_max = calc_consumed_max(an_1, an_2)
            #
            # pc_values.append(evaluate_qsurvey.
            #                  _calc_candidate_freq_from_binaries([an_1, an_2])["relative at least one vote"])
            # cohen_values.append(statistics_util.cohen_kappa_agreement(an_1, an_2))
            # cohen_mv_values.append(statistics_util.cohen_kappa_agreement(an_1, mv))
            # cohen_mv_values.append(statistics_util.cohen_kappa_agreement(an_2, mv))
            # max_mv_cohen.append(max(statistics_util.cohen_kappa_agreement(an_1, mv),
            #                         statistics_util.cohen_kappa_agreement(an_2, mv)))
            # accuracy.append(sum(agreement) / len(agreement))
            #
            # contained_zeroes.append(consumed_max)

        print(
            f"share of null model individual paraphrase candidates {sum(individual_pc_shares) / len(individual_pc_shares)} "
            f"with STD {statistics.stdev(individual_pc_shares)}")
        print(f"share of null model paraphrase candidates {sum(pc_values) / len(pc_values)} with STD "
              f"{statistics.stdev(pc_values)}")
        print(f"AVG cohen agreement null model {sum(cohen_values) / len(cohen_values)} "
              f"with STD {statistics.stdev(cohen_values)}")
        print(f"AVG majority vote cohen agreement null model {sum(cohen_mv_values) / len(cohen_mv_values)} "
              f"with STD {statistics.stdev(cohen_mv_values)}")
        print(f"AVG max majority vote cohen agreement null model {sum(max_mv_cohen) / len(max_mv_cohen)}"
              f" with STD {statistics.stdev(max_mv_cohen)}")
        print(f"AVG accuracy null model {sum(accuracy) / len(accuracy)} "
              f"with STD {statistics.stdev(accuracy)}")
        print(f"AVG maximum of directional 0 -> 0 overlap null model {sum(contained_zeroes) / len(contained_zeroes)} "
              f"with STD {statistics.stdev(contained_zeroes)}")
        print(f"Fleiss {fleiss_kappa(agg)}")

    def test_eval_demographics(self):

        demo_path = "../result/Annotations/demographics_prolific_export_6447b8a669103d69cfbd6d69.csv"
        demo_df = eval_demographics(demo_path)
        participant_ids = demo_df[demo_df['Status'] != 'RETURNED']["Participant id"]
        self.assertEqual(len(participant_ids), 61)
        self.assertSetEqual(set(PAID_ANNOTATORS_SORTED_BY_START_TIME), set(participant_ids.values.tolist()))

    def test_eval_annotator_training(self):
        n = 61
        sorted_annotator_list = PAID_ANNOTATORS_SORTED_BY_START_TIME_18
        training_path = "fixtures/annotations/23-06-13_Training_Paraphrase-Highlighting_June+26,+2023_01.01.tsv"
        # UNCOMMENT training_path for 63 annotators
        # training_path = "../result/Annotations/Paraphrase Annotations/RAND-PARA/" \
        #                 "23-06-13_Training_Paraphrase-Highlighting_July+24,+2023_01.46.tsv"
        # n = 63
        # sorted_annotator_list = PAID_ANNOTATORS_SORTED_BY_START_TIME
        # filters all cases where annotators were asked to return on prolific (failed 2 ACs or failed the initial CC)
        train_df = annotation_df.merge_transform([training_path])
        train_df.to_csv("../output/Annotator-Training.tsv", sep="\t")
        self.assertSetEqual(set(sorted_annotator_list), set(train_df["Annotator"].unique()))
        self.assertEqual(n, len(train_df[train_df["Category"] == "# failed ACs"]))
        self.assertEqual(n, len(train_df[train_df["Category"] == "Time"]))
        self.assertEqual(n, len(train_df[train_df["Category"] == "EndTime"]))

        # calculate average time
        times_sorted_start_date = train_df[train_df["Category"] == "Time"].sort_values(
            by="Annotator", key=lambda x: x.map({k: i for i, k in enumerate(sorted_annotator_list)}))
        times_sorted_start_date = times_sorted_start_date["Highlighted"].values.tolist()
        times_sorted_start_date = list(map(int, times_sorted_start_date))
        # unsorted_times = train_df[train_df["Category"] == "Time"]["Highlighted"].values.tolist()
        # unsorted_times = list(map(int, unsorted_times))
        # Remove 7h time
        times_sorted_start_date.remove(23685)
        sorted_annotator_list.remove("64136bf30b27746cb96f7db8")

        sns.lmplot(x='nth participant', y='duration',
                   data=pd.DataFrame({'nth participant': list(range(len(times_sorted_start_date))),
                                      'duration': times_sorted_start_date}))
        plt.axhline(y=900, color='red', linestyle='--')
        developing_median = [np.median(np.array(times_sorted_start_date[:i])) for i in
                             range(len(times_sorted_start_date))]
        # Plot the developing median line
        plt.plot(list(range(len(times_sorted_start_date))), developing_median, color="green", linewidth=2,
                 linestyle="--")
        plt.show()

        # get accepted annotators
        highlight_df = annotation_df.merge_transform(self.pc100_annotations_paths)
        accepted_annotators = highlight_df["Annotator"].unique().tolist()
        developing_quality = [len(set(sorted_annotator_list[:i]).intersection(set(accepted_annotators)))
                              for i in range(len(sorted_annotator_list))]
        # Convert the list to a DataFrame
        df = pd.DataFrame({'# annotators passing training': developing_quality,
                           'first n participants': list(range(len(times_sorted_start_date)))})
        sns.lineplot(data=df, x='first n participants', y='# annotators passing training')

        # Add a line for the angle bisector
        lims = [
            0,  # np.min([plt.xlim(), plt.ylim()]),  # min of both axes
            len(times_sorted_start_date)  # np.max([plt.xlim(), plt.ylim()]),  # max of both axes
        ]
        plt.plot(lims, lims, 'k-', alpha=0.5, zorder=0)
        # plt.axis('equal')
        # Show the plot
        plt.xlim(0, len(times_sorted_start_date))
        plt.ylim(0, len(times_sorted_start_date))
        # Turn off autoscaling on the x-axis
        plt.gca().set_autoscale_on(False)
        plt.margins(x=0)
        plt.show()

        sns.kdeplot(times_sorted_start_date)
        plt.axvline(x=900, color='red', linestyle='--')
        plt.show()

        times_sorted_start_date.sort()
        print(times_sorted_start_date)
        print(f"median: {statistics.median(times_sorted_start_date)}")
        print(f"mean: {statistics.mean(times_sorted_start_date)} +- {statistics.stdev(times_sorted_start_date)}")

        # For ACCEPTED ANNOTATORS
        highlight_df = annotation_df.merge_transform(self.pc100_annotations_paths)
        #   get accuracy
        binaries_dict = annotation_df.get_annotation_dict(highlight_df)
        binaries = numpy.array(binaries_dict["complete_binaries"])

        paraphrase_shares = get_vote_shares(highlight_df)["votes"]

        majority_vote = [predict(p_share) for p_share in paraphrase_shares]

        annotators = binaries_dict["annotators"]
        acc_by_completion_time = [count_same_values_same_index(binaries[annotators.index(an)], majority_vote) /
                                  len(binaries_dict["binaries"][annotators.index(an)])
                                  for an in sorted_annotator_list if an in annotators]

        sns.lmplot(x='nth annotator', y='accuracy w.r.t. majority vote',
                   data=pd.DataFrame({'nth annotator': list(range(len(acc_by_completion_time))),
                                      'accuracy w.r.t. majority vote': acc_by_completion_time}))
        plt.axhline(y=statistics.mean(acc_by_completion_time), color='red', linestyle='--')
        developing_mean = [np.mean(np.array(acc_by_completion_time[:i])) for i in
                           range(len(acc_by_completion_time))]
        # Plot the developing median line
        plt.plot(list(range(len(acc_by_completion_time))), developing_mean, color="green", linewidth=2,
                 linestyle="--")
        plt.show()


def count_same_values_same_index(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists should have the same length")

    count = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            count += 1
    return count


def get_sem_sims(relevant_hl_ids, pd_hl, interview):
    """
        for the ids given in relevant_hl_ids, calculate the semantic similarities between the matching highlighting,
        assumes only one annotator (Lead Author condition)
    :param relevant_hl_ids:
    :param pd_hl:
    :param interview:
    :return:
    """
    hl_guest, _, _, _ = annotation_data.extract_txts_to_compare('Referred',
                                                                pd_hl[pd_hl["QID"].isin(relevant_hl_ids)],
                                                                interview, single_annotator=True)
    hl_host, _, _, _ = annotation_data.extract_txts_to_compare('Paraphrase',
                                                               pd_hl[pd_hl["QID"].isin(relevant_hl_ids)],
                                                               interview, single_annotator=True)
    hl_guest = [li[0] for li in hl_guest]
    hl_host = [li[0] for li in hl_host]
    _, sem_sims = statistics_util.transformer_scores([[t1, t2] for t1, t2 in zip(hl_guest, hl_host)])
    return sem_sims


def get_latex_code_for_q_id(an_order_per_q, annotators, highlight_df, n_annotators, q_id, q_ids, interview=None, n=4):
    subselect_highligths = get_dynamic_entropy_hl_from_qid(q_id, highlight_df, an_order_per_q, annotators,
                                                           n_annotators, q_ids, thresh=0.8)

    evaluate_qsurvey._print_latex_guest_host_highlights(subselect_highligths, interview)


def get_early_stopping_hl_from_qid(q_id, highlight_df, an_order_per_q, annotators, n_annotators, q_ids, n=4):
    # get the highlights for early stopping
    q_index = q_ids.index(q_id)
    q_anns = [annotators[an_index] for an_index in an_order_per_q[q_index][0:n_annotators[n - 1][q_index]]]
    subselect_highligths = highlight_df[
        (highlight_df[annotation_df.AnnotationColumns.Question_ID] == q_id) &
        (highlight_df[annotation_df.AnnotationColumns.Annotator_ID].isin(q_anns))]
    return subselect_highligths


def get_dynamic_entropy_hl_from_qid(q_id, highlight_df, an_order_per_q, annotators, n_annotators, q_ids, thresh=0.8):
    # get the highlights for early stopping
    q_index = q_ids.index(q_id)
    q_anns = [annotators[an_index] for an_index in
              an_order_per_q[q_index][0:n_annotators[int(9 - thresh / 0.1)][q_index]]]
    subselect_highligths = highlight_df[
        (highlight_df[annotation_df.AnnotationColumns.Question_ID] == q_id) &
        (highlight_df[annotation_df.AnnotationColumns.Annotator_ID].isin(q_anns))]
    return subselect_highligths


def boxplot_w_accuracies(predictions, conv_share=None, n=18, halfwidth=False):
    """
        prints a boxplot with Distances to the converged vote on y axis for each # of annotators on the x axis
    :param predictions:
    :param conv_share:
    :param n:
    :return:
    """
    converged_share, gt_preds = _get_conv_predictions(predictions, conv_share)

    # abs_share = 1 - abs(predictions - converged_share)
    abs_distances = abs(predictions - converged_share)  # /np.maximum(converged_share, 1-converged_share)

    # get BINARY accuracy
    mv_predictions = np.vectorize(predict, otypes=[np.float])(predictions)
    abs_gt = abs(mv_predictions - gt_preds)
    accs = np.count_nonzero(abs_gt == 0, axis=1) / np.count_nonzero(abs_gt < 5, axis=1)

    ylabel = "Distance to Converged Vote"
    xlabel = "Binary Accuracies with Increasing # Annotators"

    plot_boxplot_distaccs(abs_distances, accs, xlabel, ylabel, n, halfwidth)

    return gt_preds, converged_share, accs[:n]


def plot_boxplot_distaccs(abs_distances, accs, xlabel, ylabel, n, halfwidth=False):
    print(f"MEAN Distance {np.mean(abs_distances, axis=1)} with STD {np.std(abs_distances, axis=1)}")
    # PLOT
    width = 10
    if halfwidth:
        width = 6
    fig, ax = plt.subplots(figsize=(width, 4))
    # Create scatter plot using Seaborn
    df = pd.DataFrame(abs_distances.transpose())
    # PLOT
    # plt.figure(figsize=(8, 4))
    sns.boxplot(data=df)
    plt.xlim(right=n - 1)
    plt.ylim(0.00, 1.05)
    ax.set_xticks(range(len(accs[:n])))
    ax.set_xticklabels([str(i + 1) + "\n" + str(round(acc, 2))
                        for i, acc in enumerate(accs[:n])])  ## ????
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    ax.tick_params(axis='x', which='major', labelsize=11)
    ax.tick_params(axis='y', which='major', labelsize=15)
    # Adjust the bottom of the subplots upward
    plt.subplots_adjust(bottom=0.2)
    # Display the plot
    plt.show()


def get_pred_dynamic_stopping(cumulative_matrix, fixed_predictions, ranks=None,
                              max_total_annotators=18, entropy=False, min_votes=3):
    """

    :param max_total_annotators: n close to 20 for entropy false,
    :param cumulative_matrix: cumulative # votes with increasing annotators per question, e.g.,
                                0.0,0.0,1.0,0.0,0.0
                                1.0,0.0,2.0,0.0,0.0
    :param fixed_predictions:
    :param ranks: matrix with "rank" annotators, e.g., 2 annotators and 5 questions
                    1,1,1,1,1
                    2,2,2,2,2



    :return:
    """
    if ranks is None:
        ranks = (np.arange(max_total_annotators).reshape(max_total_annotators, 1) + 1).repeat(
            cumulative_matrix.shape[1], axis=1)

    if not entropy:
        # reverse of cumulative matrix, i.e., # of annotators voted for non-paraphrase
        rev_cum_matrix = (ranks - cumulative_matrix)[:max_total_annotators]
        loop_start = 1
        loop_stopper = round(max_total_annotators / 2) + 1
        step = 1
    else:
        loop_start = 9
        loop_stopper = 0
        step = -1

    preds_lists = []
    annotator_lists = []
    for k in range(loop_start, loop_stopper, step):  # for number of agreeing annotators
        # get the predictions that would result using early stopping
        if not entropy:
            earliest_n_agreements = _get_early_stopping_ids(k, cumulative_matrix, rev_cum_matrix)
        else:
            earliest_n_agreements = _get_entropy_stopping_ids(k * 0.1, cumulative_matrix,
                                                              max_votes=max_total_annotators, min_votes=min_votes)
        nth_predictions = [fixed_predictions[agree_index, i] for i, agree_index in enumerate(earliest_n_agreements)]
        nth_annotators = [agree_index + 1 for agree_index in earliest_n_agreements]
        preds_lists.append(nth_predictions)
        annotator_lists.append(nth_annotators)
    return preds_lists, annotator_lists
