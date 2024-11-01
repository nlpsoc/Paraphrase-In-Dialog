"""
    Evaluate, print and plot results for
        - preprocessing
        - lead author annotations

"""
import random

import pandas as pd

import paraphrase.utility.annotation_df as annotation_df
import paraphrase.utility.evaluate_qsurvey as evaluate_qsurvey
from paraphrase import interview_data
import paraphrase.utility.stats as statistics_util
from paraphrase.utility.PC_utility import print_pc_stats, get_sampled_5050_pc_ids, \
    get_sampled_RAND_PARA_pc_ids, print_interview_pair_stats, get_random_q_ids, get_pc_paraphrase_ids
from paraphrase.utility.prolific_api import print_pph_project
from paraphrase.lead_pc import get_author_annotations
from paraphrase.utility.annotation_pipeline import QualtricsProlificDB, _download_qualtrics_surveys
from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.utility.annotation_df import AnnotationColumns, get_sub_df_from_qids
from preprocess_interviews import IDS_CSV
from paraphrase.interview_data import TripleIDs
from paraphrase.annotation_data import get_all_crowd_annotations, _get_sub_df_from_annotators, get_aggregated_human_anns
from paraphrase.utility.stats import plot_relative_triplets


def main(print_preprocessing_stats=False, print_lead_annotation_stats=False, print_annotation_stats=True,
         print_lead_mv_stats=False, print_single_qs=True, print_train_stats=False):
    # DATASET STATS
    if print_preprocessing_stats:  # gets the stats for the interview/g-h pair number of items
        # PREPROCESSING STATS
        #   takes some time to run because 200 token removal after the fact
        #   After preprocssing Dataset STAT (but before 200 token removal)
        preprocessed_df = pd.read_csv(IDS_CSV, sep="\t")
        pair_ids = [TripleIDs.get_unique_id(
            row[TripleIDs.CONVO_ID], row[TripleIDs.IE_UTTERANCE_ID]) for _, row in preprocessed_df.iterrows()]
        #   DO 200 token removal
        token_limit_ids = []
        interview = interview_data.MediaSumProcessor(uncut=True)
        for gh_id in pair_ids:
            g_utt = interview.get_guest_utt_from_id(gh_id)
            if len(g_utt.split()) > 200:
                continue
            h_utt = interview.get_host_utt_from_id(gh_id)
            if len(h_utt.split()) > 200:
                continue
            token_limit_ids.append(gh_id)
        print_interview_pair_stats(token_limit_ids)

    # GET IDs for RAND, all 500 and 50/50
    an500_ids = get_sampled_RAND_PARA_pc_ids()
    random_100_ids = get_random_q_ids()
    an100_ids = get_sampled_5050_pc_ids()

    if print_lead_annotation_stats:
        # LEAD AUTHOR ANNOTATION STATS
        #       load screening annotations
        lead_pc_estimations, first_batch_pcs, second_patch_pcs = get_author_annotations(print_pilot_stats=True)
        #       print stats
        first_batch_np, first_batch_p, first_batch_r = print_pc_stats(first_batch_pcs, first_batch=True)
        second_batch_np, second_batch_p, second_batch_r = print_pc_stats(second_patch_pcs)
        all_np, all_p, all_r = print_pc_stats(lead_pc_estimations)

        #           PC Screening Stats for all 4450
        #               and the first and second batch of screening
        plot_relative_triplets([
            (all_p, all_r, all_np), (first_batch_p, first_batch_r, first_batch_np),
            (second_batch_p, second_batch_r, second_batch_np)
        ], ["Total", "First Batch", "Second Batch"],
            title="Screening Stats for per batch and all 4450")
        #           PC Screening Stats for all 4450 and the annotated samples

        pilot_np, pilot_p, pilot_r = print_pc_stats(
            lead_pc_estimations[lead_pc_estimations[AnnotationColumns.Question_ID].isin(an100_ids)],
            include_ambiguous=True
        )
        pc400_np, pc400_p, pc400_r = print_pc_stats(
            lead_pc_estimations[lead_pc_estimations[AnnotationColumns.Question_ID].isin(an500_ids) &
                                ~(lead_pc_estimations[AnnotationColumns.Question_ID].isin(random_100_ids))])
        random100_np, random100_p, random100_r = print_pc_stats(
            lead_pc_estimations[lead_pc_estimations[AnnotationColumns.Question_ID].isin(random_100_ids)])

        plot_relative_triplets([
            # (all_p, all_r, all_np),
            (random100_p, random100_r, random100_np),
            (pc400_p, pc400_r, pc400_np), (pilot_p, pilot_r, pilot_np)
        ], [  # "Total",
            "RANDOM", "PARA", "50/50"])

    crowd_anns_df = get_all_crowd_annotations(include_fails=False)

    if print_single_qs:
        interview = interview_data.MediaSumProcessor()
        evaluate_qsurvey.print_latex_for_q(crowd_anns_df, "CNN-339958-3", interview)
        evaluate_qsurvey.print_latex_for_q(crowd_anns_df, "CNN-80522-7", interview)

    if print_annotation_stats:
        # ANNOTATION STATS
        #  get basic stats
        #    number of annotators
        print(f"Number of unique annotators: {len(annotation_df.get_annotators(crowd_anns_df))}")
        #    number of annotations
        print(f"Number of annotations: {annotation_df.get_number_annotations(crowd_anns_df)}")

        #    number of paraphrases acc. to majority vote per dataset
        #       RAND
        #           get annotations
        interview = interview_data.MediaSumProcessor()
        _, _, _, human_anns = get_aggregated_human_anns(interview, random_100_ids)
        classifications = [1 if value[0]/value[1] >= 0.5 else 0 for _, value in human_anns.items()]
        annotations = sum(value[1] for _, value in human_anns.items())
        print(f"Number of paraphrases in RAND: {sum(classifications)}")
        print(f"Number of annotations in RAND: {annotations}")
        #       PARA
        _, _, _, human_anns = get_aggregated_human_anns(interview, set(an500_ids) - set(random_100_ids))
        classifications = [1 if value[0]/value[1] >= 0.5 else 0 for _, value in human_anns.items()]
        annotations = sum(value[1] for _, value in human_anns.items())
        print(f"Number of paraphrases in PARA: {sum(classifications)}")
        print(f"Number of annotations in PARA: {annotations}")
        #       50/50
        _, _, _, human_anns = get_aggregated_human_anns(interview, an100_ids)
        classifications = [1 if value[0]/value[1] >= 0.5 else 0 for _, value in human_anns.items()]
        annotations = sum(value[1] for _, value in human_anns.items())
        print(f"Number of paraphrases in 50/50: {sum(classifications)}")
        print(f"Number of annotations in 50/50: {annotations}")
        #       TOTAL
        _, _, _, human_anns = get_aggregated_human_anns(interview)
        classifications = [1 if value[0]/value[1] >= 0.5 else 0 for _, value in human_anns.items()]
        annotations = sum(value[1] for _, value in human_anns.items())
        print(f"Number of paraphrases in TOTAL: {sum(classifications)}")
        print(f"Number of annotations in TOTAL: {annotations}")

        #   For RAND
        #       get all annotations
        ann_tuples = annotation_df.get_annotation_tuples(get_sub_df_from_qids(crowd_anns_df, random_100_ids))
        #       sample 100 random tuples from list
        statistics_util.set_global_seed()
        random_tuples = random.sample(ann_tuples, 100)
        print(random_tuples)
        #       print latex

        for q_id, annotator in random_tuples:
            subselected_df = _get_sub_df_from_annotators(crowd_anns_df, [annotator])
            evaluate_qsurvey.print_latex_for_q(subselected_df, q_id, interview)

        #       sample 10 random annotators, and get their annotations
        print("-------- RANDOM 10 ANNOTATORS --------")
        random_annotators = random.sample(annotation_df.get_annotators(crowd_anns_df), 10)
        nbr_annotations = 0
        for annotator in random_annotators:
            print(f"Annotator: {annotator}")
            subselected_df = _get_sub_df_from_annotators(crowd_anns_df, [annotator])
            # get all questions annotated by annotator and sample 10 of those
            question_ids = subselected_df[AnnotationColumns.Question_ID].unique()  # currently includes
            random_question_ids = random.sample(list(question_ids), min(5, len(question_ids)))
            for q_id in random_question_ids:
                evaluate_qsurvey.print_latex_for_q(subselected_df, q_id, interview)
                nbr_annotations += 1
            print(f"End Annotator: {annotator} with {len(random_question_ids)} annotations.")
        print(f"Number of annotations: {nbr_annotations}")
        print("-------- END RANDOM 10 ANNOTATORS --------")
        #   RATER AGREEMENT
        #       CLASSIFICATION
        disagreement_ids = ["CNN-98333-7", "CNN-268015-12", "CNN-15655-13", "CNN-234545-5", "CNN-154452-7",
                            "CNN-332405-27", "NPR-733-15"]
        for item_id in disagreement_ids:
            evaluate_qsurvey.print_latex_for_q(crowd_anns_df, item_id, interview)

        #           PRINT AGREEMENT EXAMPLES
        figure_1_ids = ["CNN-384581-5", "CNN-74126-3", "CNN-350238-9", "CNN-376903-8", "NPR-4239-7", "NPR-15913-5"]
        for item_id in figure_1_ids:
            evaluate_qsurvey.print_latex_for_q(crowd_anns_df, item_id, interview)
        vote_dict = annotation_df.get_vote_tuples(get_sub_df_from_qids(crowd_anns_df,
                                                                       set(an500_ids) - set(random_100_ids)))
        for vote_tuple, q_id in zip(vote_dict['votes'], vote_dict['q_ids']):
            if vote_tuple[0] / vote_tuple[1] > 0.9:
                evaluate_qsurvey.print_latex_for_q(crowd_anns_df, q_id, interview)

        #           PRINT DISAGREEMENT EXAMPLES
        vote_dict = annotation_df.get_vote_tuples(crowd_anns_df)
        for vote_tuple, q_id in zip(vote_dict['votes'], vote_dict['q_ids']):
            if (vote_tuple[0] / vote_tuple[1] > 0.45) and (vote_tuple[0] / vote_tuple[1] < 0.55):
                evaluate_qsurvey.print_latex_for_q(crowd_anns_df, q_id, interview)

        #           CALCULATE AGREEMENT STATS
        evaluate_qsurvey.calc_classification_agreement(get_sub_df_from_qids(crowd_anns_df, random_100_ids),
                                                       min_overlap=1)
        evaluate_qsurvey.calc_classification_agreement(get_sub_df_from_qids(crowd_anns_df, an100_ids), min_overlap=1)
        evaluate_qsurvey.calc_classification_agreement(get_sub_df_from_qids(crowd_anns_df,
                                                                            set(an500_ids) - set(random_100_ids)),
                                                       min_overlap=1)

        #       HIGHLIGHTING
        evaluate_qsurvey.get_hl_agreement(get_sub_df_from_qids(crowd_anns_df, random_100_ids), interview)
        evaluate_qsurvey.get_hl_agreement(get_sub_df_from_qids(crowd_anns_df, an100_ids), interview)
        evaluate_qsurvey.get_hl_agreement(get_sub_df_from_qids(crowd_anns_df, set(an500_ids) - set(random_100_ids)),
                                          interview)

    if print_lead_mv_stats:
        #   AGREEMENT LEAD Author with crowd-sourced Majority Vote
        print("Agreement Lead Author/Crowd-sourced Majority Vote")
        try:
            lead_pc_estimations
        except NameError:
            lead_pc_estimations, first_batch_pcs, second_patch_pcs = get_author_annotations()

        #       on RAND
        calculate_lead_mv_agreement(crowd_anns_df,
                                    lead_pc_estimations,
                                    random_100_ids)

        #       on PARA
        calculate_lead_mv_agreement(crowd_anns_df, lead_pc_estimations, set(an500_ids) - set(random_100_ids))
        #       on 50/50
        calculate_lead_mv_agreement(crowd_anns_df, lead_pc_estimations, an100_ids)

    if print_train_stats:
        # print pay per hour info
        print_pph_project()
        #      TRAINING
        #          first batch
        first_training_path = get_dir_to_src() + "/../result/Annotations/Paraphrase Annotations/RAND-PARA/" \
                                                 "23-06-13_Training_Paraphrase-Highlighting_July+24,+2023_01.46.tsv"
        train_df = annotation_df.merge_transform([first_training_path], include_fails=True)
        print_train_filter_stats(crowd_anns_df, train_df, an100_ids)
        #           second batch
        ann2_db = QualtricsProlificDB()
        print(f"For the second annotation batch: "
              f"Out of {len(ann2_db.get_participated_train_annotators())} annotators, "
              f"{len(ann2_db.get_valid_train_annotators())} passed training.")
        # get train ids
        train_ids = ann2_db._get_qualtrics_train_ids()
        survey_paths = _download_qualtrics_surveys(train_ids)
        train_df = annotation_df.merge_transform(survey_paths, include_fails=True)
        print_train_filter_stats(crowd_anns_df, train_df, ann_ids=ann2_db.get_valid_train_annotators())


def print_train_filter_stats(crowd_anns_df, train_df, q_ids=None, ann_ids=None):
    assert (q_ids is not None) or (ann_ids is not None)
    nbr_stopped = len(train_df[
                          (train_df[
                               AnnotationColumns.Highlight_Category] == annotation_df.AllAnnotationCategories.Stopped) &
                          (train_df[AnnotationColumns.Highlight_Choices] == 1)])
    if ann_ids is None:
        ann_ids = crowd_anns_df[crowd_anns_df[AnnotationColumns.Question_ID].isin(q_ids)][
            AnnotationColumns.Annotator_ID].nunique()
    else:
        ann_ids = len(ann_ids)

    nbr_acs = len(train_df[
                      (train_df[
                           AnnotationColumns.Highlight_Category] == annotation_df.AllAnnotationCategories.AC_fails) &
                      (train_df[AnnotationColumns.Highlight_Choices] != "0")])

    print(f"For the current annotation batch: "
          f"Out of {train_df['Annotator'].nunique()} annotators, {nbr_stopped} stopped and "
          f"{ann_ids} "
          f"passed training, "
          f"i.e., {ann_ids} / {(train_df['Annotator'].nunique() - nbr_stopped)} "
          f"= {ann_ids / (train_df['Annotator'].nunique() - nbr_stopped)}% passed training,"
          f"{nbr_acs} / {(train_df['Annotator'].nunique() - nbr_stopped)} = {nbr_acs / ((train_df['Annotator'].nunique() - nbr_stopped))} failed at least one ACs, "
          f"{(train_df['Annotator'].nunique() - nbr_stopped - ann_ids) - nbr_acs} / {(train_df['Annotator'].nunique() - nbr_stopped)} = {((train_df['Annotator'].nunique() - nbr_stopped - ann_ids) - nbr_acs) / (train_df['Annotator'].nunique() - nbr_stopped)}  failed at least one CC")


def calculate_lead_mv_agreement(all_crowd_annotations_df, lead_pc_estimations, subset_q_ids):
    """
        calculate the agreement between lead author and majority vote given
            the annotations on all PCs and
            a subset of q_ids to evaluate the agreement on
    :param all_crowd_annotations_df: dataframe with annotations from crowd
    :param lead_pc_estimations: dataframe with lead author estimations
    :param subset_q_ids: list or set
    :return:
    """
    # get crowd/lead annotations for the given set of q_ids
    crowd_anns_df = get_sub_df_from_qids(all_crowd_annotations_df, subset_q_ids)
    lead_anns_df = get_sub_df_from_qids(lead_pc_estimations, subset_q_ids)
    #   GET paraphrase ids stats for lead estimations
    paraphrase_ids = get_pc_paraphrase_ids(lead_anns_df)
    #   GET annotation DICT for crowd annotations
    result_dict = annotation_df.get_annotation_dict(crowd_anns_df)
    ordered_votes = result_dict['temporally_ordered_classifications']
    ordered_qids = result_dict['temporally_ordered_q_ids']
    # GET majority and lead votes for the given subset of q_ids
    majority_votes = []
    author_vote = []
    for q_id in subset_q_ids:
        mv = sum(ordered_votes[ordered_qids.index(q_id)]) / len(ordered_votes[ordered_qids.index(q_id)])
        mv = 1 if mv >= 0.5 else 0
        majority_votes.append(mv)
        if q_id in paraphrase_ids:
            author_vote.append(1)
        else:
            author_vote.append(0)
    # CALCULATE Cohen and Overlap
    print(f"Cohen's Kappa Agreement: {statistics_util.cohen_kappa_agreement(majority_votes, author_vote)}")
    print(f"Overlap: {statistics_util.relative_overlap(majority_votes, author_vote)}")


if __name__ == "__main__":
    main()
