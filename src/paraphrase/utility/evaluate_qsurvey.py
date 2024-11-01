"""
    utility python file to evaluate responses
"""
import math
from typing import List, Dict, Any
import ast

from itertools import chain

import numpy
import numpy as np
import pandas
import pandas as pd
import sklearn.metrics
from numpy import mean, std

import statistics
from paraphrase.annotator_allocation import get_min_nbr_anns_entropy
from paraphrase.utility.dataframe import create_condition
from paraphrase.utility.annotation_df import AnnotationColumns, AllAnnotationCategories, ContentReproductionCategories, \
    get_annotation_dict, assert_valid_category
from paraphrase.utility.annotator_allocation_util import get_cumulative_matrix
from paraphrase.annotation_data import _get_annotations_per_qid, extract_txts_to_compare
from paraphrase.utility.stats import cohen_kappa_agreement, transformer_scores, jaccard_overlap_for_highlights, \
    fleiss_kappa_agreement, calc_krippendorff_alpha, \
    set_global_seed
from paraphrase import interview_data
import paraphrase.utility.qualtrics_survey as qualtrics_survey

from paraphrase.utility.tex_highlighting import gen_guest_host_tex

import random

set_global_seed()

FIRST_100_IDs = ['CNN-57562-3', 'CNN-57562-5', 'NPR-2972-3', 'NPR-2972-6', 'NPR-20293-3', 'NPR-20293-5', 'NPR-20293-7',
                 'CNN-81467-9', 'CNN-81467-11', 'CNN-81467-19', 'CNN-33106-3', 'CNN-33106-5', 'CNN-33106-7',
                 'CNN-33106-9', 'CNN-33106-11', 'CNN-33106-13', 'CNN-33106-15', 'CNN-33106-17', 'CNN-8741-3',
                 'CNN-8741-5', 'CNN-254960-3', 'CNN-353687-3', 'NPR-9330-4', 'NPR-9330-6', 'NPR-9330-8', 'NPR-9330-11',
                 'NPR-9330-15', 'NPR-17647-6', 'NPR-17647-8', 'NPR-17647-12', 'NPR-17647-15', 'NPR-11972-4',
                 'NPR-11972-8', 'NPR-11972-10', 'CNN-42690-3', 'CNN-42690-5', 'CNN-39943-5', 'CNN-39943-7',
                 'CNN-13829-3', 'CNN-13829-5', 'CNN-13829-7', 'CNN-67148-3', 'CNN-67148-11', 'CNN-67148-13',
                 'CNN-67148-15', 'CNN-45461-3', 'CNN-45461-5', 'CNN-209089-3', 'CNN-209089-5', 'CNN-68990-3',
                 'CNN-68990-5', 'CNN-68990-7', 'CNN-80346-3', 'CNN-80346-5', 'CNN-80346-9', 'CNN-80346-11',
                 'CNN-224826-3', 'CNN-224826-5', 'CNN-246366-3', 'CNN-246366-5', 'NPR-17441-5', 'NPR-17441-11',
                 'NPR-17441-15', 'NPR-17441-17', 'NPR-17441-19', 'NPR-17441-21', 'NPR-17441-23', 'NPR-17441-25',
                 'NPR-17441-27', 'CNN-254170-5', 'CNN-254170-7', 'CNN-254170-11', 'CNN-254170-15', 'CNN-254170-17',
                 'CNN-267046-5', 'CNN-267046-7', 'CNN-267046-9', 'CNN-267046-11', 'CNN-88844-3', 'CNN-88844-5',
                 'CNN-88844-9', 'NPR-3411-4', 'NPR-3411-6', 'NPR-3411-10', 'NPR-3411-13', 'NPR-3411-15', 'CNN-368821-3',
                 'CNN-368821-5', 'CNN-368821-7', 'CNN-368821-11', 'CNN-22192-3', 'CNN-22192-9', 'CNN-22192-13',
                 'CNN-380194-3', 'CNN-380194-5', 'CNN-380194-7', 'CNN-380194-9', 'CNN-380194-11', 'CNN-380194-13',
                 'CNN-380194-15', 'CNN-380194-17']


def get_highlight_choices(q_id: str, pd_report: pd.DataFrame, category: str, annotator: str) -> List[int]:
    assert_valid_category(category)
    result = pd_report.loc[(pd_report[AnnotationColumns.Annotator_ID] == annotator)
                           & (pd_report[AnnotationColumns.Question_ID] == q_id)
                           & (pd_report[AnnotationColumns.Highlight_Category] == category)
                           ][AnnotationColumns.Highlight_Choices]
    return ast.literal_eval(result.iloc[0])


def change_report_id(report_id: str, start_nbr: int):
    if "QID" in report_id:
        report_id = (start_nbr - 1) + math.floor((int(report_id[3:]) - 15) / 7)
    if "Q" in report_id:
        report_id = (start_nbr - 1) + math.floor((int(report_id[1:]) - 15) / 7)

    return FIRST_100_IDs[report_id]


def get_freq(transformed_df, paraphrase_candidate: bool = False, content_reproduction: bool = True):
    q_ids = transformed_df[AnnotationColumns.Question_ID].unique()
    i_ids = [q_id[:q_id.rfind('-')] for q_id in q_ids if "R_" not in q_id]

    if paraphrase_candidate:
        annotators = transformed_df[AnnotationColumns.Annotator_ID].unique()

        values = [0 if yn == "No" else 1 for yn in transformed_df[
            transformed_df[AnnotationColumns.Highlight_Category] == ContentReproductionCategories.Referring_Choice
            ][AnnotationColumns.Highlight_Choices]]

        return {"Interviews": len(set(i_ids)),
                "# Annotated Paraphrase Pairs": len(i_ids),
                f"AVG # PC candidates absolute": sum(values) / len(annotators),
                f"AVG share #PC candidates": [sum(values) / len(values), statistics.stdev(values)],
                f"#PC candidates per annotator": values
                }

    abs_ref, rel_ref = calc_freq_cat(AllAnnotationCategories.Referred_to, transformed_df)

    if not content_reproduction:
        abs_reph, rel_reph = calc_freq_cat(AllAnnotationCategories.Rephrase, transformed_df)
        abs_inter, rel_inter = calc_freq_cat(AllAnnotationCategories.Interpretation, transformed_df)
        abs_info, rel_info = calc_freq_cat(AllAnnotationCategories.Elaboration_Information, transformed_df)
        abs_q, rel_q = calc_freq_cat(AllAnnotationCategories.Elaboration_Question, transformed_df)

        return {"Interviews": len(set(i_ids)),
                AllAnnotationCategories.Referred_to: rel_ref,
                AllAnnotationCategories.Referred_to + " absolute": abs_ref,
                AllAnnotationCategories.Rephrase: rel_reph,
                AllAnnotationCategories.Rephrase + " absolute": abs_reph,
                AllAnnotationCategories.Interpretation: rel_inter,
                AllAnnotationCategories.Interpretation + " absolute": abs_inter,
                AllAnnotationCategories.Elaboration_Information: rel_info,
                AllAnnotationCategories.Elaboration_Information + " absolute": abs_info,
                AllAnnotationCategories.Elaboration_Question: rel_q,
                AllAnnotationCategories.Elaboration_Question + " absolute": abs_q}

    else:
        abs_rep, rel_rep = calc_freq_cat(ContentReproductionCategories.Repeat, transformed_df)
        abs_re, rel_re = calc_freq_cat(ContentReproductionCategories.Reference, transformed_df)
        abs_para, rel_para = calc_freq_cat(ContentReproductionCategories.Paraphrase, transformed_df)

        return {"Interviews": len(set(i_ids)),
                AllAnnotationCategories.Referred_to: rel_ref,
                AllAnnotationCategories.Referred_to + " absolute": abs_ref,
                ContentReproductionCategories.Repeat: rel_rep,
                ContentReproductionCategories.Repeat + " absolute": abs_rep,
                ContentReproductionCategories.Reference: rel_re,
                ContentReproductionCategories.Reference + " absolute": abs_re,
                ContentReproductionCategories.Paraphrase: rel_para,
                ContentReproductionCategories.Paraphrase + " absolute": abs_para}


def calc_freq_cat(ref_cat, transformed_df, content_reproduction: bool = True):
    """

    :param ref_cat:
    :param transformed_df:
    :return:
    """
    assert_valid_category(ref_cat, content_reproduction)

    ref_choices = transformed_df[
        transformed_df[AnnotationColumns.Highlight_Category] == ref_cat
        ]

    # assuming at least as many sessions as annotators
    annotators = transformed_df[AnnotationColumns.Annotator_ID].unique()
    assert (len(annotators) <= len(transformed_df[AnnotationColumns.Annotation_Session_ID].unique()))

    abs_freqs = []
    max_freq = len(ref_choices.index) / len(annotators)
    for annotator in annotators:
        count = ref_choices[
            ref_choices[AnnotationColumns.Annotator_ID] == annotator
            ][AnnotationColumns.Highlight_Choices].value_counts()
        abs_freq = max_freq
        if str([]) in count.keys():
            abs_freq = (max_freq - count[str([])])
        abs_freqs.append(abs_freq)

    return np.array(abs_freqs).mean(), np.array(abs_freqs).mean() / max_freq


def calc_classification_agreement(transformed_df, paraphrase_candidate=False, include_history: bool = True,
                                  content_reproduction: bool = True, min_overlap=1):
    co_ref = calc_agreement_cat(transformed_df, min_overlap=min_overlap, paraphrase_candidate=paraphrase_candidate)
    if paraphrase_candidate:
        return {AllAnnotationCategories.Referred_to: co_ref
                }
    else:
        # co_para = calc_agreement_cat(ContentReproductionCategories.Paraphrase, transformed_df, False)
        return {AllAnnotationCategories.Referred_to: co_ref,
                ContentReproductionCategories.Paraphrase: co_ref,
                }


def calc_mv_convergence(highlight_df: pd.DataFrame, set_max: int = 0, shuffle_times: bool = False,
                        fixed_order=None) -> np.array:
    """

    :param highlight_df: dataframe of highlighting studies that comes from a merge_transform call
    :return:
    """

    binary_dict = get_annotation_dict(highlight_df)
    annotator_order = binary_dict["annotators"]
    q_order = binary_dict["q_ids"]
    binary_annotation_matrix = np.array(binary_dict["complete_binaries"])

    # sort according to completion times
    reorder_indices = []
    for q_index, q_id in enumerate(q_order):
        ann_indices = get_ordered_annotator_indices(q_id, annotator_order, highlight_df, shuffle_times=shuffle_times,
                                                    fixed_oder=fixed_order)
        binary_annotation_matrix[:, q_index] = binary_annotation_matrix[ann_indices, q_index]
        reorder_indices.append(ann_indices)

    if set_max > 0:
        binary_annotation_matrix[set_max:, :] = np.nan
    ranks = np.tile(np.arange(1, len(annotator_order) + 1)[:, np.newaxis], (1, len(q_order)))
    cumulative_matrix = get_cumulative_matrix(binary_annotation_matrix)
    non_nan_count = np.count_nonzero(~np.isnan(binary_annotation_matrix), axis=0)
    relative_cum_matrix = cumulative_matrix / ranks
    relative_cum_matrix[np.argsort(binary_annotation_matrix, axis=0).argsort(axis=0) + 1 > non_nan_count] = np.nan

    return relative_cum_matrix, cumulative_matrix, ranks, annotator_order, reorder_indices, q_order


def get_ordered_annotator_indices(q_id, annotator_order, highlight_df, shuffle_times: bool = False, fixed_oder=None):
    """
        for a given question, get the order of annotators according to handing in times
    :param q_id:
    :param annotator_order:
    :param highlight_df:
    :param shuffle_times:
    :return:
    """
    # get all sessions in which this question was annotated
    q_sns = highlight_df[
        highlight_df[AnnotationColumns.Question_ID] == q_id][AnnotationColumns.Annotation_Session_ID].unique()

    # sort according to End Times
    sorted_sessions = highlight_df[highlight_df[AnnotationColumns.Question_ID].isin(q_sns) &
                                   (highlight_df[
                                        AnnotationColumns.Highlight_Category] == AllAnnotationCategories.EndTime)
                                   ].sort_values(AnnotationColumns.Highlight_Choices)
    sorted_annotators = sorted_sessions[AnnotationColumns.Annotator_ID].values.tolist()
    if fixed_oder is not None:
        ann_indices = [annotator_order.index(ann) for ann in fixed_oder if ann in sorted_annotators]
    else:
        ann_indices = [annotator_order.index(ann) for ann in sorted_annotators]
    if shuffle_times:
        random.shuffle(ann_indices)
    # append missing annotators at the end
    ann_indices = ann_indices + [ann for ann in range(len(annotator_order)) if ann not in ann_indices]
    return ann_indices


def calc_agreement_cat(transformed_df, min_overlap=1, paraphrase_candidate=False):
    """
    :param transformed_df:
    :return:
    """
    # Reset index to avoid duplicate axis error
    transformed_df = transformed_df.reset_index(drop=True)

    # assert (ref_cat in [ContentReproductionCategories.Paraphrase, ContentReproductionCategories.Referred_to])
    annotation_dict = get_annotation_dict(transformed_df, paraphrase_candidate=paraphrase_candidate)

    # calculate average Cohen on fixed number of annotators per item
    print(f"For fixed number annotators ...")
    constant_nbr_per_item = 3  # min(len(sublist) for sublist in ordered_classifications)
    ordered_classifications = annotation_dict["temporally_ordered_classifications"]
    cut_ordered_classifications = [sublist[:constant_nbr_per_item] for sublist in ordered_classifications]
    print(f"Fleiss (for first {constant_nbr_per_item} raters): "
          f"{fleiss_kappa_agreement(np.array(cut_ordered_classifications).transpose())}")

    print(f"Calculating agreements for first {constant_nbr_per_item} annotations ... ")
    #   create a 3-15 subselection of the dataframe
    ordered_q_ids = annotation_dict["temporally_ordered_q_ids"]
    ordered_sessions = annotation_dict["temporally_ordered_sessions"]
    #       for each question, get the first constant_nbr_per_item sessions
    cut_ordered_sessions = [sublist[:constant_nbr_per_item]
                            for sublist in ordered_sessions]
    condition = create_condition(transformed_df, AnnotationColumns.Question_ID, AnnotationColumns.Annotation_Session_ID,
                                 ordered_q_ids, cut_ordered_sessions)
    # cut_ordered_sessions = [item for sublist in cut_ordered_sessions for item in sublist]
    _calc_agreement(transformed_df[condition],
                    min_overlap=min_overlap)

    print(f"Calculating agreements for simulated 3-15 dynamic recruitment, threshold of 0.8 ... ")
    entropy_ordered_sessions = []
    for cur_class, cur_sessions in zip(ordered_classifications, ordered_sessions):
        added = False
        for i in range(3, len(cur_class)):
            if get_min_nbr_anns_entropy(cur_class[:i].count(1) / i, i) > 0:
                continue
            else:
                entropy_ordered_sessions.append(cur_sessions[:i])
                added = True
                break
        if not added:
            entropy_ordered_sessions.append(cur_sessions[:len(cur_class)])
    # entropy_ordered_sessions = [item for sublist in entropy_ordered_sessions for item in sublist]
    condition = create_condition(transformed_df, AnnotationColumns.Question_ID, AnnotationColumns.Annotation_Session_ID,
                                 ordered_q_ids, entropy_ordered_sessions)
    _calc_agreement(
        transformed_df[condition],
        min_overlap=min_overlap)

    print(f"Calculating on complete data ...")
    _calc_agreement(transformed_df, min_overlap=min_overlap)


def _calc_agreement(transformed_df, min_overlap=1):
    # legacy calculations
    binary_annotations = get_annotation_dict(transformed_df)[
        "complete_binaries"]
    # bin_np = np.array(binary_annotations)

    # column_sums = np.count_nonzero(~np.isnan(bin_np), axis=0)
    # if len(set(column_sums)) > 1:
    #     # randomly replace with nan values until the same (if # raters not the same across items)
    #     lowest_raters = min(set(column_sums))
    #     for i, n_raters in enumerate(column_sums):
    #         if n_raters > lowest_raters:
    #             n_rand_remove = n_raters - lowest_raters
    #             # Get the valid indices of non-NaN values in the column
    #             valid_indices = np.where(~np.isnan(bin_np[:, i]))[0]
    #             # Randomly select n_rand_remove indices from the valid indices
    #             random_indices = np.random.choice(valid_indices, size=n_rand_remove, replace=False)
    #             # Replace values with NaN at the selected indices
    #             bin_np[random_indices, i] = np.nan
    # fleiss_agreement = fleiss_kappa_agreement(bin_np)
    # print(f"\t Fleiss (assumes same # raters and technically same raters): {fleiss_agreement}")

    # print("Calculating average Cohen")
    bin_np = np.array(binary_annotations)
    print(f"\t Krippendorff's alpha: {calc_krippendorff_alpha(bin_np)}")
    cohens = calculate_pairwise_cohen(bin_np, min_overlap=min_overlap)
    print(f"\t annotator pairs: {len(cohens)}, average cohen: {mean(cohens)} with STD {np.std(cohens)}, "
          f"median cohen: {statistics.median(cohens)}, # below 0: {len([c for c in cohens if c < 0])}, "
          f"# above 0.2: {len([c for c in cohens if c > 0.2])}")
    print(f"\t percentual agreement over every pair of raters:{calculate_pairwise_overlap_between_rows(bin_np)}")
    print(f"\t average agreement across items: {calculate_avg_item_overlap(bin_np)}")

    print(f"\t mean accuracy with majority vote: {calculate_mv_overlap_between_rows(bin_np)[0]}")

    cohens_per_study = calculate_pairwise_cohen(bin_np, per_study=True)
    print(
        f"\t annotator pairs for per study calculate: {len(cohens_per_study)}, average per study cohen: {mean(cohens_per_study)} with STD {np.std(cohens_per_study)}")
    mv_cohens = calculate_mv_cohen(bin_np)
    print(f"\t For {len(mv_cohens)} annotators, average MV cohen: {mean(mv_cohens)} with STD {np.std(mv_cohens)}")
    mv_cohens = calculate_mv_cohen(bin_np, exclude_current_rater=False)
    print(
        f"\t For {len(mv_cohens)} annotators, average MV cohen (INCLUDING RATER): {mean(mv_cohens)} with STD {np.std(mv_cohens)}")


def calculate_pairwise_overlap_between_rows(bin_np):
    # get overlap between annotators
    overlaps = 0
    n = 0
    for i in range(len(bin_np)):  # for each annotator
        for j in range(i + 1, len(bin_np)):
            overlap, non_nan_len = overlap_share(bin_np[i], bin_np[j])
            if not np.isnan(overlap):
                n += 1
                overlaps += overlap
    return overlaps / n


def calculate_avg_item_overlap(bin_np):
    agreements = []
    for i in range(bin_np.shape[1]):  # for each item
        # get the share of 1s in that column
        non_nan_len = np.count_nonzero(~np.isnan(bin_np[:, i]))
        ones = np.count_nonzero(bin_np[:, i] == 1)
        agreement = max(ones / non_nan_len, 1 - ones / non_nan_len)
        agreements.append(agreement)
    return mean(agreements), std(agreements)


def calculate_mv_overlap_between_rows(bin_np, excluding_rater=True):
    """

    :param excluding_rater: whether current rater is used to calculate the mv
    :param bin_np: ANNS x ITEMS
    :return:
    """
    accs = []
    for ann in range(bin_np.shape[0]):
        overlap_w_mv = 0
        for item in np.where(~np.isnan(bin_np[ann, :]))[0]:
            # if np.isnan(bin_np[ann, item]):  # not considered if annotator didn't annotate this item
            #     continue
            ann_vote = int(bin_np[ann, item])
            # get number non-non entries in this column
            non_nan_indices = np.where(~np.isnan(bin_np[:, item]))[0]
            para_votes = (sum(bin_np[non_nan_indices, item]) -
                          (ann_vote if excluding_rater else 0))
            total_votes = len(non_nan_indices) - (1 if excluding_rater else 0)
            assert not np.isnan(para_votes), "Value is NaN"
            assert not np.isnan(total_votes), "Value is NaN"
            # majority vote for item
            mv = 1 if para_votes / total_votes >= 0.5 else 0
            # compare with annotator vote
            overlap_w_mv += 1 if mv == ann_vote else 0
        accs.append(overlap_w_mv / len(np.where(~np.isnan(bin_np[ann, :]))[0]))
    return mean(accs), std(accs)


def overlap_share(arr1, arr2):
    # Ensure input are numpy arrays
    arr1, arr2 = np.array(arr1), np.array(arr2)

    # Identify indices where both arrays do not have NaN values
    non_nan_indices = np.where(~np.isnan(arr1) & ~np.isnan(arr2))

    if len(non_nan_indices[0]) == 0:
        return np.nan, 0

    # Extract non-NaN values from both arrays
    arr1_non_nan = arr1[non_nan_indices]
    arr2_non_nan = arr2[non_nan_indices]

    # Calculate the number of positions where the values are the same
    same_values_count = np.sum(arr1_non_nan == arr2_non_nan)

    # Calculate the share of overlap
    overlap_share = same_values_count / len(arr1_non_nan)

    return overlap_share, len(arr1_non_nan)


def calculate_pairwise_cohen(bin_np, per_study=False, min_overlap=30):
    """
        calculate all pairwise cohen values
    :param bin_np:
    :param per_study: whether pair based on annotators (default) or per study,
        i.e., treating the annotators for each study as separate as if they are different ppl
        ASSUMES a study is represented by 10 columns in the right order of bin_np
    :return:
    """
    cohens = []
    if not per_study:

        cohen_matrix = _get_cohen_matrix(bin_np, min_overlap)

        cohens = list(val for val in chain.from_iterable(cohen_matrix) if not np.isnan(val))
    else:
        for study in range(0, bin_np.shape[1], 10):
            cur_bins = bin_np[:, study:study + 10]
            cohens += calculate_pairwise_cohen(cur_bins, per_study=False)
    return cohens


def _get_cohen_matrix(bin_np, min_overlap, replace_nan=False):
    # TODO: why is this not symmetrical?
    nbr_annotators = len(bin_np)
    if replace_nan:
        cohen_matrix = np.zeros((nbr_annotators, nbr_annotators))
    else:
        cohen_matrix = np.full((nbr_annotators, nbr_annotators), np.nan)
    for i in range(nbr_annotators):
        for j in range(i + 1, nbr_annotators):
            combined_anns = bin_np[[i, j], :]
            combined_anns = combined_anns[:, ~np.any(np.isnan(combined_anns), axis=0)]
            if len(combined_anns.flatten()) >= min_overlap:
                cur_kappa = cohen_kappa_agreement(list(combined_anns[0]), list(combined_anns[1]))
                cohen_matrix[i, j] = cur_kappa
                cohen_matrix[j, i] = cur_kappa
    return cohen_matrix


def calculate_mv_cohen(bin_np, exclude_current_rater=True):
    if exclude_current_rater:
        print(f"\t Calculating Cohen with MV (excluding the current rater)")
    else:
        print(f"\t ATTENTION: Cohen with MV (INCLUDING the current rater)")
    mv_cohens = []
    for i in range(bin_np.shape[0]):
        anns_i = bin_np[i, :]
        # get indices where entry is not nan
        nan_mask = ~np.isnan(anns_i)
        non_nan_indices = np.where(nan_mask)[0]
        if exclude_current_rater:
            anns_without_i = np.delete(bin_np, i, axis=0)
        else:
            anns_without_i = bin_np
        mv_without_i = np.round(np.nanmean(anns_without_i, axis=0)).astype(int)

        mv_cohens.append(cohen_kappa_agreement(
            list(anns_i[non_nan_indices]), list(mv_without_i[non_nan_indices])))
    return mv_cohens


def calc_an_perf_onlist(transformed_df, is_prphrs_lst, ref_cat=ContentReproductionCategories.Paraphrase):
    """
        Assumes that there is a ground truth of paraphrase annotations given with is_prphrs_lst
            --> later on this could also be majority votes
    :param transformed_df:
    :param is_prphrs_lst:
    :param ref_cat:
    :return:
    """
    #   get binary predictions for the given category
    rslt_dct = get_annotation_dict(transformed_df, False)
    #   first 10 are assumed to be the "simple" cases
    simple_qids = list(is_prphrs_lst.keys())[:10]
    si_gt = [is_prphrs_lst[q_id] for q_id in rslt_dct["q_ids"] if q_id in simple_qids]
    #   second 10 are assumed to be the "difficult" cases
    difficult_qids = list(is_prphrs_lst.keys())[10:20]
    di_gt = [is_prphrs_lst[q_id] for q_id in rslt_dct["q_ids"] if q_id in difficult_qids]
    #   annotator names
    annotators = rslt_dct["annotators"]
    #   gt in the right order as the participants saw it
    gt = [is_prphrs_lst[q_id] for q_id in rslt_dct["q_ids"]]

    # calculating results
    result = dict()
    result["GT"] = gt
    result["q_ids"] = rslt_dct["q_ids"]
    si_binaries = []
    di_binaries = []
    for i, an in enumerate(annotators):
        # F1-scores
        result[f"{an}-binaries"] = rslt_dct["binaries"][i]
        cur_f1 = sklearn.metrics.f1_score(gt, rslt_dct["binaries"][i])
        result[f"{an}-f1"] = cur_f1
        si_pred = [bin for bin, q_id in zip(rslt_dct["binaries"][i], rslt_dct["q_ids"]) if q_id in simple_qids]
        si_binaries.append(si_pred)
        si_f1 = sklearn.metrics.f1_score(si_gt, si_pred)
        result[f"{an}-simple-f1"] = si_f1
        di_pred = [bin for bin, q_id in zip(rslt_dct["binaries"][i], rslt_dct["q_ids"]) if q_id in difficult_qids]
        di_binaries.append(di_pred)
        di_f1 = sklearn.metrics.f1_score(di_gt, di_pred)
        result[f"{an}-difficult-f1"] = di_f1

        # agreement with GT
        result[f"{an}-agreement-GT"] = cohen_kappa_agreement(gt, rslt_dct["binaries"][i])
        result[f"{an}-si-agreement-GT"] = cohen_kappa_agreement(si_gt, si_pred)
        result[f"{an}-di-agreement-GT"] = cohen_kappa_agreement(di_gt, di_pred)

    # if len(si_binaries) == 2:
    #     result[f"cohen-simple"] = cohen_kappa_agreement(*si_binaries)
    #     result[f"cohen-difficult"] = cohen_kappa_agreement(*di_binaries)
    #     result[f"cohen-check"] = cohen_kappa_agreement(*rslt_dct["binaries"])
    # else:
    result[f"fleiss-simple"] = fleiss_kappa_agreement(si_binaries)
    result[f"fleiss-difficult"] = fleiss_kappa_agreement(di_binaries)
    result[f"fleiss-check"] = fleiss_kappa_agreement(rslt_dct["binaries"])
    result[f"pw-cohen-simple"] = (mean(calculate_pairwise_cohen(np.array(si_binaries))),
                                  np.std(calculate_pairwise_cohen(np.array(si_binaries))))
    result[f"pw-cohen-difficult"] = (mean(calculate_pairwise_cohen(np.array(di_binaries))),
                                     np.std(calculate_pairwise_cohen(np.array(di_binaries))))
    result[f"mv-cohen-simple"] = (mean(calculate_mv_cohen(np.array(si_binaries))),
                                  np.std(calculate_mv_cohen(np.array(si_binaries))))
    result[f"mv-cohen-difficult"] = (mean(calculate_mv_cohen(np.array(di_binaries))),
                                     np.std(calculate_mv_cohen(np.array(di_binaries))))
    result[f"mv-cohen-simple-inc"] = (mean(calculate_mv_cohen(np.array(si_binaries), exclude_current_rater=False)),
                                      np.std(calculate_mv_cohen(np.array(si_binaries), exclude_current_rater=False)))
    result[f"mv-cohen-difficult-inc"] = (mean(calculate_mv_cohen(np.array(di_binaries), exclude_current_rater=False)),
                                         np.std(calculate_mv_cohen(np.array(di_binaries), exclude_current_rater=False)))

    return result


def _print_cat_hl_examples(cats: List[str], transformed_df: pd.DataFrame,
                           interview: interview_data.MediaSumProcessor, disagreements=True, agreements=False,
                           chunked=False, content_reproduction: bool = True):
    """

    :param cats: categories to consider; ordering matters currently only accepts one
    :param transformed_df:
    :param interview:
    :param disagreements:
    :param agreements:
    :param chunked:
    :return:
    """
    cat = cats[0]
    on_guest = cat == AllAnnotationCategories.Referred_to
    if not on_guest:
        cats.append(AllAnnotationCategories.Referred_to)
    binary_annotations = np.array(get_annotation_dict(transformed_df, False)["binaries"])
    highlight_choices = transformed_df[
        transformed_df[AnnotationColumns.Highlight_Category].isin(cats)
    ]
    q_ids = highlight_choices[AnnotationColumns.Question_ID].unique().tolist()
    annotators = get_unique_annotators(highlight_choices)
    nbr_dis = 0
    nbr_agr = 0
    for q_id, an_group in zip(q_ids, binary_annotations.transpose()):
        if (len(set(an_group)) > 1 and disagreements) or (len(set(an_group)) == 1 and agreements and 0 not in an_group):
            all_hl_choices = highlight_choices[highlight_choices[
                                                   AnnotationColumns.Question_ID] == q_id]  # [AnnotationColumns.Highlight_Choices].tolist()
            guest_utt, host_utt, summary = interview.get_qdict_from_qid(q_id)
            utt = host_utt

            if on_guest:
                utt = guest_utt
            else:
                guest_choices = qualtrics_survey.tokenize_for_highlight_choices(guest_utt)
            choice_to_text = qualtrics_survey.tokenize_for_highlight_choices(utt)

            print("==============================")
            if len(set(an_group)) > 1:
                nbr_dis += 1
                print("The {0}. disagreement in the category {1} was found on the pair {2}:".format(nbr_dis, cat, q_id))
            else:
                nbr_agr += 1
                print("The {0}. agreement in the category {1} was found on the pair {2}:".format(nbr_agr, cat, q_id))
            # print(guest)
            print(f"Summary: {summary}")
            print("Original Text: \n \tGuest: {0} \n \tHost: {1}".format(guest_utt.replace("\n", ""),
                                                                         host_utt.replace("\n", "")))
            print(" ---------------- ")
            for an in annotators:
                print(f"  For annotator: {an}")
                hl_choices = all_hl_choices[all_hl_choices[AnnotationColumns.Annotator_ID] == an]
                host_hls = hl_choices[
                    hl_choices[AnnotationColumns.Highlight_Category] != AllAnnotationCategories.Referred_to][
                    AnnotationColumns.Highlight_Choices].tolist()[0]
                if not on_guest:
                    print("\tGuest: ", end="")
                    guest_hls = hl_choices[
                        hl_choices[AnnotationColumns.Highlight_Category] == AllAnnotationCategories.Referred_to][
                        AnnotationColumns.Highlight_Choices].tolist()[0]  # this should only ever include one
                    if len(ast.literal_eval(guest_hls)) > 0:
                        highlighted_guest = " ".join([guest_choices[i].replace("&nbsp;", " ").upper()
                                                      if i + 1 in ast.literal_eval(guest_hls)
                                                      else guest_choices[i].replace("&nbsp;", " ")
                                                      for i in range(len(guest_choices))])
                    elif len(ast.literal_eval(host_hls)) == 0:
                        highlighted_guest = "None"
                    else:
                        highlighted_guest = "HISTORY"
                    print(highlighted_guest)
                    print("\tHost: ", end="")
                else:
                    print("\tGuest: ", end="")
                highlighted_utt = " ".join([choice_to_text[i].replace("&nbsp;", " ").upper()
                                            if i + 1 in ast.literal_eval(host_hls)
                                            else choice_to_text[i].replace("&nbsp;", " ")
                                            for i in range(len(choice_to_text))])
                print(highlighted_utt)
                if on_guest:
                    print("\tHost: SAME".format(host_utt))


def print_agreement_examples(cat: str, transformed_df: pd.DataFrame,
                             interview: interview_data.MediaSumProcessor, chunked=False):
    _print_cat_hl_examples([cat], transformed_df, interview, agreements=True, disagreements=False, chunked=chunked)


def print_disagreement_examples(cat: str, transformed_df: pd.DataFrame,
                                interview: interview_data.MediaSumProcessor, chunked=False,
                                content_reproduction: bool = True):
    """
    :param cat:
    :param transformed_df:
    :param interview:
    :param chunked:
    :return:
    """
    _print_cat_hl_examples([cat], transformed_df, interview, agreements=False,
                           disagreements=True, chunked=chunked, content_reproduction=content_reproduction)


def print_example_cat(cats: List[str], transformed_df: pd.DataFrame,
                      interview: interview_data.MediaSumProcessor, chunked=False):
    highlight_choices = transformed_df[
        (transformed_df[AnnotationColumns.Highlight_Choices] != '[]') &
        transformed_df[AnnotationColumns.Highlight_Category].isin(cats)
        ]
    q_ids = highlight_choices[AnnotationColumns.Question_ID].unique()
    q_ids = [q_id for q_id in q_ids if "R_" not in q_id]
    annotators = highlight_choices[AnnotationColumns.Annotator_ID].unique()
    nbr_ex = 0
    for q_id in q_ids:
        cur_hl = highlight_choices[highlight_choices[AnnotationColumns.Question_ID] == q_id]
        for ann in annotators:
            cur_hl = cur_hl[cur_hl[AnnotationColumns.Annotator_ID] == ann]
            if len(cur_hl.index) == len(cats):
                print("For annotator {}:".format(ann))
                guest_utt, host_utt, _ = interview.get_qdict_from_qid(q_id)
                choice_to_host = qualtrics_survey.tokenize_for_highlight_choices(host_utt)
                choice_to_guest = qualtrics_survey.tokenize_for_highlight_choices(guest_utt)
                print("==============================")
                print("Original Text: \n \tGuest: {0} \n \tHost: {1}".format(guest_utt.replace("\n", ""),
                                                                             host_utt.replace("\n", "")))

                cat_ordered = cur_hl[cur_hl[AnnotationColumns.Question_ID] == q_id][
                    AnnotationColumns.Highlight_Category].tolist()
                assert (set(cat_ordered) == set(cats))
                all_hl_choices = cur_hl[cur_hl[AnnotationColumns.Question_ID] == q_id][
                    AnnotationColumns.Highlight_Choices].tolist()

                print(" ---------------- ")
                for cat, hl_choices in zip(cat_ordered, all_hl_choices):
                    if AllAnnotationCategories.Referred_to == cat:
                        print("\tGuest: ", end="")
                        highlighted_utt = " ".join([choice_to_guest[i].replace("&nbsp;", " ").upper()
                                                    if i + 1 in ast.literal_eval(hl_choices)
                                                    else choice_to_guest[i].replace("&nbsp;", " ")
                                                    for i in range(len(choice_to_guest))])
                    else:
                        print("\tHost: ", end="")
                        highlighted_utt = " ".join([choice_to_host[i].replace("&nbsp;", " ").upper()
                                                    if i + 1 in ast.literal_eval(hl_choices)
                                                    else choice_to_host[i].replace("&nbsp;", " ")
                                                    for i in range(len(choice_to_host))])
                    print(highlighted_utt)


def print_latex_for_q(higlight_df, q_id, interview):
    """
        For a given question, print the latex code for the highlighting
    :param higlight_df:
    :param q_id: question ID to print
    :param interview: data including the interview texts
    :return:
    """
    if type(q_id) == str:
        q_id = [q_id]
    filtered_df = higlight_df[higlight_df[AnnotationColumns.Question_ID].isin(q_id)]
    _print_latex_guest_host_highlights(filtered_df, interview)


def _print_latex_guest_host_highlights(highlight_df, interview=None):
    """
    :param highlight_df:
    :param ms_path:
    :return:
    """
    guest_msgs, guest_tokens, host_msgs, host_tokens, q_ids, vote_strs = _get_annotations_per_qid(highlight_df,
                                                                                                  interview)

    for q_id, votes, guest_weights, guest_toks, host_weights, host_toks in zip(q_ids, vote_strs, guest_msgs,
                                                                               guest_tokens, host_msgs,
                                                                               host_tokens):
        if q_id == "CNN-235909-21":
            print()
        tex_str = gen_guest_host_tex([guest_toks, host_toks], [guest_weights, host_weights])
        print()
        print(r'''%''' + q_id + r'''\\''')
        print(r'''\begin{center} \textbf{''' + votes +
              r'''} annotators highlight a paraphrase \end{center} \vspace{-0.5\baselineskip}''' + "\n")
        print(tex_str)


def calc_str_agreements_cat(report_df: pandas.DataFrame, ref_cat: str, interview: interview_data.MediaSumProcessor,
                            chunked=False) -> Dict[str, Any]:
    candidates, _, _, _ = extract_txts_to_compare(ref_cat, report_df, interview)
    # only consider cases where at least two different annotators annotated
    candidates = [c_list for c_list in candidates if len(c_list) > 1]

    return {
        "mean Semantic Similarity": transformer_scores(candidates)[0],
        # "mean BERTScore": transformer_scores(candidates, bert_score=True)[0]
    }


def calc_overlap(report_df: pandas.DataFrame, ref_cat: str, interview=None):
    """
        calls Overlap
    :param interview:
    :param report_df:
    :param ref_cat:
    :return:
    """
    if interview is None:
        interview = interview_data.MediaSumProcessor()

    assert_valid_category(ref_cat)

    choices_df = report_df[
        report_df[AnnotationColumns.Highlight_Category] == ref_cat
        ]
    annotators = get_unique_annotators(report_df)

    jaccard_ov = calc_jaccard_overlap_from_choices(annotators, choices_df)

    # calculate unitizing Alpha
    # - implementation: creates one item / annotator matrix as usual for Krippendorff alpha calculation,
    #   so the "special" step is in creating the matrix:
    #       + for each question it is not just one column added but
    #           as many columns added as there are tokens in the respective host/guest message
    #           one question:
    #               [NaN... NaN] for ann1
    #               [0 0 0... 1] for ann2
    #               [1 1 0... 0] for ann3
    #               ...
    data_matrix = [[] for _ in range(len(annotators))]

    q_ids = choices_df[AnnotationColumns.Question_ID].unique()
    len_utts = []
    len_hl = []
    for i, q_id in enumerate(q_ids):
        if ref_cat == ContentReproductionCategories.Referred_to:
            cur_utt = interview.get_qdict_from_qid(q_id)["guest utterance"]
        else:
            cur_utt = interview.get_qdict_from_qid(q_id)["host utterance"]
        len_utt = len(cur_utt.split())
        len_utts.append(len_utt)
        for j, ann in enumerate(annotators):
            hl = [np.nan] * len_utt
            # get highlights from this annotator for question q_id
            hl_indices = choices_df[
                (choices_df[AnnotationColumns.Question_ID] == q_id) &
                (choices_df[AnnotationColumns.Annotator_ID] == ann)][AnnotationColumns.Highlight_Choices]
            assert ((len(hl_indices) == 1) or (len(hl_indices) == 0))
            if len(hl_indices) == 1 and len(ast.literal_eval(hl_indices.values[0])) > 0:
                hl_indices = ast.literal_eval(hl_indices.values[0])
                len_hl.append(len(hl_indices))
                hl = [1 if pos + 1 in hl_indices else 0 for pos in range(len_utt)]
            data_matrix[j] += hl
    try:
        krippendorff = calc_krippendorff_alpha(np.array(data_matrix))
    except AssertionError:
        # here has to be more than one value in the domain.
        krippendorff = np.nan

    return {
        "Mean Jaccard Overlap": jaccard_ov,
        "Krippendorff Alpha": krippendorff,
        "AVG len": statistics.mean(len_utts),
        "AVG hl len": statistics.mean(len_hl)
    }


def calc_jaccard_overlap_from_choices(annotators, choices_df):
    highlights_per_q_per_annotator = choices_df.groupby(
        [AnnotationColumns.Question_ID])[AnnotationColumns.Highlight_Choices].apply(list) \
        .apply(lambda x: [ast.literal_eval(elem) for elem in x]).to_numpy()
    if len(annotators) == 2:
        jaccard_ov = jaccard_overlap_for_highlights(highlights_per_q_per_annotator)
    else:
        jaccard_values = []
        for l_pointer in range(len(annotators) - 1):
            for r_pointer in range(l_pointer + 1, len(annotators)):
                try:
                    hl_values = [[elem[l_pointer], elem[r_pointer]] for elem in highlights_per_q_per_annotator]
                    jaccard_values.append(jaccard_overlap_for_highlights(hl_values))
                except IndexError:  # happens because not always 20 annotators
                    continue
        jaccard_ov = statistics.mean(jaccard_values)
    return jaccard_ov


def get_unique_annotators(report_df):
    annotators = report_df[AnnotationColumns.Annotator_ID].unique()
    # assert (len(annotators) == len(report_df[AnnotationColumns.Annotation_Session_ID].unique()))
    return annotators


def calc_overlaps(transformed_df, content_reproduction: bool = True, interview=None):
    bs_ref = calc_overlap(transformed_df, AllAnnotationCategories.Referred_to, interview=interview)

    if not content_reproduction:
        bs_reph = calc_overlap(transformed_df, AllAnnotationCategories.Rephrase)
        bs_inter = calc_overlap(transformed_df, AllAnnotationCategories.Interpretation)
        bs_info = calc_overlap(transformed_df, AllAnnotationCategories.Elaboration_Information)
        bs_q = calc_overlap(transformed_df, AllAnnotationCategories.Elaboration_Question)

        return {AllAnnotationCategories.Referred_to: bs_ref,
                AllAnnotationCategories.Rephrase: bs_reph,
                AllAnnotationCategories.Interpretation: bs_inter,
                AllAnnotationCategories.Elaboration_Information: bs_info,
                AllAnnotationCategories.Elaboration_Question: bs_q}
    else:
        #  bs_repeat = calc_overlap(transformed_df, ContentReproductionCategories.Repeat)
        try:
            bs_para = calc_overlap(transformed_df, ContentReproductionCategories.Paraphrase, interview=interview)
        except TypeError:  # statistics.StatisticsError or :  #  mean requires at least one data point, NoneType
            # legacy
            bs_para = calc_overlap(transformed_df, ContentReproductionCategories.Repeat, interview=interview)
        #  bs_re = calc_overlap(transformed_df, ContentReproductionCategories.Reference)

        return {AllAnnotationCategories.Referred_to: bs_ref,
                # ContentReproductionCategories.Repeat: bs_repeat,
                ContentReproductionCategories.Paraphrase: bs_para
                # ContentReproductionCategories.Reference: bs_re
                }


def calc_HighlightScores(transformed_df, chunked=False, interview=None,
                         content_reproduction: bool = True):
    if interview is None:
        interview = interview_data.MediaSumProcessor()
    bs_ref = calc_str_agreements_cat(transformed_df, AllAnnotationCategories.Referred_to, interview, chunked)

    if not content_reproduction:
        bs_reph = calc_str_agreements_cat(transformed_df, AllAnnotationCategories.Rephrase, interview, chunked)
        bs_inter = calc_str_agreements_cat(transformed_df, AllAnnotationCategories.Interpretation, interview, chunked)
        bs_info = calc_str_agreements_cat(transformed_df, AllAnnotationCategories.Elaboration_Information, interview,
                                          chunked)
        bs_q = calc_str_agreements_cat(transformed_df, AllAnnotationCategories.Elaboration_Question, interview, chunked)

        return {AllAnnotationCategories.Referred_to: bs_ref,
                AllAnnotationCategories.Rephrase: bs_reph,
                AllAnnotationCategories.Interpretation: bs_inter,
                AllAnnotationCategories.Elaboration_Information: bs_info,
                AllAnnotationCategories.Elaboration_Question: bs_q}

    else:
        bs_paraphrase = calc_str_agreements_cat(transformed_df, ContentReproductionCategories.Paraphrase, interview,
                                                chunked)

        guest_hls, _, _, _ = extract_txts_to_compare(AllAnnotationCategories.Referred_to, transformed_df, interview)
        host_hls, _, _, _ = extract_txts_to_compare(ContentReproductionCategories.Paraphrase, transformed_df, interview)
        sem_pairs = [[g, h] for g_ls, h_ls in zip(guest_hls, host_hls) for g, h in zip(g_ls, h_ls)]
        print(f"Selecting {min(150, len(sem_pairs))} out of {len(sem_pairs)} hl annotations for inspection")
        print(random.sample(sem_pairs, min(150, len(sem_pairs))))

        return {AllAnnotationCategories.Referred_to: bs_ref,
                ContentReproductionCategories.Paraphrase: bs_paraphrase,
                "Mean Self-Similarity between Guest/Host": transformer_scores(sem_pairs, bert_score=False)[0],
                "Jaccard Overlap between Guest/Host": jaccard_overlap_for_highlights(sem_pairs)
                }


def calc_pc_eval_crowd(pd_report):
    """
        only called for annotations for paraphrase candidate selection
        ASSUMES data is given in 2 ppl annotating the same instances in the right order (crowd setup)
        (right order is tested with assert)
    :param pd_report:
    :return:
    """
    paraphrase_candidate = True
    binary_dict = get_annotation_dict(pd_report, paraphrase_candidate=paraphrase_candidate)
    result = _calc_candidate_freq_from_binaries(binary_dict)

    q_ids = [q_list for q_list in binary_dict["q_ids"][::2]]
    binaries = binary_dict["binaries"]
    selected_qs = [True if 1 in elems else False for elems in zip(*binary_dict["complete_binaries"])]
    result["Resulting Paraphrase Candidates"] = [q_id for selected, q_id in zip(selected_qs, q_ids) if selected]

    print("Assuming 2 annotators each in the right order ... ")
    print(f"Total number q_ids: {len(q_ids)}")
    print(f"Total number annotators: {len(binaries)}")
    pc_shares = []
    cohen_values = []
    cohen_mv_values = []
    max_mv_cohen = []
    overlap = []
    contained_zeroes = []
    individual_pc_shares = []
    for i in range(0, len(binaries), 2):
        # assert that the chosen two annotations have the same non-nan values
        indices_nan_1 = numpy.argwhere(np.isnan(binary_dict["complete_binaries"][i]))
        indices_nan_2 = numpy.argwhere(np.isnan(binary_dict["complete_binaries"][i + 1]))
        assert (numpy.array_equal(indices_nan_1, indices_nan_2))
        # Get annotations without NaN
        an_1 = binaries[i]
        an_2 = binaries[i + 1]
        _calc_pc_eval_scores(an_1, an_2, pc_shares, cohen_values, cohen_mv_values, max_mv_cohen, overlap,
                             contained_zeroes, individual_pc_shares)

    if len(pc_shares) > 1:
        result["AVG # PCs"] = [sum(pc_shares) / len(pc_shares), statistics.stdev(pc_shares)]

        # Agreement Values
        result["fleiss"] = fleiss_kappa_agreement(binary_dict["complete_binaries"])
        result["AVG Cohen + STD"] = [sum(cohen_values) / len(cohen_values), statistics.stdev(cohen_values)]
        result["AVG MAX MV Cohen + STD"] = [sum(max_mv_cohen) / len(max_mv_cohen), statistics.stdev(max_mv_cohen)]
        result["AVG Overlap + STD"] = [sum(overlap) / len(overlap), statistics.stdev(overlap)]
        result["AVG MAX 0 -> 0 Overlap"] = [sum(contained_zeroes) / len(contained_zeroes),
                                            statistics.stdev(contained_zeroes)]

    result["AVG MV Cohen + STD"] = [sum(cohen_mv_values) / len(cohen_mv_values), statistics.stdev(cohen_mv_values)]
    result["AVG individual # PCs"] = [sum(individual_pc_shares) / len(individual_pc_shares),
                                      statistics.stdev(individual_pc_shares)]

    result["ALL Cohen"] = cohen_values
    result["ALL MV Cohen"] = cohen_mv_values
    result["ALL Overlap"] = overlap
    result["MAX Overlap"] = contained_zeroes
    result["Individual # PC"] = individual_pc_shares

    result["ALL Q IDs"] = q_ids

    return result


def _calc_pc_eval_scores(an_1, an_2, pc_shares, cohen_values, cohen_mv_values, max_mv_cohen, overlap, contained_zeroes,
                         individual_pc_shares):
    """
        calculates all interesting PC evaluation scores and saves them in the according array
            that is part of the function arguments
    :param individual_pc_shares:
    :param an_1:
    :param an_2:
    :param pc_shares:
    :param cohen_values:
    :param cohen_mv_values:
    :param max_mv_cohen:
    :param overlap:
    :param contained_zeroes:
    :return:
    """
    # calculate Agreement Scores
    #   Relative Frequency PCs
    pc_shares.append(
        _calc_candidate_freq_from_binaries({"complete_binaries": [an_1, an_2]})["relative at least one vote"]
    )
    individual_pc_shares.append(sum(an_1) / len(an_1))
    individual_pc_shares.append(sum(an_2) / len(an_2))
    #   Cohen Values
    cohen_values.append(cohen_kappa_agreement(an_1, an_2))
    #       based on "majority vote" (whether sth is selected as PC)
    mv = [1 if 1 in elems else 0 for elems in zip(an_1, an_2)]
    cohen_mv_values.append(cohen_kappa_agreement(an_1, mv))
    cohen_mv_values.append(cohen_kappa_agreement(an_2, mv))
    max_mv_cohen.append(max(cohen_kappa_agreement(an_1, mv),
                            cohen_kappa_agreement(an_2, mv)))
    #   where do annotators agree?
    agreement = [1 if a == b else 0 for a, b in zip(an_1, an_2)]
    overlap.append(sum(agreement) / len(agreement))
    #   is one annotator more "sure" than the other?
    consumed_max = calc_consumed_max(an_1, an_2)
    contained_zeroes.append(consumed_max)


def _calc_candidate_freq_from_binaries(binary_dict):
    selected_pc = [True if 1 in elems else False for elems in zip(*binary_dict["complete_binaries"])]
    # selected_pc = [True if 1 in elems else False for elems in zip(*binaries)]
    result = {
        "relative at least one vote": sum(selected_pc) / len(selected_pc),
        "absolute at least one vote": sum(selected_pc),
        "Selected PCs": selected_pc
    }
    return result


def calc_consumed_max(an_1, an_2):
    a_when_b_0 = [a for a, b in zip(an_1, an_2) if b == 0]
    b_when_a_0 = [b for a, b in zip(an_1, an_2) if a == 0]
    share_a_also_0 = 0
    share_b_also_0 = 0
    if len(a_when_b_0) > 0:
        share_a_also_0 = (len(a_when_b_0) - sum(a_when_b_0)) / len(a_when_b_0)
    if len(b_when_a_0) > 0:
        share_b_also_0 = (len(b_when_a_0) - sum(b_when_a_0)) / len(b_when_a_0)
    consumed_max = max(share_a_also_0, share_b_also_0)
    return consumed_max


def get_hl_agreement(highlight_df, interview):
    print("For Guest/Host Utterances Seperatly: ")
    result_dict = calc_overlaps(highlight_df, interview=interview)
    print(f"\t{result_dict}")
    print("For Guest/Host Utterances Combined: ")
    result_dict = calc_HighlightScores(highlight_df, interview=interview)
    print(f"\t{result_dict}")
