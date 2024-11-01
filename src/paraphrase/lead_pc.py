"""
    utility file for wrangling screening results
    DOES NOT import interview_data
"""
import itertools
import logging
from typing import List
import pandas as pd

import paraphrase.utility.annotation_df as annotation_df
from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.utility.annotation_df import read_df_from_list, AnnotationColumns, AllAnnotationCategories, \
    ContentReproductionCategories

"""
    path to the lead annotation results ("paraphrase candidates" or "pc" annotations)
"""
PC_AN_PATH_first_batch_1 = (get_dir_to_src() + "/../result/Annotations/Paraphrase Candidates/"
                                               "ANON_23-06-20_post-sampling-updates_359-908_updated-PCs.tsv")
PC_AN_PATH_first_batch_2 = (get_dir_to_src() + "/../result/Annotations/Paraphrase Candidates/"
                                               "ANON_23-06-20_post-sampling-updates_959-1158_annotations.tsv")

PC_LEGACY_FOLDER = get_dir_to_src() + "/../result/Annotations/Paraphrase Candidates/legacy files/"
PC_SECOND_BATCH_FOLDER = get_dir_to_src() + "/../result/Annotations/Paraphrase Candidates/Qualtrics/"
PC_SECOND_BATCH_BINARY_FILES = ["ANON_23-06-23_Paraphrase-Candidate_1159-1208_July+25,+2023_03.47.tsv",
                                "ANON_23-07-10_Paraphrase-Candidate_1209-1258_July+25,+2023_03.47.tsv",
                                "ANON_23-07-17_Paraphrase-Candidate_1259-1308_July+25,+2023_03.48.tsv",
                                "ANON_23-07-17_Paraphrase-Candidate_1309-1358_July+25,+2023_03.48.tsv",
                                "ANON_23-07-17_Paraphrase-Candidate_1359-1408_July+25,+2023_03.48.tsv"]
PC_SECOND_BATCH_HL_FILES = ["ANON_23-07-17_Paraphrase-Candidate_1409-1458_July+25,+2023_03.48.tsv",
                            "ANON_23-07-17_Paraphrase-Candidate_1459-1508_July+25,+2023_03.48.tsv",
                            "ANON_23-07-17_Paraphrase-Candidate_1509-1558_July+25,+2023_03.48.tsv",
                            "ANON_23-07-17_Paraphrase-Candidate_1559-1608_July+25,+2023_03.48.tsv",
                            "ANON_23-07-31_Paraphrase-Candidate_1609-1658_August+9,+2023_02.09.tsv",
                            "ANON_23-07-31_Paraphrase-Candidate_1659-1708_August+9,+2023_02.10.tsv",
                            "ANON_23-08-01_Paraphrase-Candidate_1709-1758_August+9,+2023_02.11.tsv",
                            "ANON_23-08-01_Paraphrase-Candidate_1759-1808_August+9,+2023_02.12.tsv",
                            "ANON_23-08-02_Paraphrase-Candidate_1809-1858_August+9,+2023_02.13.tsv",
                            "ANON_23-08-02_Paraphrase-Candidate_1859-1908_August+9,+2023_02.13.tsv",
                            "ANON_23-08-07_Paraphrase-Candidate_1909-1958_August+9,+2023_02.13.tsv",
                            "ANON_23-08-07_Paraphrase-Candidate_1959-2008_August+9,+2023_02.14.tsv",
                            "ANON_23-08-07_Paraphrase-Candidate_2009-2058_August+9,+2023_02.14.tsv",
                            "ANON_23-08-08_Paraphrase-Candidate_2059-2108_August+9,+2023_02.19.tsv",
                            "ANON_23-08-08_Paraphrase-Candidate_2109-2158_August+9,+2023_02.15.tsv",
                            "ANON_23-08-08_Paraphrase-Candidate_2159-2208_August+9,+2023_02.15.tsv",
                            "ANON_23-08-14_Paraphrase-Candidate_2209-2258_August+29,+2023_01.11.tsv",
                            "ANON_23-08-14_Paraphrase-Candidate_2259-2308_August+29,+2023_01.12.tsv",
                            "ANON_23-08-15_Paraphrase-Candidate_2309-2358_August+29,+2023_01.13.tsv",
                            "ANON_23-08-15_Paraphrase-Candidate_2359-2408_August+29,+2023_01.13.tsv",
                            "ANON_23-08-23_Paraphrase-Candidate_2409-2458_August+29,+2023_01.13.tsv",
                            "ANON_23-08-23_Paraphrase-Candidate_2459-2508_August+29,+2023_01.14.tsv",
                            "ANON_23-08-23_Paraphrase-Candidate_2509-2558_August+29,+2023_01.28.tsv",
                            "ANON_23-08-24_Paraphrase-Candidate_2559-2608_August+29,+2023_01.14.tsv",
                            "ANON_23-08-24_Paraphrase-Candidate_2609-2658_August+29,+2023_01.15.tsv",
                            "ANON_23-08-25_Paraphrase-Candidate_2659-2708_August+29,+2023_01.15.tsv",
                            "ANON_23-08-25_Paraphrase-Candidate_2709-2758_August+29,+2023_01.15.tsv",
                            "ANON_23-08-29_Paraphrase-Candidate_2759-2808_August+30,+2023_03.03.tsv",
                            "ANON_23-08-29_Paraphrase-Candidate_2809-2858_August+30,+2023_03.06.tsv",
                            "ANON_23-08-29_Paraphrase-Candidate_2859-2908_August+30,+2023_03.05.tsv",
                            "ANON_23-08-30_Paraphrase-Candidate_2909-2958_August+31,+2023_01.38.tsv",
                            "ANON_23-08-30_Paraphrase-Candidate_2959-3008_August+31,+2023_01.40.tsv",
                            "ANON_23-08-30_Paraphrase-Candidate_3009-3058_August+31,+2023_01.39.tsv",
                            "ANON_23-08-30_Paraphrase-Candidate_3059-3108_August+31,+2023_01.39.tsv",
                            "ANON_23-08-31_Paraphrase-Candidat_3109-3158_September+1,+2023_01.02.tsv",
                            "ANON_23-08-31_Paraphrase-Candidate_3159-3208_September+1,+2023_01.07.tsv",
                            "ANON_23-08-31_Paraphrase-Candidate_3209-3258_September+1,+2023_01.10.tsv",
                            "ANON_23-08-31_Paraphrase-Candidate_3259-3308_September+1,+2023_01.08.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3309-3358.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3359-3408.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3409-3458.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3459-3508.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3509-3558_September+1,+2023_06.34.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3559-3608.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3609-3658.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3659-3708.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3709-3758.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3759-3808.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3809-3858.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3859-3908.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3909-3958.tsv",
                            "ANON_23-09-01_Paraphrase-Candidate_3959-4008.tsv",
                            "ANON_23-09-05_Paraphrase-Candidate_4009-4058.tsv",
                            "ANON_23-09-05_Paraphrase-Candidate_4059-4108.tsv",
                            "ANON_23-09-06_Paraphrase-Candidate_4109-4158.tsv",  #
                            "ANON_23-09-06_Paraphrase-Candidate_4159-4208.tsv",
                            "ANON_23-09-06_Paraphrase-Candidate_4209-4258.tsv",
                            "ANON_23-09-06_Paraphrase-Candidate_4259-4308.tsv",
                            "ANON_23-09-06_Paraphrase-Candidate_4309-4358.tsv",
                            "ANON_23-09-06_Paraphrase-Candidate_4359-4408.tsv",
                            "ANON_23-09-06_Paraphrase-Candidate_4409-4458.tsv",
                            "ANON_23-09-06_Paraphrase-Candidate_4459-4508.tsv",
                            "ANON_23-09-07_Paraphrase-Candidate_4509-4558.tsv",
                            "ANON_23-09-07_Paraphrase-Candidate_4559-4608.tsv",
                            "ANON_23-09-07_Paraphrase-Candidate_4609-4658.tsv",
                            "ANON_23-09-08_Paraphrase-Candidate_4659-4708.tsv",
                            "ANON_23-09-08_Paraphrase-Candidate_4709-4758.tsv",
                            "ANON_23-09-11_Paraphrase-Candidate_4759-4808.tsv",
                            "ANON_23-09-11_Paraphrase-Candidate_4809-4858.tsv"]
PC_SECOND_BATCH_CORRECTION_FILES = ["../ANON_23-09_checked-ambig.tsv", "../ANON_23-09_checked-context.tsv"]  # some items were re-labeled


def get_author_annotations(print_pilot_stats=False):
    first_pc_anns_df, result_dict = get_pc_first_batch_label_dict()
    if print_pilot_stats:
        print(f"PC First Batch stats: ")
        # iterate over result dict
        for key, value in result_dict.items():
            if "#" in key:
                print(f"{key}: {value}")
    second_pc_anns_df, _ = load_pc_second_batch()
    all_pc = pd.concat([first_pc_anns_df, second_pc_anns_df])
    return all_pc, first_pc_anns_df, second_pc_anns_df


def get_pc_first_batch_label_dict(an_path_1=PC_AN_PATH_first_batch_1, an_path_2=PC_AN_PATH_first_batch_2):
    try:
        df1 = pd.read_csv(an_path_1, sep="\t", encoding="utf-16")
        df2 = pd.read_csv(an_path_2, sep="\t", encoding="utf-16")
    except UnicodeError:
        logging.info("Reading file with utf-8 encoding")
        df1 = pd.read_csv(an_path_1, sep="\t")
        df2 = pd.read_csv(an_path_2, sep="\t")
    concatenated = pd.concat([df1, df2], ignore_index=True)
    result_dict = count_stats_pc_second_batch(concatenated)
    return concatenated, result_dict


def load_pc_second_batch():
    """
        load in pc annotations
    :return: combined dataframe of binary and highlighting annotations (df_sorted),
            ids of those where highlighting exists (hl_ids)
    """
    folder = PC_SECOND_BATCH_FOLDER
    pc_binary = PC_SECOND_BATCH_BINARY_FILES
    pc_hl = PC_SECOND_BATCH_HL_FILES
    corrections = PC_SECOND_BATCH_CORRECTION_FILES
    print(f"Reading in {len(pc_binary) + len(pc_hl)} unique files, "
          f"which should add to {50 * (len(pc_binary) + len(pc_hl))} unique pairs")
    pd_binary = annotation_df.merge_transform([folder + f_name for f_name in pc_binary],
                                              paraphrase_candidate=True)
    print(f"Read in in {len(annotation_df.get_unique_q_ids(pd_binary))} unique binary ids "
          f"out of {50 * len(pc_binary)}")
    assert (len(annotation_df.get_unique_q_ids(pd_binary)) == 50 * len(pc_binary))
    pd_hl = annotation_df.merge_transform([folder + f_name for f_name in pc_hl],
                                          paraphrase_candidate=False)
    print(f"Read in in {len(annotation_df.get_unique_q_ids(pd_hl))} unique hl ids "
          f"out of {50 * len(pc_hl)}")
    assert (len(annotation_df.get_unique_q_ids(pd_hl)) == 50 * len(pc_hl))

    pd_cor = read_df_from_list([folder + f_name for f_name in corrections])
    print(f"Read in in {len(annotation_df.get_unique_q_ids(pd_cor))} unique corr. ids "
          f"out of {len(pd_cor)} files")
    cor_ids = annotation_df.get_unique_q_ids(pd_cor)

    # MERGE the two datasets
    hl_ids = pd_hl["QID"].unique().tolist()
    q_ids = annotation_df.get_unique_q_ids(pd_binary)
    rows = []
    for q_id in q_ids:
        cur_ann = pd_binary[(pd_binary["QID"] == q_id) & (pd_binary["Category"] == "Is Referring")]
        para_sig = "[]"
        if cur_ann["Highlighted"].values[0] == "Yes":
            para_sig = "-"
        rows.append([q_id, "Referred", para_sig, cur_ann["Annotator"].values[0], cur_ann["Session"].values[0]])
        rows.append([q_id, "Paraphrase", para_sig, cur_ann["Annotator"].values[0], cur_ann["Session"].values[0]])
    new_data = pd.DataFrame(rows, columns=[column for column in pd_hl.columns if column != "Session_Start"])
    pd_hl = pd.concat([pd_hl, pd_binary], ignore_index=True)
    pd_hl = pd.concat([pd_hl, new_data], ignore_index=True)
    df_sorted = pd_hl.sort_values(by='QID', ignore_index=True)
    print(f"Found {len(annotation_df.get_unique_q_ids(df_sorted))} unique pairs.")

    print(f"Updating data with AMBIGUOUS/CONTEXT corrections ...")
    # Update data with corrections
    for i, row in pd_cor.iterrows():
        if row[AnnotationColumns.Highlight_Category] == AllAnnotationCategories.Referring_Choice:
            continue
        c_id = row[AnnotationColumns.Question_ID]
        c_cat = row[AnnotationColumns.Highlight_Category]
        c_hl_choices = row[AnnotationColumns.Highlight_Choices]
        df_sorted.loc[(df_sorted[AnnotationColumns.Question_ID] == c_id) &
                      (df_sorted[AnnotationColumns.Highlight_Category] == c_cat),
        AnnotationColumns.Highlight_Choices] = c_hl_choices

    return df_sorted, hl_ids


def count_stats_pc_second_batch(pd_report):
    """
        ASSUMES only one annotator per item, so no agreement is calculated here

        Labels used during 1st pass annotation
            repetition = "REPETITION"
            reference = "REFERENCE"
            it_reference = "IT-REFERENCE"
            inspired_reference = "INSPIRED-REFERENCE"
            context = "CONTEXT"
            conclusion = "CONCLUSION"

            --> here: repetition means is a paraphrase, any of the others means no-paraphrase

        Additional labels 2nd pass see AnnotationLabels

        NOTE: "ERROR" was a keyword used for an error made by the host in the paraphrase, i.e., misrepresenting
            later, I used this to mean that the item itself is faulty ... this results in inconsistency of 2
    :param pd_report:
    :return:
    """
    print(f"PC stands for paraphrase candidate (i.e., just the reviewed cases), "
          f"while paraphrases stands for those that were actually classified as paraphrases.")
    # annotators = get_annotators(pd_report)
    result = {}

    # first pass assignments, yes_ids are the "original" paraphrases
    yes_ids = get_ids_w_label(pd_report, is_PC=True)
    no_ids = get_ids_w_label(pd_report, is_PC=False)
    # all unique pairs
    all_q_ids = pd_report[~pd_report[AnnotationColumns.Question_ID].str.startswith('R_')][
        AnnotationColumns.Question_ID].unique()
    result["# Reviewed"] = len(all_q_ids)
    result["Reviewed"] = all_q_ids

    # REMOVE unclear context cases
    #   OUTSIDE paraphrase categorization
    context_ids = get_ids_w_label(pd_report, [AnnotationLabels.context], is_PC=True) + \
                  get_ids_w_label(pd_report, [AnnotationLabels.context], is_PC=False)
    result["# unclear context PCs"] = len(context_ids)
    result[AnnotationLabels.context] = context_ids

    # AMBIGUOUS
    ambig_ids = get_ids_w_label(pd_report, with_label=[AnnotationLabels.ambiguous])
    result["# Ambiguous PCs, not Context"] = len(set(ambig_ids) - set(context_ids))
    result[AnnotationLabels.ambiguous] = ambig_ids
    result["# unclear context, classified as Ambiguous"] = len(set(context_ids).intersection(set(ambig_ids)))

    # PARAPHRASES
    pc_ids = get_ids_w_label(pd_report, with_label=[AnnotationLabels.paraphrase])
    pc_ids += get_ids_w_label(pd_report, with_label=[AnnotationLabels.repetition],
                              without_label=[AnnotationLabels.paraphrase, AnnotationLabels.non_paraphrase,
                                             AnnotationLabels.ambiguous])
    pc_ids = list(set(pc_ids) - set(context_ids))
    result["# Paraphrases"] = len(pc_ids)
    result["Paraphrases"] = pc_ids

    #   Clear paraphrases, assuming the repetitions ones are clear paraphrases
    clear_ids = get_ids_w_label(pd_report, with_label=[AnnotationLabels.paraphrase],
                                without_label=[AnnotationLabels.difficult])
    #       first pass relic: some cases do not include the AnnotationLabels.paraphrase label but ARE paraphrases
    #           --> these are only those labelled as repetitions (everything else was relabelled within the 115 set)
    clear_ids += get_ids_w_label(pd_report, with_label=[AnnotationLabels.repetition],
                                 without_label=[AnnotationLabels.paraphrase, AnnotationLabels.non_paraphrase,
                                                AnnotationLabels.ambiguous])
    clear_ids = list(set(clear_ids) & set(pc_ids))
    result["# Clear Paraphrases"] = len(clear_ids)
    result["Clear Paraphrases"] = clear_ids
    #   Repetitions
    rep_ids = get_ids_w_label(pd_report, with_label=[AnnotationLabels.repetition, AnnotationLabels.paraphrase],
                              without_label=[AnnotationLabels.repetition + "-SOME"])
    rep_ids += get_ids_w_label(pd_report, with_label=[AnnotationLabels.repetition],
                               without_label=[AnnotationLabels.paraphrase, AnnotationLabels.non_paraphrase,
                                              AnnotationLabels.ambiguous])
    rep_ids = list(set(rep_ids) & set(pc_ids))
    result["# Repetition Paraphrases"] = len(set(rep_ids) & set(pc_ids))
    # result["# unclear context, classified as Paraphrase with rep"] = len(set(context_ids).intersection(set(rep_ids)))
    result[AnnotationLabels.repetition] = rep_ids
    #   without Repetitions
    result["# Paraphrases without Repetitions"] = len(set(pc_ids) - set(rep_ids))
    result["# Clear Paraphrases without Repetition"] = len(set(clear_ids) - set(rep_ids))
    #   Universal Paraphrases
    universal_ids = get_ids_w_label(pd_report, with_label=[AnnotationLabels.universal, AnnotationLabels.paraphrase])
    universal_ids = list(set(universal_ids) & set(pc_ids))
    result["# Universal Paraphrases"] = len(universal_ids)
    result[AnnotationLabels.universal] = universal_ids
    result["# Universal Paraphrases without Repetition"] = len(set(universal_ids) - set(rep_ids))
    #   High Lex Sim Paraphrases
    high_lex_sim = get_ids_w_label(pd_report, [AnnotationLabels.high_lex_sim, AnnotationLabels.paraphrase])
    high_lex_sim += get_ids_w_label(pd_report, with_label=[AnnotationLabels.repetition],
                                    without_label=[AnnotationLabels.paraphrase, AnnotationLabels.non_paraphrase,
                                                   AnnotationLabels.ambiguous])
    high_lex_sim = list(set(high_lex_sim) & set(pc_ids))
    result[AnnotationLabels.high_lex_sim] = high_lex_sim  # high lexical similarity & paraphrase label
    result["# High Sim Paraphrases"] = len(high_lex_sim)
    result["# Clear paraphrases without high lex"] = len(set(clear_ids) - set(high_lex_sim))
    result["# Universal Paraphrases without high lex"] = len(set(universal_ids) - set(context_ids))
    result["# Paraphrases without high lex"] = len(set(pc_ids) - set(high_lex_sim))
    result["# Repetition Paraphrases without high lex"] = len(set(rep_ids) - set(high_lex_sim))
    result["# High Sim Paraphrases without Repetition"] = len(set(high_lex_sim) - set(rep_ids))
    # result[AnnotationLabels.high_sim] = high_lex_sim
    #   Directional
    directional_ids = get_ids_w_label(pd_report, with_label=[AnnotationLabels.directional, AnnotationLabels.paraphrase])
    directional_ids = list(set(directional_ids) & set(pc_ids))
    result["# Directional Paraphrases"] = len(directional_ids)
    result["# Directional Paraphrases without high lex"] = len(set(directional_ids) - set(high_lex_sim))
    result[AnnotationLabels.directional] = directional_ids
    result["# Directional Paraphrases without Repetition"] = len(set(directional_ids) - set(rep_ids))
    #   Perspective Shift
    perspective_ids = get_ids_w_label(pd_report, [AnnotationLabels.perspective_shift, AnnotationLabels.paraphrase])
    perspective_ids = list(set(perspective_ids) & set(pc_ids))
    result["# Perspective Shift Paraphrases"] = len(perspective_ids)
    result["# Perspective Shift Paraphrases without high lex"] = len(set(perspective_ids) - set(high_lex_sim))
    result[AnnotationLabels.perspective_shift] = perspective_ids
    result["# Perspective Shift Paraphrases without Repetition"] = len(
        set(perspective_ids) - set(rep_ids))
    #   Pragmatic without Perspective Shift
    pragmatic_ids = get_ids_w_label(pd_report,
                                    with_label=[AnnotationLabels.pragmatic, AnnotationLabels.paraphrase],
                                    without_label=[AnnotationLabels.perspective_shift])
    pragmatic_ids = list(set(pragmatic_ids) & set(pc_ids))
    result["# Pragmatic Paraphrases (no shift)"] = len(pragmatic_ids)
    result[AnnotationLabels.pragmatic] = pragmatic_ids
    result["# Pragmatic Paraphrases (no shift) without high lex"] = len(
        set(pragmatic_ids) - set(high_lex_sim))
    result["# Pragmatic Paraphrases (no shift) without rep"] = len(set(pragmatic_ids) - set(rep_ids))
    #   Side Paraphrase
    side_ids = get_ids_w_label(pd_report, [AnnotationLabels.side, AnnotationLabels.paraphrase])
    side_ids = list(set(side_ids) & set(pc_ids))
    result["# SIDE Paraphrases"] = len(side_ids)
    result[AnnotationLabels.side] = side_ids

    # NON-PARAPHRASES
    no_pc_ids = get_ids_w_label(pd_report, with_label=[AnnotationLabels.non_paraphrase])
    no_pc_ids += get_ids_w_label(pd_report, without_label=[AnnotationLabels.repetition, AnnotationLabels.paraphrase,
                                                           AnnotationLabels.non_paraphrase,
                                                           AnnotationLabels.ambiguous])
    no_pc_ids = list(set(no_pc_ids + no_ids) - set(context_ids) - set(ambig_ids))
    result["# No Paraphrases"] = len(no_pc_ids)
    # result["# unclear context, classified as Non Paraphrase"] = len(set(context_ids).intersection(set(no_pc_ids)))
    result[AnnotationLabels.non_paraphrase] = no_pc_ids
    #   High Lexical Similarity
    high_sim_nonp_ids = get_ids_w_label(pd_report, with_label=[AnnotationLabels.high_lex_sim,
                                                               AnnotationLabels.non_paraphrase]) \
                        + get_ids_w_label(pd_report, with_label=[AnnotationLabels.high_lex_sim,
                                                                 AnnotationLabels.non_paraphrase], is_PC=False)
    high_sim_nonp_ids = list(set(high_sim_nonp_ids) & set(no_pc_ids))
    result["# High Sim Non Paraphrases"] = len(high_sim_nonp_ids)
    result[AnnotationLabels.high_lex_sim + " " + AnnotationLabels.non_paraphrase] = high_sim_nonp_ids
    #   Only Related
    related_ids = get_ids_w_label(pd_report,
                                  with_label=[AnnotationLabels.inspired_reference],
                                  without_label=[AnnotationLabels.paraphrase, AnnotationLabels.non_paraphrase,
                                                 AnnotationLabels.ambiguous]) \
                  + get_ids_w_label(pd_report, with_label=[AnnotationLabels.related, AnnotationLabels.non_paraphrase]) \
                  + get_ids_w_label(pd_report, with_label=[AnnotationLabels.related, AnnotationLabels.non_paraphrase],
                                    is_PC=False)
    related_ids = list(set(related_ids) & set(no_pc_ids))
    result["# Related Non Paraphrases"] = len(set(related_ids))
    result[AnnotationLabels.related] = related_ids
    #   Partial
    partial_ids = get_ids_w_label(pd_report, with_label=[AnnotationLabels.partial, AnnotationLabels.non_paraphrase]) \
                  + get_ids_w_label(pd_report, with_label=[AnnotationLabels.partial, AnnotationLabels.non_paraphrase],
                                    is_PC=False)
    partial_ids = list(set(partial_ids) & set(no_pc_ids))
    result["# Partial Non Paraphrases"] = len(partial_ids)
    result[AnnotationLabels.partial] = partial_ids
    #   Conclusion
    conclusion_ids = get_ids_w_label(pd_report, [AnnotationLabels.conclusion, AnnotationLabels.non_paraphrase]) \
                     + get_ids_w_label(pd_report, [AnnotationLabels.conclusion, AnnotationLabels.non_paraphrase],
                                       is_PC=False)
    conclusion_ids += get_ids_w_label(pd_report, with_label=[AnnotationLabels.conclusion],
                                      without_label=[AnnotationLabels.paraphrase, AnnotationLabels.non_paraphrase,
                                                     AnnotationLabels.ambiguous])
    conclusion_ids = list(set(conclusion_ids) & set(no_pc_ids))
    result["# Conclusion Non Paraphrases"] = len(conclusion_ids)
    result[AnnotationLabels.conclusion] = conclusion_ids
    #   Unrelated
    unrelated_ids = get_ids_w_label(pd_report, [AnnotationLabels.unrelated, AnnotationLabels.non_paraphrase]) \
                    + get_ids_w_label(pd_report,
                                      with_label=[AnnotationLabels.unrelated, AnnotationLabels.non_paraphrase],
                                      is_PC=False)
    unrelated_ids = list(set(unrelated_ids) & set(no_pc_ids))
    result["# Unrelated Non Paraphrases"] = len(unrelated_ids)
    result[AnnotationLabels.unrelated] = unrelated_ids

    # OUTSIDE CLASSIFICATION
    #   Difficult
    difficult_ids = get_ids_w_label(pd_report, [AnnotationLabels.difficult])
    result["# Difficult Non Paraphrases"] = len(set(difficult_ids) & set(no_pc_ids))
    result["# Difficult Paraphrases"] = len(set(difficult_ids) & set(pc_ids))
    result[AnnotationLabels.difficult] = list(set(difficult_ids) - set(context_ids))
    #   META-COMMENT
    meta_ids = get_ids_w_label(pd_report, [AnnotationLabels.meta])
    result["# META COMMENT"] = len(set(meta_ids) - set(context_ids))
    #   negated paraphrase
    negated_ids = get_ids_w_label(pd_report, [AnnotationLabels.negated])
    result["# NEGATED-PARAPHRASE"] = len(set(negated_ids) - set(context_ids))
    result[AnnotationLabels.negated] = list(set(negated_ids) - set(context_ids))
    #   surrogate paraphrase
    surrogate_ids = get_ids_w_label(pd_report, [AnnotationLabels.surrogate])
    result["# SURROGATE-PARAPHRASE"] = len(set(surrogate_ids) - set(context_ids))
    #   misrepresentation paraphrase
    misrepresentation_ids = get_ids_w_label(pd_report, [AnnotationLabels.error])
    result["# MISREPRESENTATION-PARAPHRASE"] = len(set(misrepresentation_ids) - set(context_ids))

    paraphrase_vars = ['rep', 'hlex', 'uni', 'clear', 'prag', 'per', 'dir']
    paraprhase_ids = [set(rep_ids), set(high_lex_sim), set(universal_ids), set(clear_ids), set(pragmatic_ids),
                      set(perspective_ids), set(directional_ids)]
    non_para_vars = ['hlex', 'partial', 'conc', 'context', 'unrel', 'rel']
    non_paraphrase_ids = [set(high_sim_nonp_ids), set(partial_ids), set(conclusion_ids), set(context_ids),
                          set(unrelated_ids), set(related_ids)]

    # Generate all combinations
    combinations_para = list(itertools.product([0, 1], repeat=7))
    # Iterate through each combination
    for combination in combinations_para:
        remaining_set = set(pc_ids) - set(context_ids)

        for cur_id_set, is_included in zip(paraprhase_ids, combination):
            if is_included == 1:
                remaining_set = remaining_set & cur_id_set
            else:
                remaining_set -= cur_id_set

        if len(remaining_set) > 0:
            result[f"# paraprhase combination {combination}"] = len(remaining_set)

    # Generate all combinations
    combinations_non_para = list(itertools.product([0, 1], repeat=6))
    # Iterate through each combination
    for combination in combinations_non_para:
        remaining_set = set(no_pc_ids).union(set(context_ids) - set(ambig_ids) - set(pc_ids))

        for cur_id_set, is_included in zip(non_paraphrase_ids, combination):
            if is_included == 1:
                remaining_set = remaining_set & cur_id_set
            else:
                remaining_set -= cur_id_set

        if len(remaining_set) > 0:
            result[f"# non paraprhase combination {combination}"] = len(remaining_set)

    new_order_result = {key: result[key] for key in sorted(result.keys())}
    return new_order_result


class AnnotationLabels:
    repetition = "REPETITION"
    reference = "REFERENCE"
    it_reference = "IT-REFERENCE"
    inspired_reference = "INSPIRED-REFERENCE"
    context = "CONTEXT"
    conclusion = "CONCLUSION"
    # 2nd pass
    universal = "UNIVERSAL"
    paraphrase = "CLEAR-PARAPHRASE"
    non_paraphrase = "CLEAR-NON-PARAPHRASE"
    ambiguous = "AMBIGUOUS"
    difficult = "DIFFICULT"
    coreference = "COREFERENCE"
    pragmatic = "PRAGMATIC"
    perspective_shift = "PERSPECTIVE-SHIFT"
    high_lex_sim = "HIGH-LEXICAL-SIMILARITY"
    meta = "META-COMMENT"
    surrogate = "SURROGATE-PARAPHRASE"
    negated = "NEGATED-PARAPHRASE"
    background_knowledge = "BACKGROUND-KNOWLEDGE"
    directional = "DIRECTIONAL"
    partial = "PARTIAL"
    side = "SIDE-PARAPHRASE"
    unrelated = "UNRELATED"
    related = "ONLY-RELATED"
    error = "ERROR"


def get_ids_w_label(pd_report, with_label: List[str] = None, without_label: List[str] = None, is_PC=True):
    """
        get IDs of questions (e.g., "CNN-108240-3") that contain the label(s) given with with_label
        and do not contain the label(s) given with without_label,
        while having been selected as PC (AnnotationColumns.Highlight_Choices is "Yes") or not ("No") depending on is_PC
    :param without_label:
    :param is_PC:
    :param with_label:
    :param pd_report:
    :return:
    """

    # Regex patterns
    #   a positive or a negative label HAS to be given

    with_pattern = ""
    without_pattern = ""
    if with_label:
        # contains ALL values in any order
        with_pattern = '(?=.*' + ')(?=.*'.join(with_label) + ')'
    if without_label:
        # combined with negation: does not contain any value
        without_pattern = "|".join(without_label)

    # get the paraphrase candidate ids according to is_PC
    highlight_choice = "No"
    if is_PC:
        highlight_choice = "Yes"
    pc_ids = pd_report[
        (pd_report[AnnotationColumns.Highlight_Category] == ContentReproductionCategories.Referring_Choice) &
        (pd_report[AnnotationColumns.Highlight_Choices] == highlight_choice)
        ][AnnotationColumns.Question_ID].tolist()

    if with_label or without_label:
        # get the ids where label_string occurs in comments
        if with_label and without_label:
            ids_w_label = pd_report[(pd_report[AnnotationColumns.Question_ID].isin(pc_ids)) &
                                    (pd_report[AnnotationColumns.Highlight_Category]
                                     == ContentReproductionCategories.Question_Comment) &
                                    (pd_report[AnnotationColumns.Highlight_Choices].str.contains(with_pattern,
                                                                                                 regex=True)) &
                                    (~pd_report[AnnotationColumns.Highlight_Choices].str.contains(without_pattern,
                                                                                                  regex=True,
                                                                                                  na=False))][
                AnnotationColumns.Question_ID].tolist()
        elif with_label:
            ids_w_label = pd_report[(pd_report[AnnotationColumns.Question_ID].isin(pc_ids)) &
                                    (pd_report[AnnotationColumns.Highlight_Category]
                                     == ContentReproductionCategories.Question_Comment) &
                                    (pd_report[AnnotationColumns.Highlight_Choices].str.contains(with_pattern,
                                                                                                 regex=True))][
                AnnotationColumns.Question_ID].tolist()
        else:
            ids_w_label = pd_report[(pd_report[AnnotationColumns.Question_ID].isin(pc_ids)) &
                                    (pd_report[AnnotationColumns.Highlight_Category]
                                     == ContentReproductionCategories.Question_Comment) &
                                    (~pd_report[AnnotationColumns.Highlight_Choices].str.contains(without_pattern,
                                                                                                  regex=True,
                                                                                                  na=False))][
                AnnotationColumns.Question_ID].tolist()
    else:
        ids_w_label = pc_ids

    return ids_w_label
