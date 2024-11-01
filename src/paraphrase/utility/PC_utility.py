"""
    utility file for paraphrase candidate (PC) annotations by lead author

"""
from datetime import datetime
from typing import Set

import pandas as pd

import paraphrase.utility.annotation_df as annotation_df
from paraphrase import interview_data
from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.utility.annotation_df import AnnotationColumns, ContentReproductionCategories
from paraphrase.lead_pc import PC_AN_PATH_first_batch_1, PC_AN_PATH_first_batch_2, get_pc_first_batch_label_dict, AnnotationLabels, get_ids_w_label

PC_ID_COLUMN = "PC IDs"

" -------------------------- first round of paraphrase candidate annotations -------------------------- "
SAMPLE_5050_PC_IDS = get_dir_to_src() + "/../result/Annotations/Paraphrase Candidates/Sampled_100_PC_IDs.tsv"

# file path to those files used for sampling
PILOT_AN_PATH_1 = (get_dir_to_src() +
                   "/../result/Annotations/Paraphrase Candidates/legacy files/359-908_updated-PCs.tsv")
PILOT_AN_PATH_2 = (get_dir_to_src() +
                   "/../result/Annotations/Paraphrase Candidates/legacy files/959-1158_annotations.tsv")

# file path to those files with most up to date labels

" -------------------------- second round of paraphrase candidate annotations -------------------------- "
SAMPLE_RAND_PARA_PC_IDS = get_dir_to_src() + "/../result/Annotations/Paraphrase Candidates/Annotation-Sample-500.tsv"

# file path to those files used for sampling
# path to updates to initial annotations (especially ambiguous label)

" -------------------------- print STATS -------------------------- "


def print_pc_stats(df_pcs, include_ambiguous=False, first_batch=False):
    """
        print statistics about the paraphrase candidate annotations by the lead author
            depending on first_batch flag, some free-text labels assigned by the lead author differ
    :param df_pcs:
    :param include_ambiguous:
    :param first_batch:
    :return:
    """
    # interview stats
    q_ids = annotation_df.get_unique_q_ids(df_pcs)
    print_interview_pair_stats(q_ids)
    paraphrase_ids = get_pc_paraphrase_ids(df_pcs, first_batch=first_batch)
    repetition_ids = get_pc_repetition_ids(df_pcs, first_batch=first_batch)
    no_paraphrase_ids = get_non_paraphrase_ids(df_pcs, first_batch=first_batch)
    ambig_ids = get_ambiguous_ids(df_pcs, first_batch=first_batch)
    context_ids = get_context_ids(df_pcs, first_batch=first_batch)
    assert (set(paraphrase_ids) & set(repetition_ids) == set(repetition_ids))
    assert (set(paraphrase_ids) & set(no_paraphrase_ids) == set())
    # assert (set(ambig_ids) & set(context_ids) == set())
    print("Ambiguous and Missing context cases can overlap")
    assert (set(paraphrase_ids) & set(context_ids) == set())
    assert (set(paraphrase_ids) & set(ambig_ids) == set())
    assert (set(no_paraphrase_ids) & set(ambig_ids) == set())
    assert (set(no_paraphrase_ids) & set(context_ids) == set())
    print(f"Ambiguous: {len(ambig_ids)}")
    print(f"Missing Context: {len(context_ids)}")
    print(f"No Paraphrases: {len(no_paraphrase_ids)}")
    print(f"Paraphrases: {len(paraphrase_ids)}")
    print(f"\t which includes Repetitions: {len(repetition_ids)}")

    if include_ambiguous:
        return (len(set(no_paraphrase_ids) | set(ambig_ids) | set(context_ids)),
                len(paraphrase_ids) - len(repetition_ids), len(repetition_ids))
    else:
        return (len(no_paraphrase_ids), len(paraphrase_ids) - len(repetition_ids), len(repetition_ids))


def print_interview_pair_stats(q_ids):
    interview_ids = annotation_df.get_unique_interview_ids(q_ids)
    print(f"# Interviews: {len(interview_ids)}")
    npr_ids = [i_id for i_id in interview_ids if "NPR" in i_id]
    print(f"\t # NPR Interviews: {len(npr_ids)}")
    cpr_ids = [i_id for i_id in interview_ids if "CNN" in i_id]
    print(f"\t # CNN Interviews: {len(cpr_ids)}")
    print(f"# (G,H)-pairs: {len(q_ids)}")
    npr_ids = [i_id for i_id in q_ids if "NPR" in i_id]
    print(f"\t # NPR pairs: {len(npr_ids)}")
    cpr_ids = [i_id for i_id in q_ids if "CNN" in i_id]
    print(f"\t # CNN pairs: {len(cpr_ids)}")


" -------------------------- LOAD -------------------------- "


def get_sampled_5050_pc_ids() -> Set[int]:
    """
        get the 100 PC IDs as a set
    :return: set of integers
    """
    return get_qids_from_file(SAMPLE_5050_PC_IDS)


def get_sampled_RAND_PARA_pc_ids() -> Set[int]:
    return get_qids_from_file(SAMPLE_RAND_PARA_PC_IDS)


def get_random_q_ids() -> Set[int]:
    """
        get those ids for items that were sampled as part of the bigger 500 set
    :return:
    """
    sample_file = SAMPLE_RAND_PARA_PC_IDS
    sample_data = pd.read_csv(sample_file, sep='\t')
    random_qids = sample_data[sample_data["Random"] == 1]["QID"].values.tolist()
    return set(random_qids)


def get_qids_from_file(path_sampled):
    try:
        df = pd.read_csv(path_sampled, sep="\t")
    except UnicodeDecodeError:
        df = pd.read_csv(path_sampled, sep="\t", encoding="utf-16")
    if PC_ID_COLUMN in df.columns:
        df_sampled_pc = set(df[PC_ID_COLUMN].values.tolist())
    elif AnnotationColumns.Question_ID in df.columns:
        df_sampled_pc = set(df[AnnotationColumns.Question_ID].values.tolist())
    else:
        raise ValueError("Can't find column with PC IDs")
    # sort df_sampled_pc alphabetically
    df_sampled_pc = sorted(df_sampled_pc)
    return df_sampled_pc


def get_pilot_pc_labels():
    an_path_1 = PC_AN_PATH_first_batch_1
    an_path_2 = PC_AN_PATH_first_batch_2
    _, result_dict = get_pc_first_batch_label_dict(an_path_1, an_path_2)
    return result_dict


" -------------------------- Evaluate PC Annotations -------------------------- "


def get_no_yes_ids_singlepc(df_sorted, print_stats=False):
    """
        For the standard datastructure get counts of "yes"/"No" votes
        Assumes only one single annotation per PC
    :param print_stats:
    :param df_sorted:
    :return:
    """
    # Get Yes/No Selections in Qualtrics Survey
    if print_stats:
        print(df_sorted[df_sorted['Category'] == 'Is Referring']["Highlighted"].value_counts())
    yes_ids = df_sorted[(df_sorted['Category'] == 'Is Referring') & (df_sorted["Highlighted"] == 'Yes')]["QID"] \
        .values.tolist()
    no_ids = df_sorted[(df_sorted['Category'] == 'Is Referring') & (df_sorted["Highlighted"] == 'No')]["QID"] \
        .values.tolist()
    return no_ids, yes_ids


def get_pc_repetition_ids(df_sorted, print_stats=False, first_batch=False):
    """
        get those PC annotations where the "repetition" keyword was added,
            also tests that it was selected as not not selected as a paraphrase
    :param df_sorted:
    :param print_stats:
    :return:
    """
    paraphrase_ids = get_pc_paraphrase_ids(df_sorted, first_batch=first_batch)  # Repetitions included in paraphrases
    if first_batch:
        # in the first batch "-SOME" signified a non-repetition but sth that was close to one
        #   (e.g., just one word repeated)
        repetition_ids = get_ids_w_label(df_sorted, with_label=[AnnotationLabels.repetition],
                                         without_label=[AnnotationLabels.repetition + "-SOME"])
    else:
        repetition_ids = get_ids_w_label(df_sorted, with_label=[AnnotationLabels.repetition])
    repetition_ids = list(set(repetition_ids) & set(paraphrase_ids))
    if print_stats:
        print(f"# Repetitions: {len(repetition_ids)}")
    return repetition_ids


def get_error_ids(df_sorted, print_stats=False, first_batch=False):
    if first_batch:
        # in the first batch an error did not occur in the items, but instead "ERROR" was used
        #   to signify an error done by the host in the paraphrase, i.e., a misrepresentation of the guest
        error_ids = []
    else:
        # Get ERROR keyword
        error_ids = get_ids_w_label(df_sorted, with_label=[AnnotationLabels.error], is_PC=True) + \
                    get_ids_w_label(df_sorted, with_label=[AnnotationLabels.error], is_PC=False)
    if print_stats:
        print(f"# Errors: {len(error_ids)}")
    return error_ids


def get_context_ids(df_sorted, print_stats=False, first_batch=False):
    # Get CONTEXT missing cases
    # n_ids, y_ids = get_no_yes_ids_singlepc(df_sorted)
    err_ids = get_error_ids(df_sorted, first_batch=first_batch)
    context_ids = get_ids_w_label(df_sorted, with_label=[AnnotationLabels.context], is_PC=True) + \
                  get_ids_w_label(df_sorted, with_label=[AnnotationLabels.context], is_PC=False)
    # remove err_ids from context_ids
    context_ids = [c_id for c_id in context_ids if c_id not in err_ids]
    if print_stats:
        print(f"# Context Missing: {len(context_ids)}")
    return context_ids


def get_ambiguous_ids(df_sorted, print_stats=False, first_batch=False):
    # Get AMBIGUOUS: "ambiguous" in comments
    err_ids = get_error_ids(df_sorted, first_batch=first_batch)
    cntxt_ids = get_context_ids(df_sorted, first_batch=first_batch)
    ambigous_ids = get_ids_w_label(df_sorted, [AnnotationLabels.ambiguous], is_PC=True) + \
                   get_ids_w_label(df_sorted, [AnnotationLabels.ambiguous], is_PC=False)
    ambigous_ids = [a_id for a_id in ambigous_ids if a_id not in cntxt_ids and a_id not in err_ids]
    if print_stats:
        print(f"# Ambigous, but without error and without context problems: {len(ambigous_ids)}")
    return ambigous_ids


def get_pc_paraphrase_ids(df_sorted, print_stats=False, first_batch=False):
    # Get PARAPHRASES: is referring Yes + (not "AMBIGUOUS" and not "CLEAR-NON-PARAPHRASE")
    # _, y_ids = get_no_yes_ids_singlepc(df_sorted)
    if first_batch:
        err_ids = []  # "error" means sth different in the first batch annotations
    else:
        err_ids = get_error_ids(df_sorted)
    cntxt_ids = get_context_ids(df_sorted)
    ambig_ids = get_ambiguous_ids(df_sorted)
    # assert that err, cntxt and ambig ids are disjoint
    assert not set(err_ids) & set(cntxt_ids) & set(ambig_ids), "The lists are not disjoint"
    if first_batch:
        paraphrase_ids = get_ids_w_label(df_sorted, with_label=[AnnotationLabels.paraphrase], is_PC=True)
        paraphrase_ids += get_ids_w_label(df_sorted, with_label=[AnnotationLabels.repetition],
                                          without_label=[AnnotationLabels.paraphrase, AnnotationLabels.non_paraphrase],
                                          is_PC=True)
    else:
        # "CLEAR-PARAPHRASE" not necessarily part of keywords anymore in second batch,
        #   but instead the classification with yes excluding those that explcitly labelled as non-paraphrase
        paraphrase_ids = get_ids_w_label(df_sorted, without_label=[AnnotationLabels.non_paraphrase], is_PC=True)
    paraphrase_ids = [p_id for p_id in paraphrase_ids if p_id not in err_ids + cntxt_ids + ambig_ids]
    # paraphrase_ids = df_sorted[(df_sorted['QID'].isin(y_ids)) &
    #                            (df_sorted["Category"] == "Comment") &
    #                            (~df_sorted["Highlighted"].str.contains(
    #                                "AMBIGUOUS|CLEAR-NON-PARAPHRAS|CLEAR-NON-PARAPRHAS|CONTEXT",
    #                                na=False)) &
    #                            (~df_sorted['QID'].isin(err_ids)) &
    #                            (~df_sorted['QID'].isin(cntxt_ids)) &
    #                            (~df_sorted['QID'].isin(ambig_ids))
    #                            ]["QID"].values.tolist()
    if print_stats:
        print(f"# Paraphrases: {len(paraphrase_ids)}")
    return list(set(paraphrase_ids))


def get_non_paraphrase_ids(df_sorted, print_stats=False, first_batch=False):
    # n_ids, y_ids = get_no_yes_ids_singlepc(df_sorted)
    err_ids = get_error_ids(df_sorted, first_batch=first_batch)
    cntxt_ids = get_context_ids(df_sorted, first_batch=first_batch)
    ambig_ids = get_ambiguous_ids(df_sorted, first_batch=first_batch)
    no_pc_ids = get_ids_w_label(df_sorted, is_PC=False)
    no_pc_ids += get_ids_w_label(df_sorted, with_label=[AnnotationLabels.non_paraphrase], is_PC=True)
    if first_batch:
        # in the first batch, there exist pairs classified as paraphrase candidate,
        #   but that are not paraphrases, even when non_paraphrase is not present
        #   e.g., IT-REFERENCE annotations
        no_pc_ids += get_ids_w_label(df_sorted, without_label=[AnnotationLabels.repetition, AnnotationLabels.paraphrase,
                                                               AnnotationLabels.non_paraphrase,
                                                               AnnotationLabels.ambiguous], is_PC=True)
    no_pc_ids = [n_id for n_id in no_pc_ids if n_id not in err_ids + cntxt_ids + ambig_ids]
    # non_paraphrase_ids = df_sorted[(df_sorted['QID'].isin(y_ids)) &
    #                                (df_sorted["Category"] == "Comment") &
    #                                (df_sorted["Highlighted"].str.contains("CLEAR-NON-PARAPHRAS|CLEAR-NON-PARAPRHAS",
    #                                                                       na=False)) &
    #                                (~df_sorted["Highlighted"].str.contains("AMBIGUOUS", na=False)) &
    #                                (~df_sorted['QID'].isin(err_ids)) &
    #                                (~df_sorted['QID'].isin(cntxt_ids)) &
    #                                (~df_sorted['QID'].isin(ambig_ids))
    #                                ]["QID"].values.tolist()
    no_pc_ids = list(set(no_pc_ids))
    if print_stats:
        print(f"# Non-Paraphrases: "
              f"{len(no_pc_ids)}")
    return no_pc_ids


def get_non_repetitions_singlepc(df_sorted, print_stats=False):
    #   Get (Non-)Repetitions
    p_ids = get_pc_paraphrase_ids(df_sorted)
    repetition_ids = df_sorted[(df_sorted['QID'].isin(p_ids)) &
                               (df_sorted["Category"] == "Comment") &
                               (df_sorted["Highlighted"].str.contains("REPETITION", na=False))
                               ]["QID"].values.tolist()
    non_rep_paraphrase_ids = list(set(p_ids) - set(repetition_ids))

    if print_stats:
        print(f"\t # Repetitions: {len(repetition_ids)}")
        print(f"\t # Non-Repetitions: {len(non_rep_paraphrase_ids)}")

    return repetition_ids, non_rep_paraphrase_ids


def get_highlex_ids_singlepc(df_sorted, print_stats=False):
    #   Get high semantic similarity OR High-Lexical-Similarity Keyword without being Repetition
    #       First: High-Lexical-Similarity label
    p_ids = get_pc_paraphrase_ids(df_sorted)
    high_lex_sim = df_sorted[(df_sorted['QID'].isin(p_ids)) &
                             (df_sorted["Category"] == "Comment") &
                             (~df_sorted["Highlighted"].str.contains("REPETITION", na=False)) &
                             (df_sorted["Highlighted"].str.contains("HIGH-LEXICAL-SIM", na=False))
                             ]["QID"].values.tolist()
    if print_stats:
        print(f"# High lex. Paraphrases (label): {len(high_lex_sim)}")
    return high_lex_sim


def create_tsv_ids_to_check_singlepc(q_ids, df_sorted, interview):
    # save a DataFrame with text guest/host + summary + date + highlighted
    q_df = df_sorted[df_sorted['QID'].isin(q_ids)].copy()
    q_df['Utt'] = ""
    ref_row_cond = (q_df[AnnotationColumns.Highlight_Category] ==
                    ContentReproductionCategories.Referred_to)
    para_row_cond = (q_df[AnnotationColumns.Highlight_Category] ==
                     ContentReproductionCategories.Paraphrase)
    comment_row_cond = (q_df[AnnotationColumns.Highlight_Category] ==
                        ContentReproductionCategories.Question_Comment)
    q_df.loc[ref_row_cond, 'Utt'] = (
        q_df[ref_row_cond].apply(lambda row: interview.get_guest_hl_from_id(
            row[AnnotationColumns.Question_ID], row[AnnotationColumns.Highlight_Choices]),
                                 axis=1))
    q_df.loc[para_row_cond, 'Utt'] = (
        q_df[para_row_cond].apply(lambda row: interview.get_host_hl_from_id(
            row[AnnotationColumns.Question_ID], row[AnnotationColumns.Highlight_Choices]),
                                  axis=1))
    q_df.loc[comment_row_cond, 'Utt'] = q_df[comment_row_cond]['QID'].apply(interview.get_summary_from_id)
    current_time = datetime.now()
    q_df.to_csv(f"{current_time}_ids_to_check.tsv", sep="\t", index=False)


def get_diff_nonpara_ids_singlepc(df_sorted, print_stats=False):
    # Get DIFFICULT: "difficult" in comments
    n_ids, y_ids = get_no_yes_ids_singlepc(df_sorted)
    err_ids = get_error_ids(df_sorted)
    cntxt_ids = get_context_ids(df_sorted)
    ambig_ids = get_ambiguous_ids(df_sorted)
    non_para_ids = get_non_paraphrase_ids(df_sorted)
    rel_n_para_ids = get_related_nonpara_ids_singlepc(df_sorted)
    difficult_ids = df_sorted[(df_sorted['QID'].isin(y_ids) | df_sorted['QID'].isin(n_ids)) &
                              (df_sorted["Category"] == "Comment") &
                              (df_sorted["Highlighted"].str.contains("DIFFICULT", na=False)) &
                              (~df_sorted['QID'].isin(err_ids)) &
                              (~df_sorted['QID'].isin(cntxt_ids)) &
                              (~df_sorted['QID'].isin(ambig_ids)) &
                              (~df_sorted['QID'].isin(rel_n_para_ids))
                              ]["QID"].values.tolist()
    # print(f"# Difficult Parpahrases: {len([q_id for q_id in difficult_ids if q_id in paraphrase_ids])}")
    diff_non_paraphrase = [q_id for q_id in difficult_ids if q_id in non_para_ids]
    if print_stats:
        print(f"# Difficult (Non-Related) Non-Paraphrases: {len(diff_non_paraphrase)}")
    return diff_non_paraphrase


def get_related_nonpara_ids_singlepc(df_sorted, print_stats=False):
    #   Related Non-Paraphrases, i.e., highlighted but "clearn-non-paraprhase"
    n_ids, y_ids = get_no_yes_ids_singlepc(df_sorted)
    err_ids = get_error_ids(df_sorted)
    cntxt_ids = get_context_ids(df_sorted)
    non_para_ids = get_non_paraphrase_ids(df_sorted)
    # diff_non_para_ids = get_diff_nonpara_ids_singlepc(df_sorted)
    related_ids = df_sorted[(df_sorted['QID'].isin(y_ids) | df_sorted['QID'].isin(n_ids)) &
                            (df_sorted["Category"] == "Comment") &
                            (df_sorted["Highlighted"].str.contains("ONLY-RELATED", na=False)) &
                            (~df_sorted['QID'].isin(err_ids)) &
                            (~df_sorted['QID'].isin(cntxt_ids))
                            ]["QID"].values.tolist()
    rel_n_para_ids = [q_id
                      for q_id in y_ids if
                      ((q_id in non_para_ids) or (q_id in related_ids))]  # and (q_id not in diff_non_para_ids)
    if print_stats:
        print(f"# Related (incl. difficult) Non-Paraphrases: "
              f"{len(rel_n_para_ids)}")
    return rel_n_para_ids


def evaluate_singlepc_annotations(df_sorted, reannotate_ambig=False, reannotate_cntxt=False):
    get_no_yes_ids_singlepc(df_sorted, print_stats=True)
    get_error_ids(df_sorted, print_stats=True)
    cntxt_ids = get_context_ids(df_sorted, print_stats=True)
    ambigous_ids = get_ambiguous_ids(df_sorted, print_stats=True)
    get_pc_paraphrase_ids(df_sorted, print_stats=True)
    repetition_ids, nr_para_ids = get_non_repetitions_singlepc(df_sorted, print_stats=True)
    non_paraphrase_ids = get_non_paraphrase_ids(df_sorted, print_stats=True)
    diff_non_paraphrase_ids = get_diff_nonpara_ids_singlepc(df_sorted, print_stats=True)
    rel_n_para_ids = get_related_nonpara_ids_singlepc(df_sorted, print_stats=True)
    if reannotate_ambig or reannotate_cntxt:
        interview = interview_data.MediaSumProcessor()
    if reannotate_ambig:
        create_tsv_ids_to_check_singlepc(ambigous_ids, df_sorted, interview)
    if reannotate_cntxt:
        create_tsv_ids_to_check_singlepc(cntxt_ids, df_sorted, interview)
