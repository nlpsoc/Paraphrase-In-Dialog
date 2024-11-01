"""
    ---------- Annotation results constants and functions ----------
    not necessary when using the datasets dataset
"""
import ast
import os
from collections import Counter
from itertools import chain
from typing import List

import pandas as pd

import paraphrase.utility.annotation_df as annotation_df
from paraphrase import interview_data
import paraphrase.utility.qualtrics_api as qac
import paraphrase.utility.qualtrics_survey as qualtrics_survey
from paraphrase.utility.PC_utility import get_qids_from_file
from paraphrase.utility.annotation_pipeline import get_annotators_for_qid, _get_PARA_RAND_qualtrics_ids
from paraphrase.utility.stats import parse_fraction_to_tuple
from paraphrase.utility.project_functions import get_dir_to_src, get_dir_to_result
from paraphrase.utility.annotation_df import merge_transform, AnnotationColumns, get_unique_q_ids, \
    AllAnnotationCategories, \
    ContentReproductionCategories, assert_valid_category, HFIndividualColumns

PC_FOLDER = get_dir_to_result() + "/Annotations/Paraphrase Annotations/5050/Qualtrics/"
ANNOTATIONS_5050_20_PATH = ["ANON_2023-06-13_5050_0-9_July+24,+2023_01.44.tsv",
                            "ANON_2023-06-13_5050_10-19_June+26,+2023_01.01.tsv",
                            "ANON_2023-06-13_5050_20-29_July+24,+2023_01.44.tsv",
                            "ANON_2023-06-13_5050_30-39_June+26,+2023_01.01.tsv",
                            "ANON_2023-06-13_5050_40-49_June+26,+2023_01.02.tsv",
                            "ANON_2023-06-13_5050_50-59_June+26,+2023_01.02.tsv",
                            "ANON_2023-06-13_5050_60-69_June+26,+2023_01.02.tsv",
                            "ANON_2023-06-13_5050_70-79_June+26,+2023_01.02.tsv",
                            "ANON_2023-06-13_5050_80-89_July+24,+2023_01.45.tsv",
                            "ANON_2023-06-13_5050_90-99_June+26,+2023_01.02.tsv"]
ANNOTATIONS_5050_20_PATH = [PC_FOLDER + an_path for an_path in ANNOTATIONS_5050_20_PATH]
ANNOTATIONS_5050_20 = get_dir_to_src() + "/../result/Annotations/Paraphrase Annotations/5050/ANON_100_annotations.tsv"

ALL_ANNOTATIONS = get_dir_to_src() + "/../result/Annotations/Paraphrase Annotations/ANON_all_annotations.tsv"

"""
    GET crowd annotations for specific questions or annotators
"""


def get_aggregated_human_anns(interview=None, question_ids=None):
    question_ids, save_df = get_ann_data(question_ids)
    human_annotations_df = get_crowd_annotations_for_ids(question_ids)

    if interview is None:
        interview = interview_data.MediaSumProcessor()

    human_anns_per_qid = get_annotations_dict(human_annotations_df, interview)

    human_class_per_qid = {}
    for q_id in question_ids:
        human_class_per_qid[q_id] = parse_fraction_to_tuple(human_anns_per_qid[q_id]["vote_str"])
    guest_tokens_per_qid, host_tokens_per_qid = get_tokens_per_qid(human_anns_per_qid, question_ids)

    if save_df:
        save_path = f"/../result/Annotations/Paraphrase Annotations/ANON_{save_df}_TOKEN-annotations.tsv"
        # check if file already exists
        if not os.path.isfile(get_dir_to_src() + save_path):
            # save a version for easier readability (if not yet done so) and in None, DEV, TEST, TRAIN
            q_ids = list(human_anns_per_qid.keys())
            df = pd.DataFrame({"QID": q_ids,
                               "Guest Tokens": [guest_tokens_per_qid[q_id] for q_id in q_ids],
                               "Guest HLs": [human_anns_per_qid[q_id]['guest_weights'] for q_id in q_ids],
                               "Host Tokens": [host_tokens_per_qid[q_id] for q_id in q_ids],
                               "Host HLs": [human_anns_per_qid[q_id]['host_weights'] for q_id in q_ids],
                               "Vote": [human_class_per_qid[q_id] for q_id in q_ids]})

            df.to_csv(get_dir_to_src() + f"/../result/Annotations/Paraphrase Annotations/ANON_{save_df}_TOKEN-annotations.tsv",
                      sep='\t', index=False)

    return guest_tokens_per_qid, host_tokens_per_qid, human_anns_per_qid, human_class_per_qid


def get_tokens_per_qid(human_anns_per_qid, question_ids):
    host_tokens_per_qid = {}
    guest_tokens_per_qid = {}
    for q_id in question_ids:
        host_tokens_per_qid[q_id] = human_anns_per_qid[q_id]["host_tokens"]
        guest_tokens_per_qid[q_id] = human_anns_per_qid[q_id]["guest_tokens"]
    return guest_tokens_per_qid, host_tokens_per_qid


def get_ann_data(question_ids):
    save_df = False
    # question_ids can be a list or everything, or dev/test/train set
    if question_ids is None:
        question_ids = get_unique_q_ids(get_all_crowd_annotations())
        save_df = "ALL"
    elif question_ids == 'DEV':
        question_ids = get_qids_from_file(DEV_PATH)
        save_df = "DEV"
    elif question_ids == 'TEST':
        question_ids = get_qids_from_file(TEST_PATH)
        save_df = "TEST"
    elif question_ids == 'TRAIN':
        question_ids = get_qids_from_file(TRAIN_PATH)
        save_df = "TRAIN"
    return question_ids, save_df


def get_individual_human_anns(interview=None, question_ids=None):
    question_ids, save_df = get_ann_data(question_ids)
    human_annotations_df = get_crowd_annotations_for_ids(question_ids)
    if interview is None:
        interview = interview_data.MediaSumProcessor()
    # get unique annotator, session pairs
    annotator_session_pairs = human_annotations_df[[AnnotationColumns.Annotator_ID,
                                                    AnnotationColumns.Annotation_Session_ID]].drop_duplicates()
    # for each unique annotator, session pair, get the annotations
    rows_to_add = []
    print(annotator_session_pairs)
    print(human_annotations_df)
    for _, row in annotator_session_pairs.iterrows():
        annotator = row[AnnotationColumns.Annotator_ID]
        session = row[AnnotationColumns.Annotation_Session_ID]
        relevant_anns = human_annotations_df[(human_annotations_df[AnnotationColumns.Annotator_ID] == annotator) & (
                    human_annotations_df[AnnotationColumns.Annotation_Session_ID] == session)]
        human_anns_per_qid = get_annotations_dict(relevant_anns, interview)
        for q_id, ann in human_anns_per_qid.items():
            rows_to_add.append({HFIndividualColumns.Question_ID: q_id,
                                HFIndividualColumns.Annotator_ID: annotator,
                                HFIndividualColumns.Session_ID: session,
                                HFIndividualColumns.Is_Paraphrase: parse_fraction_to_tuple(ann["vote_str"])[0],
                                HFIndividualColumns.Guest_Tokens: " ".join(ann["guest_tokens"]),
                                HFIndividualColumns.Guest_Highlights: [int(g_w) for g_w in ann["guest_weights"]],
                                HFIndividualColumns.Host_Tokens: " ".join(ann["host_tokens"]),
                                HFIndividualColumns.Host_Highlights: [int(h_w) for h_w in ann["host_weights"]]})
    df = pd.DataFrame(rows_to_add)
    return df



"""
    GET crowd annotations, total or per experiment
"""


def get_all_crowd_annotations(include_fails: bool = False, recreate: bool = False) -> pd.DataFrame:
    """
        get all crowd-sourced annotations either from the tsv file
            or from the original qualitrics annotation files ("recreate")
        saves it to PC_FOLDER + ALL_ANNOTATIONS
    :param include_fails:
    :param recreate:
    :return:
    """
    if (not recreate) and os.path.isfile(ALL_ANNOTATIONS):
        try:
            return pd.read_csv(ALL_ANNOTATIONS, sep="\t")
        except UnicodeDecodeError:
            return pd.read_csv(ALL_ANNOTATIONS, sep="\t", encoding="utf-16")
    else:
        hl_df_1 = get_5050_crowd_annotations(include_fails=include_fails)
        hl_df_2 = get_PARA_RAND_crowd_annotations(include_fails=include_fails)
        concat_df = pd.concat([hl_df_1, hl_df_2])
        concat_df.to_csv(ALL_ANNOTATIONS, sep='\t', index=False)
        return concat_df


def get_5050_crowd_annotations(include_fails: bool = False) -> pd.DataFrame:
    """
        load the crowd annotations for the 50/50 set
    :param include_fails:
    :return:
    """
    crowd_ann_paths = ANNOTATIONS_5050_20_PATH
    highlight_df = annotation_df.merge_transform(crowd_ann_paths, include_fails=include_fails)
    return highlight_df


def get_PARA_RAND_crowd_annotations(include_fails: bool = False) -> pd.DataFrame:
    """
        for the bigger (500 size) sample, get the annotation data, i.e., for RANDOM and PARA set
    :return:
    """
    qualtrics_files = _get_PARA_RAND_qualtrics_paths()
    highlight_df = merge_transform(qualtrics_files, include_fails=include_fails)
    highlight_df = highlight_df.sort_values(by=AnnotationColumns.Session_Start)
    #       get all unique q ids in highlight_df
    q_ids = get_unique_q_ids(highlight_df)
    #       remove invalid sessions (i.e., those that, e.g., handed in twice etc.) that are not in DB
    invalid_sessions = set()
    for q_id in q_ids:
        # for highlight df get all rows with q_id
        annotators = get_annotators_for_qid(q_id)
        # Remove sessions of annotators that are not valid
        invalid_sessions.update(
            highlight_df[(highlight_df['QID'] == q_id) & (~highlight_df['Annotator'].isin(annotators))][
                "Session"].unique())
        # remove double sessions of annotators
        potentially_valid_sessions = \
            highlight_df[(highlight_df['QID'] == q_id) & (highlight_df['Annotator'].isin(annotators))][
                "Session"].unique()
        matching_annotators = [highlight_df[highlight_df["Session"] == session]["Annotator"].unique()[0] for session
                               in potentially_valid_sessions]
        if len(matching_annotators) > len(set(matching_annotators)):
            for ann in set(matching_annotators):
                if matching_annotators.count(ann) > 1:
                    session_indices = [i for i, x in enumerate(matching_annotators) if x == ann]
                    invalid_sessions.update(potentially_valid_sessions[session_indices[:-1]])
    if len(invalid_sessions) > 0:
        print(f"Removing invalid sessions: {invalid_sessions}")
        highlight_df = highlight_df[~highlight_df['Session'].isin(invalid_sessions)]
    return highlight_df


"""
    GET crowd annotations for specific questions or annotators
"""


def get_crowd_annotations_for_ids(q_ids: List[str]) -> pd.DataFrame:
    """
        get the crowd annotations for the given question ids
    :param q_ids:
    :return:
    """
    all_annotations = get_all_crowd_annotations()
    return all_annotations[all_annotations['QID'].isin(q_ids)].reset_index(drop=True)


def get_crowd_annotations_for_annotators(annotators: List[str]) -> pd.DataFrame:
    """
        get the crowd annotations for the given annotators
    :param annotators:
    :return:
    """
    all_annotations = get_all_crowd_annotations()
    return _get_sub_df_from_annotators(all_annotations, annotators)


"""
    helper functions
"""


def _get_sub_df_from_annotators(all_crowd_annotations_df, subset_annotators):
    """
        get the subset of the crowd annotations dataframe annotated by a given set of annotators
    :param all_crowd_annotations_df:
    :param subset_annotators:
    :return:
    """
    crowd_anns_df = all_crowd_annotations_df[
        all_crowd_annotations_df[AnnotationColumns.Annotator_ID].isin(subset_annotators)]
    return crowd_anns_df.reset_index(drop=True)


def _get_PARA_RAND_qualtrics_paths(test: bool = False, test_df: object = None) -> List[str]:
    """
        for the second experiment, we load the the original qualtrics files and build the annotations from those
        ASSUMES qualtrics files were downloaded beforehand
    :param test:
    :param test_df:
    :return:
    """
    qualtrics_ids = _get_PARA_RAND_qualtrics_ids(test=test, test_df=test_df)
    download_folder = qac.get_automated_download_folder()
    return [download_folder + f"{q_id}.tsv" for q_id in qualtrics_ids]


def get_txt_highlight(q_id: str, highlight_choices: List[List[int]], interview: interview_data.MediaSumProcessor,
                      on_guest_utt=False, chunked=False) -> List[str]:
    """

    :param q_id:
    :param highlight_choices: if None returns the whole tokenized text
    :param interview:
    :param on_guest_utt:
    :param chunked:
    :return: list of strings that includes the highlights according to the highlight_choices
    """
    if any(c_list != [] for c_list in highlight_choices):
        choice_to_text = _get_tokenized_org_utt(q_id, on_guest_utt, interview)
        return [" ".join([choice_to_text[i - 1] for i in choice])  # highlight choices start with 1
                for choice in highlight_choices]

    return ["" for _ in highlight_choices]


def _get_tokenized_org_utt(q_id, on_guest_utt, interview):
    q_dict = interview.get_qdict_from_qid(q_id)
    guest_utt = q_dict["guest utterance"]
    org_utt = q_dict["host utterance"]
    utt = org_utt
    if on_guest_utt:
        utt = guest_utt
    choice_to_text = qualtrics_survey.tokenize_for_highlight_choices(utt)
    return choice_to_text


def get_aggregated_annotations(highlight_df, interview=None, test=False):
    if interview is None:
        interview = interview_data.MediaSumProcessor()
    unique_ann_tuples = annotation_df.get_annotation_tuples(highlight_df)
    # for each unique annotation, get the annotation data
    g_weights = []
    g_tokens = []
    h_weights = []
    h_tokens = []
    votes = []
    for i, (q_id, ann) in enumerate(unique_ann_tuples):
        if test and i > 10:
            break
        ann_df = highlight_df[(highlight_df[AnnotationColumns.Question_ID] == q_id) &
                              (highlight_df[AnnotationColumns.Annotator_ID] == ann)]
        g_token_weights, guest_tokens, h_token_weights, host_tokens, q_ids, vote_strs = _get_annotations_per_qid(ann_df,
                                                                                                                 interview)
        assert (len(q_ids) == 1)
        g_weights.append(g_token_weights[0])
        g_tokens.append(guest_tokens[0])
        h_weights.append(h_token_weights[0])
        h_tokens.append(host_tokens[0])
        assert "1" in vote_strs[0] or "0" in vote_strs[0], f"vote_strs: {vote_strs}"
        votes.append(1 if "1" in vote_strs[0] else 0)

    return {
        "annotation_ids": unique_ann_tuples[:len(g_weights)],
        "g_weights": g_weights,
        "g_tokens": g_tokens,
        "h_weights": h_weights,
        "h_tokens": h_tokens,
        "votes": votes
    }


def _get_annotations_per_qid(highlight_df, interview=None):
    if interview is None:
        interview = interview_data.MediaSumProcessor()
    #     ContentReproductionCategories.Paraphrase or  AllAnnotationCategories.Referred_to
    g_token_weights, guest_tokens, q_ids, vote_strs = extract_txts_to_compare(AllAnnotationCategories.Referred_to,
                                                                              highlight_df,
                                                                              interview,
                                                                              get_hl_texts=False)
    h_token_weights, host_tokens, q_h_ids, _ = extract_txts_to_compare(ContentReproductionCategories.Paraphrase,
                                                                       highlight_df,
                                                                       interview,
                                                                       get_hl_texts=False)
    # if q_ids from the two are not of the same order, re-order them
    if q_ids != q_h_ids:
        # reorder h_token_weights and host_tokens
        h_token_weights = [h_token_weights[q_h_ids.index(q_id)] for q_id in q_ids]
        host_tokens = [host_tokens[q_h_ids.index(q_id)] for q_id in q_ids]
    return g_token_weights, guest_tokens, h_token_weights, host_tokens, q_ids, vote_strs


def extract_txts_to_compare(ref_cat, report_df, interview, chunked=False, get_hl_texts=True, single_annotator=False):
    """
        for a given "category", i.e., guest or host utterance, return
            candidates (ALL highlightings for each question as given by q_ids), votes per token, i.e., [[1.0, 0.9, 0.9, 0.9], [0, 0, 0, 0, ...], ...]
            tok_utts: for the original utterances tokenized lists, i.e., [['This', 'is', 'not', 'good.'], ['Well', 'earlier', 'this', 'month,', ...], ...]
            q_ids (all considered questions),i.e., ['CNN-177596-7', 'NPR-8678-6', ...]
            par_votes: paraphrase votes, i.e., ['10 / 20', '5 / 20', '19 / 20', ...]
        if get_hl_texts == True:
            candidates include all highlighted text

    :param chunked:
    :param interview:
    :param ref_cat:
    :param report_df:
    :return:
    """
    assert_valid_category(ref_cat)
    on_guest_utt = ref_cat == AllAnnotationCategories.Referred_to
    choices_df = report_df[report_df[AnnotationColumns.Highlight_Category] == ref_cat]
    q_ids = list(choices_df[AnnotationColumns.Question_ID].unique())
    candidates = []
    tok_utts = []
    par_votes = []
    for q_id in q_ids:

        # get the highlighting with indices per annotator
        q_choices = choices_df[choices_df[AnnotationColumns.Question_ID] == q_id][
            AnnotationColumns.Highlight_Choices].tolist()
        q_choices = [ast.literal_eval(c_list) for c_list in q_choices]
        tok_utt = _get_tokenized_org_utt(q_id, on_guest_utt, interview)
        tok_utts.append(tok_utt)
        votes_for_paraphrase = sum(1 for sublist in q_choices if len(sublist) > 0)
        par_votes.append(f"{votes_for_paraphrase} / {len(q_choices)}")

        if get_hl_texts:
            # extract the text from the highlight indices
            all_hl_text = get_txt_highlight(q_id, q_choices, interview,
                                            on_guest_utt=on_guest_utt, chunked=chunked)
            all_hl_text = [hl_txt for hl_txt in all_hl_text if hl_txt != ""]
            candidates.append(all_hl_text)
        else:
            # weight_per_token: # times a token was chosen (index from 0 +1)
            weight_per_token = Counter(chain(*q_choices))
            weight_per_token = [
                weight_per_token[i_tok + 1] / max(votes_for_paraphrase, 1) if (i_tok + 1) in weight_per_token else 0
                for i_tok in range(len(tok_utt))]
            candidates.append(weight_per_token)
    return candidates, tok_utts, q_ids, par_votes


TEST_PATH = get_dir_to_src() + "/../result/Annotations/Paraphrase Annotations/ANON_test.tsv"
DEV_PATH = get_dir_to_src() + "/../result/Annotations/Paraphrase Annotations/ANON_dev.tsv"
TRAIN_PATH = get_dir_to_src() + "/../result/Annotations/Paraphrase Annotations/ANON_train.tsv"


def get_annotations_dict(highlight_df, interview=None):
    g_token_weights, guest_tokens, h_token_weights, host_tokens, q_ids, vote_strs = (
        _get_annotations_per_qid(highlight_df, interview))
    result_dict = {}
    for q_index, q_id in enumerate(q_ids):
        result_dict[q_id] = {"guest_tokens": guest_tokens[q_index],
                             "guest_weights": g_token_weights[q_index],
                             "host_tokens": host_tokens[q_index],
                             "host_weights": h_token_weights[q_index],
                             "vote_str": vote_strs[q_index]}
    return result_dict
