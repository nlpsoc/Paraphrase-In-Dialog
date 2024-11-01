"""
    utility script for the dataframe setup in which we save highlighting of the (guest, host)-pairs
"""
import math
from typing import List

import numpy as np
import pandas as pd

import paraphrase.utility.qualtrics_survey as qualtrics_survey
from paraphrase.utility.qualtrics_survey import REFERRING_CHOICES, PARAPHRASE_CANDIDATE_CHOICES


class HFIndividualColumns:
    Question_ID = 'QID'
    Annotator_ID = 'Annotator'
    Session_ID = 'Session'
    Is_Paraphrase = 'Is Paraphrase'
    Guest_Tokens = 'Guest Tokens'
    Guest_Highlights = 'Guest Highlights'
    Host_Tokens = 'Host Tokens'
    Host_Highlights = 'Host Highlights'

class AnnotationColumns:
    Question_ID = 'QID'
    Highlight_Category = 'Category'
    Highlight_Choices = 'Highlighted'
    Annotator_ID = 'Annotator'
    Annotation_Session_ID = 'Session'
    Session_Start = 'Session_Start'


class AllAnnotationCategories:
    Referred_to = 'Referred'
    Rephrase = 'Rephrase'
    Interpretation = 'Interpretation'
    Elaboration_Information = 'Information'
    Elaboration_Question = 'Question'
    Question_Comment = 'Comment'
    Referring_Choice = 'Is Referring'
    Duration = 'Time'
    EndTime = 'EndTime'
    AC_fails = '# failed ACs'
    Stopped = 'Stopped'


class ContentReproductionCategories:
    Referred_to = 'Referred'
    Repeat = 'Repetition'
    Paraphrase = 'Paraphrase'
    Reference = 'Reference'
    Question_Comment = 'Comment'
    Referring_Choice = 'Is Referring'


class QualtricsColumns:
    Author_ID = 'Q-PID'
    Response_ID = 'ResponseId'
    Start_Date = 'StartDate'
    Progress = 'Progress'
    Q_RESPONSE_TYPE = 'Q_RESPONSE_TYPE'
    Comment = 'Q-Comment'
    Quality_Fails = 'Quality_Fails'


"""" ------------- wrangling annotation dataframe ---------------- """


def transform_hl_report(tsv_pd, paraphrase_candidate: bool = False, include_history: bool = True,
                        content_reproduction: bool = True, manual_admission: List[str] = None,
                        manual_removal=None, include_fails: bool = False):
    """
        transform from the native qualtrics report tsv format to one that only contains the relevant data

            QID	                Category	    Highlighted	                Annotator	    Session              Session_start
            CNN-254960-3	    Referred-to	    [44,45,46,47,48,49]	        Annotator1	        R_1M4I8chDvwwFCTr    Timestamp
            CNN-254960-3	    Rephrase	    [10,11,12,13,14,15,16,17]	Annotator1	        R_1M4I8chDvwwFCTr
            CNN-254960-3	    Is Referring	Yes	                        Annotator1	        R_1M4I8chDvwwFCTr
            CNN-254960-3	    Comment	        “”	                        Annotator1	        R_1M4I8chDvwwFCTr
            R_1M4I8chDvwwFCTr	Comment	        “”	                        Annotator1	        R_1M4I8chDvwwFCTr
            R_1M4I8chDvwwFCTr   Time            304934                      Annotator1            R_1M4I8chDvwwFCTr
            R_1M4I8chDvwwFCTr   EndTime         2023-06-13 06:27:11         Annotator1            R_1M4I8chDvwwFCTr
            R_1M4I8chDvwwFCTr   Quality Fails   0                           Annotator1            R_1M4I8chDvwwFCTr
            R_1M4I8chDvwwFCTr   Stopped         0                           Annotator1            R_1M4I8chDvwwFCTr

        For paraphrase candidate:

            QID	                Category	    Highlighted	                Annotator	    Session
            CNN-254960-3	    Is Referring	Yes	                        Annotator1	        R_1M4I8chDvwwFCTr
            CNN-254960-3	    Comment	        “”	                        Annotator1	        R_1M4I8chDvwwFCTr
            R_1M4I8chDvwwFCTr	Comment	        “”	                        Annotator1	        R_1M4I8chDvwwFCTr


    :param manual_removal:
    :param manual_admission: manually screen prolific users to be added
    :param paraphrase_candidate: whether it is the paraphrase candidate survey
    :param tsv_pd: pandas dataframe of the original tsv file as downloaded from qualtrics
        of the form as represented in "Dummy_Qualtrics-Annotations.tsv"
    :return:
    """
    if manual_removal is None:
        manual_removal = ["test", "63f77b9ca8439f6e1620c1cd"]
    if manual_admission is None:
        manual_admission = []
    hl_categories = [AllAnnotationCategories.Referred_to, AllAnnotationCategories.Rephrase,
                     AllAnnotationCategories.Interpretation, AllAnnotationCategories.Elaboration_Information,
                     AllAnnotationCategories.Elaboration_Question]
    referring_choice = AllAnnotationCategories.Referring_Choice
    str_categories = [AllAnnotationCategories.Question_Comment, referring_choice]
    identifier_to_category = {
        qualtrics_survey.R2_Identifier: AllAnnotationCategories.Referred_to,
        qualtrics_survey.ContentReproduction_Identifier: AllAnnotationCategories.Rephrase,
        qualtrics_survey.Interpretation_Identifier: AllAnnotationCategories.Interpretation,
        qualtrics_survey.ElaborationStatement_Identifier: AllAnnotationCategories.Elaboration_Information,
        qualtrics_survey.Question_Identifier: AllAnnotationCategories.Elaboration_Question,
        qualtrics_survey.QComment_Identifier: AllAnnotationCategories.Question_Comment,
        qualtrics_survey.Choice_Identifier: referring_choice  # 2 (not visible), 3 (not referring), 4 (standard)
    }
    if content_reproduction:
        hl_categories = [ContentReproductionCategories.Referred_to, ContentReproductionCategories.Repeat,
                         ContentReproductionCategories.Reference, ContentReproductionCategories.Paraphrase]
        referring_choice = ContentReproductionCategories.Referring_Choice
        str_categories = [ContentReproductionCategories.Question_Comment, referring_choice]
        identifier_to_category = {
            qualtrics_survey.ContentReproductionIdentifiers.R2_Identifier: ContentReproductionCategories.Referred_to,
            qualtrics_survey.ContentReproductionIdentifiers.Repeat_Identifier: ContentReproductionCategories.Repeat,
            # legacy
            qualtrics_survey.ContentReproductionIdentifiers.Reference_Identifier: ContentReproductionCategories.Reference,
            # legacy
            "I": AllAnnotationCategories.Interpretation,  # legacy
            "EI": AllAnnotationCategories.Elaboration_Information,  # legacy
            "Q": AllAnnotationCategories.Elaboration_Question,  # legacy
            qualtrics_survey.ContentReproductionIdentifiers.Paraphrase_Identifier: ContentReproductionCategories.Paraphrase,
            qualtrics_survey.QComment_Identifier: ContentReproductionCategories.Question_Comment,
            qualtrics_survey.Choice_Identifier: ContentReproductionCategories.Referring_Choice
            # 2 (not visible), 3 (not referring), 4 (standard)
        }
    cat_to_identifier = {v: k for k, v in identifier_to_category.items()}
    is_referring = REFERRING_CHOICES
    if paraphrase_candidate:
        is_referring = PARAPHRASE_CANDIDATE_CHOICES

    # future columns in the dataframe
    #   QID	                Category	    Highlighted	                Annotator	    Session
    question_ids = []
    categories = []
    highlights = []
    annotators = []
    sessions = []
    session_starts = []

    for i, annotation_row in tsv_pd.iterrows():  # iterate over the sessions

        if i < 2:  # ignore the first lines that are meta data
            continue

        annotator = annotation_row[QualtricsColumns.Author_ID]
        session = annotation_row[QualtricsColumns.Response_ID]
        session_start = annotation_row[QualtricsColumns.Start_Date]
        stopped = 1 if int(annotation_row[QualtricsColumns.Progress]) < 100 else 0

        if annotator in manual_removal:  # ignore test submissions
            continue

        if include_fails:
            # stopped = 1 if that is the case
            _append_row_tolists(session, AllAnnotationCategories.Stopped, stopped, annotator, session, session_start,
                                question_ids, categories, highlights, annotators, sessions, session_starts)

        if (annotation_row[QualtricsColumns.Q_RESPONSE_TYPE] == "In Process") and (
                (not manual_admission) or (
                annotator not in manual_admission)):  # failed attention check, failed CC, no consent or aborted
            if include_fails:
                _append_row_tolists(session, AllAnnotationCategories.AC_fails,
                                    annotation_row[QualtricsColumns.Quality_Fails],
                                    annotator, session, session_start,
                                    question_ids, categories, highlights, annotators, sessions, session_starts)
            continue

        highlight_dict = {}

        for column, value in annotation_row.items():  # iteritems()

            if column == QualtricsColumns.Comment:  # this is the "any last comments?" question
                # --> R_1M4I8chDvwwFCTr	Comment	        “”	                        Annotator1	        R_1M4I8chDvwwFCTr
                _append_row_tolists(session, AllAnnotationCategories.Question_Comment, value, annotator, session,
                                    session_start,
                                    question_ids, categories, highlights, annotators, sessions, session_starts)
                # also add the duration time here
                _append_row_tolists(session, AllAnnotationCategories.Duration, annotation_row["Duration (in seconds)"],
                                    annotator, session, session_start,
                                    question_ids, categories, highlights, annotators, sessions, session_starts)
                if QualtricsColumns.Quality_Fails in annotation_row:
                    # add quality fails
                    _append_row_tolists(session, AllAnnotationCategories.AC_fails,
                                        annotation_row[QualtricsColumns.Quality_Fails],
                                        annotator, session, session_start,
                                        question_ids, categories, highlights, annotators, sessions, session_starts)
                # add endtime
                _append_row_tolists(session, AllAnnotationCategories.EndTime, annotation_row["EndDate"],
                                    annotator, session, session_start,
                                    question_ids, categories, highlights, annotators, sessions, session_starts)
                continue

            column_name = column
            # for annotator training the column doesn't start with "Q_"
            if "CNN-13148-5" in column_name:
                column_name = "Q-" + column_name

            if "Q" == column_name[0]:  # is this one of the question ids
                if len(column_name.split('_')[0]) > 1 \
                        and (
                        "NPR" in column_name.split('_')[0] or "CNN" in column_name.split('_')[
                    0]):  # checks if mediasum triple

                    q_id = column_name.split('_')[0]
                    q_id = q_id[q_id.index('-') + 1:]  # change_report_id(column.split('_')[0], start_nbr)
                    identifier = q_id[q_id.rindex('-') + 1:]
                    if identifier == "Timer":  # for now ignore timer questions
                        continue
                    q_id = q_id[:q_id.rindex('-')]
                    # q_nbr = int(column.split('_')[0][1:])
                    # c_nbr = (q_nbr - 15) % 7
                    # category = nbr_to_category[c_nbr]
                    category = identifier_to_category[identifier]
                    highlighting_id = q_id + "_" + identifier

                    if category in hl_categories:
                        choice = int(column_name.split('_')[2])
                        selected = True

                        if highlighting_id not in highlight_dict.keys():
                            highlight_dict[highlighting_id] = []

                        if type(value) != str and math.isnan(value):  # value is nan --> token not highlighted
                            selected = False
                        if selected:
                            highlight_dict[highlighting_id].append(choice)
                    elif category in str_categories:
                        if category == referring_choice:
                            highlight_dict[highlighting_id] = is_referring[int(value)]

                            if int(value) == 2 and not include_history:
                                for hl_cat in hl_categories[1:]:
                                    highlight_dict[q_id + "_" + cat_to_identifier[hl_cat]] = []

                        else:
                            highlight_dict[highlighting_id] = value

        for key, value in highlight_dict.items():
            question_ids.append(key.split("_")[0])
            categories.append(identifier_to_category[(key.split("_")[1])])
            highlights.append(str(value) if type(value) == list else value)
            annotators.append(annotator)
            sessions.append(session)
            session_starts.append(session_start)

    columns = [AnnotationColumns.Question_ID, AnnotationColumns.Highlight_Category, AnnotationColumns.Highlight_Choices,
               AnnotationColumns.Annotator_ID, AnnotationColumns.Annotation_Session_ID, AnnotationColumns.Session_Start]
    result = pd.DataFrame(zip(question_ids, categories, highlights, annotators, sessions, session_starts),
                          columns=columns)

    return result


def get_unique_q_ids(df_or_q_ids: pd.DataFrame):
    """
        for a dataframe generated from a merge_transform call, get the unique ids for (guest, host)-pairs
    :param df_or_q_ids:
    :return:
    """
    if type(df_or_q_ids) == pd.DataFrame:
        df_ids = df_or_q_ids[AnnotationColumns.Question_ID].unique().tolist()
    elif type(df_or_q_ids) == list:
        df_ids = df_or_q_ids
    else:
        raise ValueError("df must be either a pandas dataframe or a list of question ids")
    q_ids = [q_id for q_id in df_ids if "R_" not in q_id]
    return q_ids


def get_unique_interview_ids(df_or_q_ids: pd.DataFrame):
    """
        for dataframe generated with merge_transform get the unique q ids
    :param df_or_q_ids:
    :return:
    """
    if type(df_or_q_ids) == pd.DataFrame:
        q_ids = get_unique_q_ids(df_or_q_ids)
    elif type(df_or_q_ids) == list:
        q_ids = df_or_q_ids
    else:
        raise ValueError("df must be either a pandas dataframe or a list of question ids")
    i_ids = [q_id[:q_id.rfind('-')] for q_id in q_ids if "R_" not in q_id]
    return list(set(i_ids))


def get_annotators(transformed_df):
    """
        given the dataframe of annotations, get the list of unique annotators
    :param transformed_df:
    :return: list of annotator identifiers
    """
    annotators = transformed_df[AnnotationColumns.Annotator_ID].unique()
    # if len(annotators) != len(transformed_df[AnnotationColumns.Annotation_Session_ID].unique()):
    #     warnings.warn(
    #         f"There are less annotators ({len(annotators)}) "
    #         f"than sessions ({len(transformed_df[AnnotationColumns.Annotation_Session_ID].unique())}). "
    #         f"This is okay if it was allowed for annotators to annotate several times.")
    return list(annotators)


def get_number_annotations(transformed_df):
    """

    :param transformed_df:
    :return:
    """
    vote_tuples = get_vote_tuples(transformed_df)
    total_annotations = sum(den for _, den in vote_tuples["votes"])
    return total_annotations


def get_vote_shares(highlight_df):
    """
        For a highlight dict of datastructure.transform_hl_report form,
        return the share of annotators voting for is a paraphrase ("votes")
    :param highlight_df:
    :return:
    """
    bin_dict = get_annotation_dict(highlight_df)
    complete_binaries = np.array(bin_dict["complete_binaries"])
    paraphrase_shares = calc_vote_share(complete_binaries)
    return {
        "votes": paraphrase_shares,
        "q_ids": bin_dict["q_ids"]
    }


def get_vote_tuples(highlight_df):
    """
        For a highlight dict of datastructure.transform_hl_report form,
        return the share of annotators voting for is a paraphrase ("votes")
    :param highlight_df:
    :return:
    """
    bin_dict = get_annotation_dict(highlight_df)
    complete_binaries = np.array(bin_dict["complete_binaries"])
    paraphrase_shares = _calc_vote_tuples(complete_binaries)
    return {
        "votes": paraphrase_shares,
        "q_ids": bin_dict["q_ids"]
    }


def get_annotation_tuples(highlight_df):
    """
    For a highlight dict of datastructure.transform_hl_report form,
        return the unique (Q-ID, annotator) tuples
    :param highlight_df:
    :return:
    """
    ann_dict = get_annotation_dict(highlight_df)
    anns_per_item = ann_dict["temporally_ordered_annotators"]
    q_ids = ann_dict["temporally_ordered_q_ids"]
    return [(q_id, ann) for q_id, ann_list in zip(q_ids, anns_per_item) for ann in ann_list]


""" ------------- loading annotation dataframe from tsv files ---------------- """


def merge_transform(paths: List[str], paraphrase_candidate: bool = False, include_history: bool = False,
                    manual_admission: List[str] = None, manual_removal: List[str] = None, include_fails: bool = False):
    """
        transforms a list of qualtrics results tsvs to a pandas dataframe,
        removes annotators that are not valid (signified by Submission "In Process"), i.e., fail ACs or CCs
    :param manual_admission:
    :param paraphrase_candidate:
    :param paths:
    :param include_history:
    :return:
    """
    if manual_admission is None:
        manual_admission = []
    pd_report = pd.read_csv(paths[0], sep="\t", encoding="utf-16")
    pd_report = transform_hl_report(pd_report, paraphrase_candidate=paraphrase_candidate,
                                    include_history=include_history, manual_admission=manual_admission,
                                    manual_removal=manual_removal, include_fails=include_fails)
    for p in paths[1:]:
        pd_tmp = pd.read_csv(p, sep="\t", encoding="utf-16")
        pd_tmp = transform_hl_report(pd_tmp, paraphrase_candidate=paraphrase_candidate, include_history=include_history,
                                     manual_admission=manual_admission, manual_removal=manual_removal,
                                     include_fails=include_fails)
        pd_report = pd.concat([pd_report, pd_tmp], ignore_index=True)
    return pd_report


def _append_row_tolists(q_id, category_str, highlights_str, annotator, session, session_start,
                        question_ids, categories, highlights, annotators, sessions, session_starts):
    question_ids.append(q_id)
    categories.append(category_str)
    highlights.append(highlights_str)  # if type(value) == str else ""
    annotators.append(annotator)
    sessions.append(session)
    session_starts.append(session_start)


def calc_vote_share(complete_binaries):
    """
        calculate vote share based on a binary
    :param complete_binaries:
    :return:
    """
    return [np.count_nonzero(col == 1) / np.count_nonzero((col == 0) | (col == 1))
            for col in complete_binaries.T]


def _calc_vote_tuples(complete_binaries):
    return [(np.count_nonzero(col == 1), np.count_nonzero((col == 0) | (col == 1)))
            for col in complete_binaries.T]


def get_annotation_dict(transformed_df, paraphrase_candidate: bool = False):
    """
        TODO: return with ordering of annotators
        Get binary information whether annotator selected paraphrase is present (or not)
    :param paraphrase_candidate:
    :param transformed_df: highlight dict of datastructure.transform_hl_report
    :return:
        binaries: per annotator annotations (Dimensions: Annotators x Question_IDs),
            with annotators sorted according to "annotators"
        complete_binaries: includes nan, important when an annotator didn't annotate everything
            and thus different from "binaries"
        q_ids: per annotator what questions were annotated
        annotators: annotator ids
    """
    # check if dataframe is empty
    if transformed_df.empty:
        return None
    # assert_valid_category(cat)
    annotators = get_annotators(transformed_df)
    binary_annotations = [[] for _ in range(len(annotators))]
    comment_annotations = [[] for _ in range(len(annotators))]
    q_ids_per_annotators = [[] for _ in range(len(annotators))]
    q_ids = get_unique_q_ids(transformed_df)

    merged_df = get_binary_annotation_dataframe(transformed_df, paraphrase_candidate=paraphrase_candidate)

    # order according to tempral order of annotations
    merged_df = merged_df.sort_values(by=AnnotationColumns.Session_Start)
    temporally_ordered_classifications = [[] for _ in
                                          range(len(q_ids))]  # list of lists for each question ID as given in q_ids
    temporally_ordered_annotators = [[] for _ in
                                     range(len(q_ids))]  # list of lists for each question ID as given in q_ids
    temporally_ordered_sessions = [[] for _ in
                                   range(len(q_ids))]  # list of lists for each question ID as given in q_ids
    for (i, row) in merged_df.iterrows():
        cur_q_id = row[AnnotationColumns.Question_ID]
        temporally_ordered_annotators[q_ids.index(cur_q_id)].append(row[AnnotationColumns.Annotator_ID])

        if (not paraphrase_candidate and row[AnnotationColumns.Highlight_Choices + "_x"] == "[]") \
                or (paraphrase_candidate and row[AnnotationColumns.Highlight_Choices + "_x"] == "No"):
            cur_classification = 0
        else:
            cur_classification = 1
        temporally_ordered_classifications[q_ids.index(cur_q_id)].append(cur_classification)
        temporally_ordered_sessions[q_ids.index(cur_q_id)].append(row[AnnotationColumns.Annotation_Session_ID])

        # legacy annotation data
        an_index = annotators.index(row[AnnotationColumns.Annotator_ID])
        q_ids_per_annotators[an_index].append(cur_q_id)
        comment_annotations[an_index].append(row[AnnotationColumns.Highlight_Choices + "_y"])
        binary_annotations[an_index].append(cur_classification)

    complete_nan_annotations, all_q_ids = _transform_separate_binaries_to_combined_nan_annotations(binary_annotations,
                                                                                                   q_ids_per_annotators)

    return {
        "binaries": binary_annotations,
        "qids_per_binary": q_ids_per_annotators,
        "complete_binaries": complete_nan_annotations,
        "q_ids": all_q_ids,
        "annotators": annotators,
        "comments": comment_annotations,
        "temporally_ordered_classifications": temporally_ordered_classifications,
        "temporally_ordered_annotators": temporally_ordered_annotators,
        "temporally_ordered_q_ids": q_ids,
        "temporally_ordered_sessions": temporally_ordered_sessions
    }


def get_binary_annotation_dataframe(transformed_df, paraphrase_candidate: bool = False):
    category = ContentReproductionCategories.Paraphrase  # just need to look at one category to check if yes/no
    annotators = get_annotators(transformed_df)
    if paraphrase_candidate:

        # get all choices (is possibly paraphrase or is NOT paraphrase)
        an_choices = transformed_df[
            transformed_df[AnnotationColumns.Highlight_Category] == ContentReproductionCategories.Referring_Choice
            ]
        # get all comments
        comments = transformed_df[
            (transformed_df[AnnotationColumns.Highlight_Category] == ContentReproductionCategories.Question_Comment)
            & (~transformed_df[AnnotationColumns.Question_ID].str.startswith("R_"))]

    else:
        # ref_choices = pd.DataFrame()
        an_choices = transformed_df[
            (transformed_df[AnnotationColumns.Annotator_ID].isin(annotators)) &
            (transformed_df[AnnotationColumns.Highlight_Category] == category)
            ]
        # get all comments
        comments = transformed_df[
            (transformed_df[AnnotationColumns.Highlight_Category] == ContentReproductionCategories.Question_Comment)
            & (~transformed_df[AnnotationColumns.Question_ID].str.startswith("R_"))]
    merged_df = pd.merge(an_choices, comments, on=[AnnotationColumns.Question_ID, AnnotationColumns.Annotator_ID,
                                                   AnnotationColumns.Annotation_Session_ID,
                                                   AnnotationColumns.Session_Start])
    assert ((len(merged_df) == len(comments)) & (len(comments) == len(an_choices)))
    return merged_df


def assert_valid_category(ref_cat, content_reproduction: bool = True):
    if not content_reproduction:
        assert (ref_cat == AllAnnotationCategories.Rephrase
                or ref_cat == AllAnnotationCategories.Interpretation
                or ref_cat == AllAnnotationCategories.Referred_to
                or ref_cat == AllAnnotationCategories.Elaboration_Information
                or ref_cat == AllAnnotationCategories.Elaboration_Question)
    else:
        assert (ref_cat == ContentReproductionCategories.Repeat
                or ref_cat == ContentReproductionCategories.Paraphrase
                or ref_cat == ContentReproductionCategories.Referred_to
                or ref_cat == ContentReproductionCategories.Reference)


def _transform_separate_binaries_to_combined_nan_annotations(an_per_annotator: List[List[int]],
                                                             qids_per_annotator: List[List[str]]):
    nbr_annotations = sum(len(sublist) for sublist in an_per_annotator)
    deubg_qids = ['CNN-390120-3', 'CNN-58514-3', 'CNN-401386-3', 'CNN-22810-11', 'CNN-237525-9', 'CNN-283887-9',
                  'NPR-23790-3', 'CNN-136098-3', 'CNN-68683-5']
    flat_q_ids = [q_id for an_q_ids in qids_per_annotator for q_id in an_q_ids]
    flat_q_ids = sorted(set(flat_q_ids), key=flat_q_ids.index)
    result = []
    nan = float('nan')
    for a_i in range(len(an_per_annotator)):
        result.append([])
        for q_id in flat_q_ids:
            # if 450 < len(result[a_i]) < 450:
            #     print("debug")
            # if q_id in deubg_qids:
            #     print("debug")
            if q_id in qids_per_annotator[a_i]:
                q_index = qids_per_annotator[a_i].index(q_id)
                result[a_i].append(an_per_annotator[a_i][q_index])
            else:
                result[a_i].append(nan)
    return result, flat_q_ids


def predict(x):
    if np.isnan(x):
        return -5
    elif x >= 0.5:
        return 1
    # elif x == 0.5:
    #     return 0.5
    elif x < 0.5:
        return 0
    else:
        raise ValueError


def read_df_from_list(corrections):
    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()
    # Loop through the list of file paths and read each file into a DataFrame
    for file_path in corrections:
        try:
            temp_df = pd.read_csv(file_path, sep='\t')
        except UnicodeDecodeError:
            temp_df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
        # Append each DataFrame to combined_df
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
    return combined_df


"""
    ==================================================================
    Below assumes only one single annotation per pc (filtering step)
"""


def get_sub_df_from_qids(all_crowd_annotations_df, subset_q_ids):
    crowd_anns_df = all_crowd_annotations_df[all_crowd_annotations_df[AnnotationColumns.Question_ID].isin(subset_q_ids)]
    return crowd_anns_df.reset_index(drop=True)
