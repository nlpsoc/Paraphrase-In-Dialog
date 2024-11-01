"""
    PROLIFIC/QUALTRICS CROSS-OVER
        or: utility file for creating annotation studies (qualtrics) and pushing them for annotation (prolific)
"""
import ast
import os
import warnings
from datetime import datetime
from statistics import mean

import pandas as pd

import paraphrase.qualtrics
import paraphrase.utility.qualtrics_api
import paraphrase.utility.qualtrics_survey
from paraphrase.annotator_allocation import get_min_nbr_anns_entropy
from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.utility.annotation_df import merge_transform
from paraphrase.utility.annotator_allocation_util import get_prediction_share_from_df
from paraphrase.set_id_consts import TEST_ANN_QUALTRICS_ID_1
from paraphrase.interview_data import MediaSumProcessor
from paraphrase.utility.prolific_api import create_training_survey, create_annotation_survey, TRAINING_COMPLETION_CODE, \
    ANNOTATION_COMPLETION_CODE, get_all_annotator_ids_for_study, get_approved_annotators_for_study, \
    get_status_code_for_study, \
    _delete_prolific_study, is_an_annotator_active, stop_study
from paraphrase.utility.qualtrics_api import dowload_survey
from paraphrase.utility.qualtrics_survey import publish_survey, _delete_qualtrics_survey
import paraphrase.utility.qualtrics_api as qac

EMPTY_MANUAL_ADMISSION = []
MAX_STUDIES_ONE_ANNOTATOR = 10
MAX_QUESTIONS_ONE_ANNOTATOR = 50
MIN_ANNOTATIONS = 3
ITEMS_PER_STUDY = 10
SURVEY_NAME = datetime.now().strftime("%Y-%m-%d") + "_Paraphrase-Annotation_500-SAMPLE"

""" ---------- PROLIFIC/QUALTRICS CROSS-OVER"""


def create_and_publish_training_survey(places: int, test=False, exclude=None):
    """
        create and publish qualtrics and prolific survey for training,
        prolific survey still needs to be manually started
    :param exclude:
    :param places:
    :param test:
    :return:
    """
    # create and publish qualtrics training survey
    qualtrics_survey_id, survey_name = paraphrase.qualtrics.create_survey(training=True, test=test, places=places)
    survey_url = publish_survey(qualtrics_survey_id)
    # create matching prolific survey with link to qualtrics training survey
    prolific_id = create_training_survey(survey_url, places=places, test=test, exclude=exclude)
    if test:
        # delete qualtrics survey
        _delete_qualtrics_survey(qualtrics_survey_id)
        # delete prolific survey
        _delete_prolific_study(prolific_id)
    return prolific_id, qualtrics_survey_id, survey_name


def create_and_publish_annotation_survey(
        survey_name, id_list, approved_annotators, min_annotations, interview=None, test=False, seconds_per_item=84):
    """
        create qualtrics annotation survey with given name, qualtrics ids
        and prolific survey which invites approved annotators for a total of min_annotations places

        ATTENTION: PROLIFIC API call doesn't work if approved annotators are not valid prolific IDs
    :param seconds_per_item:
    :param survey_name:
    :param id_list:
    :param approved_annotators:
    :param min_annotations:
    :param interview:
    :param test:
    :return:
    """
    if interview is None:
        interview = MediaSumProcessor()
    qualtrics_survey_id, _ = paraphrase.qualtrics.create_survey(size_sample=len(id_list),
                                                                survey_name=survey_name,
                                                                from_id_list=True,
                                                                id_list=id_list,
                                                                candidate_survey=False,
                                                                interview=interview)
    survey_url = publish_survey(qualtrics_survey_id)
    try:
        assert (min_annotations <= len(approved_annotators))
    except AssertionError:
        warnings.warn("You are publishing a survey with not enough participants to reach your entropy goals. "
                      "This might be inefficent.")
    prolific_id = create_annotation_survey(survey_url, approved_annotators, internal_name=survey_name,
                                           nbr_items=len(id_list),
                                           places=min_annotations,
                                           test=test, seconds_per_item=seconds_per_item)
    if test:
        _delete_qualtrics_survey(qualtrics_survey_id)
        _delete_prolific_study(prolific_id)
    return prolific_id, qualtrics_survey_id


def create_and_publish_annotation_surveys_from_file(approved_annotators,
                                                    filename: str = "../result/Annotations/Paraphrase Candidates/"
                                                                    "Sampled_100_PC_IDs.tsv",
                                                    min_annotations: int = MIN_ANNOTATIONS,
                                                    items_per_study: int = ITEMS_PER_STUDY, test=False, test_df=None):
    """
        create and publish a number of annotation surveys for the given annotators and question IDs (extracted from file)
            CALLS create_and_publish_annotation_survey several times
    """

    items_in_one_study = ITEMS_PER_STUDY
    max_studies_for_one_annotator = MAX_STUDIES_ONE_ANNOTATOR
    if MIN_ANNOTATIONS > len(approved_annotators):
        warnings.warn(f"There are not enough approved annotators to reach "
                      f"the minimum number of annotators per item {min_annotations}. "
                      f"This is okay during testing.")
        min_annotations = min(MIN_ANNOTATIONS, len(approved_annotators))
    q_ids_to_annotate = pd.read_csv(filename, sep="\t")
    q_ids_to_annotate = q_ids_to_annotate["QID"].values.tolist()
    if test:
        print(f"In test mode, only creating a survey for {items_in_one_study} items...")
        # q_ids_to_annotate = q_ids_to_annotate.head(items_in_one_study)
        _, q_ids_to_annotate = get_votes(TEST_ANN_QUALTRICS_ID_1)
    prolific_ids = []
    qualtrics_ids = []
    question_ids = []
    survey_names = []
    annotators = []
    interview = MediaSumProcessor()
    # only assign an annotator to a maximum of 10 studies
    #       --> make distinct groups of annotators that are assigned to a maximum of 10 studies,
    #           i.e., nbr_studies/10 groups
    #   annotators1, ...., annotatorsN -> annotators1, annotatorsnbr_studies/10+1, annotators2*nbr_studies/10+1
    nbr_studies = round(len(q_ids_to_annotate) / items_per_study)
    print(f"Creating {nbr_studies} studies ...")
    nbr_annotator_groups = max(round(nbr_studies / max_studies_for_one_annotator), 1)
    print(f"Dividing {len(approved_annotators)} annotators into {nbr_annotator_groups} groups:")
    assignment = [[] for _ in range(nbr_annotator_groups)]
    for i in range(len(approved_annotators)):
        assignment[i % nbr_annotator_groups].append(approved_annotators[i])
    print(f"\t {assignment}")
    for i in range(nbr_studies):
        survey_name = SURVEY_NAME + f"_{int(i * items_per_study)}-{int(i * items_per_study + items_per_study - 1)}"
        if test:
            survey_name = "TEST_" + survey_name
        id_list = q_ids_to_annotate[i * items_per_study:i * items_per_study + items_per_study]
        cur_anns = assignment[i % nbr_annotator_groups]
        print(f"Creating survey {survey_name} with annotation max of {min_annotations} \n"
              f"\t for annotators {cur_anns} and question IDs {id_list}")
        prolific_id, qualtrics_survey_id = create_and_publish_annotation_survey(survey_name, id_list,
                                                                                cur_anns,
                                                                                min_annotations, interview,
                                                                                test=test)

        survey_names.append(survey_name)
        qualtrics_ids.append(qualtrics_survey_id)
        prolific_ids.append(prolific_id)
        annotators.append(cur_anns)
        question_ids.append(id_list)

    # update tsv with matching annotators
    prolific_qualtrics_df = update_qualtrics_prolific_db(prolific_ids=prolific_ids, qualtrics_ids=qualtrics_ids,
                                                         survey_names=survey_names, question_ids=question_ids,
                                                         admitted_annotators=annotators, test=test, test_df=test_df)
    return prolific_qualtrics_df


def create_and_publish_nthround_annotations(db_df, place_per_study, qid_groups, remaining_anns, survey_names, test,
                                            test_df,
                                            seconds_per_item=84):
    interview = MediaSumProcessor()
    # publish annotations
    pr_ids = []
    qu_ids = []
    for survey_name, place, anns, q_ids in zip(survey_names, place_per_study, remaining_anns, qid_groups):
        # shorten name for qualtrics/prolific upload
        shortened_name = survey_name[0].replace("_Paraphrase-Annotation_500-SAMPLE_", "_")
        pr_id, qu_id = create_and_publish_annotation_survey(shortened_name, q_ids, anns, place, interview,
                                                            test=test, seconds_per_item=seconds_per_item)
        pr_ids.append(pr_id)
        qu_ids.append(qu_id)
    versions = ["|".join(str(int(ver) + 1) for ver in version.split("|")) for _, version in survey_names]
    db_df = update_qualtrics_prolific_db(prolific_ids=pr_ids, qualtrics_ids=qu_ids,
                                         survey_names=[s_name for s_name, version in survey_names],
                                         question_ids=qid_groups,
                                         versions=[
                                             int(version) + 1 if "|" not in version else
                                             "|".join(str(int(ver) + 1) for ver in version.split("|"))
                                             for s_name, version in survey_names
                                         ],
                                         admitted_annotators=remaining_anns,
                                         test=test,
                                         test_df=(None if test_df is None else db_df))
    print(f"Uploaded a total of {len(survey_names)} studies")
    return db_df


""" ------------- ANNOTATOR ALLCOATIONS (ENTROPY CALCULATIONS and the like) -------------- """


def get_votes(qualtrics_id):
    submission_filenames = [dowload_survey(qualtrics_id)]
    submissions_df = merge_transform(submission_filenames, paraphrase_candidate=False)
    if not submissions_df.empty:
        shares, q_ids, nbr_anns = get_prediction_share_from_df(submissions_df)
        return [[fraction * denominator, denominator] for fraction, denominator in zip(shares, nbr_anns)], q_ids
    else:
        raise ValueError("The qualtrics survey doesn't include valid submissions")


def get_next_round_qids(current_question_ids, current_entropy_values, entropy_thresh=0.8, max_annotators=15):
    qids_to_annotate = []
    min_anns = []
    for q_id, vote in zip(current_question_ids, current_entropy_values):
        add_anns = get_min_nbr_anns_entropy(vote[0] / vote[1], vote[1], entropy_thresh, max_annotators)
        if add_anns > 0:
            qids_to_annotate.append(q_id)
            min_anns.append(add_anns)
    if len(min_anns) > 0:
        current_place_study = min(min_anns)
    else:
        current_place_study = None
    return current_place_study, qids_to_annotate


def _update_admitted_annotators(survey_names, remaining_anns, places_per_study, q_ids_per_study,
                                test=False, test_df=None, manual_exclusion=None,
                                manual_inclusion=None, merge=None):
    """
        for all valid annotators, get those that are not yet distributed fully and distribute them to the current anns
        only alters remaining_anns
    :param manual_exclusion:
    :param survey_names: list of survey names that will be pushed after
    :param remaining_anns: list of lists that corresponds to surveys that will be pushed after
    :param test:
    :return:
    """
    # from valid annotators get how many times they already are annotating, i.e., either in progress or in
    #   participated can be redistributed if planned-in + participated < MAX_STUDIES_ONE_ANNOTATOR
    ann_counts = get_valid_ann_counts(manual_admission=manual_inclusion, test=test, test_df=test_df)
    #   remove bad quality annotators (failed AC more than 3 times)
    bad_quality_anns = get_bad_quality_anns(test=test, test_df=test_df)
    #       add the manual exclusion to "bad quality" annotators
    if manual_exclusion is not None:
        bad_quality_anns += manual_exclusion
    #       remove bad quality annotators from remaining and from counts
    remaining_anns = [[ann for ann in anns_for_study if ann not in bad_quality_anns]
                      for anns_for_study in remaining_anns]
    ann_counts = {ann: cnt for ann, cnt in ann_counts.items() if ann not in bad_quality_anns}
    # inverse values, i.e., how many items can they still annotate
    ann_counts = {ann: MAX_QUESTIONS_ONE_ANNOTATOR - cnt for ann, cnt in ann_counts.items()
                  if cnt < MAX_QUESTIONS_ONE_ANNOTATOR}

    # Remove the times that an annotaotr is already planned in acc. to "remaining_anns"
    for ann in ann_counts.keys():
        already_planned_in = 0
        for remain_for_study, next_ids_to_annotate in zip(remaining_anns, q_ids_per_study):
            if ann in remain_for_study:
                already_planned_in += len(next_ids_to_annotate)
        ann_counts[ann] -= already_planned_in
    # remove empties
    ann_counts = {ann: cnt for ann, cnt in ann_counts.items()
                  if cnt > 0}

    # set the priority for annotator distribution
    study_is_empty = [1 if len(r_anns) == 0 else 0 for r_anns in remaining_anns]
    #   first studies that currently have NO annotators,
    #   then number of question IDs to annotate, then places open in the study
    sorted_idx = sorted(enumerate(q_ids_per_study), key=lambda x: (study_is_empty[x[0]],
                                                                   len(x[1]),
                                                                   places_per_study[x[0]]),
                        reverse=True)  # sorted(enumerate(remaining_anns), key=lambda x: len(x[1]))
    sorted_idx = [idx for idx, _ in sorted_idx]
    nothing_added_last_loop = False
    while len(ann_counts.items()) > 0:  # while there are annotators left to assign
        if nothing_added_last_loop:
            break
        nothing_added_last_loop = True
        # remove empties
        ann_counts = {ann: cnt for ann, cnt in ann_counts.items()
                      if cnt > 0}
        # remove those already assigned in ALL studies (or their study history), guarantees end of loop
        ann_counts = {ann: cnt for ann, cnt in ann_counts.items()
                      if not all(ann in remaining_anns[study_id] or
                                 ann in get_participant_history(survey_names[study_id][0], test_df=test_df, test=test)
                                 for study_id in range(len(remaining_anns)))}

        for prioritized_study_id in sorted_idx:
            # for the current study get ALL previous annotators
            previous_participants = get_participant_history(survey_names[prioritized_study_id][0],
                                                            # only name, not version
                                                            test_df=test_df, test=test)
            # try adding an annotator for this study
            for ann, cnt in ann_counts.items():
                if ((ann not in remaining_anns[prioritized_study_id]) and  # annotator not already planned in
                        (ann not in previous_participants) and  # annotator has not yet annotated these ids
                        (len(q_ids_per_study[prioritized_study_id]) < cnt)):  # annotator has still enough room
                    remaining_anns[prioritized_study_id].append(ann)
                    ann_counts[ann] -= len(q_ids_per_study[prioritized_study_id])
                    nothing_added_last_loop = False
                    break
            # no annotator was found for this study ....
    assert (all(len(sublist) > 0 for sublist in remaining_anns))
    return remaining_anns


def merge_ids_for_sublist_to_length(list_of_lists, length=10):
    """
        redistribute the q_ids among the available studies to have studies of length length
    :param list_of_lists:
    :param length:
    :return:
    """
    lens = [len(subl) for subl in list_of_lists]
    if all(l == 0 for l in lens) or all(l >= length for l in lens):
        return [[i] for i, _ in enumerate(list_of_lists)]
    else:
        # Sort the sublists in descending order of their length
        sorted_sublists = sorted(enumerate(list_of_lists), key=lambda x: len(x[1]), reverse=True)

        # Initialize bins for merged sublists
        bins = []

        # Function to find a suitable bin for the sublist
        def find_bin(sublist):
            for bin in bins:
                if sum(len(list_of_lists[idx]) for idx in bin) + len(sublist) <= length:
                    return bin
            return None

        # Pack each sublist into a bin
        for idx, sublist in sorted_sublists:
            bin = find_bin(sublist)
            if bin is not None:
                bin.append(idx)
            else:
                bins.append([idx])

        # Return the bins (representing merges)
        return bins


""" ---------------- DATABASE ------------------------------- """

DB_STARTNAME = "Qualtrics_prolific_ids"
DB_FOLDER = "/../result/Annotations/Paraphrase Annotations/RAND-PARA/DB"


class QualtricsProlificDB:
    ADMITTED = "Admitted Annotators"
    PARTICIPATED = "Participated Annotators"
    VALID = "Valid Annotators"

    def __init__(self, test=False, test_df=None):
        self.db_df = read_db(test=test, test_df=test_df)
        self.test = test
        self.test_df = test_df

    def _get_qualtrics_train_ids(self):
        """
            get the qualtrics IDs of annotators that participated
        :param test:
        :param test_df:
        :return:
        """
        survey_ids = []
        for index, row in self.db_df.iterrows():
            if row["Version"] != "TRAIN":
                continue
            survey_ids.append(row["Qualtrics ID"])
        return survey_ids

    def get_valid_train_annotators(self):
        valid_train_anns = []
        for index, row in self.db_df.iterrows():
            if row["Version"] != "TRAIN":
                continue
            if row["STATUS"] != "COMPLETE":
                continue
            valid_train_anns += ast.literal_eval(str(row["Valid Annotators"]))
        return valid_train_anns

    def get_participated_train_annotators(self):
        participated_train_anns = []
        for index, row in self.db_df.iterrows():
            if row["Version"] != "TRAIN":
                continue
            if row["STATUS"] != "COMPLETE":
                continue
            participated_train_anns += ast.literal_eval(str(row["Participated Annotators"]))
        return participated_train_anns


def read_db(test=False, test_df=None):
    """
        read in database for prolific/qualtrics crossover
    :return:
    """
    if test and test_df is not None:
        return test_df
    if get_highest_integer_filename(test=test) is None:
        return None
    memory_tsv = (get_dir_to_src() + f"{DB_FOLDER}/" + get_highest_integer_filename(test=test))
    return _read_db_from_path(memory_tsv)


def _read_db_from_path(memory_tsv):
    if not os.path.exists(memory_tsv):
        return None
    pq_df = pd.read_csv(memory_tsv, sep="\t")
    pq_df["Participated Annotators"] = pq_df["Participated Annotators"].astype(object)
    pq_df["Valid Annotators"] = pq_df["Valid Annotators"].astype(object)
    pq_df["Entropy"] = pq_df["Entropy"].astype(object)
    return pq_df


def save_db(pq_df, test=False, test_df=None):
    """
        save new df for prolific/qualtrics crossover, saves in a new file
    :param pq_df:
    :param memory_tsv:
    :return:
    """
    if test and test_df is not None:
        return
    tmp_filename = get_highest_integer_filename(test=test)
    memory_tsv = (get_dir_to_src() + f"{DB_FOLDER}/" + ("" if tmp_filename is None else tmp_filename))
    if tmp_filename is None or not os.path.exists(memory_tsv):
        memory_tsv = get_dir_to_src() + f"{DB_FOLDER}/" + ("TEST_" if test else "") + f"{DB_STARTNAME}__0.tsv"
    else:
        db_df = read_db()
        if db_df.equals(pq_df):
            return

    db_ctr = int(memory_tsv.split("__")[-1].split(".")[0])
    new_path = memory_tsv.split("__")[0] + "__" + str(db_ctr + 1) + ".tsv"
    pq_df.to_csv(new_path, sep="\t", index=False)
    return


def get_highest_integer_filename(directory=get_dir_to_src() + DB_FOLDER, test=False):
    """
        generated
    :param directory:
    :return:
    """
    db_startname = DB_STARTNAME
    if test:
        db_startname = "TEST_" + db_startname
    # List all files in the directory
    files = [f for f in os.listdir(directory) if f.startswith(db_startname) and f.endswith('.tsv') and "__" in f]

    # Extract the numbers from filenames and associate with the file
    # For example: 'FILENAME_23.tsv' -> (23, 'FILENAME_23.tsv')
    numbered_files = []
    for f in files:
        try:
            number = int(f.rsplit('_', 1)[-1].split('.')[0])
            numbered_files.append((number, f))
        except ValueError:
            # Skip files that don't have an integer before the '.tsv'
            continue

    # Return the file with the highest number or None if no files found
    return max(numbered_files, default=(None, None))[1]


def get_validated_annotators(manual_admission, test=False, test_df=None):
    """
        Using ONLY the information in the db tsv file, get the approved annotators
        should be called AFTER an update call
    :param manual_admission:
    :param test:
    :return:
    """
    pq_df = read_db(test=test, test_df=test_df)
    if manual_admission is None:
        manual_admission = EMPTY_MANUAL_ADMISSION
    allowlist = manual_admission  # add manual admissions, might not have gone through this round of training
    for index, row in pq_df[(pq_df["Version"] == "TRAIN") &
                            ((pq_df["STATUS"] == "COMPLETE") | (pq_df["STATUS"] == "PENDING"))].iterrows():
        if test and "TEST" not in row["Survey Name"]:
            continue
        if not test and "TEST" in row["Survey Name"]:
            continue
        allowlist += [ann for ann in ast.literal_eval(str(row["Valid Annotators"])) if ann not in manual_admission]
    if test:
        allowlist = list(set(allowlist))
    assert (len(allowlist) == len(set(allowlist)))
    return allowlist


def get_participated_annotators(test=False, test_df=None):
    """
        Using ONLY the information in the db tsv file, get the annotators that participated at all
    :param test_df:
    :param manual_admission:
    :param test:
    :return:
    """
    pq_df = read_db(test=test, test_df=test_df)
    participants = []
    for index, row in pq_df[(pq_df["Version"] == "TRAIN") &
                            ((pq_df["STATUS"] == "COMPLETE") | (pq_df["STATUS"] == "PENDING"))].iterrows():
        if test and "TEST" not in row["Survey Name"]:
            continue
        if not test and "TEST" in row["Survey Name"]:
            continue
        participants += ast.literal_eval(str(row["Participated Annotators"]))
    if test:
        participants = list(set(participants))
    assert (len(participants) == len(set(participants)))
    return participants


def get_bad_quality_anns(bad_quality_means_failing_more_than=3, test=False, test_df=None):
    db_df = read_db(test=test, test_df=test_df)
    failed_count = {}
    for index, row in db_df.iterrows():
        if row["Version"] == "TRAIN":
            continue
        if row["STATUS"] != "COMPLETE":
            continue
        participated = ast.literal_eval(str(row["Participated Annotators"]))
        valid = ast.literal_eval(str(row["Valid Annotators"]))
        for ann in participated:
            if ann not in valid:
                if ann not in failed_count:
                    failed_count[ann] = 1
                else:
                    failed_count[ann] += 1
    return [ann for ann, cnt in failed_count.items() if cnt > bad_quality_means_failing_more_than]


def get_valid_ann_counts(manual_admission=None, db_df=None, test=False, test_df=None):
    """
        for all annotators that passed training or were manually added,
            get the number of items they annotated already
    :param test_df:
    :param manual_admission:
    :param test:
    :param db_df:
    :return: dictionary of valid annotators: assigned items to annotate
    """
    if db_df is None:
        db_df = read_db(test=test, test_df=test_df)
    if manual_admission is None:
        manual_admission = EMPTY_MANUAL_ADMISSION
    valid_anns = get_validated_annotators(manual_admission=manual_admission, test=test, test_df=test_df)
    participated_counts = {}
    for index, row in db_df.iterrows():
        if row["Version"] == "TRAIN":
            continue
        if (row["STATUS"] != "COMPLETE") and (row["STATUS"] != "PENDING"):
            anns_to_count = ast.literal_eval(str(row["Admitted Annotators"]))
        else:
            anns_to_count = ast.literal_eval(str(row["Participated Annotators"]))
        number_annotated_questions = len(ast.literal_eval(str(row["Question IDs"])))
        for ann in anns_to_count:
            if ann not in participated_counts:
                participated_counts[ann] = number_annotated_questions
            else:
                participated_counts[ann] += number_annotated_questions
    for ann in valid_anns:
        if ann not in participated_counts:  # if an annotator has not yet annotated anything
            participated_counts[ann] = 0

    return participated_counts


def get_participant_history(study_name, db_df=None, test=False, test_df=None):
    """
        get all annotators that participated in a study with the given name
        if the given name includes merged studies (i.e., "|" is present),
            get all annotators that participated in any of the studies
    :param study_name:
    :param db_df:
    :param test:
    :param test_df:
    :return:
    """
    if db_df is None:
        db_df = read_db(test=test, test_df=test_df)
    if "|" in study_name:
        study_names = study_name.split("|")
        participants_per_unique_study = [[] for _ in range(len(study_names))]
        for i, sn in enumerate(study_names):
            participants_per_unique_study[i] = get_participant_history(sn, db_df=db_df, test=test, test_df=test_df)
        return list(set([ann for anns in participants_per_unique_study for ann in anns]))
    else:  # study is not merged
        all_participated_anns_through_time = []
        for index, row in db_df.iterrows():
            if study_name not in row["Survey Name"]:
                continue
            if (row["STATUS"] != "COMPLETE") and (row["STATUS"] != "PENDING"):
                participated = ast.literal_eval(str(row["Admitted Annotators"]))
            else:
                participated = ast.literal_eval(str(row["Participated Annotators"]))
            if (len(set(participated) & set(all_participated_anns_through_time)) != 0):
                raise ValueError("Something went wrong")
            all_participated_anns_through_time += participated
        return all_participated_anns_through_time


def check_and_update_db(test=False, test_df=None, manual_inclusion=None, close_pending=False):
    """
        for all surveys in training_surveys, get those annotator IDs that passed training
            and save that in the db "Valid Annotators"
        for all annotation surveys, get those annotators that participated and are marked as valid, and save entropy

        save new db with increased int number
    :param close_pending:
    :param test:
    :return:
    """
    # get manual admissions
    # test_annotators = ["Annotator1", "Annotator2", "Annotator3"]

    pq_df = read_db(test=test, test_df=test_df)
    total_valid = []  # the valid annotators in TRAIN studies

    for index, row in pq_df.iterrows():
        if row["STATUS"] == "COMPLETE":
            if row["Version"] == "TRAIN":
                total_valid += ast.literal_eval(row["Valid Annotators"])
            continue
        if row["Version"] == "TRAIN":
            completion_code = TRAINING_COMPLETION_CODE
        else:
            completion_code = ANNOTATION_COMPLETION_CODE
        if test and "TEST" not in row["Survey Name"]:  # skip non-test cases for test setting
            continue

        anns = get_all_annotator_ids_for_study(row['Prolific ID'], test=test)
        if completion_code == ANNOTATION_COMPLETION_CODE:  # for non-training no non-admitted annotator should have
            # participated
            assert (set(anns).issubset(set(ast.literal_eval(str(row['Admitted Annotators'])))))
        qualtrics_id = row["Qualtrics ID"]
        if test:
            valid_anns = anns
            # qualtrics_id = row["Qualtrics ID"]  # TEST_ANN_QUALTRICS_ID
            pq_df.at[index, "STATUS"] = "COMPLETE"
        else:
            valid_anns = get_approved_annotators_for_study(row['Prolific ID'], completion_code=completion_code)
            # update status to complete only if study is complete in prolific system
            if not get_status_code_for_study(row['Prolific ID']) == "COMPLETED":
                warnings.warn(
                    f"Training study {row['Prolific ID']} with name {row['Survey Name']} might not be done with "
                    f"training/annotation. Setting STATUS to PENDING.")
                pq_df.at[index, "STATUS"] = "PENDING"
                if close_pending:
                    print("close_pending is set to True.")
                    if not is_an_annotator_active(row['Prolific ID']):
                        stop_study(row['Prolific ID'])
                        pq_df.at[index, "STATUS"] = "COMPLETE"
                        print("Closed the pending survey to be fed into the next round.")
                    else:
                        print("Study could not be closed. It has an active submission. Ignoring for now ...")
            else:
                pq_df.at[index, "STATUS"] = "COMPLETE"

        if (manual_inclusion is not None) and (row["Version"] == "TRAIN"):
            valid_anns += [ann for ann in anns if (ann in manual_inclusion) and (ann not in valid_anns)]
        total_valid += valid_anns

        pq_df.at[index, "Participated Annotators"] = anns
        pq_df.at[index, "Valid Annotators"] = valid_anns

        if ((row["Version"] != "TRAIN") and
                ((pq_df.at[index, "STATUS"] == "COMPLETE") or (pq_df.at[index, "STATUS"] == "PENDING"))):
            # Get votes current survey
            f_q_ids = ast.literal_eval(str(row["Question IDs"]))
            try:
                votes, q_ids = get_votes(qualtrics_id)
            except ValueError:
                votes, q_ids = [[0, 0] for _ in range(len(f_q_ids))], f_q_ids
            assert (set(f_q_ids) == set(q_ids))
            cur_votes = [votes[q_ids.index(q_id)] for q_id in f_q_ids]

            # Get previous votes if available
            if ((("|" not in row["Version"]) and (int(row["Version"]) > 0)) or
                    any(int(ver) > 0 for ver in row["Version"].split("|"))):
                # TODO: get votes for each merged study
                if "|" in row["Survey Name"]:
                    survey_names = row["Survey Name"].split("|")
                else:
                    survey_names = [row["Survey Name"]]
                cur_votes = []
                for i, survey in enumerate(survey_names):
                    org_q_ids = ast.literal_eval(str(pq_df[(pq_df["Version"] == "0") &
                                                           (pq_df["Survey Name"] == survey)]["Question IDs"].iloc[0]))
                    cur_version = int(row["Version"].split("|")[i])
                    prev_survey = pq_df[pq_df.apply(lambda x: _check_version_and_survey(x, cur_version - 1, survey),
                                                    axis=1)]
                    prev_qs = ast.literal_eval(str(prev_survey["Question IDs"].iloc[0]))
                    prev_votes = ast.literal_eval(str(prev_survey["Entropy"].iloc[0]))
                    kept_ids = [q_id for q_id in f_q_ids if q_id in org_q_ids]
                    cur_votes += [
                        [elem1 + elem2 for elem1, elem2 in
                         zip(votes[q_ids.index(q_id)], prev_votes[prev_qs.index(q_id)])]
                        for q_id in kept_ids]
            pq_df.at[index, "Entropy"] = cur_votes

    not_included = set(manual_inclusion).difference(set(total_valid))
    if (len(not_included) > 0) and not test:
        all_valid_anns_in_history = get_validated_annotators(None, test=test, test_df=pq_df)
        if not_included.issubset(set(all_valid_anns_in_history)):
            warnings.warn(f"The following annotators were NOT added: {not_included}. "
                          f"This is expected if you added them once before already.")
        else:
            warnings.warn(f"The given set of manually included prolific IDs never participated. "
                          f"{not_included.difference(set(all_valid_anns_in_history))} are missing "
                          f"and are currently not included.")
    save_db(pq_df, test=test, test_df=test_df)
    return pq_df


def _get_PARA_RAND_qualtrics_ids(test=False, test_df=None):
    """
        For second experiment (PARA/RAND),
            get all the relevant IDs that point to qualtrics surveys where there is at least one valid annotator
    :param test:
    :param test_df:
    :return:
    """
    pq_df = read_db(test=test, test_df=test_df)
    # get all qualtrics IDs where there is at least one valid annotator
    survey_ids = []
    for index, row in pq_df.iterrows():
        if row["Version"] == "TRAIN":
            continue
        if row["STATUS"] != "COMPLETE":
            continue
        if test and "TEST" not in row["Survey Name"]:
            continue
        if len(ast.literal_eval(row["Valid Annotators"])) == 0:
            continue
        survey_ids.append(row["Qualtrics ID"])
    return survey_ids


def get_annotators_for_qid(q_id, db_df=None, test=False, test_df=None):
    """
        get all annotators that annotated a given question

    :return:
    """
    if db_df is None:
        db_df = read_db(test=test, test_df=test_df)
    participated = []
    for index, row in db_df.iterrows():
        if row["Version"] == "TRAIN":
            continue
        if q_id not in ast.literal_eval(row["Question IDs"]):
            continue
        if row["STATUS"] == "COMPLETE":
            participated += ast.literal_eval(str(row["Valid Annotators"]))
    assert (len(participated) == len(set(participated)))
    return participated


def _check_version_and_survey(df_row, version, survey_name):
    """
        check if the given version and survey name are in the given row
    :param df_row:
    :param version:
    :param survey_name:
    :return:
    """
    survey_names = df_row["Survey Name"].split("|")
    try:
        survey_index = survey_names.index(survey_name)
        if df_row["Version"].split("|")[survey_index] == str(version):
            return True
        else:
            return False
    except ValueError:
        return False


def _get_finalized_survey_names(db_df):
    # Find surveys that are already done (i.e., that have version "LAST")
    names_with_last = []
    # include cases with "|" in name, i.e., merged studies
    for i, row in db_df.iterrows():
        if "LAST" in row["Version"]:
            if "|" in row["Version"]:
                versions = row["Version"].split("|")
                last_indices = [i for i, v in enumerate(versions) if v == "LAST"]
                names_with_last += [row["Survey Name"].split("|")[i] for i in last_indices]
            else:
                names_with_last += [row["Survey Name"]]
    # names_with_last = db_df[db_df["Version"] == 'LAST']["Survey Name"].unique()
    return names_with_last


def _get_highest_level_annotation_surveys(db_df=None):
    if db_df is None:
        db_df = read_db()
    last_names = _get_finalized_survey_names(db_df)
    result = [(name, "LAST") for name in last_names]
    relevant_df = _get_latest_annotation_surveys(db_df, last_names)
    result += list(zip(relevant_df['Survey Name'], relevant_df['Version']))
    return result


def print_last_annotation_statistics(db_df):
    relevant_df = _get_latest_annotation_surveys(db_df)
    failed_acs = []
    nbr_valid = []
    silent_anns = []
    annotators_needed = []
    for index, row in relevant_df.iterrows():  # only iterates over the latest, but non last surveys
        admitted = ast.literal_eval(str(row['Admitted Annotators']))
        participated = ast.literal_eval(str(row['Participated Annotators']))
        valid = ast.literal_eval(str(row['Valid Annotators']))
        min_nbr_anns, _ = get_next_round_qids(ast.literal_eval(str(row["Question IDs"])),
                                              ast.literal_eval(str(row["Entropy"])))
        # calculate the number of failed attention checks annotators for current study
        failed_acs.append(len(set(participated) - set(valid)))
        nbr_valid.append(len(valid))
        silent_anns.append(len(set(admitted) - set(participated)))
        annotators_needed.append(min_nbr_anns)
    print(f"On average, over {len(failed_acs)} studies,")
    print(f"\t {mean(failed_acs)} annotators failed ACs per study")
    print(f"\t {mean(nbr_valid)} valid annotators participated per study")
    print(f"\t {mean(silent_anns)} annotators were part of the silent audience per study")
    if len([an_n for an_n in annotators_needed if an_n is not None]) > 0:
        print(
            f"\t {mean([an_n for an_n in annotators_needed if an_n is not None])} annotators are needed next per study.")


def _get_latest_annotation_surveys(db_df, finished_surveys=None):
    """
        ignores PENDING and TRAIN surveys
    :param finished_surveys: list of surveys that are already done, i.e., version is at "LAST"
    :param db_df:
    :return:
    """
    if finished_surveys is None:
        finished_surveys = _get_finalized_survey_names(db_df)
    # Filter out rows with surveys that are not in TRAIN and not PENDING

    # Create a copy of the dataframe
    db_df_copy = db_df.copy()
    # Step 1: Split 'Survey Name' and 'Version' columns
    db_df_copy['Survey Name'] = db_df_copy['Survey Name'].str.split('|')
    db_df_copy['Version'] = db_df_copy['Version'].str.split('|')

    # Create a new dataframe with each row corresponding to a single survey name and its corresponding version
    new_df = db_df_copy.explode(['Survey Name', 'Version'])
    # Filter out rows with surveys that are not in TRAIN and not PENDING
    #   ignore surveys that are not yet finished for the current run (e.g., only 1/3 annotated in first round)
    new_df = new_df[~(new_df['Survey Name'].apply(lambda x: any(fs in x for fs in finished_surveys))) &
                    ~(new_df["Version"] == "TRAIN") & ~(new_df["STATUS"] == "PENDING")]
    new_df['Version'] = new_df['Version'].apply(lambda x: int(x))

    # Group by 'Survey Name' and get the index with the maximum 'Version' for each group
    idx = new_df.groupby('Survey Name')['Version'].idxmax()
    idx = sorted(list(set(list(idx))))

    # Use these indices to get the corresponding rows from the original dataframe
    final_df = db_df.loc[idx]

    return final_df


def get_surveys_to_rerun(entropy_thresh, max_annotators, test=False, test_df=None):
    """
        for those annotation studies that are not yet completed (i.e., not "LAST" in Version and not "TRAIN" in version)
            minimum # of annotations remaing for each q_id
            the qids,
            the remaining annotators (i.e., admitted that have not yet participated),.
            all that participated already
            the survey name
        calculated based on entropy calculation
    :param entropy_thresh:
    :param max_annotators:
    :param test:
    :return:
    """
    db_df = read_db(test=test, test_df=test_df)
    names_with_last = _get_finalized_survey_names(db_df)
    relevant_df = _get_latest_annotation_surveys(db_df, names_with_last)
    qid_groups = []
    place_per_study = []
    remaining_anns = []
    participated_anns = []
    survey_names = []
    for index, row in relevant_df.iterrows():
        if test and "TEST" not in row["Survey Name"]:
            continue
        if not test and "TEST" in row["Survey Name"]:
            continue
        if "PENDING" == row["STATUS"]:  # ignore active studies for update
            continue

        # check what q_ids need a new annotation round
        min_nbr_anns, qids_to_annotate = get_next_round_qids(ast.literal_eval(str(row["Question IDs"])),
                                                             ast.literal_eval(str(row["Entropy"])),
                                                             entropy_thresh, max_annotators)
        if len(qids_to_annotate) == 0:  # make the survey the "LAST" if no q_id needs to be annotated again
            #   if not update version to "LAST"
            db_df.loc[(db_df["Survey Name"] == row["Survey Name"]) & (db_df["Version"] == row["Version"]), "Version"] \
                = "LAST" if "|" not in row["Survey Name"] else "|".join("LAST" for _ in row["Version"].split("|"))
        else:
            #   otherwise add to list of surveys
            qid_groups.append(qids_to_annotate)
            place_per_study.append(min_nbr_anns)
            rem_ann = (set(ast.literal_eval(str(row["Admitted Annotators"])))
                       - set(ast.literal_eval(str(row["Participated Annotators"]))))
            remaining_anns.append(list(rem_ann))

            surveys = row["Survey Name"].split("|")
            participated = []
            for i, survey in enumerate(surveys):
                # get original qids
                org_q_ids = ast.literal_eval(str(db_df[(db_df["Version"] == "0") &
                                                       (db_df["Survey Name"] == survey)]["Question IDs"].iloc[0]))
                if len(set(org_q_ids) & set(qids_to_annotate)) == 0:
                    # mark survey as "LAST" as no q_id needs to be annotated again
                    surveys[i] = "LAST"
                else:
                    participated += get_participant_history(survey, db_df=db_df, test=test, test_df=test_df)
            # update the db_df with LAST signifiers
            db_df.loc[(db_df["Survey Name"] == row["Survey Name"]) & (db_df["Version"] == row["Version"]), "Version"] \
                = "|".join("LAST" if surv == "LAST" else ver for surv, ver in zip(surveys, row["Version"].split("|")))

            participated = list(set(participated))
            participated_anns.append(participated)
            new_survey_name = "|".join(surv for surv in surveys if surv != "LAST")
            new_version = "|".join(ver for surv, ver in zip(surveys, row["Version"].split("|")) if surv != "LAST")
            survey_names.append((new_survey_name, new_version))
    save_db(db_df, test=test, test_df=test_df)
    return place_per_study, qid_groups, remaining_anns, participated_anns, survey_names, db_df


def update_qualtrics_prolific_db(prolific_ids, qualtrics_ids, survey_names, question_ids, versions=None, status=None,
                                 admitted_annotators=None, part_annotators=None, valid_annotators=None, entropy=None,
                                 test=False, test_df=None):
    assert (len(prolific_ids) == len(qualtrics_ids) == len(survey_names) == len(question_ids))
    if versions is None:
        versions = [int(0) for _ in range(len(survey_names))]
    if status is None:
        status = ["INIT" for _ in range(len(survey_names))]
    # if type(question_ids) == int:
    #     items_per_study = [int(question_ids) for _ in range(len(survey_names))]
    # if type(annotators) == int:
    #     annotators = [int(annotators) for _ in range(len(survey_names))]
    prolific_qualtrics_df = pd.DataFrame({
        'Survey Name': survey_names,
        'Qualtrics ID': qualtrics_ids,
        'Prolific ID': prolific_ids,
        'Version': versions,
        'STATUS': status,
        'Question IDs': question_ids,
        'Admitted Annotators': admitted_annotators,
        'Participated Annotators': part_annotators,
        'Valid Annotators': valid_annotators,
        'Entropy': entropy
    })
    existing_df = read_db(test=test, test_df=test_df)
    if existing_df is not None:
        final_df = pd.concat([existing_df, prolific_qualtrics_df], ignore_index=True)
    else:
        final_df = prolific_qualtrics_df
    save_db(final_df, test=test, test_df=test_df)
    return final_df


"""
    Download qualtrics surveys
"""


def download_qualtrics_ann_files(test=False, test_df=None):
    """
        get all db files
    :param test:
    :return:
    """
    survey_ids = _get_PARA_RAND_qualtrics_ids(test, test_df)

    return _download_qualtrics_surveys(survey_ids)


def _download_qualtrics_surveys(survey_ids):
    file_paths = []
    for survey_id in survey_ids:
        file_paths.append(qac.dowload_survey(survey_id,
                                             dataCenter="fra1", fileFormat="tsv"))
    return file_paths
