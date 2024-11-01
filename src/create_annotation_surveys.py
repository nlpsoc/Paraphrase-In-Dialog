""""
    You probably don't want to use this

    Create Qualtrics Studies based off a list of (host, guest)-pair IDs,
        publish them and put them on Prolific
    Used only for PARA/RAND annotation

    Needs: qualtrics and prolific APIs

"""

import pandas as pd

from paraphrase.utility.PC_utility import SAMPLE_RAND_PARA_PC_IDS
from paraphrase.utility.annotation_pipeline import create_and_publish_training_survey, \
    get_validated_annotators, check_and_update_db, \
    EMPTY_MANUAL_ADMISSION, \
    get_surveys_to_rerun, _update_admitted_annotators, create_and_publish_annotation_surveys_from_file, read_db, \
    save_db, \
    create_and_publish_nthround_annotations, get_participated_annotators, print_last_annotation_statistics, \
    merge_ids_for_sublist_to_length
from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.set_id_consts import MANUAL_EXCLUSION, MANUAL_INCLUSION


TYPES = ["TRAINING", "ANNOTATOR-INIT", "ANNOTATOR-UPDATE", "UPDATE"]
TYPE = "ANNOTATOR-UPDATE"  # "UPDATE"  # "ANNOTATOR-INIT"  # "TRAINING"
TEST = False  # True
TRAIN_PLACES = 10


def main(an_type: str = TYPE, places=TRAIN_PLACES, test=TEST, test_df=None, manual_inclusion=None,
         manual_exclusion=None):
    # read in database unless in testing mode and a dataframe is given
    if manual_exclusion is None:
        manual_exclusion = MANUAL_EXCLUSION
    if manual_inclusion is None:
        manual_inclusion = MANUAL_INCLUSION
    db_df = read_db(test=test, test_df=test_df)

    assert (an_type in TYPES)
    if an_type == "TRAINING":  # create training survey
        # get previously participated annotators in training
        db_df = check_and_update_db(test=test, test_df=(None if test_df is None else db_df),
                                    manual_inclusion=manual_inclusion)
        participated_annotators = get_participated_annotators(test=test, test_df=(None if test_df is None else db_df))
        p_id, q_id, s_name = create_and_publish_training_survey(places, test=test, exclude=participated_annotators)
        # save ID of prolific training
        new_row = {
            'Survey Name': s_name,
            'Qualtrics ID': q_id,
            'Prolific ID': p_id,
            'Version': "TRAIN",
            'STATUS': "INIT",
            'Question IDs': "-",
            'Admitted Annotators': places,
            'Participated Annotators': "",
            'Valid Annotators': "",
            'Entropy': "-"
        }
        if db_df is None:
            db_df = pd.DataFrame({key: [value] for key, value in new_row.items()})
        else:
            db_df = db_df.append(new_row, ignore_index=True)
        if test_df is None:
            save_db(db_df, test=test)
        return db_df

    elif an_type == "ANNOTATOR-INIT":
        # CALL ONLY IF ALL studies are closed (i.e., prolific)
        db_df = check_and_update_db(test=test, test_df=(None if test_df is None else db_df),
                                    manual_inclusion=manual_inclusion)
        approved_annotators = get_validated_annotators(manual_admission=EMPTY_MANUAL_ADMISSION, test=test,
                                                       test_df=(None if test_df is None else db_df))
        print(f"Total Set of approved annotators: {approved_annotators}")
        # fixed assumption of up to 84 seconds per item
        db_df = create_and_publish_annotation_surveys_from_file(approved_annotators,
                                                                filename=(get_dir_to_src() + SAMPLE_RAND_PARA_PC_IDS),
                                                                test=TEST, test_df=(None if test_df is None else db_df))
        return db_df

    elif an_type == "ANNOTATOR-UPDATE":
        entropy_thresh = 0.8
        max_annotators = 15

        # get votes from previous surveys
        # AND check if some of the relevant surveys are "PENDING", if yes close them and update db_df
        db_df = check_and_update_db(test=test, test_df=(None if test_df is None else db_df),
                                    manual_inclusion=manual_inclusion, close_pending=True)
        print_last_annotation_statistics(db_df)

        # get qids for surveys that need to be rerun
        place_per_study, qid_groups, remaining_anns, participated_anns, survey_names, db_df = (
            get_surveys_to_rerun(entropy_thresh, max_annotators, test=test,
                                 test_df=(None if test_df is None else db_df)))

        if len(survey_names) > 0:
            # make sure all current surveys have the same version
            if not any("|" in ver for sname, ver in survey_names):
                version = int(survey_names[0][1])
                assert (len(set(survey_names[c_id][1] for c_id in range(len(survey_names)))) == 1)
            else:
                version = int(survey_names[0][1].split("|")[0])
                assert (len(set(ver for c_id in range(len(survey_names)) for ver in survey_names[c_id][1].split("|"))) == 1)
            # merge studies that are too small now to up to 10 items
            merge_ids = merge_ids_for_sublist_to_length(qid_groups, length=10)
            #   make sure all studies re-occur that are remaining
            assert (set([s_id for sublist in merge_ids for s_id in sublist]) == set(range(len(qid_groups))))
            merged_qids = [[q_id for c_id in id_group for q_id in qid_groups[c_id]] for id_group in merge_ids]
            merged_names = [
                ("|".join([survey_names[c_id][0] for c_id in id_group]),
                 "|".join([survey_names[c_id][1] for c_id in id_group]))  # Versions are all the same (as asserted before)
                for id_group in merge_ids]
            merged_places = [min(place_per_study[c_id] for c_id in id_group) for id_group in merge_ids]
            merged_anns = [list(set.intersection(*[set(remaining_anns[c_id]) for c_id in id_group]))
                           for id_group in merge_ids]

            # redistribute annotators that are not yet at full capacity (changes remaining_anns)
            assigned_anns = _update_admitted_annotators(merged_names, merged_anns, merged_places, merged_qids,
                                                        test=test,
                                                        test_df=(None if test_df is None else db_df),
                                                        manual_exclusion=manual_exclusion,
                                                        manual_inclusion=manual_inclusion)

            db_df = create_and_publish_nthround_annotations(db_df, merged_places, merged_qids, assigned_anns, merged_names, test,
                                                            test_df, seconds_per_item=84 + min(int(version), 10))
        return db_df

    elif an_type == "UPDATE":
        db_df = check_and_update_db(test=test, test_df=(None if test_df is None else db_df),
                                    manual_inclusion=manual_inclusion)
        print_last_annotation_statistics(db_df)
        return db_df


if __name__ == "__main__":
    main()
