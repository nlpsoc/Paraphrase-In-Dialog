
"""
    You probably don't want to use this
    script to evaluate annotations made in pilot studies via qualtric setup

    whether this still runs without a hitch is questionable, several changes were made to study setup in-between

"""

import argparse
import pandas as pd

import paraphrase.utility.annotation_df as annotation_df
import paraphrase.utility.evaluate_qsurvey as evaluate_qsurvey
from paraphrase import interview_data as interview_data
from paraphrase.utility.evaluate_qsurvey import calc_pc_eval_crowd
from paraphrase.id_pilot_studies import DT_PATH
from typing import List

MS_NEWS_DIALOGUE_JSON = "../data/MediaSum/news_dialogue.json"

AN_PATHS = [DT_PATH]  # AN_PATHS = [PC_1, PC_31, PC_61]
PARAPHRASE_CANDIDATE_SURVEY = False  # True  # False

FROM_LIST = False
# First round of creating from list (pre-determined paraphrase test list March 21)
CURATED_ID_LIST_0 = ["CNN-201862-11", "CNN-172625-9", "CNN-215723-3", "CNN-137012-3", "CNN-13148-5", "CNN-215723-11",
                     "CNN-83717-3", "CNN-165270-3", "CNN-165270-13", "CNN-1366-9", "CNN-201862-5", "CNN-40253-3",
                     "CNN-25363-7", "CNN-25363-3", "CNN-357405-7", "CNN-96467-5", "CNN-221467-5", "CNN-67742-5",
                     "CNN-66733-7", "CNN-71044-5"]
curated_id_list = CURATED_ID_LIST_0
IS_PARAPRHASE_0 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
is_paraphrase_list = IS_PARAPRHASE_0
# Second round of creating from list (pre-determined paraphrase test list March 22):
CURATED_ID_LIST = ["CNN-201862-11", "CNN-172625-9", "CNN-159242-11", "CNN-137012-3", "CNN-152024-10",
                   "CNN-215723-11", "CNN-83717-3", "CNN-165270-3", "CNN-165270-13", "CNN-1366-9",
                   "CNN-201862-5", "CNN-40253-3", "CNN-159242-3", "CNN-25363-7", "CNN-25363-3",
                   # "CNN-357405-7", "CNN-96467-5"
                   "CNN-221467-5", "CNN-327061-5", "CNN-67742-5", "CNN-66733-7", "CNN-71044-5"]
IS_PARAPRHASE = [1, 1, 1, 1, 1,
                 0, 0, 0, 0, 0,
                 1, 1, 1, 1, 1,
                 0, 0, 0, 0, 0]


def main(an_paths: List[str], paraphrase_candidate: bool, ms_path: str = MS_NEWS_DIALOGUE_JSON):
    print(f"evaluating annotations saved in {an_paths}...")
    pd_report = annotation_df.merge_transform(an_paths, paraphrase_candidate=paraphrase_candidate)
    if "Q-CNN-13148-5" in pd_report["QID"].unique().tolist():  # error in one of the question IDs on Qualtrics
        pd_report.loc[pd_report["QID"] == "Q-CNN-13148-5", "QID"] = "CNN-13148-5"
    print(pd_report['Annotator'].unique().tolist())

    print("Calculating Frequency of categories ...")
    result_dict = evaluate_qsurvey.get_freq(pd_report, paraphrase_candidate=paraphrase_candidate)
    print(f"\t{result_dict}")

    print("Calculating Agreement of binary categories ...")
    result_dict = evaluate_qsurvey.calc_classification_agreement(pd_report, paraphrase_candidate=paraphrase_candidate)
    print(f"\t{result_dict}")

    if paraphrase_candidate:
        result = calc_pc_eval_crowd(pd_report)
        print(f"\t{result}")

    if not paraphrase_candidate:
        if FROM_LIST:
            print("Study was generated from a pre-defined list with known correct answers ...")
            q_to_is_para = dict(zip(curated_id_list, is_paraphrase_list))
            result_dict = evaluate_qsurvey.calc_an_perf_onlist(transformed_df=pd_report, is_prphrs_lst=q_to_is_para)
            print(f"\t{result_dict}")

        interview = interview_data.MediaSumProcessor(ms_path, uncut=True)
        print("Calculating Overlap of highlights ...")
        result_dict = evaluate_qsurvey.calc_overlaps(pd_report, interview=interview)
        print(f"\t{result_dict}")

        print("Error evaluation ...")

        print("Print Disagreements for Paraphrase")
        try:
            evaluate_qsurvey.print_disagreement_examples(annotation_df.ContentReproductionCategories.Paraphrase,
                                                         pd_report, interview)
            evaluate_qsurvey.print_agreement_examples(annotation_df.ContentReproductionCategories.Paraphrase,
                                                      pd_report, interview)
        except AssertionError:
            # legacy --> called repeat before
            evaluate_qsurvey.print_disagreement_examples(annotation_df.ContentReproductionCategories.Repeat,
                                                         pd_report, interview)
            evaluate_qsurvey.print_agreement_examples(annotation_df.ContentReproductionCategories.Repeat,
                                                      pd_report, interview)


def eval_demographics(csv_path: str):
    demo_df = pd.read_csv(csv_path)
    demo_df = demo_df

    print(demo_df[~demo_df['Age'].isin(['CONSENT_REVOKED', 'DATA_EXPIRED'])]['Age'].astype(int).describe())
    print(demo_df['Sex'].describe())
    print(demo_df[~demo_df['Ethnicity simplified'].isin(['CONSENT_REVOKED', 'DATA_EXPIRED'])][
              'Ethnicity simplified'].value_counts())
    print(demo_df[~demo_df['Nationality'].isin(['CONSENT_REVOKED', 'DATA_EXPIRED'])]['Nationality'].value_counts())

    return demo_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare sample annotation on Qualtrics.')
    # https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python
    parser.add_argument('-an_paths', '--annotation_paths', nargs="*", type=str, default=AN_PATHS)
    parser.add_argument('-ms', '--mediasum_path',
                        default=MS_NEWS_DIALOGUE_JSON)  # path to mediasum dataset (probably fixed)
    # pc: whether or not this is paraphrase candidate selection (with only binary annotations),
    #   or full-on annotation with highligths
    parser.add_argument('-pc', '--paraphrase_candidates', dest='pc', action='store_true',
                        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
                        help="use cluster cache folders as saved in ")
    parser.add_argument('-no-pc', '--no-paraphrase_candidates', dest='pc', action='store_false')
    parser.set_defaults(pc=PARAPHRASE_CANDIDATE_SURVEY)

    args = parser.parse_args()

    main(an_paths=args.annotation_paths, paraphrase_candidate=args.pc, ms_path=args.mediasum_path)
