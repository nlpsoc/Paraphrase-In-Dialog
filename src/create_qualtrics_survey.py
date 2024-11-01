"""
    You probably don't want to use this

        script to push surveys to qualtrics for lead author annotations
"""
from datetime import datetime

import argparse
from paraphrase.qualtrics import create_survey
from preprocess_interviews import IDS_CSV

SURVEY_NAME = "23-09-11_Paraphrase-Candidate_4809-4858"
START_SAMPLE = 4809  # indices start from 1 (!)

PARAPHRASE_CANDIDATE = False  # True for different survey format without highlighting
TOTAL_NBR_QUESTIONS = 50

SECONDS = 30
FROM_ID_LIST = False  # True to create a survey from list of IDs
ID_LIST = ['CNN-121316-5']


def main(size_sample: int = TOTAL_NBR_QUESTIONS, survey_name: str = SURVEY_NAME,
         from_id_list: bool = FROM_ID_LIST,
         id_list=None, out_qsf: str = f"../output/{datetime.now()}_survey.qsf", sample_path: str = IDS_CSV,
         candidate_survey: bool = PARAPHRASE_CANDIDATE, publish=True, interview=None, training=False,
         test=False,
         places=1):
    create_survey(size_sample=size_sample, survey_name=survey_name, from_id_list=from_id_list,
                  id_list=id_list, out_qsf=out_qsf, sample_path=sample_path, candidate_survey=candidate_survey,
                  publish=publish, interview=interview, training=training, test=test, places=places,
                  start_sample=START_SAMPLE, seconds=SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare sample annotation on Qualtrics.')
    parser.add_argument('-name', '--survey_name', default=SURVEY_NAME)
    parser.add_argument('-n', '--number_sample', default=TOTAL_NBR_QUESTIONS)
    parser.add_argument('-ms', '--mediasum_path', default="../data/MediaSum/news_dialogue.json")
    parser.add_argument('-sp', '--sample_path', default=IDS_CSV)
    parser.add_argument('-qsf', '--out_qsf_path', default="../output/sample_survey.qsf")
    parser.add_argument('-fli', '--list', dest='fli', action='store_true',
                        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
                        help="generate survey from given ID list")
    parser.add_argument('-no-fli', '--no-list', dest='fli', action='store_false')
    parser.set_defaults(fli=FROM_ID_LIST)
    parser.add_argument('-pc', '--paraphrase_candidate', dest='pc', action='store_true',
                        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
                        help="whether to generate a paraphrase candidate survey")
    parser.add_argument('-no-pc', '--no-paraphrase_candidate', dest='pc', action='store_false')
    parser.set_defaults(pc=PARAPHRASE_CANDIDATE)

    args = parser.parse_args()

    main(size_sample=int(args.number_sample), survey_name=args.survey_name, from_id_list=args.fli,
         out_qsf=args.out_qsf_path, sample_path=args.sample_path, candidate_survey=args.pc)
