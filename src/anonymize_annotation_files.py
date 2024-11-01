"""
    script we ran to anonymize the prolific IDs
"""
import os

from paraphrase.annotation_data import PC_FOLDER, ANNOTATIONS_5050_20, ALL_ANNOTATIONS
import paraphrase.anonymize as anonymize
from paraphrase.lead_pc import PC_AN_PATH_first_batch_1, PC_SECOND_BATCH_FOLDER, PC_LEGACY_FOLDER
from paraphrase.utility.annotation_pipeline import DB_FOLDER
from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.utility.qualtrics_api import PARA_RAND_DOWNLOAD_FOLDER


def main():
    # anonymze all annotations, train, dev and test split
    anonymize.anonymize_folder(os.path.dirname(ALL_ANNOTATIONS))

    # anonymize 5050
    #   all Qualtrics 5050_20 files
    anonymize.anonymize_folder(PC_FOLDER)
    #   in one file
    anonymize.anonymize_folder(os.path.dirname(ANNOTATIONS_5050_20))

    # anonymize RAND-PARA
    anonymize.anonymize_folder(PARA_RAND_DOWNLOAD_FOLDER)
    anonymize.anonymize_folder(get_dir_to_src() + DB_FOLDER + "/")

    # anonymize lead annotations
    anonymize.anonymize_folder(os.path.dirname(PC_AN_PATH_first_batch_1))
    anonymize.anonymize_folder(os.path.dirname(PC_SECOND_BATCH_FOLDER))
    anonymize.anonymize_folder(os.path.dirname(PC_LEGACY_FOLDER))

    # pilots
    anonymize.anonymize_folder(get_dir_to_src() + "/../result/Annotations/Pilots/")



if __name__ == '__main__':
    main()
