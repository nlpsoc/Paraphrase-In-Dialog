"""
    preprocess MediaSum corpus for annotation (e.g., 2 person interviews)
        --> resulting corpus saved in IDS_CSV
    this file generated the file in "../result/Data/ieir_ids.csv"
"""
from paraphrase.utility.project_functions import SEED, get_dir_to_src
from paraphrase.interview_data import MediaSumProcessor
import argparse

# Dataset that resulted from this main call
IDS_CSV = get_dir_to_src() + "/../result/Data/ieir_ids.csv"


def main(ms_path: str, gh_pairs_out: str):
    # load complete MediaSum data
    interview = MediaSumProcessor(ms_path)
    # filter out non 2-person interviews and shuffle on interview level and save the triples
    interview.filter_and_shuffle_ghpairs_to_tsv(ghID_path=gh_pairs_out, seed=SEED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save MediaSum ieir ids and pairs.')
    parser.add_argument('-ms', '--mediasum_path', default="../data/MediaSum/news_dialogue.json")
    parser.add_argument('-ieir', '--ieir_out', default="../output/ieir_ids.csv", help="path to gh_pair ids")

    args = parser.parse_args()

    main(ms_path=args.mediasum_path, gh_pairs_out=args.ieir_out)

