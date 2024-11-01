"""

    script that performs the data split into train, dev, and test sets
        ==> resulting split paths are saved in the constants TEST_PATH, DEV_PATH, TRAIN_PATH
"""
import random

from paraphrase.utility.PC_utility import get_qids_from_file
from paraphrase.utility.annotation_df import get_annotation_tuples
from paraphrase.annotation_data import get_all_crowd_annotations, TEST_PATH, DEV_PATH, TRAIN_PATH
from paraphrase.utility.stats import set_global_seed


""" ======== RESULT: GET SPLIT Q-IDS ========= """


def get_dev_ids():
    return get_qids_from_file(DEV_PATH)


def get_test_ids():
    return get_qids_from_file(TEST_PATH)


def get_train_ids():
    return get_qids_from_file(TRAIN_PATH)

""" ======== MAIN: SPLIT DATA ========= """

def main():
    crowd_anns_df = get_all_crowd_annotations(include_fails=False)
    # get ALL unique annotations, i.e., (QID, Annotator) tuples
    annotations = get_annotation_tuples(crowd_anns_df)

    # count the number of occurences of each unique qid
    counted_annotations = dict()
    for qid, _ in annotations:
        if qid in counted_annotations:
            counted_annotations[qid] += 1
        else:
            counted_annotations[qid] = 1

    # randomly split annotations 70, 15, 15 into train, dev, test
    total_anns = len(annotations)
    # Calculate sizes for each dataset based on percentages
    size_70_percent = int(round(total_anns * 0.70))
    size_15_percent = int(round(total_anns * 0.15))

    # Shuffle the keys to randomize selection
    set_global_seed()
    keys = list(counted_annotations.keys())
    random.shuffle(keys)

    # Initialize datasets
    dataset_70 = []
    dataset_15_1 = []
    dataset_15_2 = []

    # Initialize sums for each dataset
    sums = {
        0: 0,  # 70%
        1: 0,  # 15%
        2: 0  # 15%
    }

    # Distribute keys into datasets according to the defined percentages
    for key in keys:
        datasets = [dataset_70, dataset_15_1, dataset_15_2]
        max_size = [size_70_percent, size_15_percent, size_15_percent]
        rand_datasets = [0, 1, 2]
        random.shuffle(rand_datasets)
        if sums[rand_datasets[0]] < max_size[rand_datasets[0]]:
            datasets[rand_datasets[0]].append(key)
            sums[rand_datasets[0]] += counted_annotations[key]
        elif sums[rand_datasets[1]] < max_size[rand_datasets[1]]:
            datasets[rand_datasets[1]].append(key)
            sums[rand_datasets[1]] += counted_annotations[key]
        elif sums[rand_datasets[2]] < max_size[rand_datasets[2]]:
            datasets[rand_datasets[2]].append(key)
            sums[rand_datasets[2]] += counted_annotations[key]
        else:
            raise ValueError("The datasets are full")

    # make sure the datasets are not overlapping
    assert len(set(dataset_70).intersection(set(dataset_15_1))) == 0
    assert len(set(dataset_70).intersection(set(dataset_15_2))) == 0
    assert len(set(dataset_15_1).intersection(set(dataset_15_2))) == 0

    # save the matching subselections from crowd_anns_df
    train_df = crowd_anns_df[crowd_anns_df["QID"].isin(dataset_70)]
    test_df = crowd_anns_df[crowd_anns_df["QID"].isin(dataset_15_1)]
    dev_df = crowd_anns_df[crowd_anns_df["QID"].isin(dataset_15_2)]

    # Save the datasets to disk
    train_df.to_csv(TRAIN_PATH, sep="\t", index=False)
    dev_df.to_csv(DEV_PATH, sep="\t", index=False)
    test_df.to_csv(TEST_PATH, sep="\t", index=False)

    # Print the sizes of the datasets
    print(f"# different QIDs in Train: {len(dataset_70)} with {len(get_annotation_tuples(train_df))} annotations")
    print(f"# different QIDs in Dev: {len(dataset_15_2)} with {len(get_annotation_tuples(dev_df))} annotations")
    print(f"# different QIDs in Test: {len(dataset_15_1)} with {len(get_annotation_tuples(test_df))} annotations")


if __name__ == '__main__':
    main()
