"""
    helper functions for calculating stats like overlap, kappa, etc.

"""
import math
import os
import random
import sys
from typing import List

import numpy
import numpy as np
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import cohen_kappa_score

# hide the loading messages
import logging
import transformers
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
import krippendorff

SEED = 3940

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


def relative_overlap(list1, list2):
    """
    :param list1:
    :param list2:
    :return:
    """
    if len(list1) != len(list2):
        return "Lists are not of the same length"

    overlap_count = sum(a == b for a, b in zip(list1, list2))
    return overlap_count / len(list1)


def cohen_kappa_agreement(annotation1: List[int], annotation2: List[int]):
    if annotation2 == annotation1:
        return 1

    # Calculate Cohen's Kappa
    return cohen_kappa_score(annotation1, annotation2)


def fleiss_kappa_agreement(annotations):
    """
        Calculate Fleiss's Kappa via
            https://www.statsmodels.org/dev/generated/statsmodels.stats.inter_rater.fleiss_kappa.html
    :param annotations: 2D Table with rows being the annotators and columns the questions
        the values represent the annotation (0 = not a parpahrase, 1 = paraphrase, nan = not annotated)
    :return: 
    """
    annotations_tr = numpy.array(annotations).transpose()
    aggregate = aggregate_raters(annotations_tr)
    nan_index = numpy.where(numpy.isnan(aggregate[1]))[0]
    agg_no_nan = numpy.delete(aggregate[0], nan_index, axis=1)
    # ratings = []
    # for an_ratings in annotations:
    #     all_categories = all_categories.union(an_ratings)
    #     ratings += [(item, annotation) for item, annotation in enumerate(an_ratings) if not numpy.isnan(annotation)]
    if len(aggregate[1]) >= 2:
        return fleiss_kappa(agg_no_nan)
        # return fleiss_kappa(ratings, n)
    else:
        raise ValueError("No categories have been annotated.")



def transformer_scores(highlights):
    """
        see https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb

    :param highlights: of the form [[h1, h2, h3], [h4, h5, h6]]
        where the sub-lists represent highlights for the same question
    :return:
    """
    print("Querying transformer models ... ")
    if len(highlights) == 0:
        return None

    h1s = []
    h2s = []
    # res = []
    for refs in highlights:
        for i, h1 in enumerate(refs):  # for each hl
            for h2 in refs[i + 1:]:  # calculate BERTScore with all others

                if h1 == "" and h2 == "":  # unless both are empty
                    continue
                    # res.append(1)
                    # warnings.warn("Considering empty highlighting might distort the BertScore artificially.")
                else:
                    h1s.append(h1)
                    h2s.append(h2)

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings1 = model.encode(h1s, convert_to_tensor=True)
    embeddings2 = model.encode(h2s, convert_to_tensor=True)
    scores = util.pairwise_cos_sim(embeddings1, embeddings2)
    mean_score = scores.mean().item()

    return mean_score, scores


def jaccard_overlap_lexical(lexical_pairs):
    """

    :param lexical_pairs: list of g,h pairs:  [[g, h], [g, h], [g, h], ...]
    :return:
    """
    overlaps = []
    for g_utt, h_utt in lexical_pairs:
        if g_utt == "" and h_utt == "":
            continue
        else:
            g_utt = g_utt.lower().split(" ")
            h_utt = h_utt.lower().split(" ")
            intersect = set(g_utt).intersection(set(h_utt))
            union = set(g_utt).union(set(h_utt))
            if len(union) == 0:
                continue
            else:
                overlap = len(intersect) / len(union)
                overlaps.append(overlap)
    return numpy.array(overlaps).mean()


def jaccard_overlap_for_highlights(highlights):
    """
        calculates the jaccard index; ignores cases where one annotator didn't annotate
    :param highlights: of the form [[[1,5,6], [1, 5], [1, 6]], [[], [], []]]
        where the sub-lists represent highlights for the same question --> CAN BE MORE THAN 2 ANNOTATORS PER Q
    :return:
    """

    if len(highlights) == 0:
        return None

    overlaps = []
    for hls in highlights:
        if any(len(hl) == 0 for hl in
               hls):  # in case one highlighting selects nothing ignore TODO: in case of more than two annotators?
            continue
        # get union & intersection of selected tokens
        intersect = set(hls[0])
        union = set(hls[0])
        for hl in hls[1:]:
            intersect = intersect.intersection(set(hl))
            union = union.union(set(hl))

        if len(union) == 0:  # in case nothing is selected on all highlighting, ignore for highlighting overlap
            # overlaps.append(1)
            continue
        else:
            overlap = len(intersect) / len(union)
            overlaps.append(overlap)

    if len(overlaps) > 0:
        return numpy.array(overlaps).mean()
    else:
        return 0


def calc_krippendorff_alpha(annotator_matrix):
    """
        https://pypi.org/project/krippendorff/
    :param annotator_matrix:
        Wikipedia example
            "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3",  # coder A
            "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *",  # coder B
            "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4",  # coder C

        becomes

        reliability_data = [[np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
                            [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ...                     [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4]]
        alpha(reliability_data=reliability_data, level_of_measurement="nominal"

    :return:
    """
    # replaced "return krippendorff.alpha(annotator_matrix)" with the below call
    return krippendorff.alpha(reliability_data=annotator_matrix, level_of_measurement="nominal")


def set_global_seed(seed=SEED, w_torch=True):
    """
    Make calculations reproducible by setting RANDOM seeds
    :param seed:
    :param w_torch:
    :return:
    """
    # set the global variable to the new var throughout
    global SEED
    SEED = seed
    if 'torch' not in sys.modules:
        w_torch = False
    if w_torch:
        import torch
        logging.info(f"Running in deterministic mode with seed {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


"""    ---------- PLOT SOME STATS """


def plot_relative_triplets(triplets, subset_labels, group_colors=['#FF934F', '#E1DAAE', '#CC2D35'],
                           group_labels=["Paraphrases", "Repetitions", "Non-Paraphrases"],
                           title=None):
    import matplotlib.pyplot as plt
    font_size = 20

    # Normalize each triplet so that the sum equals 100
    normalized_triplets = [(A1 / (A1 + A2 + B) * 100, A2 / (A1 + A2 + B) * 100, B / (A1 + A2 + B) * 100) for A1, A2, B
                           in triplets]

    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, ((A1, A2, B), subset_label) in enumerate(zip(normalized_triplets, subset_labels), start=1):
        bar_label = subset_label
        ax.barh(bar_label, A1, color=group_colors[0], height=0.5)  # A1
        ax.barh(bar_label, A2, left=A1, color=group_colors[1], height=0.5)  # A2
        ax.barh(bar_label, B, left=A1 + A2, color=group_colors[2], height=0.5)  # B

        # Adding percentages in the middle of the bars
        ax.text(1, i - 1, f'{A1:.1f}%', ha='left', va='center', color='black', fontsize=font_size,
                fontweight='bold')
        ax.text(A1 + A2 + B / 2, i - 1, f'{B:.1f}%', ha='center', va='center', color='white', fontsize=font_size,
                fontweight='bold')

        if A1 + A2 < 20:
            # Adding percentage for A2 below the bar
            ax.text(A1 + 1, i - 1.4, f'{A2:.1f}%', ha='left', va='center', color='black', fontsize=font_size,
                    fontweight='bold')
        else:
            ax.text(A1 + A2 / 2, i - 1, f'{A2:.1f}%', ha='center', va='center', color='black', fontsize=font_size,
                    fontweight='bold')

    # Adding legend, titles, and labels
    ax.set_xlabel('Percentage', fontsize=font_size)
    if title is not None:
        ax.set_title(title, fontsize=font_size)
    ax.legend(group_labels, fontsize=font_size, ncol=3)
    # ax.legend([])
    # bars = ax.barh(group_labels, A1, height=0.5, color='#FF934F')
    # ax.legend(bars, group_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(bar_label), fancybox=True,
    #           shadow=True)

    # Increase the size of the ticks
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # Set the limits of y-axis
    ax.set_ylim(-0.5, len(subset_labels)+0.5)
    ax.set_xlim(0, 100)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_relative_histogram(data):
    """ generated by GPT4
    Plot a histogram where the y-axis is relative (percentage) and displays combined Repetitions and Paraphrases.

    Args:
    data (dict): A dictionary with values for 'Repetitions', 'Paraphrases', and 'Non-Paraphrases'.
    """
    # Calculate the combined total for Repetitions and Paraphrases
    combined_total = data["Repetitions"] + data["Other"]

    # Calculate the total for all categories
    total = combined_total + data["Non-Paraphrases"]

    # Calculate relative (percentage) values for each category
    relative_combined = (combined_total / total) * 100
    relative_repetitions = (data["Repetitions"] / total) * 100
    relative_other = (data["Other"] / total) * 100
    relative_non_paraphrases = (data["Non-Paraphrases"] / total) * 100

    # Set positions for the bars
    positions_combined = [1]
    positions_repetitions = [0.9]
    positions_other = [1.1]
    positions_non_paraphrases = [2]

    # Plotting the bars
    plt.bar(positions_combined, [relative_combined], color='gray', width=0.4,
            label='Paraphrases\n(Repetitions + Other)')
    plt.bar(positions_repetitions, [relative_repetitions], color='blue', width=0.2, label='Repetitions')
    plt.bar(positions_other, [relative_other], color='green', width=0.2, label='Other')
    plt.bar(positions_non_paraphrases, [relative_non_paraphrases], color='red', width=0.4, label='Non-Paraphrases')

    # Setting the labels and title
    plt.xticks([1, 2], ['Paraphrases\n(Repetitions + Other)', 'Non-Paraphrases'])
    plt.yticks(fontsize=12)
    plt.ylim(0, 0.85)
    plt.ylabel("Percentage of Total", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()


def binary_entropy(p):
    if p == 0 or p == 1:  # Handle edge cases where p is 0 or 1
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def dynamic_entropy(p, n):
    if p == 0 or p == 1:  # Handle edge cases where p is 0 or 1
        return 0
    return binary_entropy(p)*1/math.sqrt(n)  # binary_entropy(p) - (n-1)*0.04  # binary_entropy(p)*1/n  #


def parse_fraction_to_tuple(fraction_string):
    """generated with GPT-4 on nov 23rd 2023
    Parse a fraction string with whitespaces like " 1 / 10 " and convert it to a float.

    :param fraction_string: String representing a fraction with possible whitespaces.
    :return: Float value of the fraction.
    """
    try:
        # Splitting the string by the '/' character and stripping whitespaces
        numerator, denominator = map(str.strip, fraction_string.split('/'))
        # Converting the numerator and denominator to float and dividing
        return [int(numerator), int(denominator)]
    except ValueError:
        # In case of any error, return None
        return None
