"""
    utility file for annotator allocation strategy evaluation
"""
import numpy as np
from matplotlib import pyplot as plt

from paraphrase.utility.annotator_allocation_util import ENTROPY_THRESHOLDS, shuffle_and_trim, calc_bootstrap_kRR, calc_es_bootstrap_kRR
from paraphrase.utility.stats import binary_entropy



"""
    RESULT: calculate the minimum annotators to allocate
"""


def get_min_nbr_anns_entropy(share, nbr_ann, entropy_thresh=0.8, max_ann=15):
    """
        for a given vote of share = x/nbr_ann and an entropy threshold,
        get the minimum number of further annotators necessary to go below the entropy threshold
    :param share:
    :param nbr_ann:
    :param entropy_thresh:
    :return:
    """
    if binary_entropy(share) < entropy_thresh:  # already below entropy threshold
        return max(0, 3 - nbr_ann)
    else:
        # get equivalent positive share
        share = max(share, 1 - share)
        cur_min_anns = 1
        cur_share = (share * nbr_ann + cur_min_anns) / (nbr_ann + cur_min_anns)

        while binary_entropy(cur_share) >= entropy_thresh:
            cur_min_anns += 1
            cur_share = (share * nbr_ann + cur_min_anns) / (nbr_ann + cur_min_anns)

    return max(max_ann - nbr_ann if cur_min_anns + nbr_ann > max_ann else cur_min_anns, 3 - nbr_ann)


"""
    plotting the result of simulating different annotator allocation strategies
"""


def plot_nbr_annotator_plot(bootstrap_dict, es_dict, es_abs_label, es_dict_ekey, es_dict_ekey_avg_anns,
                            es_dict_key, es_dict_key_avg_ann, vanilla_key, vanilla_label, xlabel, ylabel,
                            ylim=(0.55, 0.80), horizontal=0.7, vertical=8, zoom=False):
    """
        plotting the result of simulating different annotator allocation strategies
    :param bootstrap_dict:
    :param es_dict:
    :param es_abs_label:
    :param es_dict_ekey:
    :param es_dict_ekey_avg_anns:
    :param es_dict_key:
    :param es_dict_key_avg_ann:
    :param vanilla_key:
    :param vanilla_label:
    :param xlabel:
    :param ylabel:
    :param ylim:
    :param horizontal:
    :param vertical:
    :param zoom:
    :return:
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    lw = 3
    if zoom:
        lw = 2
    plt.plot(list(range(1, 21)), bootstrap_dict[vanilla_key], '-', marker='.', label=vanilla_label, lw=lw)
    plt.plot(es_dict[es_dict_key_avg_ann], es_dict[es_dict_key], '--.', label=es_abs_label, lw=lw)
    if zoom or not zoom:
        prev_x_val = 0
        for i, (c_x, c_y) in enumerate(zip(es_dict[es_dict_key_avg_ann], es_dict[es_dict_key])):
            if c_x < vertical and c_y > horizontal and abs(prev_x_val - c_x) > 0.1:
                plt.annotate(str(i + 1), (c_x, c_y),
                             textcoords="offset points", xytext=(0, 5), ha='center',
                             fontsize=11, fontweight='bold')

    for min_vote in [3]:
        for max_vote in [10, 15, 20]:
            plt.plot(es_dict[f"{es_dict_ekey_avg_anns} {min_vote}-{max_vote}"],
                     es_dict[f'{es_dict_ekey} {min_vote}-{max_vote})'], ':.',
                     label=f'entropy {min_vote}-{max_vote}', lw=lw)
            if zoom or not zoom:
                prev_x_val = 0
                for i, (c_x, c_y) in enumerate(zip(es_dict[f"{es_dict_ekey_avg_anns} {min_vote}-{max_vote}"],
                                                   es_dict[f'{es_dict_ekey} {min_vote}-{max_vote})'])):
                    if c_x < vertical and c_y > horizontal and abs(prev_x_val - c_x) > 0.2:
                        plt.annotate(str(ENTROPY_THRESHOLDS[i]), (c_x, c_y),
                                     textcoords="offset points", xytext=(0, 3), ha='center',
                                     fontsize=11, fontweight='bold')
                        prev_x_val = c_x
    plt.axhline(y=horizontal, color='0.7', linestyle='-.', lw=1.5)
    plt.axvline(x=vertical, color='0.7', linestyle='-.', lw=1.5)
    plt.legend(fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    plt.xticks(list(range(2, 16)))
    plt.xlim(2, 16)
    plt.ylim(ylim[0], ylim[1])

    if zoom:
        plt.xticks(list(range(5, vertical)))
        plt.xlim(5, vertical)
        plt.ylim(horizontal, ylim[1] - 0.045)
    # plt.title('Multiple Lines with Points Highlighted')

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    plt.show()


"""
    getting the result of simulating different annotator allocation strategies
"""


def get_bootstrap_kRR(n, highlight_df, sorted_valid, sorted_bootstrap_matrix=None):
    # original annotations, where nth rows represent annotations across all questions
    # by the nth annotator (or rather the nth-annotator approximation
    #                       since not all annotators annotated everything,
    #                       1st annotator row is then annotations by the first k annotators going through training
    #                       where k is the minimum # annotators you need to assign 1 annotation for each item)
    sorted_bootstrap_kRRs = []
    org_matrix = shuffle_and_trim(highlight_df, n, sorted_annotators=sorted_valid)

    accuracies = []
    for _ in range(100):
        (sorted_bootstrap_kRR, sorted_bootstrap_matrix,
         sorted_accuracies) = calc_bootstrap_kRR(org_matrix, highlight_df, n)
        sorted_bootstrap_kRRs.append(sorted_bootstrap_kRR)
        accuracies.append(sorted_accuracies)

    print(f"Bootstrap kRR Cohen (w. org. odering): {np.mean(np.array(sorted_bootstrap_kRRs), axis=0)}")
    return {
        "bootstrap kRR": np.mean(np.array(sorted_bootstrap_kRRs), axis=0),
        "accuracies": np.mean(accuracies, axis=0)
    }


def get_es_bootstrap_kRR(n, highlight_df, sorted_valid, bootstrap_matrix=None, org_matrix=None):
    if org_matrix is None:
        org_matrix = shuffle_and_trim(highlight_df, max_k=n, sorted_annotators=sorted_valid)
    cohens = []
    e_cohens = []
    mean_stop_ids = []
    e_stop_means = []
    accuracies = []
    e_accuracies = []
    min_e_votes = [3, 4]
    e_3_15_anns = []
    d_e_cohens = []
    d_e_accs = []
    d_e_stop_means = []
    for _ in range(100):  # calculate a hundred times for bootstrap
        e_cohen, e_stop_ids, es_cohen, max_votes, stop_id_matrix, accs, e_accs, d_e_cohen, d_e_acc, dyn_e_stpid = (
            calc_es_bootstrap_kRR(highlight_df, n, org_matrix, bootstrap_matrix, min_e_votes=min_e_votes))
        cohens.append(es_cohen)
        e_cohens.append(e_cohen)
        mean_stop_ids.append(np.mean(stop_id_matrix + 1, axis=1).ravel().tolist()[0])
        e_stop_means.append([np.mean(e_stop_ids[i] + 1, axis=1) for i in range(len(e_stop_ids))])
        d_e_stop_means.append(np.mean(dyn_e_stpid + 1, axis=1))
        e_3_15_anns.append((e_stop_ids[max_votes.index(15) * 2][ENTROPY_THRESHOLDS.index(0.8)] + 1).tolist())
        accuracies.append(accs)
        e_accuracies.append(e_accs)
        d_e_cohens.append(d_e_cohen)
        d_e_accs.append(d_e_acc)

    result_dict = {
        "early stopping absolute cohen": np.mean(np.array(cohens), axis=0),
        "early stopping absolute accuracy": np.mean(np.array(accuracies), axis=0),
        "stop id means": np.mean(mean_stop_ids, axis=0),
        "max votes": max_votes,  # the set maximum of votes per set of entropy early stopping values
        "min votes": min_e_votes,
        "e_3_15_anns dist": e_3_15_anns,
        "accuracy (entropy dynamic)": np.mean(np.array(d_e_accs), axis=0),
        "bs kRR (entropy dynamic)": np.mean(np.array(d_e_cohens), axis=0),
        "means entropy dynamic": np.mean(d_e_stop_means, axis=0),
    }
    for i, max_vote in enumerate(max_votes):
        for j, min_vote in enumerate(min_e_votes):
            result_dict[f'bs kRR (entropy {min_vote}-{max_vote})'] = np.mean(np.array(
                [e_cohens[row][2 * i + j] for row in range(len(e_cohens))]), axis=0)
            result_dict[f'means entropy {min_vote}-{max_vote}'] = np.mean(np.array(
                [e_stop_means[row][2 * i + j] for row in range(len(e_stop_means))]), axis=0)
            result_dict[f'accuracy (entropy {min_vote}-{max_vote})'] = np.mean(np.array(
                [e_accuracies[row][2 * i + j] for row in range(len(e_accuracies))]), axis=0)
    return result_dict
