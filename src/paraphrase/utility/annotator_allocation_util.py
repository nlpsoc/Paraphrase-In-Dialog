"""
    Utility functions for annotator allocation simulation, called by annotator_allocation
"""
import warnings

import numpy as np
import paraphrase.utility.annotation_df as annotation_df
import paraphrase.utility.stats as statistics_util
from paraphrase.utility.annotation_df import AnnotationColumns, predict
from paraphrase.utility.stats import binary_entropy, dynamic_entropy

ENTROPY_THRESHOLDS = [0.99, 0.97, 0.95, 0.93, 0.90, 0.88, 0.85, 0.83, 0.8, 0.78, 0.75, 0.73, 0.70,
                      0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
DYNAMIC_ENTROPY_THRESHOLDS = [0.30, 0.25, 0.20, 0.17, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,
                              0.04, 0.03, 0.02, 0.01]


def get_maxmin_nbr_anns(cur_annotators, highlight_df):
    """
        for given highlighting dataframe and list of annotators, return the questions that receive any annotations
        and the minimum and maximum number of annotations observed
    :param cur_annotators:
    :param highlight_df:
    :return:
    """
    cur_annotations = highlight_df[highlight_df[AnnotationColumns.Annotator_ID].isin(cur_annotators)]
    cur_ids = annotation_df.get_unique_q_ids(cur_annotations)
    #   for each q_id and annotator only occur once
    cur_annotations = cur_annotations[cur_annotations[AnnotationColumns.Question_ID].isin(cur_ids) &
                                      (cur_annotations[AnnotationColumns.Highlight_Category] ==
                                       annotation_df.AllAnnotationCategories.Referring_Choice)]
    value_counts = cur_annotations[AnnotationColumns.Question_ID].value_counts()
    # Get value with maximum and minimum occurrence
    max_occurrence_val = value_counts.idxmax()
    min_occurrence_val = value_counts.idxmin()
    max_occ = value_counts[max_occurrence_val]
    min_occ = value_counts[min_occurrence_val]
    return cur_ids, max_occ, min_occ


def get_statistics_annotator_split(sorted_valid, highlight_df):
    """
        prints the minimum/maximum number annotations for a question
        when annotators are split in first i annotators according to sorted_valid and reverse
        when minimum is equal that is the split that makes the most sense
    :param sorted_valid:
    :param highlight_df:
    :return:
    """
    best_i = 0
    k_max = 0
    for i in range(1, len(sorted_valid)):
        cur_ids, max_occ, min_occ = get_maxmin_nbr_anns(sorted_valid[:i], highlight_df)
        rev_cur_ids, rev_max_occ, rev_min_occ = get_maxmin_nbr_anns(sorted_valid[i:], highlight_df)
        assert (len(sorted_valid[:i]) + len(sorted_valid[i:]) == 35)

        print(f"For {len(sorted_valid[:i])} annotators reverse {len(sorted_valid[i:])}, over {len(cur_ids)}")
        print(f"\t maximum # annotations per Question ID "
              f"{max_occ} and reverse {rev_max_occ}\n"
              f"\t minimum # annotations per Question ID "
              f"{min_occ} and reverse {rev_min_occ}")

        if min_occ == rev_min_occ:
            best_i = i
            k_max = min_occ
    return {
        "annotator_split_index": best_i,
        "maximum k": k_max
    }


def shuffle_and_trim(highlight_df, max_k, sorted_annotators=None, w_replacement=False):
    """
        get the annotation matrix ANNOTATORSxQUESTIONS, for a maximum of max_k annotators per column,
        shuffling the votes per question
            (unless sorted_annotators is not None, then order is determined by annotator order)
    :param w_replacement: whether to shuffle the votes "with replacement"
    :param sorted_annotators: can't be a list when w_replacement is true
    :param highlight_df:
    :param max_k:
    :return:
    """
    assert ((sorted_annotators is None) or (not w_replacement))
    # if sorted_annotators is None:
    binaries_dict = annotation_df.get_annotation_dict(highlight_df)
    annotators = binaries_dict["annotators"]
    complete_binaries = binaries_dict["complete_binaries"]
    if sorted_annotators is not None:  # keep ordering acc. to sorted_annotators --> automatically not shuffled
        complete_binaries = [complete_binaries[annotators.index(ann)] for ann in sorted_annotators]
    complete_binaries = np.array(complete_binaries)

    if sorted_annotators is None:
        sorted_binaries = np.sort(complete_binaries, axis=0)  # nan values sorted to the end
        # For each column, shuffle only the non-NaN elements
        for col_idx in range(sorted_binaries.shape[1]):
            non_nan_count = np.sum(~np.isnan(sorted_binaries[:, col_idx]))
            non_nan_elements = sorted_binaries[:non_nan_count, col_idx]
            if not w_replacement:
                np.random.shuffle(non_nan_elements)
            else:
                non_nan_elements = np.random.choice(non_nan_elements, size=non_nan_count, replace=True)
            sorted_binaries[:non_nan_count, col_idx] = non_nan_elements
    else:
        # Keep the annotator order
        for col_idx in range(complete_binaries.shape[1]):
            # Get a mask for non-NaN values
            nan_mask = np.isnan(complete_binaries[:, col_idx])
            # Reorder the column: non-NaN values first, then NaN values
            complete_binaries[:, col_idx] = np.concatenate([complete_binaries[~nan_mask, col_idx],
                                                            complete_binaries[nan_mask, col_idx]])
        sorted_binaries = complete_binaries

    sorted_binaries = sorted_binaries[:max_k, :]
    return sorted_binaries


def _calc_kRR_cohen(k, sorted_bin_left, sorted_bin_right, random_ann_selection=True):
    """
        use the first k rows to aggregate the votes for the left and right matrix,
        then use those votes to calculate cohen's agreement
    :param k:
    :param sorted_bin_left:
    :param sorted_bin_right:
    :return:
    """
    if k is not None:
        if random_ann_selection:
            # sample k rows (equivalent to columns in https://aclanthology.org/2022.acl-short.42.pdf)
            #   to sample k annotators that are aggregated for kRR
            sampled_rows_left = sorted_bin_left[
                                np.random.choice(sorted_bin_left.shape[0], k, replace=False), :]
            sampled_rows_right = sorted_bin_right[
                                 np.random.choice(sorted_bin_right.shape[0], k, replace=False), :]
        else:
            sampled_rows_left = sorted_bin_left[:k, :]
            sampled_rows_right = sorted_bin_right[:k, :]
        agg_votes_left = annotation_df.calc_vote_share(sampled_rows_left)
        agg_votes_right = annotation_df.calc_vote_share(sampled_rows_right)
    else:
        agg_votes_left = annotation_df.calc_vote_share(sorted_bin_left)
        agg_votes_right = annotation_df.calc_vote_share(sorted_bin_right)

    bin_class_left = [predict(agg_vote) for agg_vote in agg_votes_left]
    bin_class_right = [predict(agg_vote) for agg_vote in agg_votes_right]
    cohen = statistics_util.cohen_kappa_agreement(bin_class_left, bin_class_right)
    return cohen, bin_class_left, bin_class_right


def calc_bootstrap_kRR(org_matrix, highlight_df, n, sorted_bootstrap_matrix=None):
    if sorted_bootstrap_matrix is None:
        # bootstrapped annotations, where nth column represents the bootstrapped annotations for the nth item
        #                           (i.e., drawing with replacement from the 20 annotations made
        #                           for the nth item anoter 20 annotations)
        sorted_bootstrap_matrix = shuffle_and_trim(highlight_df, n, w_replacement=True)
    # SKIPPING -- shuffled original annotations, where nth row represents annotations across all questions
    # by the "nth annotator" (in this case just shuffled puzzle of all annotators for this question)
    #                       makes sense when assuming annotator order/annotator "coherence" is not important
    # shuffled_org_matrix = shuffle_and_trim(highlight_df, n)
    sorted_bootstrap_kRR = []
    # shuffled_bootstrap_kRR = []
    # predictions of original annotations
    gt_predictions = get_conv_predictions(org_matrix)
    accuracies = []
    for k in range(1, n + 1):
        # _calc_kRR_cohen samples k rows (i.e., "annotators") and
        #   aggregates the votes per annotation matrix to calculate cohen's agreement
        cohen, _, org_sample_predictions = _calc_kRR_cohen(k, sorted_bootstrap_matrix, org_matrix)
        sorted_bootstrap_kRR.append(cohen)
        accuracies.append(get_accuracies_predictions(gt_predictions, org_sample_predictions))
        # cohen = _calc_kRR_cohen(k, sorted_bootstrap_matrix, shuffled_org_matrix)
        # shuffled_bootstrap_kRR.append(cohen)
    return sorted_bootstrap_kRR, sorted_bootstrap_matrix, accuracies


def get_accuracies_predictions(gt_predictions, org_sample_predictions):
    return sum([1 if gt == bs else 0
                for gt, bs in zip(gt_predictions, org_sample_predictions)]) / len(gt_predictions)


def get_cumulative_matrix(binary_annotation_matrix):
    cumulative_matrix = np.nancumsum(binary_annotation_matrix, axis=0)
    return cumulative_matrix


def calc_es_bootstrap_kRR(highlight_df, n, org_matrix, bootstrap_matrix=None, min_e_votes=[3, 4]):
    if bootstrap_matrix is None:
        # bootstrapped annotations, where nth column represents the bootstrapped annotations for the nth item
        #                           (i.e., drawing with replacement from the 20 annotations made
        #                           for the nth item anoter 20 annotations)
        bootstrap_matrix = shuffle_and_trim(highlight_df, n, w_replacement=True)
    # shuffle the rows (i.e., annotators), to approximate selecting k random rows
    #   as done in https://aclanthology.org/2022.acl-short.42.pdf
    org_matrix = shuffle_rows(org_matrix)

    bootstrap_matrix = shuffle_rows(bootstrap_matrix)
    # get the stop ids for the shuffled matrices with absolute number of agreeing annotators
    stop_id_matrix = _get_dynamic_stopids(n, org_matrix)
    stop_id_matrix_bs = _get_dynamic_stopids(n, bootstrap_matrix)
    es_cohen = []
    # predictions of original annotations
    gt_predictions = get_conv_predictions(org_matrix)
    accuracies = []
    for es in range(1, round(n / 2) + 1):
        # replace matrix with nan values where it stops early
        #   absolute
        es_matrix = _get_nanreplaced_es_matrix(es, stop_id_matrix, org_matrix)
        es_predictions = get_conv_predictions(es_matrix)
        bs_matrix = _get_nanreplaced_es_matrix(es, stop_id_matrix_bs, bootstrap_matrix)
        # calculate cohen
        es_cohen.append(_calc_kRR_cohen(None, es_matrix, bs_matrix)[0])
        # add accuracies
        accuracies.append(get_accuracies_predictions(gt_predictions, es_predictions))

    max_votes = [7, 10, 11, 12, 13, 14, 15, 16, 17, 20]
    e_cohens = []
    e_stop_ids = []
    e_accuracies = []
    for max_vote in max_votes:
        for min_vote in min_e_votes:
            # for given max and min vote calculate fixed entropy threshold early stopping values
            e_cohen, stop_id_mat_entropy, mv_accuracies = get_entropy_stopping(org_matrix, bootstrap_matrix,
                                                                               max_vote, min_votes=min_vote)
            e_cohens.append(e_cohen)
            e_stop_ids.append(stop_id_mat_entropy)
            e_accuracies.append(mv_accuracies)

    # for dynamic entropy threshold calculate entropy
    dyn_e_cohen, dyn_stop_id_mat_entropy, dyn_mv_accuracies = get_entropy_stopping(org_matrix, bootstrap_matrix,
                                                                                   max_votes=15, min_votes=3,
                                                                                   dynamic=True)

    return (e_cohens, e_stop_ids, es_cohen, max_votes, stop_id_matrix, accuracies, e_accuracies,
            dyn_e_cohen, dyn_mv_accuracies, dyn_stop_id_mat_entropy)


def get_entropy_stopping(org_matrix, bootstrap_matrix, max_votes, min_votes=3, dynamic=False):
    """

    :param dynamic:
    :param min_votes:
    :param max_votes:
    :param org_matrix:
    :param bootstrap_matrix:
    :return:
    """
    e_cohen = []
    # get the stop ids for entropy annotations
    stop_id_mat_entropy = _get_early_entropy_ids(org_matrix, max_votes=max_votes, min_votes=min_votes,
                                                 dynamic=dynamic)
    stop_id_mat_entropy_bs = _get_early_entropy_ids(bootstrap_matrix, max_votes=max_votes, min_votes=min_votes,
                                                    dynamic=dynamic)
    gt_predictions = get_conv_predictions(org_matrix)
    accuracies = []
    for es in range(1, stop_id_mat_entropy_bs.shape[0] + 1):  # es stands for entropy threshold
        #   entropy
        e_matrix = _get_nanreplaced_es_matrix(es, stop_id_mat_entropy, org_matrix)
        e_predictions = get_conv_predictions(e_matrix)
        accuracies.append(get_accuracies_predictions(gt_predictions, e_predictions))
        e_matrix_bs = _get_nanreplaced_es_matrix(es, stop_id_mat_entropy_bs, bootstrap_matrix)
        e_cohen.append(_calc_kRR_cohen(None, e_matrix, e_matrix_bs)[0])
    return e_cohen, stop_id_mat_entropy, accuracies


def shuffle_rows(org_matrix):
    shuffled_indices = np.random.permutation(org_matrix.shape[0])
    shuffled_matrix = org_matrix[shuffled_indices]
    org_matrix = shuffled_matrix
    return org_matrix


def _get_nanreplaced_es_matrix(n, stop_id_matrix, org_matrix):
    """
        given the stop_id_matrix nth row (i.e., stopping ideas for early stopping absolute at n),
        replace everything below that id in the original matrix with nan values
    :param n:
    :param stop_id_matrix:
    :param org_matrix:
    :return:
    """
    es_matrix = np.copy(org_matrix)
    for q_idx in range(es_matrix.shape[1]):
        es_matrix[stop_id_matrix[n - 1, q_idx] + 1:, q_idx] = np.nan
    return es_matrix


def _get_dynamic_stopids(n, org_matrix):
    """
        Given the original matrix of annotations (annotators in rows, questions in columns),
        and the maximum number having to agree n,
        return a list of stopping ids for each absolute number of agreeing annotators from
            i=1 ... n/2

    :param n: maximum number having to agree
    :param org_matrix: matrix of annotations (annotators in rows, questions in columns)
    :return:
        matrix, where rows are i (i.e., absolute number of agreeing annotators) and
                      columns are the questions

    """
    cumulative_matrix = get_cumulative_matrix(org_matrix)
    ranks = (np.arange(n).reshape(n, 1) + 1).repeat(cumulative_matrix.shape[1], axis=1)
    rev_cum_matrix = (ranks - cumulative_matrix)[:n]
    stop_ids = np.matrix(
        [_get_early_stopping_ids(i, cumulative_matrix, rev_cum_matrix) for i in range(1, round(n / 2) + 1)])
    return stop_ids


def _get_early_stopping_ids(n, cumulative_matrix, rev_cum_matrix):
    """
        Given a cumulative matrix and it's reverse, i.e., sth like
            0, 1, 0, 1              1, 0, 1, 0
            0, 2, 1, 1,     and     2, 0, 1, 1
            0, 3, 1, 2              3, 0, 2, 1
        for each question (columns), get the earliest agreement of an absolute of n "annotators" (rows)
        will go through annotations from top to bottom, deterministic

    :param n:
    :param cumulative_matrix:
    :param rev_cum_matrix:
    :return:
    """
    earliest_n_agreements = [min(
        (list(np.where(cum_col == n)[0]) + [len(cum_col)])[0],  # earliest agreement paraphrase
        (list(np.where(rev_col == n)[0]) + [len(cum_col)])[0]  # earliest agreement non-paraphrase
    )
        for index, (cum_col, rev_col) in enumerate(zip(cumulative_matrix.T, rev_cum_matrix.T))]
    return earliest_n_agreements


def _get_early_entropy_ids(org_matrix, max_votes=10, min_votes=3, dynamic=False):
    """
        Given a cumulative matrix and it's reverse, i.e., sth like
            0, 1, 0, 1              1, 0, 1, 0
            0, 2, 1, 1,     and     2, 0, 1, 1
            0, 3, 1, 2              3, 0, 2, 1
        for each question (columns),
            get the earliest agreement that passes a given entropy threshold
        will go through annotations from top to bottom, deterministic

    :param n:
    :param cumulative_matrix:
    :param rev_cum_matrix:
    :return:
    """
    if max_votes > 20:
        warnings.warn(f"Number of maximum voters for entropy calculation might be too big: {max_votes}")
    cumulative_matrix = get_cumulative_matrix(org_matrix)

    entropy_stopping_numbers = []
    if not dynamic:
        entropies = ENTROPY_THRESHOLDS
    else:
        entropies = DYNAMIC_ENTROPY_THRESHOLDS
    for entropy in entropies:  # for all different entropy thresholds that are considered
        e_row = []
        for index, q_col in enumerate(cumulative_matrix.T):
            n_annotators = min_votes
            votes_share = q_col[n_annotators - 1] / n_annotators
            while ((not dynamic and (binary_entropy(votes_share) > entropy) and (n_annotators < max_votes)) or
                   (dynamic and dynamic_entropy(votes_share, n_annotators) > entropy) and (n_annotators < max_votes)):
                n_annotators += 1
                votes_share = q_col[n_annotators - 1] / n_annotators
            e_row.append(n_annotators - 1)
        entropy_stopping_numbers.append(e_row)

    return np.array(entropy_stopping_numbers)


def _get_entropy_stopping_ids(entropy_thresh, cumulative_matrix, max_votes=20, min_votes=3):
    """
        for all questions (columns in the matrices), get the # of annotators necessary to
        either reach the given entropy threshold or the maximum number of annotators max_votes
    :param entropy_thresh:
    :param cumulative_matrix:
    :param rev_cum_matrix:
    :param max_votes:
    :param min_votes:
    :return:
    """
    if max_votes > 20:
        warnings.warn(f"Number of maximum voters for entropy calculation might be too big: {max_votes}")
    assert (1 > entropy_thresh > 0)
    entropy_stopping_numbers = []
    for index, cum_col in enumerate(cumulative_matrix.T):
        n_annotators = min_votes
        votes_share = cum_col[n_annotators - 1] / n_annotators
        while (binary_entropy(votes_share) > entropy_thresh) and n_annotators < max_votes:
            n_annotators += 1
            votes_share = cum_col[n_annotators - 1] / n_annotators
        entropy_stopping_numbers.append(n_annotators - 1)

    return entropy_stopping_numbers


def get_prediction_share_from_df(highlight_df):
    binary_dict = annotation_df.get_annotation_dict(highlight_df)
    shares, nbr_annotators = _get_prediction_share(binary_dict["complete_binaries"])
    return shares, binary_dict["q_ids"], nbr_annotators


def get_conv_predictions(binary_annotation_matrix):
    """
        for the binary annotation matrix, return the binary predictions for each question item
    :param binary_annotation_matrix:
    :return:
    """
    shares, _ = _get_prediction_share(binary_annotation_matrix)
    return [1 if share >= 0.5 else 0 for share in shares]


def _get_prediction_share(binary_annotation_matrix):
    if type(binary_annotation_matrix) == list:
        binary_annotation_matrix = np.array(binary_annotation_matrix)
    shares = []
    nbr_annotators = []
    for col in binary_annotation_matrix.T:
        not_nan_values = col[~np.isnan(col)]  # Filter out NaNs from each column
        shares.append(np.sum(not_nan_values == 1) / len(not_nan_values))
        nbr_annotators.append(len(not_nan_values))
    return shares, nbr_annotators


def _get_conv_predictions(relative_cum_matrix, conv_share=None):
    if conv_share is None:
        # get the indices of "ground truth", .e., converged share
        converged_share = relative_cum_matrix[
            (~np.isnan(relative_cum_matrix)).cumsum(0).argmax(0), list(range(relative_cum_matrix.shape[1]))]
    else:
        converged_share = conv_share
    gt_preds = np.vectorize(predict, otypes=[np.float])(converged_share)
    return converged_share, gt_preds
