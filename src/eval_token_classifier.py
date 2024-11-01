"""
    script to evaluate a trained token classifier
"""
import argparse
import logging
import os
from statistics import mean

from paraphrase import interview_data
from paraphrase import token_classifier
from paraphrase.utility.PC_utility import get_qids_from_file
from paraphrase.utility.project_functions import set_logging
from evaluate_icl_responses import jaccard_overlap_from_labels, \
    print_f1_classifications, sample_10_incorrect_preds
from paraphrase.annotation_data import get_aggregated_human_anns, TEST_PATH, DEV_PATH, TRAIN_PATH

PRED_THRESH = 0.50


def main(model_path, eval_set):
    """

    :param eval_set:
    :param model_path: path to _model
    :return:
    """
    set_logging()
    assert eval_set in ["train", "dev", "test"], logging.error("eval_set must be one of 'train', 'dev', or 'test'")
    if not os.path.exists(model_path):
        logging.warning(f"model_path: {model_path} does not exist. Trying to load _model from hub.")

    try:
        classifier = token_classifier.TokenClassifier(model_path)
        logging.info("Model successfully loaded.")
    except ValueError:
        logging.error(f"Could not load _model. Exiting.")
        return

    #   GET human annotations
    if eval_set == "dev":
        item_path = DEV_PATH
    elif eval_set == "test":
        item_path = TEST_PATH
    else:
        item_path = TRAIN_PATH
    q_ids = list(get_qids_from_file(item_path))
    interview = interview_data.MediaSumProcessor()
    guest_tokens_per_qid, host_tokens_per_qid, human_anns_per_qid, human_class_per_qid = (
        get_aggregated_human_anns(interview, q_ids))

    model_binary, model_classifications, model_highlights_guest, model_highlights_host, prompt_dict, _, _ = (
        get_tokenclassification_predictions(q_ids, guest_tokens_per_qid, host_tokens_per_qid, classifier,
                                            pred_thresh=PRED_THRESH))

    human_binary = [1 if (human_class_per_qid[q_id][0] / human_class_per_qid[q_id][1] >= 0.5) else 0
                    for q_id in q_ids]
    human_share = [human_class_per_qid[q_id][0] / human_class_per_qid[q_id][1] for q_id in q_ids]

    correct_mh = print_f1_classifications(human_binary, model_binary, human_share)

    # Evaluate Highlights
    human_highlights_guest = [[1 if weight >= 0.5 else 0 for weight in human_anns_per_qid[q_id]["guest_weights"]]
                              for q_id in q_ids]
    human_highlights_host = [[1 if weight >= 0.5 else 0 for weight in human_anns_per_qid[q_id]["host_weights"]]
                             for q_id in q_ids]
    avg_model_guest_hl_len = mean([sum(hl) for hl in model_highlights_guest if sum(hl) > 0])
    avg_human_guest_hl_len = mean([sum(hl) for hl in human_highlights_guest if sum(hl) > 0])
    print(f"\t {avg_human_guest_hl_len} avg human guest hl length, "
          f"{avg_model_guest_hl_len} avg _model guest hl length.")
    avg_model_host_hl_len = mean([sum(hl) for hl in model_highlights_host if sum(hl) > 0])
    avg_human_host_hl_len = mean([sum(hl) for hl in human_highlights_host if sum(hl) > 0])
    print(f"\t {avg_human_host_hl_len} avg human host hl length, "
          f"{avg_model_host_hl_len} avg _model host hl length.")

    jaccard_overlap_from_labels(human_highlights_guest, model_highlights_guest, q_ids, "guest")
    jaccard_overlap_from_labels(human_highlights_host, model_highlights_host, q_ids, "host")

    sample_10_incorrect_preds(correct_mh, guest_tokens_per_qid, host_tokens_per_qid, human_anns_per_qid,
                              human_class_per_qid, model_highlights_guest, model_highlights_host,
                              prompt_dict, q_ids)

    inconsistent_class = [1 if (1 in item[0] and 1 not in item[1]) or (1 not in item[0] and 1 in item[1]) else 0
                          for item in model_classifications]
    print(f"{sum(inconsistent_class) / len(q_ids)} = {sum(inconsistent_class)} / {len(q_ids)} "
          f"inconsistent classifications, i.e., highligth in one but not the other sentence")


def get_tokenclassification_predictions(q_ids, guest_tokens_per_qid, host_tokens_per_qid, classifier, pred_thresh=0.5):
    data_list = []
    prompt_dict = {}
    for q_id in q_ids:
        item = {
            "sentence1": guest_tokens_per_qid[q_id],
            "sentence2": host_tokens_per_qid[q_id]
        }
        data_list.append(item)
        prompt_dict[q_id] = f"Guest: {' '.join(guest_tokens_per_qid[q_id])}\n"
        prompt_dict[q_id] += f"Host: {' '.join(host_tokens_per_qid[q_id])}"
    del item
    model_classifications, model_probabilities = classifier.inference(data_list, return_prob=True,
                                                                      pred_thresh=pred_thresh)
    model_weights_guest = [item[0] for item in model_probabilities]
    model_weights_host = [item[1] for item in model_probabilities]
    model_binary = [
        1 if (1 in model_classifications[i][0]) and (1 in model_classifications[i][1])  # both sentences highlighted
        else 0 for i in range(len(q_ids))]

    model_highlights_guest = [item[0] for item in model_classifications]
    model_highlights_guest = [[1 if (weight >= 0.5) or (q_class and weight >= 0.50) else 0  # 0.45
                               for weight in model_weights_guest[q_ids.index(q_id)]]
                              for q_class, q_id in zip(model_binary, q_ids)]
    model_highlights_host = [item[1] for item in model_classifications]
    model_highlights_host = [[1 if (weight >= 0.5) or (q_class and weight >= 0.50) else 0  # 0.45
                              for weight in model_weights_host[q_ids.index(q_id)]]
                             for q_class, q_id in zip(model_binary, q_ids)]
    return model_binary, model_classifications, model_highlights_guest, model_highlights_host, prompt_dict, \
        model_weights_guest, model_weights_host


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Token Classifier Evaluation Configurations')
    parser.add_argument('-_model', '--model_path',
                        help="path where _model was saved or huggingface identifier")
    parser.add_argument('-set', '--dataset',
                        help="dataset that the _model is supposed to be evaluated on, i.e., 'train', 'dev', or 'test'")

    args = parser.parse_args()

    main(model_path=args.model_path, eval_set=args.dataset)
