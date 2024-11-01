"""
    script for error analysis between gpt4 and deberta
"""
from build_prompts import INSTRUCT_PROMPT_TEST
from conversation_outcome import DEDUPLICATE_PATH, ALL_PATH
from eval_token_classifier import get_tokenclassification_predictions
from evaluate_icl_responses import GPT4_TEST_RESPONSE_PATH, load_json_file, get_model_predictions
from paraphrase import token_classifier, interview_data
from paraphrase.annotation_data import TEST_PATH, get_aggregated_human_anns
from paraphrase.utility.PC_utility import get_qids_from_file
from paraphrase.utility.tex_highlighting import gen_guest_host_tex

classifier = token_classifier.TokenClassifier(DEDUPLICATE_PATH)

# list(get_qids_from_file(TEST_PATH))  # this does not always return the same list
org_q_ids = ['CNN-10486-11', 'CNN-111610-7', 'CNN-11451-13', 'CNN-134633-5', 'CNN-139432-8', 'CNN-14551-7',
             'CNN-148030-5', 'CNN-151830-18', 'CNN-166487-12', 'CNN-166930-17', 'CNN-168166-4', 'CNN-178518-5',
             'CNN-179766-6', 'CNN-192500-5', 'CNN-19396-7', 'CNN-208335-11', 'CNN-214354-5', 'CNN-217116-7',
             'CNN-217116-9', 'CNN-22088-3', 'CNN-22326-7', 'CNN-226205-5', 'CNN-231880-5', 'CNN-26711-7',
             'CNN-276969-5', 'CNN-286124-3', 'CNN-312194-3', 'CNN-3199-7', 'CNN-327899-13', 'CNN-339958-3',
             'CNN-341938-3', 'CNN-344762-3', 'CNN-350238-7', 'CNN-370927-7', 'CNN-378674-9', 'CNN-38493-11',
             'CNN-390220-9', 'CNN-395083-3', 'CNN-395861-6', 'CNN-41259-4', 'CNN-41861-3', 'CNN-42770-7', 'CNN-50949-3',
             'CNN-57033-5', 'CNN-58514-3', 'CNN-62525-9', 'CNN-66127-3', 'CNN-68683-5', 'CNN-72706-11', 'CNN-80522-7',
             'CNN-87376-7', 'CNN-98051-5', 'NPR-10683-11', 'NPR-10683-9', 'NPR-11718-10', 'NPR-13225-5', 'NPR-14546-4',
             'NPR-15761-9', 'NPR-15848-3', 'NPR-15913-5', 'NPR-17496-10', 'NPR-18917-8', 'NPR-19127-3', 'NPR-20376-3',
             'NPR-20571-5', 'NPR-20855-6', 'NPR-21694-3', 'NPR-22848-7', 'NPR-23629-3', 'NPR-23790-5', 'NPR-2541-23',
             'NPR-2622-10', 'NPR-30121-10', 'NPR-3135-3', 'NPR-32322-11', 'NPR-32322-9', 'NPR-32992-12', 'NPR-39855-9',
             'NPR-41134-14', 'NPR-41743-13', 'NPR-4238-7', 'NPR-42649-11', 'NPR-4329-4', 'NPR-44121-10', 'NPR-45058-8',
             'NPR-46180-5', 'NPR-5871-3', 'NPR-636-7', 'NPR-733-15', 'NPR-7619-6', 'NPR-8339-5', 'NPR-8678-6']
interview = interview_data.MediaSumProcessor()
guest_tokens_per_qid, host_tokens_per_qid, human_anns_per_qid, human_class_per_qid = (
    get_aggregated_human_anns(interview, org_q_ids))
gpt4_class_per_qid, gpt4_guest_weights_per_qid, _, gpt4_host_weights_per_qid = (
    get_model_predictions(guest_tokens_per_qid, host_tokens_per_qid, load_json_file(GPT4_TEST_RESPONSE_PATH),
                          prompt_dict=load_json_file(INSTRUCT_PROMPT_TEST), gpt4=True))

# q_ids = ['CNN-276969-5', 'CNN-139432-8', 'NPR-733-15', 'NPR-2541-23', 'CNN-22326-7']   # ['NPR-23629-3', 'NPR-19127-3', 'NPR-3135-3', 'NPR-8339-5', 'CNN-214354-5']
q_ids = org_q_ids

human_binary = [1 if (human_class_per_qid[q_id][0] / human_class_per_qid[q_id][1] >= 0.5) else 0
                for q_id in q_ids]
human_highlights_guest = [[1 if weight >= 0.5 else 0 for weight in human_anns_per_qid[q_id]["guest_weights"]]
                          for q_id in q_ids]
human_guest_weights = [[weight for weight in human_anns_per_qid[q_id]["guest_weights"]] for q_id in q_ids]
human_highlights_host = [[1 if weight >= 0.5 else 0 for weight in human_anns_per_qid[q_id]["host_weights"]]
                         for q_id in q_ids]
human_host_weights = [[weight for weight in human_anns_per_qid[q_id]["host_weights"]] for q_id in q_ids]
gpt4_binary = [1 if ((gpt4_class_per_qid[q_id][0] + gpt4_class_per_qid[q_id][1]) > 0) and  # no divide by zero
                    (gpt4_class_per_qid[q_id][0] /
                     (gpt4_class_per_qid[q_id][0] + gpt4_class_per_qid[q_id][1]) >= 0.5) else 0
               for q_id in q_ids]
deberta_binary, _, _, _, _, deberta_weights_guest, deberta_weights_host = (
    get_tokenclassification_predictions(q_ids, guest_tokens_per_qid, host_tokens_per_qid, classifier, pred_thresh=0.50))
print(f"total paraphrase classifications by deberta: {sum(deberta_binary)}")

deberta_highlights_guest = [[1 if (weight >= 0.5) or (q_class and weight >= 0.50) else 0
                             for weight in deberta_weights_guest[q_ids.index(q_id)]]
                            for q_class, q_id in zip(deberta_binary, q_ids)]
deberta_highlights_host = [[1 if (weight >= 0.5) or (q_class and weight >= 0.50) else 0
                            for weight in deberta_weights_host[q_ids.index(q_id)]]
                           for q_class, q_id in zip(deberta_binary, q_ids)]


def print_hl_turn(guest_tokens, guest_hlids, host_tokens, host_hlids):
    print(
        (" ".join([token.upper() if pred >= 0.5 else token
                   for token, pred in zip(guest_tokens, guest_hlids)]),
         " ".join([token.upper() if pred >= 0.5 else token
                   for token, pred in zip(host_tokens, host_hlids)]))
    )


q_ids_gpt4_notdeberta = [q_id for gt_class, deberta_class, gpt4_class, q_id in
                         zip(human_binary, deberta_binary, gpt4_binary, q_ids)
                         if (gt_class != deberta_class) and (gpt4_class == gt_class)]
q_ids_deberta_notgpt4 = [q_id for gt_class, deberta_class, gpt4_class, q_id in
                         zip(human_binary, deberta_binary, gpt4_binary, q_ids)
                         if (gt_class != gpt4_class) and (deberta_class == gt_class)]
q_ids_both_wrong = [q_id for gt_class, deberta_class, gpt4_class, q_id in
                    zip(human_binary, deberta_binary, gpt4_binary, q_ids)
                    if (gt_class != deberta_class) and (gt_class != gpt4_class)]
q_ids_both_correct = [q_id for gt_class, deberta_class, gpt4_class, q_id in
                      zip(human_binary, deberta_binary, gpt4_binary, q_ids)
                      if (gt_class == deberta_class) and (gt_class == gpt4_class)]

# print all examples misclassifications deberta, but not gpt4
print(f"Number of misclassifications by deberta, while gpt4 is correct: {len(q_ids_gpt4_notdeberta)}")
for gt_class, deberta_class, gpt4_class, q_id in zip(human_binary, deberta_binary, gpt4_binary, q_ids):
    if q_id in q_ids_gpt4_notdeberta:
        print(f"QID: {q_id}")
        print(f"Ground Truth: {gt_class}")
        print_hl_turn(guest_tokens_per_qid[q_id], human_highlights_guest[q_ids.index(q_id)],
                      host_tokens_per_qid[q_id], human_highlights_host[q_ids.index(q_id)])
        print(human_guest_weights[q_ids.index(q_id)])
        print(human_host_weights[q_ids.index(q_id)])
        print(f"Deberta: {deberta_class}")

        print_hl_turn(guest_tokens_per_qid[q_id], deberta_highlights_guest[q_ids.index(q_id)],
                      host_tokens_per_qid[q_id], deberta_highlights_host[q_ids.index(q_id)])
        print(deberta_weights_guest[q_ids.index(q_id)])
        print(deberta_weights_host[q_ids.index(q_id)])

        print(f"GPT4: {gpt4_class}")
        print_hl_turn(guest_tokens_per_qid[q_id], gpt4_guest_weights_per_qid[q_id],
                      host_tokens_per_qid[q_id], gpt4_host_weights_per_qid[q_id])

        print()

# print all examples misclassifications gpt4, but not deberta
print(f"Number of misclassifications by gpt4, while deberta is correct: {len(q_ids_deberta_notgpt4)}")
for gt_class, deberta_class, gpt4_class, q_id in zip(human_binary, deberta_binary, gpt4_binary, q_ids):
    if q_id in q_ids_deberta_notgpt4:
        print(f"QID: {q_id}")
        print(f"Ground Truth: {gt_class}")
        print_hl_turn(guest_tokens_per_qid[q_id], human_highlights_guest[q_ids.index(q_id)],
                      host_tokens_per_qid[q_id], human_highlights_host[q_ids.index(q_id)])
        print(human_guest_weights[q_ids.index(q_id)])
        print(human_host_weights[q_ids.index(q_id)])
        print(f"Deberta: {deberta_class}")

        print_hl_turn(guest_tokens_per_qid[q_id], deberta_highlights_guest[q_ids.index(q_id)],
                      host_tokens_per_qid[q_id], deberta_highlights_host[q_ids.index(q_id)])
        print(deberta_weights_guest[q_ids.index(q_id)])
        print(deberta_weights_host[q_ids.index(q_id)])
        print(f"GPT4: {gpt4_class}")
        print_hl_turn(guest_tokens_per_qid[q_id], gpt4_guest_weights_per_qid[q_id],
                      host_tokens_per_qid[q_id], gpt4_host_weights_per_qid[q_id])

        print()

# print all examples misclassifications both
print(f"Number of misclassifications by both: {len(q_ids_both_wrong)}")
for gt_class, deberta_class, gpt4_class, q_id in zip(human_binary, deberta_binary, gpt4_binary, q_ids):
    if q_id in q_ids_both_wrong:
        print(f"QID: {q_id}")
        print(f"Ground Truth: {gt_class}")
        print_hl_turn(guest_tokens_per_qid[q_id], human_highlights_guest[q_ids.index(q_id)],
                      host_tokens_per_qid[q_id], human_highlights_host[q_ids.index(q_id)])
        print(human_guest_weights[q_ids.index(q_id)])
        print(human_host_weights[q_ids.index(q_id)])
        print(f"Deberta: {deberta_class}")
        print(deberta_weights_guest[q_ids.index(q_id)])
        print(deberta_weights_host[q_ids.index(q_id)])
        print_hl_turn(guest_tokens_per_qid[q_id], deberta_highlights_guest[q_ids.index(q_id)],
                      host_tokens_per_qid[q_id], deberta_highlights_host[q_ids.index(q_id)])

        print(f"GPT4: {gpt4_class}")
        print_hl_turn(guest_tokens_per_qid[q_id], gpt4_guest_weights_per_qid[q_id],
                      host_tokens_per_qid[q_id], gpt4_host_weights_per_qid[q_id])

        print()

# print all examples correct classifications both
print(f"Number of correct classifications by both: {len(q_ids_both_correct)}")
for gt_class, deberta_class, gpt4_class, q_id in zip(human_binary, deberta_binary, gpt4_binary, q_ids):
    if q_id in q_ids_both_correct:
        if q_id == "NPR-8339-5":
            print("DEBUG")
        print(f"QID: {q_id}")
        print(f"Ground Truth: {gt_class}")
        print_hl_turn(guest_tokens_per_qid[q_id], human_highlights_guest[q_ids.index(q_id)],
                      host_tokens_per_qid[q_id], human_highlights_host[q_ids.index(q_id)])
        print(human_guest_weights[q_ids.index(q_id)])
        print(human_host_weights[q_ids.index(q_id)])
        print(f"Deberta: {deberta_class}")

        print_hl_turn(guest_tokens_per_qid[q_id], deberta_highlights_guest[q_ids.index(q_id)],
                      host_tokens_per_qid[q_id], deberta_highlights_host[q_ids.index(q_id)])
        print(deberta_weights_guest[q_ids.index(q_id)])
        print(deberta_weights_host[q_ids.index(q_id)])
        print(f"GPT4: {gpt4_class}")
        print_hl_turn(guest_tokens_per_qid[q_id], gpt4_guest_weights_per_qid[q_id],
                      host_tokens_per_qid[q_id], gpt4_host_weights_per_qid[q_id])

        print()
print("Number of Questions: ", len(q_ids))
