"""
    You don't want this

    legacy test
"""
from typing import List
from unittest import TestCase

import numpy as np

import paraphrase.utility.annotation_df as annotation_df
from paraphrase import annotation_data
from paraphrase.LLMs import read_in_responses
from collections import Counter

from paraphrase.utility.stats import jaccard_overlap_for_highlights
from tex_highlighting import gen_guest_host_tex


class Test(TestCase):

    def setUp(self) -> None:
        folder = "fixtures/annotations/"
        pc100_annotations_paths = ["2023-06-13_5050_0-9_July+24,+2023_01.44.tsv",
                                   # 2023-06-13_Paraphrase-Candidates_0-9_June+26,+2023_01.01.tsv
                                   "2023-06-13_5050_10-19_June+26,+2023_01.01.tsv",
                                   "2023-06-13_5050_20-29_July+24,+2023_01.44.tsv",
                                   # 2023-06-13_Paraphrase-Candidates_20-29_June+26,+2023_01.01.tsv
                                   "2023-06-13_5050_30-39_June+26,+2023_01.01.tsv",
                                   "2023-06-13_5050_40-49_June+26,+2023_01.02.tsv",
                                   "2023-06-13_5050_50-59_June+26,+2023_01.02.tsv",
                                   "2023-06-13_5050_60-69_June+26,+2023_01.02.tsv",
                                   "2023-06-13_5050_70-79_June+26,+2023_01.02.tsv",
                                   "2023-06-13_5050_80-89_July+24,+2023_01.45.tsv",
                                   # 2023-06-13_Paraphrase-Candidates_80-89_June+26,+2023_01.02.tsv
                                   "2023-06-13_5050_90-99_June+26,+2023_01.02.tsv"]
        self.pc100_annotations_paths = [folder + response_path for response_path in pc100_annotations_paths]
        self.highlight_df = annotation_df.merge_transform(self.pc100_annotations_paths)

    def test_hl_overlap_gpt(self):

        f_path = '../../result/Models/gpt4_hl-responses.txt'
        # f_path = '../../result/Models/gpt3-5_hl-responses.txt'
        hl_answers = read_in_responses(f_path)
        # print(hl_answers)
        test_ids = ['CNN-177596-7', 'NPR-8678-6', 'CNN-350097-7', 'CNN-235909-21', 'NPR-23442-3',
                    'CNN-323090-3', 'CNN-72706-11', 'CNN-378275-5', 'CNN-300212-13',
                    'CNN-376903-8', 'CNN-111931-3', 'NPR-14133-4', 'NPR-18055-16', 'CNN-19564-5',
                    'NPR-15913-5', 'CNN-333433-3', 'CNN-6197-7', 'CNN-412181-3', 'CNN-108240-3',
                    'CNN-98051-7', 'NPR-29413-12', 'NPR-6494-5', 'NPR-9809-12', 'CNN-32771-11',
                    'CNN-6322-5', 'NPR-19980-12', 'CNN-234343-5', 'CNN-67068-5', 'CNN-169621-5',
                    'CNN-363088-3', 'CNN-165408-7', 'CNN-83-7', 'CNN-377264-11', 'CNN-32771-3',
                    'CNN-55894-3', 'CNN-208335-11', 'CNN-123568-3', 'NPR-2705-4', 'CNN-350238-9',
                    'CNN-195140-5', 'NPR-35389-8', 'CNN-189512-7', 'CNN-14551-3', 'CNN-268015-12',
                    'CNN-270198-3', 'CNN-154452-7', 'NPR-9809-7', 'NPR-44959-16', 'NPR-42397-7',
                    'NPR-3135-3', 'CNN-266429-3', 'CNN-74126-3', 'CNN-13892-3', 'CNN-98333-3',
                    'CNN-179766-6', 'NPR-6494-7', 'CNN-398359-4', 'NPR-733-15', 'CNN-332405-27',
                    'CNN-14551-7', 'CNN-98333-7', 'NPR-4238-7', 'NPR-24552-10', 'CNN-4434-3',
                    'CNN-154452-5', 'CNN-79698-3', 'NPR-17734-16', 'CNN-235909-17', 'NPR-29413-7',
                    'CNN-111931-5', 'NPR-35389-10', 'CNN-66474-3', 'CNN-350238-7', 'CNN-108240-7',
                    'CNN-111931-7', 'CNN-293273-3', 'CNN-64125-11', 'CNN-30808-15', 'NPR-22848-7',
                    'CNN-108240-11', 'CNN-141300-3', 'CNN-98051-5', 'NPR-23442-5', 'CNN-19396-7',
                    'NPR-44959-24', 'NPR-9269-7', 'CNN-76224-3', 'CNN-114404-5', 'NPR-9269-19',
                    'CNN-19396-9', 'CNN-384602-3', 'CNN-371078-3', 'CNN-15655-13', 'CNN-6322-7',
                    'CNN-377264-3', 'NPR-8248-3', 'NPR-5871-3', 'CNN-368578-5', 'CNN-28089-7',
                    'NPR-25659-9']

        to_print = ["CNN-376903-8", "CNN-74126-3", "CNN-64125-11", "NPR-42397-7", "CNN-350238-9", "CNN-378275-5",
                    "NPR-23442-3",
                    "CNN-235909-21", "CNN-6197-7", "CNN-98333-7", "CNN-268015-12", "CNN-4434-3"]

        guest_msgs, guest_tokens, host_msgs, host_tokens, q_ids, vote_strs = \
            annotation_data._get_annotations_per_qid(self.highlight_df)

        self.assertListEqual(q_ids.tolist(), test_ids)
        jaccards_guest = []
        jaccards_host = []

        for q_id, cur_hl_answers, votes, guest_weights, guest_toks, host_weights, host_toks in \
                zip(q_ids, hl_answers, vote_strs, guest_msgs, guest_tokens, host_msgs, host_tokens):
            print(guest_toks)

            if len(cur_hl_answers) > 0:
                guest_str, host_str = get_hl_str(cur_hl_answers[0])
                # if "ph" in guest_str:
                #     print()
                guest_hl_ids = get_hl_ids(guest_toks, guest_str)
                print(guest_hl_ids)

                guest_votes = [get_hl_ids(guest_toks, get_hl_str(hl_a)[0]) for hl_a in cur_hl_answers]
                g_llm_weights = self.get_llm_votes(guest_votes)
                yes_no_token_guest = [i for i, g_w in enumerate(guest_weights) if g_w >= 0.5]
                if len(g_llm_weights) > 0:
                    j_o = jaccard_overlap_for_highlights([[g_llm_weights, yes_no_token_guest]])
                    jaccards_guest.append(j_o)

                host_votes = [get_hl_ids(host_toks, get_hl_str(hl_a)[1]) for hl_a in cur_hl_answers]
                h_llm_weights = self.get_llm_votes(host_votes)
                yes_no_token_host = [i for i, g_w in enumerate(host_weights) if g_w >= 0.5]
                if len(h_llm_weights) > 0:
                    j_o = jaccard_overlap_for_highlights([[h_llm_weights, yes_no_token_host]])
                    jaccards_host.append(j_o)

                if (q_id in to_print) or (
                        (len(cur_hl_answers) / 9 < 0.5) and int(votes.split("/")[0]) / int(votes.split("/")[1]) >= 0.5):
                    print(r'''\textbf{''' + q_id + r'''}:''' + f"{len(cur_hl_answers)}/9" + r''' \\''' + "\n")
                    g_counter = Counter([num for s in guest_votes for num in s])
                    h_counter = Counter([num for s in host_votes for num in s])
                    print(gen_guest_host_tex([guest_toks, host_toks], [
                        [(g_counter[i] / len(cur_hl_answers)) if i in g_counter else 0 for i in range(len(guest_toks))],
                        [(h_counter[i] / len(cur_hl_answers)) if i in h_counter else 0 for i in
                         range(len(host_toks))]]))

            elif q_id in to_print:
                print(r'''\textbf{''' + q_id + r'''}:''' + f"{len(cur_hl_answers)}/9" + r''' \\''' + "\n")
                print(gen_guest_host_tex([guest_toks, host_toks], [
                    [0 for i in range(len(guest_toks))],
                    [0 for i in range(len(host_toks))]]))

        # print(jaccards_guest)
        print(np.nanmean(jaccards_guest))
        print(np.nanmean(jaccards_host))

    def get_llm_votes(self, guest_votes):
        # flatten the list of sets into one list
        all_numbers = [num for s in guest_votes for num in s]
        # count the occurrences of each number
        counter = Counter(all_numbers)
        # calculate the threshold for majority
        majority_threshold = len(guest_votes) / 2
        # get the numbers that occur in the majority of lists
        majority_numbers = [num for num, count in counter.items() if count > majority_threshold]
        return majority_numbers


def get_hl_str(hl: str):
    guest, host = hl.split("\n")[0:2]
    guest = guest.replace('Guest: ', '').replace('Guest:', '').replace('...', '')
    if guest[0] == '"':
        guest = guest[1:]
        if guest[-1] == '"':
            guest = guest[:-1]
    if len(host) == 0:
        host = hl.split("\n")[2]
    host = host.replace('Host: ', '').replace('Host:', '').replace('...', '')
    if host[0] == '"':
        host = host[1:]
        if host[-1] == '"':
            host = host[:-1]

    return guest, host


def get_hl_ids(utt_toks: List[str], hl_str: str):
    hl_toks = hl_str.split(" ")
    cur_hl_tok = 0

    utt_ids = []
    for i, tok in enumerate(utt_toks):
        if cur_hl_tok >= len(hl_toks):
            break
        if hl_toks[cur_hl_tok] == tok:
            utt_ids.append(i)
            cur_hl_tok += 1
        elif ignore_punctuation(hl_toks[cur_hl_tok]) == ignore_punctuation(tok):
            # almost match
            utt_ids.append(i)
            cur_hl_tok += 1

    if len(utt_ids) != len(hl_toks):
        print(f"Something went wrong at {hl_toks[cur_hl_tok]} with {hl_str} and {utt_toks}")

    return utt_ids


def ignore_punctuation(utt: str):
    return utt.replace(',', '').replace('.', '').replace(';', '')
