"""
    script for sampling the initial 100 candidates of the 50/50 set for annotation by 20 annotators
"""
from typing import List, Set, Dict

from paraphrase.utility.PC_utility import PC_ID_COLUMN, PILOT_AN_PATH_1, PILOT_AN_PATH_2
from paraphrase.lead_pc import get_pc_first_batch_label_dict, AnnotationLabels
from paraphrase.utility.project_functions import SEED
from paraphrase.utility.annotation_df import AnnotationColumns
import pandas as pd
import random
from paraphrase.interview_data import MediaSumProcessor

COMBINATIONS_PARAPRHASE = [[[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
                           [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]],
                           [[1, 1, 1, 1, 0, 0, 0]],
                           [[1, 1, 0, 1, 0, 1, 0]],
                           [[1, 1, 0, 1, 1, 0, 1]],
                           [[1, 1, 0, 1, 0, 0, 0]],
                           [[0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 1]],
                           [[0, 0, 1, 1, 0, 0, 1]],
                           [[0, 0, 1, 1, 0, 0, 0]],
                           [[0, 1, 0, 1, 0, 1, 0]],
                           [[0, 0, 0, 1, 0, 1, 0]],
                           [[0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0, 1]],
                           [[0, 0, 0, 1, 1, 0, 0]],
                           [[0, 0, 0, 1, 1, 0, 1]]]

COMBINATIONS_NON_PARAPRHASE = [[[0, 0, 0, 1, 0, 0]],
                               [[0, 0, 0, 1, 1, 0]],
                               [[1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 1]],
                               [[1, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1]],
                               [[0, 1, 0, 0, 1, 0]],
                               [[0, 0, 0, 0, 1, 0]],
                               [[0, 0, 0, 0, 0, 0]],
                               [[0, 0, 1, 0, 0, 1]],
                               [[0, 0, 1, 0, 0, 0]],
                               [[0, 0, 0, 0, 0, 1]]]

NBR_PER_COMBINATION_PARAPHRASE = [5, 5, 5, 2, 1, 12, 4, 3, 3, 5, 1, 1, 1, 2]

NBR_PER_COMBINATION_NON_PARAPHRASE = [7, 3, 5, 5, 5, 5, 5, 5, 5, 5]

"""
    "Sample" 50 Paraphrases -- ['rep', 'hlex', 'uni', 'clear', 'prag', 'per', 'dir']
        5 from '# combination (0, 0, 0, 0, 1, 0, 0)': 4,  (PRAGMATIC) U '# combination (0, 0, 0, 0, 1, 0, 1)': 1,  (PRAGMATIC & DIRECTIONAL)
        5 from '# combination (0, 0, 0, 0, 0, 0, 0)': 1, U '# combination (0, 0, 0, 0, 0, 0, 1)': 4,  (DIRECTIONAL)
        5 from '# combination (1, 1, 1, 1, 0, 0, 0)': 9,  (CLEAR & REPETITION & HIGH SIM & UNIVERSAL) 
        2 from '# combination (1, 1, 0, 1, 0, 1, 0)': 2,  (CLEAR & REPETITION & HIGH SIM & PERSPECTIVE-SHIFT)
        1 from '# combination (1, 1, 0, 1, 1, 0, 1)': 1,  (CLEAR & REPETITION & HIGH SIM & PRAGMATIC & DIRECTIONAL)
        12 from '# combination (1, 1, 0, 1, 0, 0, 0)': 35, (CLEAR & REPETITION & HIGH SIM)
        4 from '# combination (0, 1, 1, 1, 0, 0, 0)': 5,  (CLEAR & HIGH SIM & UNIVERSAL) U '# combination (0, 1, 1, 1, 0, 0, 1)': 1,  (CLEAR & HIGH SIM & UNIVERSAL & DIRECTIONAL)
        3 from '# combination (0, 0, 1, 1, 0, 0, 1)': 5,  (CLEAR & UNIVERSAL & DIRECTIONAL)
        3 from '# combination (0, 0, 1, 1, 0, 0, 0)': 4,  (CLEAR & UNIVERSAL)
        5 from '# combination (0, 1, 0, 1, 0, 1, 0)': 5,  (CLEAR & HIGH SIM & PERSPECTIVE-SHIFT)
        1 from '# combination (0, 0, 0, 1, 0, 1, 0)': 1,  (CLEAR & PERSPECTIVE-SHIFT)
        1 from '# combination (0, 1, 0, 1, 0, 0, 0)': 2,  (CLEAR & HIGH SIM) U combination (0, 1, 0, 1, 0, 0, 0)': 2,  (CLEAR & HIGH SIM)
            '# combination (0, 1, 0, 1, 0, 0, 1)': 1,  (HIGH SIM & CLEAR & DIRECTIONAL)
        1 from '# combination (0, 0, 0, 1, 1, 0, 0)': 1,  (CLEAR & PRAGMATIC)
        2 from '# combination (0, 0, 0, 1, 1, 0, 1)': 2,  (CLEAR & PRAGMATIC & DIRECTIONAL)
    
    "Sample" 50 Non-Paraphrases -- ['hlex', 'partial', 'conc', 'context', 'unrel', 'rel']
        7 from '# non paraphrase combination (0, 0, 0, 1, 0, 0)': 68 -- (CONTEXT)
        3 from '# non paraphrase combination (0, 0, 0, 1, 1, 0)': 20 -- (CONTEXT & UNRELATED)
        5 from # non paraphrase combination (1, 1, 0, 0, 0, 1)': 5, -- (HIGH SIM & PARTIAL & RELATED) U '# non paraprhase combination (1, 1, 1, 0, 0, 1)': 1, -- (HIGH SIM & PARTIAL & CONCLUSION & RELATED)
        5 from '# non paraphrase combination (1, 0, 0, 0, 0, 1)': 9, -- (HIGH SIM & RELATED) U '# non paraprhase combination (1, 0, 1, 0, 0, 1)': 1, -- (HIGH SIM & CONCLUSION & RELATED)
        5 from '# non paraphrase combination (0, 1, 0, 0, 1, 0)': 10, -- (PARTIAL & UNRELATED)
        5 from '# non paraphrase combination (0, 0, 0, 0, 1, 0)': 90, -- (UNRELATED)
        5 from '# non paraphrase combination (0, 0, 0, 0, 0, 0)': 308, -- REST - "soft unrelated"
        5 from '# non paraphrase combination (0, 0, 1, 0, 0, 1)': 18, -- (CONCLUSION & RELATED)
        5 from '# non paraphrase combination (0, 0, 1, 0, 0, 0)': 23, -- (CONCLUSION)
        5 from '# non paraphrase combination (0, 0, 0, 0, 0, 1)': 40, -- (RELATED)
    
"""


def main():
    # load Annotations
    an_path_1 = PILOT_AN_PATH_1
    an_path_2 = PILOT_AN_PATH_2
    concatenated, result_dict = get_pc_first_batch_label_dict(an_path_1, an_path_2)

    random.seed(SEED)

    # sample and print
    sampled_paraphrase_ids = sample_pcs(result_dict, paraphrase=True)
    assert (len(sampled_paraphrase_ids) == 50)
    get_pc_paraphrase_statistics(result_dict, sampled_paraphrase_ids)

    sampled_non_paraphrase_ids = sample_pcs(result_dict, paraphrase=False)
    assert (len(sampled_non_paraphrase_ids) == 50)
    get_pc_non_paraphrase_statistics(result_dict, sampled_non_paraphrase_ids)

    assert (len(sampled_non_paraphrase_ids & sampled_paraphrase_ids) == 0)

    # shuffle IDs
    pc_ids = list(sampled_paraphrase_ids) + list(sampled_non_paraphrase_ids)
    random.shuffle(pc_ids)

    # save IDs in tsv
    is_paraphrase = [1 if q_id in sampled_paraphrase_ids else 0 for q_id in pc_ids]
    df = pd.DataFrame(zip(pc_ids, is_paraphrase), columns=[PC_ID_COLUMN, "Is Paraphrase"])
    df.to_csv("../output/temp_100_PC_IDs.tsv", sep='\t', index=False)

    # save for inspection with text and comments in tsv
    save_tsv_for_inspection(pc_ids, is_paraphrase, concatenated)


def save_tsv_for_inspection(pc_ids, is_paraphrase, concatenated):
    interview = MediaSumProcessor()
    columns = [PC_ID_COLUMN, "Guest Utterance", "Host Utterance", "Is Paraphrase", "Comment"]
    guest_utts = []
    host_utts = []
    filtered_rows = concatenated[concatenated[AnnotationColumns.Question_ID].isin(pc_ids)
                                 & (concatenated[AnnotationColumns.Highlight_Category] == "Comment")]
    filtered_rows = filtered_rows.set_index(AnnotationColumns.Question_ID).reindex(pc_ids).reset_index()
    comments = filtered_rows[AnnotationColumns.Highlight_Choices].values.tolist()
    for q_id in pc_ids:
        ie_utt, ir_utt, _ = interview.get_qdict_from_qid(q_id)
        guest_utts.append(ie_utt)
        host_utts.append(ir_utt)
    df_for_inspection = pd.DataFrame(zip(*[pc_ids, guest_utts, host_utts, is_paraphrase, comments]), columns=columns)
    df_for_inspection.to_csv("../output/100_PC_IDs_for_inspection.tsv", sep='\t', index=False)


def get_pc_paraphrase_statistics(id_dict: Dict, sampled_ids: Set[int], print_ex: bool = False):
    """

    :param id_dict: comes from a evaluate_qsurvey.calc_pc_eval_lead call
    :param sampled_paraphrase_ids: expects a set
    :return:
    """
    result_dict = {}
    sampled_paraphrase_ids = set(sampled_ids & set(id_dict["Paraphrases"]))
    result_dict[AnnotationLabels.paraphrase] = sampled_paraphrase_ids

    # get nbr per category
    #   not clear
    unclear_paraphrases = sampled_paraphrase_ids - set(id_dict["Clear Paraphrases"])
    result_dict[AnnotationLabels.difficult] = unclear_paraphrases
    clear_paraphrases = sampled_paraphrase_ids & set(id_dict["Clear Paraphrases"])
    repetition = sampled_paraphrase_ids & set(id_dict[AnnotationLabels.repetition])
    result_dict[AnnotationLabels.repetition] = repetition
    universal = sampled_paraphrase_ids & set(id_dict[AnnotationLabels.universal])
    result_dict[AnnotationLabels.universal] = universal
    non_universal = sampled_paraphrase_ids - set(id_dict[AnnotationLabels.universal])
    result_dict["Context-Dependent Paraphrase"] = non_universal
    perspective = sampled_paraphrase_ids & set(id_dict[AnnotationLabels.perspective_shift])
    result_dict[AnnotationLabels.perspective_shift] = perspective
    high_sim = sampled_paraphrase_ids & set(id_dict[AnnotationLabels.high_lex_sim])
    result_dict[AnnotationLabels.high_lex_sim] = high_sim
    not_high_sim = sampled_paraphrase_ids - set(id_dict[AnnotationLabels.high_lex_sim])
    pragmatic = sampled_paraphrase_ids & set(id_dict[AnnotationLabels.pragmatic])
    result_dict[AnnotationLabels.pragmatic] = pragmatic
    directional = sampled_paraphrase_ids & set(id_dict[AnnotationLabels.directional])
    result_dict[AnnotationLabels.directional] = directional

    if print_ex:
        print(sampled_paraphrase_ids)
        print(f"Number of Paraphrases: {len(sampled_paraphrase_ids)}")
        print(f'Unclear paraphrases: {len(unclear_paraphrases)}')
        print(f'Clear paraphrases: {len(clear_paraphrases)}')
        print(f'Repetition Paraphrases: {len(repetition)}')
        print(f'Universal Paraphrases: {len(universal)}')
        print(f'Perspective Shift Paraphrases: {len(perspective)}')
        print(
            f'High Similarity Paraphrases: {len(high_sim)}')
        print(f'Not High Similarity Paraphrases: '
              f'{len(not_high_sim)}')
        print(f'Pragmatic Paraphrases: {len(pragmatic)}')
        print(f'Directional Paraphrases: {len(directional)}')

        interview = MediaSumProcessor()
        print_examples(unclear_paraphrases, interview)
        print_examples(clear_paraphrases, interview)
        print_examples(repetition, interview)
        print_examples(universal, interview)
        print_examples(perspective, interview)
        print_examples(high_sim, interview)
        print_examples(not_high_sim, interview)
        print_examples(pragmatic, interview)
        print_examples(directional, interview)

    return result_dict


def print_examples(q_ids, interview):
    q_ids = random.sample(q_ids, 2)
    for q_id in q_ids:
        ie_utt, ir_utt, summary = interview.get_qdict_from_qid(q_id)
        print(f"Q-ID: {q_id}")
        print(f"Guest:\n\t{ie_utt}")
        print(f"Host:\n\t{ir_utt}")
        print("-----------")


def get_pc_non_paraphrase_statistics(id_dict, sampled_ids, print_exs: bool = False):
    result_dict = {}
    sampled_non_paraphrase_ids = set(sampled_ids & set(id_dict[AnnotationLabels.non_paraphrase]))
    result_dict[AnnotationLabels.non_paraphrase] = sampled_non_paraphrase_ids
    print(sampled_non_paraphrase_ids)

    # get nbr per category
    #   not clear
    missing_context = sampled_non_paraphrase_ids & set(id_dict[AnnotationLabels.context])
    result_dict[AnnotationLabels.context] = missing_context
    high_lex = sampled_non_paraphrase_ids & set(
        id_dict[AnnotationLabels.high_lex_sim + " " + AnnotationLabels.non_paraphrase])
    result_dict[AnnotationLabels.high_lex_sim] = high_lex
    partial = sampled_non_paraphrase_ids & set(id_dict[AnnotationLabels.partial])
    result_dict[AnnotationLabels.partial] = partial
    unrelated = sampled_non_paraphrase_ids & set(id_dict[AnnotationLabels.unrelated])
    result_dict[AnnotationLabels.unrelated] = unrelated
    more_difficult = sampled_non_paraphrase_ids - set(id_dict[AnnotationLabels.unrelated])
    result_dict["More Difficult"] = more_difficult
    conclusion = sampled_non_paraphrase_ids & set(id_dict[AnnotationLabels.conclusion])
    result_dict[AnnotationLabels.conclusion] = conclusion

    related = sampled_non_paraphrase_ids & set(id_dict[AnnotationLabels.related])
    result_dict[AnnotationLabels.related] = related


    if print_exs:
        print(f"Number of Non-Paraphrases: {len(sampled_non_paraphrase_ids)}")
        print(f'Missing context: '
              f'{len(missing_context)}')
        print(f'High Lexical Similarity:'
              f' {len(high_lex)}')
        print(f'Partial: {len(partial)}')
        print(f'UNRELATED: {len(unrelated)}')
        print(f'CONCLUSION: '
              f'{len(conclusion)}')
        print(
            f'RELATED: {len(related)}')
        interview = MediaSumProcessor()
        print_examples(missing_context, interview)
        print_examples(high_lex, interview)
        print_examples(partial, interview)
        print_examples(unrelated, interview)
        print_examples(conclusion, interview)
        print_examples(related, interview)

    return result_dict


def sample_pcs(result_dict, paraphrase=True):
    combinations, nbr_samples, omega_ids, sets_by_position = set_up_ids(result_dict, paraphrase=paraphrase)

    # sample as described above
    sampled_pc_ids = set()
    for sample_n, from_combi in zip(nbr_samples, combinations):
        sampled_ids = sample_from_combination(from_combi, sample_n, omega_ids.copy(), sets_by_position)
        sampled_pc_ids = sampled_pc_ids | sampled_ids
    return sampled_pc_ids


def set_up_ids(result_dict, paraphrase=True):
    if paraphrase:
        omega_set = get_all_paraphrase_ids(result_dict)
        sets_by_position = get_paraprhase_id_sets(result_dict)
        nbr_samples = NBR_PER_COMBINATION_PARAPHRASE
        combinations = COMBINATIONS_PARAPRHASE
    else:
        omega_set = set(result_dict[AnnotationLabels.non_paraphrase])
        sets_by_position = get_non_paraprhase_id_sets(result_dict)
        nbr_samples = NBR_PER_COMBINATION_NON_PARAPHRASE
        combinations = COMBINATIONS_NON_PARAPRHASE

    assert (sum(nbr_samples) == 50)
    assert (len(nbr_samples) == len(combinations))
    return combinations, nbr_samples, omega_set, sets_by_position


def get_all_paraphrase_ids(result_dict):
    context_ids = result_dict[AnnotationLabels.context]
    paraphrase_ids = set(result_dict["Paraphrases"]) - set(context_ids)
    return paraphrase_ids


def get_paraprhase_id_sets(result_dict):
    #   ['rep', 'hlex', 'uni', 'clear', 'prag', 'per', 'dir']
    context_ids = result_dict[AnnotationLabels.context]
    repetition_ids = set(result_dict[AnnotationLabels.repetition]) - set(context_ids)
    high_lex_ids = set(result_dict[AnnotationLabels.high_lex_sim]) - set(context_ids)
    universal_ids = set(result_dict[AnnotationLabels.universal]) - set(context_ids)
    clear_ids = set(result_dict["Clear Paraphrases"]) - set(context_ids)
    pragmatic_ids = set(result_dict[AnnotationLabels.pragmatic]) - set(context_ids)
    perspective_ids = set(result_dict[AnnotationLabels.perspective_shift]) - set(context_ids)
    directional_ids = set(result_dict[AnnotationLabels.directional]) - set(context_ids)
    sets_by_position = [repetition_ids, high_lex_ids, universal_ids, clear_ids,
                        pragmatic_ids, perspective_ids, directional_ids]
    return sets_by_position


def get_non_paraprhase_id_sets(result_dict):
    #   ['hlex', 'partial', 'conc', 'context', 'unrel', 'rel']
    context_ids = set(result_dict[AnnotationLabels.context])
    assert (len(context_ids) > 0)
    high_lex_nonp_ids = set(result_dict[AnnotationLabels.high_lex_sim + " " + AnnotationLabels.non_paraphrase])
    assert (len(high_lex_nonp_ids) > 0)
    partial_ids = set(result_dict[AnnotationLabels.partial])
    assert (len(partial_ids) > 0)
    conclusion_ids = set(result_dict[AnnotationLabels.conclusion])
    assert (len(conclusion_ids) > 0)
    unrelated_ids = set(result_dict[AnnotationLabels.unrelated])
    assert (len(unrelated_ids) > 0)
    related_ids = set(result_dict[AnnotationLabels.related])
    assert (len(related_ids) > 0)
    sets_by_position = [high_lex_nonp_ids, partial_ids, conclusion_ids, context_ids,
                        unrelated_ids, related_ids]
    return sets_by_position


def sample_from_combination(combinations: List[List[int]], nbr_sample: int,
                            paraphrase_ids: Set[str], sets_by_position: List[Set[str]]):
    """

    :param combinations: combinations to sample from based on sets_by_position,
        several combinations together represent "unions" (|),
        one combination represents whether the ids are included (& = intersection) or excluded (-)
    :param nbr_sample:
    :param paraphrase_ids:
    :param sets_by_position:
    :return:
    """
    combi_sets = set()
    for combi in combinations:
        combi_set = paraphrase_ids.copy()
        for cur_id_set, is_included in zip(sets_by_position, combi):
            if is_included == 1:
                combi_set = combi_set & cur_id_set
            else:
                combi_set -= cur_id_set
        combi_sets = combi_sets | combi_set
    sampled_ids = random.sample(combi_sets, nbr_sample)
    return set(sampled_ids)


if __name__ == "__main__":
    main()
