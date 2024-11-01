"""
    Sample for annotation for RAND and PARA pair
"""

import pandas as pd

from paraphrase.utility.PC_utility import get_ambiguous_ids, \
    get_non_paraphrase_ids, get_non_repetitions_singlepc, \
    create_tsv_ids_to_check_singlepc, get_diff_nonpara_ids_singlepc, get_related_nonpara_ids_singlepc, \
    evaluate_singlepc_annotations
from paraphrase.lead_pc import load_pc_second_batch
from paraphrase.utility.project_functions import SEED
import random
from paraphrase import interview_data
from paraphrase.utility.annotation_df import (AnnotationColumns)


REANNOTATE_CTXT = False


def main(reannotate_ambig=False, reannotate_cntxt=REANNOTATE_CTXT):
    df_sorted, _ = load_pc_second_batch()
    evaluate_singlepc_annotations(df_sorted, reannotate_ambig, reannotate_cntxt)

    ambiguous_ids = get_ambiguous_ids(df_sorted)
    repetition_ids, nr_paraphrase_ids = get_non_repetitions_singlepc(df_sorted)
    non_paraphrase_ids = get_non_paraphrase_ids(df_sorted)
    diff_non_paraphrase_ids = get_diff_nonpara_ids_singlepc(df_sorted)
    rel_n_para_ids = get_related_nonpara_ids_singlepc(df_sorted)

    # sample 100 ids from everything but errors/context
    sample_pool = ambiguous_ids + repetition_ids + nr_paraphrase_ids + non_paraphrase_ids
    assert (len(set(sample_pool)) == len(sample_pool))
    random.seed(SEED)
    sample_100 = random.sample(sample_pool, 100)
    # pd_sample = pd.DataFrame({AnnotationColumns.Question_ID: sample_100})
    # pd_sample.to_csv("Random-Sample-100.tsv", sep="\t", index=False)

    s_ambig = set(sample_100) & set(ambiguous_ids)
    s_rep = set(sample_100) & set(repetition_ids)
    s_nr_para = set(sample_100) & set(nr_paraphrase_ids)
    s_np = set(sample_100) & set(non_paraphrase_ids)
    s_diff_np = set(sample_100) & set(diff_non_paraphrase_ids)
    s_rel_np = set(sample_100) & set(rel_n_para_ids)

    print(f"Out of a sample pool of {len(sample_pool)}, sampled {len(sample_100)} unique IDs, with")
    print(f"\t Sampled {len(s_ambig)} ambiguous, initial ratio was {len(ambiguous_ids) / len(sample_pool)}")
    print(f"\t Sampled {len(s_rep)} repetitions, initial ratio was {len(repetition_ids) / len(sample_pool)}")
    print(f"\t Sampled {len(s_nr_para)} non-repetition paraphrases, "
          f"initial ratio was {len(nr_paraphrase_ids) / len(sample_pool)}")
    print(f"\t Sampled {len(s_np)} non-paraphrases, initial ratio was {len(non_paraphrase_ids) / len(sample_pool)}")
    print(f"\t\t Sampled {len(s_diff_np)} difficult non-paraphrases, "
          f"initial ratio was {len(diff_non_paraphrase_ids) / len(sample_pool)}")
    print(f"\t\t Sampled {len(s_rel_np)} topically related non-paraphrases, "
          f"initial ratio was {len(rel_n_para_ids) / len(sample_pool)}")

    # sample 400 remaining IDs
    nbr_oth_np = max(80, len(s_np) - len(s_diff_np) - len(s_rel_np))
    nbr_dif_np = 20 - (nbr_oth_np - 80) - len(s_ambig)
    nbr_rel_np = 50
    nbr_repetitions = 50
    nbr_nr_paraphrases = 300

    oth_np_ids = random.sample([oth_np for oth_np in non_paraphrase_ids
                                if oth_np not in sample_100 and
                                oth_np not in diff_non_paraphrase_ids and
                                oth_np not in rel_n_para_ids],
                               nbr_oth_np - (len(s_np) - len(s_diff_np) - len(s_rel_np)))
    dif_np_ids = random.sample([dif_np for dif_np in diff_non_paraphrase_ids if dif_np not in sample_100],
                               nbr_dif_np - len(s_diff_np))
    rel_np_ids = random.sample([rel_np for rel_np in rel_n_para_ids if rel_np not in sample_100],
                               nbr_rel_np - len(s_rel_np))
    rep_ids = random.sample([rep for rep in repetition_ids if rep not in sample_100],
                            nbr_repetitions - len(s_rep))
    nr_p_ids = random.sample([para for para in nr_paraphrase_ids if para not in sample_100],
                             nbr_nr_paraphrases - len(s_nr_para))

    sample_400 = oth_np_ids + dif_np_ids + rel_np_ids + rep_ids + nr_p_ids
    assert (len(set(sample_400)) == 400)
    print(f"Further sampled {len(set(oth_np_ids + dif_np_ids + rel_np_ids + rep_ids + nr_p_ids))} unique IDs")
    print(f"\t Sampled {len(rep_ids)} repetitions, now totaling {len(set(rep_ids) | s_rep)}")
    print(f"\t Sampled {len(nr_p_ids)} non-repetition paraphrases, "
          f"now totaling {len(set(nr_p_ids) | s_nr_para)}")
    print(f"\t Sampled {len(oth_np_ids + dif_np_ids + rel_np_ids)} non-paraphrases, "
          f"now totaling {len(set(oth_np_ids + dif_np_ids + rel_np_ids) | s_np)}")
    print(f"\t\t Sampled {len(dif_np_ids)} difficult non-paraphrases, "
          f"now totaling {len(set(dif_np_ids) | s_diff_np)}")
    print(f"\t\t Sampled {len(rel_np_ids)} topically related non-paraphrases, "
          f"now totaling {len(set(rel_np_ids) | s_rel_np)}")

    full_sample = sample_100 + sample_400
    is_random = [1] * 100 + [0] * 400
    pd_sample = pd.DataFrame({AnnotationColumns.Question_ID: full_sample, "Random": is_random})
    pd_sample = pd_sample.sample(frac=1).reset_index(drop=True)
    pd_sample.to_csv("tmp_Annotation-Sample-500.tsv", sep="\t", index=False)

    interview = interview_data.MediaSumProcessor()
    create_tsv_ids_to_check_singlepc(full_sample, df_sorted, interview)


if __name__ == "__main__":
    main()
