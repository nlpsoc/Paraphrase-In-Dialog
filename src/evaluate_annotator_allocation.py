"""
    Perform evaluation of annotator allocation strategies
"""
import paraphrase.utility.stats as stats
from paraphrase.annotator_allocation import plot_nbr_annotator_plot, get_bootstrap_kRR, \
    get_es_bootstrap_kRR
from paraphrase.set_id_consts import PAID_ANNOTATORS_SORTED_BY_START_TIME
import paraphrase.utility.annotation_df as annotation_df
import statistics
from paraphrase.utility.annotation_df import AnnotationColumns
from paraphrase.annotation_data import ANNOTATIONS_5050_20_PATH


def main():
    n = 20
    highlight_df = annotation_df.merge_transform(ANNOTATIONS_5050_20_PATH)

    # calculate median
    times_sorted_start_date = highlight_df[highlight_df["Category"] == "Time"]
    times_sorted_start_date = times_sorted_start_date["Highlighted"].values.tolist()
    times_sorted_start_date = list(map(int, times_sorted_start_date))
    print(f"median: {statistics.median(times_sorted_start_date)} over {len(times_sorted_start_date)} annotators")
    print(f"mean: {statistics.mean(times_sorted_start_date)} +- {statistics.stdev(times_sorted_start_date)}")

    # kRR calculations
    #   get the distribution of valid completions for the questions
    sorted_valid = [ann_id for ann_id in PAID_ANNOTATORS_SORTED_BY_START_TIME
                    if ann_id in highlight_df[AnnotationColumns.Annotator_ID].unique()]
    stats.set_global_seed()
    bootstrap_dict = get_bootstrap_kRR(n, highlight_df, sorted_valid)
    es_dict = get_es_bootstrap_kRR(n, highlight_df, sorted_valid,
                                   bootstrap_matrix=None)  # bootstrap_dict["bootstrap matrix"]

    plot_nbr_annotator_plot(bootstrap_dict, es_dict,
                            es_abs_label='bs kRR (absolute)',
                            es_dict_ekey='bs kRR (entropy', es_dict_ekey_avg_anns='means entropy',
                            es_dict_key="early stopping absolute cohen", es_dict_key_avg_ann="stop id means",
                            vanilla_key="bootstrap kRR", vanilla_label='bs kRR (ALL)',
                            xlabel='Average # annotators', ylabel='Bootstrapped kRR Cohen')
    plot_nbr_annotator_plot(bootstrap_dict, es_dict,
                            es_abs_label='absolute',
                            es_dict_ekey='accuracy (entropy', es_dict_ekey_avg_anns='means entropy',
                            es_dict_key="early stopping absolute accuracy", es_dict_key_avg_ann="stop id means",
                            vanilla_key="accuracies", vanilla_label='ALL annotators',
                            xlabel='Average # annotators', ylabel='Accuracy w.r.t. majority vote',
                            ylim=(0.80, 1.0), horizontal=0.9, vertical=8)


if __name__ == '__main__':
    main()
