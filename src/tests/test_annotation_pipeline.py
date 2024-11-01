"""
    testing DB from annotation pipeline
"""
from unittest import TestCase
import paraphrase.utility.annotation_pipeline as annotation_pipeline


class Test(TestCase):
    def test_print_last_annotation_statistics(self):
        memory_tsv = "../../result/Annotations/Paraphrase Annotations/DB/Qualtrics_prolific_ids__19.tsv"
        df = annotation_pipeline._read_db_from_path(memory_tsv)
        annotation_pipeline.print_last_annotation_statistics(df)
