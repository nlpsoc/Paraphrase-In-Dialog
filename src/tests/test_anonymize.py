import os
from unittest import TestCase
import pandas as pd
import paraphrase.anonymize as anonymize
from paraphrase.utility.annotation_df import QualtricsColumns, AnnotationColumns
from paraphrase.set_id_consts import LEAD


class Test(TestCase):
    def test_anonymize_qualtrics_file(self):
        # ANONYMIZE QUALTRICS ANNOTATIONS
        # delete translation file if it exists
        test_translation_file = "fixtures/annotations/Dummy_translations.tsv"
        try:
            os.remove(test_translation_file)
        except FileNotFoundError:
            pass
        anonymize.anonymize_file("fixtures/annotations/Dummy_Qualtrics-Annotations.tsv", test_translation_file)
        df = pd.read_csv("fixtures/annotations/Dummy_Qualtrics-Annotations.tsv", sep="\t", encoding="utf-16")
        an_df = pd.read_csv("fixtures/annotations/ANON_Dummy_Qualtrics-Annotations.tsv", sep="\t", encoding="utf-16")
        translated_df = pd.read_csv(test_translation_file, sep="\t")

        # check that all but the ID columns are the same in the dataframes
        self.assertTrue(
            df.drop(columns=QualtricsColumns.Author_ID).equals(an_df.drop(columns=QualtricsColumns.Author_ID)))
        # check the first two lines in Q-ID are the same (they do not include IDs)
        self.assertTrue(df[QualtricsColumns.Author_ID].iloc[:2].equals(an_df[QualtricsColumns.Author_ID].iloc[:2]))
        # check that LEAD does not occur in the anonymized data
        self.assertFalse(any(an_df[QualtricsColumns.Author_ID].str.contains(LEAD)))
        # assert that ANON_LEAD is in the anonymized data where LEAD is in the original data

        self.assertTrue(all(an_df[
                                df[QualtricsColumns.Author_ID].str.contains(LEAD)
                            ][QualtricsColumns.Author_ID].str.contains(anonymize.ANON_LEAD)))

        # check that the translations translate LEAD -> ANON_LEAD
        self.assertEqual(translated_df[anonymize.IDTranslationColumns.Anon_ID].iloc[0], anonymize.ANON_LEAD)

    def test_anonymize_hl_file(self):
        # ANONYMIZE HIGHLIGHT ANNOTATIONS
        # delete translation file if it exists
        test_translation_file = "fixtures/annotations/Dummy_translations.tsv"
        try:
            os.remove(test_translation_file)
        except FileNotFoundError:
            pass
        anonymize.anonymize_file("fixtures/annotations/Dummy_Transformed-Results.tsv", test_translation_file)
        df = pd.read_csv("fixtures/annotations/Dummy_Transformed-Results.tsv", sep="\t", encoding="utf-16")
        an_df = pd.read_csv("fixtures/annotations/ANON_Dummy_Transformed-Results.tsv", sep="\t", encoding="utf-16")
        translated_df = pd.read_csv(test_translation_file, sep="\t")

        # check that all but the ID columns are the same in the dataframes
        self.assertTrue(
            df.drop(columns=AnnotationColumns.Annotator_ID).equals(an_df.drop(columns=AnnotationColumns.Annotator_ID)))

    def test_anonymize_db_file(self):
        db_path = "fixtures/annotations/dummy_db.tsv"
        # delete translation file if it exists
        test_translation_file = "fixtures/annotations/Dummy_translations.tsv"
        try:
            os.remove(test_translation_file)
        except FileNotFoundError:
            pass
        anonymize.anonymize_file(db_path, test_translation_file)

    def test_anonymize_folder(self):
        test_translation_file = "fixtures/annotations/Dummy_translations.tsv"
        try:
            os.remove(test_translation_file)
        except FileNotFoundError:
            pass
        anonymize.anonymize_folder("fixtures/annotations", test_translation_file)


