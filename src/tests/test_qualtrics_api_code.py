"""
    testing qualtrics API
"""
from unittest import TestCase
import paraphrase.utility.qualtrics_api as qac
from paraphrase.set_id_consts import QUALTRICS_API_AUTH_TOKEN, QUALTRICS_SURVEY
from paraphrase.utility.qualtrics_survey import download_and_close_survey


class Test(TestCase):
    def test_export_survey(self):
        filename = "test.tsv"
        file = qac.dowload_survey(QUALTRICS_SURVEY,
                                  QUALTRICS_API_AUTH_TOKEN,
                                 "fra1", "tsv")

    def test_export_pc_survey(self):
        download_and_close_survey(QUALTRICS_SURVEY)

