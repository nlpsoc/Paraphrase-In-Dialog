"""
    test interactions with prolific api
"""
from unittest import TestCase

from paraphrase.set_id_consts import PROLIFIC_STUDY, PROLFIC_ANN, PROLFIC_PROJECT
from paraphrase.utility.prolific_api import send_prolific_request, create_training_survey, create_annotation_survey, \
    update_allowlist_survey, is_an_annotator_active, stop_study, print_pph_project


class Test(TestCase):

    def setUp(self) -> None:
        self.minimal_study = {"name": "Test",
                              "description": "test",
                              "external_study_url": "https://google.com",
                              "reward": 100,
                              "total_available_places": 500,
                              "prolific_id_option": "question",
                              "completion_option": "url",
                              "completion_codes": [
                                  {
                                      "code": "ABC123",
                                      "code_type": "COMPLETED",
                                      "actions": [
                                          {
                                              "action": "MANUALLY_REVIEW"
                                          }
                                      ]
                                  }
                              ],
                              "device_compatibility": [
                                  "mobile",
                                  "desktop",
                                  "tablet"
                              ],
                              "peripheral_requirements": [],
                              "estimated_completion_time": 1,
                              "filters": [],
                              "project": PROLFIC_PROJECT
                              }
        self.get_study_endpoint = {
            "state": "COMPLETED"
        }

    def test_api_endpoint(self):
        # https://docs.prolific.co/docs/api-docs/public/#tag/Users/operation/GetUser
        send_prolific_request(url="https://api.prolific.co/api/v1/users/me/", method="GET")
        # https://docs.prolific.co/docs/api-docs/public/#tag/Studies/operation/GetStudies
        send_prolific_request(data=self.get_study_endpoint, url="https://api.prolific.co/api/v1/studies/", method="GET")

    def test_get_allfilters(self):
        response = send_prolific_request(url="https://api.prolific.com/api/v1/filters/", method="GET")
        print(response)

    def test_create_minimal_survey(self):
        send_prolific_request(data=self.minimal_study, method="POST")

    def test_get_survey(self):
        send_prolific_request(url="https://api.prolific.co/api/v1/studies/64edeec2180835cf25ef53bf/", method="GET")

    def test_create_training_survey(self):
        create_training_survey()

    def test_get_eligibility_requirements(self):
        send_prolific_request(url="https://api.prolific.co/api/v1/eligibility-requirements/", method="GET")
        send_prolific_request(url="https://api.prolific.co/api/v1/filters/", method="GET")

    def test_create_annotation_survey(self):
        create_annotation_survey("https://survey.uu.nl/jfe/form/SV_b8vANBXWdb1Ybcy",
                                 allowlist=[])

    def test_update_allowlist_survey(self):
        update_allowlist_survey(PROLFIC_ANN)
        # self.fail()

    def test_is_an_annotator_active(self):
        is_an_annotator_active(PROLFIC_ANN)

    def test_stop_study(self):
        if not is_an_annotator_active(PROLIFIC_STUDY):
            stop_study(PROLIFIC_STUDY)

    def test_get_project_studies(self):
        print_pph_project()