"""
    test generating qualtrics surveys (json files)
"""
from unittest import TestCase
import qualtrics_survey
from qualtrics_survey import upload_qsf
import json
from paraphrase.interview_data import MediaSumProcessor


class Test(TestCase):
    def setUp(self) -> None:
        self.s_id = "15"
        self.summary = "Author Khaled Hosseini has made a career out of writing about his native Afghanistan, and " \
                       "now he is using his literary fame to draw attention to the plight of refugees in Afghanistan." \
                       " He speaks with guest host Mary Louise Kelly about his visit there as a goodwill envoy for " \
                       "the United Nations High Commissioner for Refugees in Afghanistan."
        self.ie_utt = "So they also allow people to have - you know, there's a saying in Afghanistan that go hungry " \
                      "if you must but may you never go homeless. And that's why I've decided to focus on " \
                      "the issue of shelter."
        self.ir_utt = "I know you've been going there the past few years and trying to help move along this process." \
                      " Are you able to see progress? Are you able to see that things look different, better than " \
                      "they did a couple of years ago?"
        self.q_id = "NPR-4-2"  # These are made up but conform to the format used
        self.q_id2 = "CNN-5-2"  # These are made up but conform to the format used
        self.date = "2016-08-30"
        self.host_name = "Mary Louise Kelly"
        self.guest_name = "Khaled Hosseini"

        # self.s_id = "22"
        # self.summary = "Author Khaled Hosseini has made a career out of writing about his native Afghanistan, and " \
        #                "now he is using his literary fame to draw attention to the plight of refugees in Afghanistan." \
        #                " He speaks with guest host Mary Louise Kelly about his visit there as a goodwill envoy for " \
        #                "the United Nations High Commissioner for Refugees in Afghanistan."
        # self.ie_utt = "And to me, that is such a symbol of the resilience of these people. And it would be such a" \
        #               " tragedy if we abandon this country and don't continue this commitment to the Afghan people."
        # self.ir_utt = "I know that there was a large number of refugees who returned right after the 2001 U.S.-led " \
        #               "invasion of Afghanistan. Since then the numbers have slowed. But this year the rate of " \
        #               "refugees returning to Afghanistan has gone up dramatically again. Tell us why. What's going on?"

    def test_upload_empty_qsf(self):
        # Upload the empty version of the qsf survey created on qualtrics
        filename = "paraphrase/utility/fixtures/qsf/QSF-Empty.qsf"
        s_id = upload_qsf(filename)
        print(s_id)

    def test_2q_upload(self):
        """
            test if the template works
        :return:
        """
        filename = "paraphrase/utility/fixtures/qsf/test_Paraphrase.qsf"
        s_id = upload_qsf(filename)
        print(s_id)

    def test_paraphrase_2qsf(self):
        q_1 = qualtrics_survey.GuestHostPairQuestionInput(QID=self.q_id,
                                                          Guest_Utterance=self.ie_utt,
                                                          Host_Utterance=self.ir_utt,
                                                          Summary=self.summary,
                                                          Date=self.date,
                                                          Host_Name=self.host_name,
                                                          Guest_Name=self.guest_name
                                                          )
        q_2 = qualtrics_survey.GuestHostPairQuestionInput(QID=self.q_id2,
                                                          Guest_Utterance=self.ie_utt,
                                                          Host_Utterance=self.ir_utt,
                                                          Summary=self.summary,
                                                          Date=self.date,
                                                          Host_Name=self.host_name,
                                                          Guest_Name=self.guest_name
                                                          )
        survey_qsf = qualtrics_survey. \
            gen_paraphrase_survey([q_1, q_2], questions_per_annotator=1, survey_name="23-03-13_Paraphrase_Template")
        survey_str = json.dumps(survey_qsf, sort_keys=True)
        fixture_json = json.load(open("paraphrase/utility/fixtures/qsf/test_Paraphrase.qsf"))
        self.assertEqual(survey_str, json.dumps(fixture_json, sort_keys=True))

    def test_paraphrase_candidate_2qsf(self):
        q_1 = qualtrics_survey.GuestHostPairQuestionInput(QID=self.q_id,
                                                          Guest_Utterance=self.ie_utt,
                                                          Host_Utterance=self.ir_utt,
                                                          Summary=self.summary,
                                                          Date=self.date,
                                                          Host_Name=self.host_name,
                                                          Guest_Name=self.guest_name
                                                          )
        q_2 = qualtrics_survey.GuestHostPairQuestionInput(QID=self.q_id2,
                                                          Guest_Utterance=self.ie_utt,
                                                          Host_Utterance=self.ir_utt,
                                                          Summary=self.summary,
                                                          Date=self.date,
                                                          Host_Name=self.host_name,
                                                          Guest_Name=self.guest_name
                                                          )
        survey_qsf = qualtrics_survey. \
            gen_paraphrase_candidate_survey([q_1, q_2], survey_name="23-04-05_Paraphrase_Candidate_Template")
        survey_str = json.dumps(survey_qsf, sort_keys=True)
        fixture_json = json.load(open("paraphrase/utility/fixtures/qsf/test_Paraphrase-Candidate.qsf"))
        self.assertEqual(survey_str, json.dumps(fixture_json, sort_keys=True))



