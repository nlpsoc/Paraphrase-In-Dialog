"""
    You probably don't need this

    highest level qualtrics functions
"""
from datetime import datetime

from paraphrase.interview_data import MediaSumProcessor
from paraphrase.utility.qualtrics_survey import GuestHostPairQuestionInput, gen_paraphrase_candidate_survey, \
    gen_paraphrase_survey, gen_training_survey, save_qsf, upload_qsf, publish_survey
from preprocess_interviews import IDS_CSV


def create_survey(size_sample: int = 10, survey_name: str = "SURVEY_NAME",
                  from_id_list: bool = False,
                  id_list=None, out_qsf: str = f"../output/{datetime.now()}_survey.qsf", sample_path: str = IDS_CSV,
                  candidate_survey: bool = False, publish=True, interview=None, training=False,
                  test=False,
                  places=1, start_sample=None, seconds=30):
    if not training:
        if from_id_list and (id_list is None):
            raise ValueError("If from_id_list is True, id_list must be provided.")
        if interview is None:
            # load complete MediaSum data
            interview = MediaSumProcessor(uncut=True)
        # collect questions to upload
        questions = []

        if not from_id_list:
            print(f"Sampling {size_sample} questions starting from id {start_sample} ...")

            assert (start_sample >= 1)
            start_sample = start_sample - 1
            i = 0
            for q_id, ie_utt, ir_utt, summary in interview. \
                    iter_for_annotation(sample_path):

                i += 1
                if i <= start_sample:
                    continue

                current_i = interview.get_interview_from_id(q_id[:q_id.rindex("-")])
                date = current_i['date']
                host_name, guest_name = interview.get_host_and_guest_name(current_i['speaker'])

                question = GuestHostPairQuestionInput(QID=q_id, Guest_Utterance=ie_utt,
                                                      Host_Utterance=ir_utt, Summary=summary,
                                                      Date=date,
                                                      Host_Name=host_name, Guest_Name=guest_name)
                questions.append(question)

                if len(questions) >= size_sample:
                    break
        else:

            print(f"Creating survey {survey_name} of length {len(id_list)} from list of Question IDs:")
            print(id_list)
            # random.shuffle(ID_LIST)
            for q_id in id_list:
                current_i = interview.get_interview_from_id(q_id[:q_id.rindex("-")])
                date = current_i['date']
                host_name, guest_name = interview.get_host_and_guest_name(current_i['speaker'])
                q_dict = interview.get_qdict_from_qid(q_id)
                ie_utt, ir_utt, summary = q_dict["guest utterance"], q_dict["host utterance"], q_dict["summary"]
                question = GuestHostPairQuestionInput(QID=q_id, Guest_Utterance=ie_utt,
                                                      Host_Utterance=ir_utt, Summary=summary,
                                                      Date=date,
                                                      Host_Name=host_name, Guest_Name=guest_name)
                questions.append(question)

        if candidate_survey:
            # print(f"{questions_per_annotator} questions are equally sampled per annotator "
            #       f"out of a total of {len(questions)}.")
            print("Creating paraphrase candidate survey ....")
            survey_qsf = gen_paraphrase_candidate_survey(questions, survey_name)
        else:
            print("Creating paraphrase survey ....")
            survey_qsf = gen_paraphrase_survey(questions, survey_name, seconds_per_question=seconds)
        print([q.Question_ID for q in questions])
    else:
        survey_qsf = gen_training_survey(test=test, places=places)
        survey_name = survey_qsf["SurveyEntry"]["SurveyName"]

    save_qsf(survey_qsf, out_qsf)

    survey_id = upload_qsf(out_qsf, survey_name)
    if publish:
        url = publish_survey(survey_id)
        print(f"Survey published with URL {url}")

    return survey_id, survey_name
