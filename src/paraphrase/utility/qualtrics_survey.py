"""
Qualtrics custom API wrapper for automatically generating the surveys


    https://www-annualreviews-org.proxy.library.uu.nl/doi/abs/10.1146/annurev-statistics-042720-125902
    https://github.com/60decibels/pytrics/blob/91ab8107433be0391f0fad06fa702eaa93f6af21/pytrics/qualtrics_api/client.py#L251
"""
import json
import logging
import os
import sys
from datetime import datetime
from typing import List
import warnings
import copy

import requests

import paraphrase.utility.qualtrics_api as qac
from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.utility.qualtrics_api import QUALTRICS_API_AUTH_TOKEN_NAME
from paraphrase.set_id_consts import QUALTRICS_API_AUTH_TOKEN

R2_Identifier = "R2"
ContentReproduction_Identifier = "R"
Interpretation_Identifier = "I"
ElaborationStatement_Identifier = "EI"
Question_Identifier = "Q"
Choice_Identifier = "C"
Table_Identifier = "T"
QComment_Identifier = "CL"
Q_BLOCK_IDENTIFIERS = ["0", R2_Identifier, "1", "2", ContentReproduction_Identifier, Interpretation_Identifier,
                       ElaborationStatement_Identifier, Question_Identifier, Choice_Identifier, Table_Identifier,
                       QComment_Identifier]

QUALTRICS_API_BASE_URL = "https://fra1.qualtrics.com/API/v3"  # other datacenter: fra1 yul1 "https://yul1.qualtrics.com/API/v3
# /surveys"
QUALTRICS_API_PATH_SURVEYS = "surveys"
QUALTRICS_API_PATH_SURVEY_DEFINITIONS = "survey-definitions"


class ContentReproductionIdentifiers:
    R2_Identifier = "R2"
    Repeat_Identifier = "R"
    Reference_Identifier = "RE"
    Paraphrase_Identifier = "P"


class GuestHostPairQuestionInput:
    """
        class including attributes necessary for generating surveys
    """

    def __init__(self, QID: str, Guest_Utterance: str, Host_Utterance: str, Summary: str, Date: str,
                 Host_Name: str = "Host", Guest_Name: str = "Guest"):
        self.Question_ID = QID  # 'QID'
        self.Guest_Utterance = Guest_Utterance  # 'Guest Utterance'
        self.Host_Utterance = Host_Utterance  # 'Host Utterance'
        self.Summary = Summary  # 'Summary'
        self.Date = Date
        self.Host_Name = Host_Name
        self.Guest_Name = Guest_Name


def _build_headers(method, auth_token: str = QUALTRICS_API_AUTH_TOKEN,
                   auth_token_name: str = QUALTRICS_API_AUTH_TOKEN_NAME, appljson=False):
    """
    Constructs a dictionary which will be used as the request headers for all API interactions
    """
    if method not in ['GET', 'DELETE', 'POST', 'PUT', 'PATCH']:
        raise Exception('Client only supports GET, DELETE, POST, PATCH and PUT methods.')

    headers = {
        auth_token_name: auth_token,
    }

    if appljson:
        if method in ['POST', 'PUT', 'PATCH']:
            headers["content-type"] = "application/json"

    return headers


def get_survey_results(survey_id: str, base_api_url: str = QUALTRICS_API_BASE_URL):
    url = f"{base_api_url}/surveys/{survey_id}"
    headers = _build_headers('PUT', appljson=True)
    response = requests.put(url, headers=headers)
    response.raise_for_status()


def publish_survey(survey_id: str):
    _change_survey_status(survey_id, is_active=True)
    return f"https://survey.uu.nl/jfe/form/{survey_id}"


def close_survey(survey_id: str):
    _change_survey_status(survey_id, is_active=False)
    return


def _change_survey_status(survey_id, is_active, base_api_url: str = QUALTRICS_API_BASE_URL):
    url = f"{base_api_url}/surveys/{survey_id}"
    headers = _build_headers('PUT', appljson=True)
    response = requests.put(
        url,
        json={
            "isActive": is_active
        },
        headers=headers
    )
    response.raise_for_status()


def _delete_qualtrics_survey(survey_id, base_api_url: str = QUALTRICS_API_BASE_URL):
    url = f"{base_api_url}/surveys/{survey_id}"
    headers = _build_headers('DELETE', appljson=True)
    response = requests.delete(
        url,
        headers=headers
    )
    response.raise_for_status()


def create_survey_from_qsf(qsf_path: str, survey_name: str, language_code='EN', project_category='CORE',
                           base_api_url: str = QUALTRICS_API_BASE_URL,
                           qualtrics_api_path_surveys: str = QUALTRICS_API_PATH_SURVEYS) -> str:
    """
        see also: https://api.qualtrics.com/7c9e686d364ac-import-survey
    :param qsf_path: path to qsf format (e.g., exported from a qualtrics survey)
    :param survey_name:
    :param language_code:
    :param project_category:
    :param base_api_url:
    :param qualtrics_api_path_surveys:
    :return: id of the survey created on Qualitrcs
    """
    url = '{0}/{1}'.format(base_api_url, qualtrics_api_path_surveys)
    headers = _build_headers('POST')
    files = {
        "file": (qsf_path, open(qsf_path, 'rb'), 'application/vnd.qualtrics.survey.qsf')
    }
    data = {"name": survey_name}

    response = requests.post(
        url,
        files=files,
        data=data,
        headers=headers
    )

    print(response.json())
    response.raise_for_status()

    result = response.json()

    survey_id = result['result']['id']
    logging.info('survey id is {}'.format(survey_id))

    return survey_id


def upload_qsf(filepath: str, survey_name="QSF-TEST"):
    return create_survey_from_qsf(filepath, survey_name)


def gen_training_survey(test: bool = False, places: int = 1):
    # get date
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%y-%m-%d")
    # load blueprint
    blueprint_qsf = get_dir_to_src() + "/utility/qualtrics_templates/YY-MM-DD_Training_Paraphrase-Highlighting.json"
    blueprint_qsf = _get_qsf_from_path(blueprint_qsf)
    blueprint_qsf["SurveyEntry"]["SurveyName"] = (("TEST_" if test else "") +
                                                  blueprint_qsf["SurveyEntry"]["SurveyName"].format(formatted_date) +
                                                  f"_{places}")

    return blueprint_qsf


def gen_paraphrase_survey(question_dicts: List[GuestHostPairQuestionInput], survey_name: str,
                          questions_per_annotator: int = 15, seconds_per_question=50, debug=False):
    """
        using "/utility/qualtrics_templates/Paraphrase-Template.json" and
        "/utility/qualtrics_templates/Paraphrase_dynamic-elements.json"
            generates the paraphrase highlighting survey

    :param questions_per_annotator:
    :param seconds_per_question:
    :param question_dicts:
    :param survey_name:
    :param debug:
    :return:
    """
    seconds_per_question = seconds_per_question
    nbr_questions = len(question_dicts)
    minutes = round(nbr_questions * seconds_per_question / 60)
    scaffolding_path = get_dir_to_src() + "/utility/qualtrics_templates/Paraphrase-Template.json"
    dynamicelm_path = get_dir_to_src() + "/utility/qualtrics_templates/Paraphrase_dynamic-elements.json"

    scaffolding_qsf = build_hl_survey(question_dicts, scaffolding_path, dynamicelm_path, minutes, seconds_per_question,
                                      survey_name, questions_per_annotator=questions_per_annotator, debug=debug)

    return scaffolding_qsf


def gen_paraphrase_candidate_survey(question_dicts: List[GuestHostPairQuestionInput], survey_name: str,
                                    seconds_per_question=30, debug=False):
    """
        using "/utility/qualtrics_templates/Paraphrase-Candidate-Template.json" and "/utility/qualtrics_templates/Paraphrase_dynamic-elements.json"
            generates the paraphrase highlighting survey

    :param seconds_per_question:
    :param question_dicts:
    :param survey_name:
    :param debug:
    :return:
    """
    seconds_per_question = seconds_per_question
    # nbr_questions = nbr_questions_per_annotator
    minutes = round(len(question_dicts) * seconds_per_question / 60)
    scaffolding_path = get_dir_to_src() + "/utility/qualtrics_templates/Paraphrase-Candidate-Template.json"
    dynamicelm_path = get_dir_to_src() + "/utility/qualtrics_templates/Paraphrase-Candidate_dynamic-elements.json"

    scaffolding_qsf = build_hl_survey(question_dicts, scaffolding_path, dynamicelm_path, minutes, seconds_per_question,
                                      survey_name, debug=debug, paraphrase_candidate=True)

    return scaffolding_qsf


def build_hl_survey(question_dicts, scaffolding_path, dynamicelm_path, minutes, seconds_per_question, survey_name,
                    questions_per_annotator=15, debug=False, paraphrase_candidate=False):
    """
        build a highlighting survey
    :param questions_per_annotator:
    :param paraphrase_candidate:
    :param question_dicts:
    :param scaffolding_path:
    :param dynamicelm_path:
    :param minutes:
    :param seconds_per_question:
    :param survey_name:
    :param debug:
    :return:
    """

    # build survey 'scaffolding' which is only instantiated once
    scaffolding_qsf = _get_qsf_from_path(scaffolding_path)
    scaffolding_qsf["SurveyEntry"]["SurveyName"] = survey_name
    scaffolding_qsf["SurveyElements"][-1]["Payload"]["QuestionText"] = scaffolding_qsf["SurveyElements"][-1]["Payload"][
        "QuestionText"] \
        .format(seconds_per_question, len(question_dicts), minutes)
    # Fixed block ID for dynamic block construction depending on type of survey (has to do with size)
    if paraphrase_candidate:
        block_nbr = 11
        flow_nbr = 25
        dynamic_q_pos = 2
        dynamic_flow_pos = -6  # -4
    else:
        block_nbr = 10
        flow_nbr = 27
        dynamic_q_pos = 2
        dynamic_flow_pos = -7

    # build dynamic elements that extend depending on the question dicts
    dynamic_txt = _get_qsf_from_path(dynamicelm_path)
    nbr_questions = len(question_dicts)

    for i, q in enumerate(question_dicts):

        # For each Question:
        #   build current Question Block
        cur_block = copy.deepcopy(dynamic_txt["SurveyElements"][0]["Payload"][str(block_nbr)])
        cur_block["Description"] = cur_block["Description"].format(q.Question_ID)
        cur_block["ID"] = cur_block["ID"].format(q.Question_ID)
        for qid_elem in cur_block["BlockElements"]:
            if "QuestionID" in qid_elem:  # e.g., "QuestionID": "QID-{}-1", i.e., not a Page Break
                qid_elem["QuestionID"] = qid_elem["QuestionID"].format(q.Question_ID)

        #   and Flow
        # cur_flow = copy.deepcopy(dynamic_txt["SurveyElements"][1]["Payload"]["Flow"][0]["Flow"][0])
        cur_flow = copy.deepcopy(dynamic_txt["SurveyElements"][1]["Payload"]["Flow"][0])
        cur_flow["ID"] = cur_flow["ID"].format(q.Question_ID)
        cur_flow["FlowID"] = f"FL_{flow_nbr + i}"

        # if paraphrase_candidate:
        # Append to Scaffolding
        scaffolding_qsf["SurveyElements"][0]["Payload"][str(block_nbr + i)] = cur_block
        # scaffolding_qsf["SurveyElements"][1]["Payload"]["Flow"][6]["Flow"].append(cur_flow)
        if i < nbr_questions / 2:  # append before CC1
            scaffolding_qsf["SurveyElements"][1]["Payload"]["Flow"].insert(dynamic_flow_pos, cur_flow)
        elif paraphrase_candidate:  # append after AC2 AND CC2
            scaffolding_qsf["SurveyElements"][1]["Payload"]["Flow"].insert(dynamic_flow_pos + 2, cur_flow)
        else:  # append after AC1 only
            scaffolding_qsf["SurveyElements"][1]["Payload"]["Flow"].insert(dynamic_flow_pos + 1, cur_flow)
            # Append to Scaffolding
            # scaffolding_qsf["SurveyElements"][0]["Payload"][str(block_nbr + i)] = cur_block
            # scaffolding_qsf["SurveyElements"][1]["Payload"]["Flow"][6]["Flow"].append(cur_flow)
            # for qid_elem in cur_block["BlockElements"]:
            #     if "QuestionID" in qid_elem:  # e.g., "QuestionID": "QID-{}-1", i.e., not a Page Break
            #         qid_elem["QuestionID"] = qid_elem["QuestionID"].format(q.Question_ID)
            #     #   append each to scaffolding Survey Block 5 in 2nd to last place
            #     scaffolding_qsf["SurveyElements"][0]["Payload"]["5"]["BlockElements"].insert(-1, qid_elem)

        # Build question elements
        q_elements = copy.deepcopy(dynamic_txt["SurveyElements"][dynamic_q_pos:])
        for q_elem in q_elements:
            # add question ID
            q_elem["PrimaryAttribute"] = q_elem["PrimaryAttribute"].format(q.Question_ID)
            q_elem["Payload"]["QuestionID"] = q_elem["Payload"]["QuestionID"].format(q.Question_ID)
            q_elem["Payload"]["DataExportTag"] = q_elem["Payload"]["DataExportTag"].format(q.Question_ID)

            if not paraphrase_candidate:
                # Build for highlighting questions
                build_qelem_hl(q_elem, q, debug)
            else:
                build_qelem_candidate(q_elem, q)

            # insert into scaffolding
            scaffolding_qsf["SurveyElements"].append(q_elem)
    return scaffolding_qsf


def build_qelem_hl(q_elem, q, debug):
    """
        for a highlighting type questionaire, dynamoically set current element
    :param q_elem:
    :param q:
    :param debug:
    :return:
    """
    guest_identifier = f"Guest ({q.Guest_Name})"
    host_identifier = f"Host ({q.Host_Name})"
    if "summary" in q_elem["Payload"]["QuestionText"]:
        q_elem["Payload"]["QuestionText"] = q_elem["Payload"]["QuestionText"].format(q.Date, q.Summary)
    if "Guest: ..." in q_elem["SecondaryAttribute"]:
        q_elem["Payload"]["QuestionText"] = q_elem["Payload"]["QuestionText"].format(guest_identifier,
                                                                                     q.Guest_Utterance)
    if "Host: ..." in q_elem["SecondaryAttribute"]:
        q_elem["Payload"]["QuestionText"] = q_elem["Payload"]["QuestionText"].format(host_identifier,
                                                                                     q.Host_Utterance)
    #   add highlighting choices
    if "HL" in q_elem["Payload"]["QuestionType"]:
        if "R2" in q_elem["PrimaryAttribute"]:
            q_elem["Payload"]["QuestionText"] = q_elem["Payload"]["QuestionText"].format(guest_identifier)
            _update_hl_qsf(qsf_txt=q_elem, txt_to_highlight=q.Guest_Utterance, debug=debug)
        else:
            q_elem["Payload"]["QuestionText"] = q_elem["Payload"]["QuestionText"].format(host_identifier)
            _update_hl_qsf(qsf_txt=q_elem, txt_to_highlight=q.Host_Utterance, debug=debug)
            # if "-RE" or "-R" in q_elem["PrimaryAttribute"]:  # Validate that something is highlighted
            #     _add_validation(q.Question_ID, q_elem)
    #   set display logic
    if "InPageDisplayLogic" in q_elem["Payload"].keys():
        _set_display_logic(q.Question_ID, q_elem)


def build_qelem_candidate(q_elem, q):
    """
        for a paraphrase candidate type survey
    :param q_elem:
    :param q:
    :return:
    """
    # if "QID-{}-1" type question
    if "summary" in q_elem["Payload"]["QuestionText"]:
        q_elem["Payload"]["QuestionText"] = q_elem["Payload"]["QuestionText"].format(
            q.Date, q.Summary, q.Guest_Name, q.Guest_Utterance, q.Host_Name, q.Host_Utterance)
    return q_elem


def save_qsf(qsf_object, path):
    with open(path, 'w') as f:
        json.dump(qsf_object, f)


def _add_q_block(block_elements, q_id):
    identifiers = Q_BLOCK_IDENTIFIERS
    for i in identifiers:
        block_elements.append({
            "Type": "Question",
            "QuestionID": "QID-{0}-{1}".format(q_id, i)
        })


def _set_display_logic(q_id, qsf_txt):
    poss_conditions = ["0", "1"]
    for cond in poss_conditions:
        if cond in qsf_txt["Payload"]["InPageDisplayLogic"]["0"]:
            qsf_txt["Payload"]["InPageDisplayLogic"]["0"][cond]["QuestionID"] = \
                qsf_txt["Payload"]["InPageDisplayLogic"]["0"][cond]["QuestionID"].format(q_id)
            qsf_txt["Payload"]["InPageDisplayLogic"]["0"][cond]["ChoiceLocator"] = \
                qsf_txt["Payload"]["InPageDisplayLogic"]["0"][cond]["ChoiceLocator"].format(q_id)
            qsf_txt["Payload"]["InPageDisplayLogic"]["0"][cond]["QuestionIDFromLocator"] = \
                qsf_txt["Payload"]["InPageDisplayLogic"]["0"][cond]["QuestionIDFromLocator"].format(q_id)
            qsf_txt["Payload"]["InPageDisplayLogic"]["0"][cond]["LeftOperand"] = \
                qsf_txt["Payload"]["InPageDisplayLogic"]["0"][cond]["LeftOperand"].format(q_id)


def _add_validation(q_id, qsf_txt):
    """
        add validation that one of the four categories needs to be marked in the host reply
    :param q_id: ID of the base IE, e.g., NPR-4-2
    :param qsf_txt:
    :return:
    """
    categories = ["0", "1", "2"]  # "3" for survey before
    for cat in categories:
        if cat in qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"].keys():
            qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"][cat]["QuestionID"] = \
                qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"][cat][
                    "QuestionID"].format(
                    q_id)
            qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"][cat]["ChoiceLocator"] = \
                qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"][cat][
                    "ChoiceLocator"].format(
                    q_id)
            qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"][cat][
                "QuestionIDFromLocator"] = \
                qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"][cat][
                    "QuestionIDFromLocator"].format(q_id)
            qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"][cat]["LeftOperand"] = \
                qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"][cat][
                    "LeftOperand"].format(
                    q_id)
        else:
            warnings.warn(f"Category {cat} not found in scaffolding. This can happen if you create a survey with only "
                          f"one category to highlight.")
    # #   or nothing was highlighted in the host text
    # qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"]["1"]["QuestionID"] = \
    #     qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"]["1"]["QuestionID"].format(q_id)
    # qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"]["1"]["ChoiceLocator"] = \
    #     qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"]["1"]["ChoiceLocator"].format(
    #         q_id)
    # qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"]["1"]["QuestionIDFromLocator"] = \
    #     qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"]["1"][
    #         "QuestionIDFromLocator"].format(q_id)
    # qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"]["1"]["LeftOperand"] = \
    #     qsf_txt["Payload"]["Validation"]["Settings"]["CustomValidation"]["Logic"]["0"]["1"]["LeftOperand"].format(q_id)


def _gen_highlight_skeleton(q_template_path, q_id, txt_to_highlight, debug=False, chunk=False):
    # load the template for question generation
    qsf_txt = _get_qsf_from_path(q_template_path)

    # generate highlight question from json template
    _set_ids(q_id, qsf_txt)

    _update_hl_qsf(qsf_txt, txt_to_highlight, debug, chunk)

    return qsf_txt


def _update_hl_qsf(qsf_txt, txt_to_highlight, debug=False, chunk=False):
    word_index = 0
    choice_order = []
    # TODO: test parsing
    #   for now DEBUGGING for combining words
    if debug and ("a couple of years" in txt_to_highlight or "the issue of shelter" in txt_to_highlight):
        if "a couple of years" in txt_to_highlight:
            tokens = txt_to_highlight.split(" ")
            tokens[37] = "a&nbsp;couple&nbsp;of&nbsp;years"
            del tokens[38:41]
        else:
            tokens = txt_to_highlight.split(" ")
            tokens[35] = "the&nbsp;issue&nbsp;of&nbsp;shelter."
            del tokens[36:]
    else:
        tokens = tokenize_for_highlight_choices(txt_to_highlight)
    qsf_txt["Payload"]["HighlightText"] = qsf_txt["Payload"]["HighlightText"].format(" ".join(tokens))
    for i, token in enumerate(tokens):
        qsf_txt["Payload"]["Choices"][f"{i + 1}"] = {"WordIndex": word_index,
                                                     "WordLength": len(token),
                                                     "Word": token,
                                                     "Display": f"{i + 1}: {token}"}
        word_index += len(token) + 1
        choice_order.append(i + 1)
    qsf_txt["Payload"]["ChoiceOrder"] = choice_order
    qsf_txt["Payload"]["WordChoiceIds"] = choice_order
    qsf_txt["Payload"]["NextChoiceId"] = len(choice_order) + 1


def tokenize_for_highlight_choices(txt_to_highlight):
    """
        called to generate highlight choices for qualtrics,
            and later to get the choices that were highlighted from the report
        the actual choice indices used and saved in the dataframe start with 1, i.e.,
            "This is a test." -> ["This", "is", "a", "test."] -> [1, 2, 3, 4]

    :param txt_to_highlight:
    :return:
    """
    chunked = txt_to_highlight
    tokens = chunked.split(" ")
    if len(tokens) > 200:
        raise ValueError(f"The text is too long to add to qualtrics for highlighting. "
                         f"On Qualtrics there is a strict 200 token limit. You currently have {len(tokens)} tokens."
                         f"This should not happen unless "
                         f"you are evaluating a survey that did not initially enforce the limit. ")
    return tokens


def _get_qsf_from_path(q_template_path):
    """

    :param q_template_path: json file path
    :return:
    """
    f = open(q_template_path)
    qsf_txt = json.load(f)
    f.close()
    return qsf_txt


def _set_ids(q_id, qsf_txt):
    qsf_txt["PrimaryAttribute"] = qsf_txt["PrimaryAttribute"].format(q_id)
    qsf_txt["Payload"]["DataExportTag"] = qsf_txt["Payload"]["DataExportTag"].format(q_id)
    qsf_txt["Payload"]["QuestionID"] = qsf_txt["Payload"]["QuestionID"].format(q_id)


REFERRING_CHOICES = {
    2: "History",
    3: "No",
    4: "Yes"
}
PARAPHRASE_CANDIDATE_CHOICES = {
    4: "No",
    3: "Yes"
}


def download_and_close_survey(survey_id):
    file = qac.dowload_survey(survey_id,
                              QUALTRICS_API_AUTH_TOKEN,
                              "fra1", "tsv")
    print(file)
    close_survey(survey_id)
