"""
    helpers for interacting with Prolific API, AUTHO tokens removed
"""
# https://docs.prolific.co/docs/api-docs/public/#tag/Studies/operation/CreateStudy
import json
from datetime import datetime
import requests

from paraphrase.set_id_consts import PROLIFIC_PARAPHRASE_PROJECT_ID, PROLIFIC_API_TOKEN, ANNOTATION_TEMPLATE_STUDY_ID
from paraphrase.utility.qualtrics_survey import _build_headers

prolific_url = "https://api.prolific.co/api/v1/studies/"
auth_name = "Authorization"

TRAINING_COMPLETION_CODE = "PROLIFIC-FOLLOW-UP=CODE"  # FOLLOW UP STUDY
ANNOTATION_COMPLETION_CODE = "PROLIFIC-AC-FAILED"  # No ACs failed

PROLIFIC_ANN_STUDY_DESCRIPTION = ("<p>This is a next part of a at least 5-part longitudinal study. You trained to "
                                  "recognize paraphrases in the previous part. Now, we ask you to apply "
                                  "what you learned (~{} minutes). Your contribution is an essential part "
                                  "of our research!</p><p>In both studies, we ask you to perform the following task: "
                                  "</p><p><b>&nbsp;&nbsp;&nbsp;&nbsp; "
                                  "Highlight text excerpts from news interviews for paraphrases</b></p><p>"
                                  "An example text excerpt could be: </p><p>&nbsp;&nbsp;&nbsp; "
                                  "<b>Summary:</b> Fresh Prince Star Alfonso Ribeiro Sues Over Dance Moves; "
                                  "Rapper 2 Milly Alleges His Dance Moves were Copied.</p><p><b>&nbsp; &nbsp; "
                                  "Guest:</b> Even like big artists, major artists like Joe Buttons and stuff, they "
                                  "have their own like show, daily struggle, they say, you, you must sue "
                                  "&quot;Fortnite&quot;, and <b><u>I'm like, &quot;Fortnite&quot;, what is that? "
                                  "I don't even know what it is</u></b> --</p><p>&nbsp; &nbsp;&nbsp;<b>Host:</b>"
                                  " So <b><u>you weren't even familiar</u></b>?</p><p>The category we are interested in:\n"
                                  "</p><ul><li>A <b><u>Paraphrase</u></b> is a  <i>rewording or repetition</i> of content in "
                                  "the guest's statement.</li><li>We also ask you to highlight the section of the guest "
                                  "statement that is paraphrased. </li></ul><p></p>There are two attention checks.")


def get_cur_day():
    current_date = datetime.now()
    return current_date.strftime('%m/%d')


def create_training_survey(qualtrics_survey_url=None, places=1, test=False, exclude=None):
    """
        Create Annotator Training Survey from the last created annotator training survey
    :param test:
    :param qualtrics_survey_url:
    :param exclude:
    :param places:
    :return:
    """
    prolific_training_id = get_newest_training_survey()
    # training_id =
    response = send_prolific_request(url=f"https://api.prolific.co/api/v1/studies/{prolific_training_id}/clone/")
    filter_list = response['filters']
    # expand blocklist for given exclude
    if exclude is not None:
        updated = False
        for filter_dict in filter_list:
            if filter_dict["filter_id"] == "custom_blocklist":
                updated_exclude = list(set(filter_dict["selected_values"]) | set(exclude))
                filter_dict["selected_values"] = updated_exclude
                updated = True
        if not updated:
            filter_list.append({
                "filter_id": "custom_blocklist",
                "selected_values": exclude
            })
    survey_id = response["id"]
    data = {  # update name and description
        "name": ("TEST " if test else "") + "1/5 Highlight Paraphrases on News Interviews",
        "internal_name": ("TEST " if test else "") + f"{get_cur_day()} Training Paraphrase on News Interviews {places}",
        "total_available_places": places,
        'study_type': "QUOTA",
        "description": "<p>This is the first of a at least 5-part longitudinal study. In the first part you will "
                       "practice a task (~15 minutes for £2.5), and in the following parts we will ask you to apply "
                       "what you have learned (on average ~14 minutes). If you qualify, you will see the next parts "
                       "on your "
                       "dashboard within 24 hours. Please participate only if you are prepared to complete at least 4 "
                       "follow up studies within the next 5 days (they will start with the name \"X/5 Highlight\"). "
                       "You are welcome to participate in more if you "
                       "like.</p><p>We train you to perform the following task: </p><p><b>&nbsp;&nbsp;&nbsp;&nbsp; "
                       "Highlight text excerpts from news interviews for paraphrases</b></p><p>An example text "
                       "excerpt could be: </p><p>&nbsp;&nbsp;&nbsp; <b>Summary:</b> Fresh Prince Star Alfonso Ribeiro "
                       "Sues Over Dance Moves; Rapper 2 Milly Alleges His Dance Moves were Copied.</p><p><b>&nbsp; "
                       "&nbsp; Guest:</b> Even like big artists, major artists like Joe Buttons and stuff, "
                       "they have their own like show, daily struggle, they say, you, you must sue "
                       "&quot;Fortnite&quot;, and <b><u>I'm like, &quot;Fortnite&quot;, what is that? I don't even "
                       "know what it is</u></b> --</p><p>&nbsp; &nbsp;&nbsp;<b>Host:</b> So <b><u>you weren't even "
                       "familiar</u></b>?</p><p>The concepts we are interested in:</p><ul><li>A "
                       "<b><u>Paraphrase</u></b> is a  <i>rewording or repetition</i> of content in the guest's "
                       "statement.</li><li>We also ask you to highlight the section of the guest statement that is "
                       "paraphrased. </li></ul><p></p><p>There is one comprehension check and two attention checks.</p>",
        "filters": (filter_list + ([{
            "filter_id": "sex",
            "selected_values": ["0", "1"],
            "weightings": {
                "0": 1,
                "1": 1,
            }
        }] if all(fltr["filter_id"] != "sex" for fltr in filter_list) else []))

    }
    if qualtrics_survey_url is not None:
        data["external_study_url"] = (
                f"{qualtrics_survey_url}" +
                "?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}")

    survey_url = f"https://api.prolific.co/api/v1/studies/{survey_id}/"

    response = send_prolific_request(url=survey_url, method="PATCH",
                                     data=data)

    return response['id']


def create_annotation_survey(survey_url, allowlist, internal_name="Paraphrase Candidates",
                             survey_name="X/5 Highlight Paraphrases on News Interviews",
                             nbr_items=10, places=1, seconds_per_item=84, test=False):
    """
        allowlist have to be valid prolific IDs (if not in test mode)
    :param seconds_per_item:
    :param survey_url:
    :param allowlist:
    :param internal_name:
    :param survey_name:
    :param nbr_items:
    :param places:
    :param test:
    :return:
    """
    response = send_prolific_request(
        url=f"https://api.prolific.co/api/v1/studies/{ANNOTATION_TEMPLATE_STUDY_ID}/clone/")
    survey_id = response["id"]
    # needs to round to be accepted by prolific
    minutes = max(round(seconds_per_item * nbr_items / 60), 1)
    data = {
        "name": ("TEST " if test else "") + survey_name,
        "internal_name": f"{get_cur_day()} {internal_name}, {minutes}mins",
        "total_available_places": places,
        "external_study_url": f"{survey_url}" +
                              "?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}",
        "estimated_completion_time": minutes,
        "reward": round((23.4 * (seconds_per_item / 84)) * nbr_items),  # 23.4 for 84 seconds
        "description": PROLIFIC_ANN_STUDY_DESCRIPTION.format(minutes),
        "filters": [
            {
                "filter_id": "custom_allowlist",
                "selected_values": allowlist
            }
        ]
    }
    if test:
        del data["filters"]
    assert (len(allowlist) > 0)
    response = send_prolific_request(url=f"https://api.prolific.co/api/v1/studies/{survey_id}/", method="PATCH",
                                     data=data)
    return survey_id


def update_allowlist_survey(prolific_id, new_participants=None):
    if new_participants is None:
        new_participants = []
    survey_url = f"https://api.prolific.co/api/v1/studies/{prolific_id}/"
    # get current allowlist
    response = send_prolific_request(url=survey_url, method="GET")
    current_allowlist = response['filters'][0]['selected_values']
    filter_set_id = response
    # pause study
    #   https://docs.prolific.com/docs/api-docs/public/#tag/Studies/operation/PublishStudy
    # response = send_prolific_request(url=survey_url, method="PATCH",
    #                                  data={
    #                                      "status": "PAUSED"
    #                                  })
    # update allowlist
    assert (len(set(current_allowlist) & set(new_participants)) == 0)
    new_allowlist = current_allowlist + new_participants
    survey_url = f"https://api.prolific.co/api/v1/studies/{prolific_id}/"
    response = send_prolific_request(url=survey_url, method="PATCH",
                                     data={
                                         "filters": [{
                                             "filter_id": 'custom_allowlist',
                                             "selected_values": new_allowlist
                                         }]
                                     })
    return response


def get_newest_training_survey():
    response = send_prolific_request(
        url=f"https://api.prolific.co/api/v1/projects/{PROLIFIC_PARAPHRASE_PROJECT_ID}/studies/",
        method="GET", data={"state": "COMPLETED"})
    listening_studies = response['results']
    # Sort the list of dictionaries by 'date_created' in ascending order, make sure there were participants before
    sorted_list = sorted([study for study in listening_studies
                          if (study["places_taken"] > 0) and
                          ('Training Paraphrase on News Interviews' in study["internal_name"])],
                         key=lambda x: x['date_created'])
    # Find the newest entry that includes "1/3 Highlight Paraphrases on News Interviews" in 'name'
    newest_entry = sorted_list[-1]
    # for entry in reversed(sorted_list):
    #     if '1/3 Highlight Paraphrases on News Interviews' in entry['name']:
    #         newest_entry = entry
    #         break
    training_id = newest_entry["id"]
    return training_id


def _delete_prolific_study(study_id):
    response = send_prolific_request(url=prolific_url + study_id + "/", method="DELETE")
    return response


def send_prolific_request(data=None, url=prolific_url, method="POST"):
    headers = _build_headers(method, auth_token="Token " + PROLIFIC_API_TOKEN, auth_token_name=auth_name, appljson=True)
    # response = requests.post(
    #     prolific_url,
    #     data=data,
    #     headers=headers
    # )
    # response = requests.get("http://www.example.com/", headers={"Content-Type": "text"})
    if method == "GET":
        if data is None:
            response = requests.get(
                url,
                headers=headers
            )
        else:
            response = requests.get(
                url,
                headers=headers,
                params=data
            )
    elif method == "POST":
        if data is None:
            response = requests.post(
                url,
                headers=headers
            )
        else:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(data)
            )
    elif method == "PATCH":
        response = requests.patch(
            url,
            headers=headers,
            data=json.dumps(data)
        )
    elif method == "DELETE":
        response = requests.delete(
            url,
            headers=headers
        )
    else:
        raise ValueError(f"method {method} is not valid")
    response.raise_for_status()
    print(f'Response: {response.status_code}')
    if response.status_code != 204:  # https://stackoverflow.com/questions/16573332/jsondecodeerror-expecting-value-line-1-column-1-char-0
        return response.json()
    else:
        return None


def get_approved_annotators_for_study(prolific_id, completion_code: str = TRAINING_COMPLETION_CODE):
    """
        for a given id to a annotator training survey on prolific (prolific_id),
        return the list of annototators that passed training
    :param prolific_id:
    :param completion_code:
    :return:
    """
    allowlist = _get_anns_wo_completion_code(prolific_id, completion_code)
    return allowlist


def get_all_annotator_ids_for_study(prolific_id, test=False):
    if not test:
        return _get_anns_wo_completion_code(prolific_id)
    else:
        if prolific_id != "65241b6ea745597359c8051f":
            return ["Annotator1", "Annotator2", "Annotator3"]
        else:
            return ['Annotator8', 'Annotator10', 'Annotator9']


def get_status_code_for_study(prolific_id):
    response = send_prolific_request(url=f"https://api.prolific.co/api/v1/studies/{prolific_id}/",
                                     method="GET")
    return response['status']


def stop_study(prolific_id):
    # see https://docs.prolific.com/docs/api-docs/public/#tag/Studies/operation/PublishStudy
    if get_status_code_for_study(prolific_id) != "AWAITING REVIEW":
        response = send_prolific_request(url=f"https://api.prolific.co/api/v1/studies/{prolific_id}/transition/",
                                         method="POST", data={'action': 'STOP'})
        return response['status']
    else:
        return None


def restart_study(
        prolific_id):  # TODO: this seems to only work whith increasing places even if before not all places were taken ...
    if get_status_code_for_study(prolific_id) == "AWAITING REVIEW":
        cur_places = get_cur_places_study(prolific_id)
        response = send_prolific_request(url=f"https://api.prolific.co/api/v1/studies/{prolific_id}",
                                         method="PATCH",
                                         data={'total_available_places': cur_places})
        return response['status']
    else:
        raise ValueError("Are you sure you want to increase places?")


def _get_anns_wo_completion_code(prolific_id, completion_code=None):
    response = send_prolific_request(url=f"https://api.prolific.co/api/v1/"
                                         f"submissions/",
                                     method="GET",
                                     data={"study": prolific_id})
    if completion_code:
        annotators = [ann["participant_id"] for ann in response['results'] if
                      ann["study_code"] == completion_code]
    else:
        annotators = [ann["participant_id"] for ann in response['results']]
    return annotators


def get_cur_places_study(prolific_id):
    response = send_prolific_request(url=f"https://api.prolific.co/api/v1/"
                                         f"studies/{prolific_id}",
                                     method="GET")
    return int(response["total_available_places"])


def is_an_annotator_active(prolific_id):
    response = send_prolific_request(url=f"https://api.prolific.co/api/v1/"
                                         f"submissions/",
                                     method="GET",
                                     data={"study": prolific_id})
    return any(ann["status"] == "ACTIVE" for ann in response['results'])


" ========================= INSPECT OBJECTS ON PROLIFIC ================== "


def get_project_studies(project_id):
    response = send_prolific_request(url=f"https://api.prolific.co/api/v1/projects/{project_id}/studies/",
                                     method="GET")
    return [study['id'] for study in response['results']]


def get_study(study_id):
    response = send_prolific_request(url=f"https://api.prolific.co/api/v1/studies/{study_id}/",
                                     method="GET")
    return response


def get_submissions_for_study(study_id):
    response = send_prolific_request(url=f"https://api.prolific.co/api/v1/submissions/",
                                     method="GET",
                                     data={"study": study_id})
    return response['results']


def get_pph_for_study(study_id):
    # "time_taken" is given in seconds
    # "reward" is given in GBP and cents*100 for some reason, i.e., 65000 is 6.5 GBP
    submissions = get_submissions_for_study(study_id)
    return [((sub['reward'] + sum(sub['bonus_payments'])) / (100 ** 2)) / (sub['time_taken'] / (60 ** 2))
            for sub in submissions if sub['status'] == "APPROVED"]


def get_reward_for_study(study_id):
    submissions = get_submissions_for_study(study_id)
    return sum(sub['reward'] for sub in submissions if sub['status'] == "APPROVED")


def print_pph_project():
    """
        get the pay per hour for the whole prolific project
    :return:
    """
    study_ids = get_project_studies(PROLIFIC_PARAPHRASE_PROJECT_ID)
    pphs = []
    valid_studies = 0
    for study_id in study_ids:
        cur_pays = get_pph_for_study(study_id)
        if len(cur_pays) > 0:
            pphs += cur_pays
            print(f"Study {study_id} has {len(cur_pays)} approved submissions.")
            valid_studies += 1
        # else:
        #     print(f"No valid submissions for study {study_id}.")
    print(pphs)
    # calculate mean pph
    mean_pph = sum(pphs) / len(pphs)
    # get median pph
    pphs.sort()
    if len(pphs) % 2 == 0:
        median_pph = (pphs[len(pphs) // 2 - 1] + pphs[len(pphs) // 2]) / 2
    else:
        median_pph = pphs[len(pphs) // 2]
    print(f"mean pph: {mean_pph}")
    print(f"median pph: {median_pph}")
    print(f"Over {len(pphs)} sessions in {valid_studies} non-empty studies.")
