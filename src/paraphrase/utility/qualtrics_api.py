"""
    helper functions for accessing the Qualtrics API
"""

import requests
import zipfile
import io, os
import re

from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.set_id_consts import QUALTRICS_API_AUTH_TOKEN

PARA_RAND_DOWNLOAD_FOLDER = (get_dir_to_src() +
                             "/../result/Annotations/Paraphrase Annotations/RAND-PARA/Automated Download/")

QUALTRICS_API_AUTH_TOKEN_NAME = 'X-API-TOKEN'

"""
    ======== downloading surveys automatically from qualtrics ========
"""


def get_automated_download_folder():
    """
        get the folder where qualtrics files that were downloaded are saved to
    :return:
    """
    working_dir = get_dir_to_src()
    submission_folder = working_dir + PARA_RAND_DOWNLOAD_FOLDER
    return submission_folder


def dowload_survey(surveyId, apiToken=QUALTRICS_API_AUTH_TOKEN, dataCenter="fra1", fileFormat="tsv"):
    """
        largely copied from https://api.qualtrics.com/u9e5lh4172v0v-survey-response-export-guide
    :param surveyId:
    :param apiToken:
    :param dataCenter:
    :param fileFormat:
    :return:
    """
    surveyId = surveyId
    fileFormat = fileFormat
    dataCenter = dataCenter

    # Setting static parameters
    requestCheckProgress = 0.0
    progressStatus = "inProgress"
    baseUrl = "https://{0}.qualtrics.com/API/v3/surveys/{1}/export-responses/".format(dataCenter, surveyId)
    headers = {
        "content-type": "application/json",
        "x-api-token": apiToken,
    }

    # Step 1: Creating Data Export
    downloadRequestUrl = baseUrl
    downloadRequestPayload = '{"format":"' + fileFormat + '"}'
    downloadRequestResponse = requests.request("POST", downloadRequestUrl, data=downloadRequestPayload,
                                               headers=headers)
    progressId = downloadRequestResponse.json()["result"]["progressId"]
    print(downloadRequestResponse.text)

    # Step 2: Checking on Data Export Progress and waiting until export is ready
    while progressStatus != "complete" and progressStatus != "failed":
        print("progressStatus=", progressStatus)
        requestCheckUrl = baseUrl + progressId
        requestCheckResponse = requests.request("GET", requestCheckUrl, headers=headers)
        requestCheckProgress = requestCheckResponse.json()["result"]["percentComplete"]
        print("Download is " + str(requestCheckProgress) + " complete")
        progressStatus = requestCheckResponse.json()["result"]["status"]

    # step 2.1: Check for error
    if progressStatus is "failed":
        raise Exception("export failed")

    fileId = requestCheckResponse.json()["result"]["fileId"]

    # Step 3: Downloading file
    requestDownloadUrl = baseUrl + fileId + '/file'
    requestDownload = requests.request("GET", requestDownloadUrl, headers=headers, stream=True)

    # Step 4: Unzipping the file
    submission_folder = get_automated_download_folder() + f"{surveyId}/"
    zipfile.ZipFile(io.BytesIO(requestDownload.content)).extractall(submission_folder)
    print('Renaming file')
    current_file_path = ((submission_folder + "/" +
                          re.search(r'=(.*?)\.zip',
                                    requestDownload.headers['content-disposition'].replace("%7C", "|")
                                    ).group(1))
                         + ".tsv")
    # Define the new file path
    new_file_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), f'{surveyId}.tsv')
    # Rename and move the file
    os.rename(current_file_path, new_file_path)
    os.rmdir(os.path.dirname(current_file_path))
    print('Complete')
    return new_file_path




