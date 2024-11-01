"""
    identifying constants and paths used in the project
        ==> should be reset to your own paths
"""

import os
from pathlib import Path
import logging

from paraphrase.utility.project_functions import get_dir_to_src

"""
    ======== PATHS =========
"""

cur_dir = os.path.dirname(os.path.realpath(__file__))
BASE_FOLDER = cur_dir + "/../"

LOCAL_HOME = str(Path.home())
STRANFORMERS_CACHE = LOCAL_HOME + "/sentence_transformer/"  # "/home/USER/sentence_transformer/" for linux
TRANSFORMERS_CACHE = LOCAL_HOME + "/huggingface/"  # "/home/USER/huggingface/" for linux

ON_HPC = False
if "KEYWORD-FOR-HPC" in BASE_FOLDER:  # for our local setup this checks if we are on HPC cluster,
    logging.info("On Cluster")
    # file structures are different there, can probably be removed for your own setup
    ON_HPC = True

# LOCAL setup relict, can be removed for your own setup
if ON_HPC:
    BASE_FOLDER = "HPC-FOLDER"
    CONVO_CACHE = BASE_FOLDER + 'cache/convokit'
    STRANFORMERS_CACHE = BASE_FOLDER + 'sentence_transformers'
    TRANSFORMERS_CACHE = BASE_FOLDER + "huggingface"


def get_dir_to_ms():
    if ON_HPC:
        return BASE_FOLDER + "data/MediaSum/news_dialogue.json"
    else:
        return get_dir_to_src() + "/../data/MediaSum/news_dialogue.json"


"""
    ========== API TOKENS ==========
"""

QUALTRICS_API_AUTH_TOKEN = "QUALTRICS-API-TOKEN"
MANUAL_EXCLUSION = [""]  #
MANUAL_INCLUSION = [""]  # previous rounds
TEST_VALID_PROLIFIC_ID = "TEST-ID"
TEST_ANN_QUALTRICS_ID_1 = "SV_4IM74gZtPIjWS1g"
TEST_ANN_QUALTRICS_ID_2 = "SV_097ANu2TQcBu6IS"


# list of annotators for 50/50 sorted by start time of training
PAID_ANNOTATORS_SORTED_BY_START_TIME = \
    [""]
PAID_ANNOTATORS_SORTED_BY_START_TIME_18 = PAID_ANNOTATORS_SORTED_BY_START_TIME[:-2]
GPT_4 = "OPEN-AI-API-KEY"
CLUSTER_OUT = BASE_FOLDER + "PARAPHRASE/"
PROLIFIC_PARAPHRASE_PROJECT_ID = "PROLIFIC-PROJECT-ID"
PROLIFIC_API_TOKEN = "PROLIFIC-API-TOKEN"
PROLIFIC_WORKSPACE_ID = "PROLIFIC-WORKSPACE-ID"
ANNOTATION_TEMPLATE_STUDY_ID = "PROLIFIC-TEMPLATE-STUDY-ID"


def set_cache():
    try:
        os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
        os.environ["HF_HOME"] = TRANSFORMERS_CACHE
        logging.info(f'setting cache to {TRANSFORMERS_CACHE}')
    except NameError:
        logging.info("using default cache for transformers")
    try:
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = STRANFORMERS_CACHE
    except NameError:
        logging.info("using default cache for sentence_transformers")


PROLIFIC_STUDY = ""
PROLFIC_ANN = ""
PROLFIC_PROJECT = ""
QUALTRICS_SURVEY = ""
LEAD = "LEAD"
