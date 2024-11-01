"""
    project wide functions and constants
"""
import os
import sys

SEED = 13497754789


def set_logging():
    """
    set logging format for calling logging.info
    :return:
    """
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def set_torch_device():
    import torch
    global device
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def get_dir_to_src():
    dir_path = os.path.dirname(os.path.normpath(__file__))
    base_dir = os.path.basename(dir_path)
    if base_dir == "utility":
        return os.path.dirname(os.path.dirname(dir_path))
    elif base_dir == "paraphrase":
        return os.path.dirname(dir_path)
    else:
        return dir_path


def get_dir_to_result():
    return get_dir_to_src() + "/../result"
