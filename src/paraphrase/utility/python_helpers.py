"""
    This module contains helper functions for python data manipulation.
"""


def to_dict_of_lists(data_items):
    """

    :param data_items:
    :return:
    """
    result_data = {}
    for item in data_items:
        for k, v in item.items():
            if k not in result_data:
                result_data[k] = []
            result_data[k].append(v)
    return result_data
