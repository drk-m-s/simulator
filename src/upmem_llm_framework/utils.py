#
# Copyright (c) 2014-2024 - UPMEM
#


def add_dictionaries(dict1, dict2):
    for key in dict2.keys():
        dict1[key] = dict1.get(key, 0) + dict2[key]

    return dict1
