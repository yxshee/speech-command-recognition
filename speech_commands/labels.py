"""The 35 words of Speech Commands v0.02, in the order of the model's outputs.

Index i of the model output is WORDS[i]. The order is alphabetical, which is
also the sorted order of the dataset's word directories; scripts/train.py
refuses to run if the dataset on disk does not match this list.
"""

WORDS = (
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
    'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes',
    'zero',
)
