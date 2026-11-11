"""Class labels of the committed model.

The model in model/wavmodel.keras has 41 outputs. They follow the order in
which the training notebook listed the extracted dataset directory, which
included six entries that are not words. Index i of the model output is
MODEL_LABELS[i], so this order must not change while that model is in use.
"""
import numpy as np

MODEL_LABELS = np.array([
    'learn', 'on', 'down', 'follow', 'cat', '.DS_Store', 'three', 'forward',
    'up', 'no', 'sheila', 'stop', 'tree', 'zero', 'LICENSE', 'six', 'four',
    'validation_list.txt', 'seven', 'one', 'right', 'off', 'marvin', 'five',
    'dog', 'two', 'go', 'left', 'testing_list.txt', 'wow', 'bird', 'backward',
    'house', 'visual', 'happy', 'nine', '_background_noise_',
    'speech_commands_v0.02.tar.gz', 'eight', 'bed', 'yes',
])
NON_WORD_LABELS = frozenset({
    '.DS_Store', 'LICENSE', 'validation_list.txt', 'testing_list.txt',
    '_background_noise_', 'speech_commands_v0.02.tar.gz',
})
WORD_MASK = np.array([label not in NON_WORD_LABELS for label in MODEL_LABELS])

# The 35 words the model can recognise, in alphabetical order.
WORDS = tuple(sorted(str(label) for label in MODEL_LABELS[WORD_MASK]))
