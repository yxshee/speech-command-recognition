# Dataset

The model is trained and evaluated on **Speech Commands v0.02** (Pete Warden, Google, 2018): 105,829 one-second recordings of 35 English words from 2,618 speakers, released under the Creative Commons BY 4.0 licence.

- **Download**: <https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz> (about 2.4 GB). `python scripts/train.py --download` fetches it and unpacks it into `data/speech_commands_v0.02/`, which is ignored by git.
- **Paper**: Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*. arXiv:1804.03209.
- **Split**: the dataset's own `validation_list.txt` and `testing_list.txt` decide which clips are held out. That gives 84,843 training, 9,981 validation and 11,005 test clips, and no speaker appears in more than one of the three.
- **Background noise**: the six recordings in `_background_noise_/` are mixed into training clips as augmentation. They are not a class.

A separate set of one-second recordings made by the author is in [this Google Drive folder](https://drive.google.com/drive/folders/1kAuKUZF1qZFbNUM1SjHRAA8Gx1Ain0Lj?usp=sharing). The committed model was not trained on it.

## Sample clips

`assets/samples/yes.wav`, `stop.wav` and `left.wav` are three unmodified clips from the dataset: `yes/022cd682_nohash_0.wav`, `stop/022cd682_nohash_0.wav` and `left/022cd682_nohash_0.wav`. Each is the first clip of its word in the sorted official test list, so the model never saw them during training.
