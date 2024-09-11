# 102166002_Speech_Report
The **[Speech Commands Dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)** is a collection of one-second audio files (.wav) containing single spoken English words. Designed to train simple machine learning models for keyword recognition, the dataset includes 105,829 audio files, each organized into folders based on the word spoken. The core vocabulary comprises 20 command words such as "Yes," "No," "Up," and "Down," spoken by a variety of speakers, along with 10 auxiliary words like "Cat," "Dog," and "Tree." This dataset was collected via crowdsourcing using open-source tools available [here](https://aiyprojects.withgoogle.com/open_speech_recording) and is licensed under the [Creative Commons BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

Files are partitioned into training, validation, and test sets using a hashing function to maintain consistency as new files are added. The data includes a `_background_noise_` folder to simulate realistic noisy environments. The dataset is particularly useful for training machine learning models to recognize specific words in low-resource, real-world applications where recording quality varies.

You can access an interactive example and explore the dataset further using this [Google Colab Notebook](https://colab.research.google.com/drive/1HQETNtmnjnFx94deACCVYfQH7zl7STf9?usp=sharing).

For more details, refer to the paper on [arXiv](https://arxiv.org/abs/1804.03209).
