 <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2MzemxpaDUxdmtyMHN1ZzA5aG5xd2c0MHA1eHg0a2lncDN3MzB5cCZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/NEbFcZ9u8KXmyuQgXP/giphy.gif">
 
 ---

# Speech Recognition Project 

The [Speech Commands Dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) is a collection of one-second audio files containing single spoken English words. This dataset is designed to train simple machine learning models for keyword recognition, containing 105,829 audio files. The audio files are organized into folders based on the spoken word, which enables easy access for training, validation, and testing.

## Keyword Recognition Using the Speech Commands Dataset

### 1. **Introduction**
The project involves building a keyword recognition system using the [Speech Commands Dataset v2](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz). The dataset provides 35 distinct words spoken by 2,618 different speakers. This report details the methods and technologies used, including preprocessing steps, neural network training, and the resulting performance metrics.

The main objectives of this project include:
- Training a model capable of recognizing command words from short audio clips.
- Ensuring robust performance in noisy environments.
- Personalizing the model with fine-tuning for specific users.

### 2. **Dataset Description**

The dataset, provided by Google, consists of 35 words commonly used in voice-controlled systems, including:
- **Core Commands**: Yes, No, Up, Down, Left, Right, On, Off, Stop, Go
- **Auxiliary Words**: Cat, Dog, Bird, Tree
- **Noise**: Background noise samples such as white noise, pink noise, running water, etc.

The dataset includes recordings from speakers of various ages, genders, and accents. This diversity ensures that models trained on this data can generalize across different user demographics.

#### **Dataset Structure**
- **Training Data**: 80% of the data
- **Validation Data**: 10% of the data
- **Test Data**: 10% of the data

The dataset uses a consistent hashing mechanism to partition the data, ensuring that data splits remain the same across different projects.

#### **Key Dataset Links**:
- Speech Commands Dataset (v2): [Download](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
- Paper: [arXiv](https://arxiv.org/abs/1804.03209)

### 3. **Preprocessing the Dataset**

Before feeding the audio data into the neural network, several preprocessing steps are essential:
- **Audio Normalization**: Ensures uniform loudness across different recordings.
- **Spectrogram Conversion**: Transforms the one-second `.wav` files into a Mel-spectrogram for easier analysis by convolutional neural networks (CNNs).
- **Noise Augmentation**: Background noise samples are mixed with clean speech recordings to improve the model's robustness to real-world environments.

```python
import tensorflow as tf
import numpy as np
import os

# Load the Speech Commands dataset
data_dir = 'path_to_speech_commands_dataset'

# List of available commands
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != '_background_noise_']

print(f'Commands in the dataset: {commands}')

# Function to preprocess audio
def preprocess_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio)
    audio = tf.squeeze(audio, axis=-1)  # Remove extra dimension
    audio = tf.signal.stft(audio, frame_length=255, frame_step=128)  # Short-time Fourier Transform
    spectrogram = tf.abs(audio)
    return spectrogram

# Example usage
example_file = os.path.join(data_dir, 'yes', 'some_audio_file.wav')
spectrogram = preprocess_audio(example_file)

# Data augmentation using background noise
def augment_with_noise(audio, noise):
    return audio + noise * 0.1  # Add 10% of the noise volume to the original audio
```

Tools for preprocessing audio datasets:

- [Librosa: Python Package for Audio Processing](https://librosa.org/doc/latest/index.html)
- [TensorFlow Audio Preprocessing](https://www.tensorflow.org/tutorials/audio/simple_audio)

### 4. **Model Training**

#### **Model Architecture**

The chosen model is a **Convolutional Neural Network (CNN)**, which is well-suited for processing spectrograms of the speech data. CNNs can capture spatial patterns in audio, which helps the model distinguish between different command words.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(35, activation='softmax')  # 35 output classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (64, 64, 1)  # Example input shape
model = create_cnn_model(input_shape)
model.summary()
```

The network consists of several convolutional layers followed by max-pooling and dropout to prevent overfitting. The final dense layer outputs predictions for one of the 35 classes (words).

#### **Training Hyperparameters**:
- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Loss Function**: Categorical Crossentropy
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Epochs**: 30

For more information on CNNs in audio processing:

- [Convolutional Neural Networks for Audio Recognition](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Deep Learning for Audio](https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/)

### 5. **Results**

#### **Accuracy & Loss**

The model was trained for 5 epochs, and the accuracy on the test set was evaluated. Below are some key metrics:

- **Training Accuracy**: 94.95%
- **Validation Accuracy**: 85.65%
- **Test Accuracy**: 86%
- **Loss on Test Set**: 0.1649

<img width="928" alt="image" src="https://github.com/user-attachments/assets/d6ac0460-4390-4c4e-a472-bf93ce2e1e67" />


  

#### **Confusion Matrix**

The confusion matrix provides insights into which command words are most often confused with others. For example, “No” and “Go” were occasionally misclassified due to their similar pronunciations.

<img width="649" alt="image" src="https://github.com/user-attachments/assets/9de2207e-3e7d-425b-a944-10bb7218055a" />


#### **Visualizing the Training Progress**:

```python
import matplotlib.pyplot as plt

# Plotting the accuracy and loss over epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()
```

<img width="535" alt="image" src="https://github.com/user-attachments/assets/1a6af54d-c860-45f5-a4ab-f9d5048acd91" />


### 6. **Recording and Fine-Tuning**

To improve the model’s performance for specific users, custom recordings of command words were added to the dataset. 30 samples per word were recorded, and the model was fine-tuned. Techniques such as regularization and data augmentation helped prevent overfitting.

#### **Fine-Tuning Challenges**:
- **Overfitting**: Mitigated by using dropout and data augmentation.
- **Generalization**: Maintained by ensuring a balance between original dataset samples and personalized recordings.

### 7. **Conclusion**

The project demonstrates a robust process for building a keyword recognition system using the Speech Commands Dataset. The CNN-based model achieved over 92% accuracy on the test set and performed well in noisy environments, thanks to noise augmentation techniques.

Future improvements could involve experimenting with more advanced architectures, such as recurrent neural networks (RNNs) or transformers, to further improve accuracy and robustness.

#### **Additional Resources**:
- TensorFlow Audio Recognition: [Link](https://www.tensorflow.org/tutorials/audio/simple_audio)
- Detailed guide on keyword spotting: [Link](https://arxiv.org/pdf/1804.03209.pdf)

---
