
<p float="left">

<img width="190"  height="190"  src="https://media.giphy.com/media/9Q5fSHyPKfrr2/giphy.gif?cid=ecf05e47qr0xp3f59aue8ckfchtvw3uxdayo3vb5pzqj99nh&ep=v1_gifs_related&rid=giphy.gif&ct=g">

<img width="190"  height="190"  src="https://media.giphy.com/media/pViWHLiQBA1q0/giphy.gif?cid=ecf05e47qr0xp3f59aue8ckfchtvw3uxdayo3vb5pzqj99nh&ep=v1_gifs_related&rid=giphy.gif&ct=g">

<!-- <img   width="190" height="200" src="https://media.giphy.com/media/Qw4X3FwLj6EgczeQOuk/giphy.gif?cid=ecf05e471w059g8j66r5ndmpksg5h4dhb8p1v2i52dxenuuz&ep=v1_gifs_related&rid=giphy.gif&ct=g"> -->

<img  width="190" height="190"  src="https://media.giphy.com/media/Bom9WWqrVVOAc5ANVE/giphy.gif?cid=ecf05e47qr0xp3f59aue8ckfchtvw3uxdayo3vb5pzqj99nh&ep=v1_gifs_related&rid=giphy.gif&ct=g">

<img width="190"  height="190"  src="https://media.giphy.com/media/d2YVJ4m4aUrNwl44/giphy.gif?cid=ecf05e47h0tc9ft2twbxb8bg0iozgpu99t7ukzbfm13u9qly&ep=v1_gifs_related&rid=giphy.gif&ct=g">

</p>

# Speech Command Recognition Project

This project focuses on building a robust keyword recognition system using the [Speech Commands Dataset v2](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz). The dataset consists of one-second audio files containing spoken English words, enabling the training of machine learning models for real-time keyword detection. The system aims to:

- Accurately recognize command words from short audio clips.
- Perform robustly in noisy environments.
- Be customizable with additional user-specific data for fine-tuning.

---

## **Dataset Description**

The [Speech Commands Dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) is a collection of 105,829 audio samples, each containing a single spoken English word. The dataset is organized into 35 categories, including:

- **Core Commands**: Yes, No, Up, Down, Left, Right, On, Off, Stop, Go.
- **Auxiliary Words**: Cat, Dog, Bird, Tree.
- **Noise Samples**: Background noise (white noise, pink noise, running water, etc.).

### **Key Features of the Dataset**:
- **Diversity**: Contains recordings from speakers with varied ages, genders, and accents.
- **Partitioning**:
  - **Training Data**: 80% of the dataset.
  - **Validation Data**: 10%.
  - **Test Data**: 10%.

---

## **Preprocessing**

To prepare the audio data for machine learning, several preprocessing steps are applied:

1. **Audio Normalization**:
   Ensures uniform loudness across recordings.

2. **Spectrogram Conversion**:
   Converts the raw audio into Mel-spectrograms for easier analysis by convolutional neural networks (CNNs).

3. **Noise Augmentation**:
   Adds background noise to training samples to enhance model robustness in real-world environments.

### **Sample Preprocessing Code**
```python
import tensorflow as tf
import numpy as np
import os

# Load the dataset
commands_dir = 'path_to_speech_commands_dataset'
commands = np.array(tf.io.gfile.listdir(commands_dir))
commands = commands[commands != '_background_noise_']

print(f'Commands: {commands}')

# Preprocess function
def preprocess_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.signal.stft(audio, frame_length=255, frame_step=128)
    spectrogram = tf.abs(audio)
    return spectrogram

# Example Usage
example_file = os.path.join(commands_dir, 'yes', 'sample_audio.wav')
spectrogram = preprocess_audio(example_file)
```

---

## **Model Architecture**

The system uses a Convolutional Neural Network (CNN), which excels in processing spectrograms. The architecture is designed to recognize spatial and temporal patterns in audio data.

![image](https://github.com/user-attachments/assets/0ac6f7f0-b644-43b4-81fb-59afec7c1462)

### **Model Details**

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
        Dense(35, activation='softmax')  # 35 classes for 35 words
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (64, 64, 1)
model = create_cnn_model(input_shape)
model.summary()
```

<img width="1397" alt="image" src="https://github.com/user-attachments/assets/a595225f-4574-4bb4-8bfc-2fd524a48efb" />


### **Training Parameters**
- **Optimizer**: Adam.
- **Loss Function**: Categorical Crossentropy.
- **Learning Rate**: 0.001.
- **Batch Size**: 64.
- **Epochs**: 30.

---

## **Results and Performance**

<p float="left">
 
### **Key Metrics**:
- **Validation Accuracy**: 86%
- **Test Accuracy**: 85.65%
- **Loss on Test Set**: 0.629

<img width="480" alt="image" src="https://github.com/user-attachments/assets/e9ddae1d-6829-4070-a573-c4734749a74d" >

</p>



### **Training Progress**:
```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()
```


<img width="1000" alt="image" src="https://github.com/user-attachments/assets/81907b17-abe8-4814-92a6-440ae8eeee17" />


### **Confusion Matrix**
The confusion matrix highlights which command words are frequently misclassified. For instance, words like “No” and “Go” were occasionally confused due to phonetic similarities.

<img width="480" alt="image" src="https://github.com/user-attachments/assets/9de2207e-3e7d-425b-a944-10bb7218055a" ></p>

---

## **Fine-Tuning and Customization**

To improve performance for specific users, the model can be fine-tuned with additional recordings. Custom datasets (e.g., 30 samples per word) were used to:

- Enhance personalization.
- Improve accuracy in specific environments.

### **Challenges and Solutions**:
- **Overfitting**: Addressed using dropout and noise augmentation.
- **Generalization**: Maintained by balancing original and custom data.

---

## **Future Improvements**

1. Experimenting with advanced architectures such as recurrent neural networks (RNNs) or transformers.
2. Deploying the model for real-time inference on mobile or edge devices.
3. Enhancing noise augmentation techniques to improve performance in challenging environments.

---

## **Resources**

- [Speech Commands Dataset v2](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
- [TensorFlow Audio Recognition Guide](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Project Repository](https://github.com/yxshee/speech-command-recognition)

---

