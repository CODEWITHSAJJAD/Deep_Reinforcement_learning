import librosa
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython import display

# Set seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = 'ds_audio'
data_dir = pathlib.Path(DATASET_PATH)

# Fixed data augmentation functions
def add_noise(audio, noise_factor=0.005):
    # Generate noise with same shape as audio
    noise = np.random.randn(*audio.shape) * noise_factor
    return audio + noise

def time_shift(audio, shift_max=0.2):
    results = np.zeros_like(audio)
    for i in range(audio.shape[0]):  # Process each sample in batch
        shift = np.random.randint(-int(shift_max * audio.shape[1]),
                                 int(shift_max * audio.shape[1]))
        if shift > 0:
            results[i, shift:] = audio[i, :-shift]
        else:
            results[i, :shift] = audio[i, -shift:]
    return results

def augment_audio(audio, label):
    audio = add_noise(audio)
    audio = time_shift(audio)
    return audio, label

# Load dataset
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=32,
    validation_split=0.2,
    seed=seed,
    output_sequence_length=16000,
    subset='both')

label_names = np.array(train_ds.class_names)
print("Label names:", label_names)

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Apply data augmentation to training set
def tf_augment(audio, label):
    # Use tf.numpy_function to wrap our numpy augmentation functions
    aug_audio, aug_label = tf.numpy_function(
        augment_audio,
        [audio, label],
        (tf.float32, tf.int32))
    aug_audio.set_shape(audio.shape)
    aug_label.set_shape(label.shape)
    return aug_audio, aug_label

train_ds = train_ds.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)

# Split validation and test sets
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=512, frame_step=256, fft_length=512)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio,label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)


# Improved model architecture
def create_model(input_shape, num_labels):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(64, 64),  # Increased resolution
        layers.Normalization(),

        # First conv block
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # Second conv block
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # Third conv block
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_labels, activation='softmax')  # Changed to softmax for direct probability output
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    return model


model = create_model(input_shape, num_labels)
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

EPOCHS = 100
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# Load best model
model = load_model('best_model.h5')

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_spectrogram_ds)
print(f'Test Accuracy: {test_acc:.2%}')

# Save final model
model.save("audiomodel.h5")

# Plot training history
metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1, 2, 2)
plt.plot(history.epoch, 100 * np.array(metrics['accuracy']), 100 * np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.show()