import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model

IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

#Loading the tf_flowers dataset
dataset, info = tfds.load(
    'tf_flowers',
    with_info=True,
    as_supervised=True,
)

#Getting the number of classes
num_classes = info.features['label'].num_classes

#Splitting the dataset into training, validation, and test sets
def get_training_and_validation_sets(dataset, train_split=0.8, val_split=0.1, test_split=0.1):
    dataset_size = info.splits['train'].num_examples
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_dataset = dataset['train'].take(train_size)
    val_dataset = dataset['train'].skip(train_size).take(val_size)
    test_dataset = dataset['train'].skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = get_training_and_validation_sets(dataset)

#Applying preprocessing and batching
train_dataset = train_dataset.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

# --- MobileNetV2 Model ---
print("\nTraining MobileNetV2 Model...")
base_model_mobilenetv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

#Adding data augmentation and custom classification head
x = data_augmentation(base_model_mobilenetv2.input)
x = base_model_mobilenetv2(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions_mobilenetv2 = Dense(num_classes, activation='softmax')(x)
model_mobilenetv2 = Model(inputs=base_model_mobilenetv2.input, outputs=predictions_mobilenetv2)

for layer in base_model_mobilenetv2.layers:
    layer.trainable = False

model_mobilenetv2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_mobilenetv2 = model_mobilenetv2.fit(
    train_dataset,
    epochs=3, # Reduceing epochs for faster training
    validation_data=val_dataset
)
model_mobilenetv2.save('flower_classifier_mobilenetv2.h5')
print("MobileNetV2 Model training complete and saved as flower_classifier_mobilenetv2.h5")


