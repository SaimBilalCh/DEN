import tensorflow as tf
import tensorflow_datasets as tfds

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

print("Dataset preprocessing complete.")
print(f"Train dataset size: {tf.data.experimental.cardinality(train_dataset).numpy() * BATCH_SIZE}")
print(f"Validation dataset size: {tf.data.experimental.cardinality(val_dataset).numpy() * BATCH_SIZE}")
print(f"Test dataset size: {tf.data.experimental.cardinality(test_dataset).numpy() * BATCH_SIZE}")


