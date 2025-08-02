import tensorflow_datasets as tfds

#Loading the tf_flowers dataset
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)

#Printing information about the dataset
print(info)

#Getting the training and validation splits
train_dataset = dataset['train']

#Taking one example to inspect its shape and data type
for image, label in train_dataset.take(1):
    print(f"Image shape: {image.shape}")
    print(f"Label: {label.numpy()}")
    print(f"Image dtype: {image.dtype}")

#Getting the number of classes
num_classes = info.features['label'].num_classes
print(f"Number of classes: {num_classes}")

#Getting the class names
class_names = info.features['label'].names
print(f"Class names: {class_names}")

