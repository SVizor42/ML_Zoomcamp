import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import xception, resnet50, VGG16


def make_dataset(
        model_type: str,
        data_dir: str,
        target_size,
        batch_size: int,
        use_augmentation: bool = False,
        use_shuffle: bool = True
) -> tf.data.Dataset:
    model_type = str.lower(model_type)
    if model_type == 'xception':
        preprocessor = xception.preprocess_input
    elif model_type == 'resnet50':
        preprocessor = resnet50.preprocess_input
    elif model_type == 'vgg16':
        preprocessor = VGG16.preprocess_input
    else:
        raise NotImplementedError()

    if use_augmentation:
        data_generator = ImageDataGenerator(
            preprocessing_function=preprocessor,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(0.5, 1.5),
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )
    else:
        data_generator = ImageDataGenerator(preprocessing_function=preprocessor)

    dataset = data_generator.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=use_shuffle
    )

    return dataset
