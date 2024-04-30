# import os
# import urllib.request as request
# from zipfile import ZipFile
# import tensorflow as tf
# import time
# from pathlib import Path
# from cnnClassifier.entity.config_entity import TrainingConfig


# class Training:
#     def __init__(self, config: TrainingConfig):
#         self.config = config

    
#     def get_base_model(self):
#         self.model = tf.keras.models.load_model(
#             self.config.updated_base_model_path
#         )

#     def train_valid_generator(self):

#         datagenerator_kwargs = dict(
#             rescale = 1./255,
#             validation_split=0.20
#         )

#         dataflow_kwargs = dict(
#             target_size=self.config.params_image_size[:-1],
#             batch_size=self.config.params_batch_size,
#             interpolation="bilinear"
#         )

#         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs
#         )

#         self.valid_generator = valid_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="validation",
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         if self.config.params_is_augmentation:
#             train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=40,
#                 horizontal_flip=True,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 shear_range=0.2,
#                 zoom_range=0.2,
#                 **datagenerator_kwargs
#             )
#         else:
#             train_datagenerator = valid_datagenerator

#         self.train_generator = train_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="training",
#             shuffle=True,
#             **dataflow_kwargs
#         )

    
#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)



    
#     def train(self):
#         self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
#         self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

#         self.model.fit(
#             self.train_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=self.steps_per_epoch,
#             validation_steps=self.validation_steps,
#             validation_data=self.valid_generator
#         )

#         self.save_model(
#             path=self.config.trained_model_path,
#             model=self.model
#         )


# import tensorflow as tf
# from cnnClassifier.entity.config_entity import TrainingConfig, PrepareBaseModelConfig
# from pathlib import Path

# class Training:
#     def __init__(self, config: TrainingConfig, base_config:PrepareBaseModelConfig):
#         self.config = config
#         self.base_config = base_config

#     def get_base_model(self):
#         self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
#         # Reinitialize the optimizer - specify the optimizer with your desired parameters
#         optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_config.params_learning_rate)

#     # Compile the model again with the new optimizer
#         self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     def train_valid_generator(self):
#         dataflow_kwargs = {
#             'label_mode': 'categorical',
#             'image_size': self.config.params_image_size[:-1],
#             'batch_size': self.config.params_batch_size,
#             'interpolation': "bilinear",
#             'seed': 123
#         }

#         valid_generator = tf.keras.utils.image_dataset_from_directory(
#             directory=self.config.training_data,
#             validation_split=0.20,
#             subset="validation",
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         train_generator = tf.keras.utils.image_dataset_from_directory(
#             directory=self.config.training_data,
#             validation_split=0.20,
#             subset="training",
#             shuffle=True,
#             **dataflow_kwargs
#         )

#         if self.config.params_is_augmentation:
#             augmentation_layers = [
#                 tf.keras.layers.Rescaling(1./255),
#                 tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#                 tf.keras.layers.RandomRotation(0.2),
#                 tf.keras.layers.RandomZoom(0.2),
#                 tf.keras.layers.RandomTranslation(0.2, 0.2)
#             ]
#             train_datagenerator = tf.keras.models.Sequential(augmentation_layers)
#             self.train_generator = train_generator.map(
#                 lambda x, y: (train_datagenerator(x, training=True), y),
#                 num_parallel_calls=tf.data.AUTOTUNE
#             )
#         else:
#             self.train_generator = train_generator.map(
#                 lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
#                 num_parallel_calls=tf.data.AUTOTUNE
#             )

#         self.valid_generator = valid_generator.map(
#             lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
#             num_parallel_calls=tf.data.AUTOTUNE
#         )

#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)

#     def train(self):
#         self.steps_per_epoch = self.train_generator.cardinality().numpy()
#         self.validation_steps = self.valid_generator.cardinality().numpy()

#         self.model.fit(
#             self.train_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=self.steps_per_epoch,
#             validation_data=self.valid_generator,
#             validation_steps=self.validation_steps
#         )

#         self.save_model(
#             path=self.config.trained_model_path,
#             model=self.model
#         )


# class Training:
#     def __init__(self, config: TrainingConfig):
#         self.config = config

#     def get_base_model(self):
#         self.model = tf.keras.models.load_model(
#             self.config.updated_base_model_path
#         )

#     def train_valid_generator(self):
#         dataflow_kwargs = {
#             'label_mode': 'int',
#             'image_size': self.config.params_image_size[:-1],
#             'batch_size': self.config.params_batch_size,
#             'interpolation': "bilinear",
#             'seed': 123
#         }

#         valid_generator = tf.keras.utils.image_dataset_from_directory(
#             directory=self.config.training_data,
#             validation_split=0.20,
#             subset="validation",
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         train_generator = tf.keras.utils.image_dataset_from_directory(
#             directory=self.config.training_data,
#             validation_split=0.20,
#             subset="training",
#             shuffle=True,
#             **dataflow_kwargs
#         )

#         if self.config.params_is_augmentation:
#             augmentation_layers = [
#                 tf.keras.layers.Rescaling(1./255),
#                 tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#                 tf.keras.layers.RandomRotation(0.2),
#                 tf.keras.layers.RandomZoom(0.2),
#                 tf.keras.layers.RandomTranslation(0.2, 0.2)
#             ]
#             train_datagenerator = tf.keras.models.Sequential(augmentation_layers)
#         else:
#             train_datagenerator = tf.keras.models.Sequential([tf.keras.layers.Rescaling(1./255)])

#         self.train_generator = train_datagenerator(train_generator)
#         self.valid_generator = train_datagenerator(valid_generator)

#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)

#     def train(self):
#         self.steps_per_epoch = self.train_generator.cardinality().numpy()
#         self.validation_steps = self.valid_generator.cardinality().numpy()

#         self.model.fit(
#             self.train_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=self.steps_per_epoch,
#             validation_data=self.valid_generator,
#             validation_steps=self.validation_steps
#         )

#         self.save_model(
#             path=self.config.trained_model_path,
#             model=self.model
#         )


import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig, PrepareBaseModelConfig
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig, base_config:PrepareBaseModelConfig):
        self.config = config
        self.base_config = base_config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        # Reinitialize the optimizer - specify the optimizer with your desired parameters
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_config.params_learning_rate)

        # Compile the model again with the new optimizer
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_valid_generator(self):
        dataflow_kwargs = {
            'label_mode': 'categorical',
            'image_size': self.config.params_image_size[:-1],
            'batch_size': self.config.params_batch_size,
            'interpolation': "bilinear",
            'seed': 123
        }

        valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        train_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            augmentation_layers = [
                tf.keras.layers.Rescaling(1./255),
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomTranslation(0.2, 0.2)
            ]
            train_datagenerator = tf.keras.models.Sequential(augmentation_layers)
            self.train_generator = train_generator.map(
                lambda x, y: (train_datagenerator(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            self.train_generator = train_generator.map(
                lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        self.valid_generator = valid_generator.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        # Ensure GPU utilization with TensorFlow Metal
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        with tf.device('/device:GPU:0'):  # This will use the GPU if available
            self.steps_per_epoch = self.train_generator.cardinality().numpy()
            self.validation_steps = self.valid_generator.cardinality().numpy()

            self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_data=self.valid_generator,
                validation_steps=self.validation_steps
            )

            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )

