import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig, PrepareBaseModelConfig
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig, base_config: PrepareBaseModelConfig):
        self.config = config
        self.base_config = base_config
        self.train_generator = None
        self.valid_generator = None
        self.model = None
        self.steps_per_epoch = 0
        self.validation_steps = 0

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_config.params_learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_valid_generator(self):
        dataflow_kwargs = {
            'label_mode': 'categorical',
            'image_size': self.config.params_image_size[:-1],
            'batch_size': self.config.params_batch_size,
            'interpolation': "bilinear",
            'seed': 123
        }

        # Load the training and validation datasets
        train_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Data augmentation for the training set
        if self.config.params_is_augmentation:
            augmentation_layers = [
                tf.keras.layers.Rescaling(1. / 255),
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

        # Scaling the validation set
        self.valid_generator = valid_generator.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Optimize dataset performance
        self.train_generator = self.train_generator.cache().shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)
        self.valid_generator = self.valid_generator.cache().repeat().prefetch(tf.data.AUTOTUNE)

        self.steps_per_epoch = len(train_generator)
        self.validation_steps = len(valid_generator)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        with tf.device('/device:GPU:0'):  # Use GPU if available
            self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_data=self.valid_generator,
                validation_steps=self.validation_steps
            )
            self.save_model(path=self.config.trained_model_path, model=self.model)


# import tensorflow as tf
# from cnnClassifier.entity.config_entity import TrainingConfig, PrepareBaseModelConfig
# from pathlib import Path


# class Training:
#     def __init__(self, config: TrainingConfig, base_config: PrepareBaseModelConfig):
#         self.config = config
#         self.base_config = base_config
#         self.train_generator = None
#         self.valid_generator = None
#         self.model = None
#         self.steps_per_epoch = 0
#         self.validation_steps = 0

#     def get_base_model(self):
#         self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
#         optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_config.params_learning_rate)
#         self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     def train_valid_generator(self):
#         dataflow_kwargs = {
#             'label_mode': 'categorical',
#             'image_size': self.config.params_image_size[:-1],
#             'batch_size': self.config.params_batch_size,
#             'interpolation': "bilinear",
#             'seed': 123
#         }

#         # Load the training and validation datasets
#         train_generator = tf.keras.utils.image_dataset_from_directory(
#             directory=self.config.training_data,
#             validation_split=0.20,
#             subset="training",
#             shuffle=True,
#             **dataflow_kwargs
#         )

#         valid_generator = tf.keras.utils.image_dataset_from_directory(
#             directory=self.config.training_data,
#             validation_split=0.20,
#             subset="validation",
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         # Data augmentation for the training set
#         if self.config.params_is_augmentation:
#             augmentation_layers = [
#                 tf.keras.layers.Rescaling(1. / 255),
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

#         # Scaling the validation set
#         self.valid_generator = valid_generator.map(
#             lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
#             num_parallel_calls=tf.data.AUTOTUNE
#         )

#         # Optimize dataset performance
#         self.train_generator = self.train_generator.cache().shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)
#         self.valid_generator = self.valid_generator.cache().repeat().prefetch(tf.data.AUTOTUNE)

#         self.steps_per_epoch = len(train_generator)
#         self.validation_steps = len(valid_generator)

#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)

#     def train(self):
#         # Callbacks
#         early_stopping = tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss', patience=5, restore_best_weights=True
#         )
#         checkpoint = tf.keras.callbacks.ModelCheckpoint(
#             filepath=str(self.config.trained_model_path).replace(".h5", ".keras"),
#             save_best_only=True,
#             monitor='val_loss',
#             mode='min'
#         )
#         reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7
#         )

#         print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#         with tf.device('/device:GPU:0'):  # Use GPU if available
#             self.model.fit(
#                 self.train_generator,
#                 epochs=self.config.params_epochs,
#                 steps_per_epoch=self.steps_per_epoch,
#                 validation_data=self.valid_generator,
#                 validation_steps=self.validation_steps,
#                 callbacks=[early_stopping, checkpoint, reduce_lr]
#             )
#             self.save_model(path=self.config.trained_model_path, model=self.model)
