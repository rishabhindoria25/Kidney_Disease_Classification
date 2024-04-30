# import tensorflow as tf
# from pathlib import Path
# import mlflow
# import mlflow.keras
# from urllib.parse import urlparse
# from cnnClassifier.entity.config_entity import EvaluationConfig
# from cnnClassifier.utils.common import read_yaml, create_directories,save_json


# class Evaluation:
#     def __init__(self, config: EvaluationConfig):
#         self.config = config

    
#     def _valid_generator(self):

#         datagenerator_kwargs = dict(
#             rescale = 1./255,
#             validation_split=0.30
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


#     @staticmethod
#     def load_model(path: Path) -> tf.keras.Model:
#         return tf.keras.models.load_model(path)
    

#     def evaluation(self):
#         self.model = self.load_model(self.config.path_of_model)
#         self._valid_generator()
#         self.score = self.model.evaluate(self.valid_generator)
#         self.save_score()

#     def save_score(self):
#         scores = {"loss": self.score[0], "accuracy": self.score[1]}
#         save_json(path=Path("scores.json"), data=scores)

    
#     def log_into_mlflow(self):
#         mlflow.set_registry_uri(self.config.mlflow_uri)
#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
#         with mlflow.start_run():
#             mlflow.log_params(self.config.all_params)
#             mlflow.log_metrics(
#                 {"loss": self.score[0], "accuracy": self.score[1]}
#             )
#             # Model registry does not work with file store
#             if tracking_url_type_store != "file":

#                 # Register the model
#                 # There are other ways to use the Model Registry, which depends on the use case,
#                 # please refer to the doc for more information:
#                 # https://mlflow.org/docs/latest/model-registry.html#api-workflow
#                 mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
#             else:
#                 mlflow.keras.log_model(self.model, "model")

import tensorflow as tf
from cnnClassifier.entity.config_entity import EvaluationConfig
from pathlib import Path
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
import mlflow

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        dataflow_kwargs = {
            'label_mode': 'categorical',
            'image_size': self.config.params_image_size[:-1],
            'batch_size': self.config.params_batch_size,
            'interpolation': "bilinear",
            'seed': 123  # Consistency in data shuffling
        }

        valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.30,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Normalization and preparation of the dataset pipeline
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        self.valid_generator = valid_generator.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE  # Use AUTOTUNE to dynamically manage parallelism
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        # Evaluate using all available GPU resources, if necessary
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        with tf.device('/device:GPU:0'):  # This will use the GPU if available
            self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")

