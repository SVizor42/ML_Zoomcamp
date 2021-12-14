import yaml
import logging
import sys
from dataclasses import dataclass
from .train_params import TrainingParams
from .image_params import ImageParams
from marshmallow_dataclass import class_schema
from marshmallow import ValidationError


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@dataclass()
class TrainingPipelineParams:
    data_dir: str
    model_dir: str
    image_params: ImageParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    try:
        with open(path, "r") as input_stream:
            schema = TrainingPipelineParamsSchema()
            return schema.load(yaml.safe_load(input_stream))
    except FileNotFoundError:
        logger.error(f"Can't load training parameters. File not found:{path}")
    except ValidationError as err:
        logger.error(f"Can't load training parameters. {err}")
