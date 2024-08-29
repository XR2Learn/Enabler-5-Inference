import unittest

import redis

from multimodal_fusion_layer.conf import REDIS_HOST, REDIS_PORT
from multimodal_fusion_layer.fusion_pub_sub import FusionPublisherSubscriberXRoomDataset
from multimodal_fusion_layer.predict import init_logger


class FusionPublisherSubscriberXRoomDatasetTestCase(unittest.TestCase):
    def setUp(self):
        logger = init_logger()
        self.redis_cli = redis.Redis(port=REDIS_PORT, host=REDIS_HOST)
        self.fusion_layer_publisher = FusionPublisherSubscriberXRoomDataset(self.redis_cli, logger)

    def tearDown(self):
        pass

    def test_unimodal_fusion_same_session_id(self):
        message = {
            "session_id": "", "modality": "", "emotion_classification_output": []
        }
        self.fusion_layer_publisher.process_unimodal_emotion_classification(message)
