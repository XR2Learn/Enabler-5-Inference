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

    def test_unimodal_bm_add_remove_from_window(self):
        message = {
            "session_id": "638461160214938655", "modality": "shimmer",
            "emotion_classification_output": [-0.31271907687187195, 0.20892557501792908, 0.07698042690753937]
        }
        self.assertFalse(self.fusion_layer_publisher.modality_windows["shimmer"])
        self.fusion_layer_publisher.process_unimodal_emotion_classification(message)
        self.assertEqual(message['session_id'], self.fusion_layer_publisher.current_session_id)
        self.assertFalse(self.fusion_layer_publisher.modality_windows["shimmer"])

    def test_unimodal_bt_add_remove_from_window(self):
        message = {
            "session_id": "638461160214938655", "modality": "body-tracking",
            "emotion_classification_output": [0.14364197850227356, 0.13539950549602509, 0.7183650135993958,
                                              0.0025935652665793896]
        }
        self.assertFalse(self.fusion_layer_publisher.modality_windows["body-tracking"])
        self.fusion_layer_publisher.process_unimodal_emotion_classification(message)
        self.assertEqual(message['session_id'], self.fusion_layer_publisher.current_session_id)
        self.assertFalse(self.fusion_layer_publisher.modality_windows["body-tracking"])
