import json
import redis

import numpy as np
from emotion_classification_body_tracking_modality.inference_functions import make_prediction_from_numpy
from emotion_classification_body_tracking_modality.conf import PUBLISHER_ON


def init_redis_emocl_pubsub(
        host,
        port,
        modality,
        channel,
        data_header,
        model,
        logger,
        id_to_emotion
):
    redis_cli = redis.Redis(host=host, port=port)
    emocl_pubsub = EmotionClassificationPubSub(
        redis_cli=redis_cli,
        modality=modality,
        channel=channel,
        data_header=data_header,
        model=model,
        logger=logger,
        id_to_emotion=id_to_emotion
    )
    return emocl_pubsub


class EmotionClassificationPubSub:
    def __init__(self, redis_cli, modality, channel, data_header, model, logger, id_to_emotion):
        self.redis_cli = redis_cli
        self.pubsub = self.redis_cli.pubsub()

        assert modality in channel, "Modality is not mentioned in the channel"
        self.modality = modality
        self.channel = channel
        self.output_event_type = "emotion_classification_output_stream"
        self.data_header = data_header
        self.model = model
        self.logger = logger
        self.id_to_emotion = id_to_emotion

        self.sub_event_types = {
            channel: self.handle_data_stream
        }

    def start_processing(self):
        self.subscribe_data_stream()

    def publish_model_output(self, session, model_output, emotion):
        event_data = {
            "session_id": session,
            "modality": self.modality,
            f"emotion_classification_output": model_output.tolist(),
            f"emotion_classification_detected_emotion": emotion
        }
        json_message = json.dumps(event_data)
        result = self.redis_cli.publish(self.output_event_type, json_message)
        return result

    def subscribe_data_stream(self):
        self.pubsub.subscribe(**self.sub_event_types)
        self.sub_thread = self.pubsub.run_in_thread(sleep_time=0.001)

    def handle_data_stream(self, message):
        data = json.loads(message["data"])
        session = data["session_id"]
        self.logger.info(f"Received a message from {self.channel} channel, {session} session")
        data_window = np.expand_dims(np.array(data[self.data_header]), axis=0)
        self.logger.info(f"Making prediction for received data of shape: {data_window.shape}")
        model_output = make_prediction_from_numpy(data_window, self.model)
        predicted_emotion = self.id_to_emotion[np.argmax(model_output)]
        self.logger.info(f"Generated output shape: {model_output.shape}")
        self.logger.info(f"Predicted emotion: {predicted_emotion}")
        self.logger.info(f"Publishing output to {self.output_event_type}")
        self.publish_model_output(session, model_output, predicted_emotion)
