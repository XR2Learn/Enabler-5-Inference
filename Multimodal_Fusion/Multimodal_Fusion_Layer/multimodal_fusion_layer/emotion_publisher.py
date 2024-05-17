import json
import time
from random import randint

import numpy as np
import redis

from multimodal_fusion_layer.conf import REDIS_HOST, REDIS_PORT, ID_TO_LABEL, MAPPING_RAVDESS_TO_THEORY_FLOW_DUMMY


# f"{self.modality}_emotion_classification_output_stream
# Make this a Pub/Sub insteado of just publisher
class EmotionPublisher:
    def __init__(self, redis_cli):
        self.redis_cli = redis_cli
        self.pubsub = self.redis_cli.pubsub()
        self.sub_event_types = {
            'shimmer_emotion_classification_output_stream': self.handle_unimodal_emotion_classification
            }

    def publish_emotion(self, label):
        event_type = 'emotion'
        event_data = {
            'emotion': label
        }
        print(event_data)
        self.publish_activity(event_type, event_data)
        # Just for dev
        time.sleep(2)

    def start_activity(self):
        self.subscribe_unimodal_emotion_classification()
        event_type = 'start_activity'
        event_data = {
            'id': 0,
            'user_level': 0,
            'activity_level': randint(0, 50) % 3
        }
        self.publish_activity(event_type, event_data)

    def end_activity(self):
        event_type = 'end_activity'
        event_data = {
            'id': 0,
            'timestamp': time.time()  # 1 is flow
        }
        self.publish_activity(event_type, event_data)

    def publish_activity(self, event_type, event_data):
        json_message = json.dumps(event_data)
        result = self.redis_cli.publish(event_type, json_message)
        return result

    def subscribe_unimodal_emotion_classification(self):
        self.pubsub.subscribe(**self.sub_event_types)
        self.sub_thread = self.pubsub.run_in_thread(sleep_time=0.001)

    def handle_next_activity_level(self, message):
        print(message['data'])
        self.sub_thread.stop()

    def handle_unimodal_emotion_classification(self, message):
        # print(message['data'])
        print('Message receive!\n')
        message_data = json.loads(message['data'])
        print(message_data)
        message_received = message_data['shimmer_emotion_classification_output']
        data_to_publish = process_prediction('XRoom', message_received)
        self.publish_emotion(data_to_publish)


def process_prediction(dataset, prediction_vector):
    prediction_vector = np.array(prediction_vector)
    majority_index = get_majority_voting_index(prediction_vector)
    prediction_label = int(majority_index)
    if dataset == 'RAVDESS':
        prediction_label = ID_TO_LABEL[dataset][majority_index]
        prediction_label = MAPPING_RAVDESS_TO_THEORY_FLOW_DUMMY[prediction_label]
    return prediction_label


def get_majority_voting_index(predictions):
    if len(predictions.shape) <= 1:
        return np.argmax(predictions)
    return np.argmax(np.sum(predictions, axis=0))


if __name__ == '__main__':
    redis_cli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    emotion_publisher = EmotionPublisher(redis_cli)
    time.sleep(5)
    emotion_publisher.start_activity()
    time.sleep(20)
    emotion_publisher.end_activity()
