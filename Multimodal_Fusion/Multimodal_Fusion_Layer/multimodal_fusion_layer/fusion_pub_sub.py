import json
import time
from random import randint

import redis

from multimodal_fusion_layer.conf import REDIS_HOST, REDIS_PORT
from multimodal_fusion_layer.fusion_schema import process_prediction


class FusionPublisherSubscriberXRoomDataset:
    def __init__(self, redis_cli, logger, is_multimodal=False):
        self.redis_cli = redis_cli
        self.pubsub = self.redis_cli.pubsub()
        self.sub_event_types = {
            'emotion_classification_output_stream': self.handle_unimodal_emotion_classification
        }
        self.logger = logger
        self.current_session_id = None
        self.is_multimodal = is_multimodal
        self.modality_windows = {'shimmer': [], 'body-tracking': []}
        # self.bm_window = []

    def publish_emotion(self, emotion_index):
        event_type = 'emotion'
        event_data = {
            'emotion': emotion_index
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
        self.logger.info(f"Message: {message['data']}")
        self.sub_thread.stop()

    def handle_unimodal_emotion_classification(self, message):
        # print(message['data'])
        self.logger.info("Message received!")
        message_data = json.loads(message['data'])
        self.logger.info(f"{message_data}")

        self.process_unimodal_emotion_classification(message_data)

    def process_modality_data(self, session_id, modality, message_received):
        print('Process modality Data')

        # If current_session_id is empty, start new session
        if not self.current_session_id:
            self.current_session_id = session_id
            self.modality_windows[modality].append(message_received)

        else:
            if self.current_session_id != session_id:
                # Clean data here & start new session
                self.clean_window_data(session_id)
                self.modality_windows[modality].append(message_received)

            # increment data here, with information of session already running
            else:
                self.modality_windows[modality].append(message_received)

    def process_fusion_data(self, modality):
        print('Process Fusion Data')
        if not self.is_multimodal:
            return self.modality_windows[modality].pop(), True
        else:
            return [], False

    def clean_window_data(self, session_id):
        self.current_session_id = session_id
        for modality in self.modality_windows.keys():
            self.modality_windows[modality] = []

    def process_unimodal_emotion_classification(self, message_data):
        session_id = message_data['session_id']
        modality = message_data['modality']
        message_received = message_data[f'emotion_classification_output']

        self.process_modality_data(session_id, modality, message_received)

        fused_data, is_ready_to_publish = self.process_fusion_data(modality)

        if is_ready_to_publish:
            fused_emotion = process_prediction('XRoom', fused_data)
            self.publish_emotion(fused_emotion)


if __name__ == '__main__':
    redis_cli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    emotion_publisher = FusionPublisherSubscriberXRoomDataset(redis_cli)
    time.sleep(5)
    emotion_publisher.start_activity()
    time.sleep(20)
    emotion_publisher.end_activity()
