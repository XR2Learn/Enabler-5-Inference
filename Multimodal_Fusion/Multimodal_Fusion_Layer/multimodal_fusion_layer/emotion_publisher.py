import json
import time
from random import randint

import redis

from multimodal_fusion_layer.conf import REDIS_HOST, REDIS_PORT


class EmotionPublisher:
    def __init__(self, redis_cli):
        self.redis_cli = redis_cli
        self.pubsub = self.redis_cli.pubsub()
        self.sub_event_types = {'next_activity_level': self.handle_next_activity_level
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
        self.subscribe_suggested_activity_level()
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

    def subscribe_suggested_activity_level(self):
        self.pubsub.subscribe(**self.sub_event_types)
        self.sub_thread = self.pubsub.run_in_thread(sleep_time=0.001)

    def handle_next_activity_level(self, message):
        print(message['data'])
        self.sub_thread.stop()


if __name__ == '__main__':
    redis_cli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    emotion_publisher = EmotionPublisher(redis_cli)
    time.sleep(5)
    emotion_publisher.start_activity()
    time.sleep(20)
    emotion_publisher.end_activity()
