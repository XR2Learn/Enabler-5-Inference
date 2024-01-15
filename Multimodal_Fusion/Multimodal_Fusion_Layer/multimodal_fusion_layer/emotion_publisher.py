import json
import time
from random import randint

import redis

from multimodal_fusion_layer.conf import REDIS_HOST, REDIS_PORT


class EmotionPublisher:
    def __init__(self, redis_cli):
        self.redis_cli = redis_cli
        self.pubsub = self.redis_cli.pubsub()
        # self.sub_event_types = {'next_activity_level': self.handle_next_activity_level,
        #                         'debug_considered_emotions': self.handle_debug_considered_emotions}

    def publish_emotion(self, label):
        event_type = 'emotion'
        event_data = {
            'emotion': label
        }
        print(event_data)
        print(self.publish_activity(event_type, event_data))
        time.sleep(2)

    def publish_activity(self, event_type, event_data):
        json_message = json.dumps(event_data)
        result = self.redis_cli.publish(event_type, json_message)
        return result


if __name__ == '__main__':
    pass
    # redis_cli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    #
    # emotion_publisher = EmotionPublisher(redis_cli)
    # for _ in range(3):
    #     emotion_publisher.run()
