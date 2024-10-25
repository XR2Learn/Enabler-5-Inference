import json
import time
from collections import Counter
from random import randint

import numpy as np
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

    def include_modality_data_into_windows(self, session_id, modality, message_received):
        print(f'Process modality Data: {modality}')

        # If current_session_id is empty, start new session
        if not self.current_session_id:
            self.current_session_id = session_id
            self.modality_windows[modality].append(message_received)

        else:
            # right now, only supports dealing with one session_id per time, by receiving a new message of a new
            # session_id, it will clear the window data from all modalities.
            if self.current_session_id != session_id:
                # Clean data here & start new session
                self.clean_window_data(session_id)
                self.modality_windows[modality].append(message_received)

            # increment data here, with information of session already running
            else:
                self.modality_windows[modality].append(message_received)

    def process_fusion_data(self, modality):
        """
        :param modality: the modality from XRoom data
        :return: (List, Boolean) a list with emotion prediction from one modality or the fusion of multimodal and
        if it is ready to publish
        """
        print('Process Fusion Data')
        if not self.is_multimodal:
            # no matter if it is the same session id or not, just get the value of the last element
            # from window and return it
            return self.modality_windows[modality].pop(), True

        else:
            # to merge bm and bt, we need:
            # at least 1 element of bm and 5 elements of bt (change these numbers to be configured and not hot coded)
            if (len(self.modality_windows['shimmer']) > 0) and (len(self.modality_windows['body-tracking']) > 4):
                fused_data = self.execute_fusion_data()
                return fused_data, True
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

        self.include_modality_data_into_windows(session_id, modality, message_received)

        fused_data, is_ready_to_publish = self.process_fusion_data(modality)

        if is_ready_to_publish:
            fused_emotion = process_prediction('XRoom', fused_data)
            self.publish_emotion(fused_emotion)

    def execute_fusion_data(self):
        bm_prediction = self.cut_modality_windows("shimmer", 1)
        bt_prediction_match_window = self.cut_modality_windows('body-tracking', 5)

        bt_prediction = self.combine_xroom_bt_window_prediction(bt_prediction_match_window)

        modalities_prediction = [bm_prediction, bt_prediction]
        fused_data = self.fusion_schema(modalities_prediction)
        return fused_data

    def combine_xroom_bt_window_prediction(self, bt_prediction_match_window):
        """
        Combined different predictions from a modality into one single prediction, to match other modality time window
        :param bt_prediction_match_window: a vector with 5 predictions from the bt modality
        :return: vector with a single prediction for bt modality
        """
        # stopped here
        # do a majority voting for body tracking emotions - see the emotion most present and randomly
        # select only prediction vector of that emotion
        modality_prediction = self.calculate_majority_vote_for_predictions(bt_prediction_match_window)
        return modality_prediction

    def fusion_schema(self, modalities_prediction):
        # check if both prediction has the same dimension vector
        if len(modalities_prediction[0]) < len(modalities_prediction[1]):
            # drop the additional dimension from bt modality
            _ = modalities_prediction[1].pop()
        # do an average in the prediction vector to make the fusion
        fused_data = np.mean(modalities_prediction, axis=0)
        return fused_data

    def calculate_majority_vote_for_predictions(self, bt_prediction_match_window):
        predicted_emotions = []
        for prediction in bt_prediction_match_window:
            predicted_emotions.append(np.argmax(prediction))

        counter = Counter(predicted_emotions)
        # check which emotion is the most present
        most_predicted_emotion = counter.most_common()[0][0]
        # get the first prediction vector of the most present emotion to return
        first_most_voted_prediction = predicted_emotions.index(most_predicted_emotion)
        first_most_voted_prediction = bt_prediction_match_window[first_most_voted_prediction]
        return first_most_voted_prediction

    def cut_modality_windows(self, modality, size):
        prediction = []
        for i in range(size):
            prediction_from_window = self.modality_windows[modality].pop(0)
            prediction.append(prediction_from_window)
        if len(prediction) == 1:
            return prediction[0]
        else:
            return prediction


if __name__ == '__main__':
    redis_cli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    emotion_publisher = FusionPublisherSubscriberXRoomDataset(redis_cli)
    time.sleep(5)
    emotion_publisher.start_activity()
    time.sleep(20)
    emotion_publisher.end_activity()
