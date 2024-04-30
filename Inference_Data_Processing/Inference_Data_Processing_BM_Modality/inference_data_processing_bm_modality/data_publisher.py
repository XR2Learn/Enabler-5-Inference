import json
import redis


def init_redis_data_publisher(host, port, modality):
    redis_cli = redis.Redis(host=host, port=port)
    data_publisher = DataPublisher(redis_cli=redis_cli, modality=modality)
    return data_publisher


class DataPublisher:
    def __init__(self, redis_cli, modality) -> None:
        self.redis_cli = redis_cli
        self.pubsub = self.redis_cli.pubsub()
        self.modality = modality
        self.sub_event_types = {
            # do we expect something here if we read from CSVs?
        }

    def publish_data(self, data):
        event_type = f"{self.modality}_data_stream"
        event_data = {
            f"{self.modality}_data": data.tolist()
        }
        json_message = json.dumps(event_data)
        result = self.redis_cli.publish(event_type, json_message)
        return result
