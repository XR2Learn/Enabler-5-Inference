import json
import redis


def init_redis_data_publisher(host, port, modality, logger):
    redis_cli = redis.Redis(host=host, port=port)
    data_publisher = DataPublisher(redis_cli=redis_cli, modality=modality, logger=logger)
    return data_publisher


class DataPublisher:
    def __init__(self, redis_cli, modality, logger) -> None:
        self.redis_cli = redis_cli
        self.pubsub = self.redis_cli.pubsub()
        self.modality = modality
        self.logger = logger

    def publish_data(self, session, data):
        event_type = f"{self.modality}_data_stream"
        event_data = {
            "session_id": session,
            f"{self.modality}_data": data.tolist()
        }
        json_message = json.dumps(event_data)
        self.logger.info(
            "Data to publish:\n"
            f"Session: {session}\n"
            f"Shape: {data.shape}\n"
            f"Channel: {event_type}"
        )
        result = self.redis_cli.publish(event_type, json_message)
        return result
