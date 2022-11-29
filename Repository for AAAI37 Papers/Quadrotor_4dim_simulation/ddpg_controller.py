"""
This script shows how to run DNNs inference to generate actions.
The observations are from Matlab and transferred via the interface described in bridge.py
After the inference, the actions are sent back to Matlab via the interface described in bridge.py

"""
import os
import pickle
import redis
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DDPGController:
    def __init__(self):
        self.pre_trained_path = 'ddpg_controller_model'
        # self.pre_trained_path = './reindrone_move_test'
        self.agent = tf.keras.models.load_model(self.pre_trained_path)
        self.pool = redis.ConnectionPool(host="192.168.1.101", port="6379", password='ubuntu')
        self.redis_channel_states = 'ch_states'
        self.redis_channel_actions = 'ch_actions'
        self.redis_channel_actions_request = "ch_action_request"

        self.conns = redis.Redis(connection_pool=self.pool)
        self.states_subscriber = self.states_subscriber()
        self.action_request_subscriber = self.action_request_subscriber()

    def generate_action(self):

        while True:
            states = self.subscribe_states()
            states_input = tf.expand_dims(states, axis=0)
            action = self.agent(states_input).numpy().squeeze()
            action_pack = pickle.dumps(action)
            self.conns.publish(channel=self.redis_channel_actions, message=action_pack)

    def sending_action(self):

        while True:
            _ = self.subscibe_action_request()
            action_pack = pickle.dumps(self.action_buffer)
            self.conns.publish(channel=self.redis_channel_actions, message=action_pack)

    def subscribe_states(self):
        states_pack = self.states_subscriber.parse_response()[2]
        states = pickle.loads(states_pack)
        return states

    def states_subscriber(self):
        substates = self.conns.pubsub()
        substates.subscribe(self.redis_channel_states)
        substates.parse_response()
        return substates

    def action_request_subscriber(self):
        subrequest = self.conns.pubsub()
        subrequest.subscribe(self.redis_channel_actions_request)
        subrequest.parse_response()
        return subrequest

    def subscibe_action_request(self):
        request_pack = self.action_request_subscriber.parse_response()[2]
        request = request_pack
        return request

    def run(self):
        self.generate_action()


if __name__ == '__main__':
    ddpg_controller = DDPGController()
    ddpg_controller.run()
