"""
This script the interface between Matlab and Python for real-world experiments.
We send information by publishing data to a channel and get information by subscribing a channel in redis-server.
"""


import redis
import pickle

pre_trained_path = 'ddpg_controller_model'
pool = redis.ConnectionPool(host="192.168.1.101", port="6379", password='ubuntu')

redis_channel_states = 'ch_states'
redis_channel_actions = 'ch_actions'
redis_channel_actions_request = "ch_action_request"

conns = redis.Redis(connection_pool=pool)
action_subscriber = conns.pubsub()
action_subscriber.subscribe(redis_channel_actions)
action_subscriber.parse_response()


def send_states_get_actions(x, x_dot, y, y_dot, x_car, x_car_dot, y_car, y_car_dot):

    states = [x, x_dot, y, y_dot, x_car-x, x_car_dot-x_dot, y_car-y, y_car_dot-y_dot]
    states_pack = pickle.dumps(states)
    conns.publish(channel=redis_channel_states, message=states_pack)
    actions_pack = action_subscriber.parse_response()[2]
    actions = pickle.loads(actions_pack) 
    x_res, x_dot_res, y_res, y_dot_res = actions.tolist()
    
    return [x_car+x_res, x_car_dot+x_dot_res, y_car+y_res, y_car_dot+y_dot_res]


def get_actions(x_car, x_car_dot, y_car, y_car_dot):
    """
    Here the actions are the drone set-points residuals [x_res, x_dot_res, y_res, y_dot_res]
    """
    conns.publish(channel=redis_channel_actions_request, message='1')
    actions_pack = action_subscriber.parse_response()[2]
    actions = pickle.loads(actions_pack)  
    x_res, x_dot_res, y_res, y_dot_res = actions.tolist()
    
    return [x_car+x_res, x_car_dot+x_dot_res, y_car+y_res, y_car_dot+y_dot_res]
