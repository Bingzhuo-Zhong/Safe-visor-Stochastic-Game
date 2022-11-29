"""
This script defines the reward function and a distance-based evaluation metric.
"""


import numpy as np


class RewardParams:
    def __init__(self):
        self.distance_score_reward = 0.5
        self.action_penalty = 0.05
        self.crash_penalty = 10
        self.distance_score_factor = 5


class RewardFcn:

    def __init__(self, params: RewardParams):
        self.params = params
        self.reward = self.distance_reward

    def distance_reward(self, drone_position, car_position, terminal, adv_training=False):

        distance_score = self.get_distance_score(drone_position, car_position, self.params.distance_score_factor)

        r = self.params.distance_score_reward * distance_score
        r -= self.params.crash_penalty * terminal

        if adv_training:
            r = -r

        return r

    @staticmethod
    def get_distance_score(drone_postion, car_position, distance_score_factor):
        """
        calculate reward
        :param pole_length: the length of the pole
        :param distance_score_factor: co-efficient of the distance score
        :param observation: [pos, vel, sin_angle, cos_angle, angle_rate]
        :param target: [pos_target, angle_target]
        """

        distance = np.linalg.norm(np.subtract(drone_postion, car_position))
        return np.exp(-distance * distance_score_factor)  # distance [0, inf) -> score [1, 0)
