"""
This script defines a monitor to log and visualize the training/testing performance.
"""


import os
import io
import datetime

import pandas as pd
import tensorflow as tf
import numpy as np
import shutil
import distutils.util
import matplotlib.pyplot as plt
from rein.env.drone_grid import DroneGrid
from rein.utils import states2observations
from rein.env.reward import RewardFcn


class ModelStatsParams:
    def __init__(self):
        self.max_episode_steps = 1000
        self.global_train_steps = int(1e7)
        self.agent_training_steps = int(2e5)
        self.agent_adv_training_steps = int(2e5)
        self.target_distance_score = 0.77880078307  # 5 cm distance from the target tape
        self.model_name = "model_name"
        self.eval_period = 20
        self.log_file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.force_override = False
        self.weights_path = None
        self.adv_weights_path = None
        self.running_mode = 'train'
        self.random_initial_ips = True
        self.visualize_eval = False
        self.reset_delay = 1.0
        self.converge_episodes = 20
        self.train_adv = True
        self.can_track_steps = 750


class ModelStats:

    def __init__(self, params: ModelStatsParams, physics: DroneGrid):
        self.params = params
        self.visualize_eval = self.params.visualize_eval
        self.physics = physics
        self.reward = []
        self.observations = []
        self.failed = None
        self.survived = True
        self.distance_scores = []
        self.drone_position = []
        self.drone_actions = []
        self.control_agent_actions = []
        self.adv_agent_actions = []
        self.car_position = []
        self.total_train_steps = 0
        self.control_agent_train_steps = 0
        self.adv_agent_train_steps = 0
        self.on_target_steps = 0
        self.average_distance_score_and_survive = 0
        self.consecutive_on_target_steps = 0
        self.critic_losses = []

        self.drone_position_dp = []

        self.log_dir = 'logs/' + self.params.log_file_name
        self.clear_cache()

        self.training_log_writer = tf.summary.create_file_writer(self.log_dir + '/training')
        self.evaluation_log_writer = tf.summary.create_file_writer(self.log_dir + '/eval')

        if self.params.train_adv:
            self.adv_training_log_writer = tf.summary.create_file_writer(self.log_dir + '/adv_training')
            self.control_training_log_writer = tf.summary.create_file_writer(self.log_dir + '/control_training')

    def init_episode(self):
        if self.params.random_initial_ips:
            self.physics.random_reset()
        else:
            self.physics.reset()

        self.reset_status()

    def reset_status(self):
        self.reward = []
        self.observations = []
        self.failed = False
        self.distance_scores = []
        self.on_target_steps = 0
        self.consecutive_on_target_steps = 0
        self.drone_position = []
        self.car_position = []
        self.drone_actions = []
        self.control_agent_actions = []
        self.adv_agent_actions = []
        self.average_distance_score_and_survive = 0
        self.survived = True
        self.critic_losses = []
        self.drone_position_dp = []

    def get_average_reward(self):
        if len(self.reward) == 0:
            return 0
        else:
            return sum(self.reward) / len(self.reward)

    def get_average_distance_score(self):
        if len(self.distance_scores) == 0:
            return 0
        else:
            return sum(self.distance_scores) / len(self.distance_scores)

    def add_critic_loss(self, loss):
        self.critic_losses.append(loss)

    def measure(self, drone_position, car_position, failed, distance_score_factor):

        distance_score = RewardFcn.get_distance_score(drone_position, car_position, distance_score_factor)

        if distance_score > self.params.target_distance_score:
            self.on_target_steps += 1
            self.consecutive_on_target_steps += 1
        else:
            self.consecutive_on_target_steps = 0

        self.distance_scores.append(distance_score)

        if failed:
            self.survived = False

    def get_steps(self):
        return len(self.reward)

    def training_monitor(self, episode, training_log_writer, current_steps):

        average_reward, on_target_steps, average_distance_score, survived, can_track, ads_sur, critic_loss = self.log_data()

        with training_log_writer.as_default():
            tf.summary.scalar('average_Reward', average_reward, current_steps)
            tf.summary.scalar('on_target_step', on_target_steps, current_steps)
            tf.summary.scalar('can_track', can_track, current_steps)
            tf.summary.scalar('distance_score', average_distance_score, current_steps)
            tf.summary.scalar('distance_score_and_survived', ads_sur, current_steps)
            tf.summary.scalar('critic_loss',critic_loss, current_steps)

        print("Training:=====>  Episode: ", episode, " Total steps:",
              self.get_steps(), " Average_reward: ", average_reward, "ds_mean", average_distance_score)

    def evaluation_monitor_scalar(self, episode, evaluation_log_writer, current_steps):

        average_reward, on_target_steps, average_distance_score, survived, can_track, ads_sur, _ = self.log_data()

        with evaluation_log_writer.as_default():
            tf.summary.scalar('average_Reward', average_reward, current_steps)
            tf.summary.scalar('on_target_step', on_target_steps, current_steps)
            tf.summary.scalar('can_track', can_track, current_steps)
            tf.summary.scalar('distance_score', average_distance_score, current_steps)
            tf.summary.scalar('distance_score_and_survived', ads_sur, current_steps)

        print("Evaluation:=====>  Episode: ", episode, " Total steps:",
              self.get_steps(), " Average_reward: ", average_reward, "ds_mean", average_distance_score)

    def evaluation_monitor_image(self, ep, evaluation_log_writer, current_steps):

        average_reward, on_target_steps, average_distance_score, survived, can_track, _, _ = self.log_data()
        tf_summary_plot_traj = self.plot_traj_to_image(average_reward, on_target_steps, average_distance_score)
        tf_summary_plot_actions_drone = self.plot_drone_actions_to_image()

        with evaluation_log_writer.as_default():
            tf.summary.image('eval_{}/summary_plot_traj'.format(ep), tf_summary_plot_traj, current_steps)
            tf.summary.image('eval_{}/summary_plot_actions_drone'.format(ep), tf_summary_plot_actions_drone,
                             current_steps)

    def get_average(self, data):
        if len(data) == 0:
            return 0
        else:
            return sum(data) / len(data)

    def log_data(self):
        critic_loss = self.get_average(self.critic_losses)
        average_reward = self.get_average_reward()
        on_target_steps = self.on_target_steps
        average_distance_score = self.get_average_distance_score()
        survived = self.get_survived()
        can_track = self.consecutive_on_target_steps >= self.params.can_track_steps
        ads_sur = average_distance_score * survived
        self.average_distance_score_and_survive = ads_sur
        return average_reward, on_target_steps, average_distance_score, survived, can_track, ads_sur, critic_loss

    def plot_traj_to_image(self, average_reward, on_target_steps, average_distance_score):
        figure = plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(np.array(self.car_position)[:, 0], np.array(self.car_position)[:, 1], label='Car_position', c='k')
        plt.plot(np.array(self.drone_position)[:, 0], np.array(self.drone_position)[:, 1], label='Drone_position')
        plt.plot(np.array(self.drone_position_dp)[:, 0], np.array(self.drone_position_dp)[:, 1], label='Drone_position_dp')
        points_x = np.array(self.drone_position)[:601, 0]
        points_y = np.array(self.drone_position)[:601, 1]

        # df = pd.DataFrame(np.vstack((points_x, points_y)))
        # df.to_excel('drone_trajectory.xlsx')

        label = 'Average_reward: {:.2} \n On_target_steps: ' \
                '{} \n average_distance_score: {:.2}'.format(average_reward, on_target_steps, average_distance_score)

        plt.scatter(self.car_position[0][0], self.car_position[0][1], label='Car_Start', marker="s", c='g', s=100,
                    zorder=2)
        plt.scatter(self.car_position[-1][0], self.car_position[-1][1], label='Car_End', marker="*", c='g', s=100,
                    zorder=2)
        plt.scatter(self.drone_position[0][0], self.drone_position[0][1], label='Drone_Start', marker="s", c='b', s=100,
                    zorder=2)
        plt.scatter(self.drone_position[-1][0], self.drone_position[-1][1], label='Drone_End', marker="*", c='b', s=100,
                    zorder=2)
        plt.scatter(self.drone_position_dp[0][0], self.drone_position_dp[0][1], label='Drone_Start_dp', marker="s",
                    c='m', s=100, zorder=2)
        plt.scatter(self.drone_position_dp[-1][0], self.drone_position_dp[-1][1], label='Drone_End_dp', marker="*",
                    c='m', s=100, zorder=2)

        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])

        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        plt.legend(loc='best', fontsize='x-small')
        plt.grid(True)

        figure.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def plot_drone_actions_to_image(self):
        figure = plt.figure()

        x = self.get_steps()

        plt.subplot(4, 1, 1)
        plt.xlabel('steps')
        plt.ylabel('x-m')
        plt.plot(np.arange(x), np.array(self.drone_actions)[:, 0], label='Drone x position')
        plt.legend(loc='best', fontsize='small')
        plt.ylim([-2.5, 2.5])
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.xlabel('steps')
        plt.ylabel('x-m/s')
        plt.plot(np.arange(x), np.array(self.drone_actions)[:, 1], label='Drone x velocity')
        plt.legend(loc='best', fontsize='small')
        plt.ylim([-2.0, 2.0])
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.xlabel('steps')
        plt.ylabel('y-m')
        plt.plot(np.arange(x), np.array(self.drone_actions)[:, 2], label='Drone y position')
        plt.legend(loc='best', fontsize='small')
        plt.ylim([-2.5, 2.5])
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.xlabel('steps')
        plt.ylabel('y-m/s')
        plt.plot(np.arange(x), np.array(self.drone_actions)[:, 3], label='Drone y velocity')
        plt.legend(loc='best', fontsize='small')
        plt.ylim([-2.0, 2.0])
        plt.grid(True)
        figure.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def add_steps(self, step):
        self.total_train_steps += step

    def get_survived(self):
        return self.survived

    def clear_cache(self):
        if os.path.isdir(self.log_dir):
            if self.params.force_override:
                shutil.rmtree(self.log_dir)
            else:
                print(self.log_dir, 'already exists.')
                resp = input('Override log file? [Y/n]\n')
                if resp == '' or distutils.util.strtobool(resp):
                    print('Deleting old log dir')
                    shutil.rmtree(self.log_dir)
                else:
                    print('Okay bye')
                    exit(1)
