"""
We build the Drone system using Open-AI gym library.
"""


import math
import gym
from gym.utils import seeding
import numpy as np
from copy import deepcopy
import pandas as pd


class DroneGridParams:
    """The physical parameters for building the system"""
    def __init__(self):
        self.x_length = 2.5
        self.y_length = 2.5
        self.v_limit = 2.0
        self.kinematics_integrator = 'euler'
        self.simulation_frequency = 10
        self.actuation_delay = 1
        self.dim_states_car = 4
        self.dim_states_drone = 6
        self.drone_ax_sat = 2.5
        self.drone_ay_sat = 2.5
        self.drone_az_sat = 1
        self.car_ax_sat = 10
        self.car_ay_sat = 10


class DroneGrid(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30  # this is not working
    }
    """
        states: [x, y]
        action: set-point [x, y]      
    """

    def __init__(self, params: DroneGridParams):

        self.params = params

        self.seed(1)
        self.viewer = None
        self.tau = 1.0 / self.params.simulation_frequency

        # The parameters to describe the dynamics of drone and car are from system modeling.

        self.drone_A = np.array([[1, self.tau, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [0, 0, 1, self.tau, 0, 0],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, self.tau],
                                 [0, 0, 0, 0, 0, 1]]).astype(np.float)

        self.drone_B = np.array([[0.5 * self.tau ** 2, 0, 0],
                                 [self.tau, 0, 0],
                                 [0, 0.5 * self.tau ** 2, 0],
                                 [0, self.tau, 0],
                                 [0, 0, 0.5 * self.tau ** 2],
                                 [0, 0, self.tau]]).astype(np.float)

        self.drone_K = np.array([[1.4781, 1.7309, 0, 0, 0, 0],
                                 [0, 0, 1.4781, 1.7309, 0, 0],
                                 [0, 0, 0, 0, 9.9107, 4.4631]]).astype(np.float)

        self.car_A = np.array([[1, self.tau, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, self.tau],
                               [0, 0, 0, 1]])

        self.car_B = np.array([[0.5 * self.tau ** 2, 0],
                               [self.tau, 0],
                               [0, 0.5 * self.tau ** 2],
                               [0, self.tau]])

        self.car_states = np.zeros(shape=self.params.dim_states_car)
        self.drone_states = np.zeros(shape=self.params.dim_states_drone)
        self.drone_states_pre = [0.0, 0.0]
        self.bg_states = [0.5 * self.params.x_length, 0.5 * self.params.y_length]
        self.car_traj = None
        self.len_traj = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action: list):
        """
        This function describes how the system evolves.
        The system takes the set-points input, and the low-level controllers yield acceleration outputs, which will be
        used for updating system states.
        params: action [[x_d, x_dot_d, y_c, y_dot_d], [x_c, x_dot_c, y_c, y_dot_c]]
                the position and velocity setpoints for car and drone.
        """
        drone_set_points = np.hstack((action[0], [0, 0]))

        drone_states_next = self.dynamics_drone_evolve(self.drone_states, drone_set_points)

        car_target = np.array(action[1])
        car_states_next = self.dynamics_car_evolve_as_drone(self.car_states, car_target)

        back_ground_states_next = self.dynamics_background_evolve(self.drone_states, drone_states_next)
        self.car_states = car_states_next
        self.drone_states = drone_states_next
        self.bg_states = back_ground_states_next
        failed = self.is_failed(self.drone_states)
        return drone_states_next, car_states_next, failed

    def reset(self, car_re_states=None, drone_re_state=None):
        """reset the states of car and drones"""
        if car_re_states is None:
            self.car_states = np.zeros(shape=self.params.dim_states_car)
        else:
            self.car_states = car_re_states

        if drone_re_state is None:
            self.drone_states = np.zeros(shape=self.params.dim_states_drone)
        else:
            self.drone_states = drone_re_state

        self.bg_states = [0.5 * self.params.x_length, 0.5 * self.params.y_length]

    def random_reset(self):
        ran_x = np.random.uniform(low=-1 * self.params.x_length, high=self.params.x_length)
        ran_y = np.random.uniform(low=-1 * self.params.y_length, high=self.params.y_length)
        # ran_x_dot = np.random.uniform(low=-1.0, high=1.0)
        # ran_y_dot = np.random.uniform(low=-1.0, high=1.0)

        ran_x_car = np.random.uniform(low=-1 * self.params.x_length, high=self.params.x_length)
        ran_y_car = np.random.uniform(low=-1 * self.params.y_length, high=self.params.y_length)
        # ran_x_dot_car = np.random.uniform(low=-1.0, high=1.0)
        # ran_y_dot_car = np.random.uniform(low=-1.0, high=1.0)

        self.drone_states = np.array([ran_x, 0, ran_y, 0, 0., 0.])
        self.car_states = np.array([ran_x_car, 0, ran_y_car, 0])
        self.bg_states = [0.5 * self.params.x_length, 0.5 * self.params.y_length]

    def render(self, mode='human', drone_states=None, car_states=None, is_normal_operation=True):
        """ Rendering scene use openAI-gym to visualize the simulation """

        screen_width = 600
        screen_height = 600
        world_width = self.params.x_length * 2
        scale = screen_width / (world_width + 1)
        drone_width = 30
        drone_height = 30
        car_width = 30
        car_height = 30
        x_offset = 0.5 * screen_width
        y_offset = 0.5 * screen_height

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # background
            background = rendering.Image('./image/background.png', width=screen_width * 2.5, height=screen_height * 2.5)
            self.background_trans = rendering.Transform()
            background.add_attr(self.background_trans)
            self.viewer.add_geom(background)

            # drone
            self.drone_trans = rendering.Transform()
            drone_1 = rendering.Image('./image/drone.png', width=drone_width, height=drone_height)
            drone_1.add_attr(self.drone_trans)
            self.viewer.add_geom(drone_1)

            # car
            self.car_trans = rendering.Transform()
            car = rendering.Image('./image/marker.png', width=car_width, height=car_height)
            car.add_attr(self.car_trans)
            self.viewer.add_geom(car)
            margin = (screen_width - world_width * scale) * 0.5
            self.bound_1 = rendering.Line((margin, margin), (screen_width - margin, margin))
            self.bound_2 = rendering.Line((margin, margin), (margin, screen_height - margin))
            self.bound_3 = rendering.Line((margin, screen_height - margin),
                                          (screen_width - margin, screen_height - margin))
            self.bound_4 = rendering.Line((screen_width - margin, margin),
                                          (screen_width - margin, screen_height - margin))

            color_r = 201 / 255.0
            color_g = 77 / 255.0
            color_b = 52 / 255.0

            self.bound_1.set_color(color_r, color_g, color_b)
            self.bound_2.set_color(color_r, color_g, color_b)
            self.bound_3.set_color(color_r, color_g, color_b)
            self.bound_4.set_color(color_r, color_g, color_b)

            self.viewer.add_geom(self.bound_1)
            self.viewer.add_geom(self.bound_2)
            self.viewer.add_geom(self.bound_3)
            self.viewer.add_geom(self.bound_4)

        if drone_states is None:
            if self.drone_states is None:
                return None
            else:
                drone_states = deepcopy(self.drone_states)
                car_states = deepcopy(self.car_states)

        bg_states = self.bg_states
        self.drone_trans.set_translation(drone_states[0] * scale + x_offset, drone_states[2] * scale + y_offset)
        self.car_trans.set_translation(car_states[0] * scale + x_offset, car_states[2] * scale + y_offset)
        self.background_trans.set_translation(bg_states[0] * scale + x_offset, bg_states[1] * scale + y_offset)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def is_failed(self, drone_states):
        """The function used for determining the crash of the drone """
        x = drone_states[0]
        y = drone_states[2]
        failed = bool(math.fabs(x) >= self.params.x_length
                      or math.fabs(y) >= self.params.y_length)
        return failed

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dynamics_drone_evolve(self, drone_states, drone_target_states):
        """The system model of Drone """
        distance = drone_target_states - drone_states
        u_k = self.drone_K @ distance
        a_x, a_y, a_z = u_k

        a_x_sat = np.clip(a_x, a_min=-1. * self.params.drone_ax_sat, a_max=self.params.drone_ax_sat)
        a_y_sat = np.clip(a_y, a_min=-1. * self.params.drone_ay_sat, a_max=self.params.drone_ay_sat)
        a_z_sat = np.clip(a_z, a_min=-1. * self.params.drone_az_sat, a_max=self.params.drone_az_sat)

        u_k_sat = np.array([a_x_sat, a_y_sat, a_z_sat])
        drone_states_next = self.drone_A @ drone_states + self.drone_B @ u_k_sat
        return drone_states_next

    def dynamics_car_evolve_as_drone(self, car_states, car_target_states):
        """We simulate the car's dynamics by using drone's model,
        which simplify the trajectory generation in the training, since the PID parameters are tuned"""
        drone_like_car_states = np.append(car_states, values=[0, 0])
        car_target_states = np.array([car_target_states[0], 0, car_target_states[1], 0])
        drone_like_car_targets_states = np.append(car_target_states, values=[0, 0])
        distance = drone_like_car_targets_states - drone_like_car_states

        u_k = self.drone_K @ distance
        a_x, a_y, a_z = u_k

        a_x_sat = np.clip(a_x, a_min=-1. * self.params.drone_ax_sat, a_max=self.params.drone_ax_sat)
        a_y_sat = np.clip(a_y, a_min=-1. * self.params.drone_ay_sat, a_max=self.params.drone_ay_sat)
        a_z_sat = np.clip(a_z, a_min=-1. * self.params.drone_az_sat, a_max=self.params.drone_az_sat)

        u_k_sat = np.array([a_x_sat, a_y_sat, a_z_sat])
        drone_like_car_states_next = self.drone_A @ drone_like_car_states + self.drone_B @ u_k_sat
        car_states_next = drone_like_car_states_next[:4]
        return car_states_next

    def dynamics_car_evolve(self, car_states, axy_input):
        """We can also directly use car's model and use PID controller to calculate the acceleration input.
            However, we need to tune the parameters of PID """
        car_states_next = self.car_A @ car_states + self.car_B @ axy_input
        return car_states_next

    def dynamics_background_evolve(self, drone_states, drone_states_next):
        """Adding relative movement of the background to make the simulation visually more realistic"""
        delta_x = drone_states_next[0] - drone_states[0]
        delta_y = drone_states_next[2] - drone_states[2]

        new_x = self.bg_states[0] - delta_x * 0.5
        new_y = self.bg_states[1] - delta_y * 0.5
        back_ground_states_next = [new_x, new_y]
        return back_ground_states_next

    @staticmethod
    def car_on_target(car_states, trajectory_point):
        x_distance = math.fabs(car_states[0] - trajectory_point[0])
        y_distance = math.fabs(car_states[1] - trajectory_point[1])
        if x_distance <= 0.1 and y_distance <= 0.1:
            return True
        else:
            return False

    def random_trajectory_generator(self, num_points_x=3, num_points_y=3):
        """
        To randomly generate the trajectory, we randomly sample the setpoints for car in the field.
        During training, the car will take these random setpoints as input and move correspondingly.
        """
        trajectory = []
        len_trajectory = int(num_points_x * num_points_y)

        x_interval_length = 2 * self.params.x_length / num_points_x
        y_interval_length = 2 * self.params.y_length / num_points_y

        for i in range(num_points_x):
            x = np.random.uniform() * (i + 1) * x_interval_length - self.params.x_length
            for j in range(num_points_y):
                y = np.random.uniform() * (j + 1) * y_interval_length - self.params.y_length
                trajectory.append([x, y])
        np.random.shuffle(trajectory)
        return trajectory, len_trajectory

    @staticmethod
    def get_clover_trajectory():
        """We load ground truth trajectory for evaluation purpose"""
        # path = "clover.xlsx"
        path = "car_setpoint.xlsx"
        data_source = pd.read_excel(path)
        points = data_source.to_numpy()
        points = points[:, :, np.newaxis]
        car_trajectory = np.concatenate(points, axis=-1)
        len_trajectory = car_trajectory.shape[0]
        return car_trajectory, len_trajectory
