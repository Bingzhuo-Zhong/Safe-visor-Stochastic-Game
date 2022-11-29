"""
In this script we build the whole system DrnSystem.
DrnSystem includes:
1. Physics: We describe the system dynamics in DroneGrid
2. Agent: we define the structure and parameters of DNNs.
3. Trainer: In the trainer, we describe sampling strategy/loss/gradients/ calculation etc.
3. Reward function: We calculate the reward function in the interaction loop instead of in Physics.
4. Monitor: We define the evaluation metrics here. And use it for logging training and testing processes.

"""

import numpy as np
import copy
from rein.monitor.monitor import ModelStatsParams, ModelStats
from rein.env.drone_grid import DroneGrid, DroneGridParams
from rein.env.reward import RewardParams, RewardFcn
from simple_pid import PID
import matplotlib.pyplot as plt


class DrnSystemParams:
    def __init__(self):
        self.physics_params = DroneGridParams()
        self.reward_params = RewardParams()
        self.stats_params = ModelStatsParams()
        self.agent_params = None


class DrnSystem:
    def __init__(self, params: DrnSystemParams):
        self.params = params

        self.physics = DroneGrid(self.params.physics_params)
        self.model_stats = ModelStats(self.params.stats_params, self.physics)
        self.reward_fcn = RewardFcn(self.params.reward_params)
        self.shape_observations_drone = 4 + 4  # 4 + 10
        self.trainer = None
        self.agent = None
        self.pid_x = PID(Kp=1.47, Kd=0.5, setpoint=0, output_limits=(-0.6, 0.6))
        self.pid_y = PID(Kp=1.47, Kd=0.5, setpoint=0, output_limits=(-0.6, 0.6))
        self.car_still = False  # False for car staying the initial position
        self.car_start_origin = True  # Ture for starting at the origin [0, 0] of the map.
        self.use_guide_setpoints = True  # True for using the car's states to calculate the final setpoints.
                                         # If true, the agent only outputs the setpoints residuals
        self.use_relative_states = True  # If ture, we take the car's relative positions to drone as the car's observations.
        self.add_observations_noise = False  # If ture, we add noise to the observations to increase the agent's robustness

    def evaluation_episode(self, agent, ep=1):
        """ We evaluate the performance of agent on the clover trajectory """
        setpoints_list = []
        trajs, len_traj = self.physics.get_clover_trajectory()

        self.physics.car_traj = trajs
        self.physics.len_traj = len_traj
        traj_index = 0

        self.model_stats.init_episode()

        if self.car_start_origin:
            self.physics.car_states = np.zeros(shape=self.physics.params.dim_states_car)

        initial_states = [copy.deepcopy(self.physics.drone_states), copy.deepcopy(self.physics.car_states)]

        self.physics.reset()  # reset all to origin

        self.initialize_observations(self.use_relative_states)

        if agent.add_actions_observations:
            action_observations = np.zeros(shape=agent.action_observations_dim)
        else:
            action_observations = []

        for step in range(self.params.stats_params.max_episode_steps):

            if self.params.stats_params.visualize_eval:
                self.physics.render()
                # time.sleep(0.1)

            observations = np.hstack((self.model_stats.observations, action_observations))

            car_position = [self.physics.car_states[0], self.physics.car_states[2]]

            drone_position = [self.physics.drone_states[0], self.physics.drone_states[2]]

            if not self.car_still:
                car_targets = self.physics.car_traj[traj_index]
            else:
                car_targets = car_position

            car_action = car_targets

            drone_action = agent.get_exploitation_action(observations).tolist()

            if self.use_guide_setpoints:
                drone_action_applied = self.actions_with_guide_setpoints(drone_action, self.physics.car_states)
            else:
                drone_action_applied = self.rescale_drone_actions(drone_action)

            d_x = drone_action_applied[0]
            d_y = drone_action_applied[2]
            setpoints_list.append([d_x, d_y])

            action = [drone_action_applied, car_action]

            if self.params.agent_params.add_actions_observations:
                action_observations = np.append(action_observations, action)[1:]

            drone_states_next, car_states_next, failed = self.physics.step(action)

            car_position_next = [car_states_next[0], car_states_next[2]]

            drone_position_next = [drone_states_next[0], drone_states_next[2]]

            r = self.reward_fcn.reward(drone_position, car_position, failed)

            if self.use_relative_states:
                relative_states = self.relative_states(drone_states_next[:4], car_states_next)

                if self.add_observations_noise:
                    noise = np.random.normal(loc=0, scale=0.05, size=8)
                    observations_next = np.hstack((drone_states_next[:4], relative_states))
                    observations_next = np.add(observations_next, noise).tolist()
                else:
                    observations_next = np.hstack((drone_states_next[:4], relative_states)).tolist()
            else:
                observations_next = np.hstack((drone_states_next[:4], car_states_next)).tolist()

            self.model_stats.observations = copy.deepcopy(observations_next)

            self.model_stats.measure(drone_position_next, car_position_next, failed,
                                     distance_score_factor=self.params.reward_params.distance_score_factor)

            self.model_stats.reward.append(r)
            self.model_stats.car_position.append(car_position)
            self.model_stats.drone_position.append(drone_position)
            self.model_stats.drone_actions.append(drone_action_applied)

            traj_index += 1

            if traj_index >= len_traj:
                traj_index = 0

            if failed:
                break

        distance_score_and_survived = float(self.model_stats.survived) * self.model_stats.get_average_distance_score()

        # evaluate heuristic setpoint provider
        self.evaluation_episode_dp(initial_states)

        self.model_stats.evaluation_monitor_image(ep, self.model_stats.evaluation_log_writer,
                                                  self.model_stats.total_train_steps)

        self.physics.close()

        self.model_stats.evaluation_monitor_scalar(ep, self.model_stats.evaluation_log_writer,
                                                   self.model_stats.total_train_steps)

        plt.plot(np.array(setpoints_list)[:, 0], np.array(setpoints_list)[:, 1])

        plt.show()

        return distance_score_and_survived

    def evaluation_episode_dp(self, initial_states):
        """
        This evaluation directly sets the positions and velocities of the car as the setpoints for controlling the drone
        """
        trajs, len_traj = self.physics.get_clover_trajectory()

        self.physics.car_traj = trajs
        self.physics.len_traj = len_traj
        traj_index = 0

        drone_initial_states, car_initial_states = initial_states

        self.physics.reset(car_re_states=car_initial_states, drone_re_state=drone_initial_states)

        for step in range(self.params.stats_params.max_episode_steps):

            if self.params.stats_params.visualize_eval:
                self.physics.render()
                # time.sleep(0.1)

            car_position = [self.physics.car_states[0], self.physics.car_states[2]]

            drone_position = [self.physics.drone_states[0], self.physics.drone_states[2]]

            if not self.car_still:
                car_targets = self.physics.car_traj[traj_index]
            else:
                car_targets = car_position

            car_action = car_targets

            drone_action = self.physics.car_states

            action = [drone_action, car_action]

            drone_states_next, car_states_next, failed = self.physics.step(action)

            car_position_next = [car_states_next[0], car_states_next[2]]

            drone_position_next = [drone_states_next[0], drone_states_next[2]]

            self.model_stats.measure(drone_position_next, car_position_next, failed,
                                     distance_score_factor=self.params.reward_params.distance_score_factor)

            self.model_stats.drone_position_dp.append(drone_position)

            traj_index += 1

            if traj_index >= len_traj:
                traj_index = 0

            if failed:
                break

        self.physics.close()

    def train(self):
        """
        This is the training interaction loop
        """

        ep = 0
        best_dsas = 0.0  # Best distance score and survived
        moving_average_dsas = 0.0

        while self.model_stats.total_train_steps < self.model_stats.params.global_train_steps:

            self.model_stats.init_episode()
            if self.car_start_origin:
                self.physics.car_states = np.zeros(shape=self.physics.params.dim_states_car)
            self.initialize_observations(self.use_relative_states)

            trajs, len_traj = self.physics.random_trajectory_generator()

            self.physics.car_traj = trajs
            self.physics.len_traj = len_traj

            ep += 1
            step = 0
            stage = 0

            if self.params.agent_params.add_actions_observations:
                action_observations = np.zeros(shape=self.params.agent_params.action_observations_dim)
            else:
                action_observations = []

            for step in range(self.params.stats_params.max_episode_steps):

                # self.physics.render()

                observations = np.hstack((self.model_stats.observations, action_observations)).tolist()

                traj_point = stage % self.physics.len_traj

                car_position = [self.physics.car_states[0], self.physics.car_states[2]]

                drone_position = [self.physics.drone_states[0], self.physics.drone_states[2]]

                if not self.car_still:
                    car_targets = self.physics.car_traj[traj_point]
                else:
                    car_targets = car_position

                car_action = car_targets

                drone_action = self.agent.get_exploration_action(observations).tolist()

                if self.params.agent_params.add_actions_observations:
                    action_observations = np.append(action_observations, drone_action)[1:]

                if self.use_guide_setpoints:
                    drone_action_applied = self.actions_with_guide_setpoints(drone_action, self.physics.car_states)
                else:
                    drone_action_applied = self.rescale_drone_actions(drone_action)

                action = [drone_action_applied, car_action]

                drone_states_next, car_states_next, failed = self.physics.step(action)  # get next-states from system

                car_position_next = [car_states_next[0], car_states_next[2]]

                drone_position_next = [drone_states_next[0], drone_states_next[2]]

                r = self.reward_fcn.reward(drone_position, car_position, failed)

                if self.use_relative_states:
                    relative_states = self.relative_states(drone_states_next[:4], car_states_next)
                    observations_next = np.hstack((drone_states_next[:4], relative_states)).tolist()
                else:
                    observations_next = np.hstack((drone_states_next[:4], car_states_next)).tolist()

                self.trainer.store_experience(observations, drone_action, r, observations_next, failed)

                self.model_stats.observations = copy.deepcopy(observations_next)

                self.model_stats.measure(drone_position_next, car_position_next,
                                         failed, distance_score_factor=self.params.reward_params.distance_score_factor)
                self.model_stats.reward.append(r)

                loss_critic = self.trainer.optimize()
                self.model_stats.critic_losses.append(loss_critic)

                if self.physics.car_on_target(car_position_next, car_targets):
                    stage += 1

                if failed:
                    break

            self.model_stats.add_steps(step)
            self.model_stats.training_monitor(ep, self.model_stats.training_log_writer,
                                              self.model_stats.total_train_steps)

            self.agent.noise_factor_decay(self.model_stats.total_train_steps)

            if ep % self.params.stats_params.eval_period == 0:
                dsal = self.evaluation_episode(self.agent, ep)
                moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * dsal
                if moving_average_dsas > best_dsas:
                    self.agent.save_weights(self.params.stats_params.model_name + '_best')
                    best_dsas = moving_average_dsas

        self.agent.save_weights(self.params.stats_params.model_name)

    def car_controller(self, position, target_position):
        self.pid_x.setpoint = target_position[0]
        self.pid_y.setpoint = target_position[1]
        a_x = self.pid_x(position[0])
        a_y = self.pid_y(position[1])
        return [a_x, a_y]

    def reset_pid(self):
        self.pid_x = PID(Kp=1.47, Kd=0.5, setpoint=0, sample_time=0.1, output_limits=(-0.6, 0.6))
        self.pid_y = PID(Kp=1.47, Kd=0.5, setpoint=0, sample_time=0.1, output_limits=(-0.6, 0.6))

    def rescale_drone_actions(self, drone_agent_output):
        """
        The output of the agent is in [-1, 1]. We need to rescale it to the corresponding magnitudes
        """
        drone_actions = drone_agent_output
        drone_actions[0] *= self.physics.params.x_length
        drone_actions[2] *= self.physics.params.y_length
        drone_actions[1] *= self.physics.params.v_limit
        drone_actions[3] *= self.physics.params.v_limit
        return drone_actions

    def actions_with_guide_setpoints(self, drone_agent_output, car_states_setpoints):
        """
        Since the car's states are observable, we can get agent only output the set-point resduals.
        The final set-points of the drone are calculated as:
        drone_setpoints = agent_output + car_states
        To note, the calculated setpoints are saturated
        """

        car_x_setpoint = np.clip(car_states_setpoints[0], a_min=-1 * self.physics.params.x_length,
                                 a_max=1 * self.physics.params.x_length)
        car_x_dot_setpoint = np.clip(car_states_setpoints[1], a_min=-1 * self.physics.params.v_limit,
                                     a_max=1 * self.physics.params.v_limit)
        car_y_setpoint = np.clip(car_states_setpoints[2], a_min=-1 * self.physics.params.y_length,
                                 a_max=1 * self.physics.params.y_length)
        car_y_dot_setpoint = np.clip(car_states_setpoints[3], a_min=-1 * self.physics.params.v_limit,
                                     a_max=1 * self.physics.params.v_limit)

        drone_x = np.clip((drone_agent_output[0] + car_x_setpoint), a_min=-1 * self.physics.params.x_length,
                          a_max=1 * self.physics.params.x_length)
        drone_x_dot = np.clip((drone_agent_output[1] + car_x_dot_setpoint), a_min=-1 * self.physics.params.v_limit,
                              a_max=1 * self.physics.params.v_limit)
        drone_y = np.clip((drone_agent_output[2] + car_y_setpoint), a_min=-1 * self.physics.params.y_length,
                          a_max=1 * self.physics.params.y_length)
        drone_y_dot = np.clip((drone_agent_output[0] + car_y_dot_setpoint), a_min=-1 * self.physics.params.v_limit,
                              a_max=1 * self.physics.params.v_limit)

        drone_actions = np.array([drone_x, drone_x_dot, drone_y, drone_y_dot]).tolist()
        return drone_actions

    def relative_states(self, drone_states, car_states):
        relative_states = np.subtract(car_states, drone_states).tolist()
        return relative_states

    def initialize_observations(self, use_relative_states=False):
        if use_relative_states:
            relative_states = self.relative_states(self.physics.drone_states[:4], self.physics.car_states)
            self.model_stats.observations = np.hstack((self.physics.drone_states[:4], relative_states))
        else:
            self.model_stats.observations = np.stack((self.physics.drone_states[:4], self.physics.car_states))
