import math
import time
import numpy as np
from rein.env.drone_grid import DroneGrid, DroneGridParams
from simple_pid import PID
import pandas as pd


drone_params = DroneGridParams()
drone_env = DroneGrid(drone_params)

pid_x = PID(Kp=1.47, Kd=0.5, setpoint=0, sample_time=0.1, output_limits=(-0.6, 0.6))
pid_y = PID(Kp=1.47, Kd=0.5, setpoint=0, sample_time=0.1, output_limits=(-0.6, 0.6))


def car_controller(states, states_target):
    pid_x.setpoint = states_target[0]
    pid_y.setpoint = states_target[1]
    a_x = pid_x(states[0])
    a_y = pid_y(states[2])
    return [a_x, a_y]


def cart_on_target(cart_states, trajectory_point):
    x_distance = math.fabs(cart_states[0] - car_trajectory[trajectory_point][0])
    y_distance = math.fabs(cart_states[2] - car_trajectory[trajectory_point][1])
    if x_distance <= 0.1 and y_distance <= 0.1:
        return True
    else:
        return False


path = "clover.xlsx"
data_source = pd.read_excel(path)
x = data_source[0]
y = data_source[1]
points = data_source.to_numpy()

points = points[:, :, np.newaxis]
car_trajectory = np.concatenate(points, axis=-1)
len_trajectory = car_trajectory.shape[0]
index = 0

while True:

    drone_env.render(mode="human")
    # set_points = [drone_env.car_states[0], 0, drone_env.car_states[2], 0]
    set_points = drone_env.car_states
    # trajectory_point = stage % len_trajectory
    # car_input = car_controller(drone_env.car_states, car_trojectory[trajectory_point])
    car_input = car_trajectory[index]
    action_list = [set_points, car_input]
    # action_list = [[-2.5, 0, -2.5, 0, 0, 0], [0.0, 0.0]]
    _, _, failed = drone_env.step(action_list)
    if failed:
        drone_env.random_reset()

    index += 1

    if index >= len_trajectory:
        index = 0

