import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def label(xy, text):
    y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)


def plot_traj(car_traj_xy, drone_uc_traj_xy_monte, drone_sav_xy_monte, drone_uc_traj_xy_phy, drone_sav_xy_phy):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))


    monte_car = ax1.plot(car_traj_xy[1], car_traj_xy[0], c='k', ls='--')
    monte_drone_uc = ax1.plot(drone_uc_traj_xy_monte[1], drone_uc_traj_xy_monte[0])
    monte_drone_sav = ax1.plot(drone_sav_xy_monte[1], drone_sav_xy_monte[0])


    x1, y1 = 0.7, 0.2
    x2, y2 = 1.2, 1.0
    el = mpatches.Ellipse((x1, y1), 1.2, 0.45, angle=0, alpha=0.3, color='red')
    ax1.add_artist(el)
    ax1.annotate("",
                xy=(x1, y1-0.1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2",
                                patchB=el)
                )
    ax1.text(.65, .90, "Safe-visor\ncorrecting", transform=ax1.transAxes, ha="left", va="top")

    ax1.set_ylabel('$E$', fontsize=15)
    ax1.set_xlabel('$N$', fontsize=15)
    ax1.set_xlim([-2.0, 2.0])
    ax1.set_ylim([-2.5, 2.0])
    ax1.grid(True)

    phy_car = ax2.plot(car_traj_xy[1], car_traj_xy[0], c='k', ls='--')
    phy_drone_uc = ax2.plot(drone_uc_traj_xy_phy[1], drone_uc_traj_xy_phy[0])
    phy_drone_sav = ax2.plot(drone_sav_xy_phy[1], drone_sav_xy_phy[0])

    ax2.set_ylabel('$E$', fontsize=15)
    ax2.set_xlabel('$N$', fontsize=15)
    ax2.set_xlim([-2.0, 2.0])
    ax2.set_ylim([-2.5, 2.0])
    ax2.grid(True)
    # ax2.set_title('(b) Physical testbed', y=-0.21, pad=0.4)

    leg = fig.legend([monte_car, monte_drone_uc, monte_drone_sav],  # The line objects
                     labels=["ground vehicle", "w.o Safe-visor", "with Safe-visor"],
                     # The labels for each line
                     loc="upper center",  # Position of legend
                     borderaxespad=0.5,  # Small spacing around legend box  # Title for the legend
                     bbox_to_anchor=(.5, 1.0),
                     ncol=3,
                     framealpha=0.0,
                     fontsize=13
                     )
    # fig.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
    plt.savefig(f"papercode/trajectory.png", dpi=300)


def plot_dims(drone_uc_traj_xy, drone_sva_xy, save_name):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # range_car = range(len(car_traj_xy[0]))
    range_drone_uc = range(len(drone_uc_traj_xy[0]))
    range_drone_sva = range(len(drone_sva_xy[0]))

    # ax1_plot = sns.lineplot(range_car, car_traj_xy[0], palette='k', ls='--', ax=ax1)
    ax1_plot = sns.lineplot(range_drone_uc, drone_uc_traj_xy[1], ax=ax1, linewidth=1.2)
    drone_sva_plot = sns.lineplot(range_drone_sva, drone_sva_xy[1], ax=ax1)
    safety_bound = ax1.hlines(y=0.5, xmin=0, xmax=600, colors="red", linestyles='--', linewidth=1.2)
    ax1_plot.set_ylabel('$y_{E_r}$', fontsize=15)

    x_ticks_positions = [x for x in range(0, 700, 100)]
    x_ticks_labels = [f"{int(x_pos * 0.1)}" for x_pos in x_ticks_positions]

    y_ticks_positions = [0.01 * y for y in range(-50, 75, 25)]
    y_ticks_labels = [f"{y_pos}" for y_pos in y_ticks_positions]

    ax1.set_xticks(x_ticks_positions)
    ax1.set_xticklabels(x_ticks_labels)
    ax1.set_yticks(y_ticks_positions)
    ax1.set_yticklabels(y_ticks_labels)

    ax1.grid(True)

    # ax2_plot = sns.lineplot(range_car, car_traj_xy[1], palette='k', ls='--', ax=ax2)
    ax2_plot = sns.lineplot(range_drone_uc, drone_uc_traj_xy[0], ax=ax2, linewidth=1.2)
    sns.lineplot(range_drone_sva, drone_sva_xy[0], ax=ax2)

    ax2.hlines(y=0.3, xmin=0, xmax=600, colors="red", linestyles='--', linewidth=1.2)
    ax2.hlines(y=-0.3, xmin=0, xmax=600, colors="red", linestyles='--', linewidth=1.2)
    ax2.hlines(y=-0.5, xmin=0, xmax=600, colors="red", linestyles='--', linewidth=1.2)
    ax2_plot.set_ylabel('$y_{N_r}$', fontsize=15)
    ax2_plot.set_xlabel('$t(s)$', fontsize=15)

    x_ticks_positions = [x for x in range(0, 700, 100)]
    x_ticks_labels = [f"{int(x_pos * 0.1)}" for x_pos in x_ticks_positions]
    y_ticks_positions = [-0.5, -0.3, 0, 0.3, 0.5]
    y_ticks_labels = [f"{y_pos}" for y_pos in y_ticks_positions]
    plt.xticks(x_ticks_positions, x_ticks_labels)
    plt.yticks(y_ticks_positions, y_ticks_labels)

    ax2.grid(True)
    # plt.legend(loc='best', fontsize='x-small')

    leg = fig.legend([ax1_plot, drone_sva_plot, safety_bound],  # The line objects
                     labels=["w.o Safe-visor", "with Safe-visor", "safety bound"],
                     # The labels for each line
                     loc="upper center",  # Position of legend
                     borderaxespad=1.0,  # Small spacing around legend box  # Title for the legend
                     bbox_to_anchor=(.5, 1.0),
                     ncol=3,
                     framealpha=0.0,
                     fontsize=13
                     )

    # fig.tight_layout()
    plt.savefig(f"papercode/{save_name}_dims.png", dpi=300)


path_car_traj = 'papercode/AAAI_DATA_CSV/phy_car_traj.csv'

path_drone_sva_monte = 'papercode/AAAI_DATA_CSV/drone_trace_sva_monte.csv'
path_drone_uc_monte = 'papercode/AAAI_DATA_CSV/drone_trace_uc_monte.csv'
path_drone_sva_phy = 'papercode/AAAI_DATA_CSV/phy_drone_sva.csv'
path_drone_uc_phy = 'papercode/AAAI_DATA_CSV/phy_drone_uc.csv'

car_traj = pd.read_csv(path_car_traj).to_numpy()

drone_uc_phy_traj = pd.read_csv(path_drone_uc_phy).to_numpy()
drone_sva_phy_traj = pd.read_csv(path_drone_sva_phy).to_numpy()

data_drone_sva_monte = pd.read_csv(path_drone_sva_monte).to_numpy()
data_drone_uc_monte = pd.read_csv(path_drone_uc_monte).to_numpy()

drone_sva_monte_traj = np.concatenate([[data_drone_sva_monte[0]], [data_drone_sva_monte[2]]], axis=0)
drone_uc_monte_traj = np.concatenate([[data_drone_uc_monte[0]], [data_drone_sva_monte[2]]], axis=0)

dis_uc_phy_traj = drone_uc_phy_traj[:, :600] - car_traj[:, :600]
dis_sva_phy_traj = drone_sva_phy_traj[:, :600] - car_traj[:, :600]

dis_uc_monte_traj = drone_uc_monte_traj - car_traj[:, :600]
dis_sva_monte_traj = drone_sva_monte_traj - car_traj[:, :600]

plot_dims(dis_uc_phy_traj, dis_sva_phy_traj, save_name="phy")
plot_dims(dis_uc_monte_traj, dis_sva_monte_traj, save_name="monte")

plot_traj(car_traj, drone_uc_monte_traj, drone_sva_monte_traj, drone_uc_phy_traj, drone_sva_phy_traj)
