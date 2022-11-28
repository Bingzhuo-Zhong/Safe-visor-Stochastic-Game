% config_simulation_Quadrotor_4d_sim: MATLABscript for running Monte Carlo
% simulation for the 4 dimensional case study
% Code for Paper "Towards Safe AI: Sandboxing DNNs-based Controllers in Stochastic Games"
%   in Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence
% Authors:
%
%    Bingzhuo Zhong , Technical University of Munich, Germany
%
% Email:
%
%   bingzhuo.zhong@tum.de
%
% Last update:
%
%   August 15, 2022
%
% Cite:
%
%   If you find the code useful and want to use it for research
%   purpose, please cite our paper following the instruction on:
%
%          https://github.com/Bingzhuo-Zhong/Safe-visor-Stochastic-Game

% loading the sva files(copy the files here)
subssys1 = load('Drone_DFA_sva.mat');
subssys2 = load('Drone_invariance_sva.mat');

safety_advisor1 = subssys1.safety_advisor;
safety_advisor2 = subssys2.safety_advisor;

mode_c = 2;
safety_advisor1.mode = mode_c;
safety_advisor2.mode = mode_c;

monte_sim = Monte_Sim_options();
monte_sim.isgame = 1;
monte_sim.loop = 10000;                                 % define number of repeat simulation
plot_num = 2;
monte_sim.core_num = 2;                                 % number of worker for parallel simulation
monte_sim.x0 = [0;0;0;0];                               % initial state of the simulation

% compute the formal gaurantee given the selected initial state
guarantee1 = safety_advisor1.inquire_init(monte_sim.x0(1:2));
guarantee2 = safety_advisor2.inquire_init(monte_sim.x0(3:4));
guarantee = guarantee1+guarantee2-guarantee1*guarantee2

monte_sim.dimx = 4;                                     % dimension of the system's state to be simulated
monte_sim.dimxab = 4;                                   % dimension of the system's state to be simulated
monte_sim.dimu = 2;                                     % dimension of the system's input to be simulated
monte_sim.dimw = 2;
monte_sim.sva = {safety_advisor1;safety_advisor2};      % safe_visor for the simulation

% running the simulation script
t_a=clock;
Quadrotor_4dim_sim_script;
time_total = etime(clock,t_a)

% 
result = monte_sim.rc_loop_analysis()
accrate = monte_sim.avg_accrate()
avgtime = mean(monte_sim.DL_time)

save('eight_monte_data','gen_exc_time','gen_flag_all','monte_sim','state_car_v')

