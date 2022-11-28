% lifting_Quadrotor_4dim_A_E: MATLABscript for computing the lifting relation for the controller
% against property (A_E,H)
% code for the submission entitled "Towards Safe AI: Sandboxing DNNs-based Controllers in Stochastic Games"
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

clear all
delta_t = 0.1;

liftingc = slprssys_apr_options();

% indicating that this is a gDTSG
liftingc.isgame = 1;
liftingc.en_para = 1;
liftingc.num_core = 4;

%% conpute abstraction of the system
liftingc.A = [1,delta_t;0,1];
liftingc.B = [delta_t^2/2;delta_t];
liftingc.D = [delta_t^2/2;delta_t];
liftingc.C = [1 0];
liftingc.Ar = liftingc.A;
liftingc.Br = liftingc.B;
liftingc.Dr = liftingc.D;
liftingc.R = [0.004 0;0 0.045];
liftingc.P = [1 0;0 1];
liftingc.P_w = 1;

% compute abstraction
liftingc = liftingc.inf_abs();

%% discretization
% configuration for region of interested
liftingc.x_l = [-0.5 -0.4];                 % Lower bound for state
liftingc.x_u = [0.5 0.4];                   % Upper bound for state
liftingc.np_x = [50 40];                    % Number of partitions for state


u_bound = 2.5;
% configuration for input set for player 1
liftingc.u_l = -u_bound;                    % Lower bound for input
liftingc.u_u = u_bound;                     % Upper bound for input
liftingc.np_u = 25;                         % Number of partitions for input

% configuration for input set for player 2
liftingc.w_l = -0.6;                        % Lower bound for disturbance input
liftingc.w_u = 0.6;                         % Lower bound for disturbance input
liftingc.np_w = 12;                         % Number of partitions for player 2's input
liftingc.M_w = 1;

% discretization
liftingc = liftingc.discretization();

%% setting expected U

ua = Lmap_options();
ua.A = [-1;1];
ua.b = [2.5;2.5];
liftingc.u_area = ua;
clear ua;

u_var1 = Lmap_options();
u_var1.A = [-1;1];
u_var1.b = [1.2;1.2];
liftingc.uab_area = {u_var1};
liftingc = liftingc.elim_uab();

% compute input constraints
liftingc = liftingc.compute_inputcon();

%% searching for solutions
liftingc.delta = 0;
eps_range = [0.05 0.4];
eps_num = 40;
kappa_range = [0.6 1];
kappa_num = 50;
liftingc = liftingc.search_solution(eps_range,eps_num,kappa_range,kappa_num);
[liftingc,bnum] = liftingc.search_bestsol();

%% copy best solution
liftingc = liftingc.copy_sol(liftingc.sol_list(bnum).best_sol);

save('lifting_drone','liftingc');
