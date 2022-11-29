% config_Quadrotor_4dim_A_E: MATLABscript for synthesizing the controller
% against property (A_E,H)
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

lifting_Quadrotor_4dim_A_E;
load('lifting_drone.mat')

% specifying the desired properties model by DFA
DFA = DFA_options();

DFA.n_state = 3;
DFA.sink_state = 3;

dfa_var.C1 = [-1;1];
dfa_var.d1 = [0.5;0.5];
dfa_var.C21 = -1;
dfa_var.d21 = -0.5;
dfa_var.C22 = 1;
dfa_var.d22 = -0.5;
DFA.dfa_var = dfa_var;

p1 = 'dfa_var.C1*x<=dfa_var.d1';
p2 = 'dfa_var.C21*x<=dfa_var.d21 ||dfa_var.C22*x<=dfa_var.d22';
p3 = '1';

DFA_m(1,2)={p1};
DFA_m(1,3)={p2};
DFA_m(2,2)={p1};
DFA_m(2,3)={p2};
DFA_m(3,3)={p3};

DFA.spec_type = 1;
DFA.acc_state = 3;
DFA.dfa = DFA_m;

for iq = 1:1:DFA.n_state
    for iq2 = 1:1:DFA.n_state
        if ~isempty(DFA.dfa{iq,iq2})
            text = ['if ',DFA.dfa{iq,iq2},';ind=1;end'];
            DFA.dfa{iq,iq2} = text;
        end
    end
end
 
DFA_emap = DFA;
DFA_emap.sink_state = 3;

cp1 = Lmap_options();
cp1.A = dfa_var.C1;
cp1.b = dfa_var.d1;

cp21 = Lmap_options();
cp21.A = dfa_var.C21;
cp21.b = dfa_var.d21;
cp22 = Lmap_options();
cp22.A = dfa_var.C22;
cp22.b = dfa_var.d22;

DFA_emap.dfa{1,2} = {cp1};
DFA_emap.dfa{1,3} = [{cp21};{cp22}];
DFA_emap.dfa{2,2} = {cp1};
DFA_emap.dfa{2,3} = [{cp21};{cp22}];
DFA_emap.dfa{3,3}= [];
DFA_emap.dfa{4,4}= [];

clear DFA_m dfa_var;
clear p1  p2  p3 iq iq2 text;
clear cp1 cp21 cp22;

%% ======= Definition of dynamic of physical system ======
phy_system = phy_system_options();

% indicating it is a gDTSG
phy_system.isgame = 1;

phy_system.sys_type = 1;
% sys_typ = 1 : system with continuous state space and contiuous input space
% sys_typ = 2 : system with continuous state space and discrete input space

phy_system.x_l = liftingc.x_l;                       % Lower bound for state
phy_system.x_u = liftingc.x_u;                       % Upper bound for state

% define the abstraction state set
xab_var = Lmap_options();
xab_var.A = [-1 0;0 -1;1 0;0 1];
xab_var.b = [0.5;0.4;0.5;0.4];
phy_system.x_range = xab_var;
clear xab_var;

% with infinite input space
phy_system.u_l = -1.3;                              % Lower bound for input
phy_system.u_u = 1.3;                               % Upper bound for input

% with input space for player 2
phy_system.w_l = liftingc.w_l;                      % Lower bound for disturbance input
phy_system.w_u = liftingc.w_u;                      % Upper bound for disturbance input

% finite input set
phy_system.Cov = [0.004^2 0;0 0.045^2];             % Diagonal of Noise Covariance Matrix

% specify the region of initial state set
phy_var.C1 = [-1 0;0 -1;1 0;0 1];
phy_var.d1 = [0.5;0.4;0.5;0.4];
phy_system.phy_var = phy_var;
phy_system.initial_state_set = ['if phy_var.C1*x<=phy_var.d1;ind=1;end'];


%% defining SVA_program_QbLSt_options
QbLSt_prog = SVA_program_QbLSt_options();

QbLSt_prog.phy_system = phy_system;
QbLSt_prog.isgame = phy_system.isgame;
QbLSt_prog.DFA = DFA;
QbLSt_prog.DFA_epmap = DFA_emap;
QbLSt_prog.lifting = liftingc;
QbLSt_prog.epsilon = QbLSt_prog.lifting.epsilon;

QbLSt_prog.np_x = liftingc.np_x;                    % Number of partitions for state
QbLSt_prog.np_u = 13;                               % Number of partitions for input
QbLSt_prog.np_w = liftingc.np_w;                    % Number of partitions for player 2's input

% specifiy parameters for safety property
QbLSt_prog.spec_type = 2;
% spec_typ = 1 : invariance specification
% spec_typ = 2 : safety property
% spec_typ = 3 : n-accepted LTL property

QbLSt_prog.terminate_type = 2;
% terminate_type = 1 : Backward iteration is terminated by pre-specified maximal tolerable unsafe probability
% terminate_type = 2 : Backward iteration is terminated by pre-specified time horizon

QbLSt_prog.max_unsafe_prob = 0.01;         % define the desired level of safety, equal to 1 - eta, with eta as in (2.2)

%specify the percentage of the states, from which a Markov policy meeting
%the desired level of safety exists
QbLSt_prog.perc_tol = 0.1;

% size of buffer for fix-point operation when synthesizing safety advisor
QbLSt_prog.buffer_length = 600;
QbLSt_prog.time_horizon = 600;

% constructing MDP
test = 0;

if test
    QbLSt_prog = QbLSt_prog.construct_MDP(0);
else
    QbLSt_prog = QbLSt_prog.construct_MDP(1);
    QbLSt_prog.en_para = 1;         % whether calling the parallel function
    QbLSt_prog.core_num = 4;
    
    %compute the NaN-1 map for the initial state
    QbLSt_prog = QbLSt_prog.compute_epmap();
    QbLSt_prog = QbLSt_prog.compute_init();
    QbLSt_prog = QbLSt_prog.compute_accmap();
    [safety_advisor,QbLSt_prog] = QbLSt_prog.syn_safe_visor();
end

% save the results
save('Drone_invariance_sva','safety_advisor','QbLSt_prog','DFA');
