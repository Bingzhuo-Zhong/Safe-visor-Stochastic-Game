% Quadrotor_4dim_sim_script: MATLABscript for running Monte Carlo
% simulation for the 4 dimensional case study (call by config_simulation_Quadrotor_4d_sim.m)

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

for i = 1:1:20
    % warming up
    input = rand(1,8);
    a=py.bridge.send_states_get_actions(input(1), input(2), input(3), input(4), input(5), input(6), input(7), input(8));
    a=double(py.array.array('d', a));
end


% some variables defined for the initialization of each parfor loop
dim_x = monte_sim.dimx;
dim_xab = monte_sim.dimxab;
dim_u = monte_sim.dimu;
dim_w = monte_sim.dimw;
is_game = monte_sim.isgame;
test_time = monte_sim.sva{1}.test_time;
safe_visor_group = monte_sim.sva;
x_int = monte_sim.x0;
para_uc_t = monte_sim.para_uc;


% initialization of the car
xcar_initialx = [x_int(1),x_int(3),0];
xcar_initialv = [x_int(2),x_int(4),0];
move = 1;

%% compute the set points for the object to be tracked
execution_time = 60;  %Total flight time
sp_rate = 10;                                                              % Setpoint rate [Hz]
deltaT= 1/sp_rate;

amp = 0;
ff  = 0.5;
execution_time_e = 60;  %Total flight time
altitude         = -1.1;                                                   % Altitude for hovering after take-off [Down]
omega            = 0.5;
scale            = 1.5;
plotf = 0;

[pos_sp,eul_att_sp] = eight_vertical_traj_gen(xcar_initialx,xcar_initialv,sp_rate,execution_time_e,altitude,omega,scale,amp,ff,plotf);

% synthesizing a clover trajectory based on double eight
x_lkk = pos_sp(340:490,1);
y_lkk = pos_sp(340:490,2);
pos_sp(340:490,1) = y_lkk;
pos_sp(340:490,2) = x_lkk;

Ae = [1,deltaT;0,1];
Be = [deltaT^2/2;deltaT];
A = [Ae zeros(2,2);zeros(2,2) Ae];
B = [Be zeros(2,1);zeros(2,1) Be];
K_cur_car=[1.4781,1.7309,0,0;0,0,1.4781,1.73089];
uxb_car=0.6;
uyb_car=0.6;


% slice variable for simulation
gen_DL_time      = zeros(monte_sim.loop,1);                     % recording average execution time in each loop
gen_rc_loop      = cell(monte_sim.loop,1);
gen_u_execute    = cell(monte_sim.loop,1);                      % recording the inputs executed in each loop
if is_game ==1
    gen_w_execute    = cell(monte_sim.loop,1);                  % recording the internal (player 2) inputs executed in each loop
end
gen_x_execute    = cell(monte_sim.loop,1);                      % recording the evolution of the systems in each loop
gen_xab_execute  = cell(monte_sim.loop,1);                      % recording the evolution of the abstraction systems in each loop
gen_q_execute    = cell(monte_sim.loop,1);                      % recording the evolution of the systems in each loop
gen_flag         = cell(monte_sim.loop,1);                      % recording acceptance of unverified controller in each loop
gen_x_uc_execute = cell(monte_sim.loop,1);                      % array for state evolution without safe-visor
gen_pro          = cell(monte_sim.loop,1);

% user defined vector
gen_exc_time     = cell(monte_sim.loop,1);%
gen_flag_all     = cell(monte_sim.loop,1);%

for loop_num=1:monte_sim.loop
    
    % getting the current safe_visor object
    cur_sva1 = safe_visor_group{1};%
    cur_sva2 = safe_visor_group{2};%
    
    % initializing cur_sva
    cur_sva1 = cur_sva1.int_sfadv(x_int(1:2));%
    cur_sva2 = cur_sva2.int_sfadv(x_int(3:4));%
    
    % get the parameter lists for the unverified controller
    cur_par_uc1 = para_uc_t;
    cur_par_uc2 = para_uc_t;
    
    % recording the time for execution of supervisor
    dec_time = 0;                                     % calculating execution time
    exe_time = test_time;                             % default execution time for the
    exits_flag = 0;                                   % indicate whether or not the simulation terminate before time is up
    
    % auxiliary variable for parallel simulation in each simulation
    temp_rc_loop = [test_time 0];                    % default rc_loop in the simulation
    temp_q_execute = zeros(2,test_time+1);           % matrix for state evolution of DFA
    temp_x_execute = zeros(dim_x,test_time+1);       % matrix for state evolution of the system controller by safe-visor architecture
    temp_xab_execute = zeros(dim_xab,test_time+1);   % matrix for state evolution of the system controller by safe-visor architecture
    temp_x_uc_execute = zeros(dim_x,test_time+1);    % matrix for state evolution of the system only controlled by unverified controller
    temp_u_execute = zeros(dim_u,test_time);         % matrix for input evolution
    if is_game ==1
        temp_w_execute = zeros(dim_w,test_time);     % matrix for internal (player 2) input evolution
    end
    temp_flag = zeros(1,test_time);                  % matrix for "flag", indicating whether or not the unverified controller is accepted in each time instant
    temp_pro = zeros(2,test_time);                   % matrix for "ac_pro", indicating the condition of the supervisor in each time instant
    
    % user define vector
    temp_exe_time = zeros(2,test_time);
    temp_flag_all = zeros(2,test_time);
    
    % partic
    drone_execute = zeros(dim_x,test_time+1);       % matrix for state evolution of the system controller by safe-visor architecture
    drone_execute_uc= zeros(dim_x,test_time+1);
    
    % save the initial state
    temp_x_execute(:,1) = x_int;
    temp_x_uc_execute(:,1) = x_int;
    x_cur = x_int;
    x_cur_uc = x_int;
    
    % initialization of a car for each run
    execution_time = 60;  %Total flight time
    sp_rate = 10;                                                              %Setpoint rate [Hz]
    deltaT= 1/sp_rate;
    tempo = 0:deltaT:execution_time;
    num = length(tempo);
    state_car_v = zeros(4,num);
    p_set_v = zeros(2,num);
    x_pre = [xcar_initialx(1);xcar_initialv(1);xcar_initialx(2);xcar_initialv(2)];
    state_car_v(:,1) = x_pre;
    acc_car = [0;0];
    cc = 1;
    init_num=1;

    for current_time=1:1:test_time
        if exits_flag == 0
            % getting the input provided by the unverified controller
            % reconstruct the state vectors
            state_drone = [x_cur(1)+state_car_v(1,current_time);x_cur(2)+state_car_v(2,current_time);...
                x_cur(3)+state_car_v(3,current_time);x_cur(4)+state_car_v(4,current_time)];
            state_drone_uc = [x_cur_uc(1)+state_car_v(1,current_time);x_cur_uc(2)+state_car_v(2,current_time);...
                x_cur_uc(3)+state_car_v(3,current_time);x_cur_uc(4)+state_car_v(4,current_time)];
            drone_execute(:,current_time)  = state_drone;
            drone_execute_uc(:,current_time)  = state_drone_uc;
            
            input=[state_drone(3),state_drone(4),state_drone(1),...
                state_drone(2),state_car_v(3,current_time),state_car_v(4,current_time),...
                state_car_v(1,current_time),state_car_v(2,current_time)];
            input_uc=[state_drone_uc(3),state_drone_uc(4),state_drone_uc(1),...
                state_drone_uc(2),state_car_v(3,current_time),state_car_v(4,current_time),...
                state_car_v(1,current_time),state_car_v(2,current_time)];
            
            % call the DNNs-based controllers
            [u_uc,cur_par_uc1] = unverified_controller(input, current_time,cur_par_uc1);
            [u_uc_uc,cur_par_uc2] = unverified_controller(input_uc,current_time,cur_par_uc2);
            
            % call the safe_visor architecture
            [stop1,controller1,flg1,time1,cur_sva1] = cur_sva1.sva_exam(current_time,x_cur(1:2),u_uc(1));
            [stop2,controller2,flg2,time2,cur_sva2] = cur_sva2.sva_exam(current_time,x_cur(3:4),u_uc(2));
            controller = [controller1;controller2];
            
            % update the abstraction system
            temp_xab_execute(:,current_time) = [cur_sva1.cur_xab;cur_sva2.cur_xab];
            
            % record current q
            temp_q_execute(:,current_time) = [cur_sva1.current_q;cur_sva2.current_q];
            
            if stop1 == 1 || stop1 == 2 || stop1 == 3 || stop2 == 1 || stop2 == 2 || stop2 == 3 
                % execution is terminated
                exe_time = current_time;
                temp_rc_loop(1,1) = exe_time; 
                temp_rc_loop(1,2) = 1;
                exits_flag = 1;
            else
                dec_time = dec_time + time1+time2;
                temp_exe_time(:,current_time) = [time1;time2];
                
                % record whether the input from the unverfied controller is
                % accepted
                if flg1==2&&flg2==2
                    flg = 2;
                else
                    flg = 1;
                end
                temp_flag(1,current_time) = flg;
                temp_flag_all(:,current_time) = [flg1;flg2];
                
                % record ac_pro
                temp_pro(:,current_time) = [cur_sva1.ac_pro;cur_sva2.ac_pro];
                
                % record the input to be executed
                temp_u_execute(:,current_time) = controller;
                
                %% ================================ calculate the state at ncurent time point =====================================
                % generate current noise (to make sure that the comparison between with and without mission controller makes sense)
                cur_disturbance = noise_gen();
                
                if init_num>0
                    w_execute(:,init_num) = acc_car;
                    % update the state of the car
                    state_car_v(:,init_num+1) = A*state_car_v(:,init_num) + B*acc_car;
                end
                init_num = init_num+1;
                
                if is_game == 1
                    % when we are simulating a game
                    
                    % get the current player'2 input according to
                    % the current u and x
                    
                    
                    % get the set point of the car
                    try
                        if move ==1
                            x_set= pos_sp(cc,1:2);
                            cc = cc+1;
                        else
                            x_set=[xcar_initialx(1);xcar_initialx(2)];
                        end
                    catch
                        x_set= x_pre;
                    end
                    x_pre = x_set;
                    
                    % compute controll input for the car
                    state_cur_car = [state_car_v(1,init_num)-x_set(1);state_car_v(2,init_num);state_car_v(3,init_num)-x_set(2);state_car_v(4,init_num)];
                    acc_temp_car = -K_cur_car*state_cur_car;
                    
                    % acceleration saturation for the car
                    acc_car = zeros(2,1);
                    acc_car(1,1) = min(max(acc_temp_car(1),-uxb_car),uxb_car);
                    acc_car(2,1) = min(max(acc_temp_car(2),-uyb_car),uyb_car);
                    
                    % computation for the car ending
                    w_cur = acc_car;
                    
                    % update the current w and w_hat
                    cur_sva1 = cur_sva1.update_adverary(w_cur(1));
                    cur_sva2 = cur_sva2.update_adverary(w_cur(2));
                    
                    % record the evolution of w
                    temp_w_execute(:,current_time) = w_cur;
                    
                    % evolution for system with safevisor architecture
                    x_cur = system_dyn(x_cur,controller,w_cur,cur_disturbance);
                    
                    % evolution for system without safevisor architecture
                    x_cur_uc = system_dyn(x_cur_uc,u_uc_uc,w_cur,cur_disturbance);
                    
                else
                    % when this is not a game
                    % evolution for system with safevisor architecture
                    x_cur = system_dyn(x_cur,controller,cur_disturbance);
                    
                    % evolution for system without safevisor architecture
                    x_cur_uc = system_dyn(x_cur_uc,u_uc_uc,cur_disturbance);
                end
                
                % record the evolution of state
                temp_x_execute(:,current_time+1) = x_cur;
                temp_x_uc_execute(:,current_time+1) = x_cur_uc;
                
            end
        end
    end
    
    %% ================================ recording result for a single loop =====================================
    % calculate the average execution time for the supervisor in this loop
    gen_DL_time(loop_num,:) = dec_time/exe_time;                                   % recording average execution time in each loop
    gen_exc_time{loop_num,:} = temp_exe_time;
    
    % recording simulation result in slice variables
    gen_rc_loop{loop_num,:} = temp_rc_loop;                                        % recording variable "rc_loop" in the current simulation
    gen_u_execute {loop_num,:} = temp_u_execute(:,1:exe_time);                     % recording the inputs executed in the current simulation
    if is_game == 1
        gen_w_execute {loop_num,:} = temp_w_execute(:,1:exe_time);                 % recording the internal (player 2) inputs executed in the current simulation
    end
    gen_x_execute {loop_num,:} = temp_x_execute(:,1:exe_time);                     % recording the evolution of the systems in the current simulation
    gen_xab_execute {loop_num,:} = temp_xab_execute(:,1:exe_time);                 % recording the evolution of the abstraction systems in the current simulation
    
    % *************** record sperately regarding two diffrent sva
    gen_q_execute {loop_num,:} = temp_q_execute(:,1:exe_time);                     % recording the evolution of the systems in the current simulation
    gen_flag{loop_num,:} = temp_flag(:,1:exe_time);                                % recording acceptance of unverified controller in the current simulation
    gen_flag_all{loop_num,:} = temp_flag_all(:,1:exe_time); 
    
    gen_x_uc_execute {loop_num,:} = temp_x_uc_execute(:,1:exe_time);               % recording the state evolution without safe-visor in the current simulation
    
    gen_pro {loop_num,:} = temp_pro(:,1:exe_time);                                 % recording the evolution of ac_pro parameters in the current simulation
end

%% save simulation results
monte_sim.DL_time = gen_DL_time;
monte_sim.rc_loop = gen_rc_loop;
monte_sim.u_execute = gen_u_execute;
if is_game == 1
    monte_sim.w_execute = gen_w_execute;
end
monte_sim.x_execute = gen_x_execute;
monte_sim.xab_execute = gen_xab_execute;
monte_sim.q_execute = gen_q_execute; 
monte_sim.flag = gen_flag;
monte_sim.x_uc_execute = gen_x_uc_execute;
monte_sim.pro = gen_pro;
