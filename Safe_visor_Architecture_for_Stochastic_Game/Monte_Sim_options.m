classdef Monte_Sim_options
    %   Monte_Sim_options: Class for monte carlo simulation
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

    properties
        % ================== variables for simulation ==================
        loop;                           % define number of repeat simulation
        core_num;                       % number of worker for parallel simulation
        x0;                             % initial state of the original system simulation
        dimx;                           % dimension of the system's state to be simulated
        dimxab;
        dimu;                           % dimension of the system's input to be simulated
        dimw;                           % dimension of the system's internal (player 2) input to be simulated
        sva;                            % safe_visor used in the simulation
        isgame;                         % integer: indicating whether the target system is a game
                                        % isgame = 1: this is a DTSG
                                        % isgame = 0; otherwise                        
        
        % ================== result for the simulation  ==================
        rc_loop;                        % rc_loop: array for recording the information in each loop, each row represent a
                                        %           loop, the first column in the loop represent the time when the execution
                                        %           exit, the second column record the reason for exits, 
                                        %           0:  execution ends without reaching the accepting state.
                                        %           1   execution ends with reaching the accepting state.
                                        %           2 	execution ends with reaching the trap state 
                                        %           3   synchronization error  
        u_execute;                      % recording the inputs executed in each loop
        w_execute;                      % recording the internal (player 2) inputs executed in each loop
        x_execute;                      % recording the evolution of the systems in each loop
        xab_execute;
        q_execute;                      % recording the evolution of the systems in each loop
        flag ;                          % recording acceptance of unverified controller in each loop
                                        %        flag = 1: input from the unverified controller is
                                        %                   rejected.
                                        %        flag = 2: input from the unverified controller is
                                        %                   accepted.
                                        %        flag = 3: no decide is made.
        DL_time ;                       % recording average execution time in each loop
        x_uc_execute;                   % array for state evolution without safe-visor
        pro;                            % history value for acc_pro during the execution
        para_uc;                        % parameters for unverified controller
    end
    
    methods
        function obj = Monte_Sim_options()
        %  Monte_Sim_options: Construct an instance of this class
            obj.para_uc = [];
            obj.dimw = 1;
            obj.isgame = 0;
        end
        
        %% parallel simulation
        function obj = par_simulation (obj)
            % parallel simulation
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
            
            % check the status of parellel pool
            poolobj = gcp('nocreate');% If no pool,  create new one.
            if isempty(poolobj)
                parpool(obj.core_num);
            else
                disp('Already initialized');
            end
            
            % some variables defined for the initialization of each parfor loop
            dim_x = obj.dimx;
            dim_xab = obj.dimxab;
            dim_u = obj.dimu;
            dim_w = obj.dimw;
            is_game = obj.isgame;
            test_time = obj.sva.test_time;
            safe_visor = obj.sva;
            x_int = obj.x0;
            para_uc_t = obj.para_uc;
            %x_intab = obj.x0_ab;
            
            % slice variable for simulation
            gen_DL_time      = zeros(obj.loop,1);                     % recording average execution time in each loop
            gen_rc_loop      = cell(obj.loop,1);
            gen_u_execute    = cell(obj.loop,1);                      % recording the inputs executed in each loop
            if is_game ==1
                gen_w_execute    = cell(obj.loop,1);                  % recording the internal (player 2) inputs executed in each loop
            end 
            gen_x_execute    = cell(obj.loop,1);                      % recording the evolution of the systems in each loop
            gen_xab_execute    = cell(obj.loop,1);                      % recording the evolution of the abstraction systems in each loop
            gen_q_execute    = cell(obj.loop,1);                      % recording the evolution of the systems in each loop
            gen_flag         = cell(obj.loop,1);                      % recording acceptance of unverified controller in each loop
            gen_x_uc_execute = cell(obj.loop,1);                      % array for state evolution without safe-visor
            gen_pro          = cell(obj.loop,1);
            
            %parfor_progress(obj.loop);
            parfor loop_num=1:obj.loop

            % getting the current safe_visor object
            cur_sva = safe_visor;
            
            % get the parameter lists for the unverified controller
            cur_par_uc1 = para_uc_t;           
            cur_par_uc2 = para_uc_t;
            
            % initializing cur_sva
            cur_sva = cur_sva.int_sfadv(x_int);
            
            % recording the time for execution of supervisor
            dec_time = 0;                               % calculating execution time
            exe_time = test_time;                       % default execution time for the
            exits_flag = 0;                             % indicate whether or not the simulation terminate before time is up
            
            % auxiliary variable for parallel simulation in each simulation
            temp_rc_loop = [test_time 0];                    % default rc_loop in the simulation
            temp_q_execute = zeros(1,test_time+1);           % matrix for state evolution of DFA
            temp_x_execute = zeros(dim_x,test_time+1);       % matrix for state evolution of the system controller by safe-visor architecture
            temp_xab_execute = zeros(dim_xab,test_time+1);       % matrix for state evolution of the system controller by safe-visor architecture
            temp_x_uc_execute = zeros(dim_x,test_time+1);    % matrix for state evolution of the system only controlled by unverified controller
            temp_u_execute = zeros(dim_u,test_time);         % matrix for input evolution
            if is_game ==1
                temp_w_execute = zeros(dim_w,test_time);         % matrix for internal (player 2) input evolution
            end
            temp_flag = zeros(1,test_time);                  % matrix for "flag", indicating whether or not the unverified controller is accepted in each time instant
            temp_pro = zeros(1,test_time);                   % matrix for "ac_pro", indicating the condition of the supervisor in each time instant
            
            % save the initial state
            temp_x_execute(:,1) = x_int;
            temp_x_uc_execute(:,1) = x_int;
            x_cur = x_int;
            x_cur_uc = x_int;
            
            for current_time=1:1:test_time
                if exits_flag == 0                    
                    % getting the input provided by the unverified controller
                    [u_uc,cur_par_uc1] = unverified_controller(x_cur, current_time,cur_par_uc1);
                    [u_uc_uc,cur_par_uc2] = unverified_controller(x_cur_uc,current_time,cur_par_uc2);
                    
                    
                    % call the safe_visor architecture
                    [stop,controller,flg,time,cur_sva] = cur_sva.sva_exam(current_time,x_cur,u_uc);
                    
                    % record current q
                    temp_q_execute(:,current_time) = cur_sva.current_q;
                    
                    if stop == 1 || stop == 2 || stop == 3
                        % execution is terminated
                        exe_time = current_time;
                        
                        temp_rc_loop(1,1) = exe_time;
                        temp_rc_loop(1,2) = stop;
                        
                        exits_flag = 1;
                    else
                        dec_time = dec_time + time;
                        
                        % record whether the input from the unverfied controller is
                        % accepted
                        temp_flag(1,current_time) = flg;
                        
                        % record ac_pro
                        temp_pro(1,current_time) = cur_sva.ac_pro;
                        
                        % record the input to be executed
                        temp_u_execute(:,current_time) = controller;
                                              
                        % ================================ calculate the state at ncurent time point =====================================
                        % generate current noise (to make sure that the comparison between with and without mission controller makes sense)
                        cur_disturbance = noise_gen();
                        
                        if is_game == 1
                            % when we are simulating a game
                            
                            % get the current player'2 input according to
                            % the current u and x
                            w_cur = adversary(x_cur,controller,current_time);
                                                        
                            % update the current w and w_hat
                            cur_sva = cur_sva.update_adverary(w_cur);
                            
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
                        
                        % abstraction system
                        temp_xab_execute(:,current_time) = cur_sva.cur_xab;
                    end
                end
            end
            
            % ================================ recording result for a single loop =====================================
            % calculate the average execution time for the supervisor in this loop
            gen_DL_time(loop_num,:) = dec_time/exe_time;                                   % recording average execution time in each loop
            
            % recording simulation result in slice variables
            gen_rc_loop{loop_num,:} = temp_rc_loop;                                        % recording variable "rc_loop" in the current simulation
            gen_u_execute {loop_num,:} = temp_u_execute(:,1:exe_time);                     % recording the inputs executed in the current simulation
            if is_game ==1
                gen_w_execute {loop_num,:} = temp_w_execute(:,1:exe_time);                     % recording the internal (player 2) inputs executed in the current simulation
            end
            gen_x_execute {loop_num,:} = temp_x_execute(:,1:exe_time);                     % recording the evolution of the systems in the current simulation
            gen_xab_execute {loop_num,:} = temp_xab_execute(:,1:exe_time);                 % recording the evolution of the abstraction systems in the current simulation
            gen_q_execute {loop_num,:} = temp_q_execute(:,1:exe_time);                     % recording the evolution of the systems in the current simulation
            gen_flag {loop_num,:} = temp_flag(:,1:exe_time);                               % recording acceptance of unverified controller in the current simulation
            gen_x_uc_execute {loop_num,:} = temp_x_uc_execute(:,1:exe_time);               % recording the state evolution without safe-visor in the current simulation
            gen_pro {loop_num,:} = temp_pro(:,1:exe_time);                                 % recording the evolution of ac_pro parameters in the current simulation
            
            %parfor_progress;
            end
           % parfor_progress(0);
            
            % output result
            obj.DL_time = gen_DL_time;
            obj.rc_loop = gen_rc_loop;
            obj.u_execute = gen_u_execute;
            if is_game ==1
                obj.w_execute = gen_w_execute;
            end
            obj.x_execute = gen_x_execute;
            obj.xab_execute = gen_xab_execute;
            obj.q_execute = gen_q_execute;
            obj.flag = gen_flag;
            obj.x_uc_execute = gen_x_uc_execute;
            obj.pro = gen_pro;
            
            % turn off par computing
            % delete(gcp('nocreate'));
        end
        
        %% simulation without parallel
        function obj = simulation (obj)
            % simulation without using parallel computing
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
            
            % some variables defined for the initialization of each parfor loop
            dim_x = obj.dimx;
            dim_xab = obj.dimxab;
            dim_u = obj.dimu;
            dim_w = obj.dimw;
            is_game = obj.isgame;
            test_time = obj.sva.test_time;
            safe_visor = obj.sva;
            x_int = obj.x0;
            para_uc_t = obj.para_uc;
            %x_intab = obj.x0_ab;
            
            % slice variable for simulation
            gen_DL_time      = zeros(obj.loop,1);                     % recording average execution time in each loop
            gen_rc_loop      = cell(obj.loop,1);
            gen_u_execute    = cell(obj.loop,1);                      % recording the inputs executed in each loop
            if is_game ==1
                gen_w_execute    = cell(obj.loop,1);                      % recording the internal (player 2) inputs executed in each loop
            end
            gen_x_execute    = cell(obj.loop,1);                      % recording the evolution of the systems in each loop
            gen_xab_execute    = cell(obj.loop,1);                      % recording the evolution of the abstraction systems in each loop
            gen_q_execute    = cell(obj.loop,1);                      % recording the evolution of the systems in each loop
            gen_flag         = cell(obj.loop,1);                      % recording acceptance of unverified controller in each loop
            gen_x_uc_execute = cell(obj.loop,1);                      % array for state evolution without safe-visor
            gen_pro          = cell(obj.loop,1);
            
            for loop_num=1:obj.loop
                
                % getting the current safe_visor object
                cur_sva = safe_visor;
                
                % initializing cur_sva
                cur_sva = cur_sva.int_sfadv(x_int);
                
                % get the parameter lists for the unverified controller
                cur_par_uc1 = para_uc_t;           
                cur_par_uc2 = para_uc_t;
                
                % recording the time for execution of supervisor
                dec_time = 0;                               % calculating execution time
                exe_time = test_time;                       % default execution time for the
                exits_flag = 0;                             % indicate whether or not the simulation terminate before time is up
                
                % auxiliary variable for parallel simulation in each simulation
                temp_rc_loop = [test_time 0];                    % default rc_loop in the simulation
                temp_q_execute = zeros(1,test_time+1);           % matrix for state evolution of DFA
                temp_x_execute = zeros(dim_x,test_time+1);       % matrix for state evolution of the system controller by safe-visor architecture
                temp_xab_execute = zeros(dim_xab,test_time+1);   % matrix for state evolution of the system controller by safe-visor architecture
                temp_x_uc_execute = zeros(dim_x,test_time+1);    % matrix for state evolution of the system only controlled by unverified controller
                temp_u_execute = zeros(dim_u,test_time);         % matrix for input evolution
                if is_game ==1
                    temp_w_execute = zeros(dim_w,test_time);         % matrix for internal (player 2) input evolution
                end
                temp_flag = zeros(1,test_time);                  % matrix for "flag", indicating whether or not the unverified controller is accepted in each time instant
                temp_pro = zeros(1,test_time);                   % matrix for "ac_pro", indicating the condition of the supervisor in each time instant
                
                % save the initial state
                temp_x_execute(:,1) = x_int;
                temp_x_uc_execute(:,1) = x_int;
%                 index = read_state(cur_sva);
%                 temp_xab_execute(:,1) = cur_sva.MDP.hx(:,index);
                x_cur = x_int;
                x_cur_uc = x_int;
                
                
                for current_time=1:1:test_time
                    if exits_flag == 0
                        % getting the input provided by the unverified controller
                        [u_uc,cur_par_uc1] = unverified_controller(x_cur, current_time,cur_par_uc1);
                        [u_uc_uc,cur_par_uc2] = unverified_controller(x_cur_uc,current_time,cur_par_uc2);
                        
                        % call the safe_visor architecture
                        [stop,controller,flg,time,cur_sva] = cur_sva.sva_exam(current_time,x_cur,u_uc);
                        
                        % abstraction system
                        temp_xab_execute(:,current_time) = cur_sva.cur_xab;
                        
                        % record current q
                        temp_q_execute(:,current_time) = cur_sva.current_q;
                        
                        if stop == 1 || stop == 2 || stop == 3
                            % execution is terminated
                            exe_time = current_time;
                            temp_rc_loop(1,1) = exe_time;
                            temp_rc_loop(1,2) = stop;
                            exits_flag = 1;
                        else
                            dec_time = dec_time + time;
                            
                            % record whether the input from the unverfied controller is
                            % accepted
                            temp_flag(1,current_time) = flg;
                            
                            % record ac_pro
                            temp_pro(1,current_time) = cur_sva.ac_pro;
                            
                            % record the input to be executed
                            temp_u_execute(:,current_time) = controller;
                            
                            % ================================ calculate the state at ncurent time point =====================================
                            % generate current noise (to make sure that the comparison between with and without mission controller makes sense)
                            cur_disturbance = noise_gen();
                            
                            if is_game == 1
                                % when we are simulating a game

                                % get the current player'2 input according to
                                % the current u and x
                                w_cur = adversary(x_cur,controller,current_time);

                                % update the current w and w_hat
                                cur_sva = cur_sva.update_adverary(w_cur);

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
                
                % ================================ recording result for a single loop =====================================
                % calculate the average execution time for the supervisor in this loop
                gen_DL_time(loop_num,:) = dec_time/exe_time;                                   % recording average execution time in each loop
                
                % recording simulation result in slice variables
                gen_rc_loop{loop_num,:} = temp_rc_loop;                                        % recording variable "rc_loop" in the current simulation
                gen_u_execute {loop_num,:} = temp_u_execute(:,1:exe_time);                     % recording the inputs executed in the current simulation
                if is_game == 1
                    gen_w_execute {loop_num,:} = temp_w_execute(:,1:exe_time);                     % recording the internal (player 2) inputs executed in the current simulation
                end 
                gen_x_execute {loop_num,:} = temp_x_execute(:,1:exe_time);                     % recording the evolution of the systems in the current simulation
                gen_xab_execute {loop_num,:} = temp_xab_execute(:,1:exe_time);                 % recording the evolution of the abstraction systems in the current simulation
                gen_q_execute {loop_num,:} = temp_q_execute(:,1:exe_time);                     % recording the evolution of the systems in the current simulation
                gen_flag {loop_num,:} = temp_flag(:,1:exe_time);                               % recording acceptance of unverified controller in the current simulation
                gen_x_uc_execute {loop_num,:} = temp_x_uc_execute(:,1:exe_time);               % recording the state evolution without safe-visor in the current simulation
                gen_pro {loop_num,:} = temp_pro(:,1:exe_time);                                 % recording the evolution of ac_pro parameters in the current simulation
            end
            
            % output result
            obj.DL_time = gen_DL_time;
            obj.rc_loop = gen_rc_loop;
            obj.u_execute = gen_u_execute;
            if is_game == 1
               obj.w_execute = gen_w_execute;  
            end
            obj.x_execute = gen_x_execute;
            obj.xab_execute = gen_xab_execute;
            obj.q_execute = gen_q_execute;
            obj.flag = gen_flag;
            obj.x_uc_execute = gen_x_uc_execute;
            obj.pro = gen_pro;
        end
        
        %% avaerage acceptance rate
        function precentage = avg_accrate(obj)
            % compute thte average acceptance rate for the simulation
            % result
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
            
            num_acc = 0;
            for i=1:1:length(obj.flag)
                temp_acc = sum(obj.flag{i}==2)/length(obj.flag{i});
                num_acc = num_acc + temp_acc;
            end
            
            precentage = num_acc/length(obj.flag);
        end

        %% analysis result for each iteration
        function result = rc_loop_analysis(obj)
            % analysing data in variable "rc_loop"
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
            
            num = length(obj.rc_loop);
            
            % initialize counters
            num0 = 0;
            num1 = 0;
            num2 = 0;
            num3 = 0;
            
            % counting
            for i = 1:1:num
                if obj.rc_loop{i}(2) == 0
                    num0 = num0 +1;
                elseif obj.rc_loop{i}(2) == 1
                    num1 = num1 +1;
                elseif obj.rc_loop{i}(2) == 2
                    num2 = num2 +1;
                elseif obj.rc_loop{i}(2) == 3
                    num3 = num3 +1;
                end
            end
            
            % record result
            result(1,:) = [0 num0 num0/num];
            result(2,:) = [1 num1 num1/num];
            result(3,:) = [2 num2 num2/num];
            result(4,:) = [3 num3 num3/num];
            
        end
        
        %% x_execute
        function outp = plot_x(obj,dim_o,test_time,ua,la,linewidth,numloop)
            %   plot_x: plot the data in x_execute (for the dimension of interests)
            % input:
            %       dim_o:the dimension of interest
            %       test_time: the maximal length of the data to be plot
            %       ua: upper bound of the set of interest
            %       la: lower bound of the set of interest
            %       linewidth: linewidth in the plotting
            %       numloop:number of data traces to be plotted
            % output:
            %       outp: data of the dimension of interest
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
            
            % check the number of input variables
            if nargin ==3
                % drawing is not required, obtain all data in the specified
                % dimension
                num = obj.loop;
                draw = 0;
            elseif nargin == 6
                % draw all data in the specified dimension
                num = obj.loop;
                draw = 1;
            elseif nargin == 7
                % draw the first numloop-th data in the specified dimension
                num = numloop;
                draw = 1;
            else
                error('Please check your input!')
            end
            
            obs_series = obj.x_execute;
            
            obs = zeros(num,test_time+1);
            outp = cell(num,1);
            for i = 1:1:num
                for j = 1:1:obj.rc_loop{i}(1,1)
                    obs(i,j)= obs_series{i}(dim_o,j);
                end
                outp{i} = obs(i,1:j);
            end
            
            if draw == 1 
                % define upper and lower bound
                time_domain = 0:1:test_time;
                upper_b = ua*ones(1,test_time+1);
                lower_b = la*ones(1,test_time+1);
                
                figure(dim_o);
                % drawing the data
                for k=1:1:num
                    t = 0:1:obj.rc_loop{k}(1,1);
                    [tn,xn] = pwc1_plot (t,obs(k,1:obj.rc_loop{k}(1,1)));
                    plot(tn,xn,'b','LineWidth',linewidth);
                    hold on
                end
                plot(time_domain,upper_b,'--r','LineWidth',linewidth);
                hold on
                plot(time_domain,lower_b,'--m','LineWidth',linewidth);
                hold on
            end
        end
        
        %% x_uc_execute
        function outp = plot_xuc(obj,dim_o,test_time,ua,la,linewidth,numloop)
            %   plot_x: plot the data in x_uc_execute (for the dimension of interests)
            % input:
            %       dim_o:the dimension of interest
            %       test_time: the maximal length of the data to be plot
            %       ua: upper bound of the set of interest
            %       la: lower bound of the set of interest
            %       linewidth: linewidth in the plotting
            %       numloop:number of data traces to be plotted
            % output:
            %       outp: data of the dimension of interest
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
            
            % check the number of input variables
            if nargin ==3
                % drawing is not required, obtain all data in the specified
                % dimension
                num = obj.loop;
                draw = 0;
            elseif nargin == 6
                % draw all data in the specified dimension
                num = obj.loop;
                draw = 1;
            elseif nargin == 7
                % draw the first numloop-th data in the specified dimension
                num = numloop;
                draw = 1;
            else
                error('Please check your input!')
            end
            
            
            obs_series = obj.x_uc_execute;
            
            obs = zeros(num,test_time+1);
            outp = cell(num,1);
            for i = 1:1:num
                for j = 1:1:obj.rc_loop{i}(1,1)
                    obs(i,j)= obs_series{i}(dim_o,j);
                end
                outp{i} = obs(i,1:j);
            end
            
            if draw == 1
                % define upper and lower bound
                time_domain = 0:1:test_time;
                upper_b = ua*ones(1,test_time+1);
                lower_b = la*ones(1,test_time+1);
                
                figure(1);
                % drawing the data
                for k=1:1:num
                    t = 0:1:obj.rc_loop{k}(1,1);
                    [tn,xn] = pwc1_plot (t,obs(k,1:obj.rc_loop{k}(1,1)));
                    plot(tn,xn,'b','LineWidth',linewidth);
                    hold on
                end
                plot(time_domain,upper_b,'--r','LineWidth',linewidth);
                hold on
                plot(time_domain,lower_b,'--m','LineWidth',linewidth);
                hold on
            end
        end
        
        %% u_execute
        function outp = plot_u(obj,dim_o,test_time,linewidth,numloop)
            %   plot_x: plot the data in u_execute (for the dimension of interests)
            % input:
            %       dim_o:the dimension of interest
            %       test_time: the maximal length of the data to be plot
            %       linewidth: linewidth in the plotting
            %       numloop:number of data traces to be plotted
            % output:
            %       outp: data of the dimension of interest
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
            
            % check the number of input variables
            if nargin ==3
                % drawing is not required, obtain all data in the specified
                % dimension
                num = obj.loop;
                draw = 0;
            elseif nargin == 4
                % draw all data in the specified dimension
                num = obj.loop;
                draw = 1;
            elseif nargin == 5
                % draw the first numloop-th data in the specified dimension
                num = numloop;
                draw = 1;
            else
                error('Please check your input!')
            end
            
            obs_series = obj.u_execute;
            
            % the evolution of action should be demonstrated
            obs = zeros(num,test_time);
            outp = cell(num,1);
            for i = 1:1:num
                for j = 1:1:obj.rc_loop{i}(1,1)
                    obs(i,j)= obs_series{i}(dim_o,j);
                end
                outp{i} = obs(i,1:j);
            end
 
            if draw ==1
                % drawing the data
                for k=1:1:num
                    t = 0:1:obj.rc_loop{k}(1,1);
                    [tn,xn] = pwc1_plot (t,obs(k,1:obj.rc_loop{k}(1,1)));
                    plot(tn,xn,'b','LineWidth',linewidth);
                    hold on
                end
            end
        end
        
        %% q_execute
        function outp = plot_q(obj,test_time,linewidth,numloop)
            %   plot_q: plot the data in q_execute 
            % input:
            %       test_time: the maximal length of the data to be plot
            %       linewidth: linewidth in the plotting
            %       numloop:number of data traces to be plotted
            % output:
            %       outp: data for output
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
            
            % check the number of input variables
            if nargin ==2
                % drawing is not required, obtain all data
                num = obj.loop;
                draw = 0;
            elseif nargin == 3
                % draw all data in the specified dimension
                num = obj.loop;
                draw = 1;
            elseif nargin == 4
                % draw the first numloop-th data in the specified dimension
                num = numloop;
                draw = 1;
            else
                error('Please check your input!')
            end
            
            obs_series = obj.q_execute;
            
            % the evolution of DFA's state should be demonstrated
            obs = zeros(num,test_time+1);
            outp = cell(num,1);
            for i = 1:1:num
                for j = 1:1:obj.rc_loop{i}(1,1)
                    obs(i,j)= obs_series{i}(1,j);
                end
                outp{i} = obs(i,1:j);
            end
            
            if draw == 1
                % drawing the data
                for k=1:1:num
                    t = 0:1:obj.rc_loop{k}(1,1);
                    [tn,xn] = pwc1_plot (t,obs(k,1:obj.rc_loop{k}(1,1)));
                    plot(tn,xn,'b','LineWidth',linewidth);
                    hold on
                end
            end
        end
        
        %% examinate x_uc
        function result = xuc_analysis(obj,DFA)
            % analysing x_uc
            % input:
            %       DFA: DFA for analysing the x_uc_execution
            % output:
            %       result: row vector: [num_F num_notF num_F/num_trace]
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
            
            num_trace = length(obj.x_uc_execute);
            num_notF = 0;
            num_F = 0;
            
            for i = 1:1:num_trace
                % go through all traces
                
                % initialization
                cur_q = 1;
                check_F = 0;
                for j = 1:1:size(obj.x_uc_execute{i},2)
                    % get the current state
                    x_cur = obj.x_uc_execute{i}(:,j);
                    
                    % compute the current output based on the current state
                    y = QbLSt_orig_output(x_cur);
                    
                    % check the current state based on the current output
                    cur_q = DFA.q_nxt(cur_q,y);
                    
                    if DFA.isacc_state(cur_q)
                        % if the accepting state is reached
                        check_F = 1;
                        break;
                    end
                end
                
                % record the result for the current trace
                if check_F == 1
                    % the accepting stae is reachted
                    num_F =num_F +1;
                else
                    num_notF = num_notF +1;
                end
            end
            result = [num_F num_notF num_F/num_trace];
            
            % display the result
            disp([num2str(num_F),' traces reach the accepting state(s), which is ',num2str(num_F/num_trace*100),'% of the total traces.'])
        end
        
        %% plot y correponding to x_execute
         function outp = plot_y(obj,dim_o,test_time,ua,la,linewidth,numloop)
            %   plot_y: plot the output corresponding to y_execute
            % input:
            %       dim_o:the dimension of interest
            %       test_time: the maximal length of the data to be plot
            %       ua: upper bound of the set of interest
            %       la: lower bound of the set of interest
            %       linewidth: linewidth in the plotting
            %       numloop:number of data traces to be plotted
            % output:
            %       outp: data of the dimension of interest
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
            
            % check the number of input variables
            if nargin ==3
                % drawing is not required, obtain all data in the specified
                % dimension
                num = obj.loop;
                draw = 0;
            elseif nargin == 6
                % draw all data in the specified dimension
                num = obj.loop;
                draw = 1;
            elseif nargin == 7
                % draw the first numloop-th data in the specified dimension
                num = numloop;
                draw = 1;
            else
                error('Please check your input!')
            end
            
            obs_series = obj.x_execute;
            
            obs = zeros(num,test_time+1);
            outp = cell(num,1);
            for i = 1:1:num
                for j = 1:1:obj.rc_loop{i}(1,1)
                    % obtain the current state
                    x_cur = obs_series{i}(:,j);
                    
                    % compute the current output based on the current state
                    y = QbLSt_orig_output(x_cur);
                    
                    obs(i,j)= y(dim_o,1);
                end
                outp{i} = obs(i,1:j);
            end
            
            if draw == 1
                % define upper and lower bound
                time_domain = 0:1:test_time;
                upper_b = ua*ones(1,test_time+1);
                lower_b = la*ones(1,test_time+1);
                
                figure(1);
                % drawing the data
                for k=1:1:num
                    t = 0:1:obj.rc_loop{k}(1,1);
                    [tn,xn] = pwc1_plot(t,obs(k,1:obj.rc_loop{k}(1,1)));
                    plot(tn,xn,'b','LineWidth',linewidth);
                    hold on
                end
                plot(time_domain,upper_b,'--r','LineWidth',linewidth);
                hold on
                plot(time_domain,lower_b,'--m','LineWidth',linewidth);
                hold on
            end
         end
        
         
        %% plot y correponding to x_execute
         function outp = plot_yuc(obj,dim_o,test_time,ua,la,linewidth,numloop)
            %   plot_yuc: plot the output corresponding to x_uc_execute 
            % input:
            %       dim_o:the dimension of interest
            %       test_time: the maximal length of the data to be plot
            %       ua: upper bound of the set of interest
            %       la: lower bound of the set of interest
            %       linewidth: linewidth in the plotting
            %       numloop:number of data traces to be plotted
            % output:
            %       outp: data of the dimension of interest
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
            
            % check the number of input variables
            if nargin ==3
                % drawing is not required, obtain all data in the specified
                % dimension
                num = obj.loop;
                draw = 0;
            elseif nargin == 6
                % draw all data in the specified dimension
                num = obj.loop;
                draw = 1;
            elseif nargin == 7
                % draw the first numloop-th data in the specified dimension
                num = numloop;
                draw = 1;
            else
                error('Please check your input!')
            end
            
            obs_series = obj.x_uc_execute;
            
            obs = zeros(num,test_time+1);
            outp = cell(num,1);
            for i = 1:1:num
                for j = 1:1:obj.rc_loop{i}(1,1)
                    % obtain the current state
                    x_cur = obs_series{i}(:,j);
                    
                    % compute the current output based on the current state
                    y = QbLSt_orig_output(x_cur);
                    
                    obs(i,j)= y(dim_o,1);
                end
                outp{i} = obs(i,1:j);
            end
            
            if draw == 1
                % define upper and lower bound
                time_domain = 0:1:test_time;
                upper_b = ua*ones(1,test_time+1);
                lower_b = la*ones(1,test_time+1);
                
                figure(1);
                % drawing the data
                for k=1:1:num
                    t = 0:1:obj.rc_loop{k}(1,1);
                    [tn,xn] = pwc1_plot(t,obs(k,1:obj.rc_loop{k}(1,1)));
                    plot(tn,xn,'b','LineWidth',linewidth);
                    hold on
                end
                plot(time_domain,upper_b,'--r','LineWidth',linewidth);
                hold on
                plot(time_domain,lower_b,'--m','LineWidth',linewidth);
                hold on
            end
        end
        
    end
end