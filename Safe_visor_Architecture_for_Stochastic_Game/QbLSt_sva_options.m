classdef QbLSt_sva_options
    %   QbLSt_sva_options: Summary of this class goes here
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
        isgame;                 % integer: indicating whether the target system is a game
        % isgame = 1: this is a DTSG
                                % isgame = 0; otherwise
        mode;                   % integer:indicating the mode for the safe-visor
                                %       mode = 1: Only using safety advisor
                                %       mode = 2: Normal mode(Default)
        current_time;           % current time instant
        current_q;              % current state
        cur_x;                  % double: memory for state of the original system
        cur_xab;                % double£ºmemory for state of the abstraction system
        cur_u;                  % double: memory of input of the original system
        cur_uab;                % double: memory of input of the abstraction system
        cur_w;                  % double: memory of internal (player 2) input of the original system
        cur_wab;                % double: memory of internal (player 2) input of the abstraction system
    
        DFA;                    % class of DFA: DFA for the evolution of the internal state
        phy_system;             % class of phy_system: information about the original system
        MDP;                    % MDP for the system
        lifting;                % class of slprssys_options: approximation probability relation between the original system and the abstraction
        hx;                     % matrix of double: state space of the safe-visor
        hu;                     % matrix of double: input space of the safe_visor
        np_x;                   % vector of integer: number of state space partition in each dimension (inherit from SVA_program_QbLSt)
        hw;                     % matrix of double: internal (player 2) input space of the safe-visor
        np_w;                   % vector of integer: number of internal (player 2) input space partition in each dimension (inherit from SVA_program_QbLSt)
        % np_u;                 % vector of integer: number of inout space dimension in each dimension (inherit from SVA_program_QbLSt)
        epmap;                  % cell of matrices of double: epsilon-map for backward iteration considering epsilon mapping(inherit from SVA_program_QbLSt)
        acc_map;                % 
        init_state;             % NaN-1 vector representing the initial state we are interested in(inherit from SVA_program_QbLSt)
        test_time;              % integer: time horizon for the safe_visor (horizon of the inout matrix)

                
        % supervisor
        safety_advisor;         % matrix of double: look up table for the safety advisor (optimal safety controller)
        value_matrix;           % matrix for value function of the safety advisor
        spec_type;              % integer: defining the type of the specification for the iteration(inherit from SVA_program_QbLSt)
                                %   spec_typ = 1 : invariance specification
                                %   spec_typ = 2 : safety-LTL property
                                %   spec_typ = 3 : co-safe-LTL property
        ac_pro;                 % accumulative product for safety probability
        sum_acpro;              % summation of ac_pro
        max_unsafe_prob;        % double: define the desired level of safety(inherit from SVA_program_QbLSt)
    end
    
    methods
        function obj = QbLSt_sva_options()
            %  QbLSt_sva_options: Construct an instance of this class
            obj.ac_pro = 1;
            obj.sum_acpro = 1;
            obj.mode = 2;
        end
        
        %% initializtion
        function obj=int_sfadv(obj,x0)
            % initialize the safety advisor to initial state and initial
            % time
            obj.current_q = 1;
            obj.current_time = 1;
            obj.ac_pro = 1;
            obj.cur_x = x0;
            
            % initialize the abstraction
            obj.cur_xab = (obj.lifting.P'*obj.lifting.M*obj.lifting.P)\obj.lifting.P'*obj.lifting.M*x0;
            
        end
        
        %% MAIN
        function [stop,controller,flag,time,obj] = sva_exam(obj,cur_time,x,uuc)
            %  Main function for the safe_visor architecture(QbLSt)
            % Input:
            %
            % Output:
            %   stop: integer: indicating the reason for exits
            %        stop = 0: execution ends without reaching the
            %                   accepting state.
            %        stop = 1: execution ends with reaching the accepting
            %                   state.
            %        stop = 2: execution ends with reaching the trap state
            %        stop = 3: synchronization error
            %  controller: double: input ought to be executed
            %  flag:  integer: indicating whether the current input from
            %                  the unverified controller could be accepted.
            %        flag = 1: input from the unverified controller is
            %                   rejected.
            %        flag = 2: input from the unverified controller is
            %                   accepted.
            %        flag = 3: no decide is made.
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
            
            
            %t1=clock;
            tic
            %  Check synchronization
            if cur_time~= obj.current_time
                % internal time and external time do not match with each
                % other
                disp('Internal time and external time do not match. Please initialize the controller');
                stop = 3;
                controller = NaN;
                flag = 3;
            else
                %  Evolution of DFA
                 y = QbLSt_orig_output(x);
                 obj.current_q = obj.DFA.q_nxt(obj.current_q,y);
                
                if obj.DFA.isacc_state(obj.current_q) == 1
                    % accepting state is reached
                    stop = 1;
                    controller = NaN;
                    flag = 3;
                elseif obj.DFA.istrap_state(obj.current_q) == 1
                    % trap state is reached
                    stop = 2;
                    controller = NaN;
                    flag = 3;
                else
                    stop = 0;
                    % in case that this is not the initial state
                    if obj.current_time > 1
                        % update the state of the system depend on whether
                        % the target system is a game
                        if obj.isgame == 1
                            % compute the noise executed on the original system
                            distb = QbLSt_noise_origsys(obj.cur_x,obj.cur_u,obj.cur_w,x);
                        
                            % save the current state of the original system
                            obj.cur_x = x;
                            
                            % compute the current state of the abstraction
                            obj.cur_xab = QbLSt_abssys_dyn(obj.cur_xab,obj.cur_uab,obj.cur_wab,distb); 
                        else
                            % compute the noise executed on the original system
                            distb = QbLSt_noise_origsys(obj.cur_x, obj.cur_u,x);
                        
                            % save the current state of the original system
                            obj.cur_x = x;
                            
                            % compute the current state of the abstraction
                            obj.cur_xab = QbLSt_abssys_dyn(obj.cur_xab,obj.cur_uab,distb);       
                        end   
                    end
                    
                    %  Read the index of the state (index in the MDP)
                    index_x = obj.read_state();  
                    obj.cur_xab = obj.hx(:,index_x);

                    %  select input
                    if obj.mode == 1
                        % only using safety advisor
                        
                        % get the safety controller
                        obj.cur_uab = obj.safety_input(index_x);  
                        
                        % refine the safety controller for the abstraction
                        % to the original syatem
                        controller = QbLSt_interface(obj.cur_x, obj.cur_xab,obj.cur_uab);
                        
                        % save the current input
                        obj.cur_u = controller;       
                        
                        flag = 1;
                    elseif obj.mode == 2
                        % normal mode
                        %  Read input
                        list = QbLSt_uhat_set(obj.cur_x,obj.cur_xab,uuc,obj.hu);% checking the non-emptyness of the set U_f, i.e. eq (3.2)
                        
                        if isempty(list)
                             % there is no feasible u_ab
                             % use the safety controller
                             obj.cur_uab = obj.safety_input(index_x);
                             
                             % refine the safety controller for the abstraction
                             % to the original syatem
                             controller = QbLSt_interface(obj.cur_x, obj.cur_xab,obj.cur_uab);
                             
                             % save the current input
                             obj.cur_u = controller;
                             
                             flag = 1;
                        else
                            v_nxt = reshape(obj.value_matrix(:,cur_time),[obj.MDP.n_x,obj.DFA.n_state-1]);
                            for kk = 1:1:length(obj.DFA.acc_state)
                                % replace the column with accepting state as one
                                v_nxt(:,obj.DFA.acc_state(kk)-1) = ones(obj.MDP.n_x,1);
                            end
                            
                            % prepare the v vector for backward iteration
                            if obj.spec_type == 1 || obj.spec_type==2
                                v_iter = max(v_nxt.*obj.epmap{obj.current_q-1},[],2);
                            elseif obj.spec_type==3
                                
                            end
                            
                            mc_safe = zeros(length(list),1);
                            if obj.spec_type == 1 || obj.spec_type==2
                                for i = 1:1:length(list)
                                    row = obj.MDP.MDP_get(index_x,list(i),0);
                                    mc_safe_temp = (1-obj.lifting.delta)*obj.ac_pro*(1-row*v_iter);% computing eq. (3.5)
                                    mc_safe(i,1) = min(mc_safe_temp);   
                                end
                            elseif obj.spec_type==3

                            end                           
                          
                            if ((min(mc_safe)/ (1- obj.max_unsafe_prob) < 1) && (obj.spec_type == 1 || obj.spec_type == 2)) ||  ((max(mc_safe) > obj.max_unsafe_prob)&& obj.spec_type == 3) 
                                % in case that the safety probability is lower than the minimnal
                                % tolerable safety probability defined by safety_tol, use the
                                % advisory input from the safety advisor to maximaise the safety
                                % probability of the system
                                
                                % get the safety controller
                                [obj.cur_uab, index_u] = obj.safety_input(index_x);
                                
                                % refine the safety controller for the abstraction
                                % to the original syatem
                                controller = QbLSt_interface(obj.cur_x, obj.cur_xab,obj.cur_uab);
                                
                                % save the current input
                                obj.cur_u = controller;

                                % set up the flag
                                flag = 1;
                            else
                                % in case that the safety probability is higher than the minimnal
                                % tolerable safety probability defined by safety_tol, use the
                                % input from the unverified controller for functionality
                                
                                if obj.spec_type == 1 || obj.spec_type == 2
                                    [~,posu] = max(mc_safe);
                                    index_u = list(posu);
                                    obj.cur_uab = obj.hu(:,index_u);
                                else

                                end
                                
                                % select uuc as the controller
                                controller = uuc;
                                
                                % save the current input
                                obj.cur_u = uuc;

                                % set up the flag
                                flag = 2;
                            end
                            % compute ac_pro
                            if obj.spec_type == 1 || obj.spec_type==2
                                row1 = obj.MDP.MDP_get(index_x,index_u,0);
                                ac_pro_temp = obj.ac_pro*(1-obj.lifting.delta)*(row1*obj.acc_map(:,obj.current_q));
                                obj.ac_pro = min(ac_pro_temp);
                            elseif obj.spec_type==3

                            end
                            
                        end
                        % index_u = obj.read_input(x);
                    else
                        disp('Mode is not correct.');
                    end

                    % setting timer
                    obj.current_time = obj.current_time +1;
                end
            end
            %time = etime(clock,t1);
            time = toc;
        end
        
        %% Read index of state
        function index_x = read_state(obj)
            % searching the index of the current state in hx
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
            
            
            position_x_cur = ceil((obj.cur_xab'-obj.phy_system.x_l)./obj.MDP.delta_x);    % searching the index in each dimension
            
            
            % auxiliary operation for simulation (in case that paths run
            % outside of the safety set)
            for i=1:1:obj.MDP.dim_x
                if position_x_cur(i) > obj.np_x(i)
                    % on one of the dimeisions, state exceed the upper bound of the
                    % state set
                    position_x_cur(i) = obj.np_x(i);
                    state.xalarm = 2;
                    if max(obj.cur_xab >obj.phy_system.x_u')| max(obj.cur_xab<obj.phy_system.x_l')
                        state.xalarm = 1;
                    end
                end
                if position_x_cur(i)<=0
                    % on one of the dimeisions, state lower than the lower bound of
                    % the state set
                    position_x_cur(i) = 1;
                    state.xalarm = 2;
                    if max(obj.cur_xab>obj.phy_system.x_u')| max(obj.cur_xab <obj.phy_system.x_l')
                        state.xalarm = 1;
                    end
                end
            end
            
            % if the current state point locates at the lower boundary on some
            % dimension in state space, attibuting it to the first grid on those dimension
            position_x_cur(position_x_cur==0)=1;
            
            % go through each dimension to calculate the index of the current state
            index_x = position_x_cur(1, obj.MDP.dim_x);     % index of x in the last dimension
            cell_n = 1;
            if obj.MDP.dim_x>1
                for i=obj.MDP.dim_x-1:-1:1
                    cell_n = cell_n*obj.np_x(1,i+1);
                    index_x = index_x+(position_x_cur(1,i)-1)*cell_n;
                end
            end
        end
                
        %% input from the safety advisor
        function [cur_uab,index_u] = safety_input(obj,index_x)
            % compute the position of the current state
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
            
            
            %pos  = index_x+(obj.current_q-1)*(length(obj.hx)+1);
            pos  = index_x+(obj.current_q-2)*(length(obj.hx)+1);
            % fetch the input at position pos and at current time
            index_u = obj.safety_advisor(pos,obj.current_time);
            cur_uab = obj.hu(:,index_u);  
        end
        
        %% inquire value function for the current initial state
        function V_value = inquire_init(obj,x0)
            %  inquire_init: inquire the value of the value function given the
            %  intial state
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
            
            
            % compute the initial state of the abstraction
            obj.cur_xab = (obj.lifting.P'*obj.lifting.M*obj.lifting.P)\obj.lifting.P'*obj.lifting.M*x0;
        
        % read the index of the initial state of the abstraction
        index_xab = obj.read_state;
        
        % compute \bar{q}_0
        q_nxt = obj.DFA.q_nxt(1,QbLSt_orig_output(x0));
        
        % inquire the value of value function % V_value = obj.value_matrix(index_xab+obj.MDP.n_x*(q_nxt-1),1);
        V_value = obj.value_matrix(index_xab+obj.MDP.n_x*(q_nxt-2),1);
        
        chk =  (x0-obj.lifting.P*obj.MDP.hx(:,index_xab))'*obj.lifting.M*(x0-obj.lifting.P*obj.MDP.hx(:,index_xab));
        %chk
            if chk>obj.lifting.epsilon^2
                disp('Warning! Something wrong with the initial state!');
            end
        end
        
        %% read index of w
         function obj = update_adverary(obj,w)
            %  update_adverary: update w and w_hat in the sva according to
            %  the provided w
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
            
             
            % searching the index of the current state in hw
            position_w_cur = ceil((w'-obj.phy_system.w_l)./obj.MDP.delta_w);    % searching the index in each dimension
                        
            % if the current input point locates at the lower boundary on some
            % dimension in input space, attibuting it to the first grid on those dimension
            position_w_cur(position_w_cur==0)=1;
            
            % go through each dimension to calculate the index of the
            % current input
            index_w = position_w_cur(1, obj.MDP.dim_w);     % index of w in the last dimension
            cell_n = 1;
            if obj.MDP.dim_w>1
                for i=obj.MDP.dim_w-1:-1:1
                    cell_n = cell_n*obj.np_w(1,i+1);
                    index_w = index_w+(position_w_cur(1,i)-1)*cell_n;
                end
            end
            
            % update w and w_hat
            obj.cur_w = w;
            obj.cur_wab = obj.hw(:,index_w);
        end
        
    end
end

