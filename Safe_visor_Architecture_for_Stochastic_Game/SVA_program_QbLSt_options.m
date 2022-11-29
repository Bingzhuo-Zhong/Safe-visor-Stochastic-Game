classdef SVA_program_QbLSt_options
    % Class for Quantization based Safe-Visor Architecture for Stochastic
    % system based on Lifting (Approximate Probabilistic Relations)
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
        % parameters of constructing finite MDP
        isgame;                 % integer: indicating whether the target system is a game
                                % isgame = 1: this is a DTSG
                                % isgame = 0; otherwise
        phy_system;             % class of phy_system_options: information for the physical systems
        np_x;                   % vector of double: Row vector for number of partitions for state on each dimension, row vector
        np_u;                   % vector of double: Row vector for number of partitions for input, only for infinite input space, i.e., sys_typ = 1
        np_w;                   % vector of double: Row vector for number of partitions for internal(player 2) input, only for game, i.e., isgame = 1
        MDP;                    % class of MDP_options: the finite MDP for backward iteration
        MDP_option;             % specify the way for getting data from the stochastic kernel of the MDP
        sdef_par;               % self define par for getting data in the salf define mode
        
        % element for iteration
        DFA;                    % class of DFA_options: the DFA backward iteration
        lifting;                % class of lifting relation, which is related to the interface funtion for the safety advisor
        DFA_epmap;              % DFA for computing the epmap
        epmap;                  % cell of matrices of double: epsilon-map for backward iteration considering epsilon mapping
        epsilon;                % double: epsilon for defining the approximate probabilistic relation       
        init_state;             % NaN-1 vector representing the initial state we are interested in
        num_init_state;         % double:number of initial state
        acc_map;                % 
        
        % parameters for synthesizing safety advisor
        en_para;                % integer: flag indicating whether parallel computing is feasible.
                                %    en_para = 0: parallel computing is disabled (Default)
                                %    en_para = 1: parallel computing is enabled
        core_num;               % number of core for parallel computing
        spec_type;              % integer: defining the type of the specification for the iteration
                                %   spec_typ = 1 : invariance specification
                                %   spec_typ = 2 : safety-LTL property
                                %   spec_typ = 3 : co-safe-LTL property
        
        terminate_type;         % interger: how the backward iteration should be terminated
                                %   terminate_type = 1 : terminated by pre-specified maximal tolerable unsafe probability
                                %   terminate_type = 2 : terminated by pre-specified time horizon
        
        % only have to be specified when terminate_type = 1
        max_unsafe_prob;        % double: define the desired level of safety
        perc_tol;               % double: specify the percentage of the states, from which a Markov policy meeting the desired level of safety exists
        buffer_length;          % double: size of buffer for fix-point operation when synthesizing safety advisor
        
        % only have to be specified when terminate_type = 2
        time_horizon;           % double: time horizon for the Bellmn interation, it must be specified for n-accepted property, -1 means unspecified
    end
    
    methods
        %% initializtion
        function obj = SVA_program_QbLSt_options()
            % SVA_program_QbLSt_options: Construct an instance of a program
            % for synthesizing safe-visor architecture based on Lifting (Approximate Probabilistic Relations)
            obj.core_num = 1;
            obj.en_para = 0;
            obj.spec_type = -1;
            obj.terminate_type = -1;
            obj.max_unsafe_prob = -1;
            obj.buffer_length = 1000;
            obj.MDP_option =0;
        end
        
        %% Constructing MDP for the backward recursion
        function obj = construct_MDP(obj,method)
            % define a new object for computing MDP
            % input
            %       method: indicating whether the MDP shpuld be
            %               synthesized 
            %               method = 0: the stochastic kernel should not be
            %               synthesized
            %               method = 1: the stochastic kernel should be
            %               synthesized
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
            
            
            MDP_program = construct_MDP_options();
            
            % define necessary parameters for constructing MDP
            MDP_program.phy_system = obj.phy_system ;
            MDP_program.np_x = obj.np_x;
            MDP_program.np_u = obj.np_u;
            if obj.isgame == 1
                MDP_program.np_w = obj.np_w;
                MDP_program.isgame = 1;
            end
            
            % MDP construction
            MDP_program = MDP_program.discretization();
            % move option and data for fetching data from the MDP
            MDP_program.MDP.option = obj.MDP_option;
            MDP_program.MDP.sdef_par = obj.sdef_par;
%             MDP_program = MDP_program.lifting_elim_input(obj.lifting);
            
            if method == 1
                MDP_program = MDP_program.MDP_generate;
            end
            % save MDP to the object
            obj.MDP = MDP_program.MDP;
        end
               
        %% Synthesizing safe_visor
        function [safe_visor,obj]= syn_safe_visor(obj)
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
            
            if ~(obj.spec_type == 1|| obj.spec_type == 2 || obj.spec_type == 3)
                error('Unacceptable specification type.');
            end
            
            % compute epmap for backward iteration
            %obj = obj.compute_epmap();
            
            % compute the NaN-1 map for the initial state
            %obj = obj.compute_init();
            
            disp(['Pre-process complete!.']);
            safe_visor = QbLSt_sva_options();
            % inheriting mecessary varables
            safe_visor.DFA = obj.DFA;                    
            safe_visor.phy_system = obj.phy_system;             
            safe_visor.hx = obj.MDP.hx;               
            safe_visor.hu = obj.MDP.hu;      
            safe_visor.MDP = obj.MDP;
            safe_visor.np_x = obj.np_x;                  
            safe_visor.epmap = obj.epmap;
            safe_visor.init_state = obj.init_state;
            safe_visor.max_unsafe_prob = obj.max_unsafe_prob;
            safe_visor.acc_map = obj.acc_map;
            safe_visor.lifting = obj.lifting;
            safe_visor.spec_type = obj.spec_type;
            safe_visor.isgame = obj.isgame;
            safe_visor.hw = obj.MDP.hw;
            safe_visor.np_w = obj.np_w;
            
            % ==================== Start iteration =================          
            % variable definitions
            num_state = obj.MDP.n_x*(obj.DFA.n_state-1);
            num_input = obj.MDP.n_u;
            num_w = obj.MDP.n_w;
            
            v = zeros(num_state,obj.buffer_length,'single');     % auxiliary matrix for saving value function at each time instant
            u_v = zeros(num_state,obj.buffer_length,'single');   % auxiliary matrix for saving optimal safety policy at each time instant
            temp_u = zeros(num_input,1);                % auxiliary array for calculating optimal safety policy
            pot = 1;                                    % pointer for the matrix for value function and the optimal policy
            
            % vector for initial value for value function: v=1 when q in F
            v0 = zeros(num_state,1);
            for kk = 1:1:length(obj.DFA.acc_state)
                v0(1+obj.MDP.n_x*(obj.DFA.acc_state(kk)-2):obj.MDP.n_x*(obj.DFA.acc_state(kk)-1),1)=ones(obj.MDP.n_x,1);
            end
            v(:,pot) = v0;
            u_v(:,pot) = v0;
            %% synthesizing the safety advisor as discussion in Definition 3
            tic
            while 1
                % initialize the value function vector and the safety input vector in the current
                % iteration
                v(:,pot+1) = v0;
                u_v(:,pot+1) = v0;
                
                % reshape the v column for iteration
                v_nxt = reshape(v(:,pot),[obj.MDP.n_x,obj.DFA.n_state-1]);
                
                % interation for the fix-point operation
                for k = 2:1:obj.DFA.n_state
                    % go through all states except for the initial state 
                    
                    if obj.DFA.isacc_state(k)
                        % if the current state is an accepting state, no
                        % iteration is needed, skip this round
                        continue;
                    end
                    
                    % prepare the v vector for backward iteration
                    if obj.spec_type == 1 || obj.spec_type==2
                        v_iter = max(v_nxt.*obj.epmap{k},[],2);
                    elseif obj.spec_type==3
                        
                    end
                    
                    for i=1:1:obj.MDP.n_x
                        % go through each states
                        for j=1:1:num_input
                            % go through each possible inputs
                            temp_w = zeros(num_w,1);
                            for iw = 1:1:num_w
                                % go through all w
                                row = obj.MDP.MDP_get(i,j,iw);
                                temp_w(iw,1)=row*v_iter;
                            end
                            % select appropriate input based on controller type
                            if obj.spec_type == 1 || obj.spec_type==2
                                 % select the input w that maximises the
                                 % value function for the current u
                                  temp_u(j,1)=max(temp_w);
                            elseif obj.spec_type==3

                            end
                            %temp_u(j,1)=obj.MDP.sto_kernel(i+(j-1)*obj.MDP.n_x,:)*v_iter;
                        end
                        
                        % select appropriate input based on controller type
                        if obj.spec_type == 1 || obj.spec_type==2
                            % select the input that minimises the value function
                            [temp_v,t_p]=min(temp_u);
                            % operator for interation regarding delta
                            temp_v = (1-obj.lifting.delta)*temp_v+obj.lifting.delta;
                        elseif obj.spec_type==3
                            
                        end
                        % if the entry of value matrix is bigger than 1 due to numerical error,
                        % set it to 1
                        if temp_v>1
                            temp_v=1;
                        end
                        % save the INDEX of the optimal inputs and the
                        % corresponding value of value function
                        v(i+obj.MDP.n_x*(k-2),pot+1) = temp_v;
                        u_v(i+obj.MDP.n_x*(k-2),pot) = t_p;
                    end
                end
                            
                % exist the iteration when it already converges
                compare_v = v(:,pot+1)-v(:,pot);
                if (min(compare_v)==0) && (max(compare_v)==0)
                    disp(['The iteration converges!.']);
                    test_time=pot;
                    break;
                end
                
                % reshape the result in the current round
                v_int = reshape(v(:,pot+1),[obj.MDP.n_x,obj.DFA.n_state-1]);
                for kk = 1:1:length(obj.DFA.acc_state)
                    % replace the column with accepting state as one
                    v_int(:,obj.DFA.acc_state(kk)-1) = ones(obj.MDP.n_x,1);
                end
                
                % calculate the percentage of states from for which a Markov policy which
                % fulfills the requirement exists
                if obj.spec_type == 1 || obj.spec_type==2
                    v_int2 = max(v_nxt.*obj.epmap{1},[],2);
                    c_v = obj.init_state.*v_int2;
                    num_fea = sum(c_v<obj.max_unsafe_prob)/obj.num_init_state;
                    
                    % let users aware of the progress
                    disp(['Time horizon: ',num2str(pot),', percentage of fea-state:',num2str(num_fea*100),'%, min v: ',num2str(min(c_v))]);
                elseif obj.spec_type==3

                end

                
                if obj.terminate_type == 1 && ((num_fea <= obj.perc_tol && (obj.spec_type ==1 || obj.spec_type ==2 ))||(num_fea >= 1- obj.perc_tol && obj.spec_type ==3))
                    % decide whether the calculation should stop according to the
                    % percentage
                    % calculate the time horizon of the safety advisor
                    test_time=pot-1;        % u(:,pot) is not qualified, so test_time is set to the previous time instant
                    break;
                elseif obj.terminate_type == 2 && pot >= obj.time_horizon
                    % decide whether the calculation should stop according to
                    % pre-defined time horizon
                    test_time=pot;
                    break;
                end
                
                % update the pointer
                pot=pot+1;
            end
            
            % obtain the matrix for value function and optimal safety policy from the
            % auxiliary matrices
            value_matrix = fliplr(v(:,1:test_time+1));
            u_matrix = fliplr(u_v(:,1:test_time));
            
            time_consume = toc;                            % record time consumed
            disp(['Time for synthesisizing safety advisor: ',num2str(time_consume),' seconds']);
            disp(['Time horizon of safety advisor: ',num2str(test_time),'.']);
            
            % save synthesis result 
            safe_visor.value_matrix = value_matrix;
            safe_visor.safety_advisor = u_matrix;
            safe_visor.test_time = test_time;
        end
        
        %% Compute epmap according to DFA_epmap
        function obj = compute_epmap(obj)
            % compute epmap according to DFA_epmap
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
            
            eps = obj.epsilon;
            
            % if parallel computing is enable, check whether works have
            % already been created
            if obj.en_para == 1
                poolobj = gcp('nocreate');% If no pool,  create new one.
                if isempty(poolobj)
                    parpool(obj.core_num);
                else
                    disp('Already initialized for computing accmap!');
                end
            end
            
            % initialize data for computing
            DFA_m = obj.DFA_epmap;
            n_x = obj.MDP.n_x;
            hx = obj.MDP.hx;
            
            % start computing epmap
            for i = 1:1:DFA_m.n_state
                % check each state of the DFA
                
                % initialize an epmap
                ep_temp = NaN*ones(n_x,DFA_m.n_state);
                
                if DFA_m.isacc_state(i) ==1 || i == DFA_m.sink_state|| DFA_m.istrap_state(i) ==1 
                    % when current state is an accepting state/ sink state/
                    % trap state
                    ep_temp(:,i) = ones(n_x,1);              % set one of the column according to the definition of accepting state  
                else
                    % when current state is not an accepting state/sink
                    % state/trap state
                    for j = 1:1:DFA_m.n_state
                        % chenck each state of the DFA
                        if ~isempty(DFA_m.dfa{i,j})
                            % if there is an transition from state i to
                            % state j
                            cur_con = DFA_m.dfa{i,j};                       % get Lmap from state i to state j
                            slice_xp = NaN*ones(n_x,1);                     % slice vector for the parfor
                            
                            if obj.en_para
                                % if parallel computing is enabled
                                parfor pi = 1:n_x-1
                                    cur_y = QbLSt_abs_output(hx(:,pi));     % output for the current state of the abstraction system
                                    iters = NaN;
                                    for k = 1:1:length(cur_con)
                                        % checking each constraints in cur_con
                                        y = sdpvar(length(cur_y),1);        % define varibale
                                        cons = (y-cur_y)'*(y-cur_y)<=eps^2;   % epsilon sphere arround cur_con
                                        conn = cur_con{k}.A*y<=cur_con{k}.b;
                                        con = conn+cons;                    % cinstruct the current constraint
                                        ops = sdpsettings('verbose', 0);
                                        r = optimize(con,'',ops);
                                        if r.problem == 0
                                            % itersection exists
                                            iters = 1;
                                            break;
                                            %k = length(cur_con)+1;         % so that the current k-loop can be ended
                                        end
                                    end
                                    slice_xp(pi,1) = iters;
                                end
                            else
                                % do not use parallel computing
                                for pi = 1:n_x-1
                                    cur_y = QbLSt_abs_output(hx(:,pi));     % output for the current state of the abstraction system
                                    iters = NaN;
                                    for k = 1:1:length(cur_con)
                                        % checking each constraints in cur_con
                                        y = sdpvar(length(cur_y),1);        % define varibale
                                        cons = (y-cur_y)'*(y-cur_y)<=eps^2;   % epsilon sphere arround cur_con
                                        conn = cur_con{k}.A*y<=cur_con{k}.b;
                                        con = conn+cons;                    % cinstruct the current constraint
                                        ops = sdpsettings('verbose', 0);
                                        r =optimize(con,'',ops);
                                        if r.problem == 0
                                            % itersection exists
                                            iters = 1;
                                            break;
                                        end
                                    end
                                    slice_xp(pi,1) = iters;
                                end
                            end
                            % save the current column for ep_temp
                            ep_temp(:,j) = slice_xp;
                        end
                    end
                    ep_temp(n_x,DFA_m.sink_state) = 1;                      % deal with the sink state
                end
                map{i} = ep_temp(:,2:DFA_m.n_state);                       % save the map for state i
            end
            
            % finally get the emp map
            obj.epmap = map;  
        end
        
        %% Compute NaN-1 map for initial state
        function obj = compute_init(obj)
            % compute NaN-1 map for initial state according to phy_system
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
            
            
            % obtain the variable list 
            phy_var = obj.phy_system.phy_var;
            
            % initialize the NaN-1 map
            init_temp = NaN*ones(obj.MDP.n_x,1);
            num_init = 0;
            
            for i = 1:1:obj.MDP.n_x-1
                x = obj.MDP.hx(:,i);
                ind = 0;
                eval(obj.phy_system.initial_state_set)
                if ind ==1
                    init_temp(i,1) = 1;
                    num_init = num_init + 1;
                end
            end
            
            % save the result
            obj.init_state = init_temp;           
            obj.num_init_state = num_init;        
        end
        
        %% compute the 0-1 map for the 
        function obj = compute_accmap(obj)
            %   compute_accmap: compute the state map for accepting state
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
            
            
            if obj.spec_type == 1 || obj.spec_type==2
                % if the specification is a safe-LTLF properties
                obj = obj.safe_compute_accmap();
            elseif obj.spec_type==3

            end
        end
        
        %% accmap for safe-LTL properties
        function obj = safe_compute_accmap(obj)
            %   compute_accmap: compute the state map for accepting state
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
            
            
            % initialize accmap
            accmap = zeros(obj.MDP.n_x,obj.DFA.n_state);
            
            for i = 1:1:obj.DFA.n_state-1
                temp_map = ones(obj.MDP.n_x,1);
                for j = 1:1:length(obj.DFA.acc_state)
                    cur = obj.epmap{i}(:,obj.DFA.acc_state(j)-1);
                    cur(cur==1)=0;
                    cur(isnan(cur))=1;
                    temp_map = temp_map.*cur;
                end
                accmap(:,i) = temp_map;
            end
            obj.acc_map = accmap;
        end
             
    end
end

