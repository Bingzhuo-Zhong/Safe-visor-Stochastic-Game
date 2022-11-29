classdef construct_MDP_options
    % construct_MDP_options: Class for construction of finite MDP based on the physical systems
    % and the requirement for partition
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
        np_w;                   % vector of double: Row vector for number of partitions for internal (player 2) input, only for game, i.e., isgame = 1
        MDP;                    % class of MDP_options: the finite MDP for backward iteration
    end
    
    methods
        function obj = construct_MDP_options()
            % construct_MDP_options: Construct an instance of the class for
            % MDP construction
            obj.isgame = 0;
        end
        
       %% Constructing MDP
        function obj = MDP_generate(obj)
            %   MDP_generate is used for generating finitie MDP for the original system
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
 
                      
            %   Preserve spaces for the stochastic kernel of the finite MDP
            %   sto_kernel=zeros(n_u*(n_x+1),n_x+1,'single');
            sto_kernel=zeros(obj.MDP.n_w*obj.MDP.n_u*obj.MDP.n_x,obj.MDP.n_x,'single');        % allocate space for the stochastic kernel
            
            % the last row of the stochastic kernel for each input is the same,
            %so assign them in advance
            temp_last_row=zeros(1,obj.MDP.n_x);
            temp_last_row(1,obj.MDP.n_x) = 1;            
            for j = 1:1:obj.MDP.n_w
                for i=1:1:obj.MDP.n_u
                    sto_kernel(obj.MDP.n_x*(i+(j-1)*obj.MDP.n_u),:)=temp_last_row;
                end
            end

            % configure the error tolerance for mvncdf
            options = statset('TolFun',1e-30);
            
            % computing stochastic matrix
            tic
            for iw = 1:1:obj.MDP.n_w
                for iu=1:1:obj.MDP.n_u
                % go through all possible inputs
                    for ix=1:1:obj.MDP.n_x-1
                        % go through all states

                        %compute the norminal value of the target point beasd on different
                        %way to define system dy different way
                        %eval(mean_cur_code);
                        if obj.isgame ==1
                            % if this is a game, input w needs to be
                            % considered
                            mean_cur = QbLSt_abssys_norm(obj.MDP.hx(:,ix),obj.MDP.hu(:,iu),obj.MDP.hw(:,iw));
                        else
                            mean_cur = QbLSt_abssys_norm(obj.MDP.hx(:,ix),obj.MDP.hu(:,iu));
                        end
                        
                        % compute the value of states in the safety set
                        sto_kernel(ix+(iu-1)*obj.MDP.n_x+(iw-1)*obj.MDP.n_u*obj.MDP.n_x,1:obj.MDP.n_x-1) = mvncdf(obj.MDP.x_lower_bound',obj.MDP.x_upper_bound',mean_cur',obj.phy_system.Cov,options);

                        % compute the value of te "sink" state (unsafe set)
                        sto_kernel(ix+(iu-1)*obj.MDP.n_x+(iw-1)*obj.MDP.n_u*obj.MDP.n_x,obj.MDP.n_x)=1-sum(sto_kernel(ix+(iu-1)*obj.MDP.n_x+(iw-1)*obj.MDP.n_u*obj.MDP.n_x,1:obj.MDP.n_x-1));
                        if sto_kernel(ix+(iu-1)*obj.MDP.n_x+(iw-1)*obj.MDP.n_u*obj.MDP.n_x,obj.MDP.n_x)<0
                            % when the probability of the sink state is smaller than 0
                            % due to numerical error, set it as zero
                            sto_kernel(ix+(iu-1)*obj.MDP.n_x+(iw-1)*obj.MDP.n_u*obj.MDP.n_x,obj.MDP.n_x)=0;
                        end
                    end
                end
            end
            
            time_consume = toc;
            disp(['Time for building MDP: ',num2str(time_consume),' seconds']);
            
            % update the MDP
            obj.MDP.sto_kernel = sto_kernel;
        end
        
        %% Discretizing state set and input set
        function obj = discretization(obj)
            %   discretization: discretizing the state set (and the input set) of the original system for generating finitie MDP 
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
            
            % checking the definition of system type
            if ~(obj.phy_system.sys_type ==1|| obj.phy_system.sys_type == 2)
                error('Unacceptable physical system type.');
            end
            
            % initialize an object for a MDP
            obj.MDP =MDP_options();
            
            % mean_cur_code = ['mean_cur =', obj.phy_system.targetFile,'(hx(:,ix),hu(:,iu));'];
            % ======================== Creating Uniform Grids on the safe set and the input space ===========================
            
            %% discretizing state space
            dim_x = length(obj.phy_system.x_l);                       % dimension of the state space being studied
            delta_x = (obj.phy_system.x_u - obj.phy_system.x_l)./ obj.np_x;    % State space discretization parameter
            
            % building the representative points on each dimension in the safety set
            for i=1:1:dim_x
                temp_x = obj.phy_system.x_l(i)+delta_x(i)/2:delta_x(i):obj.phy_system.x_l(i)+obj.np_x(i)*delta_x(i);    % calculating representative points in each dimension
                ele_x(i)={temp_x};                                                                                      % saving representative points in each dimension
                clear temp_x;
            end
            
            % Building representation points for state partition
            num_ele = length(ele_x{1});     % number of representative points in the first dimension of the state space
            f_x = ele_x{1};
            for j = 1:1:num_ele
                rp_x(j) = {f_x(j)};
            end
            
            for i=1:1:dim_x-1
                for j=1:1:num_ele
                    for k=1:1:length(ele_x{i+1})
                        temp_rp_x(k+(j-1)*length(ele_x{i+1})) = {[rp_x{j};ele_x{i+1}(k)]};  % combine representative points on different dimension
                    end
                end
                clear rp_x;
                rp_x = temp_rp_x;
                clear temp_rp_x;
                num_ele = length(rp_x);
            end
            
            % copying points in the cell to a normal matrix
            n_x  = length(rp_x);                                % number of representative point for in state space
            for i=1:1:n_x
                hx(:,i) = rp_x{i};                              % coppying points into the matrix
                x_lower_bound(:,i) = hx(:,i) - 1/2*delta_x';    % generate the lower bound of the grid in state space
                x_upper_bound(:,i) = hx(:,i) + 1/2*delta_x';    % generate the upper bound of the grid in state space
            end
            
            % saving the result for the discretize state space
            obj.MDP.n_x = n_x+1;
            obj.MDP.delta_x = delta_x;
            obj.MDP.dim_x = dim_x;
            obj.MDP.hx = hx;
            obj.MDP.x_lower_bound = x_lower_bound;
            obj.MDP.x_upper_bound = x_upper_bound;
            
            %% discretizing the input space
            if obj.phy_system.sys_type == 1
                dim_u = length(obj.phy_system.u_l);                              % dimension of the input space being studied
                delta_u = (obj.phy_system.u_u - obj.phy_system.u_l)./ obj.np_u;  % Input space discretization parameter
                
                %building the representative points on each dimension in input space
                for i=1:1:dim_u
                    temp_u = obj.phy_system.u_l(i)+delta_u(i)/2:delta_u(i):obj.phy_system.u_l(i)+obj.np_u(i)*delta_u(i);    % calculating representative points in each dimension
                    ele_u(i)={temp_u};                                                                                      % saving representative points in each dimension
                    clear temp_u;
                end
                
                % Building representation points for input partition
                num_ele = length(ele_u{1});
                f_u=ele_u{1};
                for j=1:1:num_ele
                    rp_u(j)={f_u(j)};
                end
                
                for i=1:1:dim_u-1
                    for j=1:1:num_ele
                        for k=1:1:length(ele_u{i+1})
                            temp_rp_u(k+(j-1)*length(ele_u{i+1})) = {[rp_u{j};ele_u{i+1}(k)]};
                        end
                    end
                    clear rp_u;
                    rp_u = temp_rp_u;
                    clear temp_rp_u;
                    num_ele = length(rp_u);
                end
                
                % copying points in the cell to a normal matrix
                n_u  = length(rp_u);    %number of representative point for in input space
                
                % coppying representative points into the matrix
                for i=1:1:n_u
                    hu(:,i) = rp_u{i};
                end
            elseif obj.phy_system.sys_type == 2
                dim_u = length(obj.phy_system.u_l);               % dimension of the input space being studied
                n_u  = length(obj.phy_system.hu);                 % number of representative point for in input space
                hu = obj.phy_system.hu;
                delta_u = inf;
            end
            
            % saving the result for the discretize (external) input space
            obj.MDP.n_u = n_u;
            obj.MDP.hu = hu;
            obj.MDP.delta_u = delta_u;
            obj.MDP.dim_u = dim_u;
            
            %% discretizing the internal (player 2) input space, if it is a
            % game
            if obj.isgame == 1
                dim_w = length(obj.phy_system.w_l);                       % dimension of the state space being studied
                delta_w = (obj.phy_system.w_u - obj.phy_system.w_l)./ obj.np_w;    % State space discretization parameter

                % building the representative points on each dimension in the safety set
                for i=1:1:dim_w
                    temp_w = obj.phy_system.w_l(i)+delta_w(i)/2:delta_w(i):obj.phy_system.w_l(i)+obj.np_w(i)*delta_w(i);    % calculating representative points in each dimension
                    ele_w(i)={temp_w};                                                                                      % saving representative points in each dimension
                    clear temp_w;
                end

                % Building representation points for state partition
                num_ele = length(ele_w{1});     % number of representative points in the first dimension of the state space
                f_w = ele_w{1};
                for j = 1:1:num_ele
                    rp_w(j) = {f_w(j)};
                end

                for i=1:1:dim_w-1
                    for j=1:1:num_ele
                        for k=1:1:length(ele_w{i+1})
                            temp_rp_w(k+(j-1)*length(ele_w{i+1})) = {[rp_w{j};ele_w{i+1}(k)]};  % combine representative points on different dimension
                        end
                    end
                    clear rp_w;
                    rp_w = temp_rp_w;
                    clear temp_rp_w;
                    num_ele = length(rp_w);
                end

                % copying points in the cell to a normal matrix
                n_w  = length(rp_w);                                % number of representative point for in state space
                for i=1:1:n_w
                    hw(:,i) = rp_w{i};                              % coppying points into the matrix
                end
                
                % saving the result for the discretize internal (player 2)
                % input space
                obj.MDP.n_w = n_w;
                obj.MDP.hw = hw;
                obj.MDP.delta_w = delta_w;
                obj.MDP.dim_w = dim_w;
            end     
        end
        
        %% Checking and eliminating unqualified input
        function obj = lifting_elim_input(obj,lifting)
            %   lifting_elim_input: eliminating the input set of the MDP 
            %                       according to the interface function 
            %                       associated with the lifting relation 
            %                       between the original system and the 
            %                       abstraction. 
            % Input:
            %       lifting: class of slprssys_options defining the
            %                approximate probabilistic relation.       
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
           
            % variable necessary for checking the feasibility of the inputs
            dim_x = size(lifting.A,2);
            dim_xab = obj.MDP.dim_x;
            u_var = obj.phy_system.u_crange;
            xab_var = obj.phy_system.x_range;
            
            % set the input constraints
            inputcon = ['con4 = u_var{i}.A*(',lifting.intf_text,')<=u_var{i}.b;'];
            
            % initialize the vector for qualified inputs and its pointer
            u_new = zeros(obj.MDP.dim_u,obj.MDP.n_u);
            u_pot = 0;
            
            tic
            for j = 1:1:obj.MDP.n_u
                % check all input
                
                % obtain the current input
                u_hat = obj.MDP.hu(:,j);
                
                % initialize the flag for examing an input
                u_flag = 1;
                
                for i = 1:1:length(u_var)
                    % define variables
                    x = sdpvar(dim_x,1);
                    x_hat = sdpvar(dim_xab,1);
                    
                    % constraint for epsilon
                    con1 = (x-lifting.P*x_hat)'*lifting.M*(x-lifting.P*x_hat)<=lifting.epsilon^2;
                    
                    % constraint for the state set of the abstraction system
                    con2 = xab_var.A*x_hat<=xab_var.b;
                    
                    % constraint for the state set of the original system
                    %con3 = xab_var.A*x<=xab_var.b;
                    
                    % constraint for the input
                    eval(inputcon);
%                     con3 = u_var{i}.A*(lifting.R_u*u_hat+lifting.Q*x_hat+lifting.K*(x-lifting.P*x_hat))<=u_var{i}.b;
                    
                    % define constraints
                    % con = con1+con2+con3+con4;
                    con = con1+con2+con4;
                    
                    ops = sdpsettings('verbose',0);
                    result = optimize(con,'',ops);
                    if result.problem == 0
                        u_flag = 0;
                        break;
                    end
                end
                if u_flag ==1
                    % current input pass the examination
                    u_pot = u_pot +1;
                    u_new(:,u_pot) = u_hat;
                end
            end
            time_consume = toc;
            
            % report result
            disp(['Elimination of input set is done after ',num2str(time_consume),' seconds. ',num2str(u_pot),'/',num2str(obj.MDP.n_u),' inputs are reserved.'])
            
            % update input set as well as the number of feasible input of
            % the MDP
            if u_pot == 0
                obj.MDP.n_u = u_pot;
                obj.MDP.hu = [];
            else
                obj.MDP.n_u = u_pot;
                obj.MDP.hu = u_new(:,1:u_pot);
            end
        end
        
    end
end

