classdef slprssys_apr_options
    % Class of SLope-ReStricted SYStem (Approximate Probabilitic Relation)
    % Including information for the original system, (infinite)
    % abstraction, quantization error and lifting relation between the
    % original system and the abstraction
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
        % parameters for dynamics of the original system
        A;                  % matrix of double: dynamics matrix A of the system
        B;                  % matrix of double: external input matrix B of the system
        C;                  % matrix of double: output matrix C of the system
        D;                  % matrix of double: internal matrix of D of the system (for game)
        R;                  % matrix of double: stochastic matrix of the system
        E;                  % cell of matrix of double: dynamic matrix E of the nonlinear dynamics
        F;                  % cell of matrix of double: matrix F fot the state variables in the nonlunear dynamics
        b;                  % cell of matix of double: upper and lower bound of the lipschitz constant for the nonlinear dynamics
        % parameters for infinite abstraction
        P;                  % matrix of double: abstraction matrix P
        P_w;                % matrix of double: abstraction matrix P_w for internal (player 2) input
        Ar;                 % matrix of double: dynamic matrix A of the abstraction system
        Br;                 % matrix of double: external input matrix B of the abstraction system
        Cr;                 % matrix of double: output matrix C of the abstraction system
        Dr;                 % matrix of double: internal input matrix D of the abstraction system(for game)
        Er;                 % cell of matrix of double: dynamic matrix E of the nonlinear dynamics
        Fr;                 % cell of matrix of double: matrix F fot the state variables in the nonlunear dynamics
        Rr;                 % matrix of double: stochastic matrix for the abstraction of the system
        Q;                  % matrix of double: matrix Q for infinite abstraction
        S;                  % matrix of double: matrix S for infinite abstraction
        G;                  % cell of matrix of double: L1 - L2
        % region of interested
        x_l;                % (row) vector of double: lower bounds of the region of interested             
        x_u;                % (row) vector of double: upper bounds of the region of interested 
        np_x;               % (row) vector of double: number of partitions in each dimension of the region of interested
        hx;                 %  matrix of double: state set of the abstraction
        % control input
        u_l;                % (row) vector of double: lower bounds of the control input set
        u_u;                % (row) vector of double: upper bounds of the control input set
        np_u;               % (row) vector of double: number of partitions in each dimension of the control input set
        hu;                 %  matrix of double: control input set of the abstraction
        u_area;
        uab_area;           %  class of Lmap_options: expected inputs in the control input set that can be applied for synthesis
        input_con;          %  class of Lmap_options: input constraint for compensating the error between states of the original system and the abstraciton 
        % disturbance input
        w_l;                %  (row) vector of double: lower bounds of the disturbance input set
        w_u;                %  (row) vector of double: upper bounds of the disturbance input set
        np_w;               %  (row) vector of double: number of partitions in each dimension of the disturbance input set
        hw;                 %   matrix of double: disturbance input set of the abstraction
        
        % variables for lifting relation
        K;                  % matrix of double: matrix K for the interface function
        L;                  % matrix of double: matrix L for the interface function
        R_u;                % matrix of double: R_slang for the interface function
        M;                  % matrix of double: matrix M in approximate probabilistic relation
        M_w;                % matrix of double: matrix M_w in approximate probabilistic relation for the internal input
        intf_text;          % text: expression of the interface function
        epsilon;            % double: epsilon in approximate probabilistic relation
        epsilon_w;          % double: epsilon for the relation of the internal input (for game)
        delta;              % double: probability lose in approximate probabilistic relation
        beta;               % vector of double: dicretization parameters
        gamma0;             % double: discount attributed to the quantization of disturbance input
        gamma1;             % double: discount attributed to the model order reduction in the control input
        gamma2;             % double: discount attributed to the model order reduction in the noise input
        gamma3;             % double: discount attributed to the quantization of the state space
        gamma4;             % double: discount attributed to the model order reduction in the distucbance input
        gamma_total;        % double:           sum of gamma0 to gamma 4
        sol_list;           % vector of struct: result for searching solutions
        % others
        isgame;             % isgame: 1 the system is model by gDTSG; 0 otherwise;  (for game)
        result;             % recording result for S-procedure
        solver;             % solver for LMI problem
        num_core;           % interger: number of core used for parallel computation
        en_para;            % integer: flag indicating whether parallel computing is feasible.
                            %    en_para = 0: parallel computing is disabled (Default)
                            %    en_para = 1: parallel computing is enabled
    end
    
    methods
        %% define new object
        function obj = slprssys_apr_options()
            obj.isgame = 0;             % the system is not a game by default (for game)
            obj.solver = 'SDPT3';       % use SDPT3 as default solver
            obj.intf_text = 'lifting.Q*x_hat+lifting.R_u*u_hat';
            obj.num_core = 1;
            obj.en_para = 0;
        end
        
        %% ===== Computing approximate probabilistic relation =========
        %% main function for infinite abstraction
        function obj = inf_abs(obj)
            % compute infinite abstraction based on the selection of P
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
            
            
            %  check the variables which are necessary for the infinite abstraction
            if isempty(obj.P)
                error('Please specify P for the infinite abstraction!')
            elseif isempty(obj.A)
                error('Please provide A for the original system!')
            elseif isempty(obj.B)
                error('Please provide B for the original system!')
            elseif isempty(obj.C)
                error('Please provide C for the original system!')
            elseif isempty(obj.R)
                error('Please provide R for the original system!')
            elseif isempty(obj.Br)
                error('Please provide Br for the abstraction!')
            end
            
            % solve Cr
            obj.Cr = obj.C *obj.P;
            
            % solve Ar and Q
            obj = obj.inf_Ar();
            
            if ~isempty(obj.E)
                if isempty(obj.F)
                    error('Please provide F for original system!')
                end
                % solve Fr
                temp_Fr = cell(length(obj.F),1);
                for i = 1:1:length(obj.F)
                    temp_Fr{i,1} = obj.F{i};
                end
                
                % solve L2 and Er
                obj = obj.inf_Er();
            end
            
            % solve Dr and S, when needed
            if obj.isgame == 1
               obj = obj.inf_Dr();
            end
            
            % initialize R_u
            obj = obj.init_R_u();
            
            % report result
            disp('Computation of infinite abstraction is done.');
        end
        
        %% Computing Ar and Q for the infinite abstraction
        function obj = inf_Ar(obj)
            % compute Q (and Ar) for the infinite abstraction infinite abstraction
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
            
            
            % define necessary variables for solving the LMI problem
            if isempty(obj.Ar)
                Ar_t = sdpvar(size(obj.P,2),size(obj.P,2));
            else
                Ar_t = obj.Ar;
            end
            Q_t = sdpvar(size(obj.B,2),size(obj.P,2));
            
            % define constraint
            con = obj.A*obj.P == obj.P*Ar_t-obj.B*Q_t;
            
            % mute
            ops = sdpsettings('verbose',0);
            ops.solver = obj.solver;
            
            % solve the equation
            optimize(con, '', ops);
            
            % get the answer
            if isempty(obj.Ar)
                obj.Ar = value(Ar_t);
            end
            obj.Q = value(Q_t);
            
            % check error
            %             err = obj.A*obj.P - (obj.P*obj.Ar-obj.B*obj.Q);
            %             if sqrt(err'*err)<obj.tol_err
            %                 disp(['Q is founded! Error is ',num2str(sqrt(err'*err)),'.'])
            %             else
            %                 disp('No appropriate Q is found!')
            %             end
        end
        
        %% Computing Er and L2
        function obj = inf_Er(obj)
            % compute L2 (and Er) for the infinite abstraction infinite abstraction
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
            
            
            if isempty(obj.Er)
                % if Er have note been pre-computed
                Er_temp = cell(length(obj.E),1);
                G_temp = cell(length(obj.E),1);
                for i = 1:1:length(obj.E)
                    % define necessary variables for solving the LMI problem
                    Er_t = sdpvar(size(obj.P,2),size(obj.P,2));
                    G_t = sdpvar(size(obj.B,2),size(obj.E{i},2));
                    
                    % define constraint
                    con = obj.E{i}== obj.P*Er_t-obj.B*G_t;
                    
                    % mute
                    ops = sdpsettings('verbose',0);
                    ops.solver = obj.solver;
                    
                    % solve the equation
                    optimize(con, '', ops);
                    
                    % get the answer
                    Er_temp{i} = value(Er_t);
                    G_temp{i} = value(G_t);
                end
                obj.Er = Er_temp;
                obj.G = G_temp;
            else
                % in case that Er have already been pre-computed
                G_temp = cell(length(obj.E),1);
                for i = 1:1:length(obj.E)
                    % define necessary variables for solving the LMI problem
                    G_t = sdpvar(size(obj.B,2),size(obj.E{i},2));
                    
                    % define constraint
                    con = obj.E{i} == obj.P*obj.Er{i}-obj.B*G_t;
                    
                    % mute
                    ops = sdpsettings('verbose',0);
                    ops.solver = obj.solver;
                    
                    % solve the equation
                    optimize(con, '', ops);
                    
                    % get the answer
                    G_temp{i} = value(G_t);
                end
                obj.G = G_temp;
            end
        end
        
        %% Computing Dr and S for the infinite abstraction
        function obj = inf_Dr(obj)
            % compute S (and Dr) for the infinite abstraction infinite
            % abstraction when target system is a game
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
            
            
            % define necessary variables for solving the LMI problem
            if isempty(obj.Dr)
                Dr_t = sdpvar(size(obj.P,2),size(obj.P,2));
            else
                Dr_t = obj.Dr;
            end
            S_t = sdpvar(size(obj.B,2),size(obj.D,2));
            
            % define constraint
            con = obj.D*obj.P_w == obj.P*Dr_t-obj.B*S_t;
            
            % mute
            ops = sdpsettings('verbose',0);
            ops.solver = obj.solver;
            
            % solve the equation
            optimize(con, '', ops);
            
            % get the answer
            if isempty(obj.Dr)
                obj.Dr = value(Dr_t);
            end
            obj.S = value(S_t);
        end
        
        %% Computing Rr
        function Rr = inf_Rr(obj,sol)
            % compute Rr for the infinite abstraction infinite abstraction
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
            
            
            % define variables to be optimized
            R_temp = sdpvar(size(obj.P,2),size(obj.R,2));
            s = sdpvar(1,1);
            
            R_bar = obj.R - obj.P*R_temp;
            N = sqrtm(sol.M);
            R_barN = N*R_bar;
            
            % define object function
            fun = s;
            
            % define constraint
            constraints = [s*eye(size(R_barN,1),size(R_barN,1)) R_barN;R_barN' s*eye(size(R_barN,2),size(R_barN,2))]>=0 ;
            
            % optimization
            ops = sdpsettings('verbose',0);
            ops.solver = obj.solver;
            optimize(constraints, fun, ops);
            
            % get optimal value of K
            Rr = value(R_temp);
        end
        
        %% Configure Br and compute R_slang accroding to Br
        function obj = init_R_u(obj)
            % Configure Br and compute R_slang for the interface function accroding to Br
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
            
            % define variables to be optimized
            Ru_temp = sdpvar(size(obj.B,2),size(obj.Br,2));
            s = sdpvar(1,1);
            
            Ru_bar = obj.B*Ru_temp - obj.P*obj.Br;
            
            % define object function
            fun = s;
            
            % define constraint
            constraints = [s*eye(size(Ru_bar,1),size(Ru_bar,1)) Ru_bar;Ru_bar' s*eye(size(Ru_bar,2),size(Ru_bar,2))]>=0 ;
            
            % optimization
            ops = sdpsettings('verbose',0);
            ops.solver = obj.solver;
            optimize(constraints, fun, ops);
            
            % get optimal value of K
            obj.R_u = value(Ru_temp);
        end
        
        %% Discretizing state set and input set
        function obj = discretization(obj)
            %   discretization: discretizing the state set (and the input
            %   set) of the abstraction
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
            
            
            %% discretizing state space
            dim_x = length(obj.x_l);                       % dimension of the state space being studied
            delta_x = (obj.x_u - obj.x_l)./ obj.np_x;    % State space discretization parameter
            obj.beta =  0.5*delta_x';
            
            % building the representative points on each dimension in the safety set
            for i=1:1:dim_x
                temp_x = obj.x_l(i)+delta_x(i)/2:delta_x(i):obj.x_l(i)+obj.np_x(i)*delta_x(i);    % calculating representative points in each dimension
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
                h_x(:,i) = rp_x{i};                              % coppying points into the matrix
            end
            
            % saving the result for the discretize state space
            obj.hx = h_x;
            
            
            %% discretizing the input space
            if isempty(obj.hu)
                dim_u = length(obj.u_l);                              % dimension of the input space being studied
                delta_u = (obj.u_u - obj.u_l)./ obj.np_u;  % Input space discretization parameter
                
                %building the representative points on each dimension in input space
                for i=1:1:dim_u
                    temp_u = obj.u_l(i)+delta_u(i)/2:delta_u(i):obj.u_l(i)+obj.np_u(i)*delta_u(i);    % calculating representative points in each dimension
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
                    h_u(:,i) = rp_u{i};
                end
            end
            
            % saving the result for the discretize (external) input space
            obj.hu = h_u;
            
            
            %% discretizing the internal (player 2) input space, if it is a game
            if obj.isgame == 1
                dim_w = length(obj.w_l);                        % dimension of the state space being studied
                delta_w = (obj.w_u - obj.w_l)./ obj.np_w;       % State space discretization parameter
                obj.epsilon_w = norm(obj.M_w)*norm(0.5*delta_w);
                
                % building the representative points on each dimension in the safety set
                for i=1:1:dim_w
                    temp_w = obj.w_l(i)+delta_w(i)/2:delta_w(i):obj.w_l(i)+obj.np_w(i)*delta_w(i);    % calculating representative points in each dimension
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
                    h_w(:,i) = rp_w{i};                              % coppying points into the matrix
                end
                
                % saving the result for the discretize internal (player 2)
                % input space
                obj.hw = h_w;
            end
        end
        
        %% eliminating the set of u_ab according to the expectation of the user
        function obj = elim_uab(obj)
            % eliminating the set of u_ab for synthesizing optimal
            % controller for the abstraction
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
            
            
            num_Lmap = length(obj.uab_area);
            
            if num_Lmap ~=0
                % when expected area has been configured
                uab_temp = obj.hu;
                num_uab = length(uab_temp);
                for i = 1:1:num_Lmap
                    % check every constraint
                    A_cur = obj.uab_area{i}.A;
                    b_cur = obj.uab_area{i}.b;
                    num = 0;
                    for j = 1:1:num_uab
                        % check every input given the current constraint
                        % for u
                        if A_cur*uab_temp(:,j)<=b_cur
                            num = num+1;
                            u_temp(:,num) = uab_temp(:,j);
                        end
                    end
                    
                    % save the u which remains given the current constraint
                    uab_temp = u_temp;
                    num_uab = num;
                end
                
                % save the expected input
                obj.hu = uab_temp;
            end
        end
        
        %% compute the input constraint for the LMI
        function obj = compute_inputcon(obj)
            % compute the input constraints for the LMI problem which
            % jointly solves K,M and L
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
            
            
            lifting = obj;

            A_u = obj.u_area.A;
            b_u = obj.u_area.b;
            
            % compute different b
            num_xab = size(obj.hx,2);
            if isequal(obj.P,eye(size(obj.A,2),size(obj.A,2)))
                % if there is no model order reduction, there is no effect
                % of Q
                num_xab = 1;
            end
            num_uab = size(obj.hu,2);
            b_matrix = zeros(size(b_u,1),num_xab*num_uab);
            concode = ['b_matrix(:,k) = b_u - A_u*(',lifting.intf_text,');'];
            
            for i = 1:1:num_xab
                x_hat = obj.hx(:,i);
                for j = 1:1:num_uab
                    u_hat = obj.hu(:,j);
                    k = j+(i-1)*num_uab;
                    eval(concode);
                end
            end
            
            % eliminate redundant constraints
            num_con = size(b_matrix,2);
            Poly = Polyhedron('A', A_u, 'b', b_matrix(:,1));
            tic
            for i = 2:1:num_con
                Poly_temp = Polyhedron('A', A_u, 'b', b_matrix(:,i));
                Poly = intersect (Poly,Poly_temp);
                Poly.minHRep();
            end
            time_cons = toc;
            disp(['Computation of input_con is finished within ',num2str(time_cons),' seconds.'])
                      
            
            % result
            inputcon = Lmap_options();
            inputcon.A = Poly.A;
            inputcon.b = Poly.b;
            obj.input_con = inputcon;
            
        end
        
        %% search posible solution given the range of epsilon
        function obj = search_solution(obj,eps_range,eps_num,kp_range,kappa_num)
            % compute the input constraints for the LMI problem which
            % jointly solves K,M and L
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
            
            
            tic
            if eps_range(2)<eps_range(1) || eps_range(1)<0 
                error('The range of epsilon is not acceptable!')
            elseif size(kp_range,2)==2 && (kp_range(2)<kp_range(1) || kp_range(1)<0 || kp_range(2)>1)
                error('The range of kappa is not acceptable!')
            end
            
            % formulate eps sample group
            step = (eps_range(2)-eps_range(1))/(eps_num-1);
            eps_sample = eps_range(1):step:eps_range(2);
            solutions = cell(eps_num,1);
            
            if obj.en_para
                % check the status of parellel pool
                poolobj = gcp('nocreate');% If no pool,  create new one.
                if isempty(poolobj)
                    parpool(obj.num_core);
                else
                    disp('Already initialized for searching relation');
                end
                
                parfor i = 1:eps_num
                    kappa_range = kp_range;
                    cur_sol = cell(3,1);
                    eps = eps_sample(i);
                    
                    if size(kappa_range,2)==1
                        % search the range of kappa
                        kappa_range = kp_range_cal(obj,eps);
                    end
                    
                    if kappa_range(2)>kappa_range(1)
                        % compute the solution within the range of kappa
                        stepk = (kappa_range(2)-kappa_range(1))/(kappa_num-1);
                        kappa_sample = kappa_range(1):stepk:kappa_range(2);
                        sol_group = [];
                        
                        for j = 1:1:kappa_num
                            kappa = kappa_sample(j);
                            sol = obj.solve_MK(kappa,eps);
                            if ~isnan(sol.M)
                                sol = obj.check_fea_MK(sol);
                                if isempty(sol_group)
                                    sol_group = sol;
                                else
                                    sol_group = [sol_group;sol];
                                end
                            end
                        end
                        cur_sol{1}      = eps;
                        cur_sol{2}      = kappa_range;
                        cur_sol{3}      = sol_group;
                    else
                        cur_sol{1}      = eps;
                        cur_sol{2}      = kappa_range;
                        cur_sol{3}      = [];
                    end
                    solutions{i,1}= cur_sol;
                end
                
            else
                for i = 1:eps_num
                    kappa_range = kp_range;
                    cur_sol = cell(3,1);
                    eps = eps_sample(i);
                    
                    if size(kappa_range,2)==1
                        % search the range of kappa
                        kappa_range = kp_range_cal(obj,eps);
                    end
                    
                    if kappa_range(2)>kappa_range(1)
                        % compute the solution within the range of kappa
                        stepk = (kappa_range(2)-kappa_range(1))/(kappa_num-1);
                        kappa_sample = kappa_range(1):stepk:kappa_range(2);
                        sol_group = [];
                        
                        for j = 1:1:kappa_num
                            kappa = kappa_sample(j);
                            sol = obj.solve_MK(kappa,eps);
                            if ~isnan(sol.M)
                                sol = obj.check_fea_MK(sol);
                                if isempty(sol_group)
                                    sol_group = sol;
                                else
                                    sol_group = [sol_group;sol];
                                end
                            end 
                        end
                        cur_sol{1}      = eps;
                        cur_sol{2}      = kappa_range;
                        cur_sol{3}      = sol_group;
                    else
                        cur_sol{1}      = eps;
                        cur_sol{2}      = kappa_range;
                        cur_sol{3}      = [];
                    end
                    solutions{i,1}= cur_sol;
                end
            end    
            
           % save the result
           for j = 1:1:eps_num
               temp_sol = solutions{j};
               csol.epsilon = temp_sol{1};
               csol.kappa_range = temp_sol{2};
               csol.fea = -1;
               csol.sol_group = temp_sol{3};
               csol.best_sol = csol.sol_group(1);
               final_sol(j) = csol;
           end
            obj.sol_list = final_sol;
            time_con = toc;
            
            disp(['The search for solutions is done in ',num2str(time_con),' seconds.']);
        end
        
        %% search the lower bound of kappa given epsilon
        function kappa_range = kp_range_cal(obj,epsilon)
            % compute the lower bound of kappa given epsilon
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
            
            
            max_iter = 12;          
            
            % initialize the search
            existence = 0;                  %   indicator for the existence of epsilon
            counter = 1;                    %   counter for number of iteration
            kappa = inf;                    %   initial value of epsilon
            kappa_max = 1;                  %   initialize the upper bound of the interval to be searched
            kappa_min = 0;                  %   initialize the lower bound of the interval to be searched
            kappa_temp = 1;
            
            while 1
                % check the feasibility of the current choice of kappa_temp
                sol_temp = obj.solve_MK(kappa_temp,epsilon);
                if isnan(sol_temp.kappa)
                    feas = 0;
                else
                    feas = 1;
                end
                
                if counter == 1
                    % first round is to check the feasibility when kappa == 1
                    if feas == 1 
                        kappa = kappa_temp; 
                        existence = 1;
                    end
                else
                    if feas == 1
                        kappa = kappa_temp;         % save the current eps
                        kappa_max = kappa_temp;     % update the upper bound for searching
                    else
                        kappa_min = kappa_temp;     % update the lower bound for searching
                    end
                end

                if existence == 0 || counter == max_iter
                    break;
                else
                    % update the eps_temp to be checked
                    kappa_temp = (kappa_max-kappa_min)/2+kappa_min;
                    
                    % counter plus 1 
                    counter = counter +1;
                end
            end
            
            
            if existence == 0
                kappa_range = [1 1];
            else
                kappa_range = [kappa 1];
            end
            
        end
              
        %% compute M and K given kappa and epsilon
        function sol = solve_MK(obj,kappa,epsilon)
            % jointly compute M and K given kappa and epsilon
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
            
            
            C_var = obj.C;
            A_var = obj.A;
            B_var = obj.B;
            
            % to see whether the system is nonlinear
            if ~isempty(obj.E)
                E_var = obj.E{1};
                F_var = obj.F{1};
                b_var = obj.b{1};
                nonlin = 1;             % indicate that this is a nonlinear system
            else
                nonlin = 0;
            end
                        
            % Computing constraints
            con_A = zeros(size(obj.input_con.A,1),size(obj.input_con.A,2));
            for i = 1:1:size(obj.input_con.A,2)
                con_A(i,:) = obj.input_con.A(i,:)./obj.input_con.b(i,:);
            end
            
            % define variable            
            M_var = sdpvar(size(A_var,2),size(A_var,2));
            K_var = sdpvar(size(B_var,2),size(A_var,1));
            if nonlin == 1
                L_var = sdpvar(size(B_var,2),size(A_var,1));
            end
             
            % define objective function
            fun = -logdet(M_var);
            
            % define constraints
            constraints = [M_var M_var*C_var';C_var*M_var eye(size(C_var,1))]>=0;
            constraints = [constraints,M_var>0];
            
            % stability constraint
            if nonlin == 0
                constraints = [constraints,[M_var A_var*M_var+B_var*K_var;M_var'*A_var'+K_var'*B_var' kappa*M_var]>=0];
            elseif nonlin == 1
                constraints = [constraints,[M_var A_var*M_var+B_var*K_var;M_var'*A_var'+K_var'*B_var' kappa*M_var]>=0];
                constraints = [constraints,[M_var A_var*M_var+B_var*K_var+b_var(1)*(B_var*L_var+E_var*F_var*M_var);...
                    M_var'*A_var'+K_var'*B_var'+b_var(1)*(L_var'*B_var'+M_var'*F_var'*E_var') kappa*M_var]>=0];
                constraints = [constraints,[M_var A_var*M_var+B_var*K_var+b_var(2)*(B_var*L_var+E_var*F_var*M_var);...
                    M_var'*A_var'+K_var'*B_var'+b_var(2)*(L_var'*B_var'+M_var'*F_var'*E_var') kappa*M_var]>=0];
            end
            
            % input constraints
            for j = 1:1:size(obj.input_con.A,1)
                if nonlin == 0
                    constraints = [constraints,[1/epsilon^2 con_A(j,:)*K_var; K_var'*con_A(j,:)' M_var]>=0];
                elseif nonlin == 1
                    constraints = [constraints,[1/epsilon^2 con_A(j,:)*K_var; K_var'*con_A(j,:)' M_var]>=0];
                    constraints = [constraints,[1/epsilon^2 con_A(j,:)*(K_var+b_var(1)*L_var); (K_var'+b_var(1)*L_var')*con_A(j,:)' M_var]>=0];
                    constraints = [constraints,[1/epsilon^2 con_A(j,:)*(K_var+b_var(2)*L_var); (K_var'+b_var(2)*L_var')*con_A(j,:)' M_var]>=0];
                end
            end

            % solve optimization problem using the optimization solver sedumi
            ops = sdpsettings('solver', obj.solver,'verbose',0);
            rec = optimize(constraints, fun, ops);
            
            % get optimal value and optimal solution
            M_var = inv(value(M_var));
            K_var = value(K_var)*M_var;
            if nonlin == 1
               L_var = value(L_var)*M_var;
            end
            
            % double check whether the constraints are really respected by
            % the constraints (Trust, but verify!)
            ind = 1;
            ind  = ind & min(eig(M_var))>0;
            ind = ind & min(eig(M_var-C_var'*C_var))>=0;
            
            % check stability constraints
            if nonlin == 0
                ind = ind & min(eig(kappa*M_var-(A_var+B_var*K_var)'*M_var*(A_var+B_var*K_var)))>=0;
            elseif nonlin == 1
                ind = ind & min(eig(kappa*M_var-(A_var+B_var*K_var)'*M_var*(A_var+B_var*K_var)))>=0;
                ind = ind & min(eig(kappa*M_var-(A_var+B_var*K_var+b_var(1)*(B_var*L_var + E_var*F_var))'*M_var*(A_var+B_var*K_var+b_var(1)*(B_var*L_var + E_var*F_var))))>=0;
                ind = ind & min(eig(kappa*M_var-(A_var+B_var*K_var+b_var(2)*(B_var*L_var + E_var*F_var))'*M_var*(A_var+B_var*K_var+b_var(2)*(B_var*L_var + E_var*F_var))))>=0;
            end
            
            % check input constraints
            for i = 1:1:size(obj.input_con.A,1)
                if nonlin == 0
                    for j = 1:1:size(obj.input_con.A,1)
                        ind = ind & min(eig([1/epsilon^2 con_A(j,:)*K_var; K_var'*con_A(j,:)' M_var]))>=0;
                    end
                elseif nonlin == 1
                    for j = 1:1:size(obj.input_con.A,1)
                        ind = ind & min(eig([1/epsilon^2 con_A(j,:)*K_var; K_var'*con_A(j,:)' M_var]))>=0;
                        ind = ind & min(eig([1/epsilon^2 con_A(j,:)*(K_var+b_var(1)*L_var); (K_var'+b_var(1)*L_var')*con_A(j,:)' M_var]))>=0;
                        ind = ind & min(eig([1/epsilon^2 con_A(j,:)*(K_var+b_var(2)*L_var); (K_var'+b_var(2)*L_var')*con_A(j,:)' M_var]))>=0;
                    end
                end 
            end
            
            sol = aprsol_options();
            % save the result
            if  ind ==1
                sol.epsilon = epsilon;
                sol.M = M_var;
                sol.K = K_var;
                if nonlin == 1
                    kappa1 = norm(sqrtm(M_var)*(A_var+B_var*K_var+b_var(1)*(B_var*L_var + E_var*F_var))/sqrtm(M_var))^2;
                    kappa2 = norm(sqrtm(M_var)*(A_var+B_var*K_var+b_var(2)*(B_var*L_var + E_var*F_var))/sqrtm(M_var))^2;
                    kappa3 = norm(sqrtm(M_var)*(A_var+B_var*K_var)/sqrtm(M_var))^2;
                    sol.kappa = max([kappa1 kappa2 kappa3]);
                    sol.L = L_var;
                elseif nonlin == 0
                    sol.kappa = norm(sqrtm(M_var)*(A_var+B_var*K_var)/sqrtm(M_var))^2;
                end
                sol.info = rec;
            else
                sol.epsilon = NaN;
                sol.kappa = NaN;
                sol.M = NaN;
                sol.K = NaN;
                if nonlin == 1
                    sol.L = NaN;
                end
                sol.info = rec;
            end
        end
               
        %% check the feasibility of the given MK solution
        function sol = check_fea_MK(obj,sol)
            % check the feasibility of the given sol
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
            
            
            N = sqrtm(sol.M);
            
            % ======================= gamma_0 ======================= 
            if obj.isgame == 1
                    % when the target is a game, we need to consider the
                    % effect of the internal input (Player 2)
                    sol.gamma0 = norm(N*obj.D/sqrtm(obj.M_w))*obj.epsilon_w;
            else 
                sol.gamma0 = 0;
            end
            
            % ======================= gamma_1 ======================= 
            BMB = (obj.B*obj.R_u-obj.P*obj.Br)'*sol.M*(obj.B*obj.R_u-obj.P*obj.Br);
            num_u = size(obj.hu,2);
            re_gamma1 = zeros(num_u,1);
            for i1 = 1:1:num_u
                re_gamma1(i1,1) = sqrt(obj.hu(:,i1)'*BMB*obj.hu(:,i1));
            end
            sol.gamma1 = max(re_gamma1);

            % ======================= gamma_2 ======================= 
            % discount by noise space when there is infinite abstraction
            % the internal calculation is only called when
            % sigma_discount has not been calculated
            if ~isequal(obj.P,eye(size(obj.A,2),size(obj.A,2)))
                % if P is not a identity matrix with the same dimension
                % of A, then infinite abstraction has been synthesized
                sol.Rr = inf_Rr(obj,sol);
                dim = size(obj.R,2);
                boundv = chi2inv(1-obj.delta,dim);
                sol.gamma2 = norm(N*(obj.R-obj.P*sol.Rr))*boundv;
            else
                sol.Rr = obj.R;
                sol.gamma2 = 0;
            end

            % ======================= gamma_3 ======================= 
            % discount resulted by quantization of state space
            % generate vertex according to beta
            dim_b = length(obj.beta);
            vertex = zeros(2^dim_b,dim_b);
            for j = 1:1:2^dim_b
                cd = dec2bin(j-1,dim_b);
                for jj = 1:1:dim_b
                    vertex(j,jj) = (-1)^(str2double(cd(jj)));
                end
                vertex(j,:) = vertex(j,:).*obj.beta';
            end
            vertex = vertex';
            % compute gamma3
            PMP = obj.P'*sol.M*obj.P;
            num_v = 2^dim_b;
            re_gamma3 = zeros(num_v,1);
            for i3 = 1:1:num_v
                re_gamma3(i3,1) = sqrt(vertex(:,i3)'*PMP*vertex(:,i3));
            end
            
            gamma3_1 = max(re_gamma3);
            gamma3_2 = gamma3_cal(obj,sol);
            
            sol.gamma3 = max([gamma3_1 gamma3_2]);
            
            % ======================= gamma_4 ============================
            % discount of w
            if obj.isgame == 1
                SBMBS = obj.S'*obj.B'*sol.M*obj.B*obj.S;
                num_w = size(obj.hw,2);
                re_gamma4 = zeros(num_w,1);
                for i4 = 1:1:num_w
                    re_gamma4(i4,1) = sqrt(obj.hw(:,i4)'*SBMBS*obj.hw(:,i4));
                end
                sol.gamma4 = max(re_gamma4);
            else
                sol.gamma4 = 0;
            end
            
            % ======================= sum up ======================= 
            % sum up gamma0 to gamma4
            sol.gamma_total = sol.gamma0 + sol.gamma1 + sol.gamma2 + sol.gamma3 + sol.gamma4;
            
           % decide whether current sol is feasible
           
           if sol.gamma_total<=sol.epsilon*(1-sqrt(sol.kappa))
               sol.fea = 1;
               sol.epsilon = sol.gamma_total/(1-sqrt(sol.kappa));
           else
               sol.fea = 0;
               sol.gamma_max = sol.epsilon*(1-sqrt(sol.kappa));
           end
            
        end
 
        %% search for the best solution in the current sol list
        function [obj,eps_position] = search_bestsol(obj)
            % search the best solution in each epsilon,kappa_range pair
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
            
            
            tic
            cur_smallest_eps = 10^10;
            eps_position = 1;
            num_fea = 0;
            num_notfea = 0;
            
            for i = 1:1:length(obj.sol_list)
                % get the current list of solution
                cur_sol_list = obj.sol_list(i).sol_group;
                num_sol = length(cur_sol_list);
                best_sol_temp = obj.sol_list(i).best_sol;
                cur_eps = best_sol_temp.epsilon;
                if best_sol_temp.fea ==1
                    % the initial best_sol is already a feasible answer
                    cur_gdiff = 0;
                    mode = 2;
                else 
                    % otherwise, record the necessary effort
                    cur_gdiff = best_sol_temp.gamma_total- best_sol_temp.gamma_max;
                    mode = 1;
                end
                

                for j = 2:1:num_sol
                    % go through all other solutions
                    obs_sol = cur_sol_list(j);
                    if mode == 1
                        % feasible solution has not yet been founded
                        if obs_sol.fea ==1
                            % this is a feasible solution
                            cur_eps = obs_sol.epsilon;
                            best_sol_temp = obs_sol;
                            mode = 2;
                        else
                            % this is still not a feasible solution
                            if obs_sol.gamma_total -obs_sol.gamma_max  < cur_gdiff
                                % easier to adapt
                                cur_gdiff = obs_sol.gamma_total -obs_sol.gamma_max ;
                                best_sol_temp = obs_sol;
                            end
                        end  
                    elseif mode == 2
                        % there are already feasible solutions
                        if obs_sol.fea ==1
                            % the current solution is a feasible solution
                            if obs_sol.epsilon<cur_eps
                                % if a smaller epsilon can be achieved by
                                % obs_sol
                                cur_eps = obs_sol.epsilon;
                                best_sol_temp = obs_sol;
                            end
                        end  
                    end
                end
                
                % save the result
                obj.sol_list(i).best_sol = best_sol_temp;
                if mode == 1
                    % no feasible solution has been found
                    obj.sol_list(i).fea = 0;
                    num_notfea = num_notfea +1;
                elseif mode == 2
                    % feasible solution has been found in this group
                    obj.sol_list(i).fea = 1;
                    num_fea = num_fea +1;
                    if best_sol_temp.epsilon < cur_smallest_eps
                        cur_smallest_eps = best_sol_temp.epsilon;
                        eps_position = i;
                    end
                end
            end 
            time_con = toc;
            
            % print the report
            disp([num2str(num_notfea),' unfeasible solutions and ',num2str(num_fea),' feasible solutions are founded within ',num2str(time_con),' seconds.']);
            if num_fea>0
                disp(['The ',num2str(eps_position),'-th solution is the best solution with epsilon = ',num2str(cur_smallest_eps),'!']);
            end
            
        end
        
        %% Discount attributed to beta, i.e. quantization error (\gamma_3 in the paper) 
        function gamma3 = gamma3_cal(obj,sol)
            % gamma3_cal: calculate discount attributed to quantization in state space
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
            
            
            PTMP = obj.P'*sol.M*obj.P;
            
            % optimisation variables
            beta_var = sdpvar(size(obj.Ar,2),1);
            
            % target function
            fun = -beta_var'*PTMP*beta_var;
            
            % constraints
            con = [-eye(size(obj.Ar,2),size(obj.Ar,2));eye(size(obj.Ar,2),size(obj.Ar,2))]*beta_var<=[obj.beta;obj.beta];

            % approximate the area for optimization so that the problem is
            % convex
            % con = beta_var'*beta_var<= obj.beta'*obj.beta;
            
            % set random initial value to gain global maximal
            for i = 1:1:size(obj.Ar,2)
                int_guess(i,1) = -obj.beta(i) + 2*obj.beta(i)*rand;
            end
            assign(beta_var,int_guess);
            ops = sdpsettings('verbose',0,'usex0',1,'solver','fmincon','fmincon.TolCon',1.0e-30,'fmincon.TolFun',1.0e-30);
            
            % solve the optimization problem
            res = optimize(con,fun,ops);
            
            if res.problem ~= 0
                disp('Error occur when computing gamma3!')
                res
            end
            
            % calculate the discount attributed to quantization of state
            % space
            gamma3 = sqrt(value(-fun));
        end
                
        %% copy the selected solution 
        function obj = copy_sol(obj,sol)
            % copy the solution in sol to the lifting object
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
            
            
            obj.epsilon = sol.epsilon;                          
            obj.gamma0 = sol.gamma0;                 
            obj.gamma1 = sol.gamma1;                 
            obj.gamma2 = sol.gamma2;                 
            obj.gamma3 = sol.gamma3;                 
            obj.gamma4 = sol.gamma4;                 
            obj.gamma_total = sol.gamma_total;            
            obj.M = sol.M;                      
            obj.K = sol.K;       
            if ~isempty(sol.L)
                obj.L = sol.L;
            end                    
            obj.Rr = sol.Rr;    
            obj.result = sol.info;
            
            % double check the solution copied in the lifting object
            feas = obj.Sprocedure_lambda(sol.kappa);
            if feas == 0
                disp('The provided answer is not feasible! Please check your input carefully');
            end
        end
        
       %% Checking the existency of lambda given epsilon
        function feas = Sprocedure_lambda(obj,kappa)
            % Sprocedure_lambda: checking the existence of lamda for s-procedure given epsilon
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
            
            
            if isempty(obj.b)
                nonlin = 0;
            else
                nonlin = 1;
                b_var = obj.b{1};
                E_var = obj.E{1};
                F_var = obj.F{1};
            end
            
            % dimension of the state variable
            dim_x = length(obj.A);
            
            % defining lamda
            lamda_temp = sdpvar(1,1);
            
            % matrix and vectors required for S-procedure
            F1 = obj.M;
            
            % set F2 accordingly if there are nonlinear terms
            if nonlin == 0
                F2 = (obj.A+obj.B*obj.K)'*obj.M*(obj.A+obj.B*obj.K);
            else
                for i=1:1:2
                    F2 = (obj.A+obj.B*obj.K)'*obj.M*(obj.A+obj.B*obj.K);
                    F3 = (obj.A+obj.B*obj.K + b_var(1)*(obj.B*obj.L+E_var*F_var))'*obj.M*(obj.A+obj.B*obj.K+ b_var(1)*(obj.B*obj.L+E_var*F_var));
                    F4 = (obj.A+obj.B*obj.K + b_var(2)*(obj.B*obj.L+E_var*F_var))'*obj.M*(obj.A+obj.B*obj.K+ b_var(2)*(obj.B*obj.L+E_var*F_var));
                end
            end
            
            g = zeros(dim_x,1);
            
            M1 = [F1 zeros(dim_x,1);zeros(1,dim_x) -obj.epsilon^2];
            
            M2 = [F2 g;g' -(obj.epsilon-obj.gamma_total)^2];
            
            if nonlin == 1
                M3 = [F3 g;g' -(obj.epsilon-obj.gamma_total)^2];
                M4 = [F4 g;g' -(obj.epsilon-obj.gamma_total)^2];
            end
            
            
            % defining constraints for s-procedure
            con1 = lamda_temp*M1-M2>=0;
            
            % positve lamda is required
            con2 = lamda_temp>=0;
            con3 = lamda_temp<1;
            con = (con1+con2+con3);
            
            if nonlin == 1
                con3 = lamda_temp*M1-M3>=0;
                con4 = lamda_temp*M1-M4>=0;
                con = (con1+con2+con3+con4);
            end
            
            % keep silent during the optimization
            ops = sdpsettings('verbose', 0,'solver','SDPT3');
            
            % checking the existence of lamda
            obj.result = optimize(con,'',ops);
            
            % save lamda
            lamda = value(lamda_temp);
            
            % check and verify
            ind = 1;
            ind = ind & obj.result.problem == 0;
            ind = ind & min(eig(value(lamda_temp*M1-M2)))>=-1.0e-06;
            if nonlin == 1
                ind = ind & min(eig(value(lamda_temp*M1-M3)))>=-1.0e-06;
                ind = ind & min(eig(value(lamda_temp*M1-M4)))>=-1.0e-06;
            end
            
            if ind == 1
                feas = 1;
                disp(['Positive lamda exists for solving the S-procedure. lambda = ',num2str(lamda),'.']);
            else
                % try the lambda associate with the solution
                ind2 = 1;
                ind2 = ind2 & min(eig(value(kappa*M1-M2)))>=-1.0e-07;
                if nonlin == 1
                    ind2 = ind2 & min(eig(value(kappa*M1-M3)))>=-1.0e-07;
                    ind2 = ind2 & min(eig(value(kappa*M1-M4)))>=-1.0e-07;
                end
                
                if ind2 ==1
                    feas = 1;
                    disp(['Positive lamda exists for solving the S-procedure. lambda = ',num2str(kappa),'.']);
                else
                     feas = 0;
                     disp('Lamda does not exist! Something wrong happen!');
                end
            end
        end
        
        %% interface function text
        function obj = intf_gen(obj,nonlinear)
            % intf_gen: add nonliear term for the interface function
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
            
            
            obj.intf_text = [obj.intf_text,'+',nonlinear];
        end
  
        %% find counter example
        function c_exp = linear_counter_search(obj)
            % linear_counter_search: search counter example for the current
            % setting of lifting
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
            
            
            % configure the variable
            x_bar = sdpvar(size(obj.A,2),1);
            u_hat = sdpvar(size(obj.Br,2),1);
            sigma = sdpvar(size(obj.R,2),1);
            beta_var = sdpvar(size(obj.P,2),1);
            
            % temporally set bound of u
            u_max = max(abs(max(obj.hu)),abs(min(obj.hu)));
            
            boundv = chi2inv(1-obj.delta,size(obj.R,2));
            
            % define target function
            x_nxt = (obj.A+obj.B*obj.K)*x_bar+(obj.B*obj.R_u-obj.P*obj.Br)*u_hat+(obj.R-obj.P*obj.Rr)*sigma;
            func = -x_nxt'*obj.M*x_nxt;
            
            % constraint for searching counter example
            con1 = u_hat^2<=u_max^2;
            con2 = x_bar'*obj.M*x_bar<=obj.epsilon^2;
            con3 = sigma'*sigma<=boundv^2;
            con4 = [-eye(size(obj.Ar,2),size(obj.Ar,2));eye(size(obj.Ar,2),size(obj.Ar,2))]*beta_var<=[obj.beta;obj.beta];
            con = (con1+con2+con3+con4);
            
            % set initial state for the search
            int_guess_x = -0.1*obj.epsilon+0.2*obj.epsilon.*rand(size(obj.A,2),1);
            int_guess_u = -0.1*obj.epsilon+0.2*obj.epsilon.*rand(size(obj.Br,2),1);
            int_guess_s = -0.01+0.02*rand(size(obj.R,2),1);
            int_guess_beta = -0.1+0.2*rand(size(obj.P,2),1);
            
            assign(x_bar,int_guess_x);
            assign(u_hat,int_guess_u);
            assign(sigma,int_guess_s);
            assign(beta_var,int_guess_beta);
            
            ops = sdpsettings('verbose',0,'usex0',1);
            
            % checking the existence of lamda
            re = optimize(con,func,ops);
            c_exp.epsilon_max = sqrt(-value(func));
            
            % report the result
            if c_exp.epsilon_max>obj.epsilon
                if re.problem == 0
                    disp('Counter example is found while the optimization procedure is terminated.');
                else
                    disp('Counter example is found but the optimization procedure have not been terminated.');
                end
                c_exp.result = re;
                c_exp.x_bar = value(x_bar);
                c_exp.u_hat = value(u_hat);
                c_exp.sigma = value(sigma);
                c_exp.beta = value(beta_var);
%                 disp(['x_bar = ',num2str(c_exp.x_bar),', ','u_hat = ',num2str(c_exp.u_hat),', ','sigma = ',num2str(c_exp.sigma),', ','beta = ',num2str(c_exp.beta),'.']);
                disp(['epsilon_nxt = ',num2str(c_exp.epsilon_max),'.']);
            elseif re.problem == 0 && c_exp.epsilon_max<=obj.epsilon
                disp('No counter example can be found.');
                c_exp.result = re;
            else
                disp('No counter example can be found, but the optimization did not terminate.');
            end 
        end
    end
end
