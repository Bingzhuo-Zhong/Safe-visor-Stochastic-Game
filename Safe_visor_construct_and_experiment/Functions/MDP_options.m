classdef MDP_options
    % Class for finite MDP
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
        % state space
        hx;                 % matrix of double: state set of the finite MDP (each column is a state, excluding the sink state)
        n_x;                % integer: number of state of the finite MDP
        delta_x;            % matrix of double: state set discretization parameter
        dim_x;              % integer: dimension of the state space    
        x_lower_bound;      % matrix of double: lower bound for grids in state space
        x_upper_bound;      % matrix of double: upper bound for grids in state space
        
        % input space
        hu;                 % matrix of double: input set of the finite MDP
        n_u;                % integer: number of input of the finite MDP
        delta_u;            % matrix of double: input space discretization parameter
        dim_u;              % integer: dimension of the input space
        
        % internal (player 2) input space
        hw;                 % matrix of double: internal input set of the finite MDP
        n_w;                % integer: number of internal input of the finite MDP
        delta_w;            % matrix of double: internal input space discretization parameter
        dim_w;              % integer: dimension of the internal input space
        
        % transition kernel
        sto_kernel;         % matrix of double: stochastic kernel of the finite MDP
        option;             % integer: indicating the way for getting data from the MDP
                            % option = 0: default: using the funtion
                            % implemented in the class function
                            % option = 1; self-defined way for getting data
        sdef_par;           % self define par for getting data in the salf define mode                 
    end
    
    methods
        function obj = MDP_options()
            obj.n_w = 1;    % the number is set to 1 by default, indicating that there is no internal (player 2) input (i.e. not a game)
            obj.option = 0;
            obj.sdef_par = [];
        end
        
        function re = MDP_get(obj,p_x,p_u,p_w)
            %   MDP_get: getting data from the stochastic kernel
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
            
            if obj.option ==0 || isempty(obj.option)
                % using default way to get data from the stochastic kernel
                if p_w ==0
                   % get all rows associated with the position of x and u
                   pos = p_x+(p_u-1)*obj.n_x+([1:1:obj.n_w]-1)*obj.n_x*obj.n_u;
                   re = obj.sto_kernel(pos,:);
                else
                    % get a specific row according to the position of the
                    % x, u, and w
                    re = obj.sto_kernel(p_x+(p_u-1)*obj.n_x+(p_w-1)*obj.n_x*obj.n_u,:);
                end
            else
                % using self defined way for getting data from the
                % stochastic kernel
                re = MDP_get_selfdef(obj,p_x,p_u,p_w);
            end
        end
    end
end