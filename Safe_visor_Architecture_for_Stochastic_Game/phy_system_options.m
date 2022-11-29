classdef phy_system_options
    % Parameters and function for the physical systems
    %   Including the dynamics, noise, input set as well as state set of the
    %   systems
    %   2020.09.28: add properties about game
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
        %targetFile;            % Text: target file for system dynamic
        isgame;                 % integer: indicating whether the target system is a game
                                % isgame = 1: this is a DTSG
                                % isgame = 0; otherwise
        sys_type;               % integer: indicating the type of the system
                                % sys_typ = 1 : system with continuous state space and contiuous input space
                                % sys_typ = 2 : system with continuous state space and discrete input space
        x_l;                    % vector of double: Row vector for lower bound for state 
        x_u;                    % vector of double: Row vector for upper bound for state
        x_range;                % class of Lmap: State set of the original physical system (currently only support state range defined by one Lmap class)
        u_l;                    % vector of double: Row vector for lower bound for input, only for infinite input space, i.e., sys_typ = 1
        u_u;                    % vector of double: Row vector for upper bound for input, only for infinite input space, i.e., sys_typ = 1
        u_crange;               % cell of Lmap: Complement of input set defined by a set of Lmap
        w_l;                    % vector of double: Row vector for lower bound for internal(player 2) input, only for game, i.e., isgame = 1
        w_u;                    % vector of double: Row vector for upper bound for internal(player 2) input, only for game, i.e., isgame = 1
        hu;                     % vector of double: Row vector for input set, only for finite input space, i.e., sys_typ = 2
        Cov;                    % matrix of double: Diagonal of Noise Covariance Matrix for Bellmann interation
        initial_state_set;      % text: code for initial states
        phy_var;                % struct: variable list
    end
    
    methods
        function obj = phy_system_options()
            %obj.targetFile='targetfile';
            obj.isgame = 0;
            obj.sys_type=-1;
            obj.x_l=-inf;
            obj.x_u=inf;
            obj.u_l = -inf;
            obj.u_u = inf;
            obj.hu = 0;
            obj.Cov = 0;
        end
    end
end
