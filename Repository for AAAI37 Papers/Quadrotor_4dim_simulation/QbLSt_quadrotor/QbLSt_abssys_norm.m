function x_nxt = QbLSt_abssys_norm(x,u,w)
    % QbLSt_abssys_norm: norminal(i.e. without noise term) dynamic of the abstraction system 
    %                    for computing the MDP
    % Input: 
    %   x: current state of the system
    %   u: current input of the system
    %   w: current internal (player 2) input of the system
    % Outout:
    %   x_nxt: next state of the system
    % Remark:
    %   current system: Quadrotor    
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
    
    delta_t = 0.1;
    A = [1,delta_t;0,1];
    B = [delta_t^2/2;delta_t];
    D = [delta_t^2/2;delta_t];
    
    x_nxt = A*x + B*u +D*w;   
end