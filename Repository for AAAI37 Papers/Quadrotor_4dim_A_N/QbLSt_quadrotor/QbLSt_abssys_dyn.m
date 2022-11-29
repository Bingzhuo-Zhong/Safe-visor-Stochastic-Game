function x_nxt = QbLSt_abssys_dyn(x,u,w,noise)
    % QbLSt_abssys_dyn: dynamic of the abstraction system 
    % Input: 
    %   x: previous state of the abstraction system
    %   u: previous input for the abstraction system
    %   w: previous internal (player 2) input for the abstraction system
    %   noise: noise executed on the original system
    % Outout:
    %   x_nxt: current state of the abstraction system (infinite abstraction)
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
%     R = 0.4*delta_t*eye(2);
    
    x_nxt = A*x + B*u + D*w + noise;
end

