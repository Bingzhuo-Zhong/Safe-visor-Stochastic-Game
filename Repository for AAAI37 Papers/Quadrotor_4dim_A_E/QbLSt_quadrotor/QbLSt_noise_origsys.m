function distb = QbLSt_noise_origsys(x_pre,u_pre,w_pre,x_cur)
    % QbLSt_noise_origsys: noise calculator for the original system
    % Input: 
    %   x_pre: previous state of the original system
    %   u_pre: previous input for the original system
    %   w_pre: previous internal (player 2) input for the original system
    %   x_cur: current state of the original system
    % Outout:
    %   distb: double: noise affect the original system
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
    
    distb = x_cur -(A*x_pre + B*u_pre + D*w_pre);  
end

