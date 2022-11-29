function u = QbLSt_interface(x,x_hat,u_hat)
    % QbLSt_interface: Specify the interface function for the controller
    % Input: 
    %   x: current state of the original system
    %   x_hat: current state of the abstraction system
    %   u_hat: current input on the abstraction system
    % Outout:
    %   u: current input on the original system (infinite abstraction)
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
    
    K = [-16.6608177882335,-4.83304133472051];
  
    u = K*(x-x_hat) + u_hat;
end