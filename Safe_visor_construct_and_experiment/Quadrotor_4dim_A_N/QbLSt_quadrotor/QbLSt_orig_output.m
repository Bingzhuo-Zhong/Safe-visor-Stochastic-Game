function y = QbLSt_orig_output(x)
    % QbLSt_orig_output: output function for the original system 
    % Input: 
    %   x: current state of the original system
    % Outout:
    %   y: current output of the original system
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
    C = [1 0];
    y = C*x;
end

