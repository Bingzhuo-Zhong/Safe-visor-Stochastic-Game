classdef Lmap_options
    % Lmap: class for defining L mapping for DFA
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
        A;          % matrix of double: matrix A for affine constraint
        b;          % vector of double: vector b for affine constraint
    end
    
    methods
        function obj = Lmap_options()
            % Lmap_options: Construct an instance of this class
        end
    end
end

