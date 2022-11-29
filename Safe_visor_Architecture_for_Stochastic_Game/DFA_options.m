classdef DFA_options
    % Class for DFA
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
        dfa;                % matrix of cell: matrix for the DFA
                            % Remarks:
                            %       dfa is a matrix that records how the states in the DFA transfer between each other according to the state in the physical system. 
                            %       In each entry, there are few lines of codes for judging whether the transition would take place. 
                            %       For instant, give a matrix dfa and in
                            %       entry (1,2) there is code ¡°if dfa_var.C1*x>=dfa_var.d1 && dfa_var.C2*x<=dfa_var.d2;ind=1;end¡±
                            %       Then if it is within the area specified by ¡°dfa_var.C1*x>=dfa_var.d1 && dfa_var.C2*x<=dfa_var.d2¡±, then we know that ind = 1.
        n_state;            % integer: number of state in DFA
        dfa_var;            % struct: variable list for matrix dfa
        spec_type;          % integer: safety LTL: 1 ; co-safe LTL: 2
        acc_state;          % vector of interger: accpeting state for the DFA, which is forbidden states for safety LTL and accepting state for co-safe LTL
        sink_state;         % interger: sink state 
        trap_state;         % interger: trap_state
    end
    
    methods
       %%
        function obj = DFA_options()
        % create a new object for DFA
        obj.sink_state = -1;
        end
        %%
        function nxtq = q_nxt(obj,cur_q,x)
            % Indentifying the next state
            % input:
            %   cur_q:  current state of the DFA
            %   x:      current state of the physical system
            % output:
            %   nxtq:   target state of the DFA given cur_a and x
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
            
            % obtain the variable list 
            dfa_var = obj.dfa_var;
            
            for i = 1:1:obj.n_state
                if ~isempty(obj.dfa{cur_q,i})
                    % Deciding whether there is a transition between the current
                    % state and the target state
                    ind =0;
                    
                    % execute the code in the dfa matrix
                    eval(obj.dfa{cur_q,i}); 
                    if ind == 1
                        % the current q is the target set, checking exits 
                        nxtq = i;
                        break;
                    end
                end
            end
        end
        
        %%
        function isacc = isacc_state(obj,cur_q)
            % Identifying whether the target set is reached
            % Input:
            %   cur_q: current state in the DFA
            % Output:
            %   isacc:identifier of the identification. isacc = 1 means
            %   that cur_q is the accepting state, isacc = 0 otherwise
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
            
            isacc =0;   % initialize the identifier
            for i=1:1:length(obj.acc_state)
                % go through the list of the accepting state
                if obj.acc_state(i)==cur_q
                    % if the cur_q is in the list of accepting state, set
                    % isacc to 1
                    isacc = 1;
                    break;
                end
            end
        end
        
       %% Identifying trap state
        function istrap = istrap_state(obj,cur_q)
            % Identifying whether the trap set is reached
            % Input:
            %   cur_q: current state in the DFA
            % Output:
            %   istrap:identifier of the identification. istrap = 1 means
            %   that cur_q is a trap state, isacc = 0 otherwise
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
            
            istrap =0;   % initialize the identifier
            for i=1:1:length(obj.trap_state)
                % go through the list of the accepting state
                if obj.trap_state(i)==cur_q
                    % if the cur_q is in the list of accepting state, set
                    % isacc to 1
                    istrap = 1;
                    break;
                end
            end
        end
        
    end
end
