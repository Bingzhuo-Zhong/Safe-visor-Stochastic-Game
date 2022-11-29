%% examinate x_uc
function result = xuc_analysis_quadrotor4dim(obj,DFA)
    % xuc_analysis_quadrotor4dim: analysing the data for the 4 dimensional
    % quadrotor
    % input:
    %       DFA: DFA for analysing the x_uc_execution
    % output:
    %       result: row vector: [num_F num_notF num_F/num_trace]
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

    num_trace = length(obj.x_uc_execute);
    num_notF = 0;
    num_F = 0;

    for i = 1:1:num_trace
        % go through all traces

        % initialization
        cur_q = 1;
        check_F = 0;
        for j = 1:1:size(obj.x_uc_execute{i},2)
            % get the current state
            x_cur = obj.x_uc_execute{i}(:,j);

            % compute the current output based on the current state
            y = [x_cur(1);x_cur(3)];

            % check the current state based on the current output
            cur_q = DFA.q_nxt(cur_q,y);

            if DFA.isacc_state(cur_q)
                % if the accepting state is reached
                check_F = 1;
                break;
            end
        end

        % record the result for the current trace
        if check_F == 1
            % the accepting stae is reachted
            num_F =num_F +1;
        else
            num_notF = num_notF +1;
        end
    end
    result = [num_F num_notF num_F/num_trace];

    % display the result
    disp([num2str(num_F),' traces reach the accepting state(s), which is ',num2str(num_F/num_trace*100),'% of the total traces.'])
end
