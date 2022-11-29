function  [u_uc,para_uc]= unverified_controller(x_cur,cur_time,para_uc)
%  unverified controller
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

    % definition of parameters
    K_cur = [1.4781,1.7309,0,0;0,0,1.4781,1.73089];
    uxb = 2.5;
    uyb = 2.5;

    input=x_cur;
    a=py.bridge.send_states_get_actions(input(1), input(2), input(3), input(4), input(5), input(6), input(7), input(8));
    a=double(py.array.array('d', a));
    
    drone_set=[a(3);a(1)];
    drone_vset = [a(4);a(2)];

    state_cur = [x_cur(3)-drone_set(1);x_cur(4)-drone_vset(1);x_cur(1)-drone_set(2);x_cur(2)-drone_vset(2)];
    acc_temp = -K_cur*state_cur;

    % acceleration saturation for the unverified controller
    u_uc = zeros(2,1);
    u_uc(1,1) = min(max(acc_temp(1),-uxb),uxb);
    u_uc(2,1) = min(max(acc_temp(2),-uyb),uyb);

end