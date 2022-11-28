function  x_nxt = system_dyn(x_cur, u_cur,w_cur,cur_disturbance)
% System dynamic for Quadrotor
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

    Ae = [1,delta_t;0,1];
    Be = [delta_t^2/2;delta_t];
    De = [delta_t^2/2;delta_t];
    Re = [0.004 0;0 0.045];

    A = [Ae zeros(2,2);zeros(2,2) Ae];
    B = [Be zeros(2,1);zeros(2,1) Be];
    D = [De zeros(2,1);zeros(2,1) De];
    R = [Re zeros(2,2);zeros(2,2) Re];

    x_nxt = A*x_cur + +B*u_cur + D*w_cur + R*cur_disturbance;

end