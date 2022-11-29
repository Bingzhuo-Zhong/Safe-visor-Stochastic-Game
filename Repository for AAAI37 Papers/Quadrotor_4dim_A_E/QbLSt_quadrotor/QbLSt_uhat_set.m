function list = QbLSt_uhat_set(x,x_ab,u,hu)
    % QbLSt_uhat_set:checking the emptiness of the set U_f as in (3.2)
    % Input: 
    %   x: current state of the original system
    %   x_ab: current state of the abstraction system
    %   u:  current input of the original system
    %   hu: list of input for the abstraction system
    % Outout:
    %   list: list of calculation result
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
    g = 9.8;

    P = eye(2);
    M = [1.46324748223085,0.175651348922892;0.175651348922892,0.0666024088029546];
    A = [1,delta_t;0,1];
    Ar = A;
    B = [delta_t^2/2;delta_t];
    Br= B;
    epsilon = 0.067422707510166; 
    gamma2 = 0.001481769739204+0.013715511615976; 
    
    delta = A*x - P*Ar*x_ab +B*u;
    
    a1 = Br'*P'*M*P*Br;
    a2 = -delta'*M*P*Br;
    a3 = -Br'*P'*M*delta;
    a4 = (epsilon-gamma2)^2-delta'*M*delta;
    
    % initializing list
    num_hu = length(hu);
    list_temp=zeros(1,num_hu);
    
    for i = 1:1:num_hu
        list_temp(1,i) = hu(:,i)'*a1*hu(:,i) + a2*hu(:,i) + hu(:,i)'*a3-a4;
    end
    
    % export result (the index of those input which fulfils the requirement for approximate probabilistic relation)
    list = find(list_temp<=0);

end