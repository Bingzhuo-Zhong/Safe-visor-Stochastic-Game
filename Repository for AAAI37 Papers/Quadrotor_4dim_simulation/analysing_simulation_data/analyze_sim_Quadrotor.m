% analyze_sim_Quadrotor: MATLABscript for analyzing the monte carlo simulation
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

% loading the data
load('eight_monte_data.mat')   % mat file for the simulation results

%% analyse simulation results
% analysize the trajectories of only using unverified DNNs-based
% controllers
Quadrotor_4d_DFA;
result_DFA = xuc_analysis_quadrotor4dim(monte_sim,DFA);

num_ex = monte_sim.loop;
time_horz = monte_sim.sva{1}.test_time;

time1_mat = zeros(num_ex,time_horz);
time2_mat = zeros(num_ex,time_horz);
flag_1_mat= zeros(num_ex,time_horz);
flag_2_mat= zeros(num_ex,time_horz);
for i = 1:1:num_ex
    time1_mat(i,:)=gen_exc_time{i}(1,:);
    time2_mat(i,:)=gen_exc_time{i}(2,:);
    flag_1_mat(i,:)=gen_flag_all{i}(1,:);
    flag_2_mat(i,:)=gen_flag_all{i}(2,:);
end
time1_vec = reshape(time1_mat,[num_ex*time_horz,1]);
time2_vec = reshape(time2_mat,[num_ex*time_horz,1]);
flag_1_vec = reshape(flag_1_mat,[num_ex*time_horz,1]);
flag_2_vec = reshape(flag_2_mat,[num_ex*time_horz,1]);

% compute the acceptance rate, avaeage execution time and std of the
% executiong time for sva_N
acc_N = sum(flag_1_vec==2)/length(flag_1_vec)
timeN_avg=mean(time1_vec)*1000
timeN_std=std(time1_vec)*1000

% compute the acceptance rate, avaeage execution time and std of the
% executiong time for sva_E
acc_E = sum(flag_2_vec==2)/length(flag_2_vec)
timeE_avg=mean(time2_vec)*1000
timeE_std=std(time2_vec)*1000


%% recovering the trajectory of the quadrotor from the model in (4.1) and output csv file for further analysis
drone_state_sva = monte_sim.x_execute;
drone_state_uc = monte_sim.x_uc_execute;

num_test = length(drone_state_sva);
time_horizon = length(drone_state_sva{1});
dim = 4;

drone_trace_sva = zeros(num_test*4,time_horizon);
drone_trace_uc = zeros(num_test*4,time_horizon);
car_traj = state_car_v(:,1:time_horizon);
for i = 1:1:num_test
    temp_sva= drone_state_sva{i}+car_traj;
    temp_uc= drone_state_uc{i}+car_traj;
    drone_trace_sva((i-1)*dim+1:i*dim,:) = temp_sva;
    drone_trace_uc((i-1)*dim+1:i*dim,:) = temp_uc;
    clear temp_sva temp_uc;
end

% save results
writematrix(drone_trace_sva,'drone_trace_sva_monte.csv')
writematrix(drone_trace_uc,'drone_trace_uc_monte.csv')


