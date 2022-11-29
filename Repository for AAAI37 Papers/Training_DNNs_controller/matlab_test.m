%pyversion('/home/hongi/anaconda3/envs/rips/bin/python')
% module_path = '/home/roboticlabquad/bingzhuo_Drone_Proj/ICUAS2022/Safe-visor Architecture/Quadrotor_4dim_sim/bridge.py';
% bridge = py.importlib.import_module(module_path);
% if count(py.sys.path, '' ) == 0
%     insert(py.sys.path,int32(0), '' );
% end           
for i = 1:1:20
    % warming up
    input = rand(1,8);
%     py.bridge.send_message(input(1), input(2), input(3), input(4), input(5), input(6), input(7), input(8));
%     a = py.bridge.get_actions(input(5), input(6), input(7), input(8));
    a =py.bridge.send_states_get_actions(input(1), input(2), input(3), input(4), input(5), input(6), input(7), input(8));
    a=double(py.array.array('d', a));
end
disp('pretest_done')


num_tes = 10000;
time_record = zeros(num_tes,1);
for i = 1:1:num_tes
    tic
    % code here
    input = rand(1,8);
%     py.bridge.send_message(input(1), input(2), input(3), input(4), input(5), input(6), input(7), input(8));
%     a = py.bridge.get_actions(input(5), input(6), input(7), input(8));
    a =py.bridge.send_states_get_actions(input(1), input(2), input(3), input(4), input(5), input(6), input(7), input(8));
    a=double(py.array.array('d', a));
    
    time_record(i,1)=toc;
end

avr_time = mean(time_record)

histogram(time_record)