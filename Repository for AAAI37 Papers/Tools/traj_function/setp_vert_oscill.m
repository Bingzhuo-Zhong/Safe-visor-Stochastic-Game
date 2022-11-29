function set_point = setp_vert_oscill( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init, attitude_fin, altitude, amp, ff )
%The movement is composed by the following parts:
%1- Quadrotor starts and remains in the initial pos/att
%2- Use of two 5th order polynomials (1 for pos and 1 for att) to move the
%   drone from initial pos/att to [0;0;height] and null attitude
%3- Hovering in [0;0; height] and null attitude
%4- position step from the center of the cage to the desired point
%5- Oscillation in z position

%% Timing and conditions
t1 = 5;                     %Starting time with drone on the floor
t2 = t1 + 10;               %Time to take off and go to [0;0;height] and null att.
t3 = t2 + 5;                %Time for hovering in [0;0;height]
t4 = t3 + 5;
t5 = simulation_time - 10;
t6 = simulation_time - 5;

%% Position set-point

%Conditions at the beginning of simulation
pos_i = pos_ned_init;       %Initial position
vel_i = zeros(3,1);         %Initial speed
acc_i = zeros(3,1);         %Initial acceleration
att_i = attitude_init;      %Initial attitude
ang_rates_i = zeros(3,1);   %Initial angular speed
ang_acc_i = zeros(3,1);     %Initial angular acceleration

%Conditions of hovering at the center of the cage
pos_h = [pos_ned_init(1); pos_ned_init(2); altitude];    %Hovering position
vel_h = zeros(3,1);        %Hovering speed
acc_h = zeros(3,1);        %Hovering acceleration
att_h = attitude_init;     %Hovering attitude
ang_rates_h = zeros(3,1);  %Hovering angular speed
ang_acc_h = zeros(3,1);    %Hovering angular acceleration

%Hovering conditions to start vertical oscillation
pos_v = pos_ned_fin;       %Hovering position
vel_v = zeros(3,1);        %Hovering speed
acc_v = zeros(3,1);        %Hovering acceleration
att_v = attitude_fin;      %Hovering attitude
ang_rates_v = zeros(3,1);  %Hovering angular speed
ang_acc_v = zeros(3,1);    %Hovering angular acceleration



%% Trajectory generation
if t < t1
    pos_sp = pos_i;
    vel_sp = vel_i;
    acc_sp = acc_i;
    jerk_sp = zeros(3,1);
    attitude_sp = att_i;
    ang_rates_sp = ang_rates_i;
    ang_acc_sp = ang_acc_i;
    ang_jerk_sp = zeros(3,1);
    
elseif t >= t1 && t < t2
    [pos_sp, vel_sp, acc_sp, jerk_sp] = fifth_ord_poly(t, t1, t2, pos_i, pos_h, vel_i, vel_h, acc_i, acc_h);
    [attitude_sp, ang_rates_sp, ang_acc_sp, ang_jerk_sp] = fifth_ord_poly(t, t1, t2, att_i, att_h, ang_rates_i, ang_rates_h, ang_acc_i, ang_acc_h);
    
elseif t >= t2 && t < t3
    pos_sp = pos_h;
    vel_sp = vel_h;
    acc_sp = acc_h;
    jerk_sp = zeros(3,1);
    attitude_sp = att_h;
    ang_rates_sp = ang_rates_h;
    ang_acc_sp = ang_acc_h;
    ang_jerk_sp = zeros(3,1);
    
elseif  t >= t3 && t < t4
    
    [pos_sp, vel_sp, acc_sp, jerk_sp] = fifth_ord_poly(t, t3, t4, pos_h, pos_v, vel_h, [0;0;amp*ff], acc_h, acc_v);
    [attitude_sp, ang_rates_sp, ang_acc_sp, ang_jerk_sp] = fifth_ord_poly(t, t3, t4, att_h, att_v, ang_rates_h, ang_rates_v, ang_acc_h, ang_acc_v);
    
elseif t >= t4 && t < t5
    pos_sp  = [0;0; +amp*sin(ff*(t-t4))] + pos_v;
    vel_sp  = [0;0; +amp*ff*cos(ff*(t-t4))];
    acc_sp  = [0;0; -amp*(ff^2)*sin(ff*(t-t4))];
    jerk_sp = [0;0; -amp*(ff^3)*cos(ff*(t-t4))];
    attitude_sp = att_v;
    ang_rates_sp = ang_rates_v;
    ang_acc_sp = ang_acc_v;
    ang_jerk_sp = zeros(3,1);
    
elseif t >= t5 && t < t6
    pos_vf  = [0;0; +amp*sin(ff*(t5-t4))] + pos_v;
    vel_vf  = [0;0; +amp*ff*cos(ff*(t5-t4))];
    acc_vf  = [0;0; -amp*(ff^2)*sin(ff*(t5-t4))];
    jerk_vf = [0;0; -amp*(ff^3)*cos(ff*(t5-t4))];
    att_vf = att_v;
    ang_rates_vf = ang_rates_v;
    ang_acc_vf = ang_acc_v;
    ang_jerk_vf = zeros(3,1);
    
    [pos_sp, vel_sp, acc_sp, jerk_sp] = fifth_ord_poly(t, t5, t6, pos_vf, pos_v, vel_vf, vel_v, acc_vf, acc_v);
    [attitude_sp, ang_rates_sp, ang_acc_sp, ang_jerk_sp] = fifth_ord_poly(t, t5, t6, att_vf, att_v, ang_rates_vf, ang_rates_v, ang_acc_vf, ang_acc_v);
    
else
    pos_sp  = pos_v;
    vel_sp  = vel_v;
    acc_sp  = acc_v;
    jerk_sp = zeros(3,1);
    attitude_sp = att_v;
    ang_rates_sp = ang_rates_v;
    ang_acc_sp = ang_acc_v;
    ang_jerk_sp = zeros(3,1);
    
end

set_point = [pos_sp; vel_sp; acc_sp; jerk_sp; ...
    attitude_sp; ang_rates_sp; ang_acc_sp; ang_jerk_sp];
end