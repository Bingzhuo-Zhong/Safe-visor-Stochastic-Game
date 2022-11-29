function [pos_sp,eul_att_sp] = eight_vertical_traj_gen(position_cur,euler_cur,sp_rate,execution_time,altitude,omega,scale,amp,ff,plotf)
% eight_vertical_traj_gen: generating the trajectories 

    %Options
    traj_n = 1;                                                                %Trajectory number

    % Setpoint Settings
    degToRad = pi/180;
    radToDeg = 180/pi;

    % Manual Settings
    roll_oscillation = 0;                                                      %Amplitude of roll sinusoidal oscillation
    yaw_following    = 0;                                                      %Yaw following flag                                                    %Scaling factor for traj
    delta_pos_n      = 0;                                                      %Delta for shit traj (6,7)
    delta_pos_e      = 0;                                                      %Delta for shit traj (6,7)
    pitch_deg        = 0;                                                      %Pitch for Paoll traj
    speed            = 0;                                                      %Speed parameter for traj 10

    %Setpoint rate [Hz]
    deltaT= 1/sp_rate;

    %Final condition Manual
    pos_ned_fin = [0, 0, -1.5]';
    attitude_fin = eulerToQuat( [ 0, 0, 0 ]' );

    %% Check trajectory

    pos_ned_init  = position_cur';
    attitude_init = eulerToQuat(euler_cur');

    s_p = zeros(25, execution_time*sp_rate );

    % newly added
    s_ph = zeros(24, execution_time*sp_rate );
    attitude_init_eul = quatToEuler(attitude_init');
    attitude_fin_eul = quatToEuler(attitude_fin');

    k = 1;

    for t=0:deltaT:execution_time
        s_ph(:,k) = setp_vert_oscill( t, execution_time, pos_ned_init, pos_ned_fin, attitude_init_eul, attitude_fin_eul, altitude, amp, ff );
        s_p(:,k) = set_point_generator( traj_n, t, execution_time, pos_ned_init, pos_ned_fin, attitude_init, attitude_fin, altitude, omega, roll_oscillation, yaw_following, scale, delta_pos_n, delta_pos_e, pitch_deg, speed);
        k = k+1;
    end

    tempo = 0:deltaT:execution_time;

    pos_sp = [s_p(1:2, :);s_ph(3, :)]';

    eul_att_sp = zeros(length(s_p),3);
    for i=1:length(s_p)
        eul_att_sp(i,:) = quatToEuler( s_p(13:16,i)' );
    end

    if plotf ==1
        figure;
        subplot(2,2,1)
        %plot(tempo, s_p(1:3, :));
        plot(tempo, pos_sp);
        xlabel('t [s]');ylabel('p [m]');
        legend('N','E','D');
        title('Position sp');
        grid minor

        subplot(2,2,2)
        plot(tempo, s_p(4:6, :));
        xlabel('t [s]');ylabel('v [m/s]');
        legend('N_{dot}','E_{dot}','D_{dot}');
        title('Velocity sp');
        grid minor

        subplot(2,2,3)
        plot(tempo, eul_att_sp*radToDeg);
        xlabel('t [s]');ylabel('att [deg]');
        legend('R','P','Y');
        title('Attitude sp');
        grid minor

        subplot(2,2,4)
        plot(tempo, s_p(17:19,:)*radToDeg);
        xlabel('t [s]');ylabel('as [deg/s]');
        legend('p','q','r');
        title('Angular Speed sp');
        grid minor
    end


end

