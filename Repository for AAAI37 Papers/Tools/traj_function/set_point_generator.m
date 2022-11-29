function set_point = set_point_generator( traj_n, t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init, attitude_fin, altitude, omega, roll, yaw_following, scale, delta_pos_n, delta_pos_e, pitch_deg, speed, amp, ff, theta_f)

%IF yaw_following == 2, ONLY for circular trajectory, the yaw angle would follow the tangent direction of the circular trajectory
% and the roll angle is kept constant at the 'roll' input value

attitude_init_eul = quatToEuler(attitude_init');
attitude_fin_eul = quatToEuler(attitude_fin');

switch traj_n
    case 1      %Infinity shape trajectory
        sp = setp_inf_shape( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, altitude, omega, roll, yaw_following, scale );
    
    case 2      %Eight-shape trajectory
        sp = setp_8shape( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, altitude, omega, roll, yaw_following, scale );
        
    case 3      %Circular trajectory
        sp = setp_circular( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, altitude, omega, roll, yaw_following, scale );
        
    case 4      %Polynomial trajectory
        sp = setp_poly( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, altitude );
        
    case 5      %Position step
        sp = setp_pos_step( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, attitude_fin_eul, altitude );
        
    case 6      %Infinity trajectory, alternative
        sp = setp_inf_alt( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, altitude, omega, roll, yaw_following, scale , delta_pos_n, delta_pos_e);
        
    case 7      %Circular trajectory, alternative
        sp = setp_circular_alt( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, altitude, omega, roll, yaw_following, scale, delta_pos_n, delta_pos_e );

    case 8      %paol1
        sp = setp_paol1(t, altitude, pos_ned_init, pos_ned_fin, attitude_init_eul, simulation_time, pitch_deg);
        
    case 9      %spiral trajectory
        sp = setp_spiral(t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, altitude, omega, roll, yaw_following, scale );
        
    case 10     %follow a line given the speed
        sp = setp_line( t, simulation_time, pos_ned_init, attitude_init_eul, attitude_fin_eul, altitude, speed );
        
    case 11     %from point A to point B
        sp = setp_from_a_to_b(t, pos_ned_init, pos_ned_fin, attitude_init_eul, attitude_fin_eul, simulation_time);
    
    case 12     %vertical oscillating
        sp = setp_vert_oscill( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, attitude_fin_eul, altitude, amp, ff );
    
    case 13 
        sp = setp_circular_su(  t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, altitude, omega, roll, yaw_following, scale, theta_f);
    
    case 14      %Circular landing trajectory only for simulator!!!
        sp = setp_circular_landing( t, simulation_time, pos_ned_init, pos_ned_fin, attitude_init_eul, altitude, omega, roll, yaw_following, scale );
        
    otherwise	%Stay in the origin
        sp = [ pos_ned_init ;       %Position
               zeros(3, 1)  ;       %Velocity
               zeros(3, 1)  ;       %Acceleration
               zeros(3, 1)  ;       %Jerk
               attitude_init_eul ;  %Attitude
               zeros(3, 1)  ;       %Angular speed
               zeros(3, 1)  ;       %Angular acceleration
               zeros(3, 1) ];       %Angular jerk
end

quat_att = eulerToQuat( sp(13:15) );

set_point = [  sp(1:3)  ;   %Position
               sp(4:6)  ;   %Velocity
               sp(7:9)  ;   %Acceleration
               sp(10:12);   %Jerk
               quat_att;    %Attitude
               sp(16:18);   %Angular speed
               sp(19:21);   %Angular acceleration
               sp(22:24)];  %Angular jerk
end