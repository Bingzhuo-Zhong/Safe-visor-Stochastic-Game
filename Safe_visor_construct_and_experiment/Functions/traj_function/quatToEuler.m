function euler = quatToEuler( quat )
%	Converts a quaternion vector to ZYX Euler angles (RPY)
%
%   euler = quatToEuler( quat )
%
%   The quaternion (quat) must be written as a 4-by-1 matrix with the scalar
%   element as the fourth component.
%   The returned Euler angles vector will be written in radiants as a 3-by-1 vector.
%
%   References:
%	[1] Markley, F. Landis. "Attitude error representations for Kalman filtering." 
%       Journal of guidance control and dynamics 26.2 (2003): 311-317.

norm = sqrt(quat(1)^2 + quat(2)^2 + quat(3)^2 + quat(4)^2);
quat(1) = quat(1) / norm;
quat(2) = quat(2) / norm;
quat(3) = quat(3) / norm;
quat(4) = quat(4) / norm;

A11 = quat(1)^2 - quat(2)^2 - quat(3)^2 + quat(4)^2;
A12 = 2*quat(1)*quat(2) + 2*quat(3)*quat(4);
A13 = 2*quat(1)*quat(3) - 2*quat(2)*quat(4);
A23 = 2*quat(1)*quat(4) + 2*quat(2)*quat(3);
A33 = - quat(1)^2 - quat(2)^2 + quat(3)^2 + quat(4)^2;

phi = atan2(A23, A33);
theta = - asin(A13);
psi = atan2(A12, A11);

euler = [ phi, theta, psi]';

end