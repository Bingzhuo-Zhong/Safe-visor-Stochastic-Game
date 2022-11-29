function quat = eulerToQuat( euler )
%	Converts the ZYX Euler angles (RPY) to a quaternion vector
%
%   quat = eulerToQuat( euler )
%
%   The Euler angles vector must be written in radiants as a 3-by-1 vector.
%   The output quaternion will be returned a 4-by-1 vector with the scalar
%   element as the fourth component .
%
%   References:
%	[1] Markley, F. Landis. "Attitude error representations for Kalman filtering." 
%       Journal of guidance control and dynamics 26.2 (2003): 311-317.

phi = euler(1);
theta = euler(2);
psi = euler(3);

qx = [sin(phi/2) ;
          0      ;
          0      ;
      cos(phi/2)];

qy = [     0       ;
      sin(theta/2) ;
           0       ;
      cos(theta/2)];

qz = [    0      ;
          0      ;
      sin(psi/2) ;
      cos(psi/2)];

quat = quatProd( qx, quatProd( qy, qz ) );
norm = sqrt(quat(1)^2 + quat(2)^2 + quat(3)^2 + quat(4)^2);
quat(1) = quat(1) / norm;
quat(2) = quat(2) / norm;
quat(3) = quat(3) / norm;
quat(4) = quat(4) / norm;
end