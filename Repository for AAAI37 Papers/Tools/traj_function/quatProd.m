function quat = quatProd( quat1, quat2 )
%	Calculates the Hemiltonian product between two quaternions
%
%   quat = quatProd( quat1, quat2 )
%
%   This function compute the hamiltonian product between two quaternions
%   written as 4-by-1 vectors with the scalar component as the fourth
%   element of the vector.
%
%   References:
%	[1] Markley, F. Landis. "Attitude error representations for Kalman filtering." 
%       Journal of guidance control and dynamics 26.2 (2003): 311-317.

qv1 = [quat1(1), quat1(2), quat1(3)]';
qs1 = quat1(4);

qv2 = [quat2(1), quat2(2), quat2(3)]';
qs2 = quat2(4);

quat = [qs1 * qv2 + qs2 * qv1 - cross( qv1, qv2 ) ;
              qs1 * qs2 - dot( qv1, qv2 )        ];
          
end

