function [ xp, yp ] = pwc1_plot ( x, y )

%*****************************************************************************80
%
%% PWC1_PLOT defines the lines forming a piecewise constant plot.
%
%  Discussion:
%
%    Given X1, Y1, X2, Y2, X3, ..., XN, YN, XN+1
%
%    such that F(X) = Y(I) for XI <= X <= XI+1, we can plot
%    this piecewise constant function by asking MATLAB to plot
%    a sequence of 2*N+2 points (XP,YP).
%
%               +-+
%        +------+ |
%        |        |
%    +---+        +---+
%    |                |
%    +                +
%
%    N = 4, uses 10 points.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    11 August 2016
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real X(N+1), the break points.
%
%    Input, real Y(N), the function values.
%
%    Output, real XP(2*N+2), YP(2*N+2), a sequence of points that
%    define the piecewise constant plot.
%
  n = length ( y );

  xp = zeros ( 2 * n, 1 );
  yp = zeros ( 2 * n, 1 );

  k = 0;
%
%  Essentially, we draw N+1 vertical lines and connect them.
%
  for i = 1 : n + 1

    if ( i == 1 )

%       k = k + 1;
%       xp(k) = x(i);
%       yp(k) = 0.0;

      k = k + 1;
      xp(k) = x(i);
      yp(k) = y(i);

    elseif ( i <= n )

      k = k + 1;
      xp(k) = x(i);
      yp(k) = y(i-1);

      k = k + 1;
      xp(k) = x(i);
      yp(k) = y(i);

    else

      k = k + 1;
      xp(k) = x(i);
      yp(k) = y(i-1);

%       k = k + 1;
%       xp(k) = x(i);
%       yp(k) = 0.0;

      break

    end

  end

  return
end

