function [ pos_d, vel_d, acc_d, jerk_d ] =  fifth_ord_poly ( t, ti, tf, xi, xf, vi, vf, ai, af )

a0 = xi;
a1 = vi;
a2 = ai/2;
a5 = 1 / (2 * (tf - ti)^3) * (af - ai - 6 * (vf + vi) / (tf - ti) + 12 * (xf - xi) / (tf - ti)^2);
a4 = 1 / (tf - ti)^3 * (vf + 2 * vi - 3 * (xf - xi) / (tf - ti) + 0.5 * ai * (tf - ti) - 2 * a5 * (tf - ti)^4);
a3 = 1 / (tf - ti)^3 * (xf - xi - vi * (tf - ti) - ai / 2 * (tf - ti)^2 - a4 * (tf - ti)^4 - a5 * (tf - ti)^5);
pos_d = a0 + a1 * (t - ti) + a2 * (t - ti)^2 + a3 * (t - ti)^3 + a4 * (t - ti)^4 + a5 * (t - ti)^5;
vel_d = a1 + 2 * a2 * (t - ti) + 3 * a3 * (t - ti)^2 + 4 * a4 * (t - ti)^3 + 5 * a5 * (t - ti)^4;
acc_d = 2 * a2 + 6 * a3 * (t - ti) + 12 * a4 * (t - ti)^2 + 20 * a5 * (t - ti)^3;
jerk_d = 6 * a3 + 24 * a4 * (t - ti) + 60 * a5 * (t - ti)^2;

end