% Quadrotor_2d_DFA: DFA for the 2 dimensional quadrotor (A_N)
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

DFA = DFA_options();

DFA.n_state = 6;
DFA.sink_state = 6;

bd1 = 0.3;
bd2 = 0.4;
bd3 = 0.45;
bd4 = 0.5;

dfa_var.C11 = [-1;1];
dfa_var.d11 = [-bd1;bd4];
dfa_var.C12 = [-1;1];
dfa_var.d12 = [bd4;-bd1];

dfa_var.C2 = [-1;1];
dfa_var.d2 = [bd1;bd1];

dfa_var.C3 = [-1;1];
dfa_var.d3 = [bd2;bd2];

dfa_var.C4 = [-1;1];
dfa_var.d4 = [bd3;bd3];

dfa_var.C5 = [-1;1];
dfa_var.d5 = [bd4;bd4];

dfa_var.C61 = 1;
dfa_var.d61 = -bd4;
dfa_var.C62 = -1;
dfa_var.d62 = -bd4;


dfa_var.C71 = 1;
dfa_var.d71 = -bd2;
dfa_var.C72 = -1;
dfa_var.d72 = -bd2;

dfa_var.C81 = 1;
dfa_var.d81 = -bd3;
dfa_var.C82 = -1;
dfa_var.d82 = -bd3;


DFA.dfa_var = dfa_var;

p1 = 'min(dfa_var.C11*x<=dfa_var.d11)||min(dfa_var.C12*x<=dfa_var.d12)';
p2 = 'dfa_var.C2*x<=dfa_var.d2';
p3 = 'dfa_var.C3*x<=dfa_var.d3';
p4 = 'dfa_var.C4*x<=dfa_var.d4';
p5 = 'dfa_var.C5*x<=dfa_var.d5';
p6 = 'dfa_var.C61*x<=dfa_var.d61 ||dfa_var.C62*x<=dfa_var.d62';
p7 = 'dfa_var.C71*x<=dfa_var.d71 ||dfa_var.C72*x<=dfa_var.d72';
p8 = 'dfa_var.C81*x<=dfa_var.d81 ||dfa_var.C82*x<=dfa_var.d82';
p9 = '1';

DFA_m(1,2)={p1};
DFA_m(1,3)={p2};
DFA_m(1,6)={p6};
DFA_m(2,2)={p1};
DFA_m(2,3)={p2};
DFA_m(2,6)={p6};
DFA_m(3,4)={p3};
DFA_m(3,6)={p7};
DFA_m(4,5)={p4};
DFA_m(4,6)={p8};
DFA_m(5,2)={p5};
DFA_m(5,6)={p6};
DFA_m(6,6)={p9};

DFA.spec_type = 2;
DFA.acc_state = 6;
DFA.dfa = DFA_m;

for iq = 1:1:DFA.n_state
    for iq2 = 1:1:DFA.n_state
        if ~isempty(DFA.dfa{iq,iq2})
            text = ['if ',DFA.dfa{iq,iq2},';ind=1;end'];
            DFA.dfa{iq,iq2} = text;
        end
    end
end
 
DFA_emap = DFA;
DFA_emap.sink_state = 6;

cp11 = Lmap_options();
cp11.A = dfa_var.C11;
cp11.b = dfa_var.d11;
cp12 = Lmap_options();
cp12.A = dfa_var.C12;
cp12.b = dfa_var.d12;

cp2 = Lmap_options();
cp2.A = dfa_var.C2;
cp2.b = dfa_var.d2;

cp3 = Lmap_options();
cp3.A = dfa_var.C3;
cp3.b = dfa_var.d3;

cp4 = Lmap_options();
cp4.A = dfa_var.C4;
cp4.b = dfa_var.d4;

cp5 = Lmap_options();
cp5.A = dfa_var.C5;
cp5.b = dfa_var.d5;

cp61 = Lmap_options();
cp61.A = dfa_var.C61;
cp61.b = dfa_var.d61;
cp62 = Lmap_options();
cp62.A = dfa_var.C62;
cp62.b = dfa_var.d62;

cp71 = Lmap_options();
cp71.A = dfa_var.C71;
cp71.b = dfa_var.d71;
cp72 = Lmap_options();
cp72.A = dfa_var.C72;
cp72.b = dfa_var.d72;

cp81 = Lmap_options();
cp81.A = dfa_var.C81;
cp81.b = dfa_var.d81;
cp82 = Lmap_options();
cp82.A = dfa_var.C82;
cp82.b = dfa_var.d82;



DFA_emap.dfa{1,2} = [{cp11};{cp12}];
DFA_emap.dfa{1,3} = {cp2};
DFA_emap.dfa{1,6} = [{cp61};{cp62}];
DFA_emap.dfa{2,2} = [{cp11};{cp12}];
DFA_emap.dfa{2,3} = {cp2};
DFA_emap.dfa{2,6} = [{cp61};{cp62}];
DFA_emap.dfa{3,4} = {cp3};
DFA_emap.dfa{3,6} = [{cp71};{cp72}];
DFA_emap.dfa{4,5} = {cp4};
DFA_emap.dfa{4,6} = [{cp81};{cp82}];
DFA_emap.dfa{5,2} = {cp5};
DFA_emap.dfa{5,6} = [{cp61};{cp62}];
DFA_emap.dfa{6,6}= [];

clear DFA_m dfa_var;
clear bd1 bd2 bd3 bd4
clear p1  p2  p3 p4 p5 p6 p7 p8 p9 p10 iq iq2 text;
clear cp11 cp12 cp2 cp3 cp4 cp5 cp61 cp62 cp71 cp72 cp81 cp82;
