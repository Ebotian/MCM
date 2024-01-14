%%first order differential equation
syms y(t) a;
eqn= diff(y,t)==a*y;
s1=dsolve(eqn);

%%second order differential equation
eqn2= diff(y,t,2)==a*y;
s2=dsolve(eqn2);

%%first order differential equation with initial condition
cond= y(0)==5;
s1c=dsolve(eqn,cond);

%%second order differential equation with initial condition
syms b;
eqn2c= diff(y,t,2)==a^2*y;
Dy= diff(y,t);
cond=[y(0)==b,Dy(0)==1];
s2c=dsolve(eqn2c,cond);
%ctrl+tab to switch inside vscode; alt tab to switch between app
% `matlab live script` is more powerful to use matlab functions such as applying transformation, solving equation, rewriting and simplifying equation, computing integral and derivative, plotting graph, latex copying,etc.
% but in `vscode` i have copilot and syntax lighting and more powerful text editor

%so, using the workflow,
% 1. vscode first to get ai help,
% 2. then in matlab open it as live script, process the results