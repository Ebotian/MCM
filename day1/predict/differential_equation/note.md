# note for differential equation

using matlab to solve it

## dsolve

syms == symbol scalar

```matlab
syms y(t) a;
%this create function `y`, variable `t`, parameter `a`
%and all of them are symbol scalar
eqn= diff(y,t)==a*y;
%this create equation `eqn`
dsolve(eqn);
%this solve the equation
```

other part in `diff.m`, comment and code inside cells
