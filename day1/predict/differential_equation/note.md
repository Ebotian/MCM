# note for differential equation

using matlab to solve it

- ctrl+tab to switch inside vscode; alt tab to switch between app
- `matlab live script` is more powerful to use matlab functions such as applying transformation, solving equation, rewriting and simplifying equation, computing integral and derivative, plotting graph, latex copying,etc.
- but in `vscode` i have copilot and syntax lighting and more powerful text editor

so, using the

## workflow

1.  `vscode` first to get ai help,
2.  then in matlab open it as `live script`, process the results

## usage

- 翻译拆分模型(Translate and Break Up Model)
  - 微分方程组(Differential Equations)
  - 条件(Conditions)
    - 边界条件(Boundary Conditions)
    - 初始条件(Initial Conditions)
- api(这里还未逐一完全了解)
  - symbolic
    - `syms`: define symbolic variable
    - `dsolve`: solve differential equation
    - `diff`: differential
    - `subs`: substitute
    - `solve`: solve equation
  - numeric
    - `ode45`: solve differential equation
    - `ode23`
    - `ode113`
    - `ode15s`
    - `ode23s`

## example
