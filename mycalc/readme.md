### 自制变成语言-试做一个计算器

支持括号,负数,幂运算符号(^)

### 编译

```sh
$ yacc -dv mycalc.y
$ lex mycalc.l
$ cc -o mycalc y.tab.c lex.yy.c
```
