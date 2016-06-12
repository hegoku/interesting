%{
#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#define YYDEBUG 1
%}
%union {
  int int_value;
  double double_value;
}
%token <double_value> DOUBLE_LITERAL
%token ADD SUB MUL DIV CR LP RP POWER
%type <double_value> expression term unary primary_expression
%%
line_list : line | line_list line;
line
  : expression CR{
    printf(">>%lf\n",$1);
  }
expression
  : term
  | expression ADD term{
    $$=$1+$3;
  }
  | expression SUB term{
    $$=$1-$3;
  };
term
  : unary
  | term MUL unary{
    $$=$1*$3;
  }
  | term DIV unary{
    $$=$1/$3;
  };
unary
  : primary_expression
  | unary POWER primary_expression{
    $$=pow($1, $3);
  };
primary_expression
  : DOUBLE_LITERAL
  | LP expression RP{
    $$=$2;
  }
  | SUB primary_expression{
    $$=-$2;
  };
%%
int yyerror(char const *str){
  extern char *yytext;
  fprintf(stderr,"parser error near %s\n",yytext);
  return 0;
}

int main(void){
  extern int yyparse(void);
  extern FILE *yyin;
  yyin=stdin;
  if(yyparse()){
    fprintf(stderr,"Error !\n");
    exit(1);
  }
}
