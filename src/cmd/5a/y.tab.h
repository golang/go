
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton interface for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     LTYPE1 = 258,
     LTYPE2 = 259,
     LTYPE3 = 260,
     LTYPE4 = 261,
     LTYPE5 = 262,
     LTYPE6 = 263,
     LTYPE7 = 264,
     LTYPE8 = 265,
     LTYPE9 = 266,
     LTYPEA = 267,
     LTYPEB = 268,
     LTYPEC = 269,
     LTYPED = 270,
     LTYPEE = 271,
     LTYPEF = 272,
     LTYPEG = 273,
     LTYPEH = 274,
     LTYPEI = 275,
     LTYPEJ = 276,
     LTYPEK = 277,
     LTYPEL = 278,
     LTYPEM = 279,
     LTYPEN = 280,
     LTYPEBX = 281,
     LCONST = 282,
     LSP = 283,
     LSB = 284,
     LFP = 285,
     LPC = 286,
     LTYPEX = 287,
     LR = 288,
     LREG = 289,
     LF = 290,
     LFREG = 291,
     LC = 292,
     LCREG = 293,
     LPSR = 294,
     LFCR = 295,
     LCOND = 296,
     LS = 297,
     LAT = 298,
     LFCONST = 299,
     LSCONST = 300,
     LNAME = 301,
     LLAB = 302,
     LVAR = 303
   };
#endif
/* Tokens.  */
#define LTYPE1 258
#define LTYPE2 259
#define LTYPE3 260
#define LTYPE4 261
#define LTYPE5 262
#define LTYPE6 263
#define LTYPE7 264
#define LTYPE8 265
#define LTYPE9 266
#define LTYPEA 267
#define LTYPEB 268
#define LTYPEC 269
#define LTYPED 270
#define LTYPEE 271
#define LTYPEF 272
#define LTYPEG 273
#define LTYPEH 274
#define LTYPEI 275
#define LTYPEJ 276
#define LTYPEK 277
#define LTYPEL 278
#define LTYPEM 279
#define LTYPEN 280
#define LTYPEBX 281
#define LCONST 282
#define LSP 283
#define LSB 284
#define LFP 285
#define LPC 286
#define LTYPEX 287
#define LR 288
#define LREG 289
#define LF 290
#define LFREG 291
#define LC 292
#define LCREG 293
#define LPSR 294
#define LFCR 295
#define LCOND 296
#define LS 297
#define LAT 298
#define LFCONST 299
#define LSCONST 300
#define LNAME 301
#define LLAB 302
#define LVAR 303




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 1676 of yacc.c  */
#line 38 "a.y"

	Sym	*sym;
	int32	lval;
	double	dval;
	char	sval[8];
	Gen	gen;



/* Line 1676 of yacc.c  */
#line 158 "y.tab.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;


