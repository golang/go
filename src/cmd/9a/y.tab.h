/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison interface for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
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
     LMOVW = 258,
     LMOVB = 259,
     LABS = 260,
     LLOGW = 261,
     LSHW = 262,
     LADDW = 263,
     LCMP = 264,
     LCROP = 265,
     LBRA = 266,
     LFMOV = 267,
     LFCONV = 268,
     LFCMP = 269,
     LFADD = 270,
     LFMA = 271,
     LTRAP = 272,
     LXORW = 273,
     LNOP = 274,
     LEND = 275,
     LRETT = 276,
     LWORD = 277,
     LTEXT = 278,
     LDATA = 279,
     LRETRN = 280,
     LCONST = 281,
     LSP = 282,
     LSB = 283,
     LFP = 284,
     LPC = 285,
     LCREG = 286,
     LFLUSH = 287,
     LREG = 288,
     LFREG = 289,
     LR = 290,
     LCR = 291,
     LF = 292,
     LFPSCR = 293,
     LLR = 294,
     LCTR = 295,
     LSPR = 296,
     LSPREG = 297,
     LSEG = 298,
     LMSR = 299,
     LPCDAT = 300,
     LFUNCDAT = 301,
     LSCHED = 302,
     LXLD = 303,
     LXST = 304,
     LXOP = 305,
     LXMV = 306,
     LRLWM = 307,
     LMOVMW = 308,
     LMOVEM = 309,
     LMOVFL = 310,
     LMTFSB = 311,
     LMA = 312,
     LFCONST = 313,
     LSCONST = 314,
     LNAME = 315,
     LLAB = 316,
     LVAR = 317
   };
#endif
/* Tokens.  */
#define LMOVW 258
#define LMOVB 259
#define LABS 260
#define LLOGW 261
#define LSHW 262
#define LADDW 263
#define LCMP 264
#define LCROP 265
#define LBRA 266
#define LFMOV 267
#define LFCONV 268
#define LFCMP 269
#define LFADD 270
#define LFMA 271
#define LTRAP 272
#define LXORW 273
#define LNOP 274
#define LEND 275
#define LRETT 276
#define LWORD 277
#define LTEXT 278
#define LDATA 279
#define LRETRN 280
#define LCONST 281
#define LSP 282
#define LSB 283
#define LFP 284
#define LPC 285
#define LCREG 286
#define LFLUSH 287
#define LREG 288
#define LFREG 289
#define LR 290
#define LCR 291
#define LF 292
#define LFPSCR 293
#define LLR 294
#define LCTR 295
#define LSPR 296
#define LSPREG 297
#define LSEG 298
#define LMSR 299
#define LPCDAT 300
#define LFUNCDAT 301
#define LSCHED 302
#define LXLD 303
#define LXST 304
#define LXOP 305
#define LXMV 306
#define LRLWM 307
#define LMOVMW 308
#define LMOVEM 309
#define LMOVFL 310
#define LMTFSB 311
#define LMA 312
#define LFCONST 313
#define LSCONST 314
#define LNAME 315
#define LLAB 316
#define LVAR 317




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 2068 of yacc.c  */
#line 38 "a.y"

	Sym	*sym;
	vlong	lval;
	double	dval;
	char	sval[8];
	Addr	addr;



/* Line 2068 of yacc.c  */
#line 184 "y.tab.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;


