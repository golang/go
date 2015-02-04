/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

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
     LGLOBL = 279,
     LDATA = 280,
     LRETRN = 281,
     LCONST = 282,
     LSP = 283,
     LSB = 284,
     LFP = 285,
     LPC = 286,
     LCREG = 287,
     LFLUSH = 288,
     LREG = 289,
     LFREG = 290,
     LR = 291,
     LCR = 292,
     LF = 293,
     LFPSCR = 294,
     LLR = 295,
     LCTR = 296,
     LSPR = 297,
     LSPREG = 298,
     LSEG = 299,
     LMSR = 300,
     LPCDAT = 301,
     LFUNCDAT = 302,
     LSCHED = 303,
     LXLD = 304,
     LXST = 305,
     LXOP = 306,
     LXMV = 307,
     LRLWM = 308,
     LMOVMW = 309,
     LMOVEM = 310,
     LMOVFL = 311,
     LMTFSB = 312,
     LMA = 313,
     LFCONST = 314,
     LSCONST = 315,
     LNAME = 316,
     LLAB = 317,
     LVAR = 318
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
#define LGLOBL 279
#define LDATA 280
#define LRETRN 281
#define LCONST 282
#define LSP 283
#define LSB 284
#define LFP 285
#define LPC 286
#define LCREG 287
#define LFLUSH 288
#define LREG 289
#define LFREG 290
#define LR 291
#define LCR 292
#define LF 293
#define LFPSCR 294
#define LLR 295
#define LCTR 296
#define LSPR 297
#define LSPREG 298
#define LSEG 299
#define LMSR 300
#define LPCDAT 301
#define LFUNCDAT 302
#define LSCHED 303
#define LXLD 304
#define LXST 305
#define LXOP 306
#define LXMV 307
#define LRLWM 308
#define LMOVMW 309
#define LMOVEM 310
#define LMOVFL 311
#define LMTFSB 312
#define LMA 313
#define LFCONST 314
#define LSCONST 315
#define LNAME 316
#define LLAB 317
#define LVAR 318




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 38 "a.y"
{
	Sym	*sym;
	vlong	lval;
	double	dval;
	char	sval[8];
	Addr	addr;
}
/* Line 1529 of yacc.c.  */
#line 183 "y.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

