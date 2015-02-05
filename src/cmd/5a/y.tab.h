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
     LGLOBL = 269,
     LTYPEC = 270,
     LTYPED = 271,
     LTYPEE = 272,
     LTYPEG = 273,
     LTYPEH = 274,
     LTYPEI = 275,
     LTYPEJ = 276,
     LTYPEK = 277,
     LTYPEL = 278,
     LTYPEM = 279,
     LTYPEN = 280,
     LTYPEBX = 281,
     LTYPEPLD = 282,
     LCONST = 283,
     LSP = 284,
     LSB = 285,
     LFP = 286,
     LPC = 287,
     LTYPEX = 288,
     LTYPEPC = 289,
     LTYPEF = 290,
     LR = 291,
     LREG = 292,
     LF = 293,
     LFREG = 294,
     LC = 295,
     LCREG = 296,
     LPSR = 297,
     LFCR = 298,
     LCOND = 299,
     LS = 300,
     LAT = 301,
     LFCONST = 302,
     LSCONST = 303,
     LNAME = 304,
     LLAB = 305,
     LVAR = 306
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
#define LGLOBL 269
#define LTYPEC 270
#define LTYPED 271
#define LTYPEE 272
#define LTYPEG 273
#define LTYPEH 274
#define LTYPEI 275
#define LTYPEJ 276
#define LTYPEK 277
#define LTYPEL 278
#define LTYPEM 279
#define LTYPEN 280
#define LTYPEBX 281
#define LTYPEPLD 282
#define LCONST 283
#define LSP 284
#define LSB 285
#define LFP 286
#define LPC 287
#define LTYPEX 288
#define LTYPEPC 289
#define LTYPEF 290
#define LR 291
#define LREG 292
#define LF 293
#define LFREG 294
#define LC 295
#define LCREG 296
#define LPSR 297
#define LFCR 298
#define LCOND 299
#define LS 300
#define LAT 301
#define LFCONST 302
#define LSCONST 303
#define LNAME 304
#define LLAB 305
#define LVAR 306




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 39 "a.y"
{
	Sym	*sym;
	int32	lval;
	double	dval;
	char	sval[8];
	Addr	addr;
}
/* Line 1529 of yacc.c.  */
#line 159 "y.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

