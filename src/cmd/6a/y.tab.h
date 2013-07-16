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
     LTYPE0 = 258,
     LTYPE1 = 259,
     LTYPE2 = 260,
     LTYPE3 = 261,
     LTYPE4 = 262,
     LTYPEC = 263,
     LTYPED = 264,
     LTYPEN = 265,
     LTYPER = 266,
     LTYPET = 267,
     LTYPEG = 268,
     LTYPEPC = 269,
     LTYPES = 270,
     LTYPEM = 271,
     LTYPEI = 272,
     LTYPEXC = 273,
     LTYPEX = 274,
     LTYPERT = 275,
     LCONST = 276,
     LFP = 277,
     LPC = 278,
     LSB = 279,
     LBREG = 280,
     LLREG = 281,
     LSREG = 282,
     LFREG = 283,
     LMREG = 284,
     LXREG = 285,
     LFCONST = 286,
     LSCONST = 287,
     LSP = 288,
     LNAME = 289,
     LLAB = 290,
     LVAR = 291
   };
#endif
/* Tokens.  */
#define LTYPE0 258
#define LTYPE1 259
#define LTYPE2 260
#define LTYPE3 261
#define LTYPE4 262
#define LTYPEC 263
#define LTYPED 264
#define LTYPEN 265
#define LTYPER 266
#define LTYPET 267
#define LTYPEG 268
#define LTYPEPC 269
#define LTYPES 270
#define LTYPEM 271
#define LTYPEI 272
#define LTYPEXC 273
#define LTYPEX 274
#define LTYPERT 275
#define LCONST 276
#define LFP 277
#define LPC 278
#define LSB 279
#define LBREG 280
#define LLREG 281
#define LSREG 282
#define LFREG 283
#define LMREG 284
#define LXREG 285
#define LFCONST 286
#define LSCONST 287
#define LSP 288
#define LNAME 289
#define LLAB 290
#define LVAR 291




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 37 "a.y"
{
	Sym	*sym;
	vlong	lval;
	double	dval;
	char	sval[8];
	Gen	gen;
	Gen2	gen2;
}
/* Line 1529 of yacc.c.  */
#line 130 "y.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

