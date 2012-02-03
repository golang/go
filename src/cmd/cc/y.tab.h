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
     LORE = 258,
     LXORE = 259,
     LANDE = 260,
     LLSHE = 261,
     LRSHE = 262,
     LMDE = 263,
     LDVE = 264,
     LMLE = 265,
     LME = 266,
     LPE = 267,
     LOROR = 268,
     LANDAND = 269,
     LNE = 270,
     LEQ = 271,
     LGE = 272,
     LLE = 273,
     LRSH = 274,
     LLSH = 275,
     LMG = 276,
     LPP = 277,
     LMM = 278,
     LNAME = 279,
     LTYPE = 280,
     LFCONST = 281,
     LDCONST = 282,
     LCONST = 283,
     LLCONST = 284,
     LUCONST = 285,
     LULCONST = 286,
     LVLCONST = 287,
     LUVLCONST = 288,
     LSTRING = 289,
     LLSTRING = 290,
     LAUTO = 291,
     LBREAK = 292,
     LCASE = 293,
     LCHAR = 294,
     LCONTINUE = 295,
     LDEFAULT = 296,
     LDO = 297,
     LDOUBLE = 298,
     LELSE = 299,
     LEXTERN = 300,
     LFLOAT = 301,
     LFOR = 302,
     LGOTO = 303,
     LIF = 304,
     LINT = 305,
     LLONG = 306,
     LREGISTER = 307,
     LRETURN = 308,
     LSHORT = 309,
     LSIZEOF = 310,
     LUSED = 311,
     LSTATIC = 312,
     LSTRUCT = 313,
     LSWITCH = 314,
     LTYPEDEF = 315,
     LTYPESTR = 316,
     LUNION = 317,
     LUNSIGNED = 318,
     LWHILE = 319,
     LVOID = 320,
     LENUM = 321,
     LSIGNED = 322,
     LCONSTNT = 323,
     LVOLATILE = 324,
     LSET = 325,
     LSIGNOF = 326,
     LRESTRICT = 327,
     LINLINE = 328
   };
#endif
/* Tokens.  */
#define LORE 258
#define LXORE 259
#define LANDE 260
#define LLSHE 261
#define LRSHE 262
#define LMDE 263
#define LDVE 264
#define LMLE 265
#define LME 266
#define LPE 267
#define LOROR 268
#define LANDAND 269
#define LNE 270
#define LEQ 271
#define LGE 272
#define LLE 273
#define LRSH 274
#define LLSH 275
#define LMG 276
#define LPP 277
#define LMM 278
#define LNAME 279
#define LTYPE 280
#define LFCONST 281
#define LDCONST 282
#define LCONST 283
#define LLCONST 284
#define LUCONST 285
#define LULCONST 286
#define LVLCONST 287
#define LUVLCONST 288
#define LSTRING 289
#define LLSTRING 290
#define LAUTO 291
#define LBREAK 292
#define LCASE 293
#define LCHAR 294
#define LCONTINUE 295
#define LDEFAULT 296
#define LDO 297
#define LDOUBLE 298
#define LELSE 299
#define LEXTERN 300
#define LFLOAT 301
#define LFOR 302
#define LGOTO 303
#define LIF 304
#define LINT 305
#define LLONG 306
#define LREGISTER 307
#define LRETURN 308
#define LSHORT 309
#define LSIZEOF 310
#define LUSED 311
#define LSTATIC 312
#define LSTRUCT 313
#define LSWITCH 314
#define LTYPEDEF 315
#define LTYPESTR 316
#define LUNION 317
#define LUNSIGNED 318
#define LWHILE 319
#define LVOID 320
#define LENUM 321
#define LSIGNED 322
#define LCONSTNT 323
#define LVOLATILE 324
#define LSET 325
#define LSIGNOF 326
#define LRESTRICT 327
#define LINLINE 328




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 36 "cc.y"
{
	Node*	node;
	Sym*	sym;
	Type*	type;
	struct
	{
		Type*	t;
		uchar	c;
	} tycl;
	struct
	{
		Type*	t1;
		Type*	t2;
		Type*	t3;
		uchar	c;
	} tyty;
	struct
	{
		char*	s;
		int32	l;
	} sval;
	int32	lval;
	double	dval;
	vlong	vval;
}
/* Line 1529 of yacc.c.  */
#line 221 "y.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

