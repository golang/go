/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



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
     LPREFETCH = 307,
     LREGISTER = 308,
     LRETURN = 309,
     LSHORT = 310,
     LSIZEOF = 311,
     LUSED = 312,
     LSTATIC = 313,
     LSTRUCT = 314,
     LSWITCH = 315,
     LTYPEDEF = 316,
     LTYPESTR = 317,
     LUNION = 318,
     LUNSIGNED = 319,
     LWHILE = 320,
     LVOID = 321,
     LENUM = 322,
     LSIGNED = 323,
     LCONSTNT = 324,
     LVOLATILE = 325,
     LSET = 326,
     LSIGNOF = 327,
     LRESTRICT = 328,
     LINLINE = 329
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
#define LPREFETCH 307
#define LREGISTER 308
#define LRETURN 309
#define LSHORT 310
#define LSIZEOF 311
#define LUSED 312
#define LSTATIC 313
#define LSTRUCT 314
#define LSWITCH 315
#define LTYPEDEF 316
#define LTYPESTR 317
#define LUNION 318
#define LUNSIGNED 319
#define LWHILE 320
#define LVOID 321
#define LENUM 322
#define LSIGNED 323
#define LCONSTNT 324
#define LVOLATILE 325
#define LSET 326
#define LSIGNOF 327
#define LRESTRICT 328
#define LINLINE 329




/* Copy the first part of user declarations.  */
#line 31 "cc.y"

#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and cc.h re-#defines getc */
#include "cc.h"


/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

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
/* Line 193 of yacc.c.  */
#line 276 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 289 "y.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1188

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  99
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  75
/* YYNRULES -- Number of rules.  */
#define YYNRULES  247
/* YYNRULES -- Number of states.  */
#define YYNSTATES  417

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   329

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    97,     2,     2,     2,    35,    22,     2,
      38,    93,    33,    31,     4,    32,    36,    34,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    17,     3,
      25,     5,    26,    16,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    37,     2,    94,    21,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    95,    20,    96,    98,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    18,    19,
      23,    24,    27,    28,    29,    30,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,    10,    14,    15,    16,    23,
      25,    26,    31,    35,    37,    41,    43,    47,    52,    57,
      60,    64,    66,    67,    72,    76,    77,    82,    84,    88,
      89,    94,    95,   101,   102,   104,   106,   110,   112,   116,
     119,   120,   122,   125,   129,   131,   133,   138,   143,   146,
     150,   154,   156,   160,   164,   167,   170,   173,   177,   179,
     182,   184,   186,   189,   190,   192,   194,   197,   200,   204,
     208,   212,   213,   216,   219,   221,   224,   228,   231,   234,
     237,   239,   242,   244,   247,   250,   251,   254,   260,   268,
     269,   280,   286,   294,   298,   304,   307,   310,   314,   320,
     326,   332,   333,   335,   336,   338,   340,   342,   346,   348,
     352,   356,   360,   364,   368,   372,   376,   380,   384,   388,
     392,   396,   400,   404,   408,   412,   416,   420,   426,   430,
     434,   438,   442,   446,   450,   454,   458,   462,   466,   470,
     472,   478,   486,   488,   491,   494,   497,   500,   503,   506,
     509,   512,   515,   518,   522,   528,   534,   539,   544,   548,
     552,   555,   558,   560,   562,   564,   566,   568,   570,   572,
     574,   576,   578,   580,   582,   585,   587,   590,   591,   593,
     595,   599,   600,   605,   606,   608,   610,   612,   614,   617,
     620,   624,   627,   631,   633,   635,   638,   639,   644,   647,
     650,   651,   656,   659,   662,   663,   664,   672,   673,   679,
     681,   683,   686,   687,   690,   692,   694,   696,   698,   701,
     703,   705,   707,   711,   714,   718,   720,   722,   724,   726,
     728,   730,   732,   734,   736,   738,   740,   742,   744,   746,
     748,   750,   752,   754,   756,   758,   760,   762
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     100,     0,    -1,    -1,   100,   101,    -1,   152,     3,    -1,
     152,   104,     3,    -1,    -1,    -1,   152,   106,   102,   111,
     103,   129,    -1,   106,    -1,    -1,   106,   105,     5,   123,
      -1,   104,     4,   104,    -1,   107,    -1,    33,   163,   106,
      -1,   172,    -1,    38,   106,    93,    -1,   107,    38,   127,
      93,    -1,   107,    37,   139,    94,    -1,   155,     3,    -1,
     155,   109,     3,    -1,   106,    -1,    -1,   106,   110,     5,
     123,    -1,   109,     4,   109,    -1,    -1,   111,   155,   112,
       3,    -1,   106,    -1,   112,     4,   112,    -1,    -1,   154,
     114,   116,     3,    -1,    -1,   113,   154,   115,   116,     3,
      -1,    -1,   117,    -1,   118,    -1,   117,     4,   117,    -1,
     106,    -1,   172,    17,   140,    -1,    17,   140,    -1,    -1,
     120,    -1,    33,   163,    -1,    33,   163,   120,    -1,   121,
      -1,   122,    -1,   121,    38,   127,    93,    -1,   121,    37,
     139,    94,    -1,    38,    93,    -1,    37,   139,    94,    -1,
      38,   120,    93,    -1,   142,    -1,    95,   126,    96,    -1,
      37,   140,    94,    -1,    36,   173,    -1,   124,     5,    -1,
     123,     4,    -1,   125,   123,     4,    -1,   124,    -1,   125,
     124,    -1,   125,    -1,   123,    -1,   125,   123,    -1,    -1,
     128,    -1,   171,    -1,   154,   119,    -1,   154,   106,    -1,
      36,    36,    36,    -1,   128,     4,   128,    -1,    95,   130,
      96,    -1,    -1,   130,   108,    -1,   130,   133,    -1,   132,
      -1,   131,   132,    -1,    56,   142,    17,    -1,    59,    17,
      -1,    42,    17,    -1,     1,     3,    -1,   135,    -1,   131,
     135,    -1,   138,    -1,   155,   109,    -1,   138,     3,    -1,
      -1,   136,   129,    -1,    67,    38,   141,    93,   133,    -1,
      67,    38,   141,    93,   133,    62,   133,    -1,    -1,   137,
      65,    38,   134,     3,   138,     3,   138,    93,   133,    -1,
      83,    38,   141,    93,   133,    -1,    60,   133,    83,    38,
     141,    93,     3,    -1,    72,   138,     3,    -1,    78,    38,
     141,    93,   133,    -1,    55,     3,    -1,    58,     3,    -1,
      66,   173,     3,    -1,    75,    38,   148,    93,     3,    -1,
      70,    38,   148,    93,     3,    -1,    89,    38,   148,    93,
       3,    -1,    -1,   141,    -1,    -1,   140,    -1,   142,    -1,
     142,    -1,   141,     4,   141,    -1,   143,    -1,   142,    33,
     142,    -1,   142,    34,   142,    -1,   142,    35,   142,    -1,
     142,    31,   142,    -1,   142,    32,   142,    -1,   142,    29,
     142,    -1,   142,    30,   142,    -1,   142,    25,   142,    -1,
     142,    26,   142,    -1,   142,    28,   142,    -1,   142,    27,
     142,    -1,   142,    24,   142,    -1,   142,    23,   142,    -1,
     142,    22,   142,    -1,   142,    21,   142,    -1,   142,    20,
     142,    -1,   142,    19,   142,    -1,   142,    18,   142,    -1,
     142,    16,   141,    17,   142,    -1,   142,     5,   142,    -1,
     142,    15,   142,    -1,   142,    14,   142,    -1,   142,    13,
     142,    -1,   142,    12,   142,    -1,   142,    11,   142,    -1,
     142,     9,   142,    -1,   142,    10,   142,    -1,   142,     8,
     142,    -1,   142,     7,   142,    -1,   142,     6,   142,    -1,
     144,    -1,    38,   154,   119,    93,   143,    -1,    38,   154,
     119,    93,    95,   126,    96,    -1,   145,    -1,    33,   143,
      -1,    22,   143,    -1,    31,   143,    -1,    32,   143,    -1,
      97,   143,    -1,    98,   143,    -1,    40,   143,    -1,    41,
     143,    -1,    74,   144,    -1,    90,   144,    -1,    38,   141,
      93,    -1,    74,    38,   154,   119,    93,    -1,    90,    38,
     154,   119,    93,    -1,   145,    38,   148,    93,    -1,   145,
      37,   141,    94,    -1,   145,    39,   173,    -1,   145,    36,
     173,    -1,   145,    40,    -1,   145,    41,    -1,   171,    -1,
      46,    -1,    47,    -1,    48,    -1,    49,    -1,    45,    -1,
      44,    -1,    50,    -1,    51,    -1,   146,    -1,   147,    -1,
      52,    -1,   146,    52,    -1,    53,    -1,   147,    53,    -1,
      -1,   149,    -1,   142,    -1,   149,     4,   149,    -1,    -1,
      95,   151,   113,    96,    -1,    -1,   155,    -1,   156,    -1,
     168,    -1,   165,    -1,   156,   162,    -1,   168,   162,    -1,
     165,   156,   163,    -1,   165,   168,    -1,   165,   168,   162,
      -1,   153,    -1,   153,    -1,    77,   173,    -1,    -1,    77,
     173,   157,   150,    -1,    77,   150,    -1,    81,   173,    -1,
      -1,    81,   173,   158,   150,    -1,    81,   150,    -1,    85,
     173,    -1,    -1,    -1,    85,   173,   159,    95,   160,   167,
      96,    -1,    -1,    85,    95,   161,   167,    96,    -1,    43,
      -1,   164,    -1,   162,   164,    -1,    -1,   163,   170,    -1,
     168,    -1,   170,    -1,   169,    -1,   166,    -1,   165,   166,
      -1,   170,    -1,   169,    -1,    42,    -1,    42,     5,   142,
      -1,   167,     4,    -1,   167,     4,   167,    -1,    57,    -1,
      73,    -1,    68,    -1,    69,    -1,    86,    -1,    82,    -1,
      64,    -1,    61,    -1,    84,    -1,    54,    -1,    76,    -1,
      63,    -1,    79,    -1,    80,    -1,    71,    -1,    92,    -1,
      87,    -1,    88,    -1,    91,    -1,    42,    -1,   173,    -1,
      42,    -1,    43,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   101,   101,   102,   108,   112,   114,   128,   113,   143,
     148,   147,   155,   158,   159,   166,   167,   171,   175,   184,
     188,   194,   200,   199,   211,   224,   225,   228,   232,   239,
     238,   244,   243,   250,   254,   257,   261,   264,   269,   273,
     282,   285,   288,   293,   298,   301,   302,   306,   312,   316,
     320,   326,   327,   333,   337,   342,   345,   346,   350,   351,
     357,   358,   359,   365,   368,   375,   376,   381,   386,   390,
     396,   406,   409,   413,   419,   420,   426,   430,   434,   440,
     444,   445,   451,   452,   458,   459,   459,   470,   476,   484,
     484,   495,   499,   503,   508,   522,   526,   530,   534,   538,
     542,   548,   551,   554,   557,   560,   567,   568,   574,   575,
     579,   583,   587,   591,   595,   599,   603,   607,   611,   615,
     619,   623,   627,   631,   635,   639,   643,   647,   651,   655,
     659,   663,   667,   671,   675,   679,   683,   687,   691,   697,
     698,   705,   713,   714,   718,   722,   726,   730,   734,   738,
     742,   746,   750,   756,   760,   766,   772,   780,   784,   789,
     794,   798,   802,   803,   810,   817,   824,   831,   838,   845,
     852,   859,   860,   863,   873,   891,   901,   919,   922,   925,
     926,   933,   932,   955,   959,   962,   967,   972,   978,   986,
     992,   998,  1004,  1012,  1020,  1027,  1033,  1032,  1044,  1052,
    1058,  1057,  1069,  1077,  1086,  1090,  1085,  1107,  1106,  1115,
    1121,  1122,  1128,  1131,  1137,  1138,  1139,  1142,  1143,  1149,
    1150,  1153,  1157,  1161,  1162,  1165,  1166,  1167,  1168,  1169,
    1170,  1171,  1172,  1173,  1176,  1177,  1178,  1179,  1180,  1181,
    1182,  1185,  1186,  1187,  1190,  1205,  1217,  1218
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "';'", "','", "'='", "LORE", "LXORE",
  "LANDE", "LLSHE", "LRSHE", "LMDE", "LDVE", "LMLE", "LME", "LPE", "'?'",
  "':'", "LOROR", "LANDAND", "'|'", "'^'", "'&'", "LNE", "LEQ", "'<'",
  "'>'", "LGE", "LLE", "LRSH", "LLSH", "'+'", "'-'", "'*'", "'/'", "'%'",
  "'.'", "'['", "'('", "LMG", "LPP", "LMM", "LNAME", "LTYPE", "LFCONST",
  "LDCONST", "LCONST", "LLCONST", "LUCONST", "LULCONST", "LVLCONST",
  "LUVLCONST", "LSTRING", "LLSTRING", "LAUTO", "LBREAK", "LCASE", "LCHAR",
  "LCONTINUE", "LDEFAULT", "LDO", "LDOUBLE", "LELSE", "LEXTERN", "LFLOAT",
  "LFOR", "LGOTO", "LIF", "LINT", "LLONG", "LPREFETCH", "LREGISTER",
  "LRETURN", "LSHORT", "LSIZEOF", "LUSED", "LSTATIC", "LSTRUCT", "LSWITCH",
  "LTYPEDEF", "LTYPESTR", "LUNION", "LUNSIGNED", "LWHILE", "LVOID",
  "LENUM", "LSIGNED", "LCONSTNT", "LVOLATILE", "LSET", "LSIGNOF",
  "LRESTRICT", "LINLINE", "')'", "']'", "'{'", "'}'", "'!'", "'~'",
  "$accept", "prog", "xdecl", "@1", "@2", "xdlist", "@3", "xdecor",
  "xdecor2", "adecl", "adlist", "@4", "pdecl", "pdlist", "edecl", "@5",
  "@6", "zedlist", "edlist", "edecor", "abdecor", "abdecor1", "abdecor2",
  "abdecor3", "init", "qual", "qlist", "ilist", "zarglist", "arglist",
  "block", "slist", "labels", "label", "stmnt", "forexpr", "ulstmnt", "@7",
  "@8", "zcexpr", "zexpr", "lexpr", "cexpr", "expr", "xuexpr", "uexpr",
  "pexpr", "string", "lstring", "zelist", "elist", "sbody", "@9",
  "zctlist", "types", "tlist", "ctlist", "complex", "@10", "@11", "@12",
  "@13", "@14", "gctnlist", "zgnlist", "gctname", "gcnlist", "gcname",
  "enum", "tname", "cname", "gname", "name", "tag", "ltag", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,    59,    44,    61,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,    63,    58,   268,   269,
     124,    94,    38,   270,   271,    60,    62,   272,   273,   274,
     275,    43,    45,    42,    47,    37,    46,    91,    40,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,   315,   316,
     317,   318,   319,   320,   321,   322,   323,   324,   325,   326,
     327,   328,   329,    41,    93,   123,   125,    33,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    99,   100,   100,   101,   101,   102,   103,   101,   104,
     105,   104,   104,   106,   106,   107,   107,   107,   107,   108,
     108,   109,   110,   109,   109,   111,   111,   112,   112,   114,
     113,   115,   113,   116,   116,   117,   117,   118,   118,   118,
     119,   119,   120,   120,   120,   121,   121,   121,   122,   122,
     122,   123,   123,   124,   124,   124,   125,   125,   125,   125,
     126,   126,   126,   127,   127,   128,   128,   128,   128,   128,
     129,   130,   130,   130,   131,   131,   132,   132,   132,   133,
     133,   133,   134,   134,   135,   136,   135,   135,   135,   137,
     135,   135,   135,   135,   135,   135,   135,   135,   135,   135,
     135,   138,   138,   139,   139,   140,   141,   141,   142,   142,
     142,   142,   142,   142,   142,   142,   142,   142,   142,   142,
     142,   142,   142,   142,   142,   142,   142,   142,   142,   142,
     142,   142,   142,   142,   142,   142,   142,   142,   142,   143,
     143,   143,   144,   144,   144,   144,   144,   144,   144,   144,
     144,   144,   144,   145,   145,   145,   145,   145,   145,   145,
     145,   145,   145,   145,   145,   145,   145,   145,   145,   145,
     145,   145,   145,   146,   146,   147,   147,   148,   148,   149,
     149,   151,   150,   152,   152,   153,   153,   153,   153,   153,
     153,   153,   153,   154,   155,   156,   157,   156,   156,   156,
     158,   156,   156,   156,   159,   160,   156,   161,   156,   156,
     162,   162,   163,   163,   164,   164,   164,   165,   165,   166,
     166,   167,   167,   167,   167,   168,   168,   168,   168,   168,
     168,   168,   168,   168,   169,   169,   169,   169,   169,   169,
     169,   170,   170,   170,   171,   172,   173,   173
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     2,     3,     0,     0,     6,     1,
       0,     4,     3,     1,     3,     1,     3,     4,     4,     2,
       3,     1,     0,     4,     3,     0,     4,     1,     3,     0,
       4,     0,     5,     0,     1,     1,     3,     1,     3,     2,
       0,     1,     2,     3,     1,     1,     4,     4,     2,     3,
       3,     1,     3,     3,     2,     2,     2,     3,     1,     2,
       1,     1,     2,     0,     1,     1,     2,     2,     3,     3,
       3,     0,     2,     2,     1,     2,     3,     2,     2,     2,
       1,     2,     1,     2,     2,     0,     2,     5,     7,     0,
      10,     5,     7,     3,     5,     2,     2,     3,     5,     5,
       5,     0,     1,     0,     1,     1,     1,     3,     1,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     5,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     1,
       5,     7,     1,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     3,     5,     5,     4,     4,     3,     3,
       2,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     1,     2,     0,     1,     1,
       3,     0,     4,     0,     1,     1,     1,     1,     2,     2,
       3,     2,     3,     1,     1,     2,     0,     4,     2,     2,
       0,     4,     2,     2,     0,     0,     7,     0,     5,     1,
       1,     2,     0,     2,     1,     1,     1,     1,     2,     1,
       1,     1,     3,     2,     3,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,   183,     1,   209,   234,   225,   232,   236,   231,   227,
     228,   239,   226,   235,     0,   237,   238,     0,   230,   233,
       0,   229,   241,   242,   243,   240,     3,     0,   194,   184,
     185,   187,   217,   186,   220,   219,   246,   247,   181,   198,
     195,   202,   199,   207,   203,     4,   212,     0,     0,     6,
      13,    15,   245,   188,   210,   214,   216,   215,   212,   218,
     191,   189,     0,     0,     0,     0,     0,     0,     0,     5,
       0,    25,     0,   103,    63,   211,   190,   192,     0,   193,
      29,   197,   201,   221,     0,   205,    14,   213,    16,    12,
       9,     7,     0,     0,     0,     0,     0,     0,     0,     0,
     244,   168,   167,   163,   164,   165,   166,   169,   170,   173,
     175,     0,     0,     0,     0,     0,   104,   105,   108,   139,
     142,   171,   172,   162,     0,     0,    64,    40,    65,   182,
      31,    33,     0,   223,   208,     0,     0,     0,     0,    11,
      51,   144,   145,   146,   143,     0,   106,    40,   149,   150,
       0,   151,     0,   152,   147,   148,    18,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   177,
       0,   160,   161,   174,   176,     0,    17,     0,   212,   103,
       0,    67,    66,    41,    44,    45,    33,     0,    37,     0,
      34,    35,    15,   222,   224,     0,    71,     8,    27,     0,
       0,     0,    61,    58,    60,     0,     0,   153,   212,     0,
       0,    40,    40,   128,   138,   137,   136,   134,   135,   133,
     132,   131,   130,   129,     0,   126,   125,   124,   123,   122,
     121,   120,   116,   117,   119,   118,   114,   115,   112,   113,
     109,   110,   111,   159,     0,   179,     0,   178,   158,    68,
      69,    42,     0,    48,     0,   103,    63,     0,    39,    30,
       0,     0,   206,     0,    26,     0,    54,     0,    56,    55,
      62,    59,    52,   107,    42,     0,     0,     0,     0,   157,
     156,     0,    43,    49,    50,     0,     0,    32,    36,    38,
       0,   244,     0,     0,     0,     0,     0,     0,     0,     0,
     101,     0,     0,     0,     0,    70,    72,    85,    74,    73,
      80,     0,     0,     0,   102,     0,    28,    53,    57,     0,
     140,   154,   155,   127,   180,    47,    46,    79,    78,    95,
       0,    96,    77,     0,     0,     0,   177,     0,   177,     0,
       0,   177,    75,    81,    86,     0,    84,    19,    21,     0,
       0,    76,     0,    97,     0,     0,    93,     0,     0,     0,
       0,   101,     0,    20,     0,   141,     0,     0,     0,     0,
       0,     0,     0,     0,    82,     0,     0,    24,     0,    87,
      99,    98,    94,    91,   100,   101,    83,    23,     0,     0,
       0,    92,    88,   101,     0,     0,    90
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,    26,    71,   136,    48,    72,   208,    50,   326,
     369,   382,    91,   219,    78,   131,   206,   209,   210,   211,
     202,   203,   204,   205,   222,   223,   224,   225,   125,   126,
     217,   283,   327,   328,   329,   393,   330,   331,   332,   333,
     115,   116,   334,   146,   118,   119,   120,   121,   122,   266,
     267,    39,    62,    27,    79,   127,    29,    30,    63,    64,
      66,   135,    65,    53,    67,    54,    31,    32,    84,    33,
      34,    35,   123,    51,    52
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -331
static const yytype_int16 yypact[] =
{
    -331,   548,  -331,  -331,  -331,  -331,  -331,  -331,  -331,  -331,
    -331,  -331,  -331,  -331,    -3,  -331,  -331,    -3,  -331,  -331,
     149,  -331,  -331,  -331,  -331,  -331,  -331,   264,  -331,  -331,
     965,   929,  -331,   965,  -331,  -331,  -331,  -331,  -331,  -331,
     -75,  -331,   -72,  -331,   -60,  -331,  -331,   307,    60,   270,
     156,  -331,  -331,   965,  -331,  -331,  -331,  -331,  -331,  -331,
     965,   965,   929,   -44,   -44,    29,   -15,   199,   -10,  -331,
     307,  -331,    83,   756,   849,  -331,   140,   965,   889,  -331,
    -331,  -331,  -331,    86,    12,  -331,  -331,  -331,  -331,  -331,
      90,   929,   686,   756,   756,   756,   756,   615,   756,   756,
    -331,  -331,  -331,  -331,  -331,  -331,  -331,  -331,  -331,  -331,
    -331,   791,   826,   756,   756,     9,  -331,  1084,  -331,  -331,
     708,    54,    57,  -331,   110,    56,   152,   310,  -331,  -331,
    -331,   279,   756,    29,  -331,    29,    63,   307,   165,  -331,
    1084,  -331,  -331,  -331,  -331,    30,  1084,    44,  -331,  -331,
     615,  -331,   615,  -331,  -331,  -331,  -331,   756,   756,   756,
     756,   756,   756,   756,   756,   756,   756,   756,   756,   756,
     756,   756,   756,   756,   756,   756,   756,   756,   756,   756,
     756,   756,   756,   756,   756,   756,   756,   157,   756,   756,
     157,  -331,  -331,  -331,  -331,   115,  -331,   849,  -331,   756,
     128,  -331,  -331,  -331,   182,  -331,   279,   756,  -331,   164,
     200,  -331,   208,  1084,  -331,    13,  -331,  -331,  -331,   262,
     157,   756,   225,   228,   165,    73,   756,  -331,  -331,    -7,
     150,    44,    44,  1084,  1084,  1084,  1084,  1084,  1084,  1084,
    1084,  1084,  1084,  1084,    28,   304,  1100,  1115,  1129,  1142,
    1153,  1153,   433,   433,   433,   433,   333,   333,   265,   265,
    -331,  -331,  -331,  -331,     8,  1084,   153,   236,  -331,  -331,
    -331,   147,   158,  -331,   161,   756,   849,   247,  -331,  -331,
     279,   756,  -331,   341,  -331,   307,  -331,   175,  -331,  -331,
     254,   228,  -331,  -331,   135,   721,   188,   190,   756,  -331,
    -331,   756,  -331,  -331,  -331,   191,   211,  -331,  -331,  -331,
     298,   301,   338,   756,   343,   339,   439,   157,   319,   321,
     756,   322,   323,   324,   332,  -331,  -331,   509,  -331,  -331,
    -331,    63,   306,   372,   373,   277,  -331,  -331,  -331,   165,
    -331,  -331,  -331,   425,  -331,  -331,  -331,  -331,  -331,  -331,
    1053,  -331,  -331,   293,   375,   756,   756,   400,   756,   756,
     756,   756,  -331,  -331,  -331,   396,  -331,  -331,   430,   285,
     377,  -331,   431,  -331,    55,   381,  -331,   382,    62,    64,
     383,   615,   473,  -331,   307,  -331,   756,   439,   479,   490,
     439,   439,   493,   497,  -331,   307,   686,  -331,    66,   440,
    -331,  -331,  -331,  -331,  -331,   756,   499,  -331,   498,   439,
     504,  -331,  -331,   756,   415,   439,  -331
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -331,  -331,  -331,  -331,  -331,   445,  -331,   -26,  -331,  -331,
    -330,  -331,  -331,   233,  -331,  -331,  -331,   313,   230,  -331,
    -132,  -187,  -331,  -331,   -82,   292,  -331,   181,   245,   326,
     193,  -331,  -331,   198,  -227,  -331,   203,  -331,  -331,  -309,
    -181,  -183,   -83,   -45,   -38,   243,  -331,  -331,  -331,  -175,
     226,    10,  -331,  -331,    -1,     0,   -88,   495,  -331,  -331,
    -331,  -331,  -331,   -14,   -51,   -28,  -331,   501,   -85,   218,
     231,   -24,   -52,  -127,   -12
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -205
static const yytype_int16 yytable[] =
{
      28,    49,    40,   137,   212,    42,    57,    76,    44,    57,
     139,   357,   226,   274,   145,   230,   133,   133,   272,    61,
    -196,    68,   128,  -200,   278,    75,   228,    41,   117,    57,
     199,   229,   226,    75,   226,  -204,    57,    57,   287,    36,
      37,    86,   274,    87,    90,   298,    77,   140,   214,    75,
     215,    38,    87,    57,   397,   141,   142,   143,   144,   226,
     148,   149,    80,    69,    70,   406,   226,   145,   226,   145,
     226,    83,   394,    81,    82,   154,   155,   228,   130,   212,
      85,   199,   229,    88,   302,   244,   273,   213,    92,   353,
      28,   132,    38,   140,   305,   -10,   410,   147,   309,   296,
     297,   201,   299,   156,   414,   264,   193,   302,   134,   282,
     194,   218,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   227,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   290,   293,   265,   128,   195,   271,   387,   196,
     231,   269,   232,   212,   117,   390,   197,   391,   216,   408,
     399,   198,   117,   402,   403,   199,   200,   279,   228,   292,
      36,    37,   199,   229,    68,   263,   117,   294,   268,   140,
     198,   375,   412,   377,   199,   200,   380,    93,   416,    36,
      37,    36,    37,    73,    74,   335,    94,    95,    96,    36,
      37,   220,   221,    97,   280,    98,    99,   100,   286,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   275,
     276,   273,    22,    23,   128,   281,    24,    22,    23,   288,
     117,    24,    46,   289,    22,    23,   117,    47,    24,   111,
     301,    36,    37,   295,    43,    86,   300,    87,    55,    60,
     307,    55,   303,   343,   304,   112,   265,   340,   338,   218,
     138,    56,   113,   114,    56,   284,   285,    45,   350,   337,
      87,    55,   374,    -9,    -9,   -10,   378,   379,    55,    55,
     367,   341,    28,   342,    56,   345,    22,    23,   383,   384,
      24,    56,    56,   395,   140,    55,   207,    46,   184,   185,
     186,   347,    47,   398,   346,   354,    36,    37,    56,   368,
      46,   265,    46,   265,   407,    47,   265,    47,   348,    36,
      37,    36,    37,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
      46,   349,   310,   198,  -101,    47,   351,   199,   200,    36,
      37,   140,    36,    37,   151,   153,   352,   355,   368,   356,
     358,   359,   360,    93,   182,   183,   184,   185,   186,   368,
     361,   365,    94,    95,    96,   366,   372,   226,   373,    97,
      28,    98,    99,   311,     3,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,     4,   312,   313,     5,   314,
     315,   316,     6,   376,     7,     8,   -89,   317,   318,     9,
      10,   319,    11,   320,    12,   111,   321,    13,    14,   322,
      15,    16,    17,    18,   323,    19,    20,    21,    22,    23,
     324,   112,    24,    25,   381,   -22,   -85,   325,   113,   114,
     310,   168,  -101,   169,   170,   171,   172,   173,   174,   175,
     176,   177,   178,   179,   180,   181,   182,   183,   184,   185,
     186,    93,   180,   181,   182,   183,   184,   185,   186,   386,
      94,    95,    96,   385,   388,   389,   392,    97,   396,    98,
      99,   311,   400,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   401,   312,   313,   404,   314,   315,   316,
     405,   411,   409,   384,   -89,   317,   318,   413,   415,   319,
     308,   320,  -101,   111,   321,    89,   291,   322,   336,   277,
     370,   306,   323,   270,   364,   362,    58,   344,   324,   112,
     363,    93,    59,     0,   -85,     0,   113,   114,     0,     0,
      94,    95,    96,     0,     0,     0,     0,    97,     2,    98,
      99,   311,     0,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,     0,   312,   313,     0,   314,   315,   316,
       0,     0,     0,     0,   -89,   317,   318,     0,     0,   319,
       0,   320,     0,   111,   321,     0,     0,   322,     0,     0,
       0,     3,   323,     0,     0,     0,     0,     0,   324,   112,
       0,     0,     4,     0,     0,     5,   113,   114,     0,     6,
       0,     7,     8,     0,     0,     0,     9,    10,     0,    11,
       0,    12,     0,     0,    13,    14,     0,    15,    16,    17,
      18,     0,    19,    20,    21,    22,    23,    93,     0,    24,
      25,     0,     0,     0,     0,     0,    94,    95,    96,     0,
       0,     0,     0,    97,     0,    98,    99,   100,     3,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,     4,
       0,     0,     5,     0,     0,     0,     6,     0,     7,     8,
       0,     0,     0,     9,    10,     0,    11,     0,    12,   111,
       0,    13,    14,     0,    15,    16,    17,    18,     0,    19,
      20,    21,    22,    23,     0,   112,    24,    25,    93,     0,
       0,     0,   113,   114,     0,     0,     0,    94,    95,    96,
       0,     0,     0,     0,    97,     0,    98,    99,   100,     0,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
       0,     0,     0,    93,   187,   188,   189,   190,   191,   192,
       0,     0,    94,    95,    96,     0,     0,     0,     0,    97,
     111,    98,    99,   100,     0,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,     0,   112,     0,    93,     0,
       0,   138,     0,   113,   114,     0,     0,    94,    95,    96,
       0,     0,     0,     0,    97,   111,    98,    99,   100,     0,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
       0,   112,     0,    93,     0,     0,   339,     0,   113,   114,
       0,     0,    94,    95,    96,     0,     0,     0,     0,   150,
     111,    98,    99,   100,     0,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,     0,   112,     0,    93,     0,
       0,     0,     0,   113,   114,     0,     0,    94,    95,    96,
       0,     0,     0,     0,   152,   111,    98,    99,   100,     0,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
       0,   112,     0,     0,     0,   124,     0,     0,   113,   114,
       0,   100,     3,     0,     0,     0,     0,     0,     0,     0,
     111,     0,     0,     4,     0,     0,     5,     0,     0,     0,
       6,     0,     7,     8,     0,     0,   112,     9,    10,     0,
      11,     0,    12,   113,   114,    13,    14,     0,    15,    16,
      17,    18,     3,    19,    20,    21,    22,    23,     0,     0,
      24,    25,     0,     4,     0,     0,     5,     0,     0,     0,
       6,     0,     7,     8,     0,     0,     0,     9,    10,     0,
      11,     0,    12,     0,     0,    13,    14,     0,    15,    16,
      17,    18,     3,    19,    20,    21,    22,    23,     0,     0,
      24,    25,     0,     4,     0,   129,     5,     0,     0,     0,
       6,     0,     7,     8,     0,     0,     0,     9,    10,     0,
      11,     0,    12,     0,     0,    13,    14,     0,    15,    16,
      17,    18,     0,    19,    20,    21,    22,    23,     0,     4,
      24,    25,     5,     0,     0,     0,     6,     0,     7,     8,
       0,     0,     0,     9,    10,     0,    11,     0,    12,     0,
       0,    13,     0,     0,    15,    16,     0,    18,     0,    19,
       0,    21,    22,    23,     0,     0,    24,    25,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     371,   169,   170,   171,   172,   173,   174,   175,   176,   177,
     178,   179,   180,   181,   182,   183,   184,   185,   186,   157,
     158,   159,   160,   161,   162,   163,   164,   165,   166,   167,
     168,     0,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   172,   173,   174,   175,
     176,   177,   178,   179,   180,   181,   182,   183,   184,   185,
     186,   173,   174,   175,   176,   177,   178,   179,   180,   181,
     182,   183,   184,   185,   186,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   176,   177,
     178,   179,   180,   181,   182,   183,   184,   185,   186
};

static const yytype_int16 yycheck[] =
{
       1,    27,    14,    91,   131,    17,    30,    58,    20,    33,
      92,   320,     4,   200,    97,   147,     4,     4,   199,    33,
      95,    47,    74,    95,   207,    53,    33,    17,    73,    53,
      37,    38,     4,    61,     4,    95,    60,    61,   221,    42,
      43,    67,   229,    67,    70,    17,    60,    92,   133,    77,
     135,    95,    76,    77,   384,    93,    94,    95,    96,     4,
      98,    99,    62,     3,     4,   395,     4,   150,     4,   152,
       4,    42,   381,    63,    64,   113,   114,    33,    78,   206,
      95,    37,    38,    93,   271,   168,    93,   132,     5,   316,
      91,     5,    95,   138,   275,     5,   405,    97,   281,   231,
     232,   127,    94,    94,   413,   188,    52,   294,    96,    96,
      53,   137,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,    93,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   224,   226,   189,   197,    36,   198,    93,    93,
     150,    36,   152,   280,   199,    93,     4,    93,    95,    93,
     387,    33,   207,   390,   391,    37,    38,     3,    33,    96,
      42,    43,    37,    38,   200,   187,   221,   228,   190,   224,
      33,   356,   409,   358,    37,    38,   361,    22,   415,    42,
      43,    42,    43,    37,    38,   283,    31,    32,    33,    42,
      43,    36,    37,    38,     4,    40,    41,    42,   220,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    37,
      38,    93,    87,    88,   276,    17,    91,    87,    88,     4,
     275,    91,    33,     5,    87,    88,   281,    38,    91,    74,
       4,    42,    43,    93,    95,   271,    93,   271,    30,    31,
       3,    33,    94,   298,    93,    90,   301,   295,     4,   285,
      95,    30,    97,    98,    33,     3,     4,     3,   313,    94,
     294,    53,   355,     3,     4,     5,   359,   360,    60,    61,
       3,    93,   283,    93,    53,    94,    87,    88,     3,     4,
      91,    60,    61,   381,   339,    77,    17,    33,    33,    34,
      35,     3,    38,   386,    93,   317,    42,    43,    77,   335,
      33,   356,    33,   358,   396,    38,   361,    38,    17,    42,
      43,    42,    43,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      33,     3,     1,    33,     3,    38,     3,    37,    38,    42,
      43,   396,    42,    43,   111,   112,    17,    38,   384,    38,
      38,    38,    38,    22,    31,    32,    33,    34,    35,   395,
      38,    65,    31,    32,    33,     3,    83,     4,     3,    38,
     381,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,     3,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    38,     5,    95,    96,    97,    98,
       1,    16,     3,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    22,    29,    30,    31,    32,    33,    34,    35,    38,
      31,    32,    33,    96,    93,    93,    93,    38,     5,    40,
      41,    42,     3,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,     3,    55,    56,     3,    58,    59,    60,
       3,     3,    62,     4,    65,    66,    67,     3,    93,    70,
     280,    72,     3,    74,    75,    70,   224,    78,   285,   206,
     339,   276,    83,   197,   331,   327,    31,   301,    89,    90,
     327,    22,    31,    -1,    95,    -1,    97,    98,    -1,    -1,
      31,    32,    33,    -1,    -1,    -1,    -1,    38,     0,    40,
      41,    42,    -1,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    -1,    55,    56,    -1,    58,    59,    60,
      -1,    -1,    -1,    -1,    65,    66,    67,    -1,    -1,    70,
      -1,    72,    -1,    74,    75,    -1,    -1,    78,    -1,    -1,
      -1,    43,    83,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    54,    -1,    -1,    57,    97,    98,    -1,    61,
      -1,    63,    64,    -1,    -1,    -1,    68,    69,    -1,    71,
      -1,    73,    -1,    -1,    76,    77,    -1,    79,    80,    81,
      82,    -1,    84,    85,    86,    87,    88,    22,    -1,    91,
      92,    -1,    -1,    -1,    -1,    -1,    31,    32,    33,    -1,
      -1,    -1,    -1,    38,    -1,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      -1,    -1,    57,    -1,    -1,    -1,    61,    -1,    63,    64,
      -1,    -1,    -1,    68,    69,    -1,    71,    -1,    73,    74,
      -1,    76,    77,    -1,    79,    80,    81,    82,    -1,    84,
      85,    86,    87,    88,    -1,    90,    91,    92,    22,    -1,
      -1,    -1,    97,    98,    -1,    -1,    -1,    31,    32,    33,
      -1,    -1,    -1,    -1,    38,    -1,    40,    41,    42,    -1,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      -1,    -1,    -1,    22,    36,    37,    38,    39,    40,    41,
      -1,    -1,    31,    32,    33,    -1,    -1,    -1,    -1,    38,
      74,    40,    41,    42,    -1,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    -1,    90,    -1,    22,    -1,
      -1,    95,    -1,    97,    98,    -1,    -1,    31,    32,    33,
      -1,    -1,    -1,    -1,    38,    74,    40,    41,    42,    -1,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      -1,    90,    -1,    22,    -1,    -1,    95,    -1,    97,    98,
      -1,    -1,    31,    32,    33,    -1,    -1,    -1,    -1,    38,
      74,    40,    41,    42,    -1,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    -1,    90,    -1,    22,    -1,
      -1,    -1,    -1,    97,    98,    -1,    -1,    31,    32,    33,
      -1,    -1,    -1,    -1,    38,    74,    40,    41,    42,    -1,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      -1,    90,    -1,    -1,    -1,    36,    -1,    -1,    97,    98,
      -1,    42,    43,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    -1,    -1,    54,    -1,    -1,    57,    -1,    -1,    -1,
      61,    -1,    63,    64,    -1,    -1,    90,    68,    69,    -1,
      71,    -1,    73,    97,    98,    76,    77,    -1,    79,    80,
      81,    82,    43,    84,    85,    86,    87,    88,    -1,    -1,
      91,    92,    -1,    54,    -1,    -1,    57,    -1,    -1,    -1,
      61,    -1,    63,    64,    -1,    -1,    -1,    68,    69,    -1,
      71,    -1,    73,    -1,    -1,    76,    77,    -1,    79,    80,
      81,    82,    43,    84,    85,    86,    87,    88,    -1,    -1,
      91,    92,    -1,    54,    -1,    96,    57,    -1,    -1,    -1,
      61,    -1,    63,    64,    -1,    -1,    -1,    68,    69,    -1,
      71,    -1,    73,    -1,    -1,    76,    77,    -1,    79,    80,
      81,    82,    -1,    84,    85,    86,    87,    88,    -1,    54,
      91,    92,    57,    -1,    -1,    -1,    61,    -1,    63,    64,
      -1,    -1,    -1,    68,    69,    -1,    71,    -1,    73,    -1,
      -1,    76,    -1,    -1,    79,    80,    -1,    82,    -1,    84,
      -1,    86,    87,    88,    -1,    -1,    91,    92,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    -1,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,   100,     0,    43,    54,    57,    61,    63,    64,    68,
      69,    71,    73,    76,    77,    79,    80,    81,    82,    84,
      85,    86,    87,    88,    91,    92,   101,   152,   153,   155,
     156,   165,   166,   168,   169,   170,    42,    43,    95,   150,
     173,   150,   173,    95,   173,     3,    33,    38,   104,   106,
     107,   172,   173,   162,   164,   168,   169,   170,   156,   166,
     168,   162,   151,   157,   158,   161,   159,   163,   106,     3,
       4,   102,   105,    37,    38,   164,   163,   162,   113,   153,
     154,   150,   150,    42,   167,    95,   106,   170,    93,   104,
     106,   111,     5,    22,    31,    32,    33,    38,    40,    41,
      42,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    74,    90,    97,    98,   139,   140,   142,   143,   144,
     145,   146,   147,   171,    36,   127,   128,   154,   171,    96,
     154,   114,     5,     4,    96,   160,   103,   155,    95,   123,
     142,   143,   143,   143,   143,   141,   142,   154,   143,   143,
      38,   144,    38,   144,   143,   143,    94,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    52,    53,    36,    93,     4,    33,    37,
      38,   106,   119,   120,   121,   122,   115,    17,   106,   116,
     117,   118,   172,   142,   167,   167,    95,   129,   106,   112,
      36,    37,   123,   124,   125,   126,     4,    93,    33,    38,
     119,   154,   154,   142,   142,   142,   142,   142,   142,   142,
     142,   142,   142,   142,   141,   142,   142,   142,   142,   142,
     142,   142,   142,   142,   142,   142,   142,   142,   142,   142,
     142,   142,   142,   173,   141,   142,   148,   149,   173,    36,
     128,   163,   139,    93,   120,    37,    38,   116,   140,     3,
       4,    17,    96,   130,     3,     4,   173,   140,     4,     5,
     123,   124,    96,   141,   163,    93,   119,   119,    17,    94,
      93,     4,   120,    94,    93,   139,   127,     3,   117,   140,
       1,    42,    55,    56,    58,    59,    60,    66,    67,    70,
      72,    75,    78,    83,    89,    96,   108,   131,   132,   133,
     135,   136,   137,   138,   141,   155,   112,    94,     4,    95,
     143,    93,    93,   142,   149,    94,    93,     3,    17,     3,
     142,     3,    17,   133,   173,    38,    38,   138,    38,    38,
      38,    38,   132,   135,   129,    65,     3,     3,   106,   109,
     126,    17,    83,     3,   141,   148,     3,   148,   141,   141,
     148,    38,   110,     3,     4,    96,    38,    93,    93,    93,
      93,    93,    93,   134,   138,   155,     5,   109,   141,   133,
       3,     3,   133,   133,     3,     3,   109,   123,    93,    62,
     138,     3,   133,     3,   138,    93,   133
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 4:
#line 109 "cc.y"
    {
		dodecl(xdecl, lastclass, lasttype, Z);
	}
    break;

  case 6:
#line 114 "cc.y"
    {
		lastdcl = T;
		firstarg = S;
		dodecl(xdecl, lastclass, lasttype, (yyvsp[(2) - (2)].node));
		if(lastdcl == T || lastdcl->etype != TFUNC) {
			diag((yyvsp[(2) - (2)].node), "not a function");
			lastdcl = types[TFUNC];
		}
		thisfn = lastdcl;
		markdcl();
		firstdcl = dclstack;
		argmark((yyvsp[(2) - (2)].node), 0);
	}
    break;

  case 7:
#line 128 "cc.y"
    {
		argmark((yyvsp[(2) - (4)].node), 1);
	}
    break;

  case 8:
#line 132 "cc.y"
    {
		Node *n;

		n = revertdcl();
		if(n)
			(yyvsp[(6) - (6)].node) = new(OLIST, n, (yyvsp[(6) - (6)].node));
		if(!debug['a'] && !debug['Z'])
			codgen((yyvsp[(6) - (6)].node), (yyvsp[(2) - (6)].node));
	}
    break;

  case 9:
#line 144 "cc.y"
    {
		dodecl(xdecl, lastclass, lasttype, (yyvsp[(1) - (1)].node));
	}
    break;

  case 10:
#line 148 "cc.y"
    {
		(yyvsp[(1) - (1)].node) = dodecl(xdecl, lastclass, lasttype, (yyvsp[(1) - (1)].node));
	}
    break;

  case 11:
#line 152 "cc.y"
    {
		doinit((yyvsp[(1) - (4)].node)->sym, (yyvsp[(1) - (4)].node)->type, 0L, (yyvsp[(4) - (4)].node));
	}
    break;

  case 14:
#line 160 "cc.y"
    {
		(yyval.node) = new(OIND, (yyvsp[(3) - (3)].node), Z);
		(yyval.node)->garb = simpleg((yyvsp[(2) - (3)].lval));
	}
    break;

  case 16:
#line 168 "cc.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
	}
    break;

  case 17:
#line 172 "cc.y"
    {
		(yyval.node) = new(OFUNC, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 18:
#line 176 "cc.y"
    {
		(yyval.node) = new(OARRAY, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 19:
#line 185 "cc.y"
    {
		(yyval.node) = dodecl(adecl, lastclass, lasttype, Z);
	}
    break;

  case 20:
#line 189 "cc.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
	}
    break;

  case 21:
#line 195 "cc.y"
    {
		dodecl(adecl, lastclass, lasttype, (yyvsp[(1) - (1)].node));
		(yyval.node) = Z;
	}
    break;

  case 22:
#line 200 "cc.y"
    {
		(yyvsp[(1) - (1)].node) = dodecl(adecl, lastclass, lasttype, (yyvsp[(1) - (1)].node));
	}
    break;

  case 23:
#line 204 "cc.y"
    {
		int32 w;

		w = (yyvsp[(1) - (4)].node)->sym->type->width;
		(yyval.node) = doinit((yyvsp[(1) - (4)].node)->sym, (yyvsp[(1) - (4)].node)->type, 0L, (yyvsp[(4) - (4)].node));
		(yyval.node) = contig((yyvsp[(1) - (4)].node)->sym, (yyval.node), w);
	}
    break;

  case 24:
#line 212 "cc.y"
    {
		(yyval.node) = (yyvsp[(1) - (3)].node);
		if((yyvsp[(3) - (3)].node) != Z) {
			(yyval.node) = (yyvsp[(3) - (3)].node);
			if((yyvsp[(1) - (3)].node) != Z)
				(yyval.node) = new(OLIST, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
		}
	}
    break;

  case 27:
#line 229 "cc.y"
    {
		dodecl(pdecl, lastclass, lasttype, (yyvsp[(1) - (1)].node));
	}
    break;

  case 29:
#line 239 "cc.y"
    {
		lasttype = (yyvsp[(1) - (1)].type);
	}
    break;

  case 31:
#line 244 "cc.y"
    {
		lasttype = (yyvsp[(2) - (2)].type);
	}
    break;

  case 33:
#line 250 "cc.y"
    {
		lastfield = 0;
		edecl(CXXX, lasttype, S);
	}
    break;

  case 35:
#line 258 "cc.y"
    {
		dodecl(edecl, CXXX, lasttype, (yyvsp[(1) - (1)].node));
	}
    break;

  case 37:
#line 265 "cc.y"
    {
		lastbit = 0;
		firstbit = 1;
	}
    break;

  case 38:
#line 270 "cc.y"
    {
		(yyval.node) = new(OBIT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 39:
#line 274 "cc.y"
    {
		(yyval.node) = new(OBIT, Z, (yyvsp[(2) - (2)].node));
	}
    break;

  case 40:
#line 282 "cc.y"
    {
		(yyval.node) = (Z);
	}
    break;

  case 42:
#line 289 "cc.y"
    {
		(yyval.node) = new(OIND, (Z), Z);
		(yyval.node)->garb = simpleg((yyvsp[(2) - (2)].lval));
	}
    break;

  case 43:
#line 294 "cc.y"
    {
		(yyval.node) = new(OIND, (yyvsp[(3) - (3)].node), Z);
		(yyval.node)->garb = simpleg((yyvsp[(2) - (3)].lval));
	}
    break;

  case 46:
#line 303 "cc.y"
    {
		(yyval.node) = new(OFUNC, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 47:
#line 307 "cc.y"
    {
		(yyval.node) = new(OARRAY, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 48:
#line 313 "cc.y"
    {
		(yyval.node) = new(OFUNC, (Z), Z);
	}
    break;

  case 49:
#line 317 "cc.y"
    {
		(yyval.node) = new(OARRAY, (Z), (yyvsp[(2) - (3)].node));
	}
    break;

  case 50:
#line 321 "cc.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
	}
    break;

  case 52:
#line 328 "cc.y"
    {
		(yyval.node) = new(OINIT, invert((yyvsp[(2) - (3)].node)), Z);
	}
    break;

  case 53:
#line 334 "cc.y"
    {
		(yyval.node) = new(OARRAY, (yyvsp[(2) - (3)].node), Z);
	}
    break;

  case 54:
#line 338 "cc.y"
    {
		(yyval.node) = new(OELEM, Z, Z);
		(yyval.node)->sym = (yyvsp[(2) - (2)].sym);
	}
    break;

  case 57:
#line 347 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (3)].node), (yyvsp[(2) - (3)].node));
	}
    break;

  case 59:
#line 352 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 62:
#line 360 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 63:
#line 365 "cc.y"
    {
		(yyval.node) = Z;
	}
    break;

  case 64:
#line 369 "cc.y"
    {
		(yyval.node) = invert((yyvsp[(1) - (1)].node));
	}
    break;

  case 66:
#line 377 "cc.y"
    {
		(yyval.node) = new(OPROTO, (yyvsp[(2) - (2)].node), Z);
		(yyval.node)->type = (yyvsp[(1) - (2)].type);
	}
    break;

  case 67:
#line 382 "cc.y"
    {
		(yyval.node) = new(OPROTO, (yyvsp[(2) - (2)].node), Z);
		(yyval.node)->type = (yyvsp[(1) - (2)].type);
	}
    break;

  case 68:
#line 387 "cc.y"
    {
		(yyval.node) = new(ODOTDOT, Z, Z);
	}
    break;

  case 69:
#line 391 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 70:
#line 397 "cc.y"
    {
		(yyval.node) = invert((yyvsp[(2) - (3)].node));
	//	if($2 != Z)
	//		$$ = new(OLIST, $2, $$);
		if((yyval.node) == Z)
			(yyval.node) = new(OLIST, Z, Z);
	}
    break;

  case 71:
#line 406 "cc.y"
    {
		(yyval.node) = Z;
	}
    break;

  case 72:
#line 410 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 73:
#line 414 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 75:
#line 421 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 76:
#line 427 "cc.y"
    {
		(yyval.node) = new(OCASE, (yyvsp[(2) - (3)].node), Z);
	}
    break;

  case 77:
#line 431 "cc.y"
    {
		(yyval.node) = new(OCASE, Z, Z);
	}
    break;

  case 78:
#line 435 "cc.y"
    {
		(yyval.node) = new(OLABEL, dcllabel((yyvsp[(1) - (2)].sym), 1), Z);
	}
    break;

  case 79:
#line 441 "cc.y"
    {
		(yyval.node) = Z;
	}
    break;

  case 81:
#line 446 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 83:
#line 453 "cc.y"
    {
		(yyval.node) = (yyvsp[(2) - (2)].node);
	}
    break;

  case 85:
#line 459 "cc.y"
    {
		markdcl();
	}
    break;

  case 86:
#line 463 "cc.y"
    {
		(yyval.node) = revertdcl();
		if((yyval.node))
			(yyval.node) = new(OLIST, (yyval.node), (yyvsp[(2) - (2)].node));
		else
			(yyval.node) = (yyvsp[(2) - (2)].node);
	}
    break;

  case 87:
#line 471 "cc.y"
    {
		(yyval.node) = new(OIF, (yyvsp[(3) - (5)].node), new(OLIST, (yyvsp[(5) - (5)].node), Z));
		if((yyvsp[(5) - (5)].node) == Z)
			warn((yyvsp[(3) - (5)].node), "empty if body");
	}
    break;

  case 88:
#line 477 "cc.y"
    {
		(yyval.node) = new(OIF, (yyvsp[(3) - (7)].node), new(OLIST, (yyvsp[(5) - (7)].node), (yyvsp[(7) - (7)].node)));
		if((yyvsp[(5) - (7)].node) == Z)
			warn((yyvsp[(3) - (7)].node), "empty if body");
		if((yyvsp[(7) - (7)].node) == Z)
			warn((yyvsp[(3) - (7)].node), "empty else body");
	}
    break;

  case 89:
#line 484 "cc.y"
    { markdcl(); }
    break;

  case 90:
#line 485 "cc.y"
    {
		(yyval.node) = revertdcl();
		if((yyval.node)){
			if((yyvsp[(4) - (10)].node))
				(yyvsp[(4) - (10)].node) = new(OLIST, (yyval.node), (yyvsp[(4) - (10)].node));
			else
				(yyvsp[(4) - (10)].node) = (yyval.node);
		}
		(yyval.node) = new(OFOR, new(OLIST, (yyvsp[(6) - (10)].node), new(OLIST, (yyvsp[(4) - (10)].node), (yyvsp[(8) - (10)].node))), (yyvsp[(10) - (10)].node));
	}
    break;

  case 91:
#line 496 "cc.y"
    {
		(yyval.node) = new(OWHILE, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node));
	}
    break;

  case 92:
#line 500 "cc.y"
    {
		(yyval.node) = new(ODWHILE, (yyvsp[(5) - (7)].node), (yyvsp[(2) - (7)].node));
	}
    break;

  case 93:
#line 504 "cc.y"
    {
		(yyval.node) = new(ORETURN, (yyvsp[(2) - (3)].node), Z);
		(yyval.node)->type = thisfn->link;
	}
    break;

  case 94:
#line 509 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->vconst = 0;
		(yyval.node)->type = types[TINT];
		(yyvsp[(3) - (5)].node) = new(OSUB, (yyval.node), (yyvsp[(3) - (5)].node));

		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->vconst = 0;
		(yyval.node)->type = types[TINT];
		(yyvsp[(3) - (5)].node) = new(OSUB, (yyval.node), (yyvsp[(3) - (5)].node));

		(yyval.node) = new(OSWITCH, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node));
	}
    break;

  case 95:
#line 523 "cc.y"
    {
		(yyval.node) = new(OBREAK, Z, Z);
	}
    break;

  case 96:
#line 527 "cc.y"
    {
		(yyval.node) = new(OCONTINUE, Z, Z);
	}
    break;

  case 97:
#line 531 "cc.y"
    {
		(yyval.node) = new(OGOTO, dcllabel((yyvsp[(2) - (3)].sym), 0), Z);
	}
    break;

  case 98:
#line 535 "cc.y"
    {
		(yyval.node) = new(OUSED, (yyvsp[(3) - (5)].node), Z);
	}
    break;

  case 99:
#line 539 "cc.y"
    {
		(yyval.node) = new(OPREFETCH, (yyvsp[(3) - (5)].node), Z);
	}
    break;

  case 100:
#line 543 "cc.y"
    {
		(yyval.node) = new(OSET, (yyvsp[(3) - (5)].node), Z);
	}
    break;

  case 101:
#line 548 "cc.y"
    {
		(yyval.node) = Z;
	}
    break;

  case 103:
#line 554 "cc.y"
    {
		(yyval.node) = Z;
	}
    break;

  case 105:
#line 561 "cc.y"
    {
		(yyval.node) = new(OCAST, (yyvsp[(1) - (1)].node), Z);
		(yyval.node)->type = types[TLONG];
	}
    break;

  case 107:
#line 569 "cc.y"
    {
		(yyval.node) = new(OCOMMA, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 109:
#line 576 "cc.y"
    {
		(yyval.node) = new(OMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 110:
#line 580 "cc.y"
    {
		(yyval.node) = new(ODIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 111:
#line 584 "cc.y"
    {
		(yyval.node) = new(OMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 112:
#line 588 "cc.y"
    {
		(yyval.node) = new(OADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 113:
#line 592 "cc.y"
    {
		(yyval.node) = new(OSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 114:
#line 596 "cc.y"
    {
		(yyval.node) = new(OASHR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 115:
#line 600 "cc.y"
    {
		(yyval.node) = new(OASHL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 116:
#line 604 "cc.y"
    {
		(yyval.node) = new(OLT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 117:
#line 608 "cc.y"
    {
		(yyval.node) = new(OGT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 118:
#line 612 "cc.y"
    {
		(yyval.node) = new(OLE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 119:
#line 616 "cc.y"
    {
		(yyval.node) = new(OGE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 120:
#line 620 "cc.y"
    {
		(yyval.node) = new(OEQ, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 121:
#line 624 "cc.y"
    {
		(yyval.node) = new(ONE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 122:
#line 628 "cc.y"
    {
		(yyval.node) = new(OAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 123:
#line 632 "cc.y"
    {
		(yyval.node) = new(OXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 124:
#line 636 "cc.y"
    {
		(yyval.node) = new(OOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 125:
#line 640 "cc.y"
    {
		(yyval.node) = new(OANDAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 126:
#line 644 "cc.y"
    {
		(yyval.node) = new(OOROR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 127:
#line 648 "cc.y"
    {
		(yyval.node) = new(OCOND, (yyvsp[(1) - (5)].node), new(OLIST, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node)));
	}
    break;

  case 128:
#line 652 "cc.y"
    {
		(yyval.node) = new(OAS, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 129:
#line 656 "cc.y"
    {
		(yyval.node) = new(OASADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 130:
#line 660 "cc.y"
    {
		(yyval.node) = new(OASSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 131:
#line 664 "cc.y"
    {
		(yyval.node) = new(OASMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 132:
#line 668 "cc.y"
    {
		(yyval.node) = new(OASDIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 133:
#line 672 "cc.y"
    {
		(yyval.node) = new(OASMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 134:
#line 676 "cc.y"
    {
		(yyval.node) = new(OASASHL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 135:
#line 680 "cc.y"
    {
		(yyval.node) = new(OASASHR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 136:
#line 684 "cc.y"
    {
		(yyval.node) = new(OASAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 137:
#line 688 "cc.y"
    {
		(yyval.node) = new(OASXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 138:
#line 692 "cc.y"
    {
		(yyval.node) = new(OASOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 140:
#line 699 "cc.y"
    {
		(yyval.node) = new(OCAST, (yyvsp[(5) - (5)].node), Z);
		dodecl(NODECL, CXXX, (yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].node));
		(yyval.node)->type = lastdcl;
		(yyval.node)->xcast = 1;
	}
    break;

  case 141:
#line 706 "cc.y"
    {
		(yyval.node) = new(OSTRUCT, (yyvsp[(6) - (7)].node), Z);
		dodecl(NODECL, CXXX, (yyvsp[(2) - (7)].type), (yyvsp[(3) - (7)].node));
		(yyval.node)->type = lastdcl;
	}
    break;

  case 143:
#line 715 "cc.y"
    {
		(yyval.node) = new(OIND, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 144:
#line 719 "cc.y"
    {
		(yyval.node) = new(OADDR, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 145:
#line 723 "cc.y"
    {
		(yyval.node) = new(OPOS, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 146:
#line 727 "cc.y"
    {
		(yyval.node) = new(ONEG, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 147:
#line 731 "cc.y"
    {
		(yyval.node) = new(ONOT, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 148:
#line 735 "cc.y"
    {
		(yyval.node) = new(OCOM, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 149:
#line 739 "cc.y"
    {
		(yyval.node) = new(OPREINC, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 150:
#line 743 "cc.y"
    {
		(yyval.node) = new(OPREDEC, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 151:
#line 747 "cc.y"
    {
		(yyval.node) = new(OSIZE, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 152:
#line 751 "cc.y"
    {
		(yyval.node) = new(OSIGN, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 153:
#line 757 "cc.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
	}
    break;

  case 154:
#line 761 "cc.y"
    {
		(yyval.node) = new(OSIZE, Z, Z);
		dodecl(NODECL, CXXX, (yyvsp[(3) - (5)].type), (yyvsp[(4) - (5)].node));
		(yyval.node)->type = lastdcl;
	}
    break;

  case 155:
#line 767 "cc.y"
    {
		(yyval.node) = new(OSIGN, Z, Z);
		dodecl(NODECL, CXXX, (yyvsp[(3) - (5)].type), (yyvsp[(4) - (5)].node));
		(yyval.node)->type = lastdcl;
	}
    break;

  case 156:
#line 773 "cc.y"
    {
		(yyval.node) = new(OFUNC, (yyvsp[(1) - (4)].node), Z);
		if((yyvsp[(1) - (4)].node)->op == ONAME)
		if((yyvsp[(1) - (4)].node)->type == T)
			dodecl(xdecl, CXXX, types[TINT], (yyval.node));
		(yyval.node)->right = invert((yyvsp[(3) - (4)].node));
	}
    break;

  case 157:
#line 781 "cc.y"
    {
		(yyval.node) = new(OIND, new(OADD, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node)), Z);
	}
    break;

  case 158:
#line 785 "cc.y"
    {
		(yyval.node) = new(ODOT, new(OIND, (yyvsp[(1) - (3)].node), Z), Z);
		(yyval.node)->sym = (yyvsp[(3) - (3)].sym);
	}
    break;

  case 159:
#line 790 "cc.y"
    {
		(yyval.node) = new(ODOT, (yyvsp[(1) - (3)].node), Z);
		(yyval.node)->sym = (yyvsp[(3) - (3)].sym);
	}
    break;

  case 160:
#line 795 "cc.y"
    {
		(yyval.node) = new(OPOSTINC, (yyvsp[(1) - (2)].node), Z);
	}
    break;

  case 161:
#line 799 "cc.y"
    {
		(yyval.node) = new(OPOSTDEC, (yyvsp[(1) - (2)].node), Z);
	}
    break;

  case 163:
#line 804 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TINT];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 164:
#line 811 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TLONG];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 165:
#line 818 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TUINT];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 166:
#line 825 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TULONG];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 167:
#line 832 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TDOUBLE];
		(yyval.node)->fconst = (yyvsp[(1) - (1)].dval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 168:
#line 839 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TFLOAT];
		(yyval.node)->fconst = (yyvsp[(1) - (1)].dval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 169:
#line 846 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TVLONG];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 170:
#line 853 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TUVLONG];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 173:
#line 864 "cc.y"
    {
		(yyval.node) = new(OSTRING, Z, Z);
		(yyval.node)->type = typ(TARRAY, types[TCHAR]);
		(yyval.node)->type->width = (yyvsp[(1) - (1)].sval).l + 1;
		(yyval.node)->cstring = (yyvsp[(1) - (1)].sval).s;
		(yyval.node)->sym = symstring;
		(yyval.node)->etype = TARRAY;
		(yyval.node)->class = CSTATIC;
	}
    break;

  case 174:
#line 874 "cc.y"
    {
		char *s;
		int n;

		n = (yyvsp[(1) - (2)].node)->type->width - 1;
		s = alloc(n+(yyvsp[(2) - (2)].sval).l+MAXALIGN);

		memcpy(s, (yyvsp[(1) - (2)].node)->cstring, n);
		memcpy(s+n, (yyvsp[(2) - (2)].sval).s, (yyvsp[(2) - (2)].sval).l);
		s[n+(yyvsp[(2) - (2)].sval).l] = 0;

		(yyval.node) = (yyvsp[(1) - (2)].node);
		(yyval.node)->type->width += (yyvsp[(2) - (2)].sval).l;
		(yyval.node)->cstring = s;
	}
    break;

  case 175:
#line 892 "cc.y"
    {
		(yyval.node) = new(OLSTRING, Z, Z);
		(yyval.node)->type = typ(TARRAY, types[TUSHORT]);
		(yyval.node)->type->width = (yyvsp[(1) - (1)].sval).l + sizeof(ushort);
		(yyval.node)->rstring = (ushort*)(yyvsp[(1) - (1)].sval).s;
		(yyval.node)->sym = symstring;
		(yyval.node)->etype = TARRAY;
		(yyval.node)->class = CSTATIC;
	}
    break;

  case 176:
#line 902 "cc.y"
    {
		char *s;
		int n;

		n = (yyvsp[(1) - (2)].node)->type->width - sizeof(ushort);
		s = alloc(n+(yyvsp[(2) - (2)].sval).l+MAXALIGN);

		memcpy(s, (yyvsp[(1) - (2)].node)->rstring, n);
		memcpy(s+n, (yyvsp[(2) - (2)].sval).s, (yyvsp[(2) - (2)].sval).l);
		*(ushort*)(s+n+(yyvsp[(2) - (2)].sval).l) = 0;

		(yyval.node) = (yyvsp[(1) - (2)].node);
		(yyval.node)->type->width += (yyvsp[(2) - (2)].sval).l;
		(yyval.node)->rstring = (ushort*)s;
	}
    break;

  case 177:
#line 919 "cc.y"
    {
		(yyval.node) = Z;
	}
    break;

  case 180:
#line 927 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 181:
#line 933 "cc.y"
    {
		(yyval.tyty).t1 = strf;
		(yyval.tyty).t2 = strl;
		(yyval.tyty).t3 = lasttype;
		(yyval.tyty).c = lastclass;
		strf = T;
		strl = T;
		lastbit = 0;
		firstbit = 1;
		lastclass = CXXX;
		lasttype = T;
	}
    break;

  case 182:
#line 946 "cc.y"
    {
		(yyval.type) = strf;
		strf = (yyvsp[(2) - (4)].tyty).t1;
		strl = (yyvsp[(2) - (4)].tyty).t2;
		lasttype = (yyvsp[(2) - (4)].tyty).t3;
		lastclass = (yyvsp[(2) - (4)].tyty).c;
	}
    break;

  case 183:
#line 955 "cc.y"
    {
		lastclass = CXXX;
		lasttype = types[TINT];
	}
    break;

  case 185:
#line 963 "cc.y"
    {
		(yyval.tycl).t = (yyvsp[(1) - (1)].type);
		(yyval.tycl).c = CXXX;
	}
    break;

  case 186:
#line 968 "cc.y"
    {
		(yyval.tycl).t = simplet((yyvsp[(1) - (1)].lval));
		(yyval.tycl).c = CXXX;
	}
    break;

  case 187:
#line 973 "cc.y"
    {
		(yyval.tycl).t = simplet((yyvsp[(1) - (1)].lval));
		(yyval.tycl).c = simplec((yyvsp[(1) - (1)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(1) - (1)].lval));
	}
    break;

  case 188:
#line 979 "cc.y"
    {
		(yyval.tycl).t = (yyvsp[(1) - (2)].type);
		(yyval.tycl).c = simplec((yyvsp[(2) - (2)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(2) - (2)].lval));
		if((yyvsp[(2) - (2)].lval) & ~BCLASS & ~BGARB)
			diag(Z, "duplicate types given: %T and %Q", (yyvsp[(1) - (2)].type), (yyvsp[(2) - (2)].lval));
	}
    break;

  case 189:
#line 987 "cc.y"
    {
		(yyval.tycl).t = simplet(typebitor((yyvsp[(1) - (2)].lval), (yyvsp[(2) - (2)].lval)));
		(yyval.tycl).c = simplec((yyvsp[(2) - (2)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(2) - (2)].lval));
	}
    break;

  case 190:
#line 993 "cc.y"
    {
		(yyval.tycl).t = (yyvsp[(2) - (3)].type);
		(yyval.tycl).c = simplec((yyvsp[(1) - (3)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(1) - (3)].lval)|(yyvsp[(3) - (3)].lval));
	}
    break;

  case 191:
#line 999 "cc.y"
    {
		(yyval.tycl).t = simplet((yyvsp[(2) - (2)].lval));
		(yyval.tycl).c = simplec((yyvsp[(1) - (2)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(1) - (2)].lval));
	}
    break;

  case 192:
#line 1005 "cc.y"
    {
		(yyval.tycl).t = simplet(typebitor((yyvsp[(2) - (3)].lval), (yyvsp[(3) - (3)].lval)));
		(yyval.tycl).c = simplec((yyvsp[(1) - (3)].lval)|(yyvsp[(3) - (3)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(1) - (3)].lval)|(yyvsp[(3) - (3)].lval));
	}
    break;

  case 193:
#line 1013 "cc.y"
    {
		(yyval.type) = (yyvsp[(1) - (1)].tycl).t;
		if((yyvsp[(1) - (1)].tycl).c != CXXX)
			diag(Z, "illegal combination of class 4: %s", cnames[(yyvsp[(1) - (1)].tycl).c]);
	}
    break;

  case 194:
#line 1021 "cc.y"
    {
		lasttype = (yyvsp[(1) - (1)].tycl).t;
		lastclass = (yyvsp[(1) - (1)].tycl).c;
	}
    break;

  case 195:
#line 1028 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TSTRUCT, 0);
		(yyval.type) = (yyvsp[(2) - (2)].sym)->suetag;
	}
    break;

  case 196:
#line 1033 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TSTRUCT, autobn);
	}
    break;

  case 197:
#line 1037 "cc.y"
    {
		(yyval.type) = (yyvsp[(2) - (4)].sym)->suetag;
		if((yyval.type)->link != T)
			diag(Z, "redeclare tag: %s", (yyvsp[(2) - (4)].sym)->name);
		(yyval.type)->link = (yyvsp[(4) - (4)].type);
		sualign((yyval.type));
	}
    break;

  case 198:
#line 1045 "cc.y"
    {
		taggen++;
		sprint(symb, "_%d_", taggen);
		(yyval.type) = dotag(lookup(), TSTRUCT, autobn);
		(yyval.type)->link = (yyvsp[(2) - (2)].type);
		sualign((yyval.type));
	}
    break;

  case 199:
#line 1053 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TUNION, 0);
		(yyval.type) = (yyvsp[(2) - (2)].sym)->suetag;
	}
    break;

  case 200:
#line 1058 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TUNION, autobn);
	}
    break;

  case 201:
#line 1062 "cc.y"
    {
		(yyval.type) = (yyvsp[(2) - (4)].sym)->suetag;
		if((yyval.type)->link != T)
			diag(Z, "redeclare tag: %s", (yyvsp[(2) - (4)].sym)->name);
		(yyval.type)->link = (yyvsp[(4) - (4)].type);
		sualign((yyval.type));
	}
    break;

  case 202:
#line 1070 "cc.y"
    {
		taggen++;
		sprint(symb, "_%d_", taggen);
		(yyval.type) = dotag(lookup(), TUNION, autobn);
		(yyval.type)->link = (yyvsp[(2) - (2)].type);
		sualign((yyval.type));
	}
    break;

  case 203:
#line 1078 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TENUM, 0);
		(yyval.type) = (yyvsp[(2) - (2)].sym)->suetag;
		if((yyval.type)->link == T)
			(yyval.type)->link = types[TINT];
		(yyval.type) = (yyval.type)->link;
	}
    break;

  case 204:
#line 1086 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TENUM, autobn);
	}
    break;

  case 205:
#line 1090 "cc.y"
    {
		en.tenum = T;
		en.cenum = T;
	}
    break;

  case 206:
#line 1095 "cc.y"
    {
		(yyval.type) = (yyvsp[(2) - (7)].sym)->suetag;
		if((yyval.type)->link != T)
			diag(Z, "redeclare tag: %s", (yyvsp[(2) - (7)].sym)->name);
		if(en.tenum == T) {
			diag(Z, "enum type ambiguous: %s", (yyvsp[(2) - (7)].sym)->name);
			en.tenum = types[TINT];
		}
		(yyval.type)->link = en.tenum;
		(yyval.type) = en.tenum;
	}
    break;

  case 207:
#line 1107 "cc.y"
    {
		en.tenum = T;
		en.cenum = T;
	}
    break;

  case 208:
#line 1112 "cc.y"
    {
		(yyval.type) = en.tenum;
	}
    break;

  case 209:
#line 1116 "cc.y"
    {
		(yyval.type) = tcopy((yyvsp[(1) - (1)].sym)->type);
	}
    break;

  case 211:
#line 1123 "cc.y"
    {
		(yyval.lval) = typebitor((yyvsp[(1) - (2)].lval), (yyvsp[(2) - (2)].lval));
	}
    break;

  case 212:
#line 1128 "cc.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 213:
#line 1132 "cc.y"
    {
		(yyval.lval) = typebitor((yyvsp[(1) - (2)].lval), (yyvsp[(2) - (2)].lval));
	}
    break;

  case 218:
#line 1144 "cc.y"
    {
		(yyval.lval) = typebitor((yyvsp[(1) - (2)].lval), (yyvsp[(2) - (2)].lval));
	}
    break;

  case 221:
#line 1154 "cc.y"
    {
		doenum((yyvsp[(1) - (1)].sym), Z);
	}
    break;

  case 222:
#line 1158 "cc.y"
    {
		doenum((yyvsp[(1) - (3)].sym), (yyvsp[(3) - (3)].node));
	}
    break;

  case 225:
#line 1165 "cc.y"
    { (yyval.lval) = BCHAR; }
    break;

  case 226:
#line 1166 "cc.y"
    { (yyval.lval) = BSHORT; }
    break;

  case 227:
#line 1167 "cc.y"
    { (yyval.lval) = BINT; }
    break;

  case 228:
#line 1168 "cc.y"
    { (yyval.lval) = BLONG; }
    break;

  case 229:
#line 1169 "cc.y"
    { (yyval.lval) = BSIGNED; }
    break;

  case 230:
#line 1170 "cc.y"
    { (yyval.lval) = BUNSIGNED; }
    break;

  case 231:
#line 1171 "cc.y"
    { (yyval.lval) = BFLOAT; }
    break;

  case 232:
#line 1172 "cc.y"
    { (yyval.lval) = BDOUBLE; }
    break;

  case 233:
#line 1173 "cc.y"
    { (yyval.lval) = BVOID; }
    break;

  case 234:
#line 1176 "cc.y"
    { (yyval.lval) = BAUTO; }
    break;

  case 235:
#line 1177 "cc.y"
    { (yyval.lval) = BSTATIC; }
    break;

  case 236:
#line 1178 "cc.y"
    { (yyval.lval) = BEXTERN; }
    break;

  case 237:
#line 1179 "cc.y"
    { (yyval.lval) = BTYPEDEF; }
    break;

  case 238:
#line 1180 "cc.y"
    { (yyval.lval) = BTYPESTR; }
    break;

  case 239:
#line 1181 "cc.y"
    { (yyval.lval) = BREGISTER; }
    break;

  case 240:
#line 1182 "cc.y"
    { (yyval.lval) = 0; }
    break;

  case 241:
#line 1185 "cc.y"
    { (yyval.lval) = BCONSTNT; }
    break;

  case 242:
#line 1186 "cc.y"
    { (yyval.lval) = BVOLATILE; }
    break;

  case 243:
#line 1187 "cc.y"
    { (yyval.lval) = 0; }
    break;

  case 244:
#line 1191 "cc.y"
    {
		(yyval.node) = new(ONAME, Z, Z);
		if((yyvsp[(1) - (1)].sym)->class == CLOCAL)
			(yyvsp[(1) - (1)].sym) = mkstatic((yyvsp[(1) - (1)].sym));
		(yyval.node)->sym = (yyvsp[(1) - (1)].sym);
		(yyval.node)->type = (yyvsp[(1) - (1)].sym)->type;
		(yyval.node)->etype = TVOID;
		if((yyval.node)->type != T)
			(yyval.node)->etype = (yyval.node)->type->etype;
		(yyval.node)->xoffset = (yyvsp[(1) - (1)].sym)->offset;
		(yyval.node)->class = (yyvsp[(1) - (1)].sym)->class;
		(yyvsp[(1) - (1)].sym)->aused = 1;
	}
    break;

  case 245:
#line 1206 "cc.y"
    {
		(yyval.node) = new(ONAME, Z, Z);
		(yyval.node)->sym = (yyvsp[(1) - (1)].sym);
		(yyval.node)->type = (yyvsp[(1) - (1)].sym)->type;
		(yyval.node)->etype = TVOID;
		if((yyval.node)->type != T)
			(yyval.node)->etype = (yyval.node)->type->etype;
		(yyval.node)->xoffset = (yyvsp[(1) - (1)].sym)->offset;
		(yyval.node)->class = (yyvsp[(1) - (1)].sym)->class;
	}
    break;


/* Line 1267 of yacc.c.  */
#line 3606 "y.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 1219 "cc.y"


