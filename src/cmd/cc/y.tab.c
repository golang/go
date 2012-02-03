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
#line 274 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 287 "y.tab.c"

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
#define YYNTOKENS  98
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  75
/* YYNRULES -- Number of rules.  */
#define YYNRULES  246
/* YYNRULES -- Number of states.  */
#define YYNSTATES  412

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   328

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    96,     2,     2,     2,    35,    22,     2,
      38,    92,    33,    31,     4,    32,    36,    34,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    17,     3,
      25,     5,    26,    16,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    37,     2,    93,    21,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    94,    20,    95,    97,     2,     2,     2,
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
      83,    84,    85,    86,    87,    88,    89,    90,    91
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
     326,   327,   329,   330,   332,   334,   336,   340,   342,   346,
     350,   354,   358,   362,   366,   370,   374,   378,   382,   386,
     390,   394,   398,   402,   406,   410,   414,   420,   424,   428,
     432,   436,   440,   444,   448,   452,   456,   460,   464,   466,
     472,   480,   482,   485,   488,   491,   494,   497,   500,   503,
     506,   509,   512,   516,   522,   528,   533,   538,   542,   546,
     549,   552,   554,   556,   558,   560,   562,   564,   566,   568,
     570,   572,   574,   576,   579,   581,   584,   585,   587,   589,
     593,   594,   599,   600,   602,   604,   606,   608,   611,   614,
     618,   621,   625,   627,   629,   632,   633,   638,   641,   644,
     645,   650,   653,   656,   657,   658,   666,   667,   673,   675,
     677,   680,   681,   684,   686,   688,   690,   692,   695,   697,
     699,   701,   705,   708,   712,   714,   716,   718,   720,   722,
     724,   726,   728,   730,   732,   734,   736,   738,   740,   742,
     744,   746,   748,   750,   752,   754,   756
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      99,     0,    -1,    -1,    99,   100,    -1,   151,     3,    -1,
     151,   103,     3,    -1,    -1,    -1,   151,   105,   101,   110,
     102,   128,    -1,   105,    -1,    -1,   105,   104,     5,   122,
      -1,   103,     4,   103,    -1,   106,    -1,    33,   162,   105,
      -1,   171,    -1,    38,   105,    92,    -1,   106,    38,   126,
      92,    -1,   106,    37,   138,    93,    -1,   154,     3,    -1,
     154,   108,     3,    -1,   105,    -1,    -1,   105,   109,     5,
     122,    -1,   108,     4,   108,    -1,    -1,   110,   154,   111,
       3,    -1,   105,    -1,   111,     4,   111,    -1,    -1,   153,
     113,   115,     3,    -1,    -1,   112,   153,   114,   115,     3,
      -1,    -1,   116,    -1,   117,    -1,   116,     4,   116,    -1,
     105,    -1,   171,    17,   139,    -1,    17,   139,    -1,    -1,
     119,    -1,    33,   162,    -1,    33,   162,   119,    -1,   120,
      -1,   121,    -1,   120,    38,   126,    92,    -1,   120,    37,
     138,    93,    -1,    38,    92,    -1,    37,   138,    93,    -1,
      38,   119,    92,    -1,   141,    -1,    94,   125,    95,    -1,
      37,   139,    93,    -1,    36,   172,    -1,   123,     5,    -1,
     122,     4,    -1,   124,   122,     4,    -1,   123,    -1,   124,
     123,    -1,   124,    -1,   122,    -1,   124,   122,    -1,    -1,
     127,    -1,   170,    -1,   153,   118,    -1,   153,   105,    -1,
      36,    36,    36,    -1,   127,     4,   127,    -1,    94,   129,
      95,    -1,    -1,   129,   107,    -1,   129,   132,    -1,   131,
      -1,   130,   131,    -1,    56,   141,    17,    -1,    59,    17,
      -1,    42,    17,    -1,     1,     3,    -1,   134,    -1,   130,
     134,    -1,   137,    -1,   154,   108,    -1,   137,     3,    -1,
      -1,   135,   128,    -1,    67,    38,   140,    92,   132,    -1,
      67,    38,   140,    92,   132,    62,   132,    -1,    -1,   136,
      65,    38,   133,     3,   137,     3,   137,    92,   132,    -1,
      82,    38,   140,    92,   132,    -1,    60,   132,    82,    38,
     140,    92,     3,    -1,    71,   137,     3,    -1,    77,    38,
     140,    92,   132,    -1,    55,     3,    -1,    58,     3,    -1,
      66,   172,     3,    -1,    74,    38,   147,    92,     3,    -1,
      88,    38,   147,    92,     3,    -1,    -1,   140,    -1,    -1,
     139,    -1,   141,    -1,   141,    -1,   140,     4,   140,    -1,
     142,    -1,   141,    33,   141,    -1,   141,    34,   141,    -1,
     141,    35,   141,    -1,   141,    31,   141,    -1,   141,    32,
     141,    -1,   141,    29,   141,    -1,   141,    30,   141,    -1,
     141,    25,   141,    -1,   141,    26,   141,    -1,   141,    28,
     141,    -1,   141,    27,   141,    -1,   141,    24,   141,    -1,
     141,    23,   141,    -1,   141,    22,   141,    -1,   141,    21,
     141,    -1,   141,    20,   141,    -1,   141,    19,   141,    -1,
     141,    18,   141,    -1,   141,    16,   140,    17,   141,    -1,
     141,     5,   141,    -1,   141,    15,   141,    -1,   141,    14,
     141,    -1,   141,    13,   141,    -1,   141,    12,   141,    -1,
     141,    11,   141,    -1,   141,     9,   141,    -1,   141,    10,
     141,    -1,   141,     8,   141,    -1,   141,     7,   141,    -1,
     141,     6,   141,    -1,   143,    -1,    38,   153,   118,    92,
     142,    -1,    38,   153,   118,    92,    94,   125,    95,    -1,
     144,    -1,    33,   142,    -1,    22,   142,    -1,    31,   142,
      -1,    32,   142,    -1,    96,   142,    -1,    97,   142,    -1,
      40,   142,    -1,    41,   142,    -1,    73,   143,    -1,    89,
     143,    -1,    38,   140,    92,    -1,    73,    38,   153,   118,
      92,    -1,    89,    38,   153,   118,    92,    -1,   144,    38,
     147,    92,    -1,   144,    37,   140,    93,    -1,   144,    39,
     172,    -1,   144,    36,   172,    -1,   144,    40,    -1,   144,
      41,    -1,   170,    -1,    46,    -1,    47,    -1,    48,    -1,
      49,    -1,    45,    -1,    44,    -1,    50,    -1,    51,    -1,
     145,    -1,   146,    -1,    52,    -1,   145,    52,    -1,    53,
      -1,   146,    53,    -1,    -1,   148,    -1,   141,    -1,   148,
       4,   148,    -1,    -1,    94,   150,   112,    95,    -1,    -1,
     154,    -1,   155,    -1,   167,    -1,   164,    -1,   155,   161,
      -1,   167,   161,    -1,   164,   155,   162,    -1,   164,   167,
      -1,   164,   167,   161,    -1,   152,    -1,   152,    -1,    76,
     172,    -1,    -1,    76,   172,   156,   149,    -1,    76,   149,
      -1,    80,   172,    -1,    -1,    80,   172,   157,   149,    -1,
      80,   149,    -1,    84,   172,    -1,    -1,    -1,    84,   172,
     158,    94,   159,   166,    95,    -1,    -1,    84,    94,   160,
     166,    95,    -1,    43,    -1,   163,    -1,   161,   163,    -1,
      -1,   162,   169,    -1,   167,    -1,   169,    -1,   168,    -1,
     165,    -1,   164,   165,    -1,   169,    -1,   168,    -1,    42,
      -1,    42,     5,   141,    -1,   166,     4,    -1,   166,     4,
     166,    -1,    57,    -1,    72,    -1,    68,    -1,    69,    -1,
      85,    -1,    81,    -1,    64,    -1,    61,    -1,    83,    -1,
      54,    -1,    75,    -1,    63,    -1,    78,    -1,    79,    -1,
      70,    -1,    91,    -1,    86,    -1,    87,    -1,    90,    -1,
      42,    -1,   172,    -1,    42,    -1,    43,    -1
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
     544,   547,   550,   553,   556,   563,   564,   570,   571,   575,
     579,   583,   587,   591,   595,   599,   603,   607,   611,   615,
     619,   623,   627,   631,   635,   639,   643,   647,   651,   655,
     659,   663,   667,   671,   675,   679,   683,   687,   693,   694,
     701,   709,   710,   714,   718,   722,   726,   730,   734,   738,
     742,   746,   752,   756,   762,   768,   776,   780,   785,   790,
     794,   798,   799,   806,   813,   820,   827,   834,   841,   848,
     855,   856,   859,   869,   887,   897,   915,   918,   921,   922,
     929,   928,   951,   955,   958,   963,   968,   974,   982,   988,
     994,  1000,  1008,  1016,  1023,  1029,  1028,  1040,  1048,  1054,
    1053,  1065,  1073,  1082,  1086,  1081,  1103,  1102,  1111,  1117,
    1118,  1124,  1127,  1133,  1134,  1135,  1138,  1139,  1145,  1146,
    1149,  1153,  1157,  1158,  1161,  1162,  1163,  1164,  1165,  1166,
    1167,  1168,  1169,  1172,  1173,  1174,  1175,  1176,  1177,  1178,
    1181,  1182,  1183,  1186,  1201,  1213,  1214
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
  "LFOR", "LGOTO", "LIF", "LINT", "LLONG", "LREGISTER", "LRETURN",
  "LSHORT", "LSIZEOF", "LUSED", "LSTATIC", "LSTRUCT", "LSWITCH",
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
     327,   328,    41,    93,   123,   125,    33,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    98,    99,    99,   100,   100,   101,   102,   100,   103,
     104,   103,   103,   105,   105,   106,   106,   106,   106,   107,
     107,   108,   109,   108,   108,   110,   110,   111,   111,   113,
     112,   114,   112,   115,   115,   116,   116,   117,   117,   117,
     118,   118,   119,   119,   119,   120,   120,   120,   121,   121,
     121,   122,   122,   123,   123,   123,   124,   124,   124,   124,
     125,   125,   125,   126,   126,   127,   127,   127,   127,   127,
     128,   129,   129,   129,   130,   130,   131,   131,   131,   132,
     132,   132,   133,   133,   134,   135,   134,   134,   134,   136,
     134,   134,   134,   134,   134,   134,   134,   134,   134,   134,
     137,   137,   138,   138,   139,   140,   140,   141,   141,   141,
     141,   141,   141,   141,   141,   141,   141,   141,   141,   141,
     141,   141,   141,   141,   141,   141,   141,   141,   141,   141,
     141,   141,   141,   141,   141,   141,   141,   141,   142,   142,
     142,   143,   143,   143,   143,   143,   143,   143,   143,   143,
     143,   143,   144,   144,   144,   144,   144,   144,   144,   144,
     144,   144,   144,   144,   144,   144,   144,   144,   144,   144,
     144,   144,   145,   145,   146,   146,   147,   147,   148,   148,
     150,   149,   151,   151,   152,   152,   152,   152,   152,   152,
     152,   152,   153,   154,   155,   156,   155,   155,   155,   157,
     155,   155,   155,   158,   159,   155,   160,   155,   155,   161,
     161,   162,   162,   163,   163,   163,   164,   164,   165,   165,
     166,   166,   166,   166,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   168,   168,   168,   168,   168,   168,   168,
     169,   169,   169,   170,   171,   172,   172
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
       0,     1,     0,     1,     1,     1,     3,     1,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     5,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     1,     5,
       7,     1,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     3,     5,     5,     4,     4,     3,     3,     2,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     1,     2,     0,     1,     1,     3,
       0,     4,     0,     1,     1,     1,     1,     2,     2,     3,
       2,     3,     1,     1,     2,     0,     4,     2,     2,     0,
       4,     2,     2,     0,     0,     7,     0,     5,     1,     1,
       2,     0,     2,     1,     1,     1,     1,     2,     1,     1,
       1,     3,     2,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,   182,     1,   208,   233,   224,   231,   235,   230,   226,
     227,   238,   225,   234,     0,   236,   237,     0,   229,   232,
       0,   228,   240,   241,   242,   239,     3,     0,   193,   183,
     184,   186,   216,   185,   219,   218,   245,   246,   180,   197,
     194,   201,   198,   206,   202,     4,   211,     0,     0,     6,
      13,    15,   244,   187,   209,   213,   215,   214,   211,   217,
     190,   188,     0,     0,     0,     0,     0,     0,     0,     5,
       0,    25,     0,   102,    63,   210,   189,   191,     0,   192,
      29,   196,   200,   220,     0,   204,    14,   212,    16,    12,
       9,     7,     0,     0,     0,     0,     0,     0,     0,     0,
     243,   167,   166,   162,   163,   164,   165,   168,   169,   172,
     174,     0,     0,     0,     0,     0,   103,   104,   107,   138,
     141,   170,   171,   161,     0,     0,    64,    40,    65,   181,
      31,    33,     0,   222,   207,     0,     0,     0,     0,    11,
      51,   143,   144,   145,   142,     0,   105,    40,   148,   149,
       0,   150,     0,   151,   146,   147,    18,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   176,
       0,   159,   160,   173,   175,     0,    17,     0,   211,   102,
       0,    67,    66,    41,    44,    45,    33,     0,    37,     0,
      34,    35,    15,   221,   223,     0,    71,     8,    27,     0,
       0,     0,    61,    58,    60,     0,     0,   152,   211,     0,
       0,    40,    40,   127,   137,   136,   135,   133,   134,   132,
     131,   130,   129,   128,     0,   125,   124,   123,   122,   121,
     120,   119,   115,   116,   118,   117,   113,   114,   111,   112,
     108,   109,   110,   158,     0,   178,     0,   177,   157,    68,
      69,    42,     0,    48,     0,   102,    63,     0,    39,    30,
       0,     0,   205,     0,    26,     0,    54,     0,    56,    55,
      62,    59,    52,   106,    42,     0,     0,     0,     0,   156,
     155,     0,    43,    49,    50,     0,     0,    32,    36,    38,
       0,   243,     0,     0,     0,     0,     0,     0,     0,   100,
       0,     0,     0,     0,    70,    72,    85,    74,    73,    80,
       0,     0,     0,   101,     0,    28,    53,    57,     0,   139,
     153,   154,   126,   179,    47,    46,    79,    78,    95,     0,
      96,    77,     0,     0,     0,     0,   176,     0,     0,   176,
      75,    81,    86,     0,    84,    19,    21,     0,     0,    76,
       0,    97,     0,    93,     0,     0,     0,     0,   100,     0,
      20,     0,   140,     0,     0,     0,     0,     0,     0,     0,
      82,     0,     0,    24,     0,    87,    98,    94,    91,    99,
     100,    83,    23,     0,     0,     0,    92,    88,   100,     0,
       0,    90
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,    26,    71,   136,    48,    72,   208,    50,   325,
     367,   379,    91,   219,    78,   131,   206,   209,   210,   211,
     202,   203,   204,   205,   222,   223,   224,   225,   125,   126,
     217,   283,   326,   327,   328,   389,   329,   330,   331,   332,
     115,   116,   333,   146,   118,   119,   120,   121,   122,   266,
     267,    39,    62,    27,    79,   127,    29,    30,    63,    64,
      66,   135,    65,    53,    67,    54,    31,    32,    84,    33,
      34,    35,   123,    51,    52
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -366
static const yytype_int16 yypact[] =
{
    -366,   541,  -366,  -366,  -366,  -366,  -366,  -366,  -366,  -366,
    -366,  -366,  -366,  -366,    -3,  -366,  -366,    -3,  -366,  -366,
     140,  -366,  -366,  -366,  -366,  -366,  -366,   161,  -366,  -366,
     949,   914,  -366,   949,  -366,  -366,  -366,  -366,  -366,  -366,
     -64,  -366,   -60,  -366,   -17,  -366,  -366,    21,    70,   316,
      51,  -366,  -366,   949,  -366,  -366,  -366,  -366,  -366,  -366,
     949,   949,   914,   -13,   -13,    57,     9,   154,    54,  -366,
      21,  -366,   164,   745,   836,  -366,   -41,   949,   875,  -366,
    -366,  -366,  -366,   167,     7,  -366,  -366,  -366,  -366,  -366,
     181,   914,   677,   745,   745,   745,   745,   607,   745,   745,
    -366,  -366,  -366,  -366,  -366,  -366,  -366,  -366,  -366,  -366,
    -366,   779,   813,   745,   745,    95,  -366,  1067,  -366,  -366,
     422,   141,   145,  -366,   176,   139,   225,   128,  -366,  -366,
    -366,   289,   745,    57,  -366,    57,   149,    21,   169,  -366,
    1067,  -366,  -366,  -366,  -366,    31,  1067,   130,  -366,  -366,
     607,  -366,   607,  -366,  -366,  -366,  -366,   745,   745,   745,
     745,   745,   745,   745,   745,   745,   745,   745,   745,   745,
     745,   745,   745,   745,   745,   745,   745,   745,   745,   745,
     745,   745,   745,   745,   745,   745,   745,    53,   745,   745,
      53,  -366,  -366,  -366,  -366,   199,  -366,   836,  -366,   745,
     147,  -366,  -366,  -366,   111,  -366,   289,   745,  -366,   243,
     247,  -366,   235,  1067,  -366,    13,  -366,  -366,  -366,   222,
      53,   745,   260,   255,   169,   172,   745,  -366,  -366,    -6,
     180,   130,   130,  1067,  1067,  1067,  1067,  1067,  1067,  1067,
    1067,  1067,  1067,  1067,    16,  1084,  1100,  1115,  1129,  1142,
    1153,  1153,   319,   319,   319,   319,   303,   303,   295,   295,
    -366,  -366,  -366,  -366,    11,  1067,   186,   269,  -366,  -366,
    -366,   190,   188,  -366,   187,   745,   836,   280,  -366,  -366,
     289,   745,  -366,   338,  -366,    21,  -366,   191,  -366,  -366,
     284,   255,  -366,  -366,   217,   711,   202,   204,   745,  -366,
    -366,   745,  -366,  -366,  -366,   198,   207,  -366,  -366,  -366,
     309,   296,   312,   745,   320,   307,   435,    53,   302,   745,
     305,   306,   308,   318,  -366,  -366,   504,  -366,  -366,  -366,
     149,   292,   342,   354,   259,  -366,  -366,  -366,   169,  -366,
    -366,  -366,   421,  -366,  -366,  -366,  -366,  -366,  -366,  1036,
    -366,  -366,   277,   358,   745,   359,   745,   745,   745,   745,
    -366,  -366,  -366,   325,  -366,  -366,   361,   234,   272,  -366,
     326,  -366,    64,  -366,   276,    66,    67,   281,   607,   367,
    -366,    21,  -366,   745,   435,   371,   435,   435,   372,   397,
    -366,    21,   677,  -366,    68,   368,  -366,  -366,  -366,  -366,
     745,   427,  -366,   461,   435,   462,  -366,  -366,   745,   377,
     435,  -366
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -366,  -366,  -366,  -366,  -366,   400,  -366,   -26,  -366,  -366,
    -365,  -366,  -366,   189,  -366,  -366,  -366,   266,   209,  -366,
    -122,  -187,  -366,  -366,   -82,   254,  -366,   133,   216,   299,
     168,  -366,  -366,   171,  -304,  -366,   173,  -366,  -366,  -227,
    -181,  -183,   -83,   -45,   -38,   137,  -366,  -366,  -366,  -308,
     203,     2,  -366,  -366,    -1,     0,   -88,   472,  -366,  -366,
    -366,  -366,  -366,   -10,   -51,   208,  -366,   474,    22,   256,
     265,   -24,   -52,  -127,   -12
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -204
static const yytype_int16 yytable[] =
{
      28,    49,    40,   137,   212,    42,    57,    76,    44,    57,
     139,   133,   352,   274,   145,   226,   393,   133,   272,    41,
     226,    68,   128,    61,   278,   230,   401,   228,   117,    57,
    -195,   199,   229,   298,  -199,   226,    57,    57,   287,    36,
      37,    86,   274,    87,    90,    22,    23,   140,   374,    24,
      77,   377,    87,    57,    46,   141,   142,   143,   144,    47,
     148,   149,    80,    36,    37,    81,    82,   145,   226,   145,
     226,   226,   226,    69,    70,   154,   155,  -203,   130,   212,
     395,    38,   397,   398,   302,   244,   273,   213,    73,    74,
      28,    38,   355,   140,   305,    36,    37,   147,   309,    83,
     407,   201,   134,    85,   299,   264,   411,   302,   282,   296,
     297,   218,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   227,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   290,   293,   265,   128,    88,   271,   275,   276,
     231,   390,   232,   212,   117,   214,   384,   215,   386,   387,
     403,   198,   117,   228,    45,   199,   200,   199,   229,    92,
      36,    37,   132,   405,    68,   263,   117,   294,   268,   140,
     198,   409,    36,    37,   199,   200,   -10,    46,   156,    36,
      37,    93,    47,   193,    46,   334,    36,    37,   194,    47,
      94,    95,    96,    36,    37,   220,   221,    97,   286,    98,
      99,   100,   195,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   198,   128,   284,   285,   199,   200,   197,
     117,   196,    36,    37,    43,   269,   117,   380,   381,   273,
      22,    23,   111,   216,    24,    86,   279,    87,   151,   153,
     228,   280,   281,   342,   199,   229,   265,   339,   112,   218,
     289,    75,   365,   138,   288,   113,   114,   292,   349,    75,
      87,   372,   295,   301,   375,   376,    22,    23,   300,   304,
      24,   303,    28,   307,   336,    75,    55,    60,   337,    55,
     391,   344,    46,   140,   340,    56,   341,    47,    56,   345,
     394,    36,    37,    22,    23,   353,   207,    24,   366,    55,
     402,   265,   346,   347,   265,   348,    55,    55,    56,    -9,
      -9,   -10,    46,   350,   351,    56,    56,    47,   184,   185,
     186,    36,    37,    55,   182,   183,   184,   185,   186,   310,
     354,  -100,    56,   356,   357,   364,   358,   140,   180,   181,
     182,   183,   184,   185,   186,   366,   359,   363,   226,   370,
      93,   371,   373,   378,   383,   366,   -22,   382,   385,    94,
      95,    96,   392,   388,   396,   399,    97,    28,    98,    99,
     311,     3,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,     4,   312,   313,     5,   314,   315,   316,     6,
     400,     7,     8,   -89,   317,   318,     9,    10,    11,   319,
      12,   111,   320,    13,    14,   321,    15,    16,    17,    18,
     322,    19,    20,    21,    22,    23,   323,   112,    24,    25,
     404,   381,   -85,   324,   113,   114,   310,   168,  -100,   169,
     170,   171,   172,   173,   174,   175,   176,   177,   178,   179,
     180,   181,   182,   183,   184,   185,   186,    93,   187,   188,
     189,   190,   191,   192,   406,   408,    94,    95,    96,   410,
      89,   368,   277,    97,   335,    98,    99,   311,   291,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   308,
     312,   313,   306,   314,   315,   316,   270,   360,   362,   361,
     -89,   317,   318,    58,   343,    59,   319,  -100,   111,   320,
       0,     0,   321,     0,     0,     0,     0,   322,     0,     0,
       0,     0,     0,   323,   112,     0,    93,     0,     0,   -85,
       0,   113,   114,     0,     0,    94,    95,    96,     0,     0,
       0,     2,    97,     0,    98,    99,   311,     0,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   110,     0,   312,
     313,     0,   314,   315,   316,     0,     0,     0,     0,   -89,
     317,   318,     0,     0,     0,   319,     0,   111,   320,     0,
       0,   321,     0,     0,     3,     0,   322,     0,     0,     0,
       0,     0,   323,   112,     0,     4,     0,     0,     5,     0,
     113,   114,     6,     0,     7,     8,     0,     0,     0,     9,
      10,    11,     0,    12,     0,     0,    13,    14,     0,    15,
      16,    17,    18,     0,    19,    20,    21,    22,    23,    93,
       0,    24,    25,     0,     0,     0,     0,     0,    94,    95,
      96,     0,     0,     0,     0,    97,     0,    98,    99,   100,
       3,   101,   102,   103,   104,   105,   106,   107,   108,   109,
     110,     4,     0,     0,     5,     0,     0,     0,     6,     0,
       7,     8,     0,     0,     0,     9,    10,    11,     0,    12,
     111,     0,    13,    14,     0,    15,    16,    17,    18,     0,
      19,    20,    21,    22,    23,     0,   112,    24,    25,    93,
       0,     0,     0,   113,   114,     0,     0,     0,    94,    95,
      96,     0,     0,     0,     0,    97,     0,    98,    99,   100,
       0,   101,   102,   103,   104,   105,   106,   107,   108,   109,
     110,     0,     0,    93,     0,     0,     0,     0,     0,     0,
       0,     0,    94,    95,    96,     0,     0,     0,     0,    97,
     111,    98,    99,   100,     0,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,     0,   112,    93,     0,     0,
       0,   138,     0,   113,   114,     0,    94,    95,    96,     0,
       0,     0,     0,    97,   111,    98,    99,   100,     0,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,     0,
     112,    93,     0,     0,     0,   338,     0,   113,   114,     0,
      94,    95,    96,     0,     0,     0,     0,   150,   111,    98,
      99,   100,     0,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,     0,   112,    93,     0,     0,     0,     0,
       0,   113,   114,     0,    94,    95,    96,     0,     0,     0,
       0,   152,   111,    98,    99,   100,     0,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,     0,   112,     0,
       0,     0,   124,     0,     0,   113,   114,     0,   100,     3,
       0,     0,     0,     0,     0,     0,   111,     0,     0,     0,
       4,     0,     0,     5,     0,     0,     0,     6,     0,     7,
       8,     0,   112,     0,     9,    10,    11,     0,    12,   113,
     114,    13,    14,     0,    15,    16,    17,    18,     3,    19,
      20,    21,    22,    23,     0,     0,    24,    25,     0,     4,
       0,     0,     5,     0,     0,     0,     6,     0,     7,     8,
       0,     0,     0,     9,    10,    11,     0,    12,     0,     0,
      13,    14,     0,    15,    16,    17,    18,     3,    19,    20,
      21,    22,    23,     0,     0,    24,    25,     0,     4,     0,
     129,     5,     0,     0,     0,     6,     0,     7,     8,     0,
       0,     0,     9,    10,    11,     0,    12,     0,     0,    13,
      14,     0,    15,    16,    17,    18,     0,    19,    20,    21,
      22,    23,     0,     4,    24,    25,     5,     0,     0,     0,
       6,     0,     7,     8,     0,     0,     0,     9,    10,    11,
       0,    12,     0,     0,    13,     0,     0,    15,    16,     0,
      18,     0,    19,     0,    21,    22,    23,     0,     0,    24,
      25,   157,   158,   159,   160,   161,   162,   163,   164,   165,
     166,   167,   168,   369,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,     0,   169,   170,   171,   172,   173,
     174,   175,   176,   177,   178,   179,   180,   181,   182,   183,
     184,   185,   186,   170,   171,   172,   173,   174,   175,   176,
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
      92,     4,   316,   200,    97,     4,   381,     4,   199,    17,
       4,    47,    74,    33,   207,   147,   391,    33,    73,    53,
      94,    37,    38,    17,    94,     4,    60,    61,   221,    42,
      43,    67,   229,    67,    70,    86,    87,    92,   356,    90,
      60,   359,    76,    77,    33,    93,    94,    95,    96,    38,
      98,    99,    62,    42,    43,    63,    64,   150,     4,   152,
       4,     4,     4,     3,     4,   113,   114,    94,    78,   206,
     384,    94,   386,   387,   271,   168,    92,   132,    37,    38,
      91,    94,   319,   138,   275,    42,    43,    97,   281,    42,
     404,   127,    95,    94,    93,   188,   410,   294,    95,   231,
     232,   137,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,    92,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   224,   226,   189,   197,    92,   198,    37,    38,
     150,   378,   152,   280,   199,   133,    92,   135,    92,    92,
      92,    33,   207,    33,     3,    37,    38,    37,    38,     5,
      42,    43,     5,   400,   200,   187,   221,   228,   190,   224,
      33,   408,    42,    43,    37,    38,     5,    33,    93,    42,
      43,    22,    38,    52,    33,   283,    42,    43,    53,    38,
      31,    32,    33,    42,    43,    36,    37,    38,   220,    40,
      41,    42,    36,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    33,   276,     3,     4,    37,    38,     4,
     275,    92,    42,    43,    94,    36,   281,     3,     4,    92,
      86,    87,    73,    94,    90,   271,     3,   271,   111,   112,
      33,     4,    17,   298,    37,    38,   301,   295,    89,   285,
       5,    53,     3,    94,     4,    96,    97,    95,   313,    61,
     294,   354,    92,     4,   357,   358,    86,    87,    92,    92,
      90,    93,   283,     3,    93,    77,    30,    31,     4,    33,
     378,    93,    33,   338,    92,    30,    92,    38,    33,    92,
     383,    42,    43,    86,    87,   317,    17,    90,   334,    53,
     392,   356,     3,    17,   359,     3,    60,    61,    53,     3,
       4,     5,    33,     3,    17,    60,    61,    38,    33,    34,
      35,    42,    43,    77,    31,    32,    33,    34,    35,     1,
      38,     3,    77,    38,    38,     3,    38,   392,    29,    30,
      31,    32,    33,    34,    35,   381,    38,    65,     4,    82,
      22,     3,     3,    38,    38,   391,     5,    95,    92,    31,
      32,    33,     5,    92,     3,     3,    38,   378,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
       3,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      62,     4,    94,    95,    96,    97,     1,    16,     3,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    22,    36,    37,
      38,    39,    40,    41,     3,     3,    31,    32,    33,    92,
      70,   338,   206,    38,   285,    40,    41,    42,   224,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,   280,
      55,    56,   276,    58,    59,    60,   197,   326,   330,   326,
      65,    66,    67,    31,   301,    31,    71,     3,    73,    74,
      -1,    -1,    77,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    88,    89,    -1,    22,    -1,    -1,    94,
      -1,    96,    97,    -1,    -1,    31,    32,    33,    -1,    -1,
      -1,     0,    38,    -1,    40,    41,    42,    -1,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    -1,    55,
      56,    -1,    58,    59,    60,    -1,    -1,    -1,    -1,    65,
      66,    67,    -1,    -1,    -1,    71,    -1,    73,    74,    -1,
      -1,    77,    -1,    -1,    43,    -1,    82,    -1,    -1,    -1,
      -1,    -1,    88,    89,    -1,    54,    -1,    -1,    57,    -1,
      96,    97,    61,    -1,    63,    64,    -1,    -1,    -1,    68,
      69,    70,    -1,    72,    -1,    -1,    75,    76,    -1,    78,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    22,
      -1,    90,    91,    -1,    -1,    -1,    -1,    -1,    31,    32,
      33,    -1,    -1,    -1,    -1,    38,    -1,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    -1,    -1,    57,    -1,    -1,    -1,    61,    -1,
      63,    64,    -1,    -1,    -1,    68,    69,    70,    -1,    72,
      73,    -1,    75,    76,    -1,    78,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    -1,    89,    90,    91,    22,
      -1,    -1,    -1,    96,    97,    -1,    -1,    -1,    31,    32,
      33,    -1,    -1,    -1,    -1,    38,    -1,    40,    41,    42,
      -1,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    -1,    -1,    22,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    31,    32,    33,    -1,    -1,    -1,    -1,    38,
      73,    40,    41,    42,    -1,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    -1,    89,    22,    -1,    -1,
      -1,    94,    -1,    96,    97,    -1,    31,    32,    33,    -1,
      -1,    -1,    -1,    38,    73,    40,    41,    42,    -1,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    -1,
      89,    22,    -1,    -1,    -1,    94,    -1,    96,    97,    -1,
      31,    32,    33,    -1,    -1,    -1,    -1,    38,    73,    40,
      41,    42,    -1,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    -1,    89,    22,    -1,    -1,    -1,    -1,
      -1,    96,    97,    -1,    31,    32,    33,    -1,    -1,    -1,
      -1,    38,    73,    40,    41,    42,    -1,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    -1,    89,    -1,
      -1,    -1,    36,    -1,    -1,    96,    97,    -1,    42,    43,
      -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,
      54,    -1,    -1,    57,    -1,    -1,    -1,    61,    -1,    63,
      64,    -1,    89,    -1,    68,    69,    70,    -1,    72,    96,
      97,    75,    76,    -1,    78,    79,    80,    81,    43,    83,
      84,    85,    86,    87,    -1,    -1,    90,    91,    -1,    54,
      -1,    -1,    57,    -1,    -1,    -1,    61,    -1,    63,    64,
      -1,    -1,    -1,    68,    69,    70,    -1,    72,    -1,    -1,
      75,    76,    -1,    78,    79,    80,    81,    43,    83,    84,
      85,    86,    87,    -1,    -1,    90,    91,    -1,    54,    -1,
      95,    57,    -1,    -1,    -1,    61,    -1,    63,    64,    -1,
      -1,    -1,    68,    69,    70,    -1,    72,    -1,    -1,    75,
      76,    -1,    78,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    -1,    54,    90,    91,    57,    -1,    -1,    -1,
      61,    -1,    63,    64,    -1,    -1,    -1,    68,    69,    70,
      -1,    72,    -1,    -1,    75,    -1,    -1,    78,    79,    -1,
      81,    -1,    83,    -1,    85,    86,    87,    -1,    -1,    90,
      91,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    -1,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    19,    20,    21,    22,    23,    24,    25,
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
       0,    99,     0,    43,    54,    57,    61,    63,    64,    68,
      69,    70,    72,    75,    76,    78,    79,    80,    81,    83,
      84,    85,    86,    87,    90,    91,   100,   151,   152,   154,
     155,   164,   165,   167,   168,   169,    42,    43,    94,   149,
     172,   149,   172,    94,   172,     3,    33,    38,   103,   105,
     106,   171,   172,   161,   163,   167,   168,   169,   155,   165,
     167,   161,   150,   156,   157,   160,   158,   162,   105,     3,
       4,   101,   104,    37,    38,   163,   162,   161,   112,   152,
     153,   149,   149,    42,   166,    94,   105,   169,    92,   103,
     105,   110,     5,    22,    31,    32,    33,    38,    40,    41,
      42,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    73,    89,    96,    97,   138,   139,   141,   142,   143,
     144,   145,   146,   170,    36,   126,   127,   153,   170,    95,
     153,   113,     5,     4,    95,   159,   102,   154,    94,   122,
     141,   142,   142,   142,   142,   140,   141,   153,   142,   142,
      38,   143,    38,   143,   142,   142,    93,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    52,    53,    36,    92,     4,    33,    37,
      38,   105,   118,   119,   120,   121,   114,    17,   105,   115,
     116,   117,   171,   141,   166,   166,    94,   128,   105,   111,
      36,    37,   122,   123,   124,   125,     4,    92,    33,    38,
     118,   153,   153,   141,   141,   141,   141,   141,   141,   141,
     141,   141,   141,   141,   140,   141,   141,   141,   141,   141,
     141,   141,   141,   141,   141,   141,   141,   141,   141,   141,
     141,   141,   141,   172,   140,   141,   147,   148,   172,    36,
     127,   162,   138,    92,   119,    37,    38,   115,   139,     3,
       4,    17,    95,   129,     3,     4,   172,   139,     4,     5,
     122,   123,    95,   140,   162,    92,   118,   118,    17,    93,
      92,     4,   119,    93,    92,   138,   126,     3,   116,   139,
       1,    42,    55,    56,    58,    59,    60,    66,    67,    71,
      74,    77,    82,    88,    95,   107,   130,   131,   132,   134,
     135,   136,   137,   140,   154,   111,    93,     4,    94,   142,
      92,    92,   141,   148,    93,    92,     3,    17,     3,   141,
       3,    17,   132,   172,    38,   137,    38,    38,    38,    38,
     131,   134,   128,    65,     3,     3,   105,   108,   125,    17,
      82,     3,   140,     3,   147,   140,   140,   147,    38,   109,
       3,     4,    95,    38,    92,    92,    92,    92,    92,   133,
     137,   154,     5,   108,   140,   132,     3,   132,   132,     3,
       3,   108,   122,    92,    62,   137,     3,   132,     3,   137,
      92,   132
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
		(yyval.node) = new(OSET, (yyvsp[(3) - (5)].node), Z);
	}
    break;

  case 100:
#line 544 "cc.y"
    {
		(yyval.node) = Z;
	}
    break;

  case 102:
#line 550 "cc.y"
    {
		(yyval.node) = Z;
	}
    break;

  case 104:
#line 557 "cc.y"
    {
		(yyval.node) = new(OCAST, (yyvsp[(1) - (1)].node), Z);
		(yyval.node)->type = types[TLONG];
	}
    break;

  case 106:
#line 565 "cc.y"
    {
		(yyval.node) = new(OCOMMA, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 108:
#line 572 "cc.y"
    {
		(yyval.node) = new(OMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 109:
#line 576 "cc.y"
    {
		(yyval.node) = new(ODIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 110:
#line 580 "cc.y"
    {
		(yyval.node) = new(OMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 111:
#line 584 "cc.y"
    {
		(yyval.node) = new(OADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 112:
#line 588 "cc.y"
    {
		(yyval.node) = new(OSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 113:
#line 592 "cc.y"
    {
		(yyval.node) = new(OASHR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 114:
#line 596 "cc.y"
    {
		(yyval.node) = new(OASHL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 115:
#line 600 "cc.y"
    {
		(yyval.node) = new(OLT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 116:
#line 604 "cc.y"
    {
		(yyval.node) = new(OGT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 117:
#line 608 "cc.y"
    {
		(yyval.node) = new(OLE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 118:
#line 612 "cc.y"
    {
		(yyval.node) = new(OGE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 119:
#line 616 "cc.y"
    {
		(yyval.node) = new(OEQ, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 120:
#line 620 "cc.y"
    {
		(yyval.node) = new(ONE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 121:
#line 624 "cc.y"
    {
		(yyval.node) = new(OAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 122:
#line 628 "cc.y"
    {
		(yyval.node) = new(OXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 123:
#line 632 "cc.y"
    {
		(yyval.node) = new(OOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 124:
#line 636 "cc.y"
    {
		(yyval.node) = new(OANDAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 125:
#line 640 "cc.y"
    {
		(yyval.node) = new(OOROR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 126:
#line 644 "cc.y"
    {
		(yyval.node) = new(OCOND, (yyvsp[(1) - (5)].node), new(OLIST, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node)));
	}
    break;

  case 127:
#line 648 "cc.y"
    {
		(yyval.node) = new(OAS, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 128:
#line 652 "cc.y"
    {
		(yyval.node) = new(OASADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 129:
#line 656 "cc.y"
    {
		(yyval.node) = new(OASSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 130:
#line 660 "cc.y"
    {
		(yyval.node) = new(OASMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 131:
#line 664 "cc.y"
    {
		(yyval.node) = new(OASDIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 132:
#line 668 "cc.y"
    {
		(yyval.node) = new(OASMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 133:
#line 672 "cc.y"
    {
		(yyval.node) = new(OASASHL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 134:
#line 676 "cc.y"
    {
		(yyval.node) = new(OASASHR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 135:
#line 680 "cc.y"
    {
		(yyval.node) = new(OASAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 136:
#line 684 "cc.y"
    {
		(yyval.node) = new(OASXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 137:
#line 688 "cc.y"
    {
		(yyval.node) = new(OASOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 139:
#line 695 "cc.y"
    {
		(yyval.node) = new(OCAST, (yyvsp[(5) - (5)].node), Z);
		dodecl(NODECL, CXXX, (yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].node));
		(yyval.node)->type = lastdcl;
		(yyval.node)->xcast = 1;
	}
    break;

  case 140:
#line 702 "cc.y"
    {
		(yyval.node) = new(OSTRUCT, (yyvsp[(6) - (7)].node), Z);
		dodecl(NODECL, CXXX, (yyvsp[(2) - (7)].type), (yyvsp[(3) - (7)].node));
		(yyval.node)->type = lastdcl;
	}
    break;

  case 142:
#line 711 "cc.y"
    {
		(yyval.node) = new(OIND, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 143:
#line 715 "cc.y"
    {
		(yyval.node) = new(OADDR, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 144:
#line 719 "cc.y"
    {
		(yyval.node) = new(OPOS, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 145:
#line 723 "cc.y"
    {
		(yyval.node) = new(ONEG, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 146:
#line 727 "cc.y"
    {
		(yyval.node) = new(ONOT, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 147:
#line 731 "cc.y"
    {
		(yyval.node) = new(OCOM, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 148:
#line 735 "cc.y"
    {
		(yyval.node) = new(OPREINC, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 149:
#line 739 "cc.y"
    {
		(yyval.node) = new(OPREDEC, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 150:
#line 743 "cc.y"
    {
		(yyval.node) = new(OSIZE, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 151:
#line 747 "cc.y"
    {
		(yyval.node) = new(OSIGN, (yyvsp[(2) - (2)].node), Z);
	}
    break;

  case 152:
#line 753 "cc.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
	}
    break;

  case 153:
#line 757 "cc.y"
    {
		(yyval.node) = new(OSIZE, Z, Z);
		dodecl(NODECL, CXXX, (yyvsp[(3) - (5)].type), (yyvsp[(4) - (5)].node));
		(yyval.node)->type = lastdcl;
	}
    break;

  case 154:
#line 763 "cc.y"
    {
		(yyval.node) = new(OSIGN, Z, Z);
		dodecl(NODECL, CXXX, (yyvsp[(3) - (5)].type), (yyvsp[(4) - (5)].node));
		(yyval.node)->type = lastdcl;
	}
    break;

  case 155:
#line 769 "cc.y"
    {
		(yyval.node) = new(OFUNC, (yyvsp[(1) - (4)].node), Z);
		if((yyvsp[(1) - (4)].node)->op == ONAME)
		if((yyvsp[(1) - (4)].node)->type == T)
			dodecl(xdecl, CXXX, types[TINT], (yyval.node));
		(yyval.node)->right = invert((yyvsp[(3) - (4)].node));
	}
    break;

  case 156:
#line 777 "cc.y"
    {
		(yyval.node) = new(OIND, new(OADD, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node)), Z);
	}
    break;

  case 157:
#line 781 "cc.y"
    {
		(yyval.node) = new(ODOT, new(OIND, (yyvsp[(1) - (3)].node), Z), Z);
		(yyval.node)->sym = (yyvsp[(3) - (3)].sym);
	}
    break;

  case 158:
#line 786 "cc.y"
    {
		(yyval.node) = new(ODOT, (yyvsp[(1) - (3)].node), Z);
		(yyval.node)->sym = (yyvsp[(3) - (3)].sym);
	}
    break;

  case 159:
#line 791 "cc.y"
    {
		(yyval.node) = new(OPOSTINC, (yyvsp[(1) - (2)].node), Z);
	}
    break;

  case 160:
#line 795 "cc.y"
    {
		(yyval.node) = new(OPOSTDEC, (yyvsp[(1) - (2)].node), Z);
	}
    break;

  case 162:
#line 800 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TINT];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 163:
#line 807 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TLONG];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 164:
#line 814 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TUINT];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 165:
#line 821 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TULONG];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 166:
#line 828 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TDOUBLE];
		(yyval.node)->fconst = (yyvsp[(1) - (1)].dval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 167:
#line 835 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TFLOAT];
		(yyval.node)->fconst = (yyvsp[(1) - (1)].dval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 168:
#line 842 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TVLONG];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 169:
#line 849 "cc.y"
    {
		(yyval.node) = new(OCONST, Z, Z);
		(yyval.node)->type = types[TUVLONG];
		(yyval.node)->vconst = (yyvsp[(1) - (1)].vval);
		(yyval.node)->cstring = strdup(symb);
	}
    break;

  case 172:
#line 860 "cc.y"
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

  case 173:
#line 870 "cc.y"
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

  case 174:
#line 888 "cc.y"
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

  case 175:
#line 898 "cc.y"
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

  case 176:
#line 915 "cc.y"
    {
		(yyval.node) = Z;
	}
    break;

  case 179:
#line 923 "cc.y"
    {
		(yyval.node) = new(OLIST, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 180:
#line 929 "cc.y"
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

  case 181:
#line 942 "cc.y"
    {
		(yyval.type) = strf;
		strf = (yyvsp[(2) - (4)].tyty).t1;
		strl = (yyvsp[(2) - (4)].tyty).t2;
		lasttype = (yyvsp[(2) - (4)].tyty).t3;
		lastclass = (yyvsp[(2) - (4)].tyty).c;
	}
    break;

  case 182:
#line 951 "cc.y"
    {
		lastclass = CXXX;
		lasttype = types[TINT];
	}
    break;

  case 184:
#line 959 "cc.y"
    {
		(yyval.tycl).t = (yyvsp[(1) - (1)].type);
		(yyval.tycl).c = CXXX;
	}
    break;

  case 185:
#line 964 "cc.y"
    {
		(yyval.tycl).t = simplet((yyvsp[(1) - (1)].lval));
		(yyval.tycl).c = CXXX;
	}
    break;

  case 186:
#line 969 "cc.y"
    {
		(yyval.tycl).t = simplet((yyvsp[(1) - (1)].lval));
		(yyval.tycl).c = simplec((yyvsp[(1) - (1)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(1) - (1)].lval));
	}
    break;

  case 187:
#line 975 "cc.y"
    {
		(yyval.tycl).t = (yyvsp[(1) - (2)].type);
		(yyval.tycl).c = simplec((yyvsp[(2) - (2)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(2) - (2)].lval));
		if((yyvsp[(2) - (2)].lval) & ~BCLASS & ~BGARB)
			diag(Z, "duplicate types given: %T and %Q", (yyvsp[(1) - (2)].type), (yyvsp[(2) - (2)].lval));
	}
    break;

  case 188:
#line 983 "cc.y"
    {
		(yyval.tycl).t = simplet(typebitor((yyvsp[(1) - (2)].lval), (yyvsp[(2) - (2)].lval)));
		(yyval.tycl).c = simplec((yyvsp[(2) - (2)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(2) - (2)].lval));
	}
    break;

  case 189:
#line 989 "cc.y"
    {
		(yyval.tycl).t = (yyvsp[(2) - (3)].type);
		(yyval.tycl).c = simplec((yyvsp[(1) - (3)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(1) - (3)].lval)|(yyvsp[(3) - (3)].lval));
	}
    break;

  case 190:
#line 995 "cc.y"
    {
		(yyval.tycl).t = simplet((yyvsp[(2) - (2)].lval));
		(yyval.tycl).c = simplec((yyvsp[(1) - (2)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(1) - (2)].lval));
	}
    break;

  case 191:
#line 1001 "cc.y"
    {
		(yyval.tycl).t = simplet(typebitor((yyvsp[(2) - (3)].lval), (yyvsp[(3) - (3)].lval)));
		(yyval.tycl).c = simplec((yyvsp[(1) - (3)].lval)|(yyvsp[(3) - (3)].lval));
		(yyval.tycl).t = garbt((yyval.tycl).t, (yyvsp[(1) - (3)].lval)|(yyvsp[(3) - (3)].lval));
	}
    break;

  case 192:
#line 1009 "cc.y"
    {
		(yyval.type) = (yyvsp[(1) - (1)].tycl).t;
		if((yyvsp[(1) - (1)].tycl).c != CXXX)
			diag(Z, "illegal combination of class 4: %s", cnames[(yyvsp[(1) - (1)].tycl).c]);
	}
    break;

  case 193:
#line 1017 "cc.y"
    {
		lasttype = (yyvsp[(1) - (1)].tycl).t;
		lastclass = (yyvsp[(1) - (1)].tycl).c;
	}
    break;

  case 194:
#line 1024 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TSTRUCT, 0);
		(yyval.type) = (yyvsp[(2) - (2)].sym)->suetag;
	}
    break;

  case 195:
#line 1029 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TSTRUCT, autobn);
	}
    break;

  case 196:
#line 1033 "cc.y"
    {
		(yyval.type) = (yyvsp[(2) - (4)].sym)->suetag;
		if((yyval.type)->link != T)
			diag(Z, "redeclare tag: %s", (yyvsp[(2) - (4)].sym)->name);
		(yyval.type)->link = (yyvsp[(4) - (4)].type);
		sualign((yyval.type));
	}
    break;

  case 197:
#line 1041 "cc.y"
    {
		taggen++;
		sprint(symb, "_%d_", taggen);
		(yyval.type) = dotag(lookup(), TSTRUCT, autobn);
		(yyval.type)->link = (yyvsp[(2) - (2)].type);
		sualign((yyval.type));
	}
    break;

  case 198:
#line 1049 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TUNION, 0);
		(yyval.type) = (yyvsp[(2) - (2)].sym)->suetag;
	}
    break;

  case 199:
#line 1054 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TUNION, autobn);
	}
    break;

  case 200:
#line 1058 "cc.y"
    {
		(yyval.type) = (yyvsp[(2) - (4)].sym)->suetag;
		if((yyval.type)->link != T)
			diag(Z, "redeclare tag: %s", (yyvsp[(2) - (4)].sym)->name);
		(yyval.type)->link = (yyvsp[(4) - (4)].type);
		sualign((yyval.type));
	}
    break;

  case 201:
#line 1066 "cc.y"
    {
		taggen++;
		sprint(symb, "_%d_", taggen);
		(yyval.type) = dotag(lookup(), TUNION, autobn);
		(yyval.type)->link = (yyvsp[(2) - (2)].type);
		sualign((yyval.type));
	}
    break;

  case 202:
#line 1074 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TENUM, 0);
		(yyval.type) = (yyvsp[(2) - (2)].sym)->suetag;
		if((yyval.type)->link == T)
			(yyval.type)->link = types[TINT];
		(yyval.type) = (yyval.type)->link;
	}
    break;

  case 203:
#line 1082 "cc.y"
    {
		dotag((yyvsp[(2) - (2)].sym), TENUM, autobn);
	}
    break;

  case 204:
#line 1086 "cc.y"
    {
		en.tenum = T;
		en.cenum = T;
	}
    break;

  case 205:
#line 1091 "cc.y"
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

  case 206:
#line 1103 "cc.y"
    {
		en.tenum = T;
		en.cenum = T;
	}
    break;

  case 207:
#line 1108 "cc.y"
    {
		(yyval.type) = en.tenum;
	}
    break;

  case 208:
#line 1112 "cc.y"
    {
		(yyval.type) = tcopy((yyvsp[(1) - (1)].sym)->type);
	}
    break;

  case 210:
#line 1119 "cc.y"
    {
		(yyval.lval) = typebitor((yyvsp[(1) - (2)].lval), (yyvsp[(2) - (2)].lval));
	}
    break;

  case 211:
#line 1124 "cc.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 212:
#line 1128 "cc.y"
    {
		(yyval.lval) = typebitor((yyvsp[(1) - (2)].lval), (yyvsp[(2) - (2)].lval));
	}
    break;

  case 217:
#line 1140 "cc.y"
    {
		(yyval.lval) = typebitor((yyvsp[(1) - (2)].lval), (yyvsp[(2) - (2)].lval));
	}
    break;

  case 220:
#line 1150 "cc.y"
    {
		doenum((yyvsp[(1) - (1)].sym), Z);
	}
    break;

  case 221:
#line 1154 "cc.y"
    {
		doenum((yyvsp[(1) - (3)].sym), (yyvsp[(3) - (3)].node));
	}
    break;

  case 224:
#line 1161 "cc.y"
    { (yyval.lval) = BCHAR; }
    break;

  case 225:
#line 1162 "cc.y"
    { (yyval.lval) = BSHORT; }
    break;

  case 226:
#line 1163 "cc.y"
    { (yyval.lval) = BINT; }
    break;

  case 227:
#line 1164 "cc.y"
    { (yyval.lval) = BLONG; }
    break;

  case 228:
#line 1165 "cc.y"
    { (yyval.lval) = BSIGNED; }
    break;

  case 229:
#line 1166 "cc.y"
    { (yyval.lval) = BUNSIGNED; }
    break;

  case 230:
#line 1167 "cc.y"
    { (yyval.lval) = BFLOAT; }
    break;

  case 231:
#line 1168 "cc.y"
    { (yyval.lval) = BDOUBLE; }
    break;

  case 232:
#line 1169 "cc.y"
    { (yyval.lval) = BVOID; }
    break;

  case 233:
#line 1172 "cc.y"
    { (yyval.lval) = BAUTO; }
    break;

  case 234:
#line 1173 "cc.y"
    { (yyval.lval) = BSTATIC; }
    break;

  case 235:
#line 1174 "cc.y"
    { (yyval.lval) = BEXTERN; }
    break;

  case 236:
#line 1175 "cc.y"
    { (yyval.lval) = BTYPEDEF; }
    break;

  case 237:
#line 1176 "cc.y"
    { (yyval.lval) = BTYPESTR; }
    break;

  case 238:
#line 1177 "cc.y"
    { (yyval.lval) = BREGISTER; }
    break;

  case 239:
#line 1178 "cc.y"
    { (yyval.lval) = 0; }
    break;

  case 240:
#line 1181 "cc.y"
    { (yyval.lval) = BCONSTNT; }
    break;

  case 241:
#line 1182 "cc.y"
    { (yyval.lval) = BVOLATILE; }
    break;

  case 242:
#line 1183 "cc.y"
    { (yyval.lval) = 0; }
    break;

  case 243:
#line 1187 "cc.y"
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

  case 244:
#line 1202 "cc.y"
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
#line 3596 "y.tab.c"
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


#line 1215 "cc.y"


