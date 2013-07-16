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
     LTYPEPLD = 282,
     LCONST = 283,
     LSP = 284,
     LSB = 285,
     LFP = 286,
     LPC = 287,
     LTYPEX = 288,
     LTYPEPC = 289,
     LR = 290,
     LREG = 291,
     LF = 292,
     LFREG = 293,
     LC = 294,
     LCREG = 295,
     LPSR = 296,
     LFCR = 297,
     LCOND = 298,
     LS = 299,
     LAT = 300,
     LFCONST = 301,
     LSCONST = 302,
     LNAME = 303,
     LLAB = 304,
     LVAR = 305
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
#define LTYPEPLD 282
#define LCONST 283
#define LSP 284
#define LSB 285
#define LFP 286
#define LPC 287
#define LTYPEX 288
#define LTYPEPC 289
#define LR 290
#define LREG 291
#define LF 292
#define LFREG 293
#define LC 294
#define LCREG 295
#define LPSR 296
#define LFCR 297
#define LCOND 298
#define LS 299
#define LAT 300
#define LFCONST 301
#define LSCONST 302
#define LNAME 303
#define LLAB 304
#define LVAR 305




/* Copy the first part of user declarations.  */
#line 31 "a.y"

#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and a.h re-#defines getc */
#include <libc.h>
#include "a.h"


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
#line 38 "a.y"
{
	Sym	*sym;
	int32	lval;
	double	dval;
	char	sval[8];
	Gen	gen;
}
/* Line 193 of yacc.c.  */
#line 211 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 224 "y.tab.c"

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
#define YYLAST   615

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  71
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  35
/* YYNRULES -- Number of rules.  */
#define YYNRULES  132
/* YYNRULES -- Number of states.  */
#define YYNSTATES  335

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   305

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    69,    12,     5,     2,
      67,    68,    10,     8,    64,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    61,    63,
       6,    62,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    65,     2,    66,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    70,     2,     2,     2,
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
       2,     2,     2,     2,     2,     2,     1,     2,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     5,     9,    10,    15,    16,    21,
      26,    31,    33,    36,    39,    47,    54,    60,    66,    72,
      77,    82,    86,    90,    95,   102,   110,   118,   126,   133,
     140,   144,   149,   156,   165,   172,   177,   181,   187,   193,
     201,   208,   221,   229,   239,   242,   247,   250,   251,   254,
     257,   258,   261,   266,   269,   272,   275,   278,   283,   286,
     288,   291,   295,   297,   301,   305,   307,   309,   311,   316,
     318,   320,   322,   324,   326,   328,   330,   334,   336,   341,
     343,   348,   350,   352,   354,   356,   359,   361,   367,   372,
     377,   382,   387,   389,   391,   393,   395,   400,   402,   404,
     406,   411,   413,   415,   417,   422,   427,   433,   441,   442,
     445,   448,   450,   452,   454,   456,   458,   461,   464,   467,
     471,   472,   475,   477,   481,   485,   489,   493,   497,   502,
     507,   511,   515
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      72,     0,    -1,    -1,    -1,    72,    73,    74,    -1,    -1,
      59,    61,    75,    74,    -1,    -1,    58,    61,    76,    74,
      -1,    58,    62,   105,    63,    -1,    60,    62,   105,    63,
      -1,    63,    -1,    77,    63,    -1,     1,    63,    -1,    13,
      78,    89,    64,    96,    64,    91,    -1,    13,    78,    89,
      64,    96,    64,    -1,    13,    78,    89,    64,    91,    -1,
      14,    78,    89,    64,    91,    -1,    15,    78,    84,    64,
      84,    -1,    16,    78,    79,    80,    -1,    16,    78,    79,
      85,    -1,    36,    79,    86,    -1,    17,    79,    80,    -1,
      18,    78,    79,    84,    -1,    19,    78,    89,    64,    96,
      79,    -1,    20,    78,    87,    64,    65,    83,    66,    -1,
      20,    78,    65,    83,    66,    64,    87,    -1,    21,    78,
      91,    64,    86,    64,    91,    -1,    21,    78,    91,    64,
      86,    79,    -1,    21,    78,    79,    86,    64,    91,    -1,
      22,    78,    79,    -1,    23,   100,    64,    90,    -1,    23,
     100,    64,   103,    64,    90,    -1,    23,   100,    64,   103,
      64,    90,     9,   103,    -1,    24,   100,    11,   103,    64,
      81,    -1,    25,    78,    91,    79,    -1,    29,    79,    81,
      -1,    30,    78,    99,    64,    99,    -1,    32,    78,    98,
      64,    99,    -1,    32,    78,    98,    64,    48,    64,    99,
      -1,    33,    78,    99,    64,    99,    79,    -1,    31,    78,
     103,    64,   105,    64,    96,    64,    97,    64,    97,   104,
      -1,    34,    78,    91,    64,    91,    64,    92,    -1,    35,
      78,    91,    64,    91,    64,    91,    64,    96,    -1,    37,
      88,    -1,    44,    90,    64,    90,    -1,    26,    79,    -1,
      -1,    78,    53,    -1,    78,    54,    -1,    -1,    64,    79,
      -1,   103,    67,    42,    68,    -1,    58,   101,    -1,    59,
     101,    -1,    69,   103,    -1,    69,    88,    -1,    69,    10,
      69,    88,    -1,    69,    57,    -1,    82,    -1,    69,    56,
      -1,    69,     9,    56,    -1,    96,    -1,    96,     9,    96,
      -1,    96,    79,    83,    -1,    91,    -1,    81,    -1,    93,
      -1,    93,    67,    96,    68,    -1,    51,    -1,    52,    -1,
     103,    -1,    88,    -1,    99,    -1,    86,    -1,   100,    -1,
      67,    96,    68,    -1,    86,    -1,   103,    67,    95,    68,
      -1,   100,    -1,   100,    67,    95,    68,    -1,    87,    -1,
      91,    -1,    90,    -1,    93,    -1,    69,   103,    -1,    96,
      -1,    67,    96,    64,    96,    68,    -1,    96,     6,     6,
      94,    -1,    96,     7,     7,    94,    -1,    96,     9,     7,
      94,    -1,    96,    55,     7,    94,    -1,    96,    -1,   103,
      -1,    46,    -1,    42,    -1,    45,    67,   105,    68,    -1,
      95,    -1,    39,    -1,    50,    -1,    49,    67,   105,    68,
      -1,    99,    -1,    82,    -1,    48,    -1,    47,    67,   103,
      68,    -1,   103,    67,   102,    68,    -1,    58,   101,    67,
     102,    68,    -1,    58,     6,     7,   101,    67,    40,    68,
      -1,    -1,     8,   103,    -1,     9,   103,    -1,    40,    -1,
      39,    -1,    41,    -1,    38,    -1,    60,    -1,     9,   103,
      -1,     8,   103,    -1,    70,   103,    -1,    67,   105,    68,
      -1,    -1,    64,   105,    -1,   103,    -1,   105,     8,   105,
      -1,   105,     9,   105,    -1,   105,    10,   105,    -1,   105,
      11,   105,    -1,   105,    12,   105,    -1,   105,     6,     6,
     105,    -1,   105,     7,     7,   105,    -1,   105,     5,   105,
      -1,   105,     4,   105,    -1,   105,     3,   105,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    67,    67,    69,    68,    76,    75,    83,    82,    88,
      93,    99,   100,   101,   107,   111,   115,   122,   129,   136,
     140,   147,   154,   161,   168,   175,   184,   196,   200,   204,
     211,   218,   222,   226,   239,   246,   253,   260,   264,   268,
     272,   279,   301,   309,   318,   325,   332,   338,   341,   345,
     350,   351,   354,   360,   369,   377,   383,   388,   393,   399,
     402,   408,   416,   420,   429,   435,   436,   437,   438,   443,
     449,   455,   461,   462,   465,   466,   474,   483,   484,   493,
     494,   500,   503,   504,   505,   507,   515,   523,   532,   538,
     544,   550,   558,   564,   572,   573,   577,   585,   586,   592,
     593,   601,   602,   605,   611,   619,   627,   635,   645,   648,
     652,   658,   659,   660,   663,   664,   668,   672,   676,   680,
     686,   689,   695,   696,   700,   704,   708,   712,   716,   720,
     724,   728,   732
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'|'", "'^'", "'&'", "'<'", "'>'", "'+'",
  "'-'", "'*'", "'/'", "'%'", "LTYPE1", "LTYPE2", "LTYPE3", "LTYPE4",
  "LTYPE5", "LTYPE6", "LTYPE7", "LTYPE8", "LTYPE9", "LTYPEA", "LTYPEB",
  "LTYPEC", "LTYPED", "LTYPEE", "LTYPEF", "LTYPEG", "LTYPEH", "LTYPEI",
  "LTYPEJ", "LTYPEK", "LTYPEL", "LTYPEM", "LTYPEN", "LTYPEBX", "LTYPEPLD",
  "LCONST", "LSP", "LSB", "LFP", "LPC", "LTYPEX", "LTYPEPC", "LR", "LREG",
  "LF", "LFREG", "LC", "LCREG", "LPSR", "LFCR", "LCOND", "LS", "LAT",
  "LFCONST", "LSCONST", "LNAME", "LLAB", "LVAR", "':'", "'='", "';'",
  "','", "'['", "']'", "'('", "')'", "'$'", "'~'", "$accept", "prog", "@1",
  "line", "@2", "@3", "inst", "cond", "comma", "rel", "ximm", "fcon",
  "reglist", "gen", "nireg", "ireg", "ioreg", "oreg", "imsr", "imm", "reg",
  "regreg", "shift", "rcon", "sreg", "spreg", "creg", "frcon", "freg",
  "name", "offset", "pointer", "con", "oexpr", "expr", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   124,    94,    38,    60,    62,    43,    45,
      42,    47,    37,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,    58,    61,    59,    44,    91,    93,    40,    41,    36,
     126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    71,    72,    73,    72,    75,    74,    76,    74,    74,
      74,    74,    74,    74,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    78,    78,    78,
      79,    79,    80,    80,    80,    81,    81,    81,    81,    81,
      82,    82,    83,    83,    83,    84,    84,    84,    84,    84,
      84,    84,    84,    84,    85,    85,    86,    87,    87,    88,
      88,    88,    89,    89,    89,    90,    91,    92,    93,    93,
      93,    93,    94,    94,    95,    95,    95,    96,    96,    97,
      97,    98,    98,    99,    99,   100,   100,   100,   101,   101,
     101,   102,   102,   102,   103,   103,   103,   103,   103,   103,
     104,   104,   105,   105,   105,   105,   105,   105,   105,   105,
     105,   105,   105
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     0,     4,     4,
       4,     1,     2,     2,     7,     6,     5,     5,     5,     4,
       4,     3,     3,     4,     6,     7,     7,     7,     6,     6,
       3,     4,     6,     8,     6,     4,     3,     5,     5,     7,
       6,    12,     7,     9,     2,     4,     2,     0,     2,     2,
       0,     2,     4,     2,     2,     2,     2,     4,     2,     1,
       2,     3,     1,     3,     3,     1,     1,     1,     4,     1,
       1,     1,     1,     1,     1,     1,     3,     1,     4,     1,
       4,     1,     1,     1,     1,     2,     1,     5,     4,     4,
       4,     4,     1,     1,     1,     1,     4,     1,     1,     1,
       4,     1,     1,     1,     4,     4,     5,     7,     0,     2,
       2,     1,     1,     1,     1,     1,     2,     2,     2,     3,
       0,     2,     1,     3,     3,     3,     3,     3,     4,     4,
       3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    47,    47,    47,    47,    50,
      47,    47,    47,    47,    47,     0,     0,    47,    50,    50,
      47,    47,    47,    47,    47,    47,    50,     0,     0,     0,
       0,     0,    11,     4,     0,    13,     0,     0,     0,    50,
      50,     0,    50,     0,     0,    50,    50,     0,     0,   114,
     108,   115,     0,     0,     0,     0,     0,     0,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    77,    81,
      44,    79,     0,     0,     0,     7,     0,     5,     0,    12,
      98,    95,     0,    94,    48,    49,     0,    83,    82,    84,
      97,    86,     0,     0,   103,    69,    70,     0,    66,    59,
       0,    72,    65,    67,    73,    71,     0,    51,   108,   108,
      22,     0,     0,     0,     0,     0,     0,     0,     0,    86,
      30,   117,   116,     0,     0,     0,     0,   122,     0,   118,
       0,     0,     0,    50,    36,     0,     0,     0,   102,     0,
     101,     0,     0,     0,     0,    21,     0,     0,     0,    85,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    60,    58,    56,    55,     0,
       0,   108,    19,    20,    74,    75,     0,    53,    54,     0,
      23,     0,     0,    50,     0,     0,     0,     0,   108,   109,
     110,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   119,    31,     0,   112,   111,   113,     0,     0,
      35,     0,     0,     0,     0,     0,     0,     0,    76,     0,
       0,    45,     8,     9,     6,    10,     0,    16,    86,     0,
       0,     0,     0,    17,     0,    61,     0,    18,     0,    53,
       0,     0,    50,     0,     0,     0,     0,     0,    50,     0,
       0,   132,   131,   130,     0,     0,   123,   124,   125,   126,
     127,     0,   105,     0,    37,     0,   103,    38,    50,     0,
       0,    80,    78,    96,    15,    88,    92,    93,    89,    90,
      91,   104,    57,    68,    52,    24,     0,    63,    64,     0,
      29,    50,    28,     0,   106,   128,   129,    32,    34,     0,
       0,    40,     0,     0,    14,    26,    25,    27,     0,     0,
       0,    39,     0,    42,     0,   107,    33,     0,     0,     0,
       0,    99,     0,     0,    43,     0,     0,     0,     0,   120,
      87,   100,     0,    41,   121
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    33,   153,   151,    34,    36,   107,   110,
      98,    99,   182,   100,   173,    68,    69,   101,    86,    87,
      88,   313,    89,   275,    90,   119,   322,   139,   104,    71,
     126,   208,   127,   333,   128
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -127
static const yytype_int16 yypact[] =
{
    -127,    39,  -127,   320,   -20,  -127,  -127,  -127,  -127,   -13,
    -127,  -127,  -127,  -127,  -127,   260,   260,  -127,   -13,   -13,
    -127,  -127,  -127,  -127,  -127,  -127,   -13,   421,    -6,    28,
     -12,     9,  -127,  -127,    -9,  -127,   321,   321,   350,     3,
     -13,   122,     3,   321,   397,   323,     3,   444,   444,  -127,
     385,  -127,   444,   444,    13,     7,    74,   467,  -127,    23,
      91,   417,   248,    91,   467,   467,    29,   194,  -127,  -127,
    -127,    41,    44,   444,    54,  -127,   444,  -127,   444,  -127,
    -127,  -127,    79,  -127,  -127,  -127,    77,  -127,  -127,  -127,
    -127,    69,    85,    86,  -127,  -127,  -127,    56,  -127,  -127,
      92,  -127,  -127,   109,  -127,    44,   200,  -127,   161,   161,
    -127,   112,   376,   142,    42,   160,   174,    29,   183,  -127,
    -127,  -127,  -127,   239,   444,   444,   188,  -127,    95,  -127,
     182,    20,   444,   -13,  -127,   184,   186,    -1,  -127,   192,
    -127,   193,   201,   202,    42,  -127,   204,   119,   159,  -127,
      -6,   320,   535,   320,   545,   444,    42,   275,   266,   290,
     296,    42,   444,   430,   236,  -127,  -127,  -127,    44,   376,
      42,   385,  -127,  -127,  -127,  -127,   252,  -127,  -127,   284,
    -127,    42,   262,     6,   264,   119,   268,    29,   161,  -127,
    -127,    20,   444,   444,   444,   325,   340,   444,   444,   444,
     444,   444,  -127,  -127,   297,  -127,  -127,  -127,   302,   307,
    -127,   147,   444,   317,   175,   147,    42,    42,  -127,   313,
     314,  -127,  -127,  -127,  -127,  -127,   280,  -127,   322,   194,
     194,   194,   194,  -127,   331,  -127,   421,  -127,   332,   188,
     283,   339,   -13,   345,    42,    42,    42,    42,   347,   346,
     344,   595,   585,   603,   444,   444,   123,   123,  -127,  -127,
    -127,    -6,  -127,    23,  -127,   525,   352,  -127,   -13,   367,
     368,  -127,  -127,  -127,    42,  -127,  -127,  -127,  -127,  -127,
    -127,  -127,  -127,  -127,  -127,  -127,   457,  -127,  -127,   371,
    -127,   113,  -127,   393,  -127,   267,   267,   431,  -127,    42,
     147,  -127,   374,    42,  -127,  -127,  -127,  -127,   379,   444,
     378,  -127,    42,  -127,   380,  -127,  -127,   181,   384,    42,
     382,  -127,   390,    42,  -127,   444,   181,   388,   304,   394,
    -127,  -127,   444,  -127,   575
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -127,  -127,  -127,  -126,  -127,  -127,  -127,   552,    -5,   354,
     -56,   399,   -11,  -101,  -127,   -44,   -42,   -17,    10,    -8,
     -29,  -127,   -33,   -47,   -28,   -19,   137,  -127,   -18,    15,
    -102,   278,   -15,  -127,   -26
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -63
static const yytype_int16 yytable[] =
{
      55,    55,   115,   134,    41,   103,   177,   178,   213,   102,
      70,   180,    72,    58,    59,   244,   118,    91,    91,    91,
      74,    66,   145,   105,    91,   222,   111,   224,   133,   116,
      54,    56,   121,   122,   106,   142,   143,   112,   129,     2,
     117,   120,   135,    35,   140,   141,   136,    92,   146,    77,
     152,    40,   154,   113,    79,   165,    84,    85,   149,   205,
     206,   207,   174,    73,    47,   163,   164,    40,   237,   239,
      40,    78,   -62,   186,   131,   157,   158,   130,   159,   103,
     167,    80,   168,   102,    81,   132,   249,    82,    83,    75,
      76,   176,    97,    91,    49,   183,   144,   105,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   147,   189,
     190,   148,   165,   166,    50,   204,    51,   209,   150,   219,
     220,   175,   203,    67,   160,   146,    53,   227,   210,   226,
      47,    48,   233,   199,   200,   201,   103,   228,    93,    94,
     102,   156,   221,   248,    84,    85,   155,   234,   122,   161,
      91,   238,    80,   162,   105,    81,   169,   220,    82,    83,
      49,    81,   242,   202,    82,    83,   251,   252,   253,   124,
     125,   256,   257,   258,   259,   260,   170,    40,   245,   179,
     108,   109,    51,   278,   279,   280,   265,   269,   270,    52,
      47,    48,    53,   264,    93,    94,   267,   268,   205,   206,
     207,    81,    47,    48,    82,    83,   181,   298,    47,    48,
     276,   276,   276,   276,   277,   277,   277,   277,   290,   282,
      49,    72,    93,   266,   184,   287,   183,   183,   295,   296,
     320,   321,    49,    80,   288,   289,    81,   285,    49,    82,
      83,   185,    51,   292,   305,   304,   188,   187,   211,    52,
     212,    73,    53,   297,    51,   191,   214,   215,   171,   109,
      51,    52,   307,   301,    53,   216,   217,    67,    47,    48,
      53,   116,   218,   230,   314,   197,   198,   199,   200,   201,
     310,   229,   311,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   318,   316,    93,    94,   231,    49,   328,
     324,    84,    85,   232,   327,   236,   334,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   137,    50,   240,
      51,     4,   205,   206,   207,   241,   241,    52,   243,   246,
      53,   254,   247,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,   255,   273,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    47,    48,
      80,   261,    80,    81,    28,    81,    82,    83,    82,    83,
     262,   263,   331,   235,    84,    85,    84,    85,    29,    30,
      31,   271,   272,    32,    47,    48,   274,    40,    49,    80,
      73,   123,    81,   124,   125,    82,    83,    93,    94,   281,
     283,    95,    96,    84,    85,    47,    48,   284,    50,   286,
      51,   291,   294,   293,    49,    80,   300,    67,    81,    97,
      53,    82,    83,    93,    94,    47,    48,    95,    96,    47,
      48,   302,   303,   308,    50,    49,    51,   306,    47,    48,
     309,   312,   317,    67,   319,    97,    53,   315,   323,   325,
      84,    85,    47,    48,   326,    49,   330,    51,   332,    49,
     172,   138,   114,   329,    67,    47,    48,    53,    49,   250,
      84,    85,     0,     0,     0,     0,     0,    51,     0,    50,
       0,    51,    49,     0,    52,     0,   235,    53,    67,     0,
      51,    53,     0,     0,     0,    49,     0,    52,     0,     0,
      53,     0,     0,     0,    51,     0,    80,     0,     0,    81,
       0,    52,    82,    83,    53,     0,     0,    51,     0,     0,
      84,    85,     0,     0,    67,     0,     0,    53,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,    37,    38,
      39,     0,    42,    43,    44,    45,    46,     0,     0,    57,
       0,     0,    60,    61,    62,    63,    64,    65,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,     0,   299,
     194,   195,   196,   197,   198,   199,   200,   201,   223,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   225,   195,
     196,   197,   198,   199,   200,   201
};

static const yytype_int16 yycheck[] =
{
      15,    16,    44,    59,     9,    38,   108,   109,     9,    38,
      27,   112,    27,    18,    19,     9,    45,    36,    37,    38,
      28,    26,    66,    38,    43,   151,    41,   153,    57,    44,
      15,    16,    47,    48,    39,    64,    65,    42,    53,     0,
      45,    46,    60,    63,    62,    63,    61,    37,    67,    61,
      76,    64,    78,    43,    63,    56,    53,    54,    73,    39,
      40,    41,   106,    69,     8,     9,    10,    64,   169,   171,
      64,    62,    66,   117,    67,     6,     7,    64,     9,   112,
      97,    39,    97,   112,    42,    11,   188,    45,    46,    61,
      62,   106,    69,   112,    38,   114,    67,   112,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    67,   124,
     125,    67,    56,    57,    58,   130,    60,   132,    64,   147,
     148,   106,   130,    67,    55,   144,    70,   156,   133,   155,
       8,     9,   161,    10,    11,    12,   169,   156,    47,    48,
     169,    64,   150,   187,    53,    54,    67,   162,   163,    64,
     169,   170,    39,    67,   169,    42,    64,   185,    45,    46,
      38,    42,   181,    68,    45,    46,   192,   193,   194,     8,
       9,   197,   198,   199,   200,   201,    67,    64,   183,    67,
      58,    59,    60,   230,   231,   232,   212,   216,   217,    67,
       8,     9,    70,   211,    47,    48,   214,   215,    39,    40,
      41,    42,     8,     9,    45,    46,    64,   263,     8,     9,
     229,   230,   231,   232,   229,   230,   231,   232,   247,   236,
      38,   236,    47,    48,    64,   244,   245,   246,   254,   255,
      49,    50,    38,    39,   245,   246,    42,   242,    38,    45,
      46,    67,    60,   248,   286,   274,     7,    64,    64,    67,
      64,    69,    70,   261,    60,    67,    64,    64,    58,    59,
      60,    67,   291,   268,    70,    64,    64,    67,     8,     9,
      70,   286,    68,     7,   303,     8,     9,    10,    11,    12,
     299,     6,   300,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,   312,   309,    47,    48,     7,    38,   325,
     319,    53,    54,     7,   323,    69,   332,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    69,    58,    67,
      60,     1,    39,    40,    41,    42,    42,    67,    66,    65,
      70,     6,    64,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,     7,    68,    29,
      30,    31,    32,    33,    34,    35,    36,    37,     8,     9,
      39,    64,    39,    42,    44,    42,    45,    46,    45,    46,
      68,    64,    68,    56,    53,    54,    53,    54,    58,    59,
      60,    68,    68,    63,     8,     9,    64,    64,    38,    39,
      69,     6,    42,     8,     9,    45,    46,    47,    48,    68,
      68,    51,    52,    53,    54,     8,     9,    68,    58,    64,
      60,    64,    68,    67,    38,    39,    64,    67,    42,    69,
      70,    45,    46,    47,    48,     8,     9,    51,    52,     8,
       9,    64,    64,    40,    58,    38,    60,    66,     8,     9,
       9,    67,    64,    67,    64,    69,    70,    68,    64,    67,
      53,    54,     8,     9,    64,    38,    68,    60,    64,    38,
     106,    62,    65,   326,    67,     8,     9,    70,    38,   191,
      53,    54,    -1,    -1,    -1,    -1,    -1,    60,    -1,    58,
      -1,    60,    38,    -1,    67,    -1,    56,    70,    67,    -1,
      60,    70,    -1,    -1,    -1,    38,    -1,    67,    -1,    -1,
      70,    -1,    -1,    -1,    60,    -1,    39,    -1,    -1,    42,
      -1,    67,    45,    46,    70,    -1,    -1,    60,    -1,    -1,
      53,    54,    -1,    -1,    67,    -1,    -1,    70,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,     6,     7,
       8,    -1,    10,    11,    12,    13,    14,    -1,    -1,    17,
      -1,    -1,    20,    21,    22,    23,    24,    25,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    -1,    64,
       5,     6,     7,     8,     9,    10,    11,    12,    63,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    63,     6,
       7,     8,     9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    72,     0,    73,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    44,    58,
      59,    60,    63,    74,    77,    63,    78,    78,    78,    78,
      64,    79,    78,    78,    78,    78,    78,     8,     9,    38,
      58,    60,    67,    70,   100,   103,   100,    78,    79,    79,
      78,    78,    78,    78,    78,    78,    79,    67,    86,    87,
      88,   100,   103,    69,    90,    61,    62,    61,    62,    63,
      39,    42,    45,    46,    53,    54,    89,    90,    91,    93,
      95,    96,    89,    47,    48,    51,    52,    69,    81,    82,
      84,    88,    91,    93,    99,   103,    79,    79,    58,    59,
      80,   103,    79,    89,    65,    87,   103,    79,    91,    96,
      79,   103,   103,     6,     8,     9,   101,   103,   105,   103,
      64,    67,    11,    91,    81,    99,   103,    69,    82,    98,
      99,    99,    91,    91,    67,    86,    96,    67,    67,   103,
      64,    76,   105,    75,   105,    67,    64,     6,     7,     9,
      55,    64,    67,     9,    10,    56,    57,    88,   103,    64,
      67,    58,    80,    85,    86,   100,   103,   101,   101,    67,
      84,    64,    83,    96,    64,    67,    86,    64,     7,   103,
     103,    67,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    68,    90,   103,    39,    40,    41,   102,   103,
      79,    64,    64,     9,    64,    64,    64,    64,    68,    95,
      95,    90,    74,    63,    74,    63,   105,    91,    96,     6,
       7,     7,     7,    91,   103,    56,    69,    84,    96,   101,
      67,    42,    96,    66,     9,    79,    65,    64,    86,   101,
     102,   105,   105,   105,     6,     7,   105,   105,   105,   105,
     105,    64,    68,    64,    99,   105,    48,    99,    99,    91,
      91,    68,    68,    68,    64,    94,    96,   103,    94,    94,
      94,    68,    88,    68,    68,    79,    64,    96,    83,    83,
      91,    64,    79,    67,    68,   105,   105,    90,    81,    64,
      64,    79,    64,    64,    91,    87,    66,    91,    40,     9,
      96,    99,    67,    92,    91,    68,   103,    64,    96,    64,
      49,    50,    97,    64,    96,    67,    64,    96,   105,    97,
      68,    68,    64,   104,   105
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
        case 3:
#line 69 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:
#line 76 "a.y"
    {
		if((yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->name);
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 7:
#line 83 "a.y"
    {
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 9:
#line 89 "a.y"
    {
		(yyvsp[(1) - (4)].sym)->type = LVAR;
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 10:
#line 94 "a.y"
    {
		if((yyvsp[(1) - (4)].sym)->value != (yyvsp[(3) - (4)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (4)].sym)->name);
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 14:
#line 108 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].gen), (yyvsp[(5) - (7)].lval), &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 15:
#line 112 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].gen), (yyvsp[(5) - (6)].lval), &nullgen);
	}
    break;

  case 16:
#line 116 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 17:
#line 123 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 18:
#line 130 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 19:
#line 137 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].gen));
	}
    break;

  case 20:
#line 141 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].gen));
	}
    break;

  case 21:
#line 148 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].gen));
	}
    break;

  case 22:
#line 155 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].gen));
	}
    break;

  case 23:
#line 162 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].gen));
	}
    break;

  case 24:
#line 169 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].gen), (yyvsp[(5) - (6)].lval), &nullgen);
	}
    break;

  case 25:
#line 176 "a.y"
    {
		Gen g;

		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(6) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].gen), NREG, &g);
	}
    break;

  case 26:
#line 185 "a.y"
    {
		Gen g;

		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(4) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &g, NREG, &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 27:
#line 197 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(5) - (7)].gen), (yyvsp[(3) - (7)].gen).reg, &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 28:
#line 201 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(5) - (6)].gen), (yyvsp[(3) - (6)].gen).reg, &(yyvsp[(3) - (6)].gen));
	}
    break;

  case 29:
#line 205 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(4) - (6)].gen), (yyvsp[(6) - (6)].gen).reg, &(yyvsp[(6) - (6)].gen));
	}
    break;

  case 30:
#line 212 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), (yyvsp[(2) - (3)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 31:
#line 219 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].gen), 0, &(yyvsp[(4) - (4)].gen));
	}
    break;

  case 32:
#line 223 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].gen), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].gen));
	}
    break;

  case 33:
#line 227 "a.y"
    {
		// Change explicit 0 argument size to 1
		// so that we can distinguish it from missing.
		if((yyvsp[(8) - (8)].lval) == 0)
			(yyvsp[(8) - (8)].lval) = 1;
		(yyvsp[(6) - (8)].gen).type = D_CONST2;
		(yyvsp[(6) - (8)].gen).offset2 = (yyvsp[(8) - (8)].lval);
		outcode((yyvsp[(1) - (8)].lval), Always, &(yyvsp[(2) - (8)].gen), (yyvsp[(4) - (8)].lval), &(yyvsp[(6) - (8)].gen));
	}
    break;

  case 34:
#line 240 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].gen), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].gen));
	}
    break;

  case 35:
#line 247 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &(yyvsp[(3) - (4)].gen), NREG, &nullgen);
	}
    break;

  case 36:
#line 254 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].gen));
	}
    break;

  case 37:
#line 261 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 38:
#line 265 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 39:
#line 269 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].gen), (yyvsp[(5) - (7)].lval), &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 40:
#line 273 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].gen), (yyvsp[(5) - (6)].gen).reg, &nullgen);
	}
    break;

  case 41:
#line 280 "a.y"
    {
		Gen g;

		g = nullgen;
		g.type = D_CONST;
		g.offset =
			(0xe << 24) |		/* opcode */
			((yyvsp[(1) - (12)].lval) << 20) |		/* MCR/MRC */
			((yyvsp[(2) - (12)].lval) << 28) |		/* scond */
			(((yyvsp[(3) - (12)].lval) & 15) << 8) |	/* coprocessor number */
			(((yyvsp[(5) - (12)].lval) & 7) << 21) |	/* coprocessor operation */
			(((yyvsp[(7) - (12)].lval) & 15) << 12) |	/* arm register */
			(((yyvsp[(9) - (12)].lval) & 15) << 16) |	/* Crn */
			(((yyvsp[(11) - (12)].lval) & 15) << 0) |	/* Crm */
			(((yyvsp[(12) - (12)].lval) & 7) << 5) |	/* coprocessor information */
			(1<<4);			/* must be set */
		outcode(AWORD, Always, &nullgen, NREG, &g);
	}
    break;

  case 42:
#line 302 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].gen), (yyvsp[(5) - (7)].gen).reg, &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 43:
#line 310 "a.y"
    {
		(yyvsp[(7) - (9)].gen).type = D_REGREG2;
		(yyvsp[(7) - (9)].gen).offset = (yyvsp[(9) - (9)].lval);
		outcode((yyvsp[(1) - (9)].lval), (yyvsp[(2) - (9)].lval), &(yyvsp[(3) - (9)].gen), (yyvsp[(5) - (9)].gen).reg, &(yyvsp[(7) - (9)].gen));
	}
    break;

  case 44:
#line 319 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &(yyvsp[(2) - (2)].gen), NREG, &nullgen);
	}
    break;

  case 45:
#line 326 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].gen), NREG, &(yyvsp[(4) - (4)].gen));
	}
    break;

  case 46:
#line 333 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &nullgen, NREG, &nullgen);
	}
    break;

  case 47:
#line 338 "a.y"
    {
		(yyval.lval) = Always;
	}
    break;

  case 48:
#line 342 "a.y"
    {
		(yyval.lval) = ((yyvsp[(1) - (2)].lval) & ~C_SCOND) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 49:
#line 346 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (2)].lval) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 52:
#line 355 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 53:
#line 361 "a.y"
    {
		(yyval.gen) = nullgen;
		if(pass == 2)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->name);
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 54:
#line 370 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 55:
#line 378 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST;
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 56:
#line 384 "a.y"
    {
		(yyval.gen) = (yyvsp[(2) - (2)].gen);
		(yyval.gen).type = D_CONST;
	}
    break;

  case 57:
#line 389 "a.y"
    {
		(yyval.gen) = (yyvsp[(4) - (4)].gen);
		(yyval.gen).type = D_OCONST;
	}
    break;

  case 58:
#line 394 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SCONST;
		memcpy((yyval.gen).sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.gen).sval));
	}
    break;

  case 60:
#line 403 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 61:
#line 409 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 62:
#line 417 "a.y"
    {
		(yyval.lval) = 1 << (yyvsp[(1) - (1)].lval);
	}
    break;

  case 63:
#line 421 "a.y"
    {
		int i;
		(yyval.lval)=0;
		for(i=(yyvsp[(1) - (3)].lval); i<=(yyvsp[(3) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
		for(i=(yyvsp[(3) - (3)].lval); i<=(yyvsp[(1) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
	}
    break;

  case 64:
#line 430 "a.y"
    {
		(yyval.lval) = (1<<(yyvsp[(1) - (3)].lval)) | (yyvsp[(3) - (3)].lval);
	}
    break;

  case 68:
#line 439 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (4)].gen);
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 69:
#line 444 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_PSR;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 70:
#line 450 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FPCR;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 71:
#line 456 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 75:
#line 467 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (1)].gen);
		if((yyvsp[(1) - (1)].gen).name != D_EXTERN && (yyvsp[(1) - (1)].gen).name != D_STATIC) {
		}
	}
    break;

  case 76:
#line 475 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).reg = (yyvsp[(2) - (3)].lval);
		(yyval.gen).offset = 0;
	}
    break;

  case 78:
#line 485 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 80:
#line 495 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (4)].gen);
		(yyval.gen).type = D_OREG;
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 85:
#line 508 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST;
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 86:
#line 516 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_REG;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 87:
#line 524 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_REGREG;
		(yyval.gen).reg = (yyvsp[(2) - (5)].lval);
		(yyval.gen).offset = (yyvsp[(4) - (5)].lval);
	}
    break;

  case 88:
#line 533 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (0 << 5);
	}
    break;

  case 89:
#line 539 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (1 << 5);
	}
    break;

  case 90:
#line 545 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (2 << 5);
	}
    break;

  case 91:
#line 551 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (3 << 5);
	}
    break;

  case 92:
#line 559 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 16)
			print("register value out of range\n");
		(yyval.lval) = (((yyvsp[(1) - (1)].lval)&15) << 8) | (1 << 4);
	}
    break;

  case 93:
#line 565 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 32)
			print("shift value out of range\n");
		(yyval.lval) = ((yyvsp[(1) - (1)].lval)&31) << 7;
	}
    break;

  case 95:
#line 574 "a.y"
    {
		(yyval.lval) = REGPC;
	}
    break;

  case 96:
#line 578 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 98:
#line 587 "a.y"
    {
		(yyval.lval) = REGSP;
	}
    break;

  case 100:
#line 594 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 103:
#line 606 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FREG;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 104:
#line 612 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FREG;
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 105:
#line 620 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).name = (yyvsp[(3) - (4)].lval);
		(yyval.gen).sym = S;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 106:
#line 628 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).name = (yyvsp[(4) - (5)].lval);
		(yyval.gen).sym = (yyvsp[(1) - (5)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 107:
#line 636 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).name = D_STATIC;
		(yyval.gen).sym = (yyvsp[(1) - (7)].sym);
		(yyval.gen).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 108:
#line 645 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 109:
#line 649 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 110:
#line 653 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 115:
#line 665 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 116:
#line 669 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 117:
#line 673 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 118:
#line 677 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 119:
#line 681 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 120:
#line 686 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 121:
#line 690 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 123:
#line 697 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 124:
#line 701 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:
#line 705 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:
#line 709 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 127:
#line 713 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:
#line 717 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 129:
#line 721 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 130:
#line 725 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 131:
#line 729 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 132:
#line 733 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2569 "y.tab.c"
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



