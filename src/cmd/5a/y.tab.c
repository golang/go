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
     LTYPEG = 272,
     LTYPEH = 273,
     LTYPEI = 274,
     LTYPEJ = 275,
     LTYPEK = 276,
     LTYPEL = 277,
     LTYPEM = 278,
     LTYPEN = 279,
     LTYPEBX = 280,
     LTYPEPLD = 281,
     LCONST = 282,
     LSP = 283,
     LSB = 284,
     LFP = 285,
     LPC = 286,
     LTYPEX = 287,
     LTYPEPC = 288,
     LTYPEF = 289,
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
#define LTYPEG 272
#define LTYPEH 273
#define LTYPEI 274
#define LTYPEJ 275
#define LTYPEK 276
#define LTYPEL 277
#define LTYPEM 278
#define LTYPEN 279
#define LTYPEBX 280
#define LTYPEPLD 281
#define LCONST 282
#define LSP 283
#define LSB 284
#define LFP 285
#define LPC 286
#define LTYPEX 287
#define LTYPEPC 288
#define LTYPEF 289
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
#include "../../runtime/funcdata.h"


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
#line 39 "a.y"
{
	Sym	*sym;
	int32	lval;
	double	dval;
	char	sval[8];
	Addr	addr;
}
/* Line 193 of yacc.c.  */
#line 212 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 225 "y.tab.c"

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
#define YYLAST   624

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  71
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  34
/* YYNRULES -- Number of rules.  */
#define YYNRULES  129
/* YYNRULES -- Number of states.  */
#define YYNSTATES  330

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
       0,     0,     3,     4,     5,     9,    10,    15,    20,    25,
      27,    30,    33,    41,    48,    54,    60,    66,    71,    76,
      80,    84,    89,    96,   104,   112,   120,   127,   134,   138,
     143,   150,   159,   166,   171,   175,   181,   187,   195,   202,
     215,   223,   233,   236,   241,   246,   249,   250,   253,   256,
     257,   260,   265,   268,   271,   274,   277,   279,   282,   286,
     288,   292,   296,   298,   300,   302,   307,   309,   311,   313,
     315,   317,   319,   321,   325,   327,   332,   334,   339,   341,
     343,   345,   347,   350,   352,   358,   363,   368,   373,   378,
     380,   382,   384,   386,   391,   393,   395,   397,   402,   404,
     406,   408,   413,   418,   424,   432,   433,   436,   439,   441,
     443,   445,   447,   449,   452,   455,   458,   462,   463,   466,
     468,   472,   476,   480,   484,   488,   493,   498,   502,   506
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      72,     0,    -1,    -1,    -1,    72,    73,    74,    -1,    -1,
      58,    61,    75,    74,    -1,    58,    62,   104,    63,    -1,
      60,    62,   104,    63,    -1,    63,    -1,    76,    63,    -1,
       1,    63,    -1,    13,    77,    88,    64,    95,    64,    90,
      -1,    13,    77,    88,    64,    95,    64,    -1,    13,    77,
      88,    64,    90,    -1,    14,    77,    88,    64,    90,    -1,
      15,    77,    83,    64,    83,    -1,    16,    77,    78,    79,
      -1,    16,    77,    78,    84,    -1,    35,    78,    85,    -1,
      17,    78,    79,    -1,    18,    77,    78,    83,    -1,    19,
      77,    88,    64,    95,    78,    -1,    20,    77,    86,    64,
      65,    82,    66,    -1,    20,    77,    65,    82,    66,    64,
      86,    -1,    21,    77,    90,    64,    85,    64,    90,    -1,
      21,    77,    90,    64,    85,    78,    -1,    21,    77,    78,
      85,    64,    90,    -1,    22,    77,    78,    -1,    23,    99,
      64,    89,    -1,    23,    99,    64,   102,    64,    89,    -1,
      23,    99,    64,   102,    64,    89,     9,   102,    -1,    24,
      99,    11,   102,    64,    80,    -1,    25,    77,    90,    78,
      -1,    28,    78,    80,    -1,    29,    77,    98,    64,    98,
      -1,    31,    77,    97,    64,    98,    -1,    31,    77,    97,
      64,    48,    64,    98,    -1,    32,    77,    98,    64,    98,
      78,    -1,    30,    77,   102,    64,   104,    64,    95,    64,
      96,    64,    96,   103,    -1,    33,    77,    90,    64,    90,
      64,    91,    -1,    34,    77,    90,    64,    90,    64,    90,
      64,    95,    -1,    36,    87,    -1,    43,    83,    64,    83,
      -1,    44,    83,    64,    83,    -1,    26,    78,    -1,    -1,
      77,    53,    -1,    77,    54,    -1,    -1,    64,    78,    -1,
     102,    67,    41,    68,    -1,    58,   100,    -1,    69,   102,
      -1,    69,    87,    -1,    69,    57,    -1,    81,    -1,    69,
      56,    -1,    69,     9,    56,    -1,    95,    -1,    95,     9,
      95,    -1,    95,    78,    82,    -1,    90,    -1,    80,    -1,
      92,    -1,    92,    67,    95,    68,    -1,    51,    -1,    52,
      -1,   102,    -1,    87,    -1,    98,    -1,    85,    -1,    99,
      -1,    67,    95,    68,    -1,    85,    -1,   102,    67,    94,
      68,    -1,    99,    -1,    99,    67,    94,    68,    -1,    86,
      -1,    90,    -1,    89,    -1,    92,    -1,    69,   102,    -1,
      95,    -1,    67,    95,    64,    95,    68,    -1,    95,     6,
       6,    93,    -1,    95,     7,     7,    93,    -1,    95,     9,
       7,    93,    -1,    95,    55,     7,    93,    -1,    95,    -1,
     102,    -1,    46,    -1,    41,    -1,    45,    67,   104,    68,
      -1,    94,    -1,    38,    -1,    50,    -1,    49,    67,   104,
      68,    -1,    98,    -1,    81,    -1,    48,    -1,    47,    67,
     102,    68,    -1,   102,    67,   101,    68,    -1,    58,   100,
      67,   101,    68,    -1,    58,     6,     7,   100,    67,    39,
      68,    -1,    -1,     8,   102,    -1,     9,   102,    -1,    39,
      -1,    38,    -1,    40,    -1,    37,    -1,    60,    -1,     9,
     102,    -1,     8,   102,    -1,    70,   102,    -1,    67,   104,
      68,    -1,    -1,    64,   104,    -1,   102,    -1,   104,     8,
     104,    -1,   104,     9,   104,    -1,   104,    10,   104,    -1,
     104,    11,   104,    -1,   104,    12,   104,    -1,   104,     6,
       6,   104,    -1,   104,     7,     7,   104,    -1,   104,     5,
     104,    -1,   104,     4,   104,    -1,   104,     3,   104,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    68,    68,    70,    69,    77,    76,    85,    90,    96,
      97,    98,   104,   108,   112,   119,   126,   133,   137,   144,
     151,   158,   165,   172,   181,   193,   197,   201,   208,   215,
     222,   229,   239,   246,   253,   260,   264,   268,   272,   279,
     301,   309,   318,   325,   334,   345,   351,   354,   358,   363,
     364,   367,   373,   383,   389,   394,   400,   403,   409,   417,
     421,   430,   436,   437,   438,   439,   444,   450,   456,   462,
     463,   466,   467,   475,   484,   485,   494,   495,   501,   504,
     505,   506,   508,   516,   524,   533,   539,   545,   551,   559,
     565,   573,   574,   578,   586,   587,   593,   594,   602,   603,
     606,   612,   620,   628,   636,   646,   649,   653,   659,   660,
     661,   664,   665,   669,   673,   677,   681,   687,   690,   696,
     697,   701,   705,   709,   713,   717,   721,   725,   729,   733
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
  "LTYPEC", "LTYPED", "LTYPEE", "LTYPEG", "LTYPEH", "LTYPEI", "LTYPEJ",
  "LTYPEK", "LTYPEL", "LTYPEM", "LTYPEN", "LTYPEBX", "LTYPEPLD", "LCONST",
  "LSP", "LSB", "LFP", "LPC", "LTYPEX", "LTYPEPC", "LTYPEF", "LR", "LREG",
  "LF", "LFREG", "LC", "LCREG", "LPSR", "LFCR", "LCOND", "LS", "LAT",
  "LFCONST", "LSCONST", "LNAME", "LLAB", "LVAR", "':'", "'='", "';'",
  "','", "'['", "']'", "'('", "')'", "'$'", "'~'", "$accept", "prog", "@1",
  "line", "@2", "inst", "cond", "comma", "rel", "ximm", "fcon", "reglist",
  "gen", "nireg", "ireg", "ioreg", "oreg", "imsr", "imm", "reg", "regreg",
  "shift", "rcon", "sreg", "spreg", "creg", "frcon", "freg", "name",
  "offset", "pointer", "con", "oexpr", "expr", 0
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
       0,    71,    72,    73,    72,    75,    74,    74,    74,    74,
      74,    74,    76,    76,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    76,    77,    77,    77,    78,
      78,    79,    79,    80,    80,    80,    80,    81,    81,    82,
      82,    82,    83,    83,    83,    83,    83,    83,    83,    83,
      83,    84,    84,    85,    86,    86,    87,    87,    87,    88,
      88,    88,    89,    90,    91,    92,    92,    92,    92,    93,
      93,    94,    94,    94,    95,    95,    96,    96,    97,    97,
      98,    98,    99,    99,    99,   100,   100,   100,   101,   101,
     101,   102,   102,   102,   102,   102,   102,   103,   103,   104,
     104,   104,   104,   104,   104,   104,   104,   104,   104,   104
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     4,     4,     1,
       2,     2,     7,     6,     5,     5,     5,     4,     4,     3,
       3,     4,     6,     7,     7,     7,     6,     6,     3,     4,
       6,     8,     6,     4,     3,     5,     5,     7,     6,    12,
       7,     9,     2,     4,     4,     2,     0,     2,     2,     0,
       2,     4,     2,     2,     2,     2,     1,     2,     3,     1,
       3,     3,     1,     1,     1,     4,     1,     1,     1,     1,
       1,     1,     1,     3,     1,     4,     1,     4,     1,     1,
       1,     1,     2,     1,     5,     4,     4,     4,     4,     1,
       1,     1,     1,     4,     1,     1,     1,     4,     1,     1,
       1,     4,     4,     5,     7,     0,     2,     2,     1,     1,
       1,     1,     1,     2,     2,     2,     3,     0,     2,     1,
       3,     3,     3,     3,     3,     4,     4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    46,    46,    46,    46,    49,
      46,    46,    46,    46,    46,     0,     0,    46,    49,    49,
      46,    46,    46,    46,    46,    46,    49,     0,     0,     0,
       0,     0,     9,     4,     0,    11,     0,     0,     0,    49,
      49,     0,    49,     0,     0,    49,    49,     0,     0,   111,
     105,   112,     0,     0,     0,     0,     0,     0,    45,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    74,    78,
      42,    76,     0,    95,    92,     0,    91,     0,   100,    66,
      67,     0,    63,    56,     0,    69,    62,    64,    94,    83,
      70,    68,     0,     5,     0,     0,    10,    47,    48,     0,
       0,    80,    79,    81,     0,     0,     0,    50,   105,    20,
       0,     0,     0,     0,     0,     0,     0,     0,    83,    28,
     114,   113,     0,     0,     0,     0,   119,     0,   115,     0,
       0,     0,    49,    34,     0,     0,     0,    99,     0,    98,
       0,     0,     0,     0,    19,     0,     0,     0,     0,     0,
       0,    57,    55,    54,    53,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    82,     0,     0,     0,   105,
      17,    18,    71,    72,     0,    52,     0,    21,     0,     0,
      49,     0,     0,     0,     0,   105,   106,   107,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   116,
      29,     0,   109,   108,   110,     0,     0,    33,     0,     0,
       0,     0,     0,     0,     0,    73,     0,     0,     0,     0,
      58,    43,     0,     0,     0,     0,     0,    44,     6,     7,
       8,    14,    83,    15,    16,    52,     0,     0,    49,     0,
       0,     0,     0,     0,    49,     0,     0,   129,   128,   127,
       0,     0,   120,   121,   122,   123,   124,     0,   102,     0,
      35,     0,   100,    36,    49,     0,     0,    77,    75,    93,
     101,    65,    85,    89,    90,    86,    87,    88,    13,    51,
      22,     0,    60,    61,     0,    27,    49,    26,     0,   103,
     125,   126,    30,    32,     0,     0,    38,     0,     0,    12,
      24,    23,    25,     0,     0,     0,    37,     0,    40,     0,
     104,    31,     0,     0,     0,     0,    96,     0,     0,    41,
       0,     0,     0,     0,   117,    84,    97,     0,    39,   118
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    33,   162,    34,    36,   107,   109,    82,
      83,   179,    84,   171,    68,    69,    85,   100,   101,    86,
     308,    87,   272,    88,   118,   317,   138,    90,    71,   125,
     205,   126,   328,   127
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -124
static const yytype_int16 yypact[] =
{
    -124,     5,  -124,   329,   -61,  -124,  -124,  -124,  -124,   -57,
    -124,  -124,  -124,  -124,  -124,   157,   157,  -124,   -57,   -57,
    -124,  -124,  -124,  -124,  -124,  -124,   -57,   264,   385,   385,
     -31,   -22,  -124,  -124,   -12,  -124,   512,   512,   358,   -16,
     -57,   268,   -16,   512,   191,   337,   -16,   432,   432,  -124,
     164,  -124,   432,   432,     0,     9,    58,   450,  -124,    15,
     172,    49,   235,   172,   450,   450,    11,    34,  -124,  -124,
    -124,    14,    21,  -124,  -124,    32,  -124,    46,  -124,  -124,
    -124,    87,  -124,  -124,    59,  -124,  -124,    69,  -124,    91,
    -124,    21,    75,  -124,   432,   432,  -124,  -124,  -124,   432,
      85,  -124,  -124,  -124,   129,   141,   405,  -124,    37,  -124,
      84,   385,   147,    80,   152,   151,    11,   178,  -124,  -124,
    -124,  -124,   236,   432,   432,   180,  -124,   173,  -124,   411,
     231,   432,   -57,  -124,   185,   186,    18,  -124,   189,  -124,
     193,   200,   201,    80,  -124,   199,   107,   471,   432,   432,
     430,  -124,  -124,  -124,    21,   385,    80,   269,   273,   274,
     277,   385,   329,   526,   536,  -124,    80,    80,   385,   164,
    -124,  -124,  -124,  -124,   218,  -124,   261,  -124,    80,   243,
      56,   260,   107,   263,    11,    37,  -124,  -124,   231,   432,
     432,   432,   323,   326,   432,   432,   432,   432,   432,  -124,
    -124,   272,  -124,  -124,  -124,   303,   276,  -124,    44,   432,
     281,    63,    44,    80,    80,  -124,   306,   308,   288,   311,
    -124,  -124,   312,    34,    34,    34,    34,  -124,  -124,  -124,
    -124,  -124,   317,  -124,  -124,   180,   148,   316,   -57,   321,
      80,    80,    80,    80,   322,   330,   320,   586,   605,   612,
     432,   432,   296,   296,  -124,  -124,  -124,   331,  -124,    15,
    -124,   516,   334,  -124,   -57,   338,   343,  -124,  -124,  -124,
    -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,    80,  -124,
    -124,   448,  -124,  -124,   275,  -124,   214,  -124,   369,  -124,
     225,   225,   406,  -124,    80,    44,  -124,   350,    80,  -124,
    -124,  -124,  -124,   353,   432,   360,  -124,    80,  -124,   365,
    -124,  -124,    82,   370,    80,   368,  -124,   380,    80,  -124,
     432,    82,   378,   309,   383,  -124,  -124,   432,  -124,   597
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -124,  -124,  -124,   287,  -124,  -124,   562,    10,   344,   -55,
     389,   -19,     3,  -124,   -43,   -41,   -15,   -17,  -123,    25,
    -124,   132,   144,  -122,   -28,   137,  -124,   -49,     2,   -92,
     265,     6,  -124,    12
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -60
static const yytype_int16 yytable[] =
{
      89,    89,    35,   114,   133,     2,   200,    40,    89,    89,
      89,   134,    70,   139,   140,    89,   175,    54,    56,    41,
     104,    55,    55,   144,   216,   217,   112,   210,    58,    59,
      93,    94,    92,    72,    91,    91,    66,    97,    98,   145,
      95,   105,    47,    48,    91,   123,   124,   110,    40,   106,
     115,    96,   111,   120,   121,   116,   119,    47,    48,   128,
     217,   102,   102,   172,   129,   240,   153,   135,   102,   131,
     117,    49,    73,   183,   151,    74,   130,   235,   143,    75,
      76,   146,   132,    89,    81,   180,    49,   154,   147,   141,
     142,    77,    78,   245,    51,    47,   150,   157,   158,   148,
     159,    52,    97,    98,    53,   165,   163,   164,   173,    51,
      77,   262,   174,   149,   177,   145,    52,    91,    73,    53,
      40,    74,   -59,   155,    49,    75,    76,    89,   222,   186,
     187,   315,   316,    89,   292,   201,   156,   206,   232,   161,
      89,   244,   207,   151,   152,    50,   160,    51,    74,   166,
     238,   176,    75,    76,    67,   219,   121,    53,   221,   260,
     218,    91,   263,   264,   227,    47,    48,    91,   103,   103,
     122,   234,   123,   124,    91,   103,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   202,   203,   204,   237,
     241,   231,   233,   167,    49,   273,   273,   273,   273,    47,
      48,   247,   248,   249,   293,   168,   252,   253,   254,   255,
     256,   178,   282,   180,   180,    50,   181,    51,   182,    77,
      78,   261,   283,   284,    52,    97,    98,    53,    49,   274,
     274,   274,   274,   194,   195,   196,   197,   198,   265,   266,
     300,   199,   184,   185,    97,    98,   306,   188,   280,   208,
     209,    51,    73,   211,   287,    74,   113,   212,    67,    75,
      76,    53,   290,   291,   213,   214,   305,   215,   285,   202,
     203,   204,    47,    48,   296,   223,    47,    48,    40,   313,
     224,   225,    77,    78,   226,   236,   319,   115,    97,    98,
     322,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     198,    49,   237,   299,   136,    49,   196,   197,   198,   239,
     311,   302,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,    50,   309,    51,   242,   108,   243,    51,   250,
       4,    67,   323,   251,    53,    52,   257,   220,    53,   329,
     259,   301,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,   269,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    47,    48,   275,   276,
     277,   258,    28,    29,   267,    73,   268,   326,    74,   270,
     271,   278,    75,    76,   279,   281,   286,    30,   289,    31,
      97,    98,    32,    47,    48,    49,    73,   288,   295,    74,
      99,    40,   297,    75,    76,    77,    78,   298,   303,    79,
      80,    97,    98,    47,    48,   304,    50,   307,    51,    47,
      48,   310,    49,    73,   312,    67,    74,    81,    53,   314,
      75,    76,    77,    78,   318,   320,    79,    80,    47,    48,
      47,    48,    49,    50,   321,    51,   325,   327,    49,   228,
     170,   137,    67,   246,    81,    53,    47,    48,   324,     0,
       0,     0,     0,   169,     0,    51,     0,    49,     0,    49,
       0,    51,    67,     0,     0,    53,     0,     0,    52,     0,
      99,    53,     0,     0,     0,    49,   220,     0,    73,     0,
      51,    74,    51,     0,     0,    75,    76,    52,     0,    52,
      53,     0,    53,    97,    98,     0,     0,     0,    51,   202,
     203,   204,    74,     0,     0,    67,    75,    76,    53,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   198,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   198,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   198,     0,
      73,     0,     0,    74,     0,     0,     0,    75,    76,     0,
       0,     0,     0,     0,     0,    97,    98,     0,    37,    38,
      39,     0,    42,    43,    44,    45,    46,     0,     0,    57,
     294,    99,    60,    61,    62,    63,    64,    65,     0,   229,
     190,   191,   192,   193,   194,   195,   196,   197,   198,   230,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     191,   192,   193,   194,   195,   196,   197,   198,   192,   193,
     194,   195,   196,   197,   198
};

static const yytype_int16 yycheck[] =
{
      28,    29,    63,    44,    59,     0,   129,    64,    36,    37,
      38,    60,    27,    62,    63,    43,   108,    15,    16,     9,
      37,    15,    16,    66,   146,   147,    43,     9,    18,    19,
      61,    62,    29,    27,    28,    29,    26,    53,    54,    67,
      62,    38,     8,     9,    38,     8,     9,    41,    64,    39,
      44,    63,    42,    47,    48,    45,    46,     8,     9,    53,
     182,    36,    37,   106,    64,     9,    81,    61,    43,    11,
      45,    37,    38,   116,    56,    41,    67,   169,    67,    45,
      46,    67,    57,   111,    69,   113,    37,    81,    67,    64,
      65,    47,    48,   185,    60,     8,     9,     6,     7,    67,
       9,    67,    53,    54,    70,    99,    94,    95,   106,    60,
      47,    48,   106,    67,   111,   143,    67,   111,    38,    70,
      64,    41,    66,    64,    37,    45,    46,   155,   156,   123,
     124,    49,    50,   161,   257,   129,    67,   131,   166,    64,
     168,   184,   132,    56,    57,    58,    55,    60,    41,    64,
     178,    67,    45,    46,    67,   149,   150,    70,   155,   208,
     148,   155,   211,   212,   161,     8,     9,   161,    36,    37,
       6,   168,     8,     9,   168,    43,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    38,    39,    40,    41,
     180,   166,   167,    64,    37,   223,   224,   225,   226,     8,
       9,   189,   190,   191,   259,    64,   194,   195,   196,   197,
     198,    64,   240,   241,   242,    58,    64,    60,    67,    47,
      48,   209,   241,   242,    67,    53,    54,    70,    37,   223,
     224,   225,   226,     8,     9,    10,    11,    12,   213,   214,
     281,    68,    64,     7,    53,    54,   295,    67,   238,    64,
      64,    60,    38,    64,   244,    41,    65,    64,    67,    45,
      46,    70,   250,   251,    64,    64,   294,    68,   243,    38,
      39,    40,     8,     9,   264,     6,     8,     9,    64,   307,
       7,     7,    47,    48,     7,    67,   314,   281,    53,    54,
     318,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    37,    41,   278,    69,    37,    10,    11,    12,    66,
     304,   286,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    58,   298,    60,    65,    58,    64,    60,     6,
       1,    67,   320,     7,    70,    67,    64,    56,    70,   327,
      64,    66,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    68,    28,    29,    30,
      31,    32,    33,    34,    35,    36,     8,     9,   224,   225,
     226,    68,    43,    44,    68,    38,    68,    68,    41,    68,
      68,    64,    45,    46,    68,    64,    64,    58,    68,    60,
      53,    54,    63,     8,     9,    37,    38,    67,    64,    41,
      69,    64,    64,    45,    46,    47,    48,    64,    39,    51,
      52,    53,    54,     8,     9,     9,    58,    67,    60,     8,
       9,    68,    37,    38,    64,    67,    41,    69,    70,    64,
      45,    46,    47,    48,    64,    67,    51,    52,     8,     9,
       8,     9,    37,    58,    64,    60,    68,    64,    37,   162,
     106,    62,    67,   188,    69,    70,     8,     9,   321,    -1,
      -1,    -1,    -1,    58,    -1,    60,    -1,    37,    -1,    37,
      -1,    60,    67,    -1,    -1,    70,    -1,    -1,    67,    -1,
      69,    70,    -1,    -1,    -1,    37,    56,    -1,    38,    -1,
      60,    41,    60,    -1,    -1,    45,    46,    67,    -1,    67,
      70,    -1,    70,    53,    54,    -1,    -1,    -1,    60,    38,
      39,    40,    41,    -1,    -1,    67,    45,    46,    70,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    -1,
      38,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,    -1,
      -1,    -1,    -1,    -1,    -1,    53,    54,    -1,     6,     7,
       8,    -1,    10,    11,    12,    13,    14,    -1,    -1,    17,
      64,    69,    20,    21,    22,    23,    24,    25,    -1,    63,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    63,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
       5,     6,     7,     8,     9,    10,    11,    12,     6,     7,
       8,     9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    72,     0,    73,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    43,    44,
      58,    60,    63,    74,    76,    63,    77,    77,    77,    77,
      64,    78,    77,    77,    77,    77,    77,     8,     9,    37,
      58,    60,    67,    70,    99,   102,    99,    77,    78,    78,
      77,    77,    77,    77,    77,    77,    78,    67,    85,    86,
      87,    99,   102,    38,    41,    45,    46,    47,    48,    51,
      52,    69,    80,    81,    83,    87,    90,    92,    94,    95,
      98,   102,    83,    61,    62,    62,    63,    53,    54,    69,
      88,    89,    90,    92,    88,    83,    78,    78,    58,    79,
     102,    78,    88,    65,    86,   102,    78,    90,    95,    78,
     102,   102,     6,     8,     9,   100,   102,   104,   102,    64,
      67,    11,    90,    80,    98,   102,    69,    81,    97,    98,
      98,    90,    90,    67,    85,    95,    67,    67,    67,    67,
       9,    56,    57,    87,   102,    64,    67,     6,     7,     9,
      55,    64,    75,   104,   104,   102,    64,    64,    64,    58,
      79,    84,    85,    99,   102,   100,    67,    83,    64,    82,
      95,    64,    67,    85,    64,     7,   102,   102,    67,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    68,
      89,   102,    38,    39,    40,   101,   102,    78,    64,    64,
       9,    64,    64,    64,    64,    68,    94,    94,   104,   102,
      56,    83,    95,     6,     7,     7,     7,    83,    74,    63,
      63,    90,    95,    90,    83,   100,    67,    41,    95,    66,
       9,    78,    65,    64,    85,   100,   101,   104,   104,   104,
       6,     7,   104,   104,   104,   104,   104,    64,    68,    64,
      98,   104,    48,    98,    98,    90,    90,    68,    68,    68,
      68,    68,    93,    95,   102,    93,    93,    93,    64,    68,
      78,    64,    95,    82,    82,    90,    64,    78,    67,    68,
     104,   104,    89,    80,    64,    64,    78,    64,    64,    90,
      86,    66,    90,    39,     9,    95,    98,    67,    91,    90,
      68,   102,    64,    95,    64,    49,    50,    96,    64,    95,
      67,    64,    95,   104,    96,    68,    68,    64,   103,   104
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
#line 70 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:
#line 77 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		if((yyvsp[(1) - (2)].sym)->type == LLAB && (yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 7:
#line 86 "a.y"
    {
		(yyvsp[(1) - (4)].sym)->type = LVAR;
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 8:
#line 91 "a.y"
    {
		if((yyvsp[(1) - (4)].sym)->value != (yyvsp[(3) - (4)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (4)].sym)->name);
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 12:
#line 105 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), (yyvsp[(5) - (7)].lval), &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 13:
#line 109 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].addr), (yyvsp[(5) - (6)].lval), &nullgen);
	}
    break;

  case 14:
#line 113 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 15:
#line 120 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 16:
#line 127 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 17:
#line 134 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 18:
#line 138 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 19:
#line 145 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 20:
#line 152 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 21:
#line 159 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 22:
#line 166 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].addr), (yyvsp[(5) - (6)].lval), &nullgen);
	}
    break;

  case 23:
#line 173 "a.y"
    {
		Addr g;

		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = (yyvsp[(6) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), 0, &g);
	}
    break;

  case 24:
#line 182 "a.y"
    {
		Addr g;

		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = (yyvsp[(4) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &g, 0, &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 25:
#line 194 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(5) - (7)].addr), (yyvsp[(3) - (7)].addr).reg, &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 26:
#line 198 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(5) - (6)].addr), (yyvsp[(3) - (6)].addr).reg, &(yyvsp[(3) - (6)].addr));
	}
    break;

  case 27:
#line 202 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(4) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 28:
#line 209 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), (yyvsp[(2) - (3)].lval), &nullgen, 0, &nullgen);
	}
    break;

  case 29:
#line 216 "a.y"
    {
		settext((yyvsp[(2) - (4)].addr).sym);
		(yyvsp[(4) - (4)].addr).type = TYPE_TEXTSIZE;
		(yyvsp[(4) - (4)].addr).u.argsize = ArgsSizeUnknown;
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 30:
#line 223 "a.y"
    {
		settext((yyvsp[(2) - (6)].addr).sym);
		(yyvsp[(6) - (6)].addr).type = TYPE_TEXTSIZE;
		(yyvsp[(6) - (6)].addr).u.argsize = ArgsSizeUnknown;
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 31:
#line 230 "a.y"
    {
		settext((yyvsp[(2) - (8)].addr).sym);
		(yyvsp[(6) - (8)].addr).type = TYPE_TEXTSIZE;
		(yyvsp[(6) - (8)].addr).u.argsize = (yyvsp[(8) - (8)].lval);
		outcode((yyvsp[(1) - (8)].lval), Always, &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].lval), &(yyvsp[(6) - (8)].addr));
	}
    break;

  case 32:
#line 240 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 33:
#line 247 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &(yyvsp[(3) - (4)].addr), 0, &nullgen);
	}
    break;

  case 34:
#line 254 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 35:
#line 261 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 36:
#line 265 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 37:
#line 269 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), (yyvsp[(5) - (7)].lval), &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 38:
#line 273 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].addr), (yyvsp[(5) - (6)].addr).reg, &nullgen);
	}
    break;

  case 39:
#line 280 "a.y"
    {
		Addr g;

		g = nullgen;
		g.type = TYPE_CONST;
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
		outcode(AMRC, Always, &nullgen, 0, &g);
	}
    break;

  case 40:
#line 302 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), (yyvsp[(5) - (7)].addr).reg, &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 41:
#line 310 "a.y"
    {
		(yyvsp[(7) - (9)].addr).type = TYPE_REGREG2;
		(yyvsp[(7) - (9)].addr).offset = (yyvsp[(9) - (9)].lval);
		outcode((yyvsp[(1) - (9)].lval), (yyvsp[(2) - (9)].lval), &(yyvsp[(3) - (9)].addr), (yyvsp[(5) - (9)].addr).reg, &(yyvsp[(7) - (9)].addr));
	}
    break;

  case 42:
#line 319 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &(yyvsp[(2) - (2)].addr), 0, &nullgen);
	}
    break;

  case 43:
#line 326 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != TYPE_CONST || (yyvsp[(4) - (4)].addr).type != TYPE_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 44:
#line 335 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != TYPE_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(4) - (4)].addr).type != NAME_EXTERN && (yyvsp[(4) - (4)].addr).type != NAME_STATIC && (yyvsp[(4) - (4)].addr).type != TYPE_MEM)
			yyerror("value for FUNCDATA must be symbol reference");
 		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 45:
#line 346 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &nullgen, 0, &nullgen);
	}
    break;

  case 46:
#line 351 "a.y"
    {
		(yyval.lval) = Always;
	}
    break;

  case 47:
#line 355 "a.y"
    {
		(yyval.lval) = ((yyvsp[(1) - (2)].lval) & ~C_SCOND) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 48:
#line 359 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (2)].lval) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 51:
#line 368 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 52:
#line 374 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		(yyval.addr) = nullgen;
		if(pass == 2 && (yyvsp[(1) - (2)].sym)->type != LLAB)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 53:
#line 384 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 54:
#line 390 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = TYPE_CONST;
	}
    break;

  case 55:
#line 395 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 57:
#line 404 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 58:
#line 410 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 59:
#line 418 "a.y"
    {
		(yyval.lval) = 1 << (yyvsp[(1) - (1)].lval);
	}
    break;

  case 60:
#line 422 "a.y"
    {
		int i;
		(yyval.lval)=0;
		for(i=(yyvsp[(1) - (3)].lval); i<=(yyvsp[(3) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
		for(i=(yyvsp[(3) - (3)].lval); i<=(yyvsp[(1) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
	}
    break;

  case 61:
#line 431 "a.y"
    {
		(yyval.lval) = (1<<(yyvsp[(1) - (3)].lval)) | (yyvsp[(3) - (3)].lval);
	}
    break;

  case 65:
#line 440 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (4)].addr);
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 66:
#line 445 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 67:
#line 451 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 68:
#line 457 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 72:
#line 468 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (1)].addr);
		if((yyvsp[(1) - (1)].addr).name != NAME_EXTERN && (yyvsp[(1) - (1)].addr).name != NAME_STATIC) {
		}
	}
    break;

  case 73:
#line 476 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 75:
#line 486 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 77:
#line 496 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (4)].addr);
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 82:
#line 509 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 83:
#line 517 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 84:
#line 525 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REGREG;
		(yyval.addr).reg = (yyvsp[(2) - (5)].lval);
		(yyval.addr).offset = (yyvsp[(4) - (5)].lval);
	}
    break;

  case 85:
#line 534 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval)&15 | (yyvsp[(4) - (4)].lval) | (0 << 5);
	}
    break;

  case 86:
#line 540 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval)&15 | (yyvsp[(4) - (4)].lval) | (1 << 5);
	}
    break;

  case 87:
#line 546 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval)&15 | (yyvsp[(4) - (4)].lval) | (2 << 5);
	}
    break;

  case 88:
#line 552 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval)&15 | (yyvsp[(4) - (4)].lval) | (3 << 5);
	}
    break;

  case 89:
#line 560 "a.y"
    {
		if((yyval.lval) < REG_R0 || (yyval.lval) > REG_R15)
			print("register value out of range in shift\n");
		(yyval.lval) = (((yyvsp[(1) - (1)].lval)&15) << 8) | (1 << 4);
	}
    break;

  case 90:
#line 566 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 32)
			print("shift value out of range\n");
		(yyval.lval) = ((yyvsp[(1) - (1)].lval)&31) << 7;
	}
    break;

  case 92:
#line 575 "a.y"
    {
		(yyval.lval) = REGPC;
	}
    break;

  case 93:
#line 579 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range in R(...)\n");
		(yyval.lval) = REG_R0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 95:
#line 588 "a.y"
    {
		(yyval.lval) = REGSP;
	}
    break;

  case 97:
#line 595 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range in C(...)\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval); // TODO(rsc): REG_C0+$3
	}
    break;

  case 100:
#line 607 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 101:
#line 613 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_F0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 102:
#line 621 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(3) - (4)].lval);
		(yyval.addr).sym = nil;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 103:
#line 629 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 104:
#line 637 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = NAME_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 1);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 105:
#line 646 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 106:
#line 650 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 107:
#line 654 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 112:
#line 666 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 113:
#line 670 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 114:
#line 674 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 115:
#line 678 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 116:
#line 682 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 117:
#line 687 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 118:
#line 691 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 120:
#line 698 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 121:
#line 702 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 122:
#line 706 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 123:
#line 710 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 124:
#line 714 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:
#line 718 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 126:
#line 722 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 127:
#line 726 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:
#line 730 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:
#line 734 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2556 "y.tab.c"
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



