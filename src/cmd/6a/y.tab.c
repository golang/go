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
     LTYPEF = 276,
     LCONST = 277,
     LFP = 278,
     LPC = 279,
     LSB = 280,
     LBREG = 281,
     LLREG = 282,
     LSREG = 283,
     LFREG = 284,
     LMREG = 285,
     LXREG = 286,
     LFCONST = 287,
     LSCONST = 288,
     LSP = 289,
     LNAME = 290,
     LLAB = 291,
     LVAR = 292
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
#define LTYPEF 276
#define LCONST 277
#define LFP 278
#define LPC 279
#define LSB 280
#define LBREG 281
#define LLREG 282
#define LSREG 283
#define LFREG 284
#define LMREG 285
#define LXREG 286
#define LFCONST 287
#define LSCONST 288
#define LSP 289
#define LNAME 290
#define LLAB 291
#define LVAR 292




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
#line 37 "a.y"
{
	Sym	*sym;
	vlong	lval;
	double	dval;
	char	sval[8];
	Gen	gen;
	Gen2	gen2;
}
/* Line 193 of yacc.c.  */
#line 186 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 199 "y.tab.c"

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
#define YYLAST   545

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  56
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  42
/* YYNRULES -- Number of rules.  */
#define YYNRULES  136
/* YYNRULES -- Number of states.  */
#define YYNSTATES  273

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   292

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    54,    12,     5,     2,
      52,    53,    10,     8,    51,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    48,    49,
       6,    50,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    55,     2,     2,     2,
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
      45,    46,    47
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     5,     9,    10,    15,    16,    21,
      23,    26,    29,    33,    37,    40,    43,    46,    49,    52,
      55,    58,    61,    64,    67,    70,    73,    76,    79,    82,
      85,    88,    91,    94,    95,    97,   101,   105,   108,   110,
     113,   115,   118,   120,   124,   130,   134,   140,   143,   145,
     147,   149,   153,   159,   163,   169,   172,   174,   178,   184,
     190,   191,   193,   197,   203,   207,   211,   213,   215,   217,
     219,   222,   225,   227,   229,   231,   233,   238,   241,   244,
     246,   248,   250,   252,   254,   256,   258,   261,   264,   267,
     270,   273,   278,   284,   288,   290,   292,   294,   299,   304,
     309,   316,   326,   330,   334,   340,   349,   351,   358,   364,
     372,   373,   376,   379,   381,   383,   385,   387,   389,   392,
     395,   398,   402,   404,   407,   411,   416,   418,   422,   426,
     430,   434,   438,   443,   448,   452,   456
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      57,     0,    -1,    -1,    -1,    57,    58,    59,    -1,    -1,
      46,    48,    60,    59,    -1,    -1,    45,    48,    61,    59,
      -1,    49,    -1,    62,    49,    -1,     1,    49,    -1,    45,
      50,    97,    -1,    47,    50,    97,    -1,    13,    63,    -1,
      14,    67,    -1,    15,    66,    -1,    16,    64,    -1,    17,
      65,    -1,    21,    68,    -1,    19,    69,    -1,    22,    70,
      -1,    18,    71,    -1,    20,    72,    -1,    25,    73,    -1,
      26,    74,    -1,    27,    75,    -1,    28,    76,    -1,    29,
      77,    -1,    30,    78,    -1,    23,    79,    -1,    24,    80,
      -1,    31,    81,    -1,    -1,    51,    -1,    84,    51,    82,
      -1,    82,    51,    84,    -1,    84,    51,    -1,    84,    -1,
      51,    82,    -1,    82,    -1,    51,    85,    -1,    85,    -1,
      88,    51,    85,    -1,    92,    11,    95,    51,    88,    -1,
      89,    51,    87,    -1,    89,    51,    95,    51,    87,    -1,
      51,    83,    -1,    83,    -1,    63,    -1,    67,    -1,    84,
      51,    82,    -1,    84,    51,    82,    48,    37,    -1,    84,
      51,    82,    -1,    84,    51,    82,    48,    38,    -1,    84,
      51,    -1,    84,    -1,    84,    51,    82,    -1,    86,    51,
      82,    51,    95,    -1,    88,    51,    82,    51,    86,    -1,
      -1,    88,    -1,    89,    51,    88,    -1,    89,    51,    95,
      51,    88,    -1,    88,    51,    88,    -1,    88,    51,    82,
      -1,    86,    -1,    89,    -1,    85,    -1,    91,    -1,    10,
      86,    -1,    10,    90,    -1,    86,    -1,    90,    -1,    82,
      -1,    88,    -1,    95,    52,    34,    53,    -1,    45,    93,
      -1,    46,    93,    -1,    36,    -1,    39,    -1,    37,    -1,
      40,    -1,    44,    -1,    38,    -1,    41,    -1,    54,    96,
      -1,    54,    95,    -1,    54,    92,    -1,    54,    43,    -1,
      54,    42,    -1,    54,    52,    42,    53,    -1,    54,    52,
       9,    42,    53,    -1,    54,     9,    42,    -1,    90,    -1,
      91,    -1,    95,    -1,    95,    52,    37,    53,    -1,    95,
      52,    44,    53,    -1,    95,    52,    38,    53,    -1,    95,
      52,    37,    10,    95,    53,    -1,    95,    52,    37,    53,
      52,    37,    10,    95,    53,    -1,    52,    37,    53,    -1,
      52,    44,    53,    -1,    52,    37,    10,    95,    53,    -1,
      52,    37,    53,    52,    37,    10,    95,    53,    -1,    92,
      -1,    92,    52,    37,    10,    95,    53,    -1,    45,    93,
      52,    94,    53,    -1,    45,     6,     7,    93,    52,    35,
      53,    -1,    -1,     8,    95,    -1,     9,    95,    -1,    35,
      -1,    44,    -1,    33,    -1,    32,    -1,    47,    -1,     9,
      95,    -1,     8,    95,    -1,    55,    95,    -1,    52,    97,
      53,    -1,    32,    -1,     9,    32,    -1,    32,     9,    32,
      -1,     9,    32,     9,    32,    -1,    95,    -1,    97,     8,
      97,    -1,    97,     9,    97,    -1,    97,    10,    97,    -1,
      97,    11,    97,    -1,    97,    12,    97,    -1,    97,     6,
       6,    97,    -1,    97,     7,     7,    97,    -1,    97,     5,
      97,    -1,    97,     4,    97,    -1,    97,     3,    97,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    65,    65,    67,    66,    74,    73,    81,    80,    86,
      87,    88,    91,    96,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   123,   127,   134,   141,   148,   153,   160,
     165,   172,   177,   182,   189,   197,   202,   210,   215,   222,
     223,   226,   231,   241,   246,   256,   261,   266,   273,   281,
     291,   295,   302,   307,   315,   324,   335,   336,   339,   340,
     341,   345,   349,   350,   353,   354,   357,   363,   372,   381,
     386,   391,   396,   401,   406,   411,   417,   425,   431,   442,
     448,   454,   460,   466,   474,   475,   478,   484,   490,   496,
     502,   511,   520,   525,   530,   538,   548,   552,   561,   568,
     577,   580,   584,   590,   591,   595,   598,   599,   603,   607,
     611,   615,   621,   625,   629,   638,   649,   650,   654,   658,
     662,   666,   670,   674,   678,   682,   686
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'|'", "'^'", "'&'", "'<'", "'>'", "'+'",
  "'-'", "'*'", "'/'", "'%'", "LTYPE0", "LTYPE1", "LTYPE2", "LTYPE3",
  "LTYPE4", "LTYPEC", "LTYPED", "LTYPEN", "LTYPER", "LTYPET", "LTYPEG",
  "LTYPEPC", "LTYPES", "LTYPEM", "LTYPEI", "LTYPEXC", "LTYPEX", "LTYPERT",
  "LTYPEF", "LCONST", "LFP", "LPC", "LSB", "LBREG", "LLREG", "LSREG",
  "LFREG", "LMREG", "LXREG", "LFCONST", "LSCONST", "LSP", "LNAME", "LLAB",
  "LVAR", "':'", "';'", "'='", "','", "'('", "')'", "'$'", "'~'",
  "$accept", "prog", "@1", "line", "@2", "@3", "inst", "nonnon", "rimrem",
  "remrim", "rimnon", "nonrem", "nonrel", "spec1", "spec2", "spec3",
  "spec4", "spec5", "spec6", "spec7", "spec8", "spec9", "spec10", "spec11",
  "spec12", "spec13", "rem", "rom", "rim", "rel", "reg", "imm2", "imm",
  "mem", "omem", "nmem", "nam", "offset", "pointer", "con", "con2", "expr", 0
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
     285,   286,   287,   288,   289,   290,   291,   292,    58,    59,
      61,    44,    40,    41,    36,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    56,    57,    58,    57,    60,    59,    61,    59,    59,
      59,    59,    62,    62,    62,    62,    62,    62,    62,    62,
      62,    62,    62,    62,    62,    62,    62,    62,    62,    62,
      62,    62,    62,    63,    63,    64,    65,    66,    66,    67,
      67,    68,    68,    68,    69,    70,    70,    71,    71,    72,
      72,    73,    73,    74,    74,    75,    75,    75,    76,    77,
      78,    78,    79,    79,    80,    81,    82,    82,    83,    83,
      83,    83,    83,    83,    84,    84,    85,    85,    85,    86,
      86,    86,    86,    86,    86,    86,    87,    88,    88,    88,
      88,    88,    88,    88,    89,    89,    90,    90,    90,    90,
      90,    90,    90,    90,    90,    90,    91,    91,    92,    92,
      93,    93,    93,    94,    94,    94,    95,    95,    95,    95,
      95,    95,    96,    96,    96,    96,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     0,     4,     1,
       2,     2,     3,     3,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     0,     1,     3,     3,     2,     1,     2,
       1,     2,     1,     3,     5,     3,     5,     2,     1,     1,
       1,     3,     5,     3,     5,     2,     1,     3,     5,     5,
       0,     1,     3,     5,     3,     3,     1,     1,     1,     1,
       2,     2,     1,     1,     1,     1,     4,     2,     2,     1,
       1,     1,     1,     1,     1,     1,     2,     2,     2,     2,
       2,     4,     5,     3,     1,     1,     1,     4,     4,     4,
       6,     9,     3,     3,     5,     8,     1,     6,     5,     7,
       0,     2,     2,     1,     1,     1,     1,     1,     2,     2,
       2,     3,     1,     2,     3,     4,     1,     3,     3,     3,
       3,     3,     4,     4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    33,     0,     0,     0,     0,
       0,     0,    33,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    60,     0,     0,     0,     0,     9,     4,     0,
      11,    34,    14,     0,     0,   116,    79,    81,    84,    80,
      82,    85,    83,   110,   117,     0,     0,     0,    15,    40,
      66,    67,    94,    95,   106,    96,     0,    16,    74,    38,
      75,    17,     0,    18,     0,     0,   110,   110,     0,    22,
      48,    68,    72,    73,    69,    96,    20,     0,    34,    49,
      50,    23,   110,     0,     0,    19,    42,     0,     0,    21,
       0,    30,     0,    31,     0,    24,     0,    25,     0,    26,
      56,    27,     0,    28,     0,    29,    61,    32,     0,     7,
       0,     5,     0,    10,   119,   118,     0,     0,     0,     0,
      39,     0,     0,   126,     0,   120,     0,     0,     0,    90,
      89,     0,    88,    87,    37,     0,     0,    70,    71,    77,
      78,    47,     0,     0,    77,    41,     0,     0,     0,     0,
       0,     0,     0,    55,     0,     0,     0,     0,    12,     0,
      13,   110,   111,   112,     0,     0,   102,   103,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   121,     0,
       0,     0,     0,    93,     0,     0,    35,    36,     0,     0,
      43,     0,    45,     0,    62,     0,    64,    51,    53,    57,
       0,     0,    65,     8,     6,     0,   115,   113,   114,     0,
       0,     0,   136,   135,   134,     0,     0,   127,   128,   129,
     130,   131,     0,     0,    97,    99,    98,     0,    91,    76,
       0,     0,   122,    86,     0,     0,     0,     0,     0,     0,
       0,   108,   104,     0,   132,   133,     0,     0,     0,    92,
      44,   123,     0,    46,    63,    52,    54,    58,    59,     0,
       0,   107,   100,     0,     0,   124,   109,     0,     0,   125,
     105,     0,   101
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    28,   159,   157,    29,    32,    61,    63,
      57,    48,    85,    76,    89,    69,    81,    95,    97,    99,
     101,   103,   105,    91,    93,   107,    58,    70,    59,    71,
      50,   192,    60,    51,    52,    53,    54,   119,   209,    55,
     233,   124
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -86
static const yytype_int16 yypact[] =
{
     -86,    12,   -86,   218,   -18,   -17,   264,   285,   285,   340,
     163,     9,   319,   359,   407,   407,     3,   285,   285,   285,
     306,     3,     3,     3,    -1,    13,    14,   -86,   -86,    18,
     -86,   -86,   -86,   466,   466,   -86,   -86,   -86,   -86,   -86,
     -86,   -86,   -86,    69,   -86,   340,   380,   466,   -86,   -86,
     -86,   -86,   -86,   -86,    42,    43,     8,   -86,   -86,    19,
     -86,   -86,    47,   -86,    48,    73,    69,    71,   243,   -86,
     -86,   -86,   -86,   -86,   -86,    54,   -86,   104,   340,   -86,
     -86,   -86,    71,   391,   466,   -86,   -86,    65,    67,   -86,
      75,   -86,    76,   -86,    78,   -86,    81,   -86,    83,   -86,
      89,   -86,   102,   -86,   103,   -86,   -86,   -86,   105,   -86,
     466,   -86,   466,   -86,   -86,   -86,   116,   466,   466,   110,
     -86,    16,   115,   -86,   140,   -86,   120,     4,   413,   -86,
     -86,   425,   -86,   -86,   -86,   340,   285,   -86,   -86,   110,
     -86,   -86,    59,   466,   -86,   -86,   391,   135,   432,   441,
       3,   340,   340,   340,   340,   340,   340,   218,   180,   218,
     180,    71,   -86,   -86,    -5,   466,   118,   -86,   466,   466,
     466,   169,   187,   466,   466,   466,   466,   466,   -86,   166,
      23,   143,   144,   -86,   462,   145,   -86,   -86,   158,   165,
     -86,    20,   -86,   170,   -86,   172,   -86,   176,   177,   -86,
     175,   178,   -86,   -86,   -86,   168,   -86,   -86,   -86,   164,
     201,   219,   518,   526,   533,   466,   466,    33,    33,   -86,
     -86,   -86,   466,   466,   203,   -86,   -86,   204,   -86,   -86,
       3,   227,   251,   -86,   207,     3,   225,   228,   466,   306,
     233,   -86,   -86,   259,   491,   491,   217,   221,   234,   -86,
     -86,   267,   245,   -86,   -86,   -86,   -86,   -86,   -86,   232,
     466,   -86,   -86,   268,   254,   -86,   -86,   238,   466,   -86,
     -86,   239,   -86
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -86,   -86,   -86,   -35,   -86,   -86,   -86,   287,   -86,   -86,
     -86,   294,   -86,   -86,   -86,   -86,   -86,   -86,   -86,   -86,
     -86,   -86,   -86,   -86,   -86,   -86,    26,   229,     1,   -11,
      -9,    79,    -8,    72,     0,    -6,    10,   -60,   -86,   -10,
     -86,   -85
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      75,    72,    86,    88,    74,    87,   139,   140,    94,    62,
      73,   102,     2,   104,   106,   108,    33,   128,    96,    98,
     100,    77,   144,   114,   115,   158,   165,   160,   206,   231,
     207,    30,    49,   223,    31,    64,   123,   125,    49,   208,
      35,   180,   181,   175,   176,   177,   133,   109,   182,   110,
     129,   130,   232,    43,    43,    44,   137,    56,    75,    72,
     131,   111,    74,    47,   112,   138,   132,   113,    73,   166,
     134,   120,   145,    88,   123,   116,   224,   117,   118,   117,
     118,    33,    34,   212,   213,   214,    90,    92,   217,   218,
     219,   220,   221,   188,   126,   127,   180,   181,   135,   136,
     123,   205,   123,   182,   120,    35,   142,   162,   163,    36,
      37,    38,    39,    40,    41,   143,   146,    42,   115,   147,
      44,   123,   203,   161,   204,    46,   148,   149,    47,   150,
     244,   245,   151,   189,   152,   190,    88,   187,   193,   195,
     153,   194,   196,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   154,   155,   210,   156,   179,   123,   123,
     123,   186,   164,   123,   123,   123,   123,   123,   167,   188,
     211,    33,    34,    65,   115,   215,   222,   197,   198,   199,
     200,   201,   202,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   216,    35,   225,   226,   228,    36,
      37,    38,    39,    40,    41,   123,   123,    42,    66,    67,
      44,   229,   246,   247,    68,    46,   230,   241,    47,     4,
     240,   234,   250,   235,   236,   237,   238,   254,   257,   239,
     258,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
     267,    33,    34,    65,   242,   248,   243,   249,   271,   251,
     252,   191,   255,    24,    25,    26,   256,    27,   259,   260,
     261,   263,    33,    34,   262,    35,   264,   265,   268,    36,
      37,    38,    39,    40,    41,   266,   269,    42,    66,    67,
      44,   270,   272,    33,    34,    46,    35,   141,    47,    79,
      36,    37,    38,    39,    40,    41,    80,     0,    42,    43,
       0,    44,     0,   253,     0,    45,    46,    35,     0,    47,
       0,    36,    37,    38,    39,    40,    41,    33,    34,    42,
      43,     0,    44,     0,     0,     0,     0,    46,     0,    56,
      47,     0,    36,    37,    38,    39,    40,    41,    33,    34,
      42,    35,     0,     0,     0,    36,    37,    38,    39,    40,
      41,     0,     0,    42,    43,     0,    44,    33,    34,     0,
      78,    46,    35,     0,    47,     0,    36,    37,    38,    39,
      40,    41,     0,     0,    42,    43,     0,    44,    33,    34,
       0,    35,    46,     0,     0,    47,     0,     0,     0,    33,
      34,     0,     0,     0,    82,    67,    44,     0,     0,     0,
      83,    84,    35,    56,    47,    33,    34,   121,     0,     0,
       0,    33,    34,    35,   122,     0,     0,    44,     0,     0,
       0,     0,    84,    33,   184,    47,    82,    67,    44,    35,
      33,    34,     0,    84,     0,    35,    47,     0,     0,    33,
      34,     0,    43,     0,    44,   183,     0,    35,     0,    46,
      44,     0,    47,     0,    35,    84,     0,   185,    47,     0,
      33,    34,    44,    35,    33,    34,     0,    84,     0,    44,
      47,     0,     0,     0,    84,     0,   191,    47,    44,     0,
       0,     0,     0,    84,    35,    56,    47,     0,    35,   173,
     174,   175,   176,   177,   227,     0,     0,     0,     0,    44,
       0,     0,     0,    44,    84,     0,     0,    47,    84,     0,
       0,    47,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   170,   171,   172,   173,   174,   175,   176,   177,   171,
     172,   173,   174,   175,   176,   177
};

static const yytype_int16 yycheck[] =
{
      10,    10,    13,    13,    10,    13,    66,    67,    16,     8,
      10,    20,     0,    21,    22,    23,     8,     9,    17,    18,
      19,    11,    82,    33,    34,   110,    10,   112,    33,     9,
      35,    49,     6,    10,    51,     9,    46,    47,    12,    44,
      32,    37,    38,    10,    11,    12,    56,    48,    44,    50,
      42,    43,    32,    45,    45,    47,    65,    54,    68,    68,
      52,    48,    68,    55,    50,    65,    56,    49,    68,    53,
      51,    45,    83,    83,    84,     6,    53,     8,     9,     8,
       9,     8,     9,   168,   169,   170,    14,    15,   173,   174,
     175,   176,   177,    34,    52,    52,    37,    38,    51,    51,
     110,   161,   112,    44,    78,    32,    52,   117,   118,    36,
      37,    38,    39,    40,    41,    11,    51,    44,   128,    52,
      47,   131,   157,     7,   159,    52,    51,    51,    55,    51,
     215,   216,    51,   143,    51,   146,   146,   136,   148,   149,
      51,   149,   150,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    51,    51,   165,    51,    37,   168,   169,
     170,   135,    52,   173,   174,   175,   176,   177,    53,    34,
      52,     8,     9,    10,   184,     6,    10,   151,   152,   153,
     154,   155,   156,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    53,     7,    32,    53,    53,    53,    36,
      37,    38,    39,    40,    41,   215,   216,    44,    45,    46,
      47,    53,   222,   223,    51,    52,    51,    53,    55,     1,
      52,    51,   230,    51,    48,    48,    51,   235,   238,    51,
     239,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
     260,     8,     9,    10,    53,    52,    37,    53,   268,    32,
       9,    54,    37,    45,    46,    47,    38,    49,    35,    10,
      53,    37,     8,     9,    53,    32,     9,    32,    10,    36,
      37,    38,    39,    40,    41,    53,    32,    44,    45,    46,
      47,    53,    53,     8,     9,    52,    32,    68,    55,    12,
      36,    37,    38,    39,    40,    41,    12,    -1,    44,    45,
      -1,    47,    -1,   234,    -1,    51,    52,    32,    -1,    55,
      -1,    36,    37,    38,    39,    40,    41,     8,     9,    44,
      45,    -1,    47,    -1,    -1,    -1,    -1,    52,    -1,    54,
      55,    -1,    36,    37,    38,    39,    40,    41,     8,     9,
      44,    32,    -1,    -1,    -1,    36,    37,    38,    39,    40,
      41,    -1,    -1,    44,    45,    -1,    47,     8,     9,    -1,
      51,    52,    32,    -1,    55,    -1,    36,    37,    38,    39,
      40,    41,    -1,    -1,    44,    45,    -1,    47,     8,     9,
      -1,    32,    52,    -1,    -1,    55,    -1,    -1,    -1,     8,
       9,    -1,    -1,    -1,    45,    46,    47,    -1,    -1,    -1,
      51,    52,    32,    54,    55,     8,     9,    37,    -1,    -1,
      -1,     8,     9,    32,    44,    -1,    -1,    47,    -1,    -1,
      -1,    -1,    52,     8,     9,    55,    45,    46,    47,    32,
       8,     9,    -1,    52,    -1,    32,    55,    -1,    -1,     8,
       9,    -1,    45,    -1,    47,    42,    -1,    32,    -1,    52,
      47,    -1,    55,    -1,    32,    52,    -1,    42,    55,    -1,
       8,     9,    47,    32,     8,     9,    -1,    52,    -1,    47,
      55,    -1,    -1,    -1,    52,    -1,    54,    55,    47,    -1,
      -1,    -1,    -1,    52,    32,    54,    55,    -1,    32,     8,
       9,    10,    11,    12,    42,    -1,    -1,    -1,    -1,    47,
      -1,    -1,    -1,    47,    52,    -1,    -1,    55,    52,    -1,
      -1,    55,     4,     5,     6,     7,     8,     9,    10,    11,
      12,     5,     6,     7,     8,     9,    10,    11,    12,     6,
       7,     8,     9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    57,     0,    58,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    45,    46,    47,    49,    59,    62,
      49,    51,    63,     8,     9,    32,    36,    37,    38,    39,
      40,    41,    44,    45,    47,    51,    52,    55,    67,    82,
      86,    89,    90,    91,    92,    95,    54,    66,    82,    84,
      88,    64,    84,    65,    82,    10,    45,    46,    51,    71,
      83,    85,    86,    90,    91,    95,    69,    92,    51,    63,
      67,    72,    45,    51,    52,    68,    85,    88,    95,    70,
      89,    79,    89,    80,    88,    73,    84,    74,    84,    75,
      84,    76,    86,    77,    88,    78,    88,    81,    88,    48,
      50,    48,    50,    49,    95,    95,     6,     8,     9,    93,
      82,    37,    44,    95,    97,    95,    52,    52,     9,    42,
      43,    52,    92,    95,    51,    51,    51,    86,    90,    93,
      93,    83,    52,    11,    93,    85,    51,    52,    51,    51,
      51,    51,    51,    51,    51,    51,    51,    61,    97,    60,
      97,     7,    95,    95,    52,    10,    53,    53,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    53,    37,
      37,    38,    44,    42,     9,    42,    82,    84,    34,    95,
      85,    54,    87,    95,    88,    95,    88,    82,    82,    82,
      82,    82,    82,    59,    59,    93,    33,    35,    44,    94,
      95,    52,    97,    97,    97,     6,     7,    97,    97,    97,
      97,    97,    10,    10,    53,    53,    53,    42,    53,    53,
      51,     9,    32,    96,    51,    51,    48,    48,    51,    51,
      52,    53,    53,    37,    97,    97,    95,    95,    52,    53,
      88,    32,     9,    87,    88,    37,    38,    95,    86,    35,
      10,    53,    53,    37,     9,    32,    53,    95,    10,    32,
      53,    95,    53
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
#line 67 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:
#line 74 "a.y"
    {
		if((yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->name);
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 7:
#line 81 "a.y"
    {
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 12:
#line 92 "a.y"
    {
		(yyvsp[(1) - (3)].sym)->type = LVAR;
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 13:
#line 97 "a.y"
    {
		if((yyvsp[(1) - (3)].sym)->value != (yyvsp[(3) - (3)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (3)].sym)->name);
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 14:
#line 102 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 15:
#line 103 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 16:
#line 104 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 17:
#line 105 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 18:
#line 106 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 19:
#line 107 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 20:
#line 108 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 21:
#line 109 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 22:
#line 110 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 23:
#line 111 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 24:
#line 112 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 25:
#line 113 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 26:
#line 114 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 27:
#line 115 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 28:
#line 116 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 29:
#line 117 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 30:
#line 118 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 31:
#line 119 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 32:
#line 120 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 33:
#line 123 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = nullgen;
	}
    break;

  case 34:
#line 128 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = nullgen;
	}
    break;

  case 35:
#line 135 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 36:
#line 142 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 37:
#line 149 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (2)].gen);
		(yyval.gen2).to = nullgen;
	}
    break;

  case 38:
#line 154 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (1)].gen);
		(yyval.gen2).to = nullgen;
	}
    break;

  case 39:
#line 161 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 40:
#line 166 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(1) - (1)].gen);
	}
    break;

  case 41:
#line 173 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 42:
#line 178 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(1) - (1)].gen);
	}
    break;

  case 43:
#line 183 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 44:
#line 190 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.gen2).to = (yyvsp[(5) - (5)].gen);
	}
    break;

  case 45:
#line 198 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 46:
#line 203 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.gen2).to = (yyvsp[(5) - (5)].gen);
	}
    break;

  case 47:
#line 211 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 48:
#line 216 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(1) - (1)].gen);
	}
    break;

  case 51:
#line 227 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 52:
#line 232 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (5)].gen);
		if((yyval.gen2).from.index != D_NONE)
			yyerror("dp shift with lhs index");
		(yyval.gen2).from.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 53:
#line 242 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 54:
#line 247 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (5)].gen);
		if((yyval.gen2).to.index != D_NONE)
			yyerror("dp move with lhs index");
		(yyval.gen2).to.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 55:
#line 257 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (2)].gen);
		(yyval.gen2).to = nullgen;
	}
    break;

  case 56:
#line 262 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (1)].gen);
		(yyval.gen2).to = nullgen;
	}
    break;

  case 57:
#line 267 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 58:
#line 274 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (5)].gen);
		(yyval.gen2).to.offset = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 59:
#line 282 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(3) - (5)].gen);
		(yyval.gen2).to = (yyvsp[(5) - (5)].gen);
		if((yyvsp[(1) - (5)].gen).type != D_CONST)
			yyerror("illegal constant");
		(yyval.gen2).to.offset = (yyvsp[(1) - (5)].gen).offset;
	}
    break;

  case 60:
#line 291 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = nullgen;
	}
    break;

  case 61:
#line 296 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (1)].gen);
		(yyval.gen2).to = nullgen;
	}
    break;

  case 62:
#line 303 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 63:
#line 308 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.gen2).to = (yyvsp[(5) - (5)].gen);
	}
    break;

  case 64:
#line 316 "a.y"
    {
		if((yyvsp[(1) - (3)].gen).type != D_CONST || (yyvsp[(3) - (3)].gen).type != D_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 65:
#line 325 "a.y"
    {
		if((yyvsp[(1) - (3)].gen).type != D_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(3) - (3)].gen).type != D_EXTERN)
			yyerror("value for FUNCDATA must be symbol reference");
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 70:
#line 342 "a.y"
    {
		(yyval.gen) = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 71:
#line 346 "a.y"
    {
		(yyval.gen) = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 76:
#line 358 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 77:
#line 364 "a.y"
    {
		(yyval.gen) = nullgen;
		if(pass == 2)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->name);
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 78:
#line 373 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 79:
#line 382 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 80:
#line 387 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 81:
#line 392 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 82:
#line 397 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 83:
#line 402 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SP;
	}
    break;

  case 84:
#line 407 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 85:
#line 412 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 86:
#line 418 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST;
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 87:
#line 426 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST;
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 88:
#line 432 "a.y"
    {
		(yyval.gen) = (yyvsp[(2) - (2)].gen);
		(yyval.gen).index = (yyvsp[(2) - (2)].gen).type;
		(yyval.gen).type = D_ADDR;
		/*
		if($2.type == D_AUTO || $2.type == D_PARAM)
			yyerror("constant cannot be automatic: %s",
				$2.sym->name);
		 */
	}
    break;

  case 89:
#line 443 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SCONST;
		memcpy((yyval.gen).sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.gen).sval));
	}
    break;

  case 90:
#line 449 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 91:
#line 455 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = (yyvsp[(3) - (4)].dval);
	}
    break;

  case 92:
#line 461 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = -(yyvsp[(4) - (5)].dval);
	}
    break;

  case 93:
#line 467 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 96:
#line 479 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_NONE;
		(yyval.gen).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 97:
#line 485 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 98:
#line 491 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_SP;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 99:
#line 497 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 100:
#line 503 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_NONE;
		(yyval.gen).offset = (yyvsp[(1) - (6)].lval);
		(yyval.gen).index = (yyvsp[(3) - (6)].lval);
		(yyval.gen).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 101:
#line 512 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(3) - (9)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (9)].lval);
		(yyval.gen).index = (yyvsp[(6) - (9)].lval);
		(yyval.gen).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 102:
#line 521 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(2) - (3)].lval);
	}
    break;

  case 103:
#line 526 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_SP;
	}
    break;

  case 104:
#line 531 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_NONE;
		(yyval.gen).index = (yyvsp[(2) - (5)].lval);
		(yyval.gen).scale = (yyvsp[(4) - (5)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 105:
#line 539 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(2) - (8)].lval);
		(yyval.gen).index = (yyvsp[(5) - (8)].lval);
		(yyval.gen).scale = (yyvsp[(7) - (8)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 106:
#line 549 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (1)].gen);
	}
    break;

  case 107:
#line 553 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (6)].gen);
		(yyval.gen).index = (yyvsp[(3) - (6)].lval);
		(yyval.gen).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 108:
#line 562 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(4) - (5)].lval);
		(yyval.gen).sym = (yyvsp[(1) - (5)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 109:
#line 569 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_STATIC;
		(yyval.gen).sym = (yyvsp[(1) - (7)].sym);
		(yyval.gen).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 110:
#line 577 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 111:
#line 581 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 112:
#line 585 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 114:
#line 592 "a.y"
    {
		(yyval.lval) = D_AUTO;
	}
    break;

  case 117:
#line 600 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 118:
#line 604 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 119:
#line 608 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 120:
#line 612 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 121:
#line 616 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 122:
#line 622 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].lval) & 0xffffffffLL;
	}
    break;

  case 123:
#line 626 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval) & 0xffffffffLL;
	}
    break;

  case 124:
#line 630 "a.y"
    {
		// Change explicit 0 argument size to 1
		// so that we can distinguish it from missing.
		if((yyvsp[(3) - (3)].lval) == 0)
			(yyvsp[(3) - (3)].lval) = 1;
		(yyval.lval) = ((yyvsp[(1) - (3)].lval) & 0xffffffffLL) +
			(((yyvsp[(3) - (3)].lval) & 0xffffLL) << 32);
	}
    break;

  case 125:
#line 639 "a.y"
    {
		// Change explicit 0 argument size to 1
		// so that we can distinguish it from missing.
		if((yyvsp[(4) - (4)].lval) == 0)
			(yyvsp[(4) - (4)].lval) = 1;
		(yyval.lval) = (-(yyvsp[(2) - (4)].lval) & 0xffffffffLL) +
			(((yyvsp[(4) - (4)].lval) & 0xffffLL) << 32);
	}
    break;

  case 127:
#line 651 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:
#line 655 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:
#line 659 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 130:
#line 663 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 131:
#line 667 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 132:
#line 671 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 133:
#line 675 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 134:
#line 679 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 135:
#line 683 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 136:
#line 687 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2578 "y.tab.c"
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



