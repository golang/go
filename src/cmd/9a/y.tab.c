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




/* Copy the first part of user declarations.  */
#line 30 "a.y"

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
#line 38 "a.y"
{
	Sym	*sym;
	vlong	lval;
	double	dval;
	char	sval[8];
	Addr	addr;
}
/* Line 193 of yacc.c.  */
#line 238 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 251 "y.tab.c"

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
#define YYLAST   932

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  82
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  31
/* YYNRULES -- Number of rules.  */
#define YYNRULES  186
/* YYNRULES -- Number of states.  */
#define YYNSTATES  462

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   318

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    80,    12,     5,     2,
      78,    79,    10,     8,    77,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    74,    76,
       6,    75,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    81,     2,     2,     2,
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
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,     8,    13,    18,    23,    26,
      28,    31,    34,    39,    44,    49,    54,    59,    64,    69,
      74,    79,    84,    89,    94,    99,   104,   109,   114,   119,
     124,   129,   134,   141,   146,   151,   158,   163,   168,   175,
     182,   189,   194,   199,   206,   211,   218,   223,   230,   235,
     240,   243,   250,   255,   260,   265,   272,   277,   282,   287,
     292,   297,   302,   307,   312,   315,   318,   323,   327,   331,
     337,   342,   347,   354,   359,   364,   371,   378,   385,   394,
     399,   404,   408,   411,   416,   421,   428,   437,   442,   449,
     454,   459,   466,   473,   482,   491,   500,   509,   514,   519,
     524,   531,   536,   543,   548,   553,   556,   559,   563,   567,
     571,   575,   578,   582,   586,   591,   596,   599,   605,   613,
     618,   625,   632,   639,   646,   649,   654,   657,   659,   661,
     663,   665,   667,   669,   671,   673,   678,   680,   682,   684,
     689,   691,   696,   698,   702,   704,   707,   711,   716,   719,
     722,   725,   729,   732,   734,   739,   743,   749,   751,   756,
     761,   767,   775,   776,   778,   779,   782,   785,   787,   789,
     791,   793,   795,   798,   801,   804,   808,   810,   814,   818,
     822,   826,   830,   835,   840,   844,   848
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      83,     0,    -1,    -1,    83,    84,    -1,    -1,    71,    74,
      85,    84,    -1,    71,    75,   112,    76,    -1,    73,    75,
     112,    76,    -1,    58,    76,    -1,    76,    -1,    86,    76,
      -1,     1,    76,    -1,    13,    88,    77,    88,    -1,    13,
     106,    77,    88,    -1,    13,   105,    77,    88,    -1,    14,
      88,    77,    88,    -1,    14,   106,    77,    88,    -1,    14,
     105,    77,    88,    -1,    22,   106,    77,    96,    -1,    22,
     105,    77,    96,    -1,    22,   102,    77,    96,    -1,    22,
      96,    77,    96,    -1,    22,    96,    77,   106,    -1,    22,
      96,    77,   105,    -1,    13,    88,    77,   106,    -1,    13,
      88,    77,   105,    -1,    14,    88,    77,   106,    -1,    14,
      88,    77,   105,    -1,    13,    96,    77,   106,    -1,    13,
      96,    77,   105,    -1,    13,    95,    77,    96,    -1,    13,
      96,    77,    95,    -1,    13,    96,    77,   103,    77,    95,
      -1,    13,    95,    77,    97,    -1,    67,   103,    77,   111,
      -1,    13,    88,    77,   103,    77,    91,    -1,    13,    88,
      77,    97,    -1,    13,    88,    77,    91,    -1,    18,    88,
      77,   104,    77,    88,    -1,    18,   103,    77,   104,    77,
      88,    -1,    18,    88,    77,   103,    77,    88,    -1,    18,
      88,    77,    88,    -1,    18,   103,    77,    88,    -1,    16,
      88,    77,   104,    77,    88,    -1,    16,    88,    77,    88,
      -1,    17,    88,    77,   104,    77,    88,    -1,    17,    88,
      77,    88,    -1,    17,   103,    77,   104,    77,    88,    -1,
      17,   103,    77,    88,    -1,    15,    88,    77,    88,    -1,
      15,    88,    -1,    68,    88,    77,   104,    77,    88,    -1,
      13,   103,    77,    88,    -1,    13,   101,    77,    88,    -1,
      20,    98,    77,    98,    -1,    20,    98,    77,   111,    77,
      98,    -1,    13,    97,    77,    97,    -1,    13,    94,    77,
      97,    -1,    13,    91,    77,    88,    -1,    13,    94,    77,
      88,    -1,    13,    89,    77,    88,    -1,    13,    88,    77,
      89,    -1,    13,    97,    77,    94,    -1,    13,    88,    77,
      94,    -1,    21,    87,    -1,    21,   106,    -1,    21,    78,
      89,    79,    -1,    21,    77,    87,    -1,    21,    77,   106,
      -1,    21,    77,    78,    89,    79,    -1,    21,    97,    77,
      87,    -1,    21,    97,    77,   106,    -1,    21,    97,    77,
      78,    89,    79,    -1,    21,   111,    77,    87,    -1,    21,
     111,    77,   106,    -1,    21,   111,    77,    78,    89,    79,
      -1,    21,   111,    77,   111,    77,    87,    -1,    21,   111,
      77,   111,    77,   106,    -1,    21,   111,    77,   111,    77,
      78,    89,    79,    -1,    27,    88,    77,   104,    -1,    27,
     103,    77,   104,    -1,    27,    88,   108,    -1,    27,   108,
      -1,    23,    96,    77,    96,    -1,    25,    96,    77,    96,
      -1,    25,    96,    77,    96,    77,    96,    -1,    26,    96,
      77,    96,    77,    96,    77,    96,    -1,    24,    96,    77,
      96,    -1,    24,    96,    77,    96,    77,    97,    -1,    19,
      88,    77,    88,    -1,    19,    88,    77,   103,    -1,    19,
      88,    77,    88,    77,    97,    -1,    19,    88,    77,   103,
      77,    97,    -1,    63,   103,    77,    88,    77,   103,    77,
      88,    -1,    63,   103,    77,    88,    77,    99,    77,    88,
      -1,    63,    88,    77,    88,    77,   103,    77,    88,    -1,
      63,    88,    77,    88,    77,    99,    77,    88,    -1,    64,
     106,    77,    88,    -1,    64,    88,    77,   106,    -1,    59,
     105,    77,    88,    -1,    59,   105,    77,   103,    77,    88,
      -1,    60,    88,    77,   105,    -1,    60,    88,    77,   103,
      77,   105,    -1,    62,   105,    77,    88,    -1,    62,    88,
      77,   105,    -1,    61,   105,    -1,    29,   108,    -1,    29,
      88,   108,    -1,    29,    96,   108,    -1,    29,    77,    88,
      -1,    29,    77,    96,    -1,    29,   103,    -1,    32,   103,
     108,    -1,    32,   101,   108,    -1,    56,   103,    77,   103,
      -1,    57,   103,    77,   106,    -1,    30,   108,    -1,    33,
     107,    77,    80,   100,    -1,    33,   107,    77,   111,    77,
      80,   100,    -1,    34,   107,    77,   103,    -1,    34,   107,
      77,   111,    77,   103,    -1,    35,   107,    11,   111,    77,
     103,    -1,    35,   107,    11,   111,    77,   101,    -1,    35,
     107,    11,   111,    77,   102,    -1,    36,   108,    -1,   111,
      78,    41,    79,    -1,    71,   109,    -1,   104,    -1,    90,
      -1,    92,    -1,    50,    -1,    47,    -1,    51,    -1,    55,
      -1,    53,    -1,    52,    78,   111,    79,    -1,    93,    -1,
      49,    -1,    45,    -1,    48,    78,   111,    79,    -1,    42,
      -1,    47,    78,   111,    79,    -1,   111,    -1,   111,    77,
     111,    -1,    37,    -1,     9,    37,    -1,    37,     9,    37,
      -1,     9,    37,     9,    37,    -1,    80,   106,    -1,    80,
      70,    -1,    80,    69,    -1,    80,     9,    69,    -1,    80,
     111,    -1,    44,    -1,    46,    78,   111,    79,    -1,    78,
     104,    79,    -1,    78,   104,     8,   104,    79,    -1,   107,
      -1,   111,    78,   104,    79,    -1,   111,    78,   110,    79,
      -1,    71,   109,    78,   110,    79,    -1,    71,     6,     7,
     109,    78,    39,    79,    -1,    -1,    77,    -1,    -1,     8,
     111,    -1,     9,   111,    -1,    39,    -1,    38,    -1,    40,
      -1,    37,    -1,    73,    -1,     9,   111,    -1,     8,   111,
      -1,    81,   111,    -1,    78,   112,    79,    -1,   111,    -1,
     112,     8,   112,    -1,   112,     9,   112,    -1,   112,    10,
     112,    -1,   112,    11,   112,    -1,   112,    12,   112,    -1,
     112,     6,     6,   112,    -1,   112,     7,     7,   112,    -1,
     112,     5,   112,    -1,   112,     4,   112,    -1,   112,     3,
     112,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    66,    66,    67,    71,    70,    79,    84,    90,    94,
      95,    96,   102,   106,   110,   114,   118,   122,   129,   133,
     137,   141,   145,   149,   156,   160,   164,   168,   175,   179,
     186,   190,   194,   198,   202,   209,   213,   217,   227,   231,
     235,   239,   243,   247,   251,   255,   259,   263,   267,   271,
     275,   282,   289,   293,   300,   304,   312,   316,   320,   324,
     328,   332,   336,   340,   349,   353,   357,   361,   365,   369,
     373,   377,   381,   385,   389,   393,   397,   405,   413,   424,
     428,   432,   436,   443,   447,   451,   455,   459,   463,   470,
     474,   478,   482,   489,   493,   497,   501,   508,   512,   520,
     524,   528,   532,   536,   540,   544,   551,   555,   559,   563,
     567,   571,   578,   582,   589,   598,   609,   616,   621,   633,
     638,   651,   659,   667,   678,   684,   690,   701,   709,   710,
     713,   721,   729,   737,   745,   751,   759,   762,   770,   776,
     784,   790,   798,   806,   827,   834,   841,   848,   857,   862,
     870,   876,   883,   891,   892,   900,   907,   917,   918,   927,
     935,   943,   952,   953,   956,   959,   963,   969,   970,   971,
     974,   975,   979,   983,   987,   991,   997,   998,  1002,  1006,
    1010,  1014,  1018,  1022,  1026,  1030,  1034
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'|'", "'^'", "'&'", "'<'", "'>'", "'+'",
  "'-'", "'*'", "'/'", "'%'", "LMOVW", "LMOVB", "LABS", "LLOGW", "LSHW",
  "LADDW", "LCMP", "LCROP", "LBRA", "LFMOV", "LFCONV", "LFCMP", "LFADD",
  "LFMA", "LTRAP", "LXORW", "LNOP", "LEND", "LRETT", "LWORD", "LTEXT",
  "LGLOBL", "LDATA", "LRETRN", "LCONST", "LSP", "LSB", "LFP", "LPC",
  "LCREG", "LFLUSH", "LREG", "LFREG", "LR", "LCR", "LF", "LFPSCR", "LLR",
  "LCTR", "LSPR", "LSPREG", "LSEG", "LMSR", "LPCDAT", "LFUNCDAT", "LSCHED",
  "LXLD", "LXST", "LXOP", "LXMV", "LRLWM", "LMOVMW", "LMOVEM", "LMOVFL",
  "LMTFSB", "LMA", "LFCONST", "LSCONST", "LNAME", "LLAB", "LVAR", "':'",
  "'='", "';'", "','", "'('", "')'", "'$'", "'~'", "$accept", "prog",
  "line", "@1", "inst", "rel", "rreg", "xlreg", "lr", "lcr", "ctr", "msr",
  "psr", "fpscr", "freg", "creg", "cbit", "mask", "textsize", "ximm",
  "fimm", "imm", "sreg", "regaddr", "addr", "name", "comma", "offset",
  "pointer", "con", "expr", 0
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
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,    58,    61,    59,    44,    40,    41,
      36,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    82,    83,    83,    85,    84,    84,    84,    84,    84,
      84,    84,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    87,    87,    88,    89,    89,
      90,    91,    92,    93,    94,    94,    94,    95,    96,    96,
      97,    97,    98,    99,   100,   100,   100,   100,   101,   101,
     102,   102,   103,   104,   104,   105,   105,   106,   106,   107,
     107,   107,   108,   108,   109,   109,   109,   110,   110,   110,
     111,   111,   111,   111,   111,   111,   112,   112,   112,   112,
     112,   112,   112,   112,   112,   112,   112
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     0,     4,     4,     4,     2,     1,
       2,     2,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     6,     4,     4,     6,     4,     4,     6,     6,
       6,     4,     4,     6,     4,     6,     4,     6,     4,     4,
       2,     6,     4,     4,     4,     6,     4,     4,     4,     4,
       4,     4,     4,     4,     2,     2,     4,     3,     3,     5,
       4,     4,     6,     4,     4,     6,     6,     6,     8,     4,
       4,     3,     2,     4,     4,     6,     8,     4,     6,     4,
       4,     6,     6,     8,     8,     8,     8,     4,     4,     4,
       6,     4,     6,     4,     4,     2,     2,     3,     3,     3,
       3,     2,     3,     3,     4,     4,     2,     5,     7,     4,
       6,     6,     6,     6,     2,     4,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     4,     1,     1,     1,     4,
       1,     4,     1,     3,     1,     2,     3,     4,     2,     2,
       2,     3,     2,     1,     4,     3,     5,     1,     4,     4,
       5,     7,     0,     1,     0,     2,     2,     1,     1,     1,
       1,     1,     2,     2,     2,     3,     1,     3,     3,     3,
       3,     3,     4,     4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     0,     1,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   162,   162,
     162,     0,     0,     0,     0,   162,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     9,
       3,     0,    11,     0,     0,   170,   140,   153,   138,     0,
     131,     0,   137,   130,   132,     0,   134,   133,   164,   171,
       0,     0,     0,     0,     0,   128,     0,   129,   136,     0,
       0,     0,     0,     0,     0,   127,     0,     0,   157,     0,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   142,     0,   164,     0,     0,    64,     0,
      65,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   163,   162,     0,    82,   163,   162,   162,   111,   106,
     116,   162,   162,     0,     0,     0,     0,   124,     0,     0,
       8,     0,     0,     0,   105,     0,     0,     0,     0,     0,
       0,     0,     0,     4,     0,     0,    10,   173,   172,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   176,     0,
     149,   148,   152,   174,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   152,     0,     0,     0,     0,     0,     0,   126,     0,
      67,    68,     0,     0,     0,     0,     0,     0,   150,     0,
       0,     0,     0,     0,     0,     0,     0,   163,    81,     0,
     109,   110,   107,   108,   113,   112,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   164,
     165,   166,     0,     0,   155,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   175,    12,    61,    37,    63,
      36,     0,    25,    24,    60,    58,    59,    57,    30,    33,
      31,     0,    29,    28,    62,    56,    53,    52,    14,    13,
     168,   167,   169,     0,     0,    15,    27,    26,    17,    16,
      49,    44,   127,    46,   127,    48,   127,    41,     0,   127,
      42,   127,    89,    90,    54,   142,     0,    66,     0,    70,
      71,     0,    73,    74,     0,     0,   151,    21,    23,    22,
      20,    19,    18,    83,    87,    84,     0,    79,    80,     0,
       0,   119,     0,     0,   114,   115,    99,     0,     0,   101,
     104,   103,     0,     0,    98,    97,    34,     0,     5,     6,
       7,   154,   141,   139,   135,     0,     0,     0,   186,   185,
     184,     0,     0,   177,   178,   179,   180,   181,     0,     0,
     158,   159,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    69,     0,     0,     0,   125,     0,     0,     0,     0,
     144,   117,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   160,   156,   182,   183,   131,    35,    32,    43,    45,
      47,    40,    38,    39,    91,    92,    55,    72,    75,     0,
      76,    77,    88,    85,     0,   145,     0,     0,   120,     0,
     122,   123,   121,   100,   102,     0,     0,     0,     0,     0,
      51,     0,     0,     0,     0,   146,   118,     0,     0,     0,
       0,     0,     0,   161,    78,    86,   147,    96,    95,   143,
      94,    93
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,    40,   232,    41,    98,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    92,   435,   391,    73,
     104,    74,    75,    76,   161,    78,   114,   156,   284,   158,
     159
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -179
static const yytype_int16 yypact[] =
{
    -179,   484,  -179,   -64,   581,   686,    74,    74,   -24,   -24,
      74,   845,   641,   656,   -29,   -29,   -29,   -29,    19,   -11,
     -54,   -38,   747,   747,   747,   -54,   -19,   -19,   -50,    -6,
      74,    -6,   -14,   -24,   707,   -19,    74,   -36,    18,  -179,
    -179,     2,  -179,   845,   845,  -179,  -179,  -179,  -179,    24,
      27,    48,  -179,  -179,  -179,    61,  -179,  -179,   188,  -179,
     717,   738,   845,    79,    81,  -179,    93,  -179,  -179,    99,
     107,   116,   126,   127,   130,  -179,   132,   133,  -179,    87,
     136,   138,   157,   159,   171,   845,   176,   179,   182,   184,
     186,   845,   194,  -179,    27,   188,   762,   764,  -179,   196,
    -179,    66,     8,   198,   200,   201,   202,   203,   206,   215,
     216,  -179,   217,   219,  -179,   181,   -54,   -54,  -179,  -179,
    -179,   -54,   -54,   220,   167,   221,   178,  -179,   223,   224,
    -179,    74,   225,   226,  -179,   227,   231,   232,   233,   234,
     236,   237,   238,  -179,   845,   845,  -179,  -179,  -179,   845,
     845,   845,   845,   242,   845,   845,   229,     3,  -179,   377,
    -179,  -179,    87,  -179,   629,    74,    74,   172,    26,   732,
      39,    74,    74,    74,    74,   230,   686,    74,    74,    74,
      74,  -179,    74,    74,   -24,    74,   -24,   845,   229,   764,
    -179,  -179,   241,   243,   784,   814,   111,   254,  -179,    67,
     -29,   -29,   -29,   -29,   -29,   -29,   -29,    74,  -179,    74,
    -179,  -179,  -179,  -179,  -179,  -179,   821,    45,   830,   845,
     -19,   747,   -24,    49,    -6,    74,    74,    74,   747,    74,
     845,    74,   548,   463,   518,   246,   247,   248,   249,   155,
    -179,  -179,    45,    74,  -179,   845,   845,   845,   323,   325,
     845,   845,   845,   845,   845,  -179,  -179,  -179,  -179,  -179,
    -179,   259,  -179,  -179,  -179,  -179,  -179,  -179,  -179,  -179,
    -179,   260,  -179,  -179,  -179,  -179,  -179,  -179,  -179,  -179,
    -179,  -179,  -179,   265,   266,  -179,  -179,  -179,  -179,  -179,
    -179,  -179,   269,  -179,   270,  -179,   272,  -179,   278,   279,
    -179,   280,   283,   284,  -179,   285,   275,  -179,   764,  -179,
    -179,   764,  -179,  -179,   105,   286,  -179,  -179,  -179,  -179,
    -179,  -179,  -179,  -179,   289,   296,   297,  -179,  -179,     9,
     299,  -179,   301,   319,  -179,  -179,  -179,   320,   321,  -179,
    -179,  -179,   324,   327,  -179,  -179,  -179,   328,  -179,  -179,
    -179,  -179,  -179,  -179,  -179,   329,   333,   334,   591,   430,
     451,   845,   845,    78,    78,  -179,  -179,  -179,   316,   353,
    -179,  -179,    74,    74,    74,    74,    74,    74,    20,    20,
     845,  -179,   335,   336,   841,  -179,    20,   -29,   -29,   369,
     399,  -179,   338,   -19,   339,    74,    -6,   830,   830,    74,
     382,  -179,  -179,   277,   277,  -179,  -179,  -179,  -179,  -179,
    -179,  -179,  -179,  -179,  -179,  -179,  -179,  -179,  -179,   764,
    -179,  -179,  -179,  -179,   345,   414,   387,     9,  -179,   322,
    -179,  -179,  -179,  -179,  -179,   350,   351,   352,   354,   355,
    -179,   366,   372,   -29,   393,  -179,  -179,   851,    74,    74,
     845,    74,    74,  -179,  -179,  -179,  -179,  -179,  -179,  -179,
    -179,  -179
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -179,  -179,   222,  -179,  -179,   -72,    -5,   -61,  -179,  -157,
    -179,  -179,  -149,  -161,    38,    31,  -178,    50,    28,   -15,
      58,    98,   168,    82,    96,   112,    25,   -85,   211,    36,
      88
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      80,    83,    84,    86,    88,    90,   121,   258,   270,   304,
     188,   243,    42,   112,   116,   259,    48,   197,   389,    51,
      47,   274,    49,   111,   190,   133,   130,   135,   137,   139,
      47,   142,    49,    47,    48,    49,   193,    51,   143,   144,
      79,    79,    61,    99,   119,   120,   390,    93,   101,    79,
     127,   103,   107,   108,   109,   110,    85,   117,   124,   124,
     124,    85,    46,    47,   131,    49,   115,    94,    46,    85,
      79,    48,   131,    94,    51,    43,    44,   198,   146,   147,
     148,    46,   244,   280,   281,   282,    94,    81,   252,   253,
     254,    55,    56,   145,    57,   105,   111,   162,   163,    85,
      77,    82,   149,   257,    45,   150,    87,    89,   100,   106,
     210,   132,    48,   134,   136,    51,   113,   118,    47,   122,
      49,   181,   309,   312,   128,   129,   151,   131,   306,    85,
     140,   138,   192,   141,   123,   125,   126,   208,    58,   152,
      59,   212,   213,   195,   196,    60,   214,   215,    62,   280,
     281,   282,   315,   211,   355,    47,   164,    49,   165,   256,
     264,   265,   266,   154,   155,   175,   276,   277,   278,   279,
     166,   285,   288,   289,   290,   291,   167,   293,   295,   297,
     300,   302,   384,   196,   168,   235,   236,   237,   238,   219,
     240,   241,   191,   169,   153,   260,   154,   155,   267,   269,
      79,   275,   416,   170,   171,    79,   268,   172,   407,   173,
     174,   406,    79,   176,    46,   177,    47,   336,    49,    94,
     341,   342,   343,   305,   345,    47,    48,    49,   157,    51,
     192,   314,   233,   234,   178,    79,   179,   317,   320,   321,
     322,   323,   324,   325,   326,   217,   262,   382,   180,   239,
     383,   272,   330,   182,   332,   333,   183,    79,   286,   184,
     263,   185,   261,   186,    79,   273,   346,   271,   280,   281,
     282,   187,   287,   194,    47,   199,    49,   200,   201,   202,
     203,   318,   298,   204,   303,   250,   251,   252,   253,   254,
     310,   313,   205,   206,   207,   319,   209,   216,   218,   157,
     220,   221,   222,   223,   224,   339,   340,   242,   225,   226,
     227,   228,   420,   229,   230,   231,   331,   335,   334,   196,
     337,   338,   307,   316,   344,   351,   352,   353,   354,   361,
      43,   447,   362,   358,   359,   360,   368,   369,   363,   364,
     365,   366,   367,   283,   370,   371,   372,   373,   292,   374,
     294,   296,   299,   301,   381,   375,   376,   377,   442,    45,
     378,   379,   380,   405,   283,   385,   386,   408,   409,   410,
     411,   412,   413,   387,   388,   327,   392,   328,   393,   430,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     433,   198,   160,    58,   440,    59,   394,   395,   396,   347,
      91,   397,    52,    62,   398,   399,   425,   400,   426,   414,
     415,   357,   401,   402,   417,   418,    93,   422,   427,   429,
     192,   441,   443,   444,   445,   423,   424,   448,   449,   450,
     456,   451,   452,   437,   437,   247,   248,   249,   250,   251,
     252,   253,   254,   457,   458,   453,   460,   461,   438,   403,
     404,   454,   431,   356,   348,   446,   255,   248,   249,   250,
     251,   252,   253,   254,     0,   162,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,     0,     0,   434,     0,
     421,   455,     0,   148,     2,     3,   459,     0,     0,     0,
       0,   428,   432,     0,     0,   436,   439,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,     0,    19,    20,     0,    21,    22,    23,    24,
      25,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,     0,     0,     0,     0,     0,     0,     0,     0,   349,
      26,    27,    28,    29,    30,    31,    32,    33,    34,     3,
       0,    35,    36,     0,     0,    37,     0,    38,     0,     0,
      39,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,     0,    19,    20,     0,
      21,    22,    23,    24,    25,     0,     0,     0,     0,    43,
      44,     0,     0,     0,   350,   246,   247,   248,   249,   250,
     251,   252,   253,   254,    26,    27,    28,    29,    30,    31,
      32,    33,    34,     0,     0,    35,    36,     0,    45,    37,
       0,    38,     0,    46,    39,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,     0,    57,    43,    44,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,    58,     0,    59,     0,     0,     0,     0,    60,
       0,    61,    62,     0,    43,    44,    45,     0,     0,     0,
       0,    46,     0,    47,     0,    49,    50,     0,    45,    53,
      54,    55,    56,    46,    57,     0,     0,     0,    94,     0,
       0,     0,     0,    45,    43,    44,     0,     0,     0,     0,
      58,    48,    59,     0,    51,     0,     0,    60,     0,    85,
      62,     0,    95,     0,    59,    43,    44,     0,    96,    97,
       0,     0,    62,    45,     0,    43,    44,    58,     0,    59,
      47,     0,    49,     0,    60,     0,   102,    62,     0,     0,
      43,    44,     0,     0,    45,     0,    43,    44,     0,     0,
       0,    47,     0,    49,    45,    43,    44,    58,     0,    59,
       0,    47,     0,    49,    60,     0,     0,    62,     0,    45,
      43,    44,    43,    44,     0,    45,     0,     0,    58,     0,
      59,    52,     0,     0,    45,    91,     0,     0,    62,     0,
      59,     0,    43,    44,     0,    91,     0,     0,    62,    45,
       0,    45,     0,    58,     0,    59,     0,     0,   160,    58,
      60,    59,    85,    62,    53,    54,    91,     0,    58,    62,
      59,    45,    43,    44,     0,    91,     0,     0,    62,    43,
      44,     0,     0,    95,     0,    59,     0,    59,    43,    44,
     189,     0,    91,    62,     0,    62,     0,     0,     0,    43,
      44,    45,     0,    43,    44,    95,     0,    59,    45,    43,
      44,     0,   308,     0,     0,    62,     0,    45,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    45,     0,
       0,     0,    45,     0,     0,    95,     0,    59,    45,     0,
       0,     0,   311,     0,    59,    62,     0,     0,     0,    91,
       0,   329,    62,    59,     0,     0,     0,     0,    91,     0,
      85,    62,    95,     0,    59,     0,     0,     0,    59,   419,
     316,     0,    62,    91,    59,     0,    62,     0,     0,    91,
       0,     0,    62
};

static const yytype_int16 yycheck[] =
{
       5,     6,     7,     8,     9,    10,    21,   164,   169,   187,
      95,     8,    76,    18,    19,   164,    45,     9,     9,    48,
      44,   170,    46,    77,    96,    30,    76,    32,    33,    34,
      44,    36,    46,    44,    45,    46,    97,    48,    74,    75,
       4,     5,    80,    12,    19,    20,    37,    11,    12,    13,
      25,    13,    14,    15,    16,    17,    80,    19,    22,    23,
      24,    80,    42,    44,    78,    46,    77,    47,    42,    80,
      34,    45,    78,    47,    48,     8,     9,    69,    76,    43,
      44,    42,    79,    38,    39,    40,    47,     5,    10,    11,
      12,    52,    53,    75,    55,    13,    77,    61,    62,    80,
       4,     5,    78,   164,    37,    78,     8,     9,    12,    13,
     115,    29,    45,    31,    32,    48,    18,    19,    44,    21,
      46,    85,   194,   195,    26,    27,    78,    78,   189,    80,
      34,    33,    96,    35,    22,    23,    24,   112,    71,    78,
      73,   116,   117,    77,    78,    78,   121,   122,    81,    38,
      39,    40,    41,   115,   239,    44,    77,    46,    77,   164,
     165,   166,   167,     8,     9,    78,   171,   172,   173,   174,
      77,   176,   177,   178,   179,   180,    77,   182,   183,   184,
     185,   186,    77,    78,    77,   149,   150,   151,   152,    11,
     154,   155,    96,    77,     6,   164,     8,     9,   167,   168,
     164,   170,   380,    77,    77,   169,   168,    77,   369,    77,
      77,   368,   176,    77,    42,    77,    44,   222,    46,    47,
     225,   226,   227,   187,   229,    44,    45,    46,    60,    48,
     194,   195,   144,   145,    77,   199,    77,   199,   200,   201,
     202,   203,   204,   205,   206,    78,   164,   308,    77,     7,
     311,   169,   216,    77,   218,   219,    77,   221,   176,    77,
     164,    77,   164,    77,   228,   169,   230,   169,    38,    39,
      40,    77,   176,    77,    44,    77,    46,    77,    77,    77,
      77,   199,   184,    77,   186,     8,     9,    10,    11,    12,
     194,   195,    77,    77,    77,   199,    77,    77,    77,   131,
      77,    77,    77,    77,    77,   223,   224,    78,    77,    77,
      77,    77,   384,    77,    77,    77,   218,   221,   220,    78,
     222,   223,    79,    69,   228,    79,    79,    79,    79,     6,
       8,     9,     7,   245,   246,   247,    77,    77,   250,   251,
     252,   253,   254,   175,    79,    79,    77,    77,   180,    77,
     182,   183,   184,   185,    79,    77,    77,    77,   419,    37,
      77,    77,    77,    47,   196,    79,    77,   372,   373,   374,
     375,   376,   377,    77,    77,   207,    77,   209,    77,   394,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
     395,    69,    70,    71,   399,    73,    77,    77,    77,   231,
      78,    77,    49,    81,    77,    77,    37,    78,     9,   378,
     379,   243,    79,    79,    79,    79,   380,   386,    80,    80,
     384,    39,    77,     9,    37,   387,   388,    77,    77,    77,
      37,    77,    77,   397,   398,     5,     6,     7,     8,     9,
      10,    11,    12,   448,   449,    79,   451,   452,   398,   361,
     362,    79,   394,   242,   232,   427,    79,     6,     7,     8,
       9,    10,    11,    12,    -1,   429,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    -1,    -1,   396,    -1,
     384,   443,    -1,   447,     0,     1,   450,    -1,    -1,    -1,
      -1,   393,   394,    -1,    -1,   397,   398,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    -1,    29,    30,    -1,    32,    33,    34,    35,
      36,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      56,    57,    58,    59,    60,    61,    62,    63,    64,     1,
      -1,    67,    68,    -1,    -1,    71,    -1,    73,    -1,    -1,
      76,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    -1,    29,    30,    -1,
      32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,     8,
       9,    -1,    -1,    -1,    76,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    -1,    -1,    67,    68,    -1,    37,    71,
      -1,    73,    -1,    42,    76,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    -1,    55,     8,     9,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     8,
       9,    -1,    71,    -1,    73,    -1,    -1,    -1,    -1,    78,
      -1,    80,    81,    -1,     8,     9,    37,    -1,    -1,    -1,
      -1,    42,    -1,    44,    -1,    46,    47,    -1,    37,    50,
      51,    52,    53,    42,    55,    -1,    -1,    -1,    47,    -1,
      -1,    -1,    -1,    37,     8,     9,    -1,    -1,    -1,    -1,
      71,    45,    73,    -1,    48,    -1,    -1,    78,    -1,    80,
      81,    -1,    71,    -1,    73,     8,     9,    -1,    77,    78,
      -1,    -1,    81,    37,    -1,     8,     9,    71,    -1,    73,
      44,    -1,    46,    -1,    78,    -1,    80,    81,    -1,    -1,
       8,     9,    -1,    -1,    37,    -1,     8,     9,    -1,    -1,
      -1,    44,    -1,    46,    37,     8,     9,    71,    -1,    73,
      -1,    44,    -1,    46,    78,    -1,    -1,    81,    -1,    37,
       8,     9,     8,     9,    -1,    37,    -1,    -1,    71,    -1,
      73,    49,    -1,    -1,    37,    78,    -1,    -1,    81,    -1,
      73,    -1,     8,     9,    -1,    78,    -1,    -1,    81,    37,
      -1,    37,    -1,    71,    -1,    73,    -1,    -1,    70,    71,
      78,    73,    80,    81,    50,    51,    78,    -1,    71,    81,
      73,    37,     8,     9,    -1,    78,    -1,    -1,    81,     8,
       9,    -1,    -1,    71,    -1,    73,    -1,    73,     8,     9,
      78,    -1,    78,    81,    -1,    81,    -1,    -1,    -1,     8,
       9,    37,    -1,     8,     9,    71,    -1,    73,    37,     8,
       9,    -1,    78,    -1,    -1,    81,    -1,    37,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,
      -1,    -1,    37,    -1,    -1,    71,    -1,    73,    37,    -1,
      -1,    -1,    78,    -1,    73,    81,    -1,    -1,    -1,    78,
      -1,    80,    81,    73,    -1,    -1,    -1,    -1,    78,    -1,
      80,    81,    71,    -1,    73,    -1,    -1,    -1,    73,    78,
      69,    -1,    81,    78,    73,    -1,    81,    -1,    -1,    78,
      -1,    -1,    81
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    83,     0,     1,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    29,
      30,    32,    33,    34,    35,    36,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    67,    68,    71,    73,    76,
      84,    86,    76,     8,     9,    37,    42,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    55,    71,    73,
      78,    80,    81,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,   101,   103,   104,   105,   106,   107,   111,
      88,   105,   106,    88,    88,    80,    88,   103,    88,   103,
      88,    78,    98,   111,    47,    71,    77,    78,    87,    97,
     106,   111,    80,    96,   102,   105,   106,    96,    96,    96,
      96,    77,    88,   103,   108,    77,    88,    96,   103,   108,
     108,   101,   103,   107,   111,   107,   107,   108,   103,   103,
      76,    78,   105,    88,   105,    88,   105,    88,   103,    88,
     106,   103,    88,    74,    75,    75,    76,   111,   111,    78,
      78,    78,    78,     6,     8,     9,   109,   104,   111,   112,
      70,   106,   111,   111,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    78,    77,    77,    77,    77,
      77,   111,    77,    77,    77,    77,    77,    77,   109,    78,
      87,   106,   111,    89,    77,    77,    78,     9,    69,    77,
      77,    77,    77,    77,    77,    77,    77,    77,   108,    77,
      88,    96,   108,   108,   108,   108,    77,    78,    77,    11,
      77,    77,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    85,   112,   112,   111,   111,   111,   111,     7,
     111,   111,    78,     8,    79,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    79,    88,    89,    91,    94,
      97,   103,   105,   106,    88,    88,    88,    97,    96,    97,
      95,   103,   105,   106,    94,    97,    88,    88,    88,    88,
      38,    39,    40,   104,   110,    88,   105,   106,    88,    88,
      88,    88,   104,    88,   104,    88,   104,    88,   103,   104,
      88,   104,    88,   103,    98,   111,    89,    79,    78,    87,
     106,    78,    87,   106,   111,    41,    69,    96,   105,   106,
      96,    96,    96,    96,    96,    96,    96,   104,   104,    80,
     111,   103,   111,   111,   103,   106,    88,   103,   103,   105,
     105,    88,    88,    88,   106,    88,   111,   104,    84,    76,
      76,    79,    79,    79,    79,   109,   110,   104,   112,   112,
     112,     6,     7,   112,   112,   112,   112,   112,    77,    77,
      79,    79,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    79,    89,    89,    77,    79,    77,    77,    77,     9,
      37,   100,    77,    77,    77,    77,    77,    77,    77,    77,
      78,    79,    79,   112,   112,    47,    91,    95,    88,    88,
      88,    88,    88,    88,    97,    97,    98,    79,    79,    78,
      87,   106,    97,    96,    96,    37,     9,    80,   103,    80,
     101,   102,   103,    88,   105,    99,   103,   111,    99,   103,
      88,    39,    89,    77,     9,    37,   100,     9,    77,    77,
      77,    77,    77,    79,    79,    96,    37,    88,    88,   111,
      88,    88
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
#line 71 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		if((yyvsp[(1) - (2)].sym)->type == LLAB && (yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 6:
#line 80 "a.y"
    {
		(yyvsp[(1) - (4)].sym)->type = LVAR;
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 7:
#line 85 "a.y"
    {
		if((yyvsp[(1) - (4)].sym)->value != (yyvsp[(3) - (4)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (4)].sym)->name);
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 8:
#line 91 "a.y"
    {
		nosched = (yyvsp[(1) - (2)].lval);
	}
    break;

  case 12:
#line 103 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 13:
#line 107 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 14:
#line 111 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 15:
#line 115 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 16:
#line 119 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 17:
#line 123 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 18:
#line 130 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 19:
#line 134 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 20:
#line 138 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 21:
#line 142 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 22:
#line 146 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 23:
#line 150 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 24:
#line 157 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 25:
#line 161 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 26:
#line 165 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 27:
#line 169 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 28:
#line 176 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 29:
#line 180 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 30:
#line 187 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 31:
#line 191 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 32:
#line 195 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 33:
#line 199 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 34:
#line 203 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].lval), &nullgen);
	}
    break;

  case 35:
#line 210 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 36:
#line 214 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 37:
#line 218 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 38:
#line 228 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 39:
#line 232 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 40:
#line 236 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 41:
#line 240 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 42:
#line 244 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 43:
#line 248 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 44:
#line 252 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 45:
#line 256 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 46:
#line 260 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 47:
#line 264 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 48:
#line 268 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 49:
#line 272 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 50:
#line 276 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), 0, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 51:
#line 283 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 52:
#line 290 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 53:
#line 294 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 54:
#line 301 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].addr).reg, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 55:
#line 305 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 56:
#line 313 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 57:
#line 317 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 58:
#line 321 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 59:
#line 325 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 60:
#line 329 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 61:
#line 333 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 62:
#line 337 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 63:
#line 341 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 64:
#line 350 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 65:
#line 354 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 66:
#line 358 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &nullgen, 0, &(yyvsp[(3) - (4)].addr));
	}
    break;

  case 67:
#line 362 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 68:
#line 366 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 69:
#line 370 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), &nullgen, 0, &(yyvsp[(4) - (5)].addr));
	}
    break;

  case 70:
#line 374 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 71:
#line 378 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 72:
#line 382 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(5) - (6)].addr));
	}
    break;

  case 73:
#line 386 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &nullgen, (yyvsp[(2) - (4)].lval), &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 74:
#line 390 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &nullgen, (yyvsp[(2) - (4)].lval), &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 75:
#line 394 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &nullgen, (yyvsp[(2) - (6)].lval), &(yyvsp[(5) - (6)].addr));
	}
    break;

  case 76:
#line 398 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = (yyvsp[(2) - (6)].lval);
		outcode((yyvsp[(1) - (6)].lval), &g, REG_R0+(yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 77:
#line 406 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = (yyvsp[(2) - (6)].lval);
		outcode((yyvsp[(1) - (6)].lval), &g, REG_R0+(yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 78:
#line 414 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = (yyvsp[(2) - (8)].lval);
		outcode((yyvsp[(1) - (8)].lval), &g, REG_R0+(yyvsp[(4) - (8)].lval), &(yyvsp[(7) - (8)].addr));
	}
    break;

  case 79:
#line 425 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].lval), &nullgen);
	}
    break;

  case 80:
#line 429 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].lval), &nullgen);
	}
    break;

  case 81:
#line 433 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), 0, &nullgen);
	}
    break;

  case 82:
#line 437 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &nullgen);
	}
    break;

  case 83:
#line 444 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 84:
#line 448 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 85:
#line 452 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].addr).reg, &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 86:
#line 456 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 87:
#line 460 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 88:
#line 464 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(4) - (6)].addr));
	}
    break;

  case 89:
#line 471 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 90:
#line 475 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 91:
#line 479 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(4) - (6)].addr));
	}
    break;

  case 92:
#line 483 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(4) - (6)].addr));
	}
    break;

  case 93:
#line 490 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 94:
#line 494 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 95:
#line 498 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 96:
#line 502 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 97:
#line 509 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 98:
#line 513 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 99:
#line 521 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 100:
#line 525 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 101:
#line 529 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 102:
#line 533 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 103:
#line 537 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 104:
#line 541 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 105:
#line 545 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), 0, &nullgen);
	}
    break;

  case 106:
#line 552 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &nullgen);
	}
    break;

  case 107:
#line 556 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), 0, &nullgen);
	}
    break;

  case 108:
#line 560 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), 0, &nullgen);
	}
    break;

  case 109:
#line 564 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 110:
#line 568 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 111:
#line 572 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), 0, &nullgen);
	}
    break;

  case 112:
#line 579 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), 0, &nullgen);
	}
    break;

  case 113:
#line 583 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), 0, &nullgen);
	}
    break;

  case 114:
#line 590 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != TYPE_CONST || (yyvsp[(4) - (4)].addr).type != TYPE_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 115:
#line 599 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != TYPE_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(4) - (4)].addr).type != NAME_EXTERN && (yyvsp[(4) - (4)].addr).type != NAME_STATIC && (yyvsp[(4) - (4)].addr).type != TYPE_MEM)
			yyerror("value for FUNCDATA must be symbol reference");
 		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 116:
#line 610 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &nullgen);
	}
    break;

  case 117:
#line 617 "a.y"
    {
		settext((yyvsp[(2) - (5)].addr).sym);
		outcode((yyvsp[(1) - (5)].lval), &(yyvsp[(2) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 118:
#line 622 "a.y"
    {
		settext((yyvsp[(2) - (7)].addr).sym);
		outcode((yyvsp[(1) - (7)].lval), &(yyvsp[(2) - (7)].addr), 0, &(yyvsp[(7) - (7)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (7)].lval);
		}
	}
    break;

  case 119:
#line 634 "a.y"
    {
		settext((yyvsp[(2) - (4)].addr).sym);
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 120:
#line 639 "a.y"
    {
		settext((yyvsp[(2) - (6)].addr).sym);
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 121:
#line 652 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 122:
#line 660 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 123:
#line 668 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 124:
#line 679 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &nullgen);
	}
    break;

  case 125:
#line 685 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 126:
#line 691 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		(yyval.addr) = nullgen;
		if(pass == 2 && (yyvsp[(1) - (2)].sym)->type != LLAB)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 127:
#line 702 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 130:
#line 714 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 131:
#line 722 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);	/* whole register */
	}
    break;

  case 132:
#line 730 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 133:
#line 738 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 134:
#line 746 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 135:
#line 752 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= 1024)
			yyerror("SPR/DCR out of range");
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (4)].lval) + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 137:
#line 763 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 138:
#line 771 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 139:
#line 777 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_F0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 140:
#line 785 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 141:
#line 791 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_C0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 142:
#line 799 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 143:
#line 807 "a.y"
    {
		int mb, me;
		uint32 v;

		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		mb = (yyvsp[(1) - (3)].lval);
		me = (yyvsp[(3) - (3)].lval);
		if(mb < 0 || mb > 31 || me < 0 || me > 31){
			yyerror("illegal mask start/end value(s)");
			mb = me = 0;
		}
		if(mb <= me)
			v = ((uint32)~0L>>mb) & (~0L<<(31-me));
		else
			v = ~(((uint32)~0L>>(me+1)) & (~0L<<(31-(mb-1))));
		(yyval.addr).offset = v;
	}
    break;

  case 144:
#line 828 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 145:
#line 835 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (2)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 146:
#line 842 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (3)].lval);
		(yyval.addr).u.argsize = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 147:
#line 849 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (4)].lval);
		(yyval.addr).u.argsize = (yyvsp[(4) - (4)].lval);
	}
    break;

  case 148:
#line 858 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = TYPE_ADDR;
	}
    break;

  case 149:
#line 863 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 150:
#line 871 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 151:
#line 877 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 152:
#line 884 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 154:
#line 893 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = REG_R0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 155:
#line 901 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 156:
#line 908 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 158:
#line 919 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 159:
#line 928 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(3) - (4)].lval);
		(yyval.addr).sym = nil;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 160:
#line 936 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 161:
#line 944 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = NAME_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 164:
#line 956 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 165:
#line 960 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 166:
#line 964 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 171:
#line 976 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 172:
#line 980 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 173:
#line 984 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 174:
#line 988 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 175:
#line 992 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 177:
#line 999 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 178:
#line 1003 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 179:
#line 1007 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 180:
#line 1011 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 181:
#line 1015 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 182:
#line 1019 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 183:
#line 1023 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 184:
#line 1027 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 185:
#line 1031 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 186:
#line 1035 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 3256 "y.tab.c"
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



