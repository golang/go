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
#define YYLAST   862

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  82
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  30
/* YYNRULES -- Number of rules.  */
#define YYNRULES  183
/* YYNRULES -- Number of states.  */
#define YYNSTATES  455

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
     571,   575,   578,   582,   586,   591,   596,   599,   604,   611,
     620,   625,   632,   639,   646,   653,   656,   661,   664,   666,
     668,   670,   672,   674,   676,   678,   680,   685,   687,   689,
     691,   696,   698,   703,   705,   709,   712,   715,   718,   722,
     725,   727,   732,   736,   742,   744,   749,   754,   760,   768,
     769,   771,   772,   775,   778,   780,   782,   784,   786,   788,
     791,   794,   797,   801,   803,   807,   811,   815,   819,   823,
     828,   833,   837,   841
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      83,     0,    -1,    -1,    83,    84,    -1,    -1,    71,    74,
      85,    84,    -1,    71,    75,   111,    76,    -1,    73,    75,
     111,    76,    -1,    58,    76,    -1,    76,    -1,    86,    76,
      -1,     1,    76,    -1,    13,    88,    77,    88,    -1,    13,
     105,    77,    88,    -1,    13,   104,    77,    88,    -1,    14,
      88,    77,    88,    -1,    14,   105,    77,    88,    -1,    14,
     104,    77,    88,    -1,    22,   105,    77,    96,    -1,    22,
     104,    77,    96,    -1,    22,   101,    77,    96,    -1,    22,
      96,    77,    96,    -1,    22,    96,    77,   105,    -1,    22,
      96,    77,   104,    -1,    13,    88,    77,   105,    -1,    13,
      88,    77,   104,    -1,    14,    88,    77,   105,    -1,    14,
      88,    77,   104,    -1,    13,    96,    77,   105,    -1,    13,
      96,    77,   104,    -1,    13,    95,    77,    96,    -1,    13,
      96,    77,    95,    -1,    13,    96,    77,   102,    77,    95,
      -1,    13,    95,    77,    97,    -1,    67,   102,    77,   110,
      -1,    13,    88,    77,   102,    77,    91,    -1,    13,    88,
      77,    97,    -1,    13,    88,    77,    91,    -1,    18,    88,
      77,   103,    77,    88,    -1,    18,   102,    77,   103,    77,
      88,    -1,    18,    88,    77,   102,    77,    88,    -1,    18,
      88,    77,    88,    -1,    18,   102,    77,    88,    -1,    16,
      88,    77,   103,    77,    88,    -1,    16,    88,    77,    88,
      -1,    17,    88,    77,   103,    77,    88,    -1,    17,    88,
      77,    88,    -1,    17,   102,    77,   103,    77,    88,    -1,
      17,   102,    77,    88,    -1,    15,    88,    77,    88,    -1,
      15,    88,    -1,    68,    88,    77,   103,    77,    88,    -1,
      13,   102,    77,    88,    -1,    13,   100,    77,    88,    -1,
      20,    98,    77,    98,    -1,    20,    98,    77,   110,    77,
      98,    -1,    13,    97,    77,    97,    -1,    13,    94,    77,
      97,    -1,    13,    91,    77,    88,    -1,    13,    94,    77,
      88,    -1,    13,    89,    77,    88,    -1,    13,    88,    77,
      89,    -1,    13,    97,    77,    94,    -1,    13,    88,    77,
      94,    -1,    21,    87,    -1,    21,   105,    -1,    21,    78,
      89,    79,    -1,    21,    77,    87,    -1,    21,    77,   105,
      -1,    21,    77,    78,    89,    79,    -1,    21,    97,    77,
      87,    -1,    21,    97,    77,   105,    -1,    21,    97,    77,
      78,    89,    79,    -1,    21,   110,    77,    87,    -1,    21,
     110,    77,   105,    -1,    21,   110,    77,    78,    89,    79,
      -1,    21,   110,    77,   110,    77,    87,    -1,    21,   110,
      77,   110,    77,   105,    -1,    21,   110,    77,   110,    77,
      78,    89,    79,    -1,    27,    88,    77,   103,    -1,    27,
     102,    77,   103,    -1,    27,    88,   107,    -1,    27,   107,
      -1,    23,    96,    77,    96,    -1,    25,    96,    77,    96,
      -1,    25,    96,    77,    96,    77,    96,    -1,    26,    96,
      77,    96,    77,    96,    77,    96,    -1,    24,    96,    77,
      96,    -1,    24,    96,    77,    96,    77,    97,    -1,    19,
      88,    77,    88,    -1,    19,    88,    77,   102,    -1,    19,
      88,    77,    88,    77,    97,    -1,    19,    88,    77,   102,
      77,    97,    -1,    63,   102,    77,    88,    77,   102,    77,
      88,    -1,    63,   102,    77,    88,    77,    99,    77,    88,
      -1,    63,    88,    77,    88,    77,   102,    77,    88,    -1,
      63,    88,    77,    88,    77,    99,    77,    88,    -1,    64,
     105,    77,    88,    -1,    64,    88,    77,   105,    -1,    59,
     104,    77,    88,    -1,    59,   104,    77,   102,    77,    88,
      -1,    60,    88,    77,   104,    -1,    60,    88,    77,   102,
      77,   104,    -1,    62,   104,    77,    88,    -1,    62,    88,
      77,   104,    -1,    61,   104,    -1,    29,   107,    -1,    29,
      88,   107,    -1,    29,    96,   107,    -1,    29,    77,    88,
      -1,    29,    77,    96,    -1,    29,   102,    -1,    32,   102,
     107,    -1,    32,   100,   107,    -1,    56,   102,    77,   102,
      -1,    57,   102,    77,   105,    -1,    30,   107,    -1,    33,
     106,    77,   102,    -1,    33,   106,    77,   110,    77,   102,
      -1,    33,   106,    77,   110,    77,   102,     9,   110,    -1,
      34,   106,    77,   102,    -1,    34,   106,    77,   110,    77,
     102,    -1,    35,   106,    11,   110,    77,   102,    -1,    35,
     106,    11,   110,    77,   100,    -1,    35,   106,    11,   110,
      77,   101,    -1,    36,   107,    -1,   110,    78,    41,    79,
      -1,    71,   108,    -1,   103,    -1,    90,    -1,    92,    -1,
      50,    -1,    47,    -1,    51,    -1,    55,    -1,    53,    -1,
      52,    78,   110,    79,    -1,    93,    -1,    49,    -1,    45,
      -1,    48,    78,   110,    79,    -1,    42,    -1,    47,    78,
     110,    79,    -1,   110,    -1,   110,    77,   110,    -1,    80,
     105,    -1,    80,    70,    -1,    80,    69,    -1,    80,     9,
      69,    -1,    80,   110,    -1,    44,    -1,    46,    78,   110,
      79,    -1,    78,   103,    79,    -1,    78,   103,     8,   103,
      79,    -1,   106,    -1,   110,    78,   103,    79,    -1,   110,
      78,   109,    79,    -1,    71,   108,    78,   109,    79,    -1,
      71,     6,     7,   108,    78,    39,    79,    -1,    -1,    77,
      -1,    -1,     8,   110,    -1,     9,   110,    -1,    39,    -1,
      38,    -1,    40,    -1,    37,    -1,    73,    -1,     9,   110,
      -1,     8,   110,    -1,    81,   110,    -1,    78,   111,    79,
      -1,   110,    -1,   111,     8,   111,    -1,   111,     9,   111,
      -1,   111,    10,   111,    -1,   111,    11,   111,    -1,   111,
      12,   111,    -1,   111,     6,     6,   111,    -1,   111,     7,
       7,   111,    -1,   111,     5,   111,    -1,   111,     4,   111,
      -1,   111,     3,   111,    -1
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
     567,   571,   578,   582,   589,   598,   609,   616,   621,   628,
     638,   643,   652,   656,   660,   667,   673,   679,   690,   698,
     699,   702,   710,   718,   726,   734,   740,   748,   751,   759,
     765,   773,   779,   787,   795,   816,   821,   829,   835,   842,
     850,   851,   859,   866,   876,   877,   886,   894,   902,   911,
     912,   915,   918,   922,   928,   929,   930,   933,   934,   938,
     942,   946,   950,   956,   957,   961,   965,   969,   973,   977,
     981,   985,   989,   993
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
  "psr", "fpscr", "freg", "creg", "cbit", "mask", "ximm", "fimm", "imm",
  "sreg", "regaddr", "addr", "name", "comma", "offset", "pointer", "con",
  "expr", 0
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
      86,    86,    86,    86,    86,    86,    87,    87,    88,    89,
      89,    90,    91,    92,    93,    94,    94,    94,    95,    96,
      96,    97,    97,    98,    99,   100,   100,   101,   101,   102,
     103,   103,   104,   104,   105,   105,   106,   106,   106,   107,
     107,   108,   108,   108,   109,   109,   109,   110,   110,   110,
     110,   110,   110,   111,   111,   111,   111,   111,   111,   111,
     111,   111,   111,   111
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
       3,     2,     3,     3,     4,     4,     2,     4,     6,     8,
       4,     6,     6,     6,     6,     2,     4,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     4,     1,     1,     1,
       4,     1,     4,     1,     3,     2,     2,     2,     3,     2,
       1,     4,     3,     5,     1,     4,     4,     5,     7,     0,
       1,     0,     2,     2,     1,     1,     1,     1,     1,     2,
       2,     2,     3,     1,     3,     3,     3,     3,     3,     4,
       4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     0,     1,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   159,   159,
     159,     0,     0,     0,     0,   159,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     9,
       3,     0,    11,     0,     0,   167,   141,   150,   139,     0,
     132,     0,   138,   131,   133,     0,   135,   134,   161,   168,
       0,     0,     0,     0,     0,   129,     0,   130,   137,     0,
       0,     0,     0,     0,     0,   128,     0,     0,   154,     0,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   143,     0,   161,     0,     0,    64,     0,
      65,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   160,   159,     0,    82,   160,   159,   159,   111,   106,
     116,   159,   159,     0,     0,     0,     0,   125,     0,     0,
       8,     0,     0,     0,   105,     0,     0,     0,     0,     0,
       0,     0,     0,     4,     0,     0,    10,   170,   169,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   173,     0,
     146,   145,   149,   171,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   149,     0,     0,     0,     0,     0,     0,   127,     0,
      67,    68,     0,     0,     0,     0,     0,     0,   147,     0,
       0,     0,     0,     0,     0,     0,     0,   160,    81,     0,
     109,   110,   107,   108,   113,   112,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   161,
     162,   163,     0,     0,   152,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   172,    12,    61,    37,    63,
      36,     0,    25,    24,    60,    58,    59,    57,    30,    33,
      31,     0,    29,    28,    62,    56,    53,    52,    14,    13,
     165,   164,   166,     0,     0,    15,    27,    26,    17,    16,
      49,    44,   128,    46,   128,    48,   128,    41,     0,   128,
      42,   128,    89,    90,    54,   143,     0,    66,     0,    70,
      71,     0,    73,    74,     0,     0,   148,    21,    23,    22,
      20,    19,    18,    83,    87,    84,     0,    79,    80,   117,
       0,   120,     0,     0,   114,   115,    99,     0,     0,   101,
     104,   103,     0,     0,    98,    97,    34,     0,     5,     6,
       7,   151,   142,   140,   136,     0,     0,     0,   183,   182,
     181,     0,     0,   174,   175,   176,   177,   178,     0,     0,
     155,   156,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    69,     0,     0,     0,   126,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   157,   153,
     179,   180,   132,    35,    32,    43,    45,    47,    40,    38,
      39,    91,    92,    55,    72,    75,     0,    76,    77,    88,
      85,     0,   118,   121,     0,   123,   124,   122,   100,   102,
       0,     0,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   158,    78,    86,   119,
      96,    95,   144,    94,    93
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,    40,   232,    41,    98,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    92,   430,    73,   104,
      74,    75,    76,   161,    78,   114,   156,   284,   158,   159
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -178
static const yytype_int16 yypact[] =
{
    -178,   494,  -178,   -50,   591,    30,    -1,    -1,    20,    20,
      -1,   317,   315,   615,    17,    17,    17,    17,    48,     9,
     -42,   -17,   720,   720,   720,   -42,   -12,   -12,    26,    -3,
      -1,    -3,     0,    20,   639,   -12,    -1,    63,    13,  -178,
    -178,    33,  -178,   317,   317,  -178,  -178,  -178,  -178,    38,
      42,    52,  -178,  -178,  -178,    55,  -178,  -178,   198,  -178,
     690,   696,   317,    35,    72,  -178,    76,  -178,  -178,    81,
      93,    99,   105,   107,   117,  -178,   124,   131,  -178,   135,
     155,   160,   161,   164,   165,   317,   171,   172,   176,   179,
     182,   317,   184,  -178,    42,   198,   743,   722,  -178,   185,
    -178,   121,     3,   200,   202,   204,   209,   214,   215,   218,
     220,  -178,   223,   225,  -178,   181,   -42,   -42,  -178,  -178,
    -178,   -42,   -42,   230,   180,   233,    79,  -178,   234,   238,
    -178,    -1,   240,   250,  -178,   251,   253,   254,   255,   256,
     264,   266,   269,  -178,   317,   317,  -178,  -178,  -178,   317,
     317,   317,   317,   277,   317,   317,   271,    34,  -178,    12,
    -178,  -178,   135,  -178,   371,    -1,    -1,   259,   109,   645,
     192,    -1,    -1,    -1,    -1,   320,    30,    -1,    -1,    -1,
      -1,  -178,    -1,    -1,    20,    -1,    20,   317,   271,   722,
    -178,  -178,   275,   265,   747,   767,   106,   286,  -178,   666,
      17,    17,    17,    17,    17,    17,    17,    -1,  -178,    -1,
    -178,  -178,  -178,  -178,  -178,  -178,   267,   125,   267,   317,
     -12,   720,    20,    -7,    -3,    -1,    -1,    -1,   720,    -1,
     317,    -1,   558,   458,   532,   296,   299,   302,   303,   207,
    -178,  -178,   125,    -1,  -178,   317,   317,   317,   350,   358,
     317,   317,   317,   317,   317,  -178,  -178,  -178,  -178,  -178,
    -178,   300,  -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178,
    -178,   306,  -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178,
    -178,  -178,  -178,   305,   310,  -178,  -178,  -178,  -178,  -178,
    -178,  -178,   322,  -178,   323,  -178,   324,  -178,   328,   329,
    -178,   330,   332,   333,  -178,   335,   340,  -178,   722,  -178,
    -178,   722,  -178,  -178,   141,   346,  -178,  -178,  -178,  -178,
    -178,  -178,  -178,  -178,   337,   351,   352,  -178,  -178,  -178,
     355,  -178,   356,   364,  -178,  -178,  -178,   368,   370,  -178,
    -178,  -178,   373,   376,  -178,  -178,  -178,   377,  -178,  -178,
    -178,  -178,  -178,  -178,  -178,   316,   348,   369,   825,   850,
     493,   317,   317,    40,    40,  -178,  -178,  -178,   388,   409,
    -178,  -178,    -1,    -1,    -1,    -1,    -1,    -1,    14,    14,
     317,  -178,   380,   392,   773,  -178,    14,    17,    17,   -12,
     -12,   393,    -1,    -3,   267,   267,    -1,   433,  -178,  -178,
     472,   472,  -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178,
    -178,  -178,  -178,  -178,  -178,  -178,   722,  -178,  -178,  -178,
    -178,   397,   468,  -178,   672,  -178,  -178,  -178,  -178,  -178,
     401,   411,   412,   415,   416,  -178,   417,   418,    17,   317,
     365,    -1,    -1,   317,    -1,    -1,  -178,  -178,  -178,  -178,
    -178,  -178,  -178,  -178,  -178
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -178,  -178,   290,  -178,  -178,   -88,    -5,   -65,  -178,  -157,
    -178,  -178,  -134,  -160,    68,    25,  -177,   130,   -15,   140,
      96,   154,    64,   114,   298,   177,   -84,   291,    36,  -111
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      80,    83,    84,    86,    88,    90,   121,   258,   190,   270,
     304,   188,   197,   112,   116,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   133,    42,   135,   137,   139,
     259,   142,   193,   233,   234,   111,   274,    99,    43,    44,
      79,    79,   243,    47,    47,    49,    49,    93,   101,    79,
     252,   253,   254,    47,    48,    49,    46,    51,   124,   124,
     124,    94,    48,    61,    47,    51,    49,    45,    85,    81,
      79,   131,   198,    85,    47,   131,    49,   105,   131,   147,
     148,   103,   107,   108,   109,   110,   115,   117,   145,    85,
     219,   255,    47,   132,    49,   134,   136,   162,   163,   257,
      85,    58,   130,    59,    87,    89,   309,   312,    60,   146,
     210,    62,   164,   244,   113,   118,   149,   122,    77,    82,
     150,   181,   128,   129,   306,   111,   100,   106,    85,   138,
     151,   141,   192,   152,   358,   359,   360,   143,   144,   363,
     364,   365,   366,   367,   280,   281,   282,   315,   140,   165,
      47,    46,    49,   166,    48,   355,    94,    51,   167,   256,
     264,   265,   266,   280,   281,   282,   276,   277,   278,   279,
     168,   285,   288,   289,   290,   291,   169,   293,   295,   297,
     300,   302,   170,   211,   171,   235,   236,   237,   238,   260,
     240,   241,   267,   269,   172,   275,   119,   120,   195,   196,
      79,   173,   127,   413,   153,    79,   154,   155,   174,   404,
     191,   403,    79,   175,   157,   154,   155,   336,   384,   196,
     341,   342,   343,   305,   345,    47,    48,    49,   262,    51,
     192,   314,   176,   272,    46,    79,   268,   177,   178,    94,
     286,   179,   180,   382,    55,    56,   383,    57,   182,   183,
     400,   401,   330,   184,   332,   333,   185,    79,   217,   186,
     261,   187,   194,   318,    79,   271,   346,   317,   320,   321,
     322,   323,   324,   325,   326,    43,    44,   199,   263,   200,
     298,   201,   303,   273,   239,   157,   202,   339,   340,   208,
     287,   203,   204,   212,   213,   205,   417,   206,   214,   215,
     207,    46,   209,    47,    45,    49,    94,   216,   310,   313,
     218,   220,   329,   319,   331,   221,   334,   222,   337,   338,
     123,   125,   126,    43,    44,    43,    44,   223,   224,   283,
     225,   226,   227,   228,   292,   335,   294,   296,   299,   301,
      59,   229,   344,   230,   307,    91,   231,    85,    62,   242,
     283,   437,    45,   196,    45,   316,   361,    46,   280,   281,
     282,   327,    94,   328,    47,   362,    49,   405,   406,   407,
     408,   409,   410,    43,    44,   351,   425,   368,   352,    43,
      44,   353,   354,   369,   370,   347,    95,   428,    59,   371,
      59,   435,    96,    97,   397,    91,    62,   357,    62,   372,
     373,   374,    45,   411,   412,   375,   376,   377,    45,   378,
     379,   419,   380,    46,   386,    47,    93,    49,    50,   381,
     192,    53,    54,    55,    56,   385,    57,   398,   387,   388,
     432,   432,   389,   390,   316,   402,   450,   451,    59,   453,
     454,   391,    58,    91,    59,   392,    62,   393,   399,    60,
     394,    85,    62,   395,   396,   420,   421,   429,    52,   414,
     162,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   415,   436,   424,   438,   449,   148,   439,   441,   452,
     250,   251,   252,   253,   254,   422,   423,   427,   442,   443,
     431,   434,   444,   445,     2,     3,   446,   447,   418,   248,
     249,   250,   251,   252,   253,   254,   448,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,   348,    19,    20,   433,    21,    22,    23,    24,
      25,   426,     0,   356,   349,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,     0,     0,     0,     0,     0,
      26,    27,    28,    29,    30,    31,    32,    33,    34,     3,
       0,    35,    36,     0,     0,    37,     0,    38,     0,     0,
      39,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,     0,    19,    20,     0,
      21,    22,    23,    24,    25,     0,     0,     0,     0,    43,
      44,     0,     0,     0,     0,     0,     0,     0,   350,     0,
       0,     0,     0,     0,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    43,    44,    35,    36,     0,    45,    37,
       0,    38,     0,    46,    39,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,     0,    57,    43,    44,     0,
       0,     0,    45,    43,    44,     0,     0,     0,     0,     0,
      48,     0,    58,    51,    59,     0,     0,     0,     0,    60,
       0,    61,    62,     0,    43,    44,    45,     0,     0,     0,
      43,   440,    45,    47,     0,    49,    58,     0,    59,     0,
       0,     0,     0,    60,    52,   102,    62,     0,    43,    44,
       0,     0,     0,    45,    43,    44,     0,     0,     0,    45,
      58,    48,    59,     0,    51,     0,    58,    91,    59,     0,
      62,     0,     0,    60,     0,    85,    62,    45,    43,    44,
      43,    44,     0,    45,    47,     0,    49,    58,     0,    59,
       0,   198,   160,    58,    60,    59,     0,    62,     0,     0,
      91,    43,    44,    62,     0,    43,    44,    45,     0,    45,
       0,     0,     0,    59,     0,     0,   160,    58,    91,    59,
       0,    62,    53,    54,    91,    43,    44,    62,     0,     0,
      45,    43,    44,     0,    45,     0,     0,     0,     0,     0,
       0,    58,     0,    59,     0,    59,     0,     0,    91,     0,
      91,    62,     0,    62,    45,     0,     0,     0,     0,     0,
      45,     0,     0,     0,    95,     0,    59,     0,    95,     0,
      59,   189,     0,     0,    62,   308,     0,     0,    62,   246,
     247,   248,   249,   250,   251,   252,   253,   254,    95,     0,
      59,     0,     0,     0,    95,   311,    59,     0,    62,     0,
       0,   416,     0,     0,    62,   247,   248,   249,   250,   251,
     252,   253,   254
};

static const yytype_int16 yycheck[] =
{
       5,     6,     7,     8,     9,    10,    21,   164,    96,   169,
     187,    95,     9,    18,    19,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    30,    76,    32,    33,    34,
     164,    36,    97,   144,   145,    77,   170,    12,     8,     9,
       4,     5,     8,    44,    44,    46,    46,    11,    12,    13,
      10,    11,    12,    44,    45,    46,    42,    48,    22,    23,
      24,    47,    45,    80,    44,    48,    46,    37,    80,     5,
      34,    78,    69,    80,    44,    78,    46,    13,    78,    43,
      44,    13,    14,    15,    16,    17,    77,    19,    75,    80,
      11,    79,    44,    29,    46,    31,    32,    61,    62,   164,
      80,    71,    76,    73,     8,     9,   194,   195,    78,    76,
     115,    81,    77,    79,    18,    19,    78,    21,     4,     5,
      78,    85,    26,    27,   189,    77,    12,    13,    80,    33,
      78,    35,    96,    78,   245,   246,   247,    74,    75,   250,
     251,   252,   253,   254,    38,    39,    40,    41,    34,    77,
      44,    42,    46,    77,    45,   239,    47,    48,    77,   164,
     165,   166,   167,    38,    39,    40,   171,   172,   173,   174,
      77,   176,   177,   178,   179,   180,    77,   182,   183,   184,
     185,   186,    77,   115,    77,   149,   150,   151,   152,   164,
     154,   155,   167,   168,    77,   170,    19,    20,    77,    78,
     164,    77,    25,   380,     6,   169,     8,     9,    77,   369,
      96,   368,   176,    78,    60,     8,     9,   222,    77,    78,
     225,   226,   227,   187,   229,    44,    45,    46,   164,    48,
     194,   195,    77,   169,    42,   199,   168,    77,    77,    47,
     176,    77,    77,   308,    52,    53,   311,    55,    77,    77,
     361,   362,   216,    77,   218,   219,    77,   221,    78,    77,
     164,    77,    77,   199,   228,   169,   230,   199,   200,   201,
     202,   203,   204,   205,   206,     8,     9,    77,   164,    77,
     184,    77,   186,   169,     7,   131,    77,   223,   224,   112,
     176,    77,    77,   116,   117,    77,   384,    77,   121,   122,
      77,    42,    77,    44,    37,    46,    47,    77,   194,   195,
      77,    77,   216,   199,   218,    77,   220,    77,   222,   223,
      22,    23,    24,     8,     9,     8,     9,    77,    77,   175,
      77,    77,    77,    77,   180,   221,   182,   183,   184,   185,
      73,    77,   228,    77,    79,    78,    77,    80,    81,    78,
     196,   416,    37,    78,    37,    69,     6,    42,    38,    39,
      40,   207,    47,   209,    44,     7,    46,   372,   373,   374,
     375,   376,   377,     8,     9,    79,   391,    77,    79,     8,
       9,    79,    79,    77,    79,   231,    71,   392,    73,    79,
      73,   396,    77,    78,    78,    78,    81,   243,    81,    77,
      77,    77,    37,   378,   379,    77,    77,    77,    37,    77,
      77,   386,    77,    42,    77,    44,   380,    46,    47,    79,
     384,    50,    51,    52,    53,    79,    55,    79,    77,    77,
     394,   395,    77,    77,    69,    47,   441,   442,    73,   444,
     445,    77,    71,    78,    73,    77,    81,    77,    79,    78,
      77,    80,    81,    77,    77,   387,   388,   393,    49,    79,
     424,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    79,    39,    80,    77,   439,   440,     9,    77,   443,
       8,     9,    10,    11,    12,   389,   390,   391,    77,    77,
     394,   395,    77,    77,     0,     1,    79,    79,   384,     6,
       7,     8,     9,    10,    11,    12,   438,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,   232,    29,    30,   395,    32,    33,    34,    35,
      36,   391,    -1,   242,    76,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    -1,    -1,    -1,    -1,    -1,
      56,    57,    58,    59,    60,    61,    62,    63,    64,     1,
      -1,    67,    68,    -1,    -1,    71,    -1,    73,    -1,    -1,
      76,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    -1,    29,    30,    -1,
      32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,     8,
       9,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,
      -1,    -1,    -1,    -1,    56,    57,    58,    59,    60,    61,
      62,    63,    64,     8,     9,    67,    68,    -1,    37,    71,
      -1,    73,    -1,    42,    76,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    -1,    55,     8,     9,    -1,
      -1,    -1,    37,     8,     9,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    71,    48,    73,    -1,    -1,    -1,    -1,    78,
      -1,    80,    81,    -1,     8,     9,    37,    -1,    -1,    -1,
       8,     9,    37,    44,    -1,    46,    71,    -1,    73,    -1,
      -1,    -1,    -1,    78,    49,    80,    81,    -1,     8,     9,
      -1,    -1,    -1,    37,     8,     9,    -1,    -1,    -1,    37,
      71,    45,    73,    -1,    48,    -1,    71,    78,    73,    -1,
      81,    -1,    -1,    78,    -1,    80,    81,    37,     8,     9,
       8,     9,    -1,    37,    44,    -1,    46,    71,    -1,    73,
      -1,    69,    70,    71,    78,    73,    -1,    81,    -1,    -1,
      78,     8,     9,    81,    -1,     8,     9,    37,    -1,    37,
      -1,    -1,    -1,    73,    -1,    -1,    70,    71,    78,    73,
      -1,    81,    50,    51,    78,     8,     9,    81,    -1,    -1,
      37,     8,     9,    -1,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    71,    -1,    73,    -1,    73,    -1,    -1,    78,    -1,
      78,    81,    -1,    81,    37,    -1,    -1,    -1,    -1,    -1,
      37,    -1,    -1,    -1,    71,    -1,    73,    -1,    71,    -1,
      73,    78,    -1,    -1,    81,    78,    -1,    -1,    81,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    71,    -1,
      73,    -1,    -1,    -1,    71,    78,    73,    -1,    81,    -1,
      -1,    78,    -1,    -1,    81,     5,     6,     7,     8,     9,
      10,    11,    12
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
      95,    96,    97,   100,   102,   103,   104,   105,   106,   110,
      88,   104,   105,    88,    88,    80,    88,   102,    88,   102,
      88,    78,    98,   110,    47,    71,    77,    78,    87,    97,
     105,   110,    80,    96,   101,   104,   105,    96,    96,    96,
      96,    77,    88,   102,   107,    77,    88,    96,   102,   107,
     107,   100,   102,   106,   110,   106,   106,   107,   102,   102,
      76,    78,   104,    88,   104,    88,   104,    88,   102,    88,
     105,   102,    88,    74,    75,    75,    76,   110,   110,    78,
      78,    78,    78,     6,     8,     9,   108,   103,   110,   111,
      70,   105,   110,   110,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    78,    77,    77,    77,    77,
      77,   110,    77,    77,    77,    77,    77,    77,   108,    78,
      87,   105,   110,    89,    77,    77,    78,     9,    69,    77,
      77,    77,    77,    77,    77,    77,    77,    77,   107,    77,
      88,    96,   107,   107,   107,   107,    77,    78,    77,    11,
      77,    77,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    85,   111,   111,   110,   110,   110,   110,     7,
     110,   110,    78,     8,    79,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    79,    88,    89,    91,    94,
      97,   102,   104,   105,    88,    88,    88,    97,    96,    97,
      95,   102,   104,   105,    94,    97,    88,    88,    88,    88,
      38,    39,    40,   103,   109,    88,   104,   105,    88,    88,
      88,    88,   103,    88,   103,    88,   103,    88,   102,   103,
      88,   103,    88,   102,    98,   110,    89,    79,    78,    87,
     105,    78,    87,   105,   110,    41,    69,    96,   104,   105,
      96,    96,    96,    96,    96,    96,    96,   103,   103,   102,
     110,   102,   110,   110,   102,   105,    88,   102,   102,   104,
     104,    88,    88,    88,   105,    88,   110,   103,    84,    76,
      76,    79,    79,    79,    79,   108,   109,   103,   111,   111,
     111,     6,     7,   111,   111,   111,   111,   111,    77,    77,
      79,    79,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    79,    89,    89,    77,    79,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    78,    79,    79,
     111,   111,    47,    91,    95,    88,    88,    88,    88,    88,
      88,    97,    97,    98,    79,    79,    78,    87,   105,    97,
      96,    96,   102,   102,    80,   100,   101,   102,    88,   104,
      99,   102,   110,    99,   102,    88,    39,    89,    77,     9,
       9,    77,    77,    77,    77,    77,    79,    79,    96,   110,
      88,    88,   110,    88,    88
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
		settext((yyvsp[(2) - (4)].addr).sym);
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 118:
#line 622 "a.y"
    {
		settext((yyvsp[(2) - (6)].addr).sym);
		(yyvsp[(6) - (6)].addr).offset &= 0xffffffffull;
		(yyvsp[(6) - (6)].addr).offset |= (vlong)ArgsSizeUnknown << 32;
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 119:
#line 629 "a.y"
    {
		settext((yyvsp[(2) - (8)].addr).sym);
		(yyvsp[(6) - (8)].addr).offset &= 0xffffffffull;
		(yyvsp[(6) - (8)].addr).offset |= ((yyvsp[(8) - (8)].lval) & 0xffffffffull) << 32;
		outcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].lval), &(yyvsp[(6) - (8)].addr));
	}
    break;

  case 120:
#line 639 "a.y"
    {
		settext((yyvsp[(2) - (4)].addr).sym);
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 121:
#line 644 "a.y"
    {
		settext((yyvsp[(2) - (6)].addr).sym);
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 122:
#line 653 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 123:
#line 657 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 124:
#line 661 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 125:
#line 668 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &nullgen);
	}
    break;

  case 126:
#line 674 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 127:
#line 680 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		(yyval.addr) = nullgen;
		if(pass == 2 && (yyvsp[(1) - (2)].sym)->type != LLAB)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 128:
#line 691 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 131:
#line 703 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 132:
#line 711 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);	/* whole register */
	}
    break;

  case 133:
#line 719 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 134:
#line 727 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 135:
#line 735 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 136:
#line 741 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= 1024)
			yyerror("SPR/DCR out of range");
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (4)].lval) + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 138:
#line 752 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 139:
#line 760 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 140:
#line 766 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_F0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 141:
#line 774 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 142:
#line 780 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_C0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 143:
#line 788 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 144:
#line 796 "a.y"
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

  case 145:
#line 817 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = TYPE_CONST;
	}
    break;

  case 146:
#line 822 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 147:
#line 830 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 148:
#line 836 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 149:
#line 843 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 151:
#line 852 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = REG_R0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 152:
#line 860 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 153:
#line 867 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 155:
#line 878 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 156:
#line 887 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(3) - (4)].lval);
		(yyval.addr).sym = nil;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 157:
#line 895 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 158:
#line 903 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = NAME_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 161:
#line 915 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 162:
#line 919 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 163:
#line 923 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 168:
#line 935 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 169:
#line 939 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 170:
#line 943 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 171:
#line 947 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 172:
#line 951 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 174:
#line 958 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 175:
#line 962 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 176:
#line 966 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 177:
#line 970 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 178:
#line 974 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 179:
#line 978 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 180:
#line 982 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 181:
#line 986 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 182:
#line 990 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 183:
#line 994 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 3188 "y.tab.c"
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



