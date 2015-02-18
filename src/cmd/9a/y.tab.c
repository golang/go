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
#define YYLAST   880

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  82
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  32
/* YYNRULES -- Number of rules.  */
#define YYNRULES  187
/* YYNRULES -- Number of states.  */
#define YYNSTATES  463

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
       0,     0,     3,     4,     5,     9,    10,    15,    20,    25,
      28,    30,    33,    36,    41,    46,    51,    56,    61,    66,
      71,    76,    81,    86,    91,    96,   101,   106,   111,   116,
     121,   126,   131,   136,   143,   148,   153,   160,   165,   170,
     177,   184,   191,   196,   201,   208,   213,   220,   225,   232,
     237,   242,   245,   252,   257,   262,   267,   274,   279,   284,
     289,   294,   299,   304,   309,   314,   317,   320,   325,   329,
     333,   339,   344,   349,   356,   361,   366,   373,   380,   387,
     396,   401,   406,   410,   413,   418,   423,   430,   439,   444,
     451,   456,   461,   468,   475,   484,   493,   502,   511,   516,
     521,   526,   533,   538,   545,   550,   555,   558,   561,   565,
     569,   573,   577,   580,   584,   588,   593,   598,   601,   607,
     615,   620,   627,   634,   641,   648,   651,   656,   659,   661,
     663,   665,   667,   669,   671,   673,   675,   680,   682,   684,
     686,   691,   693,   698,   700,   704,   706,   709,   713,   718,
     721,   724,   727,   731,   734,   736,   741,   745,   751,   753,
     758,   763,   769,   777,   778,   780,   781,   784,   787,   789,
     791,   793,   795,   797,   800,   803,   806,   810,   812,   816,
     820,   824,   828,   832,   837,   842,   846,   850
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      83,     0,    -1,    -1,    -1,    83,    84,    85,    -1,    -1,
      71,    74,    86,    85,    -1,    71,    75,   113,    76,    -1,
      73,    75,   113,    76,    -1,    58,    76,    -1,    76,    -1,
      87,    76,    -1,     1,    76,    -1,    13,    89,    77,    89,
      -1,    13,   107,    77,    89,    -1,    13,   106,    77,    89,
      -1,    14,    89,    77,    89,    -1,    14,   107,    77,    89,
      -1,    14,   106,    77,    89,    -1,    22,   107,    77,    97,
      -1,    22,   106,    77,    97,    -1,    22,   103,    77,    97,
      -1,    22,    97,    77,    97,    -1,    22,    97,    77,   107,
      -1,    22,    97,    77,   106,    -1,    13,    89,    77,   107,
      -1,    13,    89,    77,   106,    -1,    14,    89,    77,   107,
      -1,    14,    89,    77,   106,    -1,    13,    97,    77,   107,
      -1,    13,    97,    77,   106,    -1,    13,    96,    77,    97,
      -1,    13,    97,    77,    96,    -1,    13,    97,    77,   104,
      77,    96,    -1,    13,    96,    77,    98,    -1,    67,   104,
      77,   112,    -1,    13,    89,    77,   104,    77,    92,    -1,
      13,    89,    77,    98,    -1,    13,    89,    77,    92,    -1,
      18,    89,    77,   105,    77,    89,    -1,    18,   104,    77,
     105,    77,    89,    -1,    18,    89,    77,   104,    77,    89,
      -1,    18,    89,    77,    89,    -1,    18,   104,    77,    89,
      -1,    16,    89,    77,   105,    77,    89,    -1,    16,    89,
      77,    89,    -1,    17,    89,    77,   105,    77,    89,    -1,
      17,    89,    77,    89,    -1,    17,   104,    77,   105,    77,
      89,    -1,    17,   104,    77,    89,    -1,    15,    89,    77,
      89,    -1,    15,    89,    -1,    68,    89,    77,   105,    77,
      89,    -1,    13,   104,    77,    89,    -1,    13,   102,    77,
      89,    -1,    20,    99,    77,    99,    -1,    20,    99,    77,
     112,    77,    99,    -1,    13,    98,    77,    98,    -1,    13,
      95,    77,    98,    -1,    13,    92,    77,    89,    -1,    13,
      95,    77,    89,    -1,    13,    90,    77,    89,    -1,    13,
      89,    77,    90,    -1,    13,    98,    77,    95,    -1,    13,
      89,    77,    95,    -1,    21,    88,    -1,    21,   107,    -1,
      21,    78,    90,    79,    -1,    21,    77,    88,    -1,    21,
      77,   107,    -1,    21,    77,    78,    90,    79,    -1,    21,
      98,    77,    88,    -1,    21,    98,    77,   107,    -1,    21,
      98,    77,    78,    90,    79,    -1,    21,   112,    77,    88,
      -1,    21,   112,    77,   107,    -1,    21,   112,    77,    78,
      90,    79,    -1,    21,   112,    77,   112,    77,    88,    -1,
      21,   112,    77,   112,    77,   107,    -1,    21,   112,    77,
     112,    77,    78,    90,    79,    -1,    27,    89,    77,   105,
      -1,    27,   104,    77,   105,    -1,    27,    89,   109,    -1,
      27,   109,    -1,    23,    97,    77,    97,    -1,    25,    97,
      77,    97,    -1,    25,    97,    77,    97,    77,    97,    -1,
      26,    97,    77,    97,    77,    97,    77,    97,    -1,    24,
      97,    77,    97,    -1,    24,    97,    77,    97,    77,    98,
      -1,    19,    89,    77,    89,    -1,    19,    89,    77,   104,
      -1,    19,    89,    77,    89,    77,    98,    -1,    19,    89,
      77,   104,    77,    98,    -1,    63,   104,    77,    89,    77,
     104,    77,    89,    -1,    63,   104,    77,    89,    77,   100,
      77,    89,    -1,    63,    89,    77,    89,    77,   104,    77,
      89,    -1,    63,    89,    77,    89,    77,   100,    77,    89,
      -1,    64,   107,    77,    89,    -1,    64,    89,    77,   107,
      -1,    59,   106,    77,    89,    -1,    59,   106,    77,   104,
      77,    89,    -1,    60,    89,    77,   106,    -1,    60,    89,
      77,   104,    77,   106,    -1,    62,   106,    77,    89,    -1,
      62,    89,    77,   106,    -1,    61,   106,    -1,    29,   109,
      -1,    29,    89,   109,    -1,    29,    97,   109,    -1,    29,
      77,    89,    -1,    29,    77,    97,    -1,    29,   104,    -1,
      32,   104,   109,    -1,    32,   102,   109,    -1,    56,   104,
      77,   104,    -1,    57,   104,    77,   107,    -1,    30,   109,
      -1,    33,   108,    77,    80,   101,    -1,    33,   108,    77,
     112,    77,    80,   101,    -1,    34,   108,    77,   104,    -1,
      34,   108,    77,   112,    77,   104,    -1,    35,   108,    11,
     112,    77,   104,    -1,    35,   108,    11,   112,    77,   102,
      -1,    35,   108,    11,   112,    77,   103,    -1,    36,   109,
      -1,   112,    78,    41,    79,    -1,    71,   110,    -1,   105,
      -1,    91,    -1,    93,    -1,    50,    -1,    47,    -1,    51,
      -1,    55,    -1,    53,    -1,    52,    78,   112,    79,    -1,
      94,    -1,    49,    -1,    45,    -1,    48,    78,   112,    79,
      -1,    42,    -1,    47,    78,   112,    79,    -1,   112,    -1,
     112,    77,   112,    -1,    37,    -1,     9,    37,    -1,    37,
       9,    37,    -1,     9,    37,     9,    37,    -1,    80,   107,
      -1,    80,    70,    -1,    80,    69,    -1,    80,     9,    69,
      -1,    80,   112,    -1,    44,    -1,    46,    78,   112,    79,
      -1,    78,   105,    79,    -1,    78,   105,     8,   105,    79,
      -1,   108,    -1,   112,    78,   105,    79,    -1,   112,    78,
     111,    79,    -1,    71,   110,    78,   111,    79,    -1,    71,
       6,     7,   110,    78,    39,    79,    -1,    -1,    77,    -1,
      -1,     8,   112,    -1,     9,   112,    -1,    39,    -1,    38,
      -1,    40,    -1,    37,    -1,    73,    -1,     9,   112,    -1,
       8,   112,    -1,    81,   112,    -1,    78,   113,    79,    -1,
     112,    -1,   113,     8,   113,    -1,   113,     9,   113,    -1,
     113,    10,   113,    -1,   113,    11,   113,    -1,   113,    12,
     113,    -1,   113,     6,     6,   113,    -1,   113,     7,     7,
     113,    -1,   113,     5,   113,    -1,   113,     4,   113,    -1,
     113,     3,   113,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    66,    66,    68,    67,    75,    74,    83,    88,    94,
      98,    99,   100,   106,   110,   114,   118,   122,   126,   133,
     137,   141,   145,   149,   153,   160,   164,   168,   172,   179,
     183,   190,   194,   198,   202,   206,   213,   217,   221,   231,
     235,   239,   243,   247,   251,   255,   259,   263,   267,   271,
     275,   279,   286,   293,   297,   304,   308,   316,   320,   324,
     328,   332,   336,   340,   344,   353,   357,   361,   365,   369,
     373,   377,   381,   385,   389,   393,   397,   401,   409,   417,
     428,   432,   436,   440,   447,   451,   455,   459,   463,   467,
     474,   478,   482,   486,   493,   497,   501,   505,   512,   516,
     524,   528,   532,   536,   540,   544,   548,   555,   559,   563,
     567,   571,   575,   582,   586,   593,   602,   613,   620,   625,
     637,   642,   655,   663,   671,   682,   688,   694,   705,   713,
     714,   717,   725,   733,   741,   749,   755,   763,   766,   774,
     780,   788,   794,   802,   810,   831,   838,   845,   852,   861,
     866,   874,   880,   887,   895,   896,   904,   911,   921,   922,
     931,   939,   947,   956,   957,   960,   963,   967,   973,   974,
     975,   978,   979,   983,   987,   991,   995,  1001,  1002,  1006,
    1010,  1014,  1018,  1022,  1026,  1030,  1034,  1038
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
  "'='", "';'", "','", "'('", "')'", "'$'", "'~'", "$accept", "prog", "@1",
  "line", "@2", "inst", "rel", "rreg", "xlreg", "lr", "lcr", "ctr", "msr",
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
       0,    82,    83,    84,    83,    86,    85,    85,    85,    85,
      85,    85,    85,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    88,    88,    89,    90,
      90,    91,    92,    93,    94,    95,    95,    95,    96,    97,
      97,    98,    98,    99,   100,   101,   101,   101,   101,   102,
     102,   103,   103,   104,   105,   105,   106,   106,   107,   107,
     108,   108,   108,   109,   109,   110,   110,   110,   111,   111,
     111,   112,   112,   112,   112,   112,   112,   113,   113,   113,
     113,   113,   113,   113,   113,   113,   113,   113
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     4,     4,     2,
       1,     2,     2,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     6,     4,     4,     6,     4,     4,     6,
       6,     6,     4,     4,     6,     4,     6,     4,     6,     4,
       4,     2,     6,     4,     4,     4,     6,     4,     4,     4,
       4,     4,     4,     4,     4,     2,     2,     4,     3,     3,
       5,     4,     4,     6,     4,     4,     6,     6,     6,     8,
       4,     4,     3,     2,     4,     4,     6,     8,     4,     6,
       4,     4,     6,     6,     8,     8,     8,     8,     4,     4,
       4,     6,     4,     6,     4,     4,     2,     2,     3,     3,
       3,     3,     2,     3,     3,     4,     4,     2,     5,     7,
       4,     6,     6,     6,     6,     2,     4,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     4,     1,     1,     1,
       4,     1,     4,     1,     3,     1,     2,     3,     4,     2,
       2,     2,     3,     2,     1,     4,     3,     5,     1,     4,
       4,     5,     7,     0,     1,     0,     2,     2,     1,     1,
       1,     1,     1,     2,     2,     2,     3,     1,     3,     3,
       3,     3,     3,     4,     4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   163,
     163,   163,     0,     0,     0,     0,   163,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      10,     4,     0,    12,     0,     0,   171,   141,   154,   139,
       0,   132,     0,   138,   131,   133,     0,   135,   134,   165,
     172,     0,     0,     0,     0,     0,   129,     0,   130,   137,
       0,     0,     0,     0,     0,     0,   128,     0,     0,   158,
       0,     0,     0,     0,    51,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   143,     0,   165,     0,     0,    65,
       0,    66,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   164,   163,     0,    83,   164,   163,   163,   112,
     107,   117,   163,   163,     0,     0,     0,     0,   125,     0,
       0,     9,     0,     0,     0,   106,     0,     0,     0,     0,
       0,     0,     0,     0,     5,     0,     0,    11,   174,   173,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   177,
       0,   150,   149,   153,   175,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   153,     0,     0,     0,     0,     0,     0,   127,
       0,    68,    69,     0,     0,     0,     0,     0,     0,   151,
       0,     0,     0,     0,     0,     0,     0,     0,   164,    82,
       0,   110,   111,   108,   109,   114,   113,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     165,   166,   167,     0,     0,   156,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   176,    13,    62,    38,
      64,    37,     0,    26,    25,    61,    59,    60,    58,    31,
      34,    32,     0,    30,    29,    63,    57,    54,    53,    15,
      14,   169,   168,   170,     0,     0,    16,    28,    27,    18,
      17,    50,    45,   128,    47,   128,    49,   128,    42,     0,
     128,    43,   128,    90,    91,    55,   143,     0,    67,     0,
      71,    72,     0,    74,    75,     0,     0,   152,    22,    24,
      23,    21,    20,    19,    84,    88,    85,     0,    80,    81,
       0,     0,   120,     0,     0,   115,   116,   100,     0,     0,
     102,   105,   104,     0,     0,    99,    98,    35,     0,     6,
       7,     8,   155,   142,   140,   136,     0,     0,     0,   187,
     186,   185,     0,     0,   178,   179,   180,   181,   182,     0,
       0,   159,   160,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    70,     0,     0,     0,   126,     0,     0,     0,
       0,   145,   118,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   161,   157,   183,   184,   132,    36,    33,    44,
      46,    48,    41,    39,    40,    92,    93,    56,    73,    76,
       0,    77,    78,    89,    86,     0,   146,     0,     0,   121,
       0,   123,   124,   122,   101,   103,     0,     0,     0,     0,
       0,    52,     0,     0,     0,     0,   147,   119,     0,     0,
       0,     0,     0,     0,   162,    79,    87,   148,    97,    96,
     144,    95,    94
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    41,   233,    42,    99,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    93,   436,   392,
      74,   105,    75,    76,    77,   162,    79,   115,   157,   285,
     159,   160
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -180
static const yytype_int16 yypact[] =
{
    -180,    12,  -180,   484,   -53,   517,   619,    28,    28,   -24,
     -24,    28,   799,   577,   596,   -29,   -29,   -29,   -29,    19,
     -11,   -51,   -38,   701,   701,   701,   -51,   -19,   -19,    -8,
      -7,    28,    -7,   -14,   -24,   643,   -19,    28,   -36,     6,
    -180,  -180,    26,  -180,   799,   799,  -180,  -180,  -180,  -180,
       7,    27,    51,  -180,  -180,  -180,    61,  -180,  -180,   188,
    -180,   662,   674,   799,    79,    93,  -180,    99,  -180,  -180,
     112,   116,   126,   136,   138,   157,  -180,   168,   176,  -180,
      80,   179,   182,   184,   186,   194,   799,   196,   198,   200,
     201,   202,   799,   203,  -180,    27,   188,   714,   676,  -180,
     206,  -180,    49,     8,   215,   216,   217,   219,   220,   221,
     223,   224,  -180,   225,   226,  -180,   181,   -51,   -51,  -180,
    -180,  -180,   -51,   -51,   227,   158,   231,   296,  -180,   232,
     233,  -180,    28,   234,   236,  -180,   237,   238,   242,   245,
     246,   248,   249,   250,  -180,   799,   799,  -180,  -180,  -180,
     799,   799,   799,   799,   321,   799,   799,   251,     3,  -180,
     377,  -180,  -180,    80,  -180,   565,    28,    28,   172,   162,
     623,    31,    28,    28,    28,    28,   230,   619,    28,    28,
      28,    28,  -180,    28,    28,   -24,    28,   -24,   799,   251,
     676,  -180,  -180,   254,   257,   723,   753,   111,   268,  -180,
      67,   -29,   -29,   -29,   -29,   -29,   -29,   -29,    28,  -180,
      28,  -180,  -180,  -180,  -180,  -180,  -180,   733,    96,   760,
     799,   -19,   701,   -24,    40,    -7,    28,    28,    28,   701,
      28,   799,    28,   484,   463,   524,   265,   266,   267,   270,
     135,  -180,  -180,    96,    28,  -180,   799,   799,   799,   341,
     347,   799,   799,   799,   799,   799,  -180,  -180,  -180,  -180,
    -180,  -180,   278,  -180,  -180,  -180,  -180,  -180,  -180,  -180,
    -180,  -180,   279,  -180,  -180,  -180,  -180,  -180,  -180,  -180,
    -180,  -180,  -180,  -180,   281,   282,  -180,  -180,  -180,  -180,
    -180,  -180,  -180,   280,  -180,   285,  -180,   286,  -180,   288,
     289,  -180,   297,   299,   301,  -180,   319,   294,  -180,   676,
    -180,  -180,   676,  -180,  -180,   171,   318,  -180,  -180,  -180,
    -180,  -180,  -180,  -180,  -180,   324,   325,   327,  -180,  -180,
       9,   328,  -180,   329,   330,  -180,  -180,  -180,   331,   335,
    -180,  -180,  -180,   336,   337,  -180,  -180,  -180,   338,  -180,
    -180,  -180,  -180,  -180,  -180,  -180,   320,   339,   340,   571,
     430,    82,   799,   799,   153,   153,  -180,  -180,  -180,   374,
     373,  -180,  -180,    28,    28,    28,    28,    28,    28,    20,
      20,   799,  -180,   344,   345,   772,  -180,    20,   -29,   -29,
     390,   419,  -180,   349,   -19,   350,    28,    -7,   760,   760,
      28,   392,  -180,  -180,   277,   277,  -180,  -180,  -180,  -180,
    -180,  -180,  -180,  -180,  -180,  -180,  -180,  -180,  -180,  -180,
     676,  -180,  -180,  -180,  -180,   355,   436,   411,     9,  -180,
     322,  -180,  -180,  -180,  -180,  -180,   375,   376,   378,   380,
     381,  -180,   372,   382,   -29,   417,  -180,  -180,   790,    28,
      28,   799,    28,    28,  -180,  -180,  -180,  -180,  -180,  -180,
    -180,  -180,  -180
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -180,  -180,  -180,   229,  -180,  -180,   -73,    -6,   -62,  -180,
    -158,  -180,  -180,  -150,  -162,    37,    30,  -179,    60,    32,
     -16,    68,    97,   167,    81,    95,   159,    24,   -86,   239,
      35,    87
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      81,    84,    85,    87,    89,    91,   122,   259,   271,   305,
     189,   244,     2,   113,   117,   260,    49,   198,   390,    52,
      48,   275,    50,    43,   191,   134,   112,   136,   138,   140,
      48,   143,    50,    48,    49,    50,   194,    52,   144,   145,
      80,    80,    62,   100,   120,   121,   391,    94,   102,    80,
     128,   104,   108,   109,   110,   111,    86,   118,   125,   125,
     125,    86,    47,    48,   132,    50,   116,    95,   131,    86,
      80,   132,    48,    47,    50,    44,    45,   199,    95,   148,
     149,   146,   245,    56,    57,   150,    58,    82,   249,   250,
     251,   252,   253,   254,   255,   106,   112,   163,   164,    86,
      78,    83,   147,   258,    46,   151,    88,    90,   101,   107,
     211,   133,    49,   135,   137,    52,   114,   119,   132,   123,
      86,   182,   310,   313,   129,   130,   196,   197,   307,   152,
     141,   139,   193,   142,   281,   282,   283,   209,    59,   153,
      60,   213,   214,   155,   156,    61,   215,   216,    63,   281,
     282,   283,   316,   212,   356,    48,   165,    50,   176,   257,
     265,   266,   267,   253,   254,   255,   277,   278,   279,   280,
     166,   286,   289,   290,   291,   292,   167,   294,   296,   298,
     301,   303,   124,   126,   127,   236,   237,   238,   239,   168,
     241,   242,   192,   169,   154,   261,   155,   156,   268,   270,
      80,   276,   417,   170,    47,    80,   269,    49,   408,    95,
      52,   407,    80,   171,    47,   172,    48,   337,    50,    95,
     342,   343,   344,   306,   346,    48,    49,    50,   158,    52,
     193,   315,   234,   235,   173,    80,   218,   318,   321,   322,
     323,   324,   325,   326,   327,   174,   263,   383,   385,   197,
     384,   273,   331,   175,   333,   334,   177,    80,   287,   178,
     264,   179,   262,   180,    80,   274,   347,   272,   281,   282,
     283,   181,   288,   183,    48,   184,    50,   185,   186,   187,
     188,   319,   299,   195,   304,   251,   252,   253,   254,   255,
     311,   314,   200,   201,   202,   320,   203,   204,   205,   158,
     206,   207,   208,   210,   217,   340,   341,   220,   219,   221,
     222,   223,   421,   224,   225,   226,   332,   336,   335,   227,
     338,   339,   228,   229,   345,   230,   231,   232,   240,   243,
      44,   448,   197,   359,   360,   361,   308,   317,   364,   365,
     366,   367,   368,   284,   352,   353,   354,   362,   293,   355,
     295,   297,   300,   302,   363,   369,   370,   373,   443,    46,
     371,   372,   374,   375,   284,   376,   377,   409,   410,   411,
     412,   413,   414,   382,   378,   328,   379,   329,   380,   431,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     434,   199,   161,    59,   441,    60,   381,   386,   401,   348,
      92,   387,   388,    63,   389,   393,   394,   395,   396,   415,
     416,   358,   397,   398,   399,   400,    94,   423,   402,   403,
     193,   406,    53,   418,   419,   424,   425,   426,   427,   428,
     430,   442,   444,   438,   438,   248,   249,   250,   251,   252,
     253,   254,   255,   458,   459,   445,   461,   462,   446,   404,
     405,   454,   449,   450,   457,   451,   256,   452,   453,   439,
     447,   455,   349,   432,     0,   163,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,     0,     0,   435,     0,
     422,   456,   357,   149,     0,     4,   460,     0,     0,     0,
       0,   429,   433,     0,     0,   437,   440,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,     0,    20,    21,     0,    22,    23,    24,    25,
      26,     0,     0,     0,     0,    44,    45,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,     0,     0,   350,
      27,    28,    29,    30,    31,    32,    33,    34,    35,     0,
       0,    36,    37,     0,    46,    38,     0,    39,     0,    47,
      40,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,     0,    58,    44,    45,   247,   248,   249,   250,   251,
     252,   253,   254,   255,     0,    44,    45,     0,    59,     0,
      60,     0,     0,     0,     0,    61,     0,    62,    63,     0,
     351,     0,    46,     0,    44,    45,     0,    47,     0,    48,
       0,    50,    51,     0,    46,    54,    55,    56,    57,    47,
      58,     0,     0,     0,    95,     0,     0,    44,    45,     0,
       0,    44,    45,    46,     0,     0,    59,     0,    60,     0,
       0,    49,     0,    61,    52,    86,    63,     0,    96,     0,
      60,    44,    45,     0,    97,    98,    46,     0,    63,     0,
      46,     0,     0,    48,     0,    50,     0,    59,     0,    60,
      44,    45,    53,     0,    61,     0,   103,    63,     0,     0,
      46,     0,    44,    45,    44,    45,     0,    48,     0,    50,
      59,     0,    60,     0,    59,     0,    60,    61,     0,    46,
      63,    61,     0,    86,    63,     0,    48,     0,    50,    44,
      45,    46,     0,    46,    59,     0,    60,     0,     0,     0,
       0,    92,    44,    45,    63,     0,    54,    55,     0,     0,
       0,    44,    45,     0,     0,    60,     0,     0,    46,     0,
      92,    44,    45,    63,   161,    59,     0,    60,     0,    60,
       0,    46,    92,     0,    92,    63,     0,    63,     0,     0,
      46,    44,    45,     0,     0,     0,     0,     0,    44,    45,
      46,     0,    59,     0,    60,     0,     0,     0,     0,    92,
      44,    45,    63,     0,     0,    96,     0,    60,     0,     0,
      46,     0,   190,     0,    96,    63,    60,    46,    44,    45,
       0,   309,     0,     0,    63,     0,    60,    44,    45,    46,
       0,    92,     0,   330,    63,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    96,     0,    60,    46,     0,     0,
       0,   312,     0,    60,    63,     0,    46,     0,    92,     0,
      86,    63,     0,    96,     0,    60,     0,     0,     0,     0,
     420,     0,     0,    63,     0,     0,     0,     0,     0,   317,
       0,     0,     0,    60,     0,     0,     0,     0,    92,     0,
       0,    63,    60,     0,     0,     0,     0,    92,     0,     0,
      63
};

static const yytype_int16 yycheck[] =
{
       6,     7,     8,     9,    10,    11,    22,   165,   170,   188,
      96,     8,     0,    19,    20,   165,    45,     9,     9,    48,
      44,   171,    46,    76,    97,    31,    77,    33,    34,    35,
      44,    37,    46,    44,    45,    46,    98,    48,    74,    75,
       5,     6,    80,    13,    20,    21,    37,    12,    13,    14,
      26,    14,    15,    16,    17,    18,    80,    20,    23,    24,
      25,    80,    42,    44,    78,    46,    77,    47,    76,    80,
      35,    78,    44,    42,    46,     8,     9,    69,    47,    44,
      45,    75,    79,    52,    53,    78,    55,     6,     6,     7,
       8,     9,    10,    11,    12,    14,    77,    62,    63,    80,
       5,     6,    76,   165,    37,    78,     9,    10,    13,    14,
     116,    30,    45,    32,    33,    48,    19,    20,    78,    22,
      80,    86,   195,   196,    27,    28,    77,    78,   190,    78,
      35,    34,    97,    36,    38,    39,    40,   113,    71,    78,
      73,   117,   118,     8,     9,    78,   122,   123,    81,    38,
      39,    40,    41,   116,   240,    44,    77,    46,    78,   165,
     166,   167,   168,    10,    11,    12,   172,   173,   174,   175,
      77,   177,   178,   179,   180,   181,    77,   183,   184,   185,
     186,   187,    23,    24,    25,   150,   151,   152,   153,    77,
     155,   156,    97,    77,     6,   165,     8,     9,   168,   169,
     165,   171,   381,    77,    42,   170,   169,    45,   370,    47,
      48,   369,   177,    77,    42,    77,    44,   223,    46,    47,
     226,   227,   228,   188,   230,    44,    45,    46,    61,    48,
     195,   196,   145,   146,    77,   200,    78,   200,   201,   202,
     203,   204,   205,   206,   207,    77,   165,   309,    77,    78,
     312,   170,   217,    77,   219,   220,    77,   222,   177,    77,
     165,    77,   165,    77,   229,   170,   231,   170,    38,    39,
      40,    77,   177,    77,    44,    77,    46,    77,    77,    77,
      77,   200,   185,    77,   187,     8,     9,    10,    11,    12,
     195,   196,    77,    77,    77,   200,    77,    77,    77,   132,
      77,    77,    77,    77,    77,   224,   225,    11,    77,    77,
      77,    77,   385,    77,    77,    77,   219,   222,   221,    77,
     223,   224,    77,    77,   229,    77,    77,    77,     7,    78,
       8,     9,    78,   246,   247,   248,    79,    69,   251,   252,
     253,   254,   255,   176,    79,    79,    79,     6,   181,    79,
     183,   184,   185,   186,     7,    77,    77,    77,   420,    37,
      79,    79,    77,    77,   197,    77,    77,   373,   374,   375,
     376,   377,   378,    79,    77,   208,    77,   210,    77,   395,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
     396,    69,    70,    71,   400,    73,    77,    79,    78,   232,
      78,    77,    77,    81,    77,    77,    77,    77,    77,   379,
     380,   244,    77,    77,    77,    77,   381,   387,    79,    79,
     385,    47,    49,    79,    79,   388,   389,    37,     9,    80,
      80,    39,    77,   398,   399,     5,     6,     7,     8,     9,
      10,    11,    12,   449,   450,     9,   452,   453,    37,   362,
     363,    79,    77,    77,    37,    77,    79,    77,    77,   399,
     428,    79,   233,   395,    -1,   430,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    -1,    -1,   397,    -1,
     385,   444,   243,   448,    -1,     1,   451,    -1,    -1,    -1,
      -1,   394,   395,    -1,    -1,   398,   399,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    -1,    29,    30,    -1,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,     8,     9,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    -1,    -1,    76,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    -1,
      -1,    67,    68,    -1,    37,    71,    -1,    73,    -1,    42,
      76,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    -1,    55,     8,     9,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    -1,     8,     9,    -1,    71,    -1,
      73,    -1,    -1,    -1,    -1,    78,    -1,    80,    81,    -1,
      76,    -1,    37,    -1,     8,     9,    -1,    42,    -1,    44,
      -1,    46,    47,    -1,    37,    50,    51,    52,    53,    42,
      55,    -1,    -1,    -1,    47,    -1,    -1,     8,     9,    -1,
      -1,     8,     9,    37,    -1,    -1,    71,    -1,    73,    -1,
      -1,    45,    -1,    78,    48,    80,    81,    -1,    71,    -1,
      73,     8,     9,    -1,    77,    78,    37,    -1,    81,    -1,
      37,    -1,    -1,    44,    -1,    46,    -1,    71,    -1,    73,
       8,     9,    49,    -1,    78,    -1,    80,    81,    -1,    -1,
      37,    -1,     8,     9,     8,     9,    -1,    44,    -1,    46,
      71,    -1,    73,    -1,    71,    -1,    73,    78,    -1,    37,
      81,    78,    -1,    80,    81,    -1,    44,    -1,    46,     8,
       9,    37,    -1,    37,    71,    -1,    73,    -1,    -1,    -1,
      -1,    78,     8,     9,    81,    -1,    50,    51,    -1,    -1,
      -1,     8,     9,    -1,    -1,    73,    -1,    -1,    37,    -1,
      78,     8,     9,    81,    70,    71,    -1,    73,    -1,    73,
      -1,    37,    78,    -1,    78,    81,    -1,    81,    -1,    -1,
      37,     8,     9,    -1,    -1,    -1,    -1,    -1,     8,     9,
      37,    -1,    71,    -1,    73,    -1,    -1,    -1,    -1,    78,
       8,     9,    81,    -1,    -1,    71,    -1,    73,    -1,    -1,
      37,    -1,    78,    -1,    71,    81,    73,    37,     8,     9,
      -1,    78,    -1,    -1,    81,    -1,    73,     8,     9,    37,
      -1,    78,    -1,    80,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    -1,    73,    37,    -1,    -1,
      -1,    78,    -1,    73,    81,    -1,    37,    -1,    78,    -1,
      80,    81,    -1,    71,    -1,    73,    -1,    -1,    -1,    -1,
      78,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    69,
      -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,    78,    -1,
      -1,    81,    73,    -1,    -1,    -1,    -1,    78,    -1,    -1,
      81
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    83,     0,    84,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      29,    30,    32,    33,    34,    35,    36,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    67,    68,    71,    73,
      76,    85,    87,    76,     8,     9,    37,    42,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    55,    71,
      73,    78,    80,    81,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,   102,   104,   105,   106,   107,   108,
     112,    89,   106,   107,    89,    89,    80,    89,   104,    89,
     104,    89,    78,    99,   112,    47,    71,    77,    78,    88,
      98,   107,   112,    80,    97,   103,   106,   107,    97,    97,
      97,    97,    77,    89,   104,   109,    77,    89,    97,   104,
     109,   109,   102,   104,   108,   112,   108,   108,   109,   104,
     104,    76,    78,   106,    89,   106,    89,   106,    89,   104,
      89,   107,   104,    89,    74,    75,    75,    76,   112,   112,
      78,    78,    78,    78,     6,     8,     9,   110,   105,   112,
     113,    70,   107,   112,   112,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    78,    77,    77,    77,
      77,    77,   112,    77,    77,    77,    77,    77,    77,   110,
      78,    88,   107,   112,    90,    77,    77,    78,     9,    69,
      77,    77,    77,    77,    77,    77,    77,    77,    77,   109,
      77,    89,    97,   109,   109,   109,   109,    77,    78,    77,
      11,    77,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    86,   113,   113,   112,   112,   112,   112,
       7,   112,   112,    78,     8,    79,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    79,    89,    90,    92,
      95,    98,   104,   106,   107,    89,    89,    89,    98,    97,
      98,    96,   104,   106,   107,    95,    98,    89,    89,    89,
      89,    38,    39,    40,   105,   111,    89,   106,   107,    89,
      89,    89,    89,   105,    89,   105,    89,   105,    89,   104,
     105,    89,   105,    89,   104,    99,   112,    90,    79,    78,
      88,   107,    78,    88,   107,   112,    41,    69,    97,   106,
     107,    97,    97,    97,    97,    97,    97,    97,   105,   105,
      80,   112,   104,   112,   112,   104,   107,    89,   104,   104,
     106,   106,    89,    89,    89,   107,    89,   112,   105,    85,
      76,    76,    79,    79,    79,    79,   110,   111,   105,   113,
     113,   113,     6,     7,   113,   113,   113,   113,   113,    77,
      77,    79,    79,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    79,    90,    90,    77,    79,    77,    77,    77,
       9,    37,   101,    77,    77,    77,    77,    77,    77,    77,
      77,    78,    79,    79,   113,   113,    47,    92,    96,    89,
      89,    89,    89,    89,    89,    98,    98,    99,    79,    79,
      78,    88,   107,    98,    97,    97,    37,     9,    80,   104,
      80,   102,   103,   104,    89,   106,   100,   104,   112,   100,
     104,    89,    39,    90,    77,     9,    37,   101,     9,    77,
      77,    77,    77,    77,    79,    79,    97,    37,    89,    89,
     112,    89,    89
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
#line 68 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:
#line 75 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		if((yyvsp[(1) - (2)].sym)->type == LLAB && (yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 7:
#line 84 "a.y"
    {
		(yyvsp[(1) - (4)].sym)->type = LVAR;
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 8:
#line 89 "a.y"
    {
		if((yyvsp[(1) - (4)].sym)->value != (yyvsp[(3) - (4)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (4)].sym)->name);
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 9:
#line 95 "a.y"
    {
		nosched = (yyvsp[(1) - (2)].lval);
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
#line 127 "a.y"
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
#line 154 "a.y"
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
#line 173 "a.y"
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
#line 184 "a.y"
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 33:
#line 199 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 34:
#line 203 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 35:
#line 207 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].lval), &nullgen);
	}
    break;

  case 36:
#line 214 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 37:
#line 218 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 38:
#line 222 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
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
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 41:
#line 240 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 44:
#line 252 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 45:
#line 256 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 46:
#line 260 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 47:
#line 264 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 48:
#line 268 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 51:
#line 280 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), 0, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 52:
#line 287 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 53:
#line 294 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 54:
#line 298 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 55:
#line 305 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].addr).reg, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 56:
#line 309 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
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
#line 345 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
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
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 67:
#line 362 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &nullgen, 0, &(yyvsp[(3) - (4)].addr));
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
		outcode((yyvsp[(1) - (3)].lval), &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 70:
#line 374 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), &nullgen, 0, &(yyvsp[(4) - (5)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 73:
#line 386 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(5) - (6)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &nullgen, (yyvsp[(2) - (4)].lval), &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 76:
#line 398 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &nullgen, (yyvsp[(2) - (6)].lval), &(yyvsp[(5) - (6)].addr));
	}
    break;

  case 77:
#line 402 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = (yyvsp[(2) - (6)].lval);
		outcode((yyvsp[(1) - (6)].lval), &g, REG_R0+(yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 78:
#line 410 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = (yyvsp[(2) - (6)].lval);
		outcode((yyvsp[(1) - (6)].lval), &g, REG_R0+(yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 79:
#line 418 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = TYPE_CONST;
		g.offset = (yyvsp[(2) - (8)].lval);
		outcode((yyvsp[(1) - (8)].lval), &g, REG_R0+(yyvsp[(4) - (8)].lval), &(yyvsp[(7) - (8)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].lval), &nullgen);
	}
    break;

  case 82:
#line 437 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), 0, &nullgen);
	}
    break;

  case 83:
#line 441 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &nullgen);
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 86:
#line 456 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].addr).reg, &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 87:
#line 460 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 88:
#line 464 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 89:
#line 468 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(4) - (6)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 92:
#line 483 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(4) - (6)].addr));
	}
    break;

  case 93:
#line 487 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(4) - (6)].addr));
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
#line 506 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 98:
#line 513 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 99:
#line 517 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 100:
#line 525 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 101:
#line 529 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 102:
#line 533 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 103:
#line 537 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 106:
#line 549 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), 0, &nullgen);
	}
    break;

  case 107:
#line 556 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &nullgen);
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
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), 0, &nullgen);
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
		outcode((yyvsp[(1) - (3)].lval), &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 112:
#line 576 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), 0, &nullgen);
	}
    break;

  case 113:
#line 583 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), 0, &nullgen);
	}
    break;

  case 114:
#line 587 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), 0, &nullgen);
	}
    break;

  case 115:
#line 594 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != TYPE_CONST || (yyvsp[(4) - (4)].addr).type != TYPE_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 116:
#line 603 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != TYPE_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(4) - (4)].addr).type != NAME_EXTERN && (yyvsp[(4) - (4)].addr).type != NAME_STATIC && (yyvsp[(4) - (4)].addr).type != TYPE_MEM)
			yyerror("value for FUNCDATA must be symbol reference");
 		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 117:
#line 614 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &nullgen);
	}
    break;

  case 118:
#line 621 "a.y"
    {
		settext((yyvsp[(2) - (5)].addr).sym);
		outcode((yyvsp[(1) - (5)].lval), &(yyvsp[(2) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 119:
#line 626 "a.y"
    {
		settext((yyvsp[(2) - (7)].addr).sym);
		outcode((yyvsp[(1) - (7)].lval), &(yyvsp[(2) - (7)].addr), 0, &(yyvsp[(7) - (7)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (7)].lval);
		}
	}
    break;

  case 120:
#line 638 "a.y"
    {
		settext((yyvsp[(2) - (4)].addr).sym);
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 121:
#line 643 "a.y"
    {
		settext((yyvsp[(2) - (6)].addr).sym);
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 122:
#line 656 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 123:
#line 664 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 124:
#line 672 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 125:
#line 683 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, 0, &nullgen);
	}
    break;

  case 126:
#line 689 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 127:
#line 695 "a.y"
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
#line 706 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 131:
#line 718 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 132:
#line 726 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);	/* whole register */
	}
    break;

  case 133:
#line 734 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 134:
#line 742 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 135:
#line 750 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 136:
#line 756 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= 1024)
			yyerror("SPR/DCR out of range");
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (4)].lval) + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 138:
#line 767 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 139:
#line 775 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 140:
#line 781 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_F0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 141:
#line 789 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 142:
#line 795 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_C0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 143:
#line 803 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 144:
#line 811 "a.y"
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
#line 832 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 146:
#line 839 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (2)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 147:
#line 846 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (3)].lval);
		(yyval.addr).u.argsize = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 148:
#line 853 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (4)].lval);
		(yyval.addr).u.argsize = (yyvsp[(4) - (4)].lval);
	}
    break;

  case 149:
#line 862 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = TYPE_ADDR;
	}
    break;

  case 150:
#line 867 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 151:
#line 875 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 152:
#line 881 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 153:
#line 888 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 155:
#line 897 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = REG_R0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 156:
#line 905 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 157:
#line 912 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 159:
#line 923 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 160:
#line 932 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(3) - (4)].lval);
		(yyval.addr).sym = nil;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 161:
#line 940 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 162:
#line 948 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = NAME_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 165:
#line 960 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 166:
#line 964 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 167:
#line 968 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 172:
#line 980 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 173:
#line 984 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 174:
#line 988 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 175:
#line 992 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 176:
#line 996 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 178:
#line 1003 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 179:
#line 1007 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 180:
#line 1011 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 181:
#line 1015 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 182:
#line 1019 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 183:
#line 1023 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 184:
#line 1027 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 185:
#line 1031 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 186:
#line 1035 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 187:
#line 1039 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 3253 "y.tab.c"
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



