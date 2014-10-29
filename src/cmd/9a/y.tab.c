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
     LDATA = 279,
     LRETRN = 280,
     LCONST = 281,
     LSP = 282,
     LSB = 283,
     LFP = 284,
     LPC = 285,
     LCREG = 286,
     LFLUSH = 287,
     LREG = 288,
     LFREG = 289,
     LR = 290,
     LCR = 291,
     LF = 292,
     LFPSCR = 293,
     LLR = 294,
     LCTR = 295,
     LSPR = 296,
     LSPREG = 297,
     LSEG = 298,
     LMSR = 299,
     LPCDAT = 300,
     LFUNCDAT = 301,
     LSCHED = 302,
     LXLD = 303,
     LXST = 304,
     LXOP = 305,
     LXMV = 306,
     LRLWM = 307,
     LMOVMW = 308,
     LMOVEM = 309,
     LMOVFL = 310,
     LMTFSB = 311,
     LMA = 312,
     LFCONST = 313,
     LSCONST = 314,
     LNAME = 315,
     LLAB = 316,
     LVAR = 317
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
#define LDATA 279
#define LRETRN 280
#define LCONST 281
#define LSP 282
#define LSB 283
#define LFP 284
#define LPC 285
#define LCREG 286
#define LFLUSH 287
#define LREG 288
#define LFREG 289
#define LR 290
#define LCR 291
#define LF 292
#define LFPSCR 293
#define LLR 294
#define LCTR 295
#define LSPR 296
#define LSPREG 297
#define LSEG 298
#define LMSR 299
#define LPCDAT 300
#define LFUNCDAT 301
#define LSCHED 302
#define LXLD 303
#define LXST 304
#define LXOP 305
#define LXMV 306
#define LRLWM 307
#define LMOVMW 308
#define LMOVEM 309
#define LMOVFL 310
#define LMTFSB 311
#define LMA 312
#define LFCONST 313
#define LSCONST 314
#define LNAME 315
#define LLAB 316
#define LVAR 317




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
#line 236 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 249 "y.tab.c"

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
#define YYNTOKENS  81
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  31
/* YYNRULES -- Number of rules.  */
#define YYNRULES  183
/* YYNRULES -- Number of states.  */
#define YYNSTATES  453

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   317

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    79,    12,     5,     2,
      77,    78,    10,     8,    76,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    73,    75,
       6,    74,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    80,     2,     2,     2,
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
      65,    66,    67,    68,    69,    70,    71,    72
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,     8,    13,    18,    23,    26,
      28,    31,    34,    39,    44,    49,    54,    59,    64,    69,
      74,    79,    84,    89,    94,    99,   104,   109,   114,   119,
     124,   129,   134,   141,   146,   151,   156,   163,   168,   173,
     180,   187,   194,   199,   204,   211,   216,   223,   228,   235,
     240,   245,   248,   255,   260,   265,   270,   277,   282,   287,
     292,   297,   302,   307,   312,   317,   320,   323,   328,   332,
     336,   342,   347,   352,   359,   364,   369,   376,   383,   390,
     399,   404,   409,   413,   416,   421,   426,   433,   442,   447,
     454,   459,   464,   471,   478,   487,   496,   505,   514,   519,
     524,   529,   536,   541,   548,   553,   558,   561,   564,   568,
     572,   576,   580,   583,   587,   591,   596,   601,   604,   609,
     616,   625,   632,   639,   646,   649,   654,   657,   659,   661,
     663,   665,   667,   669,   671,   673,   678,   680,   682,   687,
     689,   694,   696,   701,   703,   707,   710,   713,   716,   720,
     723,   725,   730,   734,   740,   742,   747,   752,   758,   766,
     767,   769,   770,   773,   776,   778,   780,   782,   784,   786,
     789,   792,   795,   799,   801,   805,   809,   813,   817,   821,
     826,   831,   835,   839
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      82,     0,    -1,    -1,    82,    83,    -1,    -1,    70,    73,
      84,    83,    -1,    70,    74,   111,    75,    -1,    72,    74,
     111,    75,    -1,    57,    75,    -1,    75,    -1,    85,    75,
      -1,     1,    75,    -1,    13,    87,    76,    87,    -1,    13,
     105,    76,    87,    -1,    13,   104,    76,    87,    -1,    14,
      87,    76,    87,    -1,    14,   105,    76,    87,    -1,    14,
     104,    76,    87,    -1,    22,   105,    76,    96,    -1,    22,
     104,    76,    96,    -1,    22,   101,    76,    96,    -1,    22,
      96,    76,    96,    -1,    22,    96,    76,   105,    -1,    22,
      96,    76,   104,    -1,    13,    87,    76,   105,    -1,    13,
      87,    76,   104,    -1,    14,    87,    76,   105,    -1,    14,
      87,    76,   104,    -1,    13,    96,    76,   105,    -1,    13,
      96,    76,   104,    -1,    13,    94,    76,    96,    -1,    13,
      96,    76,    94,    -1,    13,    96,    76,   102,    76,    94,
      -1,    13,    94,    76,    97,    -1,    13,   102,    76,    95,
      -1,    66,   102,    76,   110,    -1,    13,    87,    76,   102,
      76,    90,    -1,    13,    87,    76,    97,    -1,    13,    87,
      76,    90,    -1,    18,    87,    76,   103,    76,    87,    -1,
      18,   102,    76,   103,    76,    87,    -1,    18,    87,    76,
     102,    76,    87,    -1,    18,    87,    76,    87,    -1,    18,
     102,    76,    87,    -1,    16,    87,    76,   103,    76,    87,
      -1,    16,    87,    76,    87,    -1,    17,    87,    76,   103,
      76,    87,    -1,    17,    87,    76,    87,    -1,    17,   102,
      76,   103,    76,    87,    -1,    17,   102,    76,    87,    -1,
      15,    87,    76,    87,    -1,    15,    87,    -1,    67,    87,
      76,   103,    76,    87,    -1,    13,   102,    76,    87,    -1,
      13,   100,    76,    87,    -1,    20,    98,    76,    98,    -1,
      20,    98,    76,   110,    76,    98,    -1,    13,    97,    76,
      97,    -1,    13,    93,    76,    97,    -1,    13,    90,    76,
      87,    -1,    13,    93,    76,    87,    -1,    13,    88,    76,
      87,    -1,    13,    87,    76,    88,    -1,    13,    97,    76,
      93,    -1,    13,    87,    76,    93,    -1,    21,    86,    -1,
      21,   105,    -1,    21,    77,    88,    78,    -1,    21,    76,
      86,    -1,    21,    76,   105,    -1,    21,    76,    77,    88,
      78,    -1,    21,    97,    76,    86,    -1,    21,    97,    76,
     105,    -1,    21,    97,    76,    77,    88,    78,    -1,    21,
     110,    76,    86,    -1,    21,   110,    76,   105,    -1,    21,
     110,    76,    77,    88,    78,    -1,    21,   110,    76,   110,
      76,    86,    -1,    21,   110,    76,   110,    76,   105,    -1,
      21,   110,    76,   110,    76,    77,    88,    78,    -1,    27,
      87,    76,   103,    -1,    27,   102,    76,   103,    -1,    27,
      87,   107,    -1,    27,   107,    -1,    23,    96,    76,    96,
      -1,    25,    96,    76,    96,    -1,    25,    96,    76,    96,
      76,    96,    -1,    26,    96,    76,    96,    76,    96,    76,
      96,    -1,    24,    96,    76,    96,    -1,    24,    96,    76,
      96,    76,    97,    -1,    19,    87,    76,    87,    -1,    19,
      87,    76,   102,    -1,    19,    87,    76,    87,    76,    97,
      -1,    19,    87,    76,   102,    76,    97,    -1,    62,   102,
      76,    87,    76,   102,    76,    87,    -1,    62,   102,    76,
      87,    76,    99,    76,    87,    -1,    62,    87,    76,    87,
      76,   102,    76,    87,    -1,    62,    87,    76,    87,    76,
      99,    76,    87,    -1,    63,   105,    76,    87,    -1,    63,
      87,    76,   105,    -1,    58,   104,    76,    87,    -1,    58,
     104,    76,   102,    76,    87,    -1,    59,    87,    76,   104,
      -1,    59,    87,    76,   102,    76,   104,    -1,    61,   104,
      76,    87,    -1,    61,    87,    76,   104,    -1,    60,   104,
      -1,    29,   107,    -1,    29,    87,   107,    -1,    29,    96,
     107,    -1,    29,    76,    87,    -1,    29,    76,    96,    -1,
      29,   102,    -1,    32,   102,   107,    -1,    32,   100,   107,
      -1,    55,   102,    76,   102,    -1,    56,   102,    76,   105,
      -1,    30,   107,    -1,    33,   106,    76,   102,    -1,    33,
     106,    76,   110,    76,   102,    -1,    33,   106,    76,   110,
      76,   102,     9,   110,    -1,    34,   106,    11,   110,    76,
     102,    -1,    34,   106,    11,   110,    76,   100,    -1,    34,
     106,    11,   110,    76,   101,    -1,    35,   107,    -1,   110,
      77,    40,    78,    -1,    70,   108,    -1,   103,    -1,    89,
      -1,    91,    -1,    49,    -1,    46,    -1,    50,    -1,    54,
      -1,    52,    -1,    51,    77,   110,    78,    -1,    92,    -1,
      48,    -1,    48,    77,   110,    78,    -1,    44,    -1,    47,
      77,   110,    78,    -1,    41,    -1,    46,    77,   110,    78,
      -1,   110,    -1,   110,    76,   110,    -1,    79,   105,    -1,
      79,    69,    -1,    79,    68,    -1,    79,     9,    68,    -1,
      79,   110,    -1,    43,    -1,    45,    77,   110,    78,    -1,
      77,   103,    78,    -1,    77,   103,     8,   103,    78,    -1,
     106,    -1,   110,    77,   103,    78,    -1,   110,    77,   109,
      78,    -1,    70,   108,    77,   109,    78,    -1,    70,     6,
       7,   108,    77,    38,    78,    -1,    -1,    76,    -1,    -1,
       8,   110,    -1,     9,   110,    -1,    38,    -1,    37,    -1,
      39,    -1,    36,    -1,    72,    -1,     9,   110,    -1,     8,
     110,    -1,    80,   110,    -1,    77,   111,    78,    -1,   110,
      -1,   111,     8,   111,    -1,   111,     9,   111,    -1,   111,
      10,   111,    -1,   111,    11,   111,    -1,   111,    12,   111,
      -1,   111,     6,     6,   111,    -1,   111,     7,     7,   111,
      -1,   111,     5,   111,    -1,   111,     4,   111,    -1,   111,
       3,   111,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    66,    66,    67,    71,    70,    79,    84,    90,    94,
      95,    96,   102,   106,   110,   114,   118,   122,   129,   133,
     137,   141,   145,   149,   156,   160,   164,   168,   175,   179,
     186,   190,   194,   198,   202,   206,   213,   217,   221,   231,
     235,   239,   243,   247,   251,   255,   259,   263,   267,   271,
     275,   279,   286,   293,   297,   304,   308,   316,   320,   324,
     328,   332,   336,   340,   344,   353,   357,   361,   365,   369,
     373,   377,   381,   385,   389,   393,   397,   401,   409,   417,
     428,   432,   436,   440,   447,   451,   455,   459,   463,   467,
     474,   478,   482,   486,   493,   497,   501,   505,   512,   516,
     524,   528,   532,   536,   540,   544,   548,   555,   559,   563,
     567,   571,   575,   582,   586,   593,   602,   613,   620,   625,
     632,   642,   646,   650,   657,   663,   669,   680,   688,   689,
     692,   700,   708,   716,   723,   729,   735,   738,   746,   754,
     760,   768,   774,   782,   790,   811,   816,   824,   830,   837,
     845,   846,   854,   861,   871,   872,   881,   889,   897,   906,
     907,   910,   913,   917,   923,   924,   925,   928,   929,   933,
     937,   941,   945,   951,   952,   956,   960,   964,   968,   972,
     976,   980,   984,   988
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
  "LDATA", "LRETRN", "LCONST", "LSP", "LSB", "LFP", "LPC", "LCREG",
  "LFLUSH", "LREG", "LFREG", "LR", "LCR", "LF", "LFPSCR", "LLR", "LCTR",
  "LSPR", "LSPREG", "LSEG", "LMSR", "LPCDAT", "LFUNCDAT", "LSCHED", "LXLD",
  "LXST", "LXOP", "LXMV", "LRLWM", "LMOVMW", "LMOVEM", "LMOVFL", "LMTFSB",
  "LMA", "LFCONST", "LSCONST", "LNAME", "LLAB", "LVAR", "':'", "'='",
  "';'", "','", "'('", "')'", "'$'", "'~'", "$accept", "prog", "line",
  "@1", "inst", "rel", "rreg", "xlreg", "lr", "lcr", "ctr", "msr", "psr",
  "fpscr", "fpscrf", "freg", "creg", "cbit", "mask", "ximm", "fimm", "imm",
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
     315,   316,   317,    58,    61,    59,    44,    40,    41,    36,
     126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    81,    82,    82,    84,    83,    83,    83,    83,    83,
      83,    83,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    86,    86,    87,    88,    88,
      89,    90,    91,    92,    93,    93,    93,    94,    95,    96,
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
       4,     4,     6,     4,     4,     4,     6,     4,     4,     6,
       6,     6,     4,     4,     6,     4,     6,     4,     6,     4,
       4,     2,     6,     4,     4,     4,     6,     4,     4,     4,
       4,     4,     4,     4,     4,     2,     2,     4,     3,     3,
       5,     4,     4,     6,     4,     4,     6,     6,     6,     8,
       4,     4,     3,     2,     4,     4,     6,     8,     4,     6,
       4,     4,     6,     6,     8,     8,     8,     8,     4,     4,
       4,     6,     4,     6,     4,     4,     2,     2,     3,     3,
       3,     3,     2,     3,     3,     4,     4,     2,     4,     6,
       8,     6,     6,     6,     2,     4,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     4,     1,     1,     4,     1,
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
     159,     0,     0,     0,   159,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     9,     3,
       0,    11,     0,     0,   167,   141,   150,   139,     0,   131,
       0,   137,   130,   132,     0,   134,   133,   161,   168,     0,
       0,     0,     0,     0,   128,     0,   129,   136,     0,     0,
       0,     0,     0,     0,   127,     0,     0,   154,     0,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   143,     0,   161,     0,     0,    65,     0,    66,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     160,   159,     0,    83,   160,   159,   159,   112,   107,   117,
     159,   159,     0,     0,     0,   124,     0,     0,     8,     0,
       0,     0,   106,     0,     0,     0,     0,     0,     0,     0,
       0,     4,     0,     0,    10,   170,   169,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   173,     0,   146,   145,
     149,   171,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   149,
       0,     0,     0,     0,     0,     0,   126,     0,    68,    69,
       0,     0,     0,     0,     0,     0,   147,     0,     0,     0,
       0,     0,     0,     0,     0,   160,    82,     0,   110,   111,
     108,   109,   114,   113,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   161,   162,   163,     0,
       0,   152,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   172,    12,    62,    38,    64,    37,     0,    25,
      24,    61,    59,    60,    58,    30,    33,    31,     0,    29,
      28,    63,    57,    54,     0,    53,    34,    14,    13,   165,
     164,   166,     0,     0,    15,    27,    26,    17,    16,    50,
      45,   127,    47,   127,    49,   127,    42,     0,   127,    43,
     127,    90,    91,    55,   143,     0,    67,     0,    71,    72,
       0,    74,    75,     0,     0,   148,    21,    23,    22,    20,
      19,    18,    84,    88,    85,     0,    80,    81,   118,     0,
       0,   115,   116,   100,     0,     0,   102,   105,   104,     0,
       0,    99,    98,    35,     0,     5,     6,     7,   151,   142,
     140,   135,     0,     0,     0,   183,   182,   181,     0,     0,
     174,   175,   176,   177,   178,     0,     0,     0,   155,   156,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    70,
       0,     0,     0,   125,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   157,   153,   179,   180,   131,
      36,    32,     0,    44,    46,    48,    41,    39,    40,    92,
      93,    56,    73,    76,     0,    77,    78,    89,    86,     0,
     119,     0,   122,   123,   121,   101,   103,     0,     0,     0,
       0,     0,    52,     0,   138,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   158,    79,    87,   120,    97,    96,
     144,    95,    94
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,    39,   229,    40,    97,    62,    63,    64,    65,
      66,    67,    68,    69,   276,    70,    71,    91,   427,    72,
     103,    73,    74,    75,   159,    77,   113,   154,   283,   156,
     157
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -176
static const yytype_int16 yypact[] =
{
    -176,   464,  -176,   -63,   560,   637,    97,    97,   -24,   -24,
      97,   556,   317,   618,    12,    12,    12,    12,   -27,    47,
     -47,   -29,   725,   725,   -47,   -26,   -26,   -43,   -17,    97,
     -17,   -23,   -24,   658,   -26,    97,    51,   -11,  -176,  -176,
      -2,  -176,   556,   556,  -176,  -176,  -176,  -176,    -1,     2,
      11,  -176,  -176,  -176,    24,  -176,  -176,    91,  -176,    26,
     716,   556,    57,    65,  -176,    85,  -176,  -176,    92,    98,
     104,   110,   119,   134,  -176,   155,   160,  -176,    69,   162,
     165,   170,   172,   176,   556,   179,   180,   182,   183,   185,
     556,   187,  -176,     2,    91,   736,   326,  -176,   196,  -176,
      52,     6,   197,   198,   202,   203,   215,   216,   217,   222,
    -176,   223,   235,  -176,    73,   -47,   -47,  -176,  -176,  -176,
     -47,   -47,   239,    79,   178,  -176,   240,   246,  -176,    97,
     247,   248,  -176,   252,   253,   255,   262,   263,   266,   267,
     268,  -176,   556,   556,  -176,  -176,  -176,   556,   556,   556,
     556,   193,   556,   556,   166,     9,  -176,   278,  -176,  -176,
      69,  -176,   607,    97,    97,   109,    20,   683,    61,    97,
      27,    97,    97,   340,   637,    97,    97,    97,    97,  -176,
      97,    97,   -24,    97,   -24,   556,   166,   326,  -176,  -176,
     199,   152,   742,   762,   153,   283,  -176,   696,    12,    12,
      12,    12,    12,    12,    12,    97,  -176,    97,  -176,  -176,
    -176,  -176,  -176,  -176,   382,     4,   556,   -26,   725,   -24,
      72,   -17,    97,    97,    97,   725,    97,   556,    97,   527,
     436,   567,   274,   276,   277,   279,   154,  -176,  -176,     4,
      97,  -176,   556,   556,   556,   353,   339,   556,   556,   556,
     556,   556,  -176,  -176,  -176,  -176,  -176,  -176,   284,  -176,
    -176,  -176,  -176,  -176,  -176,  -176,  -176,  -176,   295,  -176,
    -176,  -176,  -176,  -176,   303,  -176,  -176,  -176,  -176,  -176,
    -176,  -176,   304,   308,  -176,  -176,  -176,  -176,  -176,  -176,
    -176,   305,  -176,   316,  -176,   323,  -176,   325,   333,  -176,
     334,   335,   336,  -176,   343,   342,  -176,   326,  -176,  -176,
     326,  -176,  -176,   139,   344,  -176,  -176,  -176,  -176,  -176,
    -176,  -176,  -176,   345,   348,   349,  -176,  -176,  -176,   354,
     355,  -176,  -176,  -176,   356,   357,  -176,  -176,  -176,   360,
     373,  -176,  -176,  -176,   374,  -176,  -176,  -176,  -176,  -176,
    -176,  -176,   327,   377,   379,   298,   612,   506,   556,   556,
     125,   125,  -176,  -176,  -176,   405,   410,   556,  -176,  -176,
      97,    97,    97,    97,    97,    97,    -8,    -8,   556,  -176,
     385,   388,   782,  -176,    -8,    12,    12,   -26,   381,    97,
     -17,   382,   382,    97,   429,  -176,  -176,   498,   498,  -176,
    -176,  -176,   390,  -176,  -176,  -176,  -176,  -176,  -176,  -176,
    -176,  -176,  -176,  -176,   326,  -176,  -176,  -176,  -176,   393,
     462,   712,  -176,  -176,  -176,  -176,  -176,   398,   399,   416,
     419,   426,  -176,   451,  -176,   454,    12,   556,   328,    97,
      97,   556,    97,    97,  -176,  -176,  -176,  -176,  -176,  -176,
    -176,  -176,  -176
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -176,  -176,   306,  -176,  -176,   -88,    -5,   -73,  -176,  -154,
    -176,  -176,  -137,  -158,  -176,    67,    39,  -175,   141,   -15,
     149,   113,   167,    80,    32,   200,   124,   -83,   299,    35,
      70
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      79,    82,    83,    85,    87,    89,   120,   188,   255,   267,
     303,   186,    41,   111,   115,   195,    46,   240,    48,    46,
      46,    48,    48,   191,   131,   256,   133,   135,   137,   110,
     140,   271,   128,    45,    42,    43,    76,    81,    93,    78,
      78,   279,   280,   281,    99,   105,    92,   100,    78,   110,
      60,    98,    84,    84,   129,    84,    47,   123,   123,    50,
     129,    45,    44,   143,    47,   138,    93,    50,    78,    46,
      46,    48,    48,   144,   196,   274,   147,   145,   146,   148,
     102,   106,   107,   108,   109,    80,   116,   241,   149,   254,
      46,    47,    48,   104,    50,   160,   161,   151,    58,   152,
     153,   150,    45,    90,   308,   311,    61,    93,   130,   208,
     132,   134,    54,    55,   305,    56,    46,    47,    48,   179,
      50,    86,    88,   114,   141,   142,    84,   189,   193,   194,
     190,   112,   117,   162,   121,   249,   250,   251,   126,   127,
      46,   163,    48,   118,   119,   136,   173,   139,   125,   129,
      45,    84,    46,   352,    48,    93,   215,   253,   261,   262,
     263,   164,   152,   153,   273,   275,   277,   278,   165,   284,
     287,   288,   289,   290,   166,   292,   294,   296,   299,   301,
     167,   209,   232,   233,   234,   235,   168,   237,   238,   216,
     279,   280,   281,   314,   260,   169,    46,    78,    48,   270,
     236,   257,    78,   411,   264,   266,   286,   272,   401,    78,
     170,   400,   230,   231,   333,   382,   194,   338,   339,   340,
     304,   342,   122,   124,   309,   312,   155,   190,   313,   318,
     306,   171,    78,   265,   380,   206,   172,   381,   174,   210,
     211,   175,   259,   239,   212,   213,   176,   269,   177,   329,
     332,   330,   178,    78,   285,   180,   181,   341,   182,   183,
      78,   184,   343,   185,   316,   319,   320,   321,   322,   323,
     324,   325,   192,   197,   198,   258,   194,   317,   199,   200,
     268,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   201,   202,   203,   415,   297,   155,   302,   204,   205,
     336,   337,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   207,   355,   356,   357,   214,   217,   360,   361,   362,
     363,   364,   218,   219,   220,    42,    43,   328,   221,   222,
     331,   223,   334,   335,    42,    43,    42,    43,   224,   225,
     282,   435,   226,   227,   228,   291,   359,   293,   295,   298,
     300,   315,   348,    44,   349,   350,   252,   351,    45,   358,
     365,   282,    44,    93,    44,   403,   404,   405,   406,   407,
     408,   366,   326,   422,   327,    52,    53,   279,   280,   281,
     367,   370,   368,    46,   425,    48,   369,    94,   432,    58,
      42,    43,   371,    95,    96,   344,   315,    61,    58,   372,
      58,   373,   402,    90,   394,    90,    61,   354,    61,   374,
     375,   376,   377,    92,   416,   409,   410,   190,    44,   378,
     379,   384,   383,   417,   385,   386,   429,   429,   397,   398,
     387,   388,   389,   390,   448,   449,   391,   451,   452,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   392,
     393,   399,   418,   419,    58,   395,   160,   396,    51,    90,
     421,    84,    61,   412,     2,     3,   413,   433,   434,   436,
     426,   437,   447,   146,   439,   440,   450,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,   441,    19,    20,   442,    21,    22,    23,    24,
     420,   424,   443,   446,   428,   431,   247,   248,   249,   250,
     251,   346,   245,   246,   247,   248,   249,   250,   251,    25,
      26,    27,    28,    29,    30,    31,    32,    33,     3,   444,
      34,    35,   445,   430,    36,   345,    37,   423,   353,    38,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,     0,    19,    20,     0,    21,
      22,    23,    24,     0,    42,    43,     0,     0,    42,    43,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
       0,     0,    25,    26,    27,    28,    29,    30,    31,    32,
      33,     0,    44,    34,    35,     0,    44,    36,     0,    37,
       0,    45,    38,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,     0,    56,    42,    43,   244,   245,   246,
     247,   248,   249,   250,   251,     0,    42,    43,    58,     0,
      57,     0,    58,    90,     0,     0,    61,    59,     0,    60,
      61,     0,   347,    44,     0,    42,    43,     0,    45,     0,
      46,     0,    48,    49,    44,     0,    52,    53,    54,    55,
       0,    56,    47,     0,     0,    50,    42,    43,     0,     0,
       0,     0,     0,    44,     0,     0,     0,    57,     0,    58,
      46,     0,    48,     0,    59,     0,    84,    61,    57,     0,
      58,    42,    43,     0,    44,    59,     0,   101,    61,     0,
       0,    46,     0,    48,    42,    43,     0,    57,     0,    58,
       0,     0,     0,     0,    59,     0,     0,    61,     0,    44,
      42,   438,     0,     0,    42,    43,     0,     0,    57,     0,
      58,    51,    44,    42,    43,    90,     0,     0,    61,     0,
      47,     0,     0,    50,    42,    43,     0,     0,    44,     0,
      42,    43,    44,    57,     0,    58,     0,     0,     0,     0,
      59,    44,    84,    61,     0,     0,    57,     0,    58,     0,
      42,    43,    44,    59,     0,     0,    61,     0,    44,     0,
     196,   158,    57,     0,    58,   158,    57,     0,    58,    90,
      42,    43,    61,    90,     0,    57,    61,    58,    44,     0,
       0,     0,    90,     0,     0,    61,    94,     0,    58,     0,
       0,     0,    94,   187,    58,     0,    61,     0,    44,   307,
       0,     0,    61,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    94,     0,    58,     0,     0,     0,     0,   310,
       0,     0,    61,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    94,     0,    58,     0,     0,     0,     0,   414,
       0,     0,    61
};

static const yytype_int16 yycheck[] =
{
       5,     6,     7,     8,     9,    10,    21,    95,   162,   167,
     185,    94,    75,    18,    19,     9,    43,     8,    45,    43,
      43,    45,    45,    96,    29,   162,    31,    32,    33,    76,
      35,   168,    75,    41,     8,     9,     4,     5,    46,     4,
       5,    37,    38,    39,    12,    13,    11,    12,    13,    76,
      79,    12,    79,    79,    77,    79,    44,    22,    23,    47,
      77,    41,    36,    74,    44,    33,    46,    47,    33,    43,
      43,    45,    45,    75,    68,    48,    77,    42,    43,    77,
      13,    14,    15,    16,    17,     5,    19,    78,    77,   162,
      43,    44,    45,    13,    47,    60,    61,     6,    72,     8,
       9,    77,    41,    77,   192,   193,    80,    46,    28,   114,
      30,    31,    51,    52,   187,    54,    43,    44,    45,    84,
      47,     8,     9,    76,    73,    74,    79,    95,    76,    77,
      95,    18,    19,    76,    21,    10,    11,    12,    25,    26,
      43,    76,    45,    19,    20,    32,    77,    34,    24,    77,
      41,    79,    43,   236,    45,    46,    77,   162,   163,   164,
     165,    76,     8,     9,   169,   170,   171,   172,    76,   174,
     175,   176,   177,   178,    76,   180,   181,   182,   183,   184,
      76,   114,   147,   148,   149,   150,    76,   152,   153,    11,
      37,    38,    39,    40,   162,    76,    43,   162,    45,   167,
       7,   162,   167,   378,   165,   166,   174,   168,   366,   174,
      76,   365,   142,   143,   219,    76,    77,   222,   223,   224,
     185,   226,    22,    23,   192,   193,    59,   192,   193,   197,
      78,    76,   197,   166,   307,   111,    76,   310,    76,   115,
     116,    76,   162,    77,   120,   121,    76,   167,    76,   214,
     218,   216,    76,   218,   174,    76,    76,   225,    76,    76,
     225,    76,   227,    76,   197,   198,   199,   200,   201,   202,
     203,   204,    76,    76,    76,   162,    77,   197,    76,    76,
     167,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    76,    76,    76,   382,   182,   129,   184,    76,    76,
     220,   221,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    76,   242,   243,   244,    76,    76,   247,   248,   249,
     250,   251,    76,    76,    76,     8,     9,   214,    76,    76,
     217,    76,   219,   220,     8,     9,     8,     9,    76,    76,
     173,   414,    76,    76,    76,   178,     7,   180,   181,   182,
     183,    68,    78,    36,    78,    78,    78,    78,    41,     6,
      76,   194,    36,    46,    36,   370,   371,   372,   373,   374,
     375,    76,   205,   388,   207,    49,    50,    37,    38,    39,
      77,    76,    78,    43,   389,    45,    78,    70,   393,    72,
       8,     9,    76,    76,    77,   228,    68,    80,    72,    76,
      72,    76,   367,    77,    77,    77,    80,   240,    80,    76,
      76,    76,    76,   378,   382,   376,   377,   382,    36,    76,
      78,    76,    78,   384,    76,    76,   391,   392,   358,   359,
      76,    76,    76,    76,   439,   440,    76,   442,   443,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    76,
      76,    46,   385,   386,    72,    78,   421,    78,    48,    77,
      79,    79,    80,    78,     0,     1,    78,    38,    78,    76,
     390,     9,   437,   438,    76,    76,   441,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    76,    29,    30,    76,    32,    33,    34,    35,
     387,   388,    76,   436,   391,   392,     8,     9,    10,    11,
      12,    75,     6,     7,     8,     9,    10,    11,    12,    55,
      56,    57,    58,    59,    60,    61,    62,    63,     1,    78,
      66,    67,    78,   392,    70,   229,    72,   388,   239,    75,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    -1,    29,    30,    -1,    32,
      33,    34,    35,    -1,     8,     9,    -1,    -1,     8,     9,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      -1,    -1,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    -1,    36,    66,    67,    -1,    36,    70,    -1,    72,
      -1,    41,    75,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    -1,    54,     8,     9,     5,     6,     7,
       8,     9,    10,    11,    12,    -1,     8,     9,    72,    -1,
      70,    -1,    72,    77,    -1,    -1,    80,    77,    -1,    79,
      80,    -1,    75,    36,    -1,     8,     9,    -1,    41,    -1,
      43,    -1,    45,    46,    36,    -1,    49,    50,    51,    52,
      -1,    54,    44,    -1,    -1,    47,     8,     9,    -1,    -1,
      -1,    -1,    -1,    36,    -1,    -1,    -1,    70,    -1,    72,
      43,    -1,    45,    -1,    77,    -1,    79,    80,    70,    -1,
      72,     8,     9,    -1,    36,    77,    -1,    79,    80,    -1,
      -1,    43,    -1,    45,     8,     9,    -1,    70,    -1,    72,
      -1,    -1,    -1,    -1,    77,    -1,    -1,    80,    -1,    36,
       8,     9,    -1,    -1,     8,     9,    -1,    -1,    70,    -1,
      72,    48,    36,     8,     9,    77,    -1,    -1,    80,    -1,
      44,    -1,    -1,    47,     8,     9,    -1,    -1,    36,    -1,
       8,     9,    36,    70,    -1,    72,    -1,    -1,    -1,    -1,
      77,    36,    79,    80,    -1,    -1,    70,    -1,    72,    -1,
       8,     9,    36,    77,    -1,    -1,    80,    -1,    36,    -1,
      68,    69,    70,    -1,    72,    69,    70,    -1,    72,    77,
       8,     9,    80,    77,    -1,    70,    80,    72,    36,    -1,
      -1,    -1,    77,    -1,    -1,    80,    70,    -1,    72,    -1,
      -1,    -1,    70,    77,    72,    -1,    80,    -1,    36,    77,
      -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    70,    -1,    72,    -1,    -1,    -1,    -1,    77,
      -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    70,    -1,    72,    -1,    -1,    -1,    -1,    77,
      -1,    -1,    80
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    82,     0,     1,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    29,
      30,    32,    33,    34,    35,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    66,    67,    70,    72,    75,    83,
      85,    75,     8,     9,    36,    41,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    54,    70,    72,    77,
      79,    80,    87,    88,    89,    90,    91,    92,    93,    94,
      96,    97,   100,   102,   103,   104,   105,   106,   110,    87,
     104,   105,    87,    87,    79,    87,   102,    87,   102,    87,
      77,    98,   110,    46,    70,    76,    77,    86,    97,   105,
     110,    79,    96,   101,   104,   105,    96,    96,    96,    96,
      76,    87,   102,   107,    76,    87,    96,   102,   107,   107,
     100,   102,   106,   110,   106,   107,   102,   102,    75,    77,
     104,    87,   104,    87,   104,    87,   102,    87,   105,   102,
      87,    73,    74,    74,    75,   110,   110,    77,    77,    77,
      77,     6,     8,     9,   108,   103,   110,   111,    69,   105,
     110,   110,    76,    76,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    77,    76,    76,    76,    76,    76,   110,
      76,    76,    76,    76,    76,    76,   108,    77,    86,   105,
     110,    88,    76,    76,    77,     9,    68,    76,    76,    76,
      76,    76,    76,    76,    76,    76,   107,    76,    87,    96,
     107,   107,   107,   107,    76,    77,    11,    76,    76,    76,
      76,    76,    76,    76,    76,    76,    76,    76,    76,    84,
     111,   111,   110,   110,   110,   110,     7,   110,   110,    77,
       8,    78,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    78,    87,    88,    90,    93,    97,   102,   104,
     105,    87,    87,    87,    97,    96,    97,    94,   102,   104,
     105,    93,    97,    87,    48,    87,    95,    87,    87,    37,
      38,    39,   103,   109,    87,   104,   105,    87,    87,    87,
      87,   103,    87,   103,    87,   103,    87,   102,   103,    87,
     103,    87,   102,    98,   110,    88,    78,    77,    86,   105,
      77,    86,   105,   110,    40,    68,    96,   104,   105,    96,
      96,    96,    96,    96,    96,    96,   103,   103,   102,   110,
     110,   102,   105,    87,   102,   102,   104,   104,    87,    87,
      87,   105,    87,   110,   103,    83,    75,    75,    78,    78,
      78,    78,   108,   109,   103,   111,   111,   111,     6,     7,
     111,   111,   111,   111,   111,    76,    76,    77,    78,    78,
      76,    76,    76,    76,    76,    76,    76,    76,    76,    78,
      88,    88,    76,    78,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    77,    78,    78,   111,   111,    46,
      90,    94,   110,    87,    87,    87,    87,    87,    87,    97,
      97,    98,    78,    78,    77,    86,   105,    97,    96,    96,
     102,    79,   100,   101,   102,    87,   104,    99,   102,   110,
      99,   102,    87,    38,    78,    88,    76,     9,     9,    76,
      76,    76,    76,    76,    78,    78,    96,   110,    87,    87,
     110,    87,    87
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 13:
#line 107 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 14:
#line 111 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 15:
#line 115 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 16:
#line 119 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 17:
#line 123 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 18:
#line 130 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 19:
#line 134 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 20:
#line 138 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 21:
#line 142 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 22:
#line 146 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 23:
#line 150 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 24:
#line 157 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 25:
#line 161 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 26:
#line 165 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 27:
#line 169 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 28:
#line 176 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 29:
#line 180 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 30:
#line 187 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 31:
#line 191 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 32:
#line 195 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 33:
#line 199 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 34:
#line 203 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
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
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 37:
#line 218 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 38:
#line 222 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
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
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 42:
#line 244 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 43:
#line 248 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 50:
#line 276 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 51:
#line 280 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), NREG, &(yyvsp[(2) - (2)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 54:
#line 298 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 58:
#line 321 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 59:
#line 325 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 60:
#line 329 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 61:
#line 333 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 62:
#line 337 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 63:
#line 341 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 64:
#line 345 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 65:
#line 354 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 66:
#line 358 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 67:
#line 362 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &nullgen, NREG, &(yyvsp[(3) - (4)].addr));
	}
    break;

  case 68:
#line 366 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 69:
#line 370 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 70:
#line 374 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), &nullgen, NREG, &(yyvsp[(4) - (5)].addr));
	}
    break;

  case 71:
#line 378 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 72:
#line 382 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 73:
#line 386 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(5) - (6)].addr));
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
		g.type = D_CONST;
		g.offset = (yyvsp[(2) - (6)].lval);
		outcode((yyvsp[(1) - (6)].lval), &g, (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 78:
#line 410 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(2) - (6)].lval);
		outcode((yyvsp[(1) - (6)].lval), &g, (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 79:
#line 418 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(2) - (8)].lval);
		outcode((yyvsp[(1) - (8)].lval), &g, (yyvsp[(4) - (8)].lval), &(yyvsp[(7) - (8)].addr));
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
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 83:
#line 441 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 84:
#line 448 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 85:
#line 452 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 91:
#line 479 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
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
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 99:
#line 517 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 100:
#line 525 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 101:
#line 529 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 102:
#line 533 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 103:
#line 537 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 104:
#line 541 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 105:
#line 545 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 106:
#line 549 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), NREG, &nullgen);
	}
    break;

  case 107:
#line 556 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 108:
#line 560 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 109:
#line 564 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 110:
#line 568 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 111:
#line 572 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 112:
#line 576 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), NREG, &nullgen);
	}
    break;

  case 113:
#line 583 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 114:
#line 587 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 115:
#line 594 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != D_CONST || (yyvsp[(4) - (4)].addr).type != D_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 116:
#line 603 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != D_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(4) - (4)].addr).type != D_EXTERN && (yyvsp[(4) - (4)].addr).type != D_STATIC && (yyvsp[(4) - (4)].addr).type != D_OREG)
			yyerror("value for FUNCDATA must be symbol reference");
 		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 117:
#line 614 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 118:
#line 621 "a.y"
    {
		settext((yyvsp[(2) - (4)].addr).sym);
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 119:
#line 626 "a.y"
    {
		settext((yyvsp[(2) - (6)].addr).sym);
		(yyvsp[(6) - (6)].addr).offset &= 0xffffffffull;
		(yyvsp[(6) - (6)].addr).offset |= (vlong)ArgsSizeUnknown << 32;
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 120:
#line 633 "a.y"
    {
		settext((yyvsp[(2) - (8)].addr).sym);
		(yyvsp[(6) - (8)].addr).offset &= 0xffffffffull;
		(yyvsp[(6) - (8)].addr).offset |= ((yyvsp[(8) - (8)].lval) & 0xffffffffull) << 32;
		outcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].lval), &(yyvsp[(6) - (8)].addr));
	}
    break;

  case 121:
#line 643 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 122:
#line 647 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 123:
#line 651 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 124:
#line 658 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 125:
#line 664 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 126:
#line 670 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		(yyval.addr) = nullgen;
		if(pass == 2 && (yyvsp[(1) - (2)].sym)->type != LLAB)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 127:
#line 681 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 130:
#line 693 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SPR;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 131:
#line 701 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CREG;
		(yyval.addr).reg = NREG;	/* whole register */
	}
    break;

  case 132:
#line 709 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SPR;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 133:
#line 717 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_MSR;
	}
    break;

  case 134:
#line 724 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SPR;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 135:
#line 730 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 137:
#line 739 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FPSCR;
		(yyval.addr).reg = NREG;
	}
    break;

  case 138:
#line 747 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FPSCR;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 139:
#line 755 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FREG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 140:
#line 761 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 141:
#line 769 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CREG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 142:
#line 775 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 143:
#line 783 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 144:
#line 791 "a.y"
    {
		int mb, me;
		uint32 v;

		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
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
#line 812 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = D_CONST;
	}
    break;

  case 146:
#line 817 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 147:
#line 825 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 148:
#line 831 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 149:
#line 838 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 151:
#line 847 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 152:
#line 855 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 153:
#line 862 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 155:
#line 873 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 156:
#line 882 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = (yyvsp[(3) - (4)].lval);
		(yyval.addr).sym = nil;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 157:
#line 890 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 158:
#line 898 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = D_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 161:
#line 910 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 162:
#line 914 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 163:
#line 918 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 168:
#line 930 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 169:
#line 934 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 170:
#line 938 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 171:
#line 942 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 172:
#line 946 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 174:
#line 953 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 175:
#line 957 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 176:
#line 961 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 177:
#line 965 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 178:
#line 969 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 179:
#line 973 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 180:
#line 977 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 181:
#line 981 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 182:
#line 985 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 183:
#line 989 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 3185 "y.tab.c"
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



