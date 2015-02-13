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
     LGLOBL = 269,
     LTYPEC = 270,
     LTYPED = 271,
     LTYPEE = 272,
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
     LTYPEF = 290,
     LR = 291,
     LREG = 292,
     LF = 293,
     LFREG = 294,
     LC = 295,
     LCREG = 296,
     LPSR = 297,
     LFCR = 298,
     LCOND = 299,
     LS = 300,
     LAT = 301,
     LFCONST = 302,
     LSCONST = 303,
     LNAME = 304,
     LLAB = 305,
     LVAR = 306
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
#define LGLOBL 269
#define LTYPEC 270
#define LTYPED 271
#define LTYPEE 272
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
#define LTYPEF 290
#define LR 291
#define LREG 292
#define LF 293
#define LFREG 294
#define LC 295
#define LCREG 296
#define LPSR 297
#define LFCR 298
#define LCOND 299
#define LS 300
#define LAT 301
#define LFCONST 302
#define LSCONST 303
#define LNAME 304
#define LLAB 305
#define LVAR 306




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
#line 214 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 227 "y.tab.c"

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
#define YYLAST   655

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  72
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  35
/* YYNRULES -- Number of rules.  */
#define YYNRULES  134
/* YYNRULES -- Number of states.  */
#define YYNSTATES  344

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   306

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    68,    12,     5,     2,
      69,    70,    10,     8,    65,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    62,    64,
       6,    63,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    66,     2,    67,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    71,     2,     2,     2,
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
      55,    56,    57,    58,    59,    60,    61
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     5,     9,    10,    15,    20,    25,
      27,    30,    33,    41,    48,    54,    60,    66,    71,    76,
      80,    84,    89,    96,   104,   112,   120,   127,   134,   138,
     144,   152,   157,   164,   171,   176,   180,   186,   192,   200,
     207,   220,   228,   238,   241,   246,   251,   254,   255,   258,
     261,   262,   265,   270,   273,   275,   278,   282,   287,   290,
     293,   296,   298,   301,   305,   307,   311,   315,   317,   319,
     321,   326,   328,   330,   332,   334,   336,   338,   340,   344,
     346,   351,   353,   358,   360,   362,   364,   366,   369,   371,
     377,   382,   387,   392,   397,   399,   401,   403,   405,   410,
     412,   414,   416,   421,   423,   425,   427,   432,   437,   443,
     451,   452,   455,   458,   460,   462,   464,   466,   468,   471,
     474,   477,   481,   482,   485,   487,   491,   495,   499,   503,
     507,   512,   517,   521,   525
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      73,     0,    -1,    -1,    -1,    73,    74,    75,    -1,    -1,
      59,    62,    76,    75,    -1,    59,    63,   106,    64,    -1,
      61,    63,   106,    64,    -1,    64,    -1,    77,    64,    -1,
       1,    64,    -1,    13,    78,    90,    65,    97,    65,    92,
      -1,    13,    78,    90,    65,    97,    65,    -1,    13,    78,
      90,    65,    92,    -1,    14,    78,    90,    65,    92,    -1,
      15,    78,    85,    65,    85,    -1,    16,    78,    79,    80,
      -1,    16,    78,    79,    86,    -1,    36,    79,    87,    -1,
      17,    79,    80,    -1,    18,    78,    79,    85,    -1,    19,
      78,    90,    65,    97,    79,    -1,    20,    78,    88,    65,
      66,    84,    67,    -1,    20,    78,    66,    84,    67,    65,
      88,    -1,    21,    78,    92,    65,    87,    65,    92,    -1,
      21,    78,    92,    65,    87,    79,    -1,    21,    78,    79,
      87,    65,    92,    -1,    22,    78,    79,    -1,    23,   101,
      65,    68,    81,    -1,    23,   101,    65,   104,    65,    68,
      81,    -1,    24,   101,    65,    91,    -1,    24,   101,    65,
     104,    65,    91,    -1,    25,   101,    11,   104,    65,    82,
      -1,    26,    78,    92,    79,    -1,    29,    79,    82,    -1,
      30,    78,   100,    65,   100,    -1,    32,    78,    99,    65,
     100,    -1,    32,    78,    99,    65,    49,    65,   100,    -1,
      33,    78,   100,    65,   100,    79,    -1,    31,    78,   104,
      65,   106,    65,    97,    65,    98,    65,    98,   105,    -1,
      34,    78,    92,    65,    92,    65,    93,    -1,    35,    78,
      92,    65,    92,    65,    92,    65,    97,    -1,    37,    89,
      -1,    44,    85,    65,    85,    -1,    45,    85,    65,    85,
      -1,    27,    79,    -1,    -1,    78,    54,    -1,    78,    55,
      -1,    -1,    65,    79,    -1,   104,    69,    42,    70,    -1,
      59,   102,    -1,    38,    -1,     9,    38,    -1,    38,     9,
      38,    -1,     9,    38,     9,    38,    -1,    68,   104,    -1,
      68,    89,    -1,    68,    58,    -1,    83,    -1,    68,    57,
      -1,    68,     9,    57,    -1,    97,    -1,    97,     9,    97,
      -1,    97,    79,    84,    -1,    92,    -1,    82,    -1,    94,
      -1,    94,    69,    97,    70,    -1,    52,    -1,    53,    -1,
     104,    -1,    89,    -1,   100,    -1,    87,    -1,   101,    -1,
      69,    97,    70,    -1,    87,    -1,   104,    69,    96,    70,
      -1,   101,    -1,   101,    69,    96,    70,    -1,    88,    -1,
      92,    -1,    91,    -1,    94,    -1,    68,   104,    -1,    97,
      -1,    69,    97,    65,    97,    70,    -1,    97,     6,     6,
      95,    -1,    97,     7,     7,    95,    -1,    97,     9,     7,
      95,    -1,    97,    56,     7,    95,    -1,    97,    -1,   104,
      -1,    47,    -1,    42,    -1,    46,    69,   106,    70,    -1,
      96,    -1,    39,    -1,    51,    -1,    50,    69,   106,    70,
      -1,   100,    -1,    83,    -1,    49,    -1,    48,    69,   104,
      70,    -1,   104,    69,   103,    70,    -1,    59,   102,    69,
     103,    70,    -1,    59,     6,     7,   102,    69,    40,    70,
      -1,    -1,     8,   104,    -1,     9,   104,    -1,    40,    -1,
      39,    -1,    41,    -1,    38,    -1,    61,    -1,     9,   104,
      -1,     8,   104,    -1,    71,   104,    -1,    69,   106,    70,
      -1,    -1,    65,   106,    -1,   104,    -1,   106,     8,   106,
      -1,   106,     9,   106,    -1,   106,    10,   106,    -1,   106,
      11,   106,    -1,   106,    12,   106,    -1,   106,     6,     6,
     106,    -1,   106,     7,     7,   106,    -1,   106,     5,   106,
      -1,   106,     4,   106,    -1,   106,     3,   106,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    68,    68,    70,    69,    77,    76,    85,    90,    96,
      97,    98,   104,   108,   112,   119,   126,   133,   137,   144,
     151,   158,   165,   172,   181,   193,   197,   201,   208,   215,
     220,   232,   237,   249,   260,   267,   274,   278,   282,   286,
     293,   315,   323,   332,   339,   348,   359,   365,   368,   372,
     377,   378,   381,   387,   398,   405,   412,   419,   427,   433,
     438,   444,   447,   453,   461,   468,   483,   492,   493,   494,
     495,   500,   506,   512,   518,   519,   522,   523,   531,   540,
     541,   550,   551,   557,   560,   561,   562,   564,   572,   580,
     589,   595,   601,   607,   615,   621,   629,   630,   634,   642,
     643,   649,   650,   658,   659,   662,   668,   676,   684,   692,
     702,   705,   709,   715,   716,   717,   720,   721,   725,   729,
     733,   737,   743,   746,   752,   753,   757,   761,   765,   769,
     773,   777,   781,   785,   789
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
  "LGLOBL", "LTYPEC", "LTYPED", "LTYPEE", "LTYPEG", "LTYPEH", "LTYPEI",
  "LTYPEJ", "LTYPEK", "LTYPEL", "LTYPEM", "LTYPEN", "LTYPEBX", "LTYPEPLD",
  "LCONST", "LSP", "LSB", "LFP", "LPC", "LTYPEX", "LTYPEPC", "LTYPEF",
  "LR", "LREG", "LF", "LFREG", "LC", "LCREG", "LPSR", "LFCR", "LCOND",
  "LS", "LAT", "LFCONST", "LSCONST", "LNAME", "LLAB", "LVAR", "':'", "'='",
  "';'", "','", "'['", "']'", "'$'", "'('", "')'", "'~'", "$accept",
  "prog", "@1", "line", "@2", "inst", "cond", "comma", "rel", "textsize",
  "ximm", "fcon", "reglist", "gen", "nireg", "ireg", "ioreg", "oreg",
  "imsr", "imm", "reg", "regreg", "shift", "rcon", "sreg", "spreg", "creg",
  "frcon", "freg", "name", "offset", "pointer", "con", "oexpr", "expr", 0
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
     305,   306,    58,    61,    59,    44,    91,    93,    36,    40,
      41,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    72,    73,    74,    73,    76,    75,    75,    75,    75,
      75,    75,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    78,    78,    78,
      79,    79,    80,    80,    81,    81,    81,    81,    82,    82,
      82,    82,    83,    83,    84,    84,    84,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    86,    86,    87,    88,
      88,    89,    89,    89,    90,    90,    90,    91,    92,    93,
      94,    94,    94,    94,    95,    95,    96,    96,    96,    97,
      97,    98,    98,    99,    99,   100,   100,   101,   101,   101,
     102,   102,   102,   103,   103,   103,   104,   104,   104,   104,
     104,   104,   105,   105,   106,   106,   106,   106,   106,   106,
     106,   106,   106,   106,   106
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     4,     4,     1,
       2,     2,     7,     6,     5,     5,     5,     4,     4,     3,
       3,     4,     6,     7,     7,     7,     6,     6,     3,     5,
       7,     4,     6,     6,     4,     3,     5,     5,     7,     6,
      12,     7,     9,     2,     4,     4,     2,     0,     2,     2,
       0,     2,     4,     2,     1,     2,     3,     4,     2,     2,
       2,     1,     2,     3,     1,     3,     3,     1,     1,     1,
       4,     1,     1,     1,     1,     1,     1,     1,     3,     1,
       4,     1,     4,     1,     1,     1,     1,     2,     1,     5,
       4,     4,     4,     4,     1,     1,     1,     1,     4,     1,
       1,     1,     4,     1,     1,     1,     4,     4,     5,     7,
       0,     2,     2,     1,     1,     1,     1,     1,     2,     2,
       2,     3,     0,     2,     1,     3,     3,     3,     3,     3,
       4,     4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    47,    47,    47,    47,    50,
      47,    47,    47,    47,    47,     0,     0,     0,    47,    50,
      50,    47,    47,    47,    47,    47,    47,    50,     0,     0,
       0,     0,     0,     9,     4,     0,    11,     0,     0,     0,
      50,    50,     0,    50,     0,     0,    50,    50,     0,     0,
     116,   110,   117,     0,     0,     0,     0,     0,     0,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      79,    83,    43,    81,     0,   100,    97,     0,    96,     0,
     105,    71,    72,     0,    68,    61,     0,    74,    67,    69,
      99,    88,    75,    73,     0,     5,     0,     0,    10,    48,
      49,     0,     0,    85,    84,    86,     0,     0,     0,    51,
     110,    20,     0,     0,     0,     0,     0,     0,     0,     0,
      88,    28,   119,   118,     0,     0,     0,     0,   124,     0,
     120,     0,     0,     0,     0,    50,    35,     0,     0,     0,
     104,     0,   103,     0,     0,     0,     0,    19,     0,     0,
       0,     0,     0,     0,    62,    60,    59,    58,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    87,     0,
       0,     0,   110,    17,    18,    76,    77,     0,    53,     0,
      21,     0,     0,    50,     0,     0,     0,     0,   110,   111,
     112,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   121,     0,     0,   114,   113,   115,     0,    31,
       0,     0,    34,     0,     0,     0,     0,     0,     0,     0,
      78,     0,     0,     0,     0,    63,    44,     0,     0,     0,
       0,     0,    45,     6,     7,     8,    14,    88,    15,    16,
      53,     0,     0,    50,     0,     0,     0,     0,     0,    50,
       0,     0,   134,   133,   132,     0,     0,   125,   126,   127,
     128,   129,     0,    54,    29,     0,   107,     0,     0,    36,
       0,   105,    37,    50,     0,     0,    82,    80,    98,   106,
      70,    90,    94,    95,    91,    92,    93,    13,    52,    22,
       0,    65,    66,     0,    27,    50,    26,     0,   108,   130,
     131,    55,     0,     0,    32,    33,     0,     0,    39,     0,
       0,    12,    24,    23,    25,     0,     0,    56,    30,     0,
      38,     0,    41,     0,   109,    57,     0,     0,     0,     0,
     101,     0,     0,    42,     0,     0,     0,     0,   122,    89,
     102,     0,    40,   123
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    34,   165,    35,    37,   109,   111,   264,
      84,    85,   182,    86,   174,    70,    71,    87,   102,   103,
      88,   322,    89,   281,    90,   120,   331,   141,    92,    73,
     127,   208,   128,   342,   129
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -130
static const yytype_int16 yypact[] =
{
    -130,    12,  -130,   305,   -35,  -130,  -130,  -130,  -130,   -33,
    -130,  -130,  -130,  -130,  -130,   427,   427,   427,  -130,   -33,
     -33,  -130,  -130,  -130,  -130,  -130,  -130,   -33,   445,   370,
     370,    10,   -24,  -130,  -130,   -30,  -130,   140,   140,   335,
     -13,   -33,   449,   -13,   140,    70,   484,   -13,   359,   359,
    -130,   114,  -130,   359,   359,    27,   -11,    62,    76,   167,
    -130,     7,   171,   424,    64,   171,   167,   167,    65,   405,
    -130,  -130,  -130,    68,    74,  -130,  -130,    78,  -130,    86,
    -130,  -130,  -130,   402,  -130,  -130,    96,  -130,  -130,   107,
    -130,    21,  -130,    74,   118,  -130,   359,   359,  -130,  -130,
    -130,   359,   131,  -130,  -130,  -130,   150,   158,   473,  -130,
      98,  -130,   155,   370,   187,    43,   192,   199,    65,   205,
    -130,  -130,  -130,  -130,   262,   359,   359,   203,  -130,    90,
    -130,   106,   164,   233,   359,   -33,  -130,   211,   223,    -3,
    -130,   226,  -130,   231,   235,   240,    43,  -130,   220,   122,
     608,   359,   359,   491,  -130,  -130,  -130,    74,   370,    43,
     287,   291,   300,   301,   370,   305,   560,   582,  -130,    43,
      43,   370,   114,  -130,  -130,  -130,  -130,   264,  -130,   267,
    -130,    43,   279,    -4,   281,   122,   283,    65,    98,  -130,
    -130,   164,   359,   359,   359,   345,   356,   359,   359,   359,
     359,   359,  -130,    15,   306,  -130,  -130,  -130,   282,  -130,
     307,   310,  -130,    32,   359,   308,   132,    32,    43,    43,
    -130,   315,   316,   225,   321,  -130,  -130,   322,   405,   405,
     405,   405,  -130,  -130,  -130,  -130,  -130,   311,  -130,  -130,
     203,   204,   323,   -33,   333,    43,    43,    43,    43,   334,
     331,   337,   631,   611,   547,   359,   359,   228,   228,  -130,
    -130,  -130,   332,   371,  -130,   353,  -130,   357,     7,  -130,
     350,   336,  -130,   -33,   340,   361,  -130,  -130,  -130,  -130,
    -130,  -130,  -130,  -130,  -130,  -130,  -130,    43,  -130,  -130,
     513,  -130,  -130,   360,  -130,   146,  -130,   384,  -130,   303,
     303,   425,   399,    15,  -130,  -130,    43,    32,  -130,   373,
      43,  -130,  -130,  -130,  -130,   375,   408,  -130,  -130,   383,
    -130,    43,  -130,   385,  -130,  -130,   120,   390,    43,   380,
    -130,   391,    43,  -130,   359,   120,   394,   275,   403,  -130,
    -130,   359,  -130,   622
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -130,  -130,  -130,   302,  -130,  -130,   589,    24,   362,   166,
     -58,   411,   -57,    -8,  -130,   -61,   -43,    -7,    22,  -129,
     -21,  -130,    72,    34,   -94,   -29,   137,  -130,   -51,     3,
     -84,   286,    20,  -130,    61
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -65
static const yytype_int16 yytable[] =
{
      91,    91,   116,   136,   209,   245,   215,   147,    91,    91,
      91,   137,     2,   142,   143,    91,   104,   104,    55,    57,
      58,    72,    94,   104,   262,   119,   178,   160,   161,    36,
     162,   107,    41,    42,    98,    56,    56,    56,   135,    97,
     148,    99,   100,    60,    61,   144,   145,   175,    74,    93,
      93,    68,    41,   263,   154,   221,   222,   186,   132,    93,
     106,    41,   112,   -64,   108,   117,   114,   113,   122,   123,
     118,   121,    95,    96,   130,    83,   156,   163,    48,    49,
      79,    80,    75,   138,    91,    76,   183,   134,   240,    77,
      78,   222,   131,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   157,   250,   180,   125,   126,    50,   105,
     105,   176,    79,    80,    48,    49,   105,   148,    99,   100,
     124,   168,   125,   126,    99,   100,   249,   133,   177,    91,
     227,    52,   139,    93,   146,    91,   115,   149,   304,    69,
     237,    54,    91,   150,    50,   189,   190,   151,   236,   238,
     226,   204,   243,   210,   211,   152,   232,   166,   167,   212,
     202,   158,   269,   239,    76,   272,   273,    52,    77,    78,
     329,   330,   224,   123,   203,    53,   159,    54,    93,    75,
      79,   271,    76,   164,    93,    75,    77,    78,    76,   292,
     293,    93,    77,    78,    99,   100,   169,   274,   275,   282,
     282,   282,   282,   205,   206,   207,    75,   246,   101,    76,
     305,    41,   223,    77,    78,   170,   291,   183,   183,    79,
      80,    99,   100,   171,   179,    99,   100,   294,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   199,   200,
     201,    48,    49,   205,   206,   207,   242,   312,   283,   283,
     283,   283,   181,   252,   253,   254,   320,   184,   257,   258,
     259,   260,   261,   284,   285,   286,   311,   289,   185,   188,
     187,    50,   191,   296,   314,   270,   213,   319,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   214,   323,
     220,   216,   327,   228,    52,   278,   217,   308,   229,   333,
     218,   101,    53,   336,    54,   219,     4,   230,   231,   242,
     117,   197,   198,   199,   200,   201,   299,   300,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,   241,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    48,    49,   340,   244,   247,   248,    29,
      30,   255,   266,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   256,    31,   225,    32,    48,    49,    33,
     301,   265,   267,    50,    75,   268,   287,    76,    48,    49,
     302,    77,    78,    79,    80,   276,   277,    81,    82,    99,
     100,   279,   280,   288,    51,   337,    52,    50,   290,   295,
     297,   307,   343,    83,    69,   309,    54,   298,    50,    75,
      48,   153,    76,    48,    49,   306,    77,    78,    79,    80,
      52,   303,    81,    82,   315,   101,   310,   313,    53,    51,
      54,    52,    48,    49,   316,    48,    49,   317,    83,    69,
      50,    54,   321,    50,    75,   324,   325,    76,   326,   334,
     328,    77,    78,    48,    49,   332,   335,    48,    49,   154,
     155,    51,    50,    52,   339,    50,    52,   233,   341,   318,
     173,    69,   338,    54,    53,   140,    54,   251,    99,   100,
       0,    48,    49,    50,     0,    52,    51,    50,    52,     0,
       0,     0,     0,    53,     0,    54,    53,     0,    54,    48,
      49,     0,     0,     0,    51,     0,    52,     0,   110,     0,
      52,    50,     0,     0,    69,     0,    54,     0,    53,     0,
      54,    48,    49,    75,     0,     0,    76,     0,     0,    50,
      77,    78,   172,     0,    52,     0,     0,     0,    99,   100,
       0,     0,    69,     0,    54,     0,     0,     0,   225,    41,
       0,    50,    52,   195,   196,   197,   198,   199,   200,   201,
      53,     0,    54,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,     0,    52,     0,     0,     0,     0,     0,
       0,     0,    69,     0,    54,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,    38,    39,    40,     0,    43,
      44,    45,    46,    47,     0,     0,     0,    59,     0,     0,
      62,    63,    64,    65,    66,    67,   194,   195,   196,   197,
     198,   199,   200,   201,   234,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   193,   194,   195,   196,   197,
     198,   199,   200,   201,     0,     0,   235,   205,   206,   207,
      76,     0,     0,     0,    77,    78
};

static const yytype_int16 yycheck[] =
{
      29,    30,    45,    61,   133,     9,     9,    68,    37,    38,
      39,    62,     0,    64,    65,    44,    37,    38,    15,    16,
      17,    28,    30,    44,     9,    46,   110,     6,     7,    64,
       9,    39,    65,     9,    64,    15,    16,    17,    59,    63,
      69,    54,    55,    19,    20,    66,    67,   108,    28,    29,
      30,    27,    65,    38,    57,   149,   150,   118,    69,    39,
      38,    65,    42,    67,    40,    45,    44,    43,    48,    49,
      46,    47,    62,    63,    54,    68,    83,    56,     8,     9,
      48,    49,    39,    63,   113,    42,   115,    11,   172,    46,
      47,   185,    65,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    83,   188,   113,     8,     9,    38,    37,
      38,   108,    48,    49,     8,     9,    44,   146,    54,    55,
       6,   101,     8,     9,    54,    55,   187,    65,   108,   158,
     159,    61,    68,   113,    69,   164,    66,    69,   267,    69,
     169,    71,   171,    69,    38,   125,   126,    69,   169,   170,
     158,   131,   181,   133,   134,    69,   164,    96,    97,   135,
      70,    65,   213,   171,    42,   216,   217,    61,    46,    47,
      50,    51,   152,   153,    68,    69,    69,    71,   158,    39,
      48,    49,    42,    65,   164,    39,    46,    47,    42,   246,
     247,   171,    46,    47,    54,    55,    65,   218,   219,   228,
     229,   230,   231,    39,    40,    41,    39,   183,    68,    42,
     268,    65,   151,    46,    47,    65,   245,   246,   247,    48,
      49,    54,    55,    65,    69,    54,    55,   248,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    10,    11,
      12,     8,     9,    39,    40,    41,    42,   290,   228,   229,
     230,   231,    65,   192,   193,   194,   307,    65,   197,   198,
     199,   200,   201,   229,   230,   231,   287,   243,    69,     7,
      65,    38,    69,   249,   295,   214,    65,   306,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    65,   310,
      70,    65,   321,     6,    61,    70,    65,   273,     7,   328,
      65,    68,    69,   332,    71,    65,     1,     7,     7,    42,
     290,     8,     9,    10,    11,    12,   255,   256,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    69,    29,    30,    31,    32,    33,    34,
      35,    36,    37,     8,     9,    70,    67,    66,    65,    44,
      45,     6,    70,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,     7,    59,    57,    61,     8,     9,    64,
      38,    65,    65,    38,    39,    65,    65,    42,     8,     9,
       9,    46,    47,    48,    49,    70,    70,    52,    53,    54,
      55,    70,    70,    70,    59,   334,    61,    38,    65,    65,
      69,    65,   341,    68,    69,    65,    71,    70,    38,    39,
       8,     9,    42,     8,     9,    65,    46,    47,    48,    49,
      61,    68,    52,    53,    40,    68,    65,    67,    69,    59,
      71,    61,     8,     9,     9,     8,     9,    38,    68,    69,
      38,    71,    69,    38,    39,    70,    38,    42,    65,    69,
      65,    46,    47,     8,     9,    65,    65,     8,     9,    57,
      58,    59,    38,    61,    70,    38,    61,   165,    65,   303,
     108,    69,   335,    71,    69,    64,    71,   191,    54,    55,
      -1,     8,     9,    38,    -1,    61,    59,    38,    61,    -1,
      -1,    -1,    -1,    69,    -1,    71,    69,    -1,    71,     8,
       9,    -1,    -1,    -1,    59,    -1,    61,    -1,    59,    -1,
      61,    38,    -1,    -1,    69,    -1,    71,    -1,    69,    -1,
      71,     8,     9,    39,    -1,    -1,    42,    -1,    -1,    38,
      46,    47,    59,    -1,    61,    -1,    -1,    -1,    54,    55,
      -1,    -1,    69,    -1,    71,    -1,    -1,    -1,    57,    65,
      -1,    38,    61,     6,     7,     8,     9,    10,    11,    12,
      69,    -1,    71,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    69,    -1,    71,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,     6,     7,     8,    -1,    10,
      11,    12,    13,    14,    -1,    -1,    -1,    18,    -1,    -1,
      21,    22,    23,    24,    25,    26,     5,     6,     7,     8,
       9,    10,    11,    12,    64,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    -1,    -1,    64,    39,    40,    41,
      42,    -1,    -1,    -1,    46,    47
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    73,     0,    74,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    44,
      45,    59,    61,    64,    75,    77,    64,    78,    78,    78,
      78,    65,    79,    78,    78,    78,    78,    78,     8,     9,
      38,    59,    61,    69,    71,   101,   104,   101,   101,    78,
      79,    79,    78,    78,    78,    78,    78,    78,    79,    69,
      87,    88,    89,   101,   104,    39,    42,    46,    47,    48,
      49,    52,    53,    68,    82,    83,    85,    89,    92,    94,
      96,    97,   100,   104,    85,    62,    63,    63,    64,    54,
      55,    68,    90,    91,    92,    94,    90,    85,    79,    79,
      59,    80,   104,    79,    90,    66,    88,   104,    79,    92,
      97,    79,   104,   104,     6,     8,     9,   102,   104,   106,
     104,    65,    69,    65,    11,    92,    82,   100,   104,    68,
      83,    99,   100,   100,    92,    92,    69,    87,    97,    69,
      69,    69,    69,     9,    57,    58,    89,   104,    65,    69,
       6,     7,     9,    56,    65,    76,   106,   106,   104,    65,
      65,    65,    59,    80,    86,    87,   101,   104,   102,    69,
      85,    65,    84,    97,    65,    69,    87,    65,     7,   104,
     104,    69,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    70,    68,   104,    39,    40,    41,   103,    91,
     104,   104,    79,    65,    65,     9,    65,    65,    65,    65,
      70,    96,    96,   106,   104,    57,    85,    97,     6,     7,
       7,     7,    85,    75,    64,    64,    92,    97,    92,    85,
     102,    69,    42,    97,    67,     9,    79,    66,    65,    87,
     102,   103,   106,   106,   106,     6,     7,   106,   106,   106,
     106,   106,     9,    38,    81,    65,    70,    65,    65,   100,
     106,    49,   100,   100,    92,    92,    70,    70,    70,    70,
      70,    95,    97,   104,    95,    95,    95,    65,    70,    79,
      65,    97,    84,    84,    92,    65,    79,    69,    70,   106,
     106,    38,     9,    68,    91,    82,    65,    65,    79,    65,
      65,    92,    88,    67,    92,    40,     9,    38,    81,    97,
     100,    69,    93,    92,    70,    38,    65,    97,    65,    50,
      51,    98,    65,    97,    69,    65,    97,   106,    98,    70,
      70,    65,   105,   106
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
		settext((yyvsp[(2) - (5)].addr).sym);
		outcode((yyvsp[(1) - (5)].lval), Always, &(yyvsp[(2) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 30:
#line 221 "a.y"
    {
		settext((yyvsp[(2) - (7)].addr).sym);
		outcode((yyvsp[(1) - (7)].lval), Always, &(yyvsp[(2) - (7)].addr), 0, &(yyvsp[(7) - (7)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (7)].lval);
		}
	}
    break;

  case 31:
#line 233 "a.y"
    {
		settext((yyvsp[(2) - (4)].addr).sym);
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 32:
#line 238 "a.y"
    {
		settext((yyvsp[(2) - (6)].addr).sym);
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 33:
#line 250 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].addr), 0, &(yyvsp[(6) - (6)].addr));
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 34:
#line 261 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &(yyvsp[(3) - (4)].addr), 0, &nullgen);
	}
    break;

  case 35:
#line 268 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, 0, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 36:
#line 275 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 37:
#line 279 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), 0, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 38:
#line 283 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), (yyvsp[(5) - (7)].lval), &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 39:
#line 287 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].addr), (yyvsp[(5) - (6)].addr).reg, &nullgen);
	}
    break;

  case 40:
#line 294 "a.y"
    {
		Addr g;

		g = nullgen;
		g.type = TYPE_CONST;
		g.offset =
			(0xe << 24) |		/* opcode */
			((yyvsp[(1) - (12)].lval) << 20) |		/* MCR/MRC */
			(((yyvsp[(2) - (12)].lval)^C_SCOND_XOR) << 28) |		/* scond */
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

  case 41:
#line 316 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), (yyvsp[(5) - (7)].addr).reg, &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 42:
#line 324 "a.y"
    {
		(yyvsp[(7) - (9)].addr).type = TYPE_REGREG2;
		(yyvsp[(7) - (9)].addr).offset = (yyvsp[(9) - (9)].lval);
		outcode((yyvsp[(1) - (9)].lval), (yyvsp[(2) - (9)].lval), &(yyvsp[(3) - (9)].addr), (yyvsp[(5) - (9)].addr).reg, &(yyvsp[(7) - (9)].addr));
	}
    break;

  case 43:
#line 333 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &(yyvsp[(2) - (2)].addr), 0, &nullgen);
	}
    break;

  case 44:
#line 340 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != TYPE_CONST || (yyvsp[(4) - (4)].addr).type != TYPE_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 45:
#line 349 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != TYPE_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(4) - (4)].addr).type != NAME_EXTERN && (yyvsp[(4) - (4)].addr).type != NAME_STATIC && (yyvsp[(4) - (4)].addr).type != TYPE_MEM)
			yyerror("value for FUNCDATA must be symbol reference");
 		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 46:
#line 360 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &nullgen, 0, &nullgen);
	}
    break;

  case 47:
#line 365 "a.y"
    {
		(yyval.lval) = Always;
	}
    break;

  case 48:
#line 369 "a.y"
    {
		(yyval.lval) = ((yyvsp[(1) - (2)].lval) & ~C_SCOND) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 49:
#line 373 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (2)].lval) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 52:
#line 382 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 53:
#line 388 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		(yyval.addr) = nullgen;
		if(pass == 2 && (yyvsp[(1) - (2)].sym)->type != LLAB)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 54:
#line 399 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 55:
#line 406 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (2)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 56:
#line 413 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (3)].lval);
		(yyval.addr).u.argsize = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 57:
#line 420 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (4)].lval);
		(yyval.addr).u.argsize = (yyvsp[(4) - (4)].lval);
	}
    break;

  case 58:
#line 428 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 59:
#line 434 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = TYPE_ADDR;
	}
    break;

  case 60:
#line 439 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 62:
#line 448 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 63:
#line 454 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 64:
#line 462 "a.y"
    {
		if((yyvsp[(1) - (1)].lval) < REG_R0 || (yyvsp[(1) - (1)].lval) > REG_R15)
			yyerror("invalid register in reglist");

		(yyval.lval) = 1 << ((yyvsp[(1) - (1)].lval)&15);
	}
    break;

  case 65:
#line 469 "a.y"
    {
		int i;

		if((yyvsp[(1) - (3)].lval) < REG_R0 || (yyvsp[(1) - (3)].lval) > REG_R15)
			yyerror("invalid register in reglist");
		if((yyvsp[(3) - (3)].lval) < REG_R0 || (yyvsp[(3) - (3)].lval) > REG_R15)
			yyerror("invalid register in reglist");

		(yyval.lval)=0;
		for(i=(yyvsp[(1) - (3)].lval); i<=(yyvsp[(3) - (3)].lval); i++)
			(yyval.lval) |= 1<<(i&15);
		for(i=(yyvsp[(3) - (3)].lval); i<=(yyvsp[(1) - (3)].lval); i++)
			(yyval.lval) |= 1<<(i&15);
	}
    break;

  case 66:
#line 484 "a.y"
    {
		if((yyvsp[(1) - (3)].lval) < REG_R0 || (yyvsp[(1) - (3)].lval) > REG_R15)
			yyerror("invalid register in reglist");

		(yyval.lval) = (1<<((yyvsp[(1) - (3)].lval)&15)) | (yyvsp[(3) - (3)].lval);
	}
    break;

  case 70:
#line 496 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (4)].addr);
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 71:
#line 501 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 72:
#line 507 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 73:
#line 513 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 77:
#line 524 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (1)].addr);
		if((yyvsp[(1) - (1)].addr).name != NAME_EXTERN && (yyvsp[(1) - (1)].addr).name != NAME_STATIC) {
		}
	}
    break;

  case 78:
#line 532 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 80:
#line 542 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 82:
#line 552 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (4)].addr);
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 87:
#line 565 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 88:
#line 573 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 89:
#line 581 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REGREG;
		(yyval.addr).reg = (yyvsp[(2) - (5)].lval);
		(yyval.addr).offset = (yyvsp[(4) - (5)].lval);
	}
    break;

  case 90:
#line 590 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval)&15 | (yyvsp[(4) - (4)].lval) | (0 << 5);
	}
    break;

  case 91:
#line 596 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval)&15 | (yyvsp[(4) - (4)].lval) | (1 << 5);
	}
    break;

  case 92:
#line 602 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval)&15 | (yyvsp[(4) - (4)].lval) | (2 << 5);
	}
    break;

  case 93:
#line 608 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval)&15 | (yyvsp[(4) - (4)].lval) | (3 << 5);
	}
    break;

  case 94:
#line 616 "a.y"
    {
		if((yyval.lval) < REG_R0 || (yyval.lval) > REG_R15)
			print("register value out of range in shift\n");
		(yyval.lval) = (((yyvsp[(1) - (1)].lval)&15) << 8) | (1 << 4);
	}
    break;

  case 95:
#line 622 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 32)
			print("shift value out of range\n");
		(yyval.lval) = ((yyvsp[(1) - (1)].lval)&31) << 7;
	}
    break;

  case 97:
#line 631 "a.y"
    {
		(yyval.lval) = REGPC;
	}
    break;

  case 98:
#line 635 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range in R(...)\n");
		(yyval.lval) = REG_R0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 100:
#line 644 "a.y"
    {
		(yyval.lval) = REGSP;
	}
    break;

  case 102:
#line 651 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range in C(...)\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval); // TODO(rsc): REG_C0+$3
	}
    break;

  case 105:
#line 663 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 106:
#line 669 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_F0 + (yyvsp[(3) - (4)].lval);
	}
    break;

  case 107:
#line 677 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(3) - (4)].lval);
		(yyval.addr).sym = nil;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 108:
#line 685 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 109:
#line 693 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = NAME_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 1);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 110:
#line 702 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 111:
#line 706 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 112:
#line 710 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 117:
#line 722 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 118:
#line 726 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 119:
#line 730 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 120:
#line 734 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 121:
#line 738 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 122:
#line 743 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 123:
#line 747 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 125:
#line 754 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:
#line 758 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 127:
#line 762 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:
#line 766 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:
#line 770 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 130:
#line 774 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 131:
#line 778 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 132:
#line 782 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 133:
#line 786 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 134:
#line 790 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2642 "y.tab.c"
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



