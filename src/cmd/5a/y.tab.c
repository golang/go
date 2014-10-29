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
#define YYLAST   640

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  71
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  34
/* YYNRULES -- Number of rules.  */
#define YYNRULES  130
/* YYNRULES -- Number of states.  */
#define YYNSTATES  333

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
     257,   260,   265,   268,   271,   274,   279,   282,   284,   287,
     291,   293,   297,   301,   303,   305,   307,   312,   314,   316,
     318,   320,   322,   324,   326,   330,   332,   337,   339,   344,
     346,   348,   350,   352,   355,   357,   363,   368,   373,   378,
     383,   385,   387,   389,   391,   396,   398,   400,   402,   407,
     409,   411,   413,   418,   423,   429,   437,   438,   441,   444,
     446,   448,   450,   452,   454,   457,   460,   463,   467,   468,
     471,   473,   477,   481,   485,   489,   493,   498,   503,   507,
     511
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
      -1,    69,    87,    -1,    69,    10,    69,    87,    -1,    69,
      57,    -1,    81,    -1,    69,    56,    -1,    69,     9,    56,
      -1,    95,    -1,    95,     9,    95,    -1,    95,    78,    82,
      -1,    90,    -1,    80,    -1,    92,    -1,    92,    67,    95,
      68,    -1,    51,    -1,    52,    -1,   102,    -1,    87,    -1,
      98,    -1,    85,    -1,    99,    -1,    67,    95,    68,    -1,
      85,    -1,   102,    67,    94,    68,    -1,    99,    -1,    99,
      67,    94,    68,    -1,    86,    -1,    90,    -1,    89,    -1,
      92,    -1,    69,   102,    -1,    95,    -1,    67,    95,    64,
      95,    68,    -1,    95,     6,     6,    93,    -1,    95,     7,
       7,    93,    -1,    95,     9,     7,    93,    -1,    95,    55,
       7,    93,    -1,    95,    -1,   102,    -1,    46,    -1,    41,
      -1,    45,    67,   104,    68,    -1,    94,    -1,    38,    -1,
      50,    -1,    49,    67,   104,    68,    -1,    98,    -1,    81,
      -1,    48,    -1,    47,    67,   102,    68,    -1,   102,    67,
     101,    68,    -1,    58,   100,    67,   101,    68,    -1,    58,
       6,     7,   100,    67,    39,    68,    -1,    -1,     8,   102,
      -1,     9,   102,    -1,    39,    -1,    38,    -1,    40,    -1,
      37,    -1,    60,    -1,     9,   102,    -1,     8,   102,    -1,
      70,   102,    -1,    67,   104,    68,    -1,    -1,    64,   104,
      -1,   102,    -1,   104,     8,   104,    -1,   104,     9,   104,
      -1,   104,    10,   104,    -1,   104,    11,   104,    -1,   104,
      12,   104,    -1,   104,     6,     6,   104,    -1,   104,     7,
       7,   104,    -1,   104,     5,   104,    -1,   104,     4,   104,
      -1,   104,     3,   104,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    68,    68,    70,    69,    77,    76,    85,    90,    96,
      97,    98,   104,   108,   112,   119,   126,   133,   137,   144,
     151,   158,   165,   172,   181,   193,   197,   201,   208,   215,
     222,   229,   239,   246,   253,   260,   264,   268,   272,   279,
     301,   309,   318,   325,   334,   345,   351,   354,   358,   363,
     364,   367,   373,   383,   389,   394,   399,   405,   408,   414,
     422,   426,   435,   441,   442,   443,   444,   449,   455,   461,
     467,   468,   471,   472,   480,   489,   490,   499,   500,   506,
     509,   510,   511,   513,   521,   529,   538,   544,   550,   556,
     564,   570,   578,   579,   583,   591,   592,   598,   599,   607,
     608,   611,   617,   625,   633,   641,   651,   654,   658,   664,
     665,   666,   669,   670,   674,   678,   682,   686,   692,   695,
     701,   702,   706,   710,   714,   718,   722,   726,   730,   734,
     738
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
      78,    79,    79,    80,    80,    80,    80,    80,    81,    81,
      82,    82,    82,    83,    83,    83,    83,    83,    83,    83,
      83,    83,    84,    84,    85,    86,    86,    87,    87,    87,
      88,    88,    88,    89,    90,    91,    92,    92,    92,    92,
      93,    93,    94,    94,    94,    95,    95,    96,    96,    97,
      97,    98,    98,    99,    99,    99,   100,   100,   100,   101,
     101,   101,   102,   102,   102,   102,   102,   102,   103,   103,
     104,   104,   104,   104,   104,   104,   104,   104,   104,   104,
     104
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     4,     4,     1,
       2,     2,     7,     6,     5,     5,     5,     4,     4,     3,
       3,     4,     6,     7,     7,     7,     6,     6,     3,     4,
       6,     8,     6,     4,     3,     5,     5,     7,     6,    12,
       7,     9,     2,     4,     4,     2,     0,     2,     2,     0,
       2,     4,     2,     2,     2,     4,     2,     1,     2,     3,
       1,     3,     3,     1,     1,     1,     4,     1,     1,     1,
       1,     1,     1,     1,     3,     1,     4,     1,     4,     1,
       1,     1,     1,     2,     1,     5,     4,     4,     4,     4,
       1,     1,     1,     1,     4,     1,     1,     1,     4,     1,
       1,     1,     4,     4,     5,     7,     0,     2,     2,     1,
       1,     1,     1,     1,     2,     2,     2,     3,     0,     2,
       1,     3,     3,     3,     3,     3,     4,     4,     3,     3,
       3
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
      49,     0,    49,     0,     0,    49,    49,     0,     0,   112,
     106,   113,     0,     0,     0,     0,     0,     0,    45,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    75,    79,
      42,    77,     0,    96,    93,     0,    92,     0,   101,    67,
      68,     0,    64,    57,     0,    70,    63,    65,    95,    84,
      71,    69,     0,     5,     0,     0,    10,    47,    48,     0,
       0,    81,    80,    82,     0,     0,     0,    50,   106,    20,
       0,     0,     0,     0,     0,     0,     0,     0,    84,    28,
     115,   114,     0,     0,     0,     0,   120,     0,   116,     0,
       0,     0,    49,    34,     0,     0,     0,   100,     0,    99,
       0,     0,     0,     0,    19,     0,     0,     0,     0,     0,
       0,     0,    58,    56,    54,    53,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    83,     0,     0,     0,
     106,    17,    18,    72,    73,     0,    52,     0,    21,     0,
       0,    49,     0,     0,     0,     0,   106,   107,   108,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     117,    29,     0,   110,   109,   111,     0,     0,    33,     0,
       0,     0,     0,     0,     0,     0,    74,     0,     0,     0,
       0,    59,     0,    43,     0,     0,     0,     0,     0,    44,
       6,     7,     8,    14,    84,    15,    16,    52,     0,     0,
      49,     0,     0,     0,     0,     0,    49,     0,     0,   130,
     129,   128,     0,     0,   121,   122,   123,   124,   125,     0,
     103,     0,    35,     0,   101,    36,    49,     0,     0,    78,
      76,    94,   102,    55,    66,    86,    90,    91,    87,    88,
      89,    13,    51,    22,     0,    61,    62,     0,    27,    49,
      26,     0,   104,   126,   127,    30,    32,     0,     0,    38,
       0,     0,    12,    24,    23,    25,     0,     0,     0,    37,
       0,    40,     0,   105,    31,     0,     0,     0,     0,    97,
       0,     0,    41,     0,     0,     0,     0,   118,    85,    98,
       0,    39,   119
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    33,   163,    34,    36,   107,   109,    82,
      83,   180,    84,   172,    68,    69,    85,   100,   101,    86,
     311,    87,   275,    88,   118,   320,   138,    90,    71,   125,
     206,   126,   331,   127
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -125
static const yytype_int16 yypact[] =
{
    -125,     7,  -125,   308,   -41,  -125,  -125,  -125,  -125,   -19,
    -125,  -125,  -125,  -125,  -125,    80,    80,  -125,   -19,   -19,
    -125,  -125,  -125,  -125,  -125,  -125,   -19,   405,   364,   364,
     -31,   -15,  -125,  -125,    -2,  -125,   528,   528,   337,   -18,
     -19,   409,   -18,   528,   230,   187,   -18,   448,   448,  -125,
     257,  -125,   448,   448,    -6,    15,    94,   309,  -125,    49,
      19,    44,    95,    19,   309,   309,    63,   391,  -125,  -125,
    -125,    90,   137,  -125,  -125,   145,  -125,   146,  -125,  -125,
    -125,    66,  -125,  -125,    52,  -125,  -125,   150,  -125,   147,
    -125,   137,    57,  -125,   448,   448,  -125,  -125,  -125,   448,
     167,  -125,  -125,  -125,   184,   200,   431,  -125,    47,  -125,
     201,   364,   217,   189,   223,   221,    63,   228,  -125,  -125,
    -125,  -125,   289,   448,   448,   231,  -125,   181,  -125,   411,
      54,   448,   -19,  -125,   237,   238,    12,  -125,   240,  -125,
     241,   244,   246,   189,  -125,   245,   114,   319,   448,   448,
     417,   243,  -125,  -125,  -125,   137,   364,   189,   293,   312,
     313,   341,   364,   308,   542,   552,  -125,   189,   189,   364,
     257,  -125,  -125,  -125,  -125,   282,  -125,   315,  -125,   189,
     287,    42,   296,   114,   303,    63,    47,  -125,  -125,    54,
     448,   448,   448,   363,   369,   448,   448,   448,   448,   448,
    -125,  -125,   306,  -125,  -125,  -125,   311,   316,  -125,    53,
     448,   321,    65,    53,   189,   189,  -125,   318,   324,   250,
     325,  -125,   405,  -125,   326,   391,   391,   391,   391,  -125,
    -125,  -125,  -125,  -125,   317,  -125,  -125,   231,   130,   328,
     -19,   323,   189,   189,   189,   189,   334,   336,   340,   602,
     621,   628,   448,   448,   197,   197,  -125,  -125,  -125,   352,
    -125,    49,  -125,   516,   359,  -125,   -19,   366,   371,  -125,
    -125,  -125,  -125,  -125,  -125,  -125,  -125,  -125,  -125,  -125,
    -125,   189,  -125,  -125,   474,  -125,  -125,   361,  -125,   165,
    -125,   399,  -125,   235,   235,   432,  -125,   189,    53,  -125,
     376,   189,  -125,  -125,  -125,  -125,   377,   448,   380,  -125,
     189,  -125,   383,  -125,  -125,   112,   385,   189,   386,  -125,
     388,   189,  -125,   448,   112,   382,   267,   395,  -125,  -125,
     448,  -125,   613
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -125,  -125,  -125,   292,  -125,  -125,   578,    45,   354,   -56,
     400,   -48,   -25,  -125,    -7,   -42,   -21,    -5,  -124,     5,
    -125,   -10,    89,  -118,   -28,   140,  -125,   -46,     4,   -90,
     277,    -4,  -125,   -16
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -61
static const yytype_int16 yytable[] =
{
      89,    89,   114,   133,    92,   201,    70,     2,    89,    89,
      89,    55,    55,   105,   134,    89,   139,   140,   176,    54,
      56,   211,    35,    72,    91,    91,   103,   103,   217,   218,
      93,    94,   104,   103,    91,    97,    98,   110,   112,   145,
     115,   102,   102,   120,   121,    40,    40,    95,   102,   128,
     117,   242,    47,    48,    41,   123,   124,   135,   129,   144,
     154,    96,   132,    58,    59,   218,    77,    78,   152,   141,
     142,    66,    97,    98,    47,   150,   151,   155,   164,   165,
     237,    49,   130,    89,   106,   181,   178,   111,    47,    48,
     116,   119,   203,   204,   205,   166,   247,    97,    98,   173,
      77,    78,   175,    49,    51,   131,    40,    91,   -60,   184,
     174,    52,    77,   264,    53,   145,   156,    49,    81,   187,
     188,   162,   152,   153,    50,   202,    51,   207,    89,   224,
     143,   223,   219,    67,    89,   295,    53,   229,    50,   234,
      51,    89,    77,    78,   236,   220,   121,    52,    97,    98,
      53,   240,    91,   158,   159,    74,   160,   146,    91,    75,
      76,   318,   319,   262,   136,    91,   265,   266,   203,   204,
     205,   239,   233,   235,   249,   250,   251,   208,   246,   254,
     255,   256,   257,   258,   190,   191,   192,   193,   194,   195,
     196,   197,   198,   199,   263,   286,   287,   276,   276,   276,
     276,   273,   161,    73,   147,   296,    74,   197,   198,   199,
      75,    76,   148,   149,   285,   181,   181,   157,    72,   267,
     268,   277,   277,   277,   277,    73,   243,    73,    74,    40,
      74,   167,    75,    76,    75,    76,   293,   294,    47,    48,
      97,    98,   303,   195,   196,   197,   198,   199,   168,   200,
     288,    40,   309,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   122,   169,   123,   124,    49,   177,   308,
     190,   191,   192,   193,   194,   195,   196,   197,   198,   199,
     115,   179,   316,    97,    98,   283,   302,   182,   183,   322,
      51,   290,   185,   325,   305,   113,   186,    67,   189,   225,
      53,   209,   210,   314,   212,   213,   312,   326,   214,     4,
     215,   299,   222,   216,   332,   278,   279,   280,   271,   226,
     227,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,   329,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    47,    48,    73,   228,   238,
      74,    28,    29,   241,    75,    76,   239,   203,   204,   205,
      74,   244,    97,    98,    75,    76,    30,   245,    31,   252,
     259,    32,    47,    48,    49,    73,   253,   221,    74,   260,
     261,   281,    75,    76,    77,    78,   269,   284,    79,    80,
      97,    98,   270,   272,   274,    50,   282,    51,   289,    47,
      48,    49,    73,   291,    67,    74,    81,    53,   292,    75,
      76,    77,    78,    47,    48,    79,    80,    47,    48,    47,
      48,    99,    50,   298,    51,    47,    48,   304,    49,    73,
     300,    67,    74,    81,    53,   301,    75,    76,   306,    47,
      48,   307,    49,   310,   315,   313,    49,   317,    49,   321,
     328,    51,   324,   323,    49,   230,    47,    48,    52,   330,
     171,    53,   137,    50,   327,    51,   248,   108,    49,    51,
       0,    51,    67,   221,     0,    53,    52,    51,    52,    53,
      99,    53,    47,    48,    52,    49,     0,    53,     0,   170,
       0,    51,     0,     0,     0,     0,     0,     0,    67,     0,
       0,    53,     0,     0,     0,     0,     0,     0,    51,     0,
       0,    49,     0,     0,     0,    52,     0,     0,    53,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,     0,
       0,     0,     0,     0,    51,     0,     0,     0,     0,     0,
       0,    67,     0,     0,    53,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,     0,    73,     0,     0,    74,
       0,     0,     0,    75,    76,     0,     0,     0,     0,     0,
     297,    97,    98,     0,    37,    38,    39,     0,    42,    43,
      44,    45,    46,     0,     0,    57,     0,    99,    60,    61,
      62,    63,    64,    65,     0,   231,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   232,   190,   191,   192,   193,
     194,   195,   196,   197,   198,   199,   192,   193,   194,   195,
     196,   197,   198,   199,   193,   194,   195,   196,   197,   198,
     199
};

static const yytype_int16 yycheck[] =
{
      28,    29,    44,    59,    29,   129,    27,     0,    36,    37,
      38,    15,    16,    38,    60,    43,    62,    63,   108,    15,
      16,     9,    63,    27,    28,    29,    36,    37,   146,   147,
      61,    62,    37,    43,    38,    53,    54,    41,    43,    67,
      44,    36,    37,    47,    48,    64,    64,    62,    43,    53,
      45,     9,     8,     9,     9,     8,     9,    61,    64,    66,
      81,    63,    57,    18,    19,   183,    47,    48,    56,    64,
      65,    26,    53,    54,     8,     9,    10,    81,    94,    95,
     170,    37,    67,   111,    39,   113,   111,    42,     8,     9,
      45,    46,    38,    39,    40,    99,   186,    53,    54,   106,
      47,    48,   106,    37,    60,    11,    64,   111,    66,   116,
     106,    67,    47,    48,    70,   143,    64,    37,    69,   123,
     124,    64,    56,    57,    58,   129,    60,   131,   156,   157,
      67,   156,   148,    67,   162,   259,    70,   162,    58,   167,
      60,   169,    47,    48,   169,   149,   150,    67,    53,    54,
      70,   179,   156,     6,     7,    41,     9,    67,   162,    45,
      46,    49,    50,   209,    69,   169,   212,   213,    38,    39,
      40,    41,   167,   168,   190,   191,   192,   132,   185,   195,
     196,   197,   198,   199,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,   210,   243,   244,   225,   226,   227,
     228,   222,    55,    38,    67,   261,    41,    10,    11,    12,
      45,    46,    67,    67,   242,   243,   244,    67,   222,   214,
     215,   225,   226,   227,   228,    38,   181,    38,    41,    64,
      41,    64,    45,    46,    45,    46,   252,   253,     8,     9,
      53,    54,   284,     8,     9,    10,    11,    12,    64,    68,
     245,    64,   298,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,     6,    64,     8,     9,    37,    67,   297,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
     284,    64,   310,    53,    54,   240,   281,    64,    67,   317,
      60,   246,    64,   321,   289,    65,     7,    67,    67,     6,
      70,    64,    64,   307,    64,    64,   301,   323,    64,     1,
      64,   266,    69,    68,   330,   226,   227,   228,    68,     7,
       7,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    68,    28,    29,    30,    31,
      32,    33,    34,    35,    36,     8,     9,    38,     7,    67,
      41,    43,    44,    66,    45,    46,    41,    38,    39,    40,
      41,    65,    53,    54,    45,    46,    58,    64,    60,     6,
      64,    63,     8,     9,    37,    38,     7,    56,    41,    68,
      64,    64,    45,    46,    47,    48,    68,    64,    51,    52,
      53,    54,    68,    68,    68,    58,    68,    60,    64,     8,
       9,    37,    38,    67,    67,    41,    69,    70,    68,    45,
      46,    47,    48,     8,     9,    51,    52,     8,     9,     8,
       9,    69,    58,    64,    60,     8,     9,    66,    37,    38,
      64,    67,    41,    69,    70,    64,    45,    46,    39,     8,
       9,     9,    37,    67,    64,    68,    37,    64,    37,    64,
      68,    60,    64,    67,    37,   163,     8,     9,    67,    64,
     106,    70,    62,    58,   324,    60,   189,    58,    37,    60,
      -1,    60,    67,    56,    -1,    70,    67,    60,    67,    70,
      69,    70,     8,     9,    67,    37,    -1,    70,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    67,    -1,
      -1,    70,    -1,    -1,    -1,    -1,    -1,    -1,    60,    -1,
      -1,    37,    -1,    -1,    -1,    67,    -1,    -1,    70,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    -1,
      -1,    -1,    -1,    -1,    60,    -1,    -1,    -1,    -1,    -1,
      -1,    67,    -1,    -1,    70,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    -1,    38,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    -1,    -1,    -1,    -1,    -1,
      64,    53,    54,    -1,     6,     7,     8,    -1,    10,    11,
      12,    13,    14,    -1,    -1,    17,    -1,    69,    20,    21,
      22,    23,    24,    25,    -1,    63,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    63,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,     5,     6,     7,     8,
       9,    10,    11,    12,     6,     7,     8,     9,    10,    11,
      12
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
       9,    10,    56,    57,    87,   102,    64,    67,     6,     7,
       9,    55,    64,    75,   104,   104,   102,    64,    64,    64,
      58,    79,    84,    85,    99,   102,   100,    67,    83,    64,
      82,    95,    64,    67,    85,    64,     7,   102,   102,    67,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      68,    89,   102,    38,    39,    40,   101,   102,    78,    64,
      64,     9,    64,    64,    64,    64,    68,    94,    94,   104,
     102,    56,    69,    83,    95,     6,     7,     7,     7,    83,
      74,    63,    63,    90,    95,    90,    83,   100,    67,    41,
      95,    66,     9,    78,    65,    64,    85,   100,   101,   104,
     104,   104,     6,     7,   104,   104,   104,   104,   104,    64,
      68,    64,    98,   104,    48,    98,    98,    90,    90,    68,
      68,    68,    68,    87,    68,    93,    95,   102,    93,    93,
      93,    64,    68,    78,    64,    95,    82,    82,    90,    64,
      78,    67,    68,   104,   104,    89,    80,    64,    64,    78,
      64,    64,    90,    86,    66,    90,    39,     9,    95,    98,
      67,    91,    90,    68,   102,    64,    95,    64,    49,    50,
      96,    64,    95,    67,    64,    95,   104,    96,    68,    68,
      64,   103,   104
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
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 15:
#line 120 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 16:
#line 127 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 17:
#line 134 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 18:
#line 138 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 19:
#line 145 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 20:
#line 152 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 21:
#line 159 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].addr));
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
		g.type = D_CONST;
		g.offset = (yyvsp[(6) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), NREG, &g);
	}
    break;

  case 24:
#line 182 "a.y"
    {
		Addr g;

		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(4) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &g, NREG, &(yyvsp[(7) - (7)].addr));
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
		outcode((yyvsp[(1) - (3)].lval), (yyvsp[(2) - (3)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 29:
#line 216 "a.y"
    {
		settext((yyvsp[(2) - (4)].addr).sym);
		(yyvsp[(4) - (4)].addr).type = D_CONST2;
		(yyvsp[(4) - (4)].addr).offset2 = ArgsSizeUnknown;
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 30:
#line 223 "a.y"
    {
		settext((yyvsp[(2) - (6)].addr).sym);
		(yyvsp[(6) - (6)].addr).type = D_CONST2;
		(yyvsp[(6) - (6)].addr).offset2 = ArgsSizeUnknown;
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 31:
#line 230 "a.y"
    {
		settext((yyvsp[(2) - (8)].addr).sym);
		(yyvsp[(6) - (8)].addr).type = D_CONST2;
		(yyvsp[(6) - (8)].addr).offset2 = (yyvsp[(8) - (8)].lval);
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
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &(yyvsp[(3) - (4)].addr), NREG, &nullgen);
	}
    break;

  case 34:
#line 254 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 35:
#line 261 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 36:
#line 265 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
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
		outcode(AMRC, Always, &nullgen, NREG, &g);
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
		(yyvsp[(7) - (9)].addr).type = D_REGREG2;
		(yyvsp[(7) - (9)].addr).offset = (yyvsp[(9) - (9)].lval);
		outcode((yyvsp[(1) - (9)].lval), (yyvsp[(2) - (9)].lval), &(yyvsp[(3) - (9)].addr), (yyvsp[(5) - (9)].addr).reg, &(yyvsp[(7) - (9)].addr));
	}
    break;

  case 42:
#line 319 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &(yyvsp[(2) - (2)].addr), NREG, &nullgen);
	}
    break;

  case 43:
#line 326 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != D_CONST || (yyvsp[(4) - (4)].addr).type != D_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 44:
#line 335 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != D_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(4) - (4)].addr).type != D_EXTERN && (yyvsp[(4) - (4)].addr).type != D_STATIC && (yyvsp[(4) - (4)].addr).type != D_OREG)
			yyerror("value for FUNCDATA must be symbol reference");
 		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 45:
#line 346 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &nullgen, NREG, &nullgen);
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
		(yyval.addr).type = D_BRANCH;
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
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 53:
#line 384 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 54:
#line 390 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = D_CONST;
	}
    break;

  case 55:
#line 395 "a.y"
    {
		(yyval.addr) = (yyvsp[(4) - (4)].addr);
		(yyval.addr).type = D_OCONST;
	}
    break;

  case 56:
#line 400 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 58:
#line 409 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 59:
#line 415 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 60:
#line 423 "a.y"
    {
		(yyval.lval) = 1 << (yyvsp[(1) - (1)].lval);
	}
    break;

  case 61:
#line 427 "a.y"
    {
		int i;
		(yyval.lval)=0;
		for(i=(yyvsp[(1) - (3)].lval); i<=(yyvsp[(3) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
		for(i=(yyvsp[(3) - (3)].lval); i<=(yyvsp[(1) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
	}
    break;

  case 62:
#line 436 "a.y"
    {
		(yyval.lval) = (1<<(yyvsp[(1) - (3)].lval)) | (yyvsp[(3) - (3)].lval);
	}
    break;

  case 66:
#line 445 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (4)].addr);
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 67:
#line 450 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_PSR;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 68:
#line 456 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FPCR;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 69:
#line 462 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 73:
#line 473 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (1)].addr);
		if((yyvsp[(1) - (1)].addr).name != D_EXTERN && (yyvsp[(1) - (1)].addr).name != D_STATIC) {
		}
	}
    break;

  case 74:
#line 481 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 76:
#line 491 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 78:
#line 501 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (4)].addr);
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 83:
#line 514 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 84:
#line 522 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 85:
#line 530 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_REGREG;
		(yyval.addr).reg = (yyvsp[(2) - (5)].lval);
		(yyval.addr).offset = (yyvsp[(4) - (5)].lval);
	}
    break;

  case 86:
#line 539 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (0 << 5);
	}
    break;

  case 87:
#line 545 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (1 << 5);
	}
    break;

  case 88:
#line 551 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (2 << 5);
	}
    break;

  case 89:
#line 557 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (3 << 5);
	}
    break;

  case 90:
#line 565 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 16)
			print("register value out of range\n");
		(yyval.lval) = (((yyvsp[(1) - (1)].lval)&15) << 8) | (1 << 4);
	}
    break;

  case 91:
#line 571 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 32)
			print("shift value out of range\n");
		(yyval.lval) = ((yyvsp[(1) - (1)].lval)&31) << 7;
	}
    break;

  case 93:
#line 580 "a.y"
    {
		(yyval.lval) = REGPC;
	}
    break;

  case 94:
#line 584 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 96:
#line 593 "a.y"
    {
		(yyval.lval) = REGSP;
	}
    break;

  case 98:
#line 600 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 101:
#line 612 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FREG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 102:
#line 618 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 103:
#line 626 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = (yyvsp[(3) - (4)].lval);
		(yyval.addr).sym = nil;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 104:
#line 634 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 105:
#line 642 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = D_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 1);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 106:
#line 651 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 107:
#line 655 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 108:
#line 659 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 113:
#line 671 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 114:
#line 675 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 115:
#line 679 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 116:
#line 683 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 117:
#line 687 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 118:
#line 692 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 119:
#line 696 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 121:
#line 703 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 122:
#line 707 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 123:
#line 711 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 124:
#line 715 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:
#line 719 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:
#line 723 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 127:
#line 727 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 128:
#line 731 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:
#line 735 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 130:
#line 739 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2576 "y.tab.c"
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



