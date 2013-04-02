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
     LLITERAL = 258,
     LASOP = 259,
     LCOLAS = 260,
     LBREAK = 261,
     LCASE = 262,
     LCHAN = 263,
     LCONST = 264,
     LCONTINUE = 265,
     LDDD = 266,
     LDEFAULT = 267,
     LDEFER = 268,
     LELSE = 269,
     LFALL = 270,
     LFOR = 271,
     LFUNC = 272,
     LGO = 273,
     LGOTO = 274,
     LIF = 275,
     LIMPORT = 276,
     LINTERFACE = 277,
     LMAP = 278,
     LNAME = 279,
     LPACKAGE = 280,
     LRANGE = 281,
     LRETURN = 282,
     LSELECT = 283,
     LSTRUCT = 284,
     LSWITCH = 285,
     LTYPE = 286,
     LVAR = 287,
     LANDAND = 288,
     LANDNOT = 289,
     LBODY = 290,
     LCOMM = 291,
     LDEC = 292,
     LEQ = 293,
     LGE = 294,
     LGT = 295,
     LIGNORE = 296,
     LINC = 297,
     LLE = 298,
     LLSH = 299,
     LLT = 300,
     LNE = 301,
     LOROR = 302,
     LRSH = 303,
     NotPackage = 304,
     NotParen = 305,
     PreferToRightParen = 306
   };
#endif
/* Tokens.  */
#define LLITERAL 258
#define LASOP 259
#define LCOLAS 260
#define LBREAK 261
#define LCASE 262
#define LCHAN 263
#define LCONST 264
#define LCONTINUE 265
#define LDDD 266
#define LDEFAULT 267
#define LDEFER 268
#define LELSE 269
#define LFALL 270
#define LFOR 271
#define LFUNC 272
#define LGO 273
#define LGOTO 274
#define LIF 275
#define LIMPORT 276
#define LINTERFACE 277
#define LMAP 278
#define LNAME 279
#define LPACKAGE 280
#define LRANGE 281
#define LRETURN 282
#define LSELECT 283
#define LSTRUCT 284
#define LSWITCH 285
#define LTYPE 286
#define LVAR 287
#define LANDAND 288
#define LANDNOT 289
#define LBODY 290
#define LCOMM 291
#define LDEC 292
#define LEQ 293
#define LGE 294
#define LGT 295
#define LIGNORE 296
#define LINC 297
#define LLE 298
#define LLSH 299
#define LLT 300
#define LNE 301
#define LOROR 302
#define LRSH 303
#define NotPackage 304
#define NotParen 305
#define PreferToRightParen 306




/* Copy the first part of user declarations.  */
#line 20 "go.y"

#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and go.h re-#defines getc */
#include <libc.h>
#include "go.h"

static void fixlbrace(int);


/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 1
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 28 "go.y"
{
	Node*		node;
	NodeList*		list;
	Type*		type;
	Sym*		sym;
	struct	Val	val;
	int		i;
}
/* Line 193 of yacc.c.  */
#line 216 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 229 "y.tab.c"

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
#define YYFINAL  4
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   2194

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  76
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  142
/* YYNRULES -- Number of rules.  */
#define YYNRULES  349
/* YYNRULES -- Number of states.  */
#define YYNSTATES  663

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
       2,     2,     2,    69,     2,     2,    64,    55,    56,     2,
      59,    60,    53,    49,    75,    50,    63,    54,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    66,    62,
       2,    65,     2,    73,    74,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    71,     2,    72,    52,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    67,    51,    68,    70,     2,     2,     2,
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
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    57,    58,    61
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     8,     9,    13,    14,    18,    19,    23,
      26,    32,    36,    40,    43,    45,    49,    51,    54,    57,
      62,    63,    65,    66,    71,    72,    74,    76,    78,    80,
      83,    89,    93,    96,   102,   110,   114,   117,   123,   127,
     129,   132,   137,   141,   146,   150,   152,   155,   157,   159,
     162,   164,   168,   172,   176,   179,   182,   186,   192,   198,
     201,   202,   207,   208,   212,   213,   216,   217,   222,   227,
     232,   238,   240,   242,   245,   246,   250,   252,   256,   257,
     258,   259,   268,   269,   275,   276,   279,   280,   283,   284,
     285,   293,   294,   300,   302,   306,   310,   314,   318,   322,
     326,   330,   334,   338,   342,   346,   350,   354,   358,   362,
     366,   370,   374,   378,   382,   384,   387,   390,   393,   396,
     399,   402,   405,   408,   412,   418,   425,   427,   429,   433,
     439,   445,   450,   457,   459,   465,   471,   477,   485,   487,
     488,   492,   494,   499,   501,   506,   508,   512,   514,   516,
     518,   520,   522,   524,   526,   527,   529,   531,   533,   535,
     540,   542,   544,   546,   549,   551,   553,   555,   557,   559,
     563,   565,   567,   569,   572,   574,   576,   578,   580,   584,
     586,   588,   590,   592,   594,   596,   598,   600,   602,   606,
     611,   616,   619,   623,   629,   631,   633,   636,   640,   646,
     650,   656,   660,   664,   670,   679,   685,   694,   700,   701,
     705,   706,   708,   712,   714,   719,   722,   723,   727,   729,
     733,   735,   739,   741,   745,   747,   751,   753,   757,   761,
     764,   769,   773,   779,   785,   787,   791,   793,   796,   798,
     802,   807,   809,   812,   815,   817,   819,   823,   824,   827,
     828,   830,   832,   834,   836,   838,   840,   842,   844,   846,
     847,   852,   854,   857,   860,   863,   866,   869,   872,   874,
     878,   880,   884,   886,   890,   892,   896,   898,   902,   904,
     906,   910,   914,   915,   918,   919,   921,   922,   924,   925,
     927,   928,   930,   931,   933,   934,   936,   937,   939,   940,
     942,   943,   945,   950,   955,   961,   968,   973,   978,   980,
     982,   984,   986,   988,   990,   992,   994,   996,  1000,  1005,
    1011,  1016,  1021,  1024,  1027,  1032,  1036,  1040,  1046,  1050,
    1055,  1059,  1065,  1067,  1068,  1070,  1074,  1076,  1078,  1081,
    1083,  1085,  1091,  1092,  1095,  1097,  1101,  1103,  1107,  1109
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      77,     0,    -1,    79,    78,    81,   166,    -1,    -1,    25,
     141,    62,    -1,    -1,    80,    86,    88,    -1,    -1,    81,
      82,    62,    -1,    21,    83,    -1,    21,    59,    84,   190,
      60,    -1,    21,    59,    60,    -1,    85,    86,    88,    -1,
      85,    88,    -1,    83,    -1,    84,    62,    83,    -1,     3,
      -1,   141,     3,    -1,    63,     3,    -1,    25,    24,    87,
      62,    -1,    -1,    24,    -1,    -1,    89,   214,    64,    64,
      -1,    -1,    91,    -1,   158,    -1,   181,    -1,     1,    -1,
      32,    93,    -1,    32,    59,   167,   190,    60,    -1,    32,
      59,    60,    -1,    92,    94,    -1,    92,    59,    94,   190,
      60,    -1,    92,    59,    94,    62,   168,   190,    60,    -1,
      92,    59,    60,    -1,    31,    97,    -1,    31,    59,   169,
     190,    60,    -1,    31,    59,    60,    -1,     9,    -1,   185,
     146,    -1,   185,   146,    65,   186,    -1,   185,    65,   186,
      -1,   185,   146,    65,   186,    -1,   185,    65,   186,    -1,
      94,    -1,   185,   146,    -1,   185,    -1,   141,    -1,    96,
     146,    -1,   126,    -1,   126,     4,   126,    -1,   186,    65,
     186,    -1,   186,     5,   186,    -1,   126,    42,    -1,   126,
      37,    -1,     7,   187,    66,    -1,     7,   187,    65,   126,
      66,    -1,     7,   187,     5,   126,    66,    -1,    12,    66,
      -1,    -1,    67,   101,   183,    68,    -1,    -1,    99,   103,
     183,    -1,    -1,   104,   102,    -1,    -1,    35,   106,   183,
      68,    -1,   186,    65,    26,   126,    -1,   186,     5,    26,
     126,    -1,   194,    62,   194,    62,   194,    -1,   194,    -1,
     107,    -1,   108,   105,    -1,    -1,    16,   111,   109,    -1,
     194,    -1,   194,    62,   194,    -1,    -1,    -1,    -1,    20,
     114,   112,   115,   105,   116,   119,   120,    -1,    -1,    14,
      20,   118,   112,   105,    -1,    -1,   119,   117,    -1,    -1,
      14,   100,    -1,    -1,    -1,    30,   122,   112,   123,    35,
     104,    68,    -1,    -1,    28,   125,    35,   104,    68,    -1,
     127,    -1,   126,    47,   126,    -1,   126,    33,   126,    -1,
     126,    38,   126,    -1,   126,    46,   126,    -1,   126,    45,
     126,    -1,   126,    43,   126,    -1,   126,    39,   126,    -1,
     126,    40,   126,    -1,   126,    49,   126,    -1,   126,    50,
     126,    -1,   126,    51,   126,    -1,   126,    52,   126,    -1,
     126,    53,   126,    -1,   126,    54,   126,    -1,   126,    55,
     126,    -1,   126,    56,   126,    -1,   126,    34,   126,    -1,
     126,    44,   126,    -1,   126,    48,   126,    -1,   126,    36,
     126,    -1,   134,    -1,    53,   127,    -1,    56,   127,    -1,
      49,   127,    -1,    50,   127,    -1,    69,   127,    -1,    70,
     127,    -1,    52,   127,    -1,    36,   127,    -1,   134,    59,
      60,    -1,   134,    59,   187,   191,    60,    -1,   134,    59,
     187,    11,   191,    60,    -1,     3,    -1,   143,    -1,   134,
      63,   141,    -1,   134,    63,    59,   135,    60,    -1,   134,
      63,    59,    31,    60,    -1,   134,    71,   126,    72,    -1,
     134,    71,   192,    66,   192,    72,    -1,   128,    -1,   149,
      59,   126,   191,    60,    -1,   150,   137,   130,   189,    68,
      -1,   129,    67,   130,   189,    68,    -1,    59,   135,    60,
      67,   130,   189,    68,    -1,   165,    -1,    -1,   126,    66,
     133,    -1,   126,    -1,    67,   130,   189,    68,    -1,   126,
      -1,    67,   130,   189,    68,    -1,   129,    -1,    59,   135,
      60,    -1,   126,    -1,   147,    -1,   146,    -1,    35,    -1,
      67,    -1,   141,    -1,   141,    -1,    -1,   138,    -1,    24,
      -1,   142,    -1,    73,    -1,    74,     3,    63,    24,    -1,
     141,    -1,   138,    -1,    11,    -1,    11,   146,    -1,   155,
      -1,   161,    -1,   153,    -1,   154,    -1,   152,    -1,    59,
     146,    60,    -1,   155,    -1,   161,    -1,   153,    -1,    53,
     147,    -1,   161,    -1,   153,    -1,   154,    -1,   152,    -1,
      59,   146,    60,    -1,   161,    -1,   153,    -1,   153,    -1,
     155,    -1,   161,    -1,   153,    -1,   154,    -1,   152,    -1,
     143,    -1,   143,    63,   141,    -1,    71,   192,    72,   146,
      -1,    71,    11,    72,   146,    -1,     8,   148,    -1,     8,
      36,   146,    -1,    23,    71,   146,    72,   146,    -1,   156,
      -1,   157,    -1,    53,   146,    -1,    36,     8,   146,    -1,
      29,   137,   170,   190,    68,    -1,    29,   137,    68,    -1,
      22,   137,   171,   190,    68,    -1,    22,   137,    68,    -1,
      17,   159,   162,    -1,   141,    59,   179,    60,   163,    -1,
      59,   179,    60,   141,    59,   179,    60,   163,    -1,   200,
      59,   195,    60,   210,    -1,    59,   215,    60,   141,    59,
     195,    60,   210,    -1,    17,    59,   179,    60,   163,    -1,
      -1,    67,   183,    68,    -1,    -1,   151,    -1,    59,   179,
      60,    -1,   161,    -1,   164,   137,   183,    68,    -1,   164,
       1,    -1,    -1,   166,    90,    62,    -1,    93,    -1,   167,
      62,    93,    -1,    95,    -1,   168,    62,    95,    -1,    97,
      -1,   169,    62,    97,    -1,   172,    -1,   170,    62,   172,
      -1,   175,    -1,   171,    62,   175,    -1,   184,   146,   198,
      -1,   174,   198,    -1,    59,   174,    60,   198,    -1,    53,
     174,   198,    -1,    59,    53,   174,    60,   198,    -1,    53,
      59,   174,    60,   198,    -1,    24,    -1,    24,    63,   141,
      -1,   173,    -1,   138,   176,    -1,   173,    -1,    59,   173,
      60,    -1,    59,   179,    60,   163,    -1,   136,    -1,   141,
     136,    -1,   141,   145,    -1,   145,    -1,   177,    -1,   178,
      75,   177,    -1,    -1,   178,   191,    -1,    -1,   100,    -1,
      91,    -1,   181,    -1,     1,    -1,    98,    -1,   110,    -1,
     121,    -1,   124,    -1,   113,    -1,    -1,   144,    66,   182,
     180,    -1,    15,    -1,     6,   140,    -1,    10,   140,    -1,
      18,   128,    -1,    13,   128,    -1,    19,   138,    -1,    27,
     193,    -1,   180,    -1,   183,    62,   180,    -1,   138,    -1,
     184,    75,   138,    -1,   139,    -1,   185,    75,   139,    -1,
     126,    -1,   186,    75,   126,    -1,   135,    -1,   187,    75,
     135,    -1,   131,    -1,   132,    -1,   188,    75,   131,    -1,
     188,    75,   132,    -1,    -1,   188,   191,    -1,    -1,    62,
      -1,    -1,    75,    -1,    -1,   126,    -1,    -1,   186,    -1,
      -1,    98,    -1,    -1,   215,    -1,    -1,   216,    -1,    -1,
     217,    -1,    -1,     3,    -1,    21,    24,     3,    62,    -1,
      32,   200,   202,    62,    -1,     9,   200,    65,   213,    62,
      -1,     9,   200,   202,    65,   213,    62,    -1,    31,   201,
     202,    62,    -1,    17,   160,   162,    62,    -1,   142,    -1,
     200,    -1,   204,    -1,   205,    -1,   206,    -1,   204,    -1,
     206,    -1,   142,    -1,    24,    -1,    71,    72,   202,    -1,
      71,     3,    72,   202,    -1,    23,    71,   202,    72,   202,
      -1,    29,    67,   196,    68,    -1,    22,    67,   197,    68,
      -1,    53,   202,    -1,     8,   203,    -1,     8,    59,   205,
      60,    -1,     8,    36,   202,    -1,    36,     8,   202,    -1,
      17,    59,   195,    60,   210,    -1,   141,   202,   198,    -1,
     141,    11,   202,   198,    -1,   141,   202,   198,    -1,   141,
      59,   195,    60,   210,    -1,   202,    -1,    -1,   211,    -1,
      59,   195,    60,    -1,   202,    -1,     3,    -1,    50,     3,
      -1,   141,    -1,   212,    -1,    59,   212,    49,   212,    60,
      -1,    -1,   214,   199,    -1,   207,    -1,   215,    75,   207,
      -1,   208,    -1,   216,    62,   208,    -1,   209,    -1,   217,
      62,   209,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   124,   124,   133,   140,   151,   151,   166,   167,   170,
     171,   172,   175,   208,   219,   220,   223,   230,   237,   246,
     260,   261,   268,   268,   281,   285,   286,   290,   295,   301,
     305,   309,   313,   319,   325,   331,   336,   340,   344,   350,
     356,   360,   364,   370,   374,   380,   381,   385,   391,   400,
     406,   424,   429,   441,   457,   462,   469,   489,   507,   516,
     535,   534,   549,   548,   579,   582,   589,   588,   599,   605,
     614,   625,   631,   634,   642,   641,   652,   658,   670,   674,
     679,   669,   700,   699,   712,   715,   721,   724,   736,   740,
     735,   758,   757,   773,   774,   778,   782,   786,   790,   794,
     798,   802,   806,   810,   814,   818,   822,   826,   830,   834,
     838,   842,   846,   851,   857,   858,   862,   873,   877,   881,
     885,   890,   894,   904,   908,   913,   921,   925,   926,   937,
     941,   945,   949,   953,   954,   960,   967,   973,   980,   983,
     990,   996,  1013,  1020,  1021,  1028,  1029,  1048,  1049,  1052,
    1055,  1059,  1070,  1079,  1085,  1088,  1091,  1098,  1099,  1105,
    1120,  1128,  1140,  1145,  1151,  1152,  1153,  1154,  1155,  1156,
    1162,  1163,  1164,  1165,  1171,  1172,  1173,  1174,  1175,  1181,
    1182,  1185,  1188,  1189,  1190,  1191,  1192,  1195,  1196,  1209,
    1213,  1218,  1223,  1228,  1232,  1233,  1236,  1242,  1249,  1255,
    1262,  1268,  1279,  1293,  1322,  1362,  1387,  1405,  1414,  1417,
    1425,  1429,  1433,  1440,  1446,  1451,  1463,  1466,  1476,  1477,
    1483,  1484,  1490,  1494,  1500,  1501,  1507,  1511,  1517,  1540,
    1545,  1551,  1557,  1564,  1573,  1582,  1597,  1603,  1608,  1612,
    1619,  1632,  1633,  1639,  1645,  1648,  1652,  1658,  1661,  1670,
    1673,  1674,  1678,  1679,  1685,  1686,  1687,  1688,  1689,  1691,
    1690,  1705,  1710,  1714,  1718,  1722,  1726,  1731,  1750,  1756,
    1764,  1768,  1774,  1778,  1784,  1788,  1794,  1798,  1807,  1811,
    1815,  1819,  1825,  1828,  1836,  1837,  1839,  1840,  1843,  1846,
    1849,  1852,  1855,  1858,  1861,  1864,  1867,  1870,  1873,  1876,
    1879,  1882,  1888,  1892,  1896,  1900,  1904,  1908,  1928,  1935,
    1946,  1947,  1948,  1951,  1952,  1955,  1959,  1969,  1973,  1977,
    1981,  1985,  1989,  1993,  1999,  2005,  2013,  2021,  2027,  2034,
    2050,  2068,  2072,  2078,  2081,  2084,  2088,  2098,  2102,  2117,
    2125,  2126,  2138,  2139,  2142,  2146,  2152,  2156,  2162,  2166
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
const char *yytname[] =
{
  "$end", "error", "$undefined", "LLITERAL", "LASOP", "LCOLAS", "LBREAK",
  "LCASE", "LCHAN", "LCONST", "LCONTINUE", "LDDD", "LDEFAULT", "LDEFER",
  "LELSE", "LFALL", "LFOR", "LFUNC", "LGO", "LGOTO", "LIF", "LIMPORT",
  "LINTERFACE", "LMAP", "LNAME", "LPACKAGE", "LRANGE", "LRETURN",
  "LSELECT", "LSTRUCT", "LSWITCH", "LTYPE", "LVAR", "LANDAND", "LANDNOT",
  "LBODY", "LCOMM", "LDEC", "LEQ", "LGE", "LGT", "LIGNORE", "LINC", "LLE",
  "LLSH", "LLT", "LNE", "LOROR", "LRSH", "'+'", "'-'", "'|'", "'^'", "'*'",
  "'/'", "'%'", "'&'", "NotPackage", "NotParen", "'('", "')'",
  "PreferToRightParen", "';'", "'.'", "'$'", "'='", "':'", "'{'", "'}'",
  "'!'", "'~'", "'['", "']'", "'?'", "'@'", "','", "$accept", "file",
  "package", "loadsys", "@1", "imports", "import", "import_stmt",
  "import_stmt_list", "import_here", "import_package", "import_safety",
  "import_there", "@2", "xdcl", "common_dcl", "lconst", "vardcl",
  "constdcl", "constdcl1", "typedclname", "typedcl", "simple_stmt", "case",
  "compound_stmt", "@3", "caseblock", "@4", "caseblock_list", "loop_body",
  "@5", "range_stmt", "for_header", "for_body", "for_stmt", "@6",
  "if_header", "if_stmt", "@7", "@8", "@9", "elseif", "@10", "elseif_list",
  "else", "switch_stmt", "@11", "@12", "select_stmt", "@13", "expr",
  "uexpr", "pseudocall", "pexpr_no_paren", "start_complit", "keyval",
  "bare_complitexpr", "complitexpr", "pexpr", "expr_or_type",
  "name_or_type", "lbrace", "new_name", "dcl_name", "onew_name", "sym",
  "hidden_importsym", "name", "labelname", "dotdotdot", "ntype",
  "non_expr_type", "non_recvchantype", "convtype", "comptype",
  "fnret_type", "dotname", "othertype", "ptrtype", "recvchantype",
  "structtype", "interfacetype", "xfndcl", "fndcl", "hidden_fndcl",
  "fntype", "fnbody", "fnres", "fnlitdcl", "fnliteral", "xdcl_list",
  "vardcl_list", "constdcl_list", "typedcl_list", "structdcl_list",
  "interfacedcl_list", "structdcl", "packname", "embed", "interfacedcl",
  "indcl", "arg_type", "arg_type_list", "oarg_type_list_ocomma", "stmt",
  "non_dcl_stmt", "@14", "stmt_list", "new_name_list", "dcl_name_list",
  "expr_list", "expr_or_type_list", "keyval_list", "braced_keyval_list",
  "osemi", "ocomma", "oexpr", "oexpr_list", "osimple_stmt",
  "ohidden_funarg_list", "ohidden_structdcl_list",
  "ohidden_interfacedcl_list", "oliteral", "hidden_import",
  "hidden_pkg_importsym", "hidden_pkgtype", "hidden_type",
  "hidden_type_non_recv_chan", "hidden_type_misc", "hidden_type_recv_chan",
  "hidden_type_func", "hidden_funarg", "hidden_structdcl",
  "hidden_interfacedcl", "ohidden_funres", "hidden_funres",
  "hidden_literal", "hidden_constant", "hidden_import_list",
  "hidden_funarg_list", "hidden_structdcl_list",
  "hidden_interfacedcl_list", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,    43,
      45,   124,    94,    42,    47,    37,    38,   304,   305,    40,
      41,   306,    59,    46,    36,    61,    58,   123,   125,    33,
     126,    91,    93,    63,    64,    44
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    76,    77,    78,    78,    80,    79,    81,    81,    82,
      82,    82,    83,    83,    84,    84,    85,    85,    85,    86,
      87,    87,    89,    88,    90,    90,    90,    90,    90,    91,
      91,    91,    91,    91,    91,    91,    91,    91,    91,    92,
      93,    93,    93,    94,    94,    95,    95,    95,    96,    97,
      98,    98,    98,    98,    98,    98,    99,    99,    99,    99,
     101,   100,   103,   102,   104,   104,   106,   105,   107,   107,
     108,   108,   108,   109,   111,   110,   112,   112,   114,   115,
     116,   113,   118,   117,   119,   119,   120,   120,   122,   123,
     121,   125,   124,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   126,   127,   127,   127,   127,   127,   127,
     127,   127,   127,   128,   128,   128,   129,   129,   129,   129,
     129,   129,   129,   129,   129,   129,   129,   129,   129,   130,
     131,   132,   132,   133,   133,   134,   134,   135,   135,   136,
     137,   137,   138,   139,   140,   140,   141,   141,   141,   142,
     143,   144,   145,   145,   146,   146,   146,   146,   146,   146,
     147,   147,   147,   147,   148,   148,   148,   148,   148,   149,
     149,   150,   151,   151,   151,   151,   151,   152,   152,   153,
     153,   153,   153,   153,   153,   153,   154,   155,   156,   156,
     157,   157,   158,   159,   159,   160,   160,   161,   162,   162,
     163,   163,   163,   164,   165,   165,   166,   166,   167,   167,
     168,   168,   169,   169,   170,   170,   171,   171,   172,   172,
     172,   172,   172,   172,   173,   173,   174,   175,   175,   175,
     176,   177,   177,   177,   177,   178,   178,   179,   179,   180,
     180,   180,   180,   180,   181,   181,   181,   181,   181,   182,
     181,   181,   181,   181,   181,   181,   181,   181,   183,   183,
     184,   184,   185,   185,   186,   186,   187,   187,   188,   188,
     188,   188,   189,   189,   190,   190,   191,   191,   192,   192,
     193,   193,   194,   194,   195,   195,   196,   196,   197,   197,
     198,   198,   199,   199,   199,   199,   199,   199,   200,   201,
     202,   202,   202,   203,   203,   204,   204,   204,   204,   204,
     204,   204,   204,   204,   204,   204,   205,   206,   207,   207,
     208,   209,   209,   210,   210,   211,   211,   212,   212,   212,
     213,   213,   214,   214,   215,   215,   216,   216,   217,   217
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     4,     0,     3,     0,     3,     0,     3,     2,
       5,     3,     3,     2,     1,     3,     1,     2,     2,     4,
       0,     1,     0,     4,     0,     1,     1,     1,     1,     2,
       5,     3,     2,     5,     7,     3,     2,     5,     3,     1,
       2,     4,     3,     4,     3,     1,     2,     1,     1,     2,
       1,     3,     3,     3,     2,     2,     3,     5,     5,     2,
       0,     4,     0,     3,     0,     2,     0,     4,     4,     4,
       5,     1,     1,     2,     0,     3,     1,     3,     0,     0,
       0,     8,     0,     5,     0,     2,     0,     2,     0,     0,
       7,     0,     5,     1,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     1,     2,     2,     2,     2,     2,
       2,     2,     2,     3,     5,     6,     1,     1,     3,     5,
       5,     4,     6,     1,     5,     5,     5,     7,     1,     0,
       3,     1,     4,     1,     4,     1,     3,     1,     1,     1,
       1,     1,     1,     1,     0,     1,     1,     1,     1,     4,
       1,     1,     1,     2,     1,     1,     1,     1,     1,     3,
       1,     1,     1,     2,     1,     1,     1,     1,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     4,
       4,     2,     3,     5,     1,     1,     2,     3,     5,     3,
       5,     3,     3,     5,     8,     5,     8,     5,     0,     3,
       0,     1,     3,     1,     4,     2,     0,     3,     1,     3,
       1,     3,     1,     3,     1,     3,     1,     3,     3,     2,
       4,     3,     5,     5,     1,     3,     1,     2,     1,     3,
       4,     1,     2,     2,     1,     1,     3,     0,     2,     0,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     0,
       4,     1,     2,     2,     2,     2,     2,     2,     1,     3,
       1,     3,     1,     3,     1,     3,     1,     3,     1,     1,
       3,     3,     0,     2,     0,     1,     0,     1,     0,     1,
       0,     1,     0,     1,     0,     1,     0,     1,     0,     1,
       0,     1,     4,     4,     5,     6,     4,     4,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     4,     5,
       4,     4,     2,     2,     4,     3,     3,     5,     3,     4,
       3,     5,     1,     0,     1,     3,     1,     1,     2,     1,
       1,     5,     0,     2,     1,     3,     1,     3,     1,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       5,     0,     3,     0,     1,     0,     7,     0,    22,   156,
     158,     0,     0,   157,   216,    20,     6,   342,     0,     4,
       0,     0,     0,    21,     0,     0,     0,    16,     0,     0,
       9,    22,     0,     8,    28,   126,   154,     0,    39,   154,
       0,   261,    74,     0,     0,     0,    78,     0,     0,   290,
      91,     0,    88,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   288,     0,    25,     0,   254,   255,
     258,   256,   257,    50,    93,   133,   145,   114,   161,   160,
     127,     0,     0,     0,   181,   194,   195,    26,   213,     0,
     138,    27,     0,    19,     0,     0,     0,     0,     0,     0,
     343,   159,    11,    14,   284,    18,    22,    13,    17,   155,
     262,   152,     0,     0,     0,     0,   160,   187,   191,   177,
     175,   176,   174,   263,   133,     0,   292,   247,     0,   208,
     133,   266,   292,   150,   151,     0,     0,   274,   291,   267,
       0,     0,   292,     0,     0,    36,    48,     0,    29,   272,
     153,     0,   122,   117,   118,   121,   115,   116,     0,     0,
     147,     0,   148,   172,   170,   171,   119,   120,     0,   289,
       0,   217,     0,    32,     0,     0,     0,     0,     0,    55,
       0,     0,     0,    54,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   139,     0,
       0,   288,   259,     0,   139,   215,     0,     0,     0,     0,
     308,     0,     0,   208,     0,     0,   309,     0,     0,    23,
     285,     0,    12,   247,     0,     0,   192,   168,   166,   167,
     164,   165,   196,     0,     0,   293,    72,     0,    75,     0,
      71,   162,   241,   160,   244,   149,   245,   286,     0,   247,
       0,   202,    79,    76,   156,     0,   201,     0,   284,   238,
     226,     0,    64,     0,     0,   199,   270,   284,   224,   236,
     300,     0,    89,    38,   222,   284,    49,    31,   218,   284,
       0,     0,    40,     0,   173,   146,     0,     0,    35,   284,
       0,     0,    51,    95,   110,   113,    96,   100,   101,    99,
     111,    98,    97,    94,   112,   102,   103,   104,   105,   106,
     107,   108,   109,   282,   123,   276,   286,     0,   128,   289,
       0,     0,   286,   282,   253,    60,   251,   250,   268,   252,
       0,    53,    52,   275,     0,     0,     0,     0,   316,     0,
       0,     0,     0,     0,   315,     0,   310,   311,   312,     0,
     344,     0,     0,   294,     0,     0,     0,    15,    10,     0,
       0,     0,   178,   188,    66,    73,     0,     0,   292,   163,
     242,   243,   287,   248,   210,     0,     0,     0,   292,     0,
     234,     0,   247,   237,   285,     0,     0,     0,     0,   300,
       0,     0,   285,     0,   301,   229,     0,   300,     0,   285,
       0,   285,     0,    42,   273,     0,     0,     0,   197,   168,
     166,   167,   165,   139,   190,   189,   285,     0,    44,     0,
     139,   141,   278,   279,   286,     0,   286,   287,     0,     0,
       0,   131,   288,   260,   287,     0,     0,     0,     0,   214,
       0,     0,   323,   313,   314,   294,   298,     0,   296,     0,
     322,   337,     0,     0,   339,   340,     0,     0,     0,     0,
       0,   300,     0,     0,   307,     0,   295,   302,   306,   303,
     210,   169,     0,     0,     0,     0,   246,   247,   160,   211,
     186,   184,   185,   182,   183,   207,   210,   209,    80,    77,
     235,   239,     0,   227,   200,   193,     0,     0,    92,    62,
      65,     0,   231,     0,   300,   225,   198,   271,   228,    64,
     223,    37,   219,    30,    41,     0,   282,    45,   220,   284,
      47,    33,    43,   282,     0,   287,   283,   136,     0,   277,
     124,   130,   129,     0,   134,   135,     0,   269,   325,     0,
       0,   316,     0,   315,     0,   332,   348,   299,     0,     0,
       0,   346,   297,   326,   338,     0,   304,     0,   317,     0,
     300,   328,     0,   345,   333,     0,    69,    68,   292,     0,
     247,   203,    84,   210,     0,    59,     0,   300,   300,   230,
       0,   169,     0,   285,     0,    46,     0,   139,   143,   140,
     280,   281,   125,   132,    61,   324,   333,   294,   321,     0,
       0,   300,   320,     0,     0,   318,   305,   329,   294,   294,
     336,   205,   334,    67,    70,   212,     0,    86,   240,     0,
       0,    56,     0,    63,   233,   232,    90,   137,   221,    34,
     142,   282,   327,     0,   349,   319,   330,   347,     0,     0,
       0,   210,     0,    85,    81,     0,     0,     0,   333,   341,
     333,   335,   204,    82,    87,    58,    57,   144,   331,   206,
     292,     0,    83
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     6,     2,     3,    14,    21,    30,   104,    31,
       8,    24,    16,    17,    65,   326,    67,   148,   517,   518,
     144,   145,    68,   499,   327,   437,   500,   576,   387,   365,
     472,   236,   237,   238,    69,   126,   252,    70,   132,   377,
     572,   643,   660,   617,   644,    71,   142,   398,    72,   140,
      73,    74,    75,    76,   313,   422,   423,   589,    77,   315,
     242,   135,    78,   149,   110,   116,    13,    80,    81,   244,
     245,   162,   118,    82,    83,   479,   227,    84,   229,   230,
      85,    86,    87,   129,   213,    88,   251,   485,    89,    90,
      22,   279,   519,   275,   267,   258,   268,   269,   270,   260,
     383,   246,   247,   248,   328,   329,   321,   330,   271,   151,
      92,   316,   424,   425,   221,   373,   170,   139,   253,   465,
     550,   544,   395,   100,   211,   217,   610,   442,   346,   347,
     348,   350,   551,   546,   611,   612,   455,   456,    25,   466,
     552,   547
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -485
static const yytype_int16 yypact[] =
{
    -485,    67,    35,    55,  -485,    44,  -485,    64,  -485,  -485,
    -485,    96,    38,  -485,    77,    85,  -485,  -485,    66,  -485,
      34,    84,  1059,  -485,    86,   294,   147,  -485,   165,   210,
    -485,    55,   221,  -485,  -485,  -485,    44,  1762,  -485,    44,
     290,  -485,  -485,   442,   290,    44,  -485,    80,    69,  1608,
    -485,    80,  -485,   450,   452,  1608,  1608,  1608,  1608,  1608,
    1608,  1651,  1608,  1608,   920,   157,  -485,   460,  -485,  -485,
    -485,  -485,  -485,   718,  -485,  -485,   167,   344,  -485,   176,
    -485,   180,   193,    80,   206,  -485,  -485,  -485,   218,    91,
    -485,  -485,    76,  -485,   205,    10,   260,   205,   205,   223,
    -485,  -485,  -485,  -485,   230,  -485,  -485,  -485,  -485,  -485,
    -485,  -485,   237,  1770,  1770,  1770,  -485,   236,  -485,  -485,
    -485,  -485,  -485,  -485,   220,   344,  1608,   990,   241,   235,
     262,  -485,  1608,  -485,  -485,   405,  1770,  2090,   254,  -485,
     297,   444,  1608,    61,  1770,  -485,  -485,   271,  -485,  -485,
    -485,   671,  -485,  -485,  -485,  -485,  -485,  -485,  1694,  1651,
    2090,   291,  -485,   181,  -485,    60,  -485,  -485,   287,  2090,
     301,  -485,   496,  -485,   912,  1608,  1608,  1608,  1608,  -485,
    1608,  1608,  1608,  -485,  1608,  1608,  1608,  1608,  1608,  1608,
    1608,  1608,  1608,  1608,  1608,  1608,  1608,  1608,  -485,  1290,
     468,  1608,  -485,  1608,  -485,  -485,  1221,  1608,  1608,  1608,
    -485,   573,    44,   235,   275,   347,  -485,  1301,  1301,  -485,
     113,   302,  -485,   990,   358,  1770,  -485,  -485,  -485,  -485,
    -485,  -485,  -485,   316,    44,  -485,  -485,   340,  -485,    78,
     318,  1770,  -485,   990,  -485,  -485,  -485,   307,   325,   990,
    1221,  -485,  -485,   324,   117,   365,  -485,   343,   337,  -485,
    -485,   333,  -485,    32,    23,  -485,  -485,   350,  -485,  -485,
     406,  1737,  -485,  -485,  -485,   351,  -485,  -485,  -485,   352,
    1608,    44,   354,  1796,  -485,   353,  1770,  1770,  -485,   359,
    1608,   357,  2090,  1928,  -485,  2114,  1212,  1212,  1212,  1212,
    -485,  1212,  1212,  2138,  -485,   566,   566,   566,   566,  -485,
    -485,  -485,  -485,  1345,  -485,  -485,    31,  1400,  -485,  1988,
     360,  1147,  1955,  1345,  -485,  -485,  -485,  -485,  -485,  -485,
      95,   254,   254,  2090,  1857,   368,   361,   371,  -485,   363,
     427,  1301,   247,    51,  -485,   374,  -485,  -485,  -485,  1890,
    -485,    36,   382,    44,   384,   385,   387,  -485,  -485,   391,
    1770,   395,  -485,  -485,  -485,  -485,  1455,  1510,  1608,  -485,
    -485,  -485,   990,  -485,  1823,   399,   135,   340,  1608,    44,
     397,   403,   990,  -485,   542,   407,  1770,   278,   365,   406,
     365,   411,   364,   413,  -485,  -485,    44,   406,   430,    44,
     423,    44,   425,   254,  -485,  1608,  1849,  1770,  -485,   216,
     219,   274,   288,  -485,  -485,  -485,    44,   426,   254,  1608,
    -485,  2018,  -485,  -485,   414,   422,   416,  1651,   433,   434,
     436,  -485,  1608,  -485,  -485,   439,   437,  1221,  1147,  -485,
    1301,   466,  -485,  -485,  -485,    44,  1882,  1301,    44,  1301,
    -485,  -485,   504,   207,  -485,  -485,   446,   438,  1301,   247,
    1301,   406,    44,    44,  -485,   453,   455,  -485,  -485,  -485,
    1823,  -485,  1221,  1608,  1608,   467,  -485,   990,   472,  -485,
    -485,  -485,  -485,  -485,  -485,  -485,  1823,  -485,  -485,  -485,
    -485,  -485,   475,  -485,  -485,  -485,  1651,   470,  -485,  -485,
    -485,   490,  -485,   493,   406,  -485,  -485,  -485,  -485,  -485,
    -485,  -485,  -485,  -485,   254,   495,  1345,  -485,  -485,   498,
     912,  -485,   254,  1345,  1553,  1345,  -485,  -485,   497,  -485,
    -485,  -485,  -485,   486,  -485,  -485,   143,  -485,  -485,   501,
     502,   473,   508,   513,   505,  -485,  -485,   515,   503,  1301,
     511,  -485,   518,  -485,  -485,   533,  -485,  1301,  -485,   522,
     406,  -485,   526,  -485,  1916,   144,  2090,  2090,  1608,   527,
     990,  -485,  -485,  1823,    39,  -485,  1147,   406,   406,  -485,
     315,   293,   521,    44,   548,   357,   525,  -485,  2090,  -485,
    -485,  -485,  -485,  -485,  -485,  -485,  1916,    44,  -485,  1882,
    1301,   406,  -485,    44,   207,  -485,  -485,  -485,    44,    44,
    -485,  -485,  -485,  -485,  -485,  -485,   551,   572,  -485,  1608,
    1608,  -485,  1651,   550,  -485,  -485,  -485,  -485,  -485,  -485,
    -485,  1345,  -485,   558,  -485,  -485,  -485,  -485,   563,   564,
     565,  1823,    46,  -485,  -485,  2042,  2066,   559,  1916,  -485,
    1916,  -485,  -485,  -485,  -485,  -485,  -485,  -485,  -485,  -485,
    1608,   340,  -485
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -485,  -485,  -485,  -485,  -485,  -485,  -485,    -6,  -485,  -485,
     597,  -485,    -3,  -485,  -485,   608,  -485,  -131,   -28,    50,
    -485,  -135,  -106,  -485,    -7,  -485,  -485,  -485,   125,  -370,
    -485,  -485,  -485,  -485,  -485,  -485,  -138,  -485,  -485,  -485,
    -485,  -485,  -485,  -485,  -485,  -485,  -485,  -485,  -485,  -485,
     665,    15,   116,  -485,  -190,   111,   112,  -485,   164,   -59,
     398,   137,    14,   367,   603,    -5,   454,   432,  -485,   402,
     -50,   491,  -485,  -485,  -485,  -485,   -36,    18,   -34,    -9,
    -485,  -485,  -485,  -485,  -485,   257,   441,  -445,  -485,  -485,
    -485,  -485,  -485,  -485,  -485,  -485,   259,  -116,  -218,   265,
    -485,   284,  -485,  -217,  -286,   636,  -485,  -237,  -485,   -62,
     -24,   166,  -485,  -314,  -246,  -265,  -177,  -485,  -115,  -415,
    -485,  -485,  -379,  -485,    -8,  -485,   435,  -485,   326,   225,
     327,   204,    65,    70,  -484,  -485,  -426,   211,  -485,   462,
    -485,  -485
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -275
static const yytype_int16 yytable[] =
{
      12,   119,   161,   121,   272,   174,   359,   488,   274,   436,
     502,   240,   385,   376,   323,    32,   278,    79,   508,   259,
     235,   393,   103,    32,   320,   138,   235,   555,   107,   400,
     540,   111,   375,   402,   111,   433,   235,    27,   128,   173,
     111,   571,   426,   417,   619,   389,   391,   380,   146,   150,
     109,   428,   164,   109,   457,   120,   380,   435,     9,   131,
       5,  -213,   150,   226,   232,   233,   653,     4,     9,   212,
     152,   153,   154,   155,   156,   157,   390,   166,   167,   163,
       7,   207,   561,   366,    11,     9,   261,   214,    15,   216,
     218,   388,   205,    28,   276,  -213,   462,    29,    20,    18,
      19,   282,   239,   222,   620,   621,   427,    10,    11,    23,
     174,   463,   632,   325,   622,   133,    27,    10,    11,  -179,
    -234,   273,   243,   458,   291,   579,   133,  -213,   618,    26,
     111,   228,   228,   228,    10,    11,   111,     9,   146,   381,
     136,   208,   150,   367,   289,   228,    33,   134,    93,   257,
     164,   209,   537,   209,   228,   266,   124,   438,   134,   526,
     130,   528,   228,   439,   658,   492,   659,   150,    27,   228,
     501,   101,   503,   152,   156,   361,    29,   163,   638,  -234,
     379,   607,   633,   331,   332,  -234,    10,    11,   141,     9,
     164,   369,   228,   639,   640,   318,   652,   438,   624,   625,
     536,    79,   582,   487,   125,   438,   438,   349,   125,   586,
     451,   594,   613,   105,   357,    32,  -181,   163,   243,   171,
     204,   397,   636,   516,   108,   102,   206,  -265,    29,   363,
     523,     9,  -265,   408,   198,   565,   414,   415,    10,    11,
    -180,   228,  -152,   228,   243,    79,   202,   409,  -181,   411,
     451,  -177,   203,   475,  -175,   533,   403,   452,   430,   228,
     569,   228,   235,   489,   510,  -180,   418,   228,   259,  -264,
     512,     9,   235,   584,  -264,  -177,   150,  -179,  -175,    11,
      10,    11,  -265,  -177,   215,   496,  -175,   219,  -265,   228,
     497,   662,   220,    35,   122,     9,   223,   452,    37,   234,
     249,   410,   250,    94,   228,   228,   453,   112,   164,  -176,
     408,    95,    47,    48,     9,    96,    79,   647,   165,    51,
      10,    11,   496,  -174,  -264,    97,    98,   497,  -178,   209,
    -264,   277,   262,  -176,   353,   163,   495,   454,   480,   623,
     482,  -176,   331,   332,    10,    11,   498,  -174,   349,    61,
     354,   285,  -178,   616,   520,  -174,   226,   515,    99,   286,
    -178,    64,   358,    10,    11,   483,   360,   243,   529,   478,
     231,   231,   231,   287,   490,   364,   362,   243,   228,   111,
     368,   514,   372,   626,   231,   374,   378,   111,   254,   380,
     228,   111,   481,   231,   146,   522,   150,   631,   257,   384,
     228,   231,   382,   199,   228,   386,   266,   200,   231,   394,
     507,   150,   392,   399,   401,   201,   165,   263,   164,   405,
     413,   416,   419,   264,   228,   228,   432,   445,   446,   254,
     448,   231,    79,    79,   480,   449,   482,    10,    11,   459,
     349,   542,   447,   549,   464,   163,   467,   468,   454,   469,
     480,   470,   482,   614,   454,   471,   165,   562,   349,   486,
     379,   483,   235,   491,   255,   509,     9,    79,   254,   117,
     585,   504,   243,   256,     9,   494,     9,   483,    10,    11,
     231,   506,   231,   511,     9,   513,   521,   164,   481,   525,
     527,   434,     9,   530,   531,   228,   532,   263,   231,   534,
     231,   127,   340,   264,   481,   535,   231,   554,   556,   143,
     557,   147,   265,   564,   163,    10,    11,    10,    11,   172,
       9,   520,   661,    10,    11,    10,    11,   317,   231,   568,
     463,   570,  -156,    10,    11,   573,   575,   480,   228,   482,
     412,    10,    11,   231,   231,   117,   117,   117,   210,   210,
     577,   210,   210,   578,   235,   581,   288,   592,   593,   117,
     583,   595,   596,   529,   483,   243,   254,   597,   117,    10,
      11,    79,  -157,   598,   165,   600,   117,   599,   150,   602,
     603,   334,   604,   117,   606,   608,   642,   615,   228,   627,
     335,   481,   349,   630,   542,   336,   337,   338,   549,   454,
     177,   255,   339,   349,   349,   480,   117,   482,   629,   340,
     185,   641,   438,   164,   189,    10,    11,   231,   648,   194,
     195,   196,   197,   649,   650,   651,   341,   657,   106,   231,
      66,   484,   483,   628,   580,   654,   590,   591,   342,   231,
     163,   370,   123,   231,   343,   371,   345,    11,   404,   493,
     284,   505,   355,   356,   352,   117,   476,   117,    91,   481,
     443,   444,   574,   231,   231,   344,   539,   563,   637,   634,
     559,   344,   344,   117,   351,   117,     0,     0,     0,    37,
       0,   117,     0,     0,   165,     0,     0,     0,   112,     0,
       0,     0,     0,    47,    48,     9,     0,     0,     0,     0,
      51,     0,     0,   117,     0,     0,     0,   224,     0,     0,
       0,     0,     0,     0,   137,   117,     0,     0,   117,   117,
       0,     0,   175,  -274,   114,     0,   160,   484,     0,   169,
     225,     0,     0,     0,   231,     0,   280,     0,     0,     0,
       0,     0,    64,   484,    10,    11,   281,     0,     0,     0,
       0,   176,   177,   165,   178,   179,   180,   181,   182,     0,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,     0,   450,   231,     0,     0,
       0,     0,     0,  -274,   461,     0,     0,     0,   344,     0,
       0,     0,   117,  -274,     0,   344,     0,     0,     0,     0,
       0,     0,     0,   344,   117,     0,   117,     0,     0,     0,
       0,     0,     0,     0,   117,     0,     0,     0,   117,     0,
       0,     0,     0,     0,     0,     0,     0,   231,     0,     0,
     484,     0,     0,     0,     0,     0,     0,     0,   117,   117,
     292,   293,   294,   295,     0,   296,   297,   298,     0,   299,
     300,   301,   302,   303,   304,   305,   306,   307,   308,   309,
     310,   311,   312,     0,   160,     0,   319,     0,   322,     0,
       0,     0,   137,   137,   333,   538,     0,     0,     0,   165,
       0,   545,   548,     0,   553,     0,     0,     0,     0,     0,
       0,     0,     0,   558,   344,   560,     0,     0,   484,     0,
     543,   344,   117,   344,     0,     0,     0,     0,     0,   117,
       0,     0,   344,     0,   344,     0,     0,     0,   117,     0,
      37,     0,     0,    35,     0,     0,     0,     0,    37,   112,
       0,   168,     0,     0,    47,    48,     9,   112,     0,     0,
       0,    51,    47,    48,     9,   137,     0,     0,   224,    51,
       0,     0,   117,     0,     0,   137,    55,     0,     0,     0,
       0,     0,     0,     0,     0,   114,     0,     0,     0,    56,
      57,   225,    58,    59,     0,     0,    60,   290,   421,    61,
       0,     0,   160,    64,   601,    10,    11,   281,   421,    62,
      63,    64,   605,    10,    11,     0,     0,     0,    37,     0,
       0,   241,   117,   344,     0,   117,     0,   112,     0,     0,
       0,   344,    47,    48,     9,     0,     0,     0,   344,    51,
       0,     0,     0,     0,     0,     0,   224,     0,     0,     0,
       0,   137,   137,     0,   545,   635,     0,     0,     0,     0,
       0,     0,     0,   114,     0,     0,     0,     0,     0,   225,
     344,     0,     0,   543,   344,     0,     0,     0,     0,    -2,
      34,    64,    35,    10,    11,    36,     0,    37,    38,    39,
     137,     0,    40,   117,    41,    42,    43,    44,    45,    46,
       0,    47,    48,     9,   137,     0,    49,    50,    51,    52,
      53,    54,   160,     0,     0,    55,     0,   169,     0,     0,
       0,     0,   344,     0,   344,     0,     0,     0,    56,    57,
       0,    58,    59,     0,     0,    60,     0,     0,    61,     0,
       0,   -24,     0,     0,     0,     0,     0,     0,    62,    63,
      64,     0,    10,    11,     0,     0,     0,     0,   566,   567,
       0,     0,     0,     0,     0,     0,     0,     0,   324,     0,
      35,     0,     0,    36,  -249,    37,    38,    39,     0,  -249,
      40,   160,    41,    42,   112,    44,    45,    46,     0,    47,
      48,     9,     0,     0,    49,    50,    51,    52,    53,    54,
       0,   421,     0,    55,     0,     0,     0,     0,   421,   588,
     421,     0,     0,     0,     0,     0,    56,    57,     0,    58,
      59,     0,     0,    60,     0,     0,    61,     0,     0,  -249,
       0,     0,     0,     0,   325,  -249,    62,    63,    64,     0,
      10,    11,   324,     0,    35,     0,     0,    36,     0,    37,
      38,    39,     0,     0,    40,     0,    41,    42,   112,    44,
      45,    46,     0,    47,    48,     9,   177,     0,    49,    50,
      51,    52,    53,    54,     0,     0,   185,    55,     0,     0,
     189,   190,   191,   192,   193,   194,   195,   196,   197,     0,
      56,    57,     0,    58,    59,     0,     0,    60,     0,     0,
      61,     0,     0,  -249,   645,   646,     0,   160,   325,  -249,
      62,    63,    64,    35,    10,    11,   421,     0,    37,     0,
       0,     0,     0,     0,     0,     0,     0,   112,     0,   334,
       0,     0,    47,    48,     9,     0,     0,     0,   335,    51,
       0,     0,     0,   336,   337,   338,   158,     0,     0,     0,
     339,     0,     0,     0,     0,     0,     0,   340,     0,    56,
      57,     0,    58,   159,     0,     0,    60,     0,    35,    61,
     314,     0,     0,    37,   341,     0,     0,     0,     0,    62,
      63,    64,   112,    10,    11,     0,     0,    47,    48,     9,
       0,     0,   343,     0,    51,    11,     0,     0,     0,     0,
       0,    55,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    56,    57,     0,    58,    59,     0,
       0,    60,     0,    35,    61,     0,     0,     0,    37,     0,
       0,     0,   420,     0,    62,    63,    64,   112,    10,    11,
       0,     0,    47,    48,     9,     0,     0,     0,     0,    51,
       0,   429,     0,     0,     0,     0,   158,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    56,
      57,     0,    58,   159,     0,     0,    60,     0,    35,    61,
       0,     0,     0,    37,     0,     0,     0,     0,     0,    62,
      63,    64,   112,    10,    11,     0,     0,    47,    48,     9,
       0,   473,     0,     0,    51,     0,     0,     0,     0,     0,
       0,    55,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    56,    57,     0,    58,    59,     0,
       0,    60,     0,    35,    61,     0,     0,     0,    37,     0,
       0,     0,     0,     0,    62,    63,    64,   112,    10,    11,
       0,     0,    47,    48,     9,     0,   474,     0,     0,    51,
       0,     0,     0,     0,     0,     0,    55,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    35,     0,     0,    56,
      57,    37,    58,    59,     0,     0,    60,     0,     0,    61,
     112,     0,     0,     0,     0,    47,    48,     9,     0,    62,
      63,    64,    51,    10,    11,     0,     0,     0,     0,    55,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    56,    57,     0,    58,    59,     0,     0,    60,
       0,    35,    61,     0,     0,     0,    37,     0,     0,     0,
     587,     0,    62,    63,    64,   112,    10,    11,     0,     0,
      47,    48,     9,     0,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,    55,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    35,     0,     0,    56,    57,    37,
      58,    59,     0,     0,    60,     0,     0,    61,   112,     0,
       0,     0,     0,    47,    48,     9,     0,    62,    63,    64,
      51,    10,    11,     0,     0,     0,     0,   158,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    35,     0,     0,
      56,    57,   283,    58,   159,     0,     0,    60,     0,     0,
      61,   112,     0,     0,     0,     0,    47,    48,     9,     0,
      62,    63,    64,    51,    10,    11,     0,     0,     0,     0,
      55,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    56,    57,    37,    58,    59,     0,     0,
      60,     0,     0,    61,   112,     0,     0,     0,     0,    47,
      48,     9,     0,    62,    63,    64,    51,    10,    11,     0,
      37,     0,     0,   224,     0,     0,     0,     0,    37,   112,
       0,     0,     0,     0,    47,    48,     9,   112,     0,     0,
     114,    51,    47,    48,     9,     0,   225,     0,   113,    51,
       0,     0,     0,     0,    37,     0,   224,     0,    64,     0,
      10,    11,   396,   112,     0,   114,     0,     0,    47,    48,
       9,   115,     0,   114,     0,    51,     0,     0,     0,   225,
       0,    37,   406,    64,     0,    10,    11,     0,     0,     0,
     112,    64,     0,    10,    11,    47,    48,     9,     0,   114,
       0,     0,    51,     0,     0,   407,     0,   283,     0,   224,
       0,     0,     0,     0,     0,   334,   112,    64,     0,    10,
      11,    47,    48,     9,   335,     0,   114,     0,    51,   336,
     337,   338,   477,     0,     0,   224,   339,     0,     0,     0,
     334,     0,     0,   440,    64,     0,    10,    11,   334,   335,
       0,   460,   114,     0,   336,   337,   541,   335,   225,     0,
     341,   339,   336,   337,   338,     0,   441,     0,   340,   339,
      64,     0,    10,    11,   334,     0,   340,     0,   343,     0,
       0,    11,     0,   335,     0,   341,     0,     0,   336,   337,
     338,     0,     0,   341,     0,   339,     0,     0,     0,     0,
       0,     0,   340,   343,     0,    10,    11,     0,     0,     0,
       0,   343,   177,     0,    11,     0,   180,   181,   182,   341,
       0,   184,   185,   186,   187,   609,   189,   190,   191,   192,
     193,   194,   195,   196,   197,     0,     0,   343,   176,   177,
      11,   178,     0,   180,   181,   182,     0,     0,   184,   185,
     186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,   197,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   176,   177,     0,   178,     0,   180,   181,   182,     0,
     434,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,     0,     0,     0,     0,     0,
       0,   176,   177,     0,   178,     0,   180,   181,   182,     0,
     431,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   176,   177,     0,   178,     0,
     180,   181,   182,     0,   524,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   176,
     177,     0,   178,     0,   180,   181,   182,     0,   655,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   176,   177,     0,   178,     0,   180,   181,
     182,     0,   656,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   176,   177,     0,
       0,     0,   180,   181,   182,     0,     0,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   176,   177,     0,     0,     0,   180,   181,   182,     0,
       0,   184,   185,   186,   187,     0,   189,   190,   191,   192,
     193,   194,   195,   196,   197
};

static const yytype_int16 yycheck[] =
{
       5,    37,    61,    37,   142,    67,   223,   377,   143,   323,
     389,   126,   258,   250,   204,    20,   147,    22,   397,   135,
     126,   267,    28,    28,   201,    49,   132,   453,    31,   275,
     445,    36,   249,   279,    39,   321,   142,     3,    43,    67,
      45,   486,    11,   289,     5,   263,   264,    24,    53,    54,
      36,   316,    61,    39,     3,    37,    24,   322,    24,    45,
      25,     1,    67,   113,   114,   115,    20,     0,    24,    59,
      55,    56,    57,    58,    59,    60,    53,    62,    63,    61,
      25,     5,   461,     5,    74,    24,   136,    95,    24,    97,
      98,    59,     1,    59,   144,    35,    60,    63,    21,     3,
      62,   151,   126,   106,    65,    66,    75,    73,    74,    24,
     172,    75,   596,    67,    75,    35,     3,    73,    74,    59,
       3,    60,   127,    72,   174,   504,    35,    67,   573,    63,
     135,   113,   114,   115,    73,    74,   141,    24,   143,   255,
      71,    65,   147,    65,   172,   127,    62,    67,    62,   135,
     159,    75,   438,    75,   136,   141,    40,    62,    67,   424,
      44,   426,   144,    68,   648,   382,   650,   172,     3,   151,
     388,    24,   390,   158,   159,   225,    63,   159,   604,    62,
      63,   560,   597,   207,   208,    68,    73,    74,    51,    24,
     199,   241,   174,   608,   609,   200,   641,    62,   577,   578,
     437,   206,   516,    68,    40,    62,    62,   212,    44,   523,
       3,    68,    68,     3,   220,   220,    35,   199,   223,    62,
      83,   271,   601,   413,     3,    60,    89,     7,    63,   234,
     420,    24,    12,   283,    67,   472,   286,   287,    73,    74,
      59,   223,    66,   225,   249,   250,    66,   283,    67,   283,
       3,    35,    59,   368,    35,   432,   280,    50,   317,   241,
     477,   243,   368,   378,   399,    59,   290,   249,   384,     7,
     401,    24,   378,   519,    12,    59,   281,    59,    59,    74,
      73,    74,    62,    67,    24,     7,    67,    64,    68,   271,
      12,   661,    62,     3,    37,    24,    59,    50,     8,    63,
      59,   283,    67,     9,   286,   287,    59,    17,   317,    35,
     360,    17,    22,    23,    24,    21,   321,   631,    61,    29,
      73,    74,     7,    35,    62,    31,    32,    12,    35,    75,
      68,    60,    35,    59,    59,   317,   386,   342,   374,   576,
     374,    67,   366,   367,    73,    74,    68,    59,   353,    59,
       3,    60,    59,   570,   416,    67,   406,   407,    64,    72,
      67,    71,    60,    73,    74,   374,     8,   372,   427,   374,
     113,   114,   115,    72,   379,    35,    60,   382,   360,   384,
      62,   405,    75,    68,   127,    60,    62,   392,    24,    24,
     372,   396,   374,   136,   399,   419,   401,   587,   384,    62,
     382,   144,    59,    59,   386,    72,   392,    63,   151,     3,
     396,   416,    62,    62,    62,    71,   159,    53,   427,    65,
      67,    62,    65,    59,   406,   407,    66,    59,    67,    24,
      67,   174,   437,   438,   470,     8,   470,    73,    74,    65,
     445,   446,    71,   448,    62,   427,    62,    62,   453,    62,
     486,    60,   486,   568,   459,    60,   199,   462,   463,    60,
      63,   470,   568,    60,    59,    35,    24,   472,    24,    37,
     520,    60,   477,    68,    24,    68,    24,   486,    73,    74,
     223,    68,   225,    60,    24,    60,    60,   496,   470,    75,
      68,    75,    24,    60,    60,   477,    60,    53,   241,    60,
     243,    59,    36,    59,   486,    68,   249,     3,    62,    59,
      72,    59,    68,    60,   496,    73,    74,    73,    74,    59,
      24,   583,   660,    73,    74,    73,    74,    59,   271,    62,
      75,    59,    59,    73,    74,    60,    66,   573,   520,   573,
     283,    73,    74,   286,   287,   113,   114,   115,    94,    95,
      60,    97,    98,    60,   660,    60,    60,    60,    72,   127,
      62,    60,    60,   622,   573,   570,    24,    59,   136,    73,
      74,   576,    59,    68,   317,    72,   144,    62,   583,    68,
      62,     8,    49,   151,    62,    59,    14,    60,   570,    68,
      17,   573,   597,    68,   599,    22,    23,    24,   603,   604,
      34,    59,    29,   608,   609,   641,   174,   641,    60,    36,
      44,    60,    62,   622,    48,    73,    74,   360,    60,    53,
      54,    55,    56,    60,    60,    60,    53,    68,    31,   372,
      22,   374,   641,   583,   509,   642,   525,   525,    65,   382,
     622,   243,    39,   386,    71,   243,   211,    74,   281,   384,
     159,   392,   217,   218,   213,   223,   372,   225,    22,   641,
     334,   334,   496,   406,   407,   211,   441,   463,   603,   599,
     459,   217,   218,   241,   212,   243,    -1,    -1,    -1,     8,
      -1,   249,    -1,    -1,   427,    -1,    -1,    -1,    17,    -1,
      -1,    -1,    -1,    22,    23,    24,    -1,    -1,    -1,    -1,
      29,    -1,    -1,   271,    -1,    -1,    -1,    36,    -1,    -1,
      -1,    -1,    -1,    -1,    49,   283,    -1,    -1,   286,   287,
      -1,    -1,     4,     5,    53,    -1,    61,   470,    -1,    64,
      59,    -1,    -1,    -1,   477,    -1,    65,    -1,    -1,    -1,
      -1,    -1,    71,   486,    73,    74,    75,    -1,    -1,    -1,
      -1,    33,    34,   496,    36,    37,    38,    39,    40,    -1,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    -1,   341,   520,    -1,    -1,
      -1,    -1,    -1,    65,   349,    -1,    -1,    -1,   334,    -1,
      -1,    -1,   360,    75,    -1,   341,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   349,   372,    -1,   374,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   382,    -1,    -1,    -1,   386,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   570,    -1,    -1,
     573,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   406,   407,
     175,   176,   177,   178,    -1,   180,   181,   182,    -1,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,    -1,   199,    -1,   201,    -1,   203,    -1,
      -1,    -1,   207,   208,   209,   440,    -1,    -1,    -1,   622,
      -1,   446,   447,    -1,   449,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   458,   440,   460,    -1,    -1,   641,    -1,
     446,   447,   470,   449,    -1,    -1,    -1,    -1,    -1,   477,
      -1,    -1,   458,    -1,   460,    -1,    -1,    -1,   486,    -1,
       8,    -1,    -1,     3,    -1,    -1,    -1,    -1,     8,    17,
      -1,    11,    -1,    -1,    22,    23,    24,    17,    -1,    -1,
      -1,    29,    22,    23,    24,   280,    -1,    -1,    36,    29,
      -1,    -1,   520,    -1,    -1,   290,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    53,    -1,    -1,    -1,    49,
      50,    59,    52,    53,    -1,    -1,    56,    65,   313,    59,
      -1,    -1,   317,    71,   549,    73,    74,    75,   323,    69,
      70,    71,   557,    73,    74,    -1,    -1,    -1,     8,    -1,
      -1,    11,   570,   549,    -1,   573,    -1,    17,    -1,    -1,
      -1,   557,    22,    23,    24,    -1,    -1,    -1,   564,    29,
      -1,    -1,    -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,
      -1,   366,   367,    -1,   599,   600,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    53,    -1,    -1,    -1,    -1,    -1,    59,
     596,    -1,    -1,   599,   600,    -1,    -1,    -1,    -1,     0,
       1,    71,     3,    73,    74,     6,    -1,     8,     9,    10,
     405,    -1,    13,   641,    15,    16,    17,    18,    19,    20,
      -1,    22,    23,    24,   419,    -1,    27,    28,    29,    30,
      31,    32,   427,    -1,    -1,    36,    -1,   432,    -1,    -1,
      -1,    -1,   648,    -1,   650,    -1,    -1,    -1,    49,    50,
      -1,    52,    53,    -1,    -1,    56,    -1,    -1,    59,    -1,
      -1,    62,    -1,    -1,    -1,    -1,    -1,    -1,    69,    70,
      71,    -1,    73,    74,    -1,    -1,    -1,    -1,   473,   474,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
       3,    -1,    -1,     6,     7,     8,     9,    10,    -1,    12,
      13,   496,    15,    16,    17,    18,    19,    20,    -1,    22,
      23,    24,    -1,    -1,    27,    28,    29,    30,    31,    32,
      -1,   516,    -1,    36,    -1,    -1,    -1,    -1,   523,   524,
     525,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,    52,
      53,    -1,    -1,    56,    -1,    -1,    59,    -1,    -1,    62,
      -1,    -1,    -1,    -1,    67,    68,    69,    70,    71,    -1,
      73,    74,     1,    -1,     3,    -1,    -1,     6,    -1,     8,
       9,    10,    -1,    -1,    13,    -1,    15,    16,    17,    18,
      19,    20,    -1,    22,    23,    24,    34,    -1,    27,    28,
      29,    30,    31,    32,    -1,    -1,    44,    36,    -1,    -1,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    -1,
      49,    50,    -1,    52,    53,    -1,    -1,    56,    -1,    -1,
      59,    -1,    -1,    62,   619,   620,    -1,   622,    67,    68,
      69,    70,    71,     3,    73,    74,   631,    -1,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    17,    -1,     8,
      -1,    -1,    22,    23,    24,    -1,    -1,    -1,    17,    29,
      -1,    -1,    -1,    22,    23,    24,    36,    -1,    -1,    -1,
      29,    -1,    -1,    -1,    -1,    -1,    -1,    36,    -1,    49,
      50,    -1,    52,    53,    -1,    -1,    56,    -1,     3,    59,
      60,    -1,    -1,     8,    53,    -1,    -1,    -1,    -1,    69,
      70,    71,    17,    73,    74,    -1,    -1,    22,    23,    24,
      -1,    -1,    71,    -1,    29,    74,    -1,    -1,    -1,    -1,
      -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    50,    -1,    52,    53,    -1,
      -1,    56,    -1,     3,    59,    -1,    -1,    -1,     8,    -1,
      -1,    -1,    67,    -1,    69,    70,    71,    17,    73,    74,
      -1,    -1,    22,    23,    24,    -1,    -1,    -1,    -1,    29,
      -1,    31,    -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,
      50,    -1,    52,    53,    -1,    -1,    56,    -1,     3,    59,
      -1,    -1,    -1,     8,    -1,    -1,    -1,    -1,    -1,    69,
      70,    71,    17,    73,    74,    -1,    -1,    22,    23,    24,
      -1,    26,    -1,    -1,    29,    -1,    -1,    -1,    -1,    -1,
      -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    50,    -1,    52,    53,    -1,
      -1,    56,    -1,     3,    59,    -1,    -1,    -1,     8,    -1,
      -1,    -1,    -1,    -1,    69,    70,    71,    17,    73,    74,
      -1,    -1,    22,    23,    24,    -1,    26,    -1,    -1,    29,
      -1,    -1,    -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    49,
      50,     8,    52,    53,    -1,    -1,    56,    -1,    -1,    59,
      17,    -1,    -1,    -1,    -1,    22,    23,    24,    -1,    69,
      70,    71,    29,    73,    74,    -1,    -1,    -1,    -1,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,     3,    59,    -1,    -1,    -1,     8,    -1,    -1,    -1,
      67,    -1,    69,    70,    71,    17,    73,    74,    -1,    -1,
      22,    23,    24,    -1,    -1,    -1,    -1,    29,    -1,    -1,
      -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     3,    -1,    -1,    49,    50,     8,
      52,    53,    -1,    -1,    56,    -1,    -1,    59,    17,    -1,
      -1,    -1,    -1,    22,    23,    24,    -1,    69,    70,    71,
      29,    73,    74,    -1,    -1,    -1,    -1,    36,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,
      49,    50,     8,    52,    53,    -1,    -1,    56,    -1,    -1,
      59,    17,    -1,    -1,    -1,    -1,    22,    23,    24,    -1,
      69,    70,    71,    29,    73,    74,    -1,    -1,    -1,    -1,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    49,    50,     8,    52,    53,    -1,    -1,
      56,    -1,    -1,    59,    17,    -1,    -1,    -1,    -1,    22,
      23,    24,    -1,    69,    70,    71,    29,    73,    74,    -1,
       8,    -1,    -1,    36,    -1,    -1,    -1,    -1,     8,    17,
      -1,    -1,    -1,    -1,    22,    23,    24,    17,    -1,    -1,
      53,    29,    22,    23,    24,    -1,    59,    -1,    36,    29,
      -1,    -1,    -1,    -1,     8,    -1,    36,    -1,    71,    -1,
      73,    74,    75,    17,    -1,    53,    -1,    -1,    22,    23,
      24,    59,    -1,    53,    -1,    29,    -1,    -1,    -1,    59,
      -1,     8,    36,    71,    -1,    73,    74,    -1,    -1,    -1,
      17,    71,    -1,    73,    74,    22,    23,    24,    -1,    53,
      -1,    -1,    29,    -1,    -1,    59,    -1,     8,    -1,    36,
      -1,    -1,    -1,    -1,    -1,     8,    17,    71,    -1,    73,
      74,    22,    23,    24,    17,    -1,    53,    -1,    29,    22,
      23,    24,    59,    -1,    -1,    36,    29,    -1,    -1,    -1,
       8,    -1,    -1,    36,    71,    -1,    73,    74,     8,    17,
      -1,    11,    53,    -1,    22,    23,    24,    17,    59,    -1,
      53,    29,    22,    23,    24,    -1,    59,    -1,    36,    29,
      71,    -1,    73,    74,     8,    -1,    36,    -1,    71,    -1,
      -1,    74,    -1,    17,    -1,    53,    -1,    -1,    22,    23,
      24,    -1,    -1,    53,    -1,    29,    -1,    -1,    -1,    -1,
      -1,    -1,    36,    71,    -1,    73,    74,    -1,    -1,    -1,
      -1,    71,    34,    -1,    74,    -1,    38,    39,    40,    53,
      -1,    43,    44,    45,    46,    59,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    -1,    -1,    71,    33,    34,
      74,    36,    -1,    38,    39,    40,    -1,    -1,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    33,    34,    -1,    36,    -1,    38,    39,    40,    -1,
      75,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    -1,    -1,    -1,    -1,    -1,
      -1,    33,    34,    -1,    36,    -1,    38,    39,    40,    -1,
      72,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    33,    34,    -1,    36,    -1,
      38,    39,    40,    -1,    66,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    33,
      34,    -1,    36,    -1,    38,    39,    40,    -1,    66,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    33,    34,    -1,    36,    -1,    38,    39,
      40,    -1,    66,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    33,    34,    -1,
      -1,    -1,    38,    39,    40,    -1,    -1,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    33,    34,    -1,    -1,    -1,    38,    39,    40,    -1,
      -1,    43,    44,    45,    46,    -1,    48,    49,    50,    51,
      52,    53,    54,    55,    56
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    77,    79,    80,     0,    25,    78,    25,    86,    24,
      73,    74,   141,   142,    81,    24,    88,    89,     3,    62,
      21,    82,   166,    24,    87,   214,    63,     3,    59,    63,
      83,    85,   141,    62,     1,     3,     6,     8,     9,    10,
      13,    15,    16,    17,    18,    19,    20,    22,    23,    27,
      28,    29,    30,    31,    32,    36,    49,    50,    52,    53,
      56,    59,    69,    70,    71,    90,    91,    92,    98,   110,
     113,   121,   124,   126,   127,   128,   129,   134,   138,   141,
     143,   144,   149,   150,   153,   156,   157,   158,   161,   164,
     165,   181,   186,    62,     9,    17,    21,    31,    32,    64,
     199,    24,    60,    83,    84,     3,    86,    88,     3,   138,
     140,   141,    17,    36,    53,    59,   141,   143,   148,   152,
     153,   154,   161,   140,   128,   134,   111,    59,   141,   159,
     128,   138,   114,    35,    67,   137,    71,   126,   186,   193,
     125,   137,   122,    59,    96,    97,   141,    59,    93,   139,
     141,   185,   127,   127,   127,   127,   127,   127,    36,    53,
     126,   135,   147,   153,   155,   161,   127,   127,    11,   126,
     192,    62,    59,    94,   185,     4,    33,    34,    36,    37,
      38,    39,    40,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    67,    59,
      63,    71,    66,    59,   137,     1,   137,     5,    65,    75,
     142,   200,    59,   160,   200,    24,   200,   201,   200,    64,
      62,   190,    88,    59,    36,    59,   146,   152,   153,   154,
     155,   161,   146,   146,    63,    98,   107,   108,   109,   186,
     194,    11,   136,   141,   145,   146,   177,   178,   179,    59,
      67,   162,   112,   194,    24,    59,    68,   138,   171,   173,
     175,   146,    35,    53,    59,    68,   138,   170,   172,   173,
     174,   184,   112,    60,    97,   169,   146,    60,    93,   167,
      65,    75,   146,     8,   147,    60,    72,    72,    60,    94,
      65,   146,   126,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   130,    60,   135,   187,    59,   141,   126,
     192,   182,   126,   130,     1,    67,    91,   100,   180,   181,
     183,   186,   186,   126,     8,    17,    22,    23,    24,    29,
      36,    53,    65,    71,   142,   202,   204,   205,   206,   141,
     207,   215,   162,    59,     3,   202,   202,    83,    60,   179,
       8,   146,    60,   141,    35,   105,     5,    65,    62,   146,
     136,   145,    75,   191,    60,   179,   183,   115,    62,    63,
      24,   173,    59,   176,    62,   190,    72,   104,    59,   174,
      53,   174,    62,   190,     3,   198,    75,   146,   123,    62,
     190,    62,   190,   186,   139,    65,    36,    59,   146,   152,
     153,   154,   161,    67,   146,   146,    62,   190,   186,    65,
      67,   126,   131,   132,   188,   189,    11,    75,   191,    31,
     135,    72,    66,   180,    75,   191,   189,   101,    62,    68,
      36,    59,   203,   204,   206,    59,    67,    71,    67,     8,
     202,     3,    50,    59,   141,   212,   213,     3,    72,    65,
      11,   202,    60,    75,    62,   195,   215,    62,    62,    62,
      60,    60,   106,    26,    26,   194,   177,    59,   141,   151,
     152,   153,   154,   155,   161,   163,    60,    68,   105,   194,
     141,    60,   179,   175,    68,   146,     7,    12,    68,    99,
     102,   174,   198,   174,    60,   172,    68,   138,   198,    35,
      97,    60,    93,    60,   186,   146,   130,    94,    95,   168,
     185,    60,   186,   130,    66,    75,   191,    68,   191,   135,
      60,    60,    60,   192,    60,    68,   183,   180,   202,   205,
     195,    24,   141,   142,   197,   202,   209,   217,   202,   141,
     196,   208,   216,   202,     3,   212,    62,    72,   202,   213,
     202,   198,   141,   207,    60,   183,   126,   126,    62,   179,
      59,   163,   116,    60,   187,    66,   103,    60,    60,   198,
     104,    60,   189,    62,   190,   146,   189,    67,   126,   133,
     131,   132,    60,    72,    68,    60,    60,    59,    68,    62,
      72,   202,    68,    62,    49,   202,    62,   198,    59,    59,
     202,   210,   211,    68,   194,    60,   179,   119,   163,     5,
      65,    66,    75,   183,   198,   198,    68,    68,    95,    60,
      68,   130,   210,   195,   209,   202,   198,   208,   212,   195,
     195,    60,    14,   117,   120,   126,   126,   189,    60,    60,
      60,    60,   163,    20,   100,    66,    66,    68,   210,   210,
     118,   112,   105
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
int yychar, yystate;

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
        case 2:
#line 128 "go.y"
    {
		xtop = concat(xtop, (yyvsp[(4) - (4)].list));
	}
    break;

  case 3:
#line 134 "go.y"
    {
		prevlineno = lineno;
		yyerror("package statement must be first");
		flusherrors();
		mkpackage("main");
	}
    break;

  case 4:
#line 141 "go.y"
    {
		mkpackage((yyvsp[(2) - (3)].sym)->name);
	}
    break;

  case 5:
#line 151 "go.y"
    {
		importpkg = runtimepkg;

		if(debug['A'])
			cannedimports("runtime.builtin", "package runtime\n\n$$\n\n");
		else
			cannedimports("runtime.builtin", runtimeimport);
		curio.importsafe = 1;
	}
    break;

  case 6:
#line 162 "go.y"
    {
		importpkg = nil;
	}
    break;

  case 12:
#line 176 "go.y"
    {
		Pkg *ipkg;
		Sym *my;
		Node *pack;
		
		ipkg = importpkg;
		my = importmyname;
		importpkg = nil;
		importmyname = S;

		if(my == nil)
			my = lookup(ipkg->name);

		pack = nod(OPACK, N, N);
		pack->sym = my;
		pack->pkg = ipkg;
		pack->lineno = (yyvsp[(1) - (3)].i);

		if(my->name[0] == '.') {
			importdot(ipkg, pack);
			break;
		}
		if(my->name[0] == '_' && my->name[1] == '\0')
			break;
		if(my->def) {
			lineno = (yyvsp[(1) - (3)].i);
			redeclare(my, "as imported package name");
		}
		my->def = pack;
		my->lastlineno = (yyvsp[(1) - (3)].i);
		my->block = 1;	// at top level
	}
    break;

  case 13:
#line 209 "go.y"
    {
		// When an invalid import path is passed to importfile,
		// it calls yyerror and then sets up a fake import with
		// no package statement. This allows us to test more
		// than one invalid import statement in a single file.
		if(nerrors == 0)
			fatal("phase error in import");
	}
    break;

  case 16:
#line 224 "go.y"
    {
		// import with original name
		(yyval.i) = parserline();
		importmyname = S;
		importfile(&(yyvsp[(1) - (1)].val), (yyval.i));
	}
    break;

  case 17:
#line 231 "go.y"
    {
		// import with given name
		(yyval.i) = parserline();
		importmyname = (yyvsp[(1) - (2)].sym);
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 18:
#line 238 "go.y"
    {
		// import into my name space
		(yyval.i) = parserline();
		importmyname = lookup(".");
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 19:
#line 247 "go.y"
    {
		if(importpkg->name == nil) {
			importpkg->name = (yyvsp[(2) - (4)].sym)->name;
			pkglookup((yyvsp[(2) - (4)].sym)->name, nil)->npkg++;
		} else if(strcmp(importpkg->name, (yyvsp[(2) - (4)].sym)->name) != 0)
			yyerror("conflicting names %s and %s for package \"%Z\"", importpkg->name, (yyvsp[(2) - (4)].sym)->name, importpkg->path);
		importpkg->direct = 1;
		importpkg->safe = curio.importsafe;

		if(safemode && !curio.importsafe)
			yyerror("cannot import unsafe package \"%Z\"", importpkg->path);
	}
    break;

  case 21:
#line 262 "go.y"
    {
		if(strcmp((yyvsp[(1) - (1)].sym)->name, "safe") == 0)
			curio.importsafe = 1;
	}
    break;

  case 22:
#line 268 "go.y"
    {
		defercheckwidth();
	}
    break;

  case 23:
#line 272 "go.y"
    {
		resumecheckwidth();
		unimportfile();
	}
    break;

  case 24:
#line 281 "go.y"
    {
		yyerror("empty top-level declaration");
		(yyval.list) = nil;
	}
    break;

  case 26:
#line 287 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 27:
#line 291 "go.y"
    {
		yyerror("non-declaration statement outside function body");
		(yyval.list) = nil;
	}
    break;

  case 28:
#line 296 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 29:
#line 302 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
	}
    break;

  case 30:
#line 306 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 31:
#line 310 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 32:
#line 314 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 33:
#line 320 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 34:
#line 326 "go.y"
    {
		(yyval.list) = concat((yyvsp[(3) - (7)].list), (yyvsp[(5) - (7)].list));
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 35:
#line 332 "go.y"
    {
		(yyval.list) = nil;
		iota = -100000;
	}
    break;

  case 36:
#line 337 "go.y"
    {
		(yyval.list) = list1((yyvsp[(2) - (2)].node));
	}
    break;

  case 37:
#line 341 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 38:
#line 345 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 39:
#line 351 "go.y"
    {
		iota = 0;
	}
    break;

  case 40:
#line 357 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 41:
#line 361 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 42:
#line 365 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (3)].list), nil, (yyvsp[(3) - (3)].list));
	}
    break;

  case 43:
#line 371 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 44:
#line 375 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (3)].list), N, (yyvsp[(3) - (3)].list));
	}
    break;

  case 46:
#line 382 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 47:
#line 386 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (1)].list), N, nil);
	}
    break;

  case 48:
#line 392 "go.y"
    {
		// different from dclname because the name
		// becomes visible right here, not at the end
		// of the declaration.
		(yyval.node) = typedcl0((yyvsp[(1) - (1)].sym));
	}
    break;

  case 49:
#line 401 "go.y"
    {
		(yyval.node) = typedcl1((yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node), 1);
	}
    break;

  case 50:
#line 407 "go.y"
    {
		(yyval.node) = (yyvsp[(1) - (1)].node);

		// These nodes do not carry line numbers.
		// Since a bare name used as an expression is an error,
		// introduce a wrapper node to give the correct line.
		switch((yyval.node)->op) {
		case ONAME:
		case ONONAME:
		case OTYPE:
		case OPACK:
		case OLITERAL:
			(yyval.node) = nod(OPAREN, (yyval.node), N);
			(yyval.node)->implicit = 1;
			break;
		}
	}
    break;

  case 51:
#line 425 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
		(yyval.node)->etype = (yyvsp[(2) - (3)].i);			// rathole to pass opcode
	}
    break;

  case 52:
#line 430 "go.y"
    {
		if((yyvsp[(1) - (3)].list)->next == nil && (yyvsp[(3) - (3)].list)->next == nil) {
			// simple
			(yyval.node) = nod(OAS, (yyvsp[(1) - (3)].list)->n, (yyvsp[(3) - (3)].list)->n);
			break;
		}
		// multiple
		(yyval.node) = nod(OAS2, N, N);
		(yyval.node)->list = (yyvsp[(1) - (3)].list);
		(yyval.node)->rlist = (yyvsp[(3) - (3)].list);
	}
    break;

  case 53:
#line 442 "go.y"
    {
		if((yyvsp[(3) - (3)].list)->n->op == OTYPESW) {
			(yyval.node) = nod(OTYPESW, N, (yyvsp[(3) - (3)].list)->n->right);
			if((yyvsp[(3) - (3)].list)->next != nil)
				yyerror("expr.(type) must be alone in list");
			if((yyvsp[(1) - (3)].list)->next != nil)
				yyerror("argument count mismatch: %d = %d", count((yyvsp[(1) - (3)].list)), 1);
			else if(((yyvsp[(1) - (3)].list)->n->op != ONAME && (yyvsp[(1) - (3)].list)->n->op != OTYPE && (yyvsp[(1) - (3)].list)->n->op != ONONAME) || isblank((yyvsp[(1) - (3)].list)->n))
				yyerror("invalid variable name %N in type switch", (yyvsp[(1) - (3)].list)->n);
			else
				(yyval.node)->left = dclname((yyvsp[(1) - (3)].list)->n->sym);  // it's a colas, so must not re-use an oldname.
			break;
		}
		(yyval.node) = colas((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list), (yyvsp[(2) - (3)].i));
	}
    break;

  case 54:
#line 458 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OADD;
	}
    break;

  case 55:
#line 463 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OSUB;
	}
    break;

  case 56:
#line 470 "go.y"
    {
		Node *n, *nn;

		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		(yyval.node) = nod(OXCASE, N, N);
		(yyval.node)->list = (yyvsp[(2) - (3)].list);
		if(typesw != N && typesw->right != N && (n=typesw->right->left) != N) {
			// type switch - declare variable
			nn = newname(n->sym);
			declare(nn, dclcontext);
			(yyval.node)->nname = nn;

			// keep track of the instances for reporting unused
			nn->defn = typesw->right;
		}
	}
    break;

  case 57:
#line 490 "go.y"
    {
		Node *n;

		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		(yyval.node) = nod(OXCASE, N, N);
		if((yyvsp[(2) - (5)].list)->next == nil)
			n = nod(OAS, (yyvsp[(2) - (5)].list)->n, (yyvsp[(4) - (5)].node));
		else {
			n = nod(OAS2, N, N);
			n->list = (yyvsp[(2) - (5)].list);
			n->rlist = list1((yyvsp[(4) - (5)].node));
		}
		(yyval.node)->list = list1(n);
	}
    break;

  case 58:
#line 508 "go.y"
    {
		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		(yyval.node) = nod(OXCASE, N, N);
		(yyval.node)->list = list1(colas((yyvsp[(2) - (5)].list), list1((yyvsp[(4) - (5)].node)), (yyvsp[(3) - (5)].i)));
	}
    break;

  case 59:
#line 517 "go.y"
    {
		Node *n, *nn;

		markdcl();
		(yyval.node) = nod(OXCASE, N, N);
		if(typesw != N && typesw->right != N && (n=typesw->right->left) != N) {
			// type switch - declare variable
			nn = newname(n->sym);
			declare(nn, dclcontext);
			(yyval.node)->nname = nn;

			// keep track of the instances for reporting unused
			nn->defn = typesw->right;
		}
	}
    break;

  case 60:
#line 535 "go.y"
    {
		markdcl();
	}
    break;

  case 61:
#line 539 "go.y"
    {
		if((yyvsp[(3) - (4)].list) == nil)
			(yyval.node) = nod(OEMPTY, N, N);
		else
			(yyval.node) = liststmt((yyvsp[(3) - (4)].list));
		popdcl();
	}
    break;

  case 62:
#line 549 "go.y"
    {
		// If the last token read by the lexer was consumed
		// as part of the case, clear it (parser has cleared yychar).
		// If the last token read by the lexer was the lookahead
		// leave it alone (parser has it cached in yychar).
		// This is so that the stmt_list action doesn't look at
		// the case tokens if the stmt_list is empty.
		yylast = yychar;
	}
    break;

  case 63:
#line 559 "go.y"
    {
		int last;

		// This is the only place in the language where a statement
		// list is not allowed to drop the final semicolon, because
		// it's the only place where a statement list is not followed 
		// by a closing brace.  Handle the error for pedantry.

		// Find the final token of the statement list.
		// yylast is lookahead; yyprev is last of stmt_list
		last = yyprev;

		if(last > 0 && last != ';' && yychar != '}')
			yyerror("missing statement after label");
		(yyval.node) = (yyvsp[(1) - (3)].node);
		(yyval.node)->nbody = (yyvsp[(3) - (3)].list);
		popdcl();
	}
    break;

  case 64:
#line 579 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 65:
#line 583 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node));
	}
    break;

  case 66:
#line 589 "go.y"
    {
		markdcl();
	}
    break;

  case 67:
#line 593 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (4)].list);
		popdcl();
	}
    break;

  case 68:
#line 600 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->etype = 0;	// := flag
	}
    break;

  case 69:
#line 606 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->colas = 1;
		colasdefn((yyvsp[(1) - (4)].list), (yyval.node));
	}
    break;

  case 70:
#line 615 "go.y"
    {
		// init ; test ; incr
		if((yyvsp[(5) - (5)].node) != N && (yyvsp[(5) - (5)].node)->colas != 0)
			yyerror("cannot declare in the for-increment");
		(yyval.node) = nod(OFOR, N, N);
		if((yyvsp[(1) - (5)].node) != N)
			(yyval.node)->ninit = list1((yyvsp[(1) - (5)].node));
		(yyval.node)->ntest = (yyvsp[(3) - (5)].node);
		(yyval.node)->nincr = (yyvsp[(5) - (5)].node);
	}
    break;

  case 71:
#line 626 "go.y"
    {
		// normal test
		(yyval.node) = nod(OFOR, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 73:
#line 635 "go.y"
    {
		(yyval.node) = (yyvsp[(1) - (2)].node);
		(yyval.node)->nbody = concat((yyval.node)->nbody, (yyvsp[(2) - (2)].list));
	}
    break;

  case 74:
#line 642 "go.y"
    {
		markdcl();
	}
    break;

  case 75:
#line 646 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (3)].node);
		popdcl();
	}
    break;

  case 76:
#line 653 "go.y"
    {
		// test
		(yyval.node) = nod(OIF, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 77:
#line 659 "go.y"
    {
		// init ; test
		(yyval.node) = nod(OIF, N, N);
		if((yyvsp[(1) - (3)].node) != N)
			(yyval.node)->ninit = list1((yyvsp[(1) - (3)].node));
		(yyval.node)->ntest = (yyvsp[(3) - (3)].node);
	}
    break;

  case 78:
#line 670 "go.y"
    {
		markdcl();
	}
    break;

  case 79:
#line 674 "go.y"
    {
		if((yyvsp[(3) - (3)].node)->ntest == N)
			yyerror("missing condition in if statement");
	}
    break;

  case 80:
#line 679 "go.y"
    {
		(yyvsp[(3) - (5)].node)->nbody = (yyvsp[(5) - (5)].list);
	}
    break;

  case 81:
#line 683 "go.y"
    {
		Node *n;
		NodeList *nn;

		(yyval.node) = (yyvsp[(3) - (8)].node);
		n = (yyvsp[(3) - (8)].node);
		popdcl();
		for(nn = concat((yyvsp[(7) - (8)].list), (yyvsp[(8) - (8)].list)); nn; nn = nn->next) {
			if(nn->n->op == OIF)
				popdcl();
			n->nelse = list1(nn->n);
			n = nn->n;
		}
	}
    break;

  case 82:
#line 700 "go.y"
    {
		markdcl();
	}
    break;

  case 83:
#line 704 "go.y"
    {
		if((yyvsp[(4) - (5)].node)->ntest == N)
			yyerror("missing condition in if statement");
		(yyvsp[(4) - (5)].node)->nbody = (yyvsp[(5) - (5)].list);
		(yyval.list) = list1((yyvsp[(4) - (5)].node));
	}
    break;

  case 84:
#line 712 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 85:
#line 716 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].list));
	}
    break;

  case 86:
#line 721 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 87:
#line 725 "go.y"
    {
		NodeList *node;
		
		node = mal(sizeof *node);
		node->n = (yyvsp[(2) - (2)].node);
		node->end = node;
		(yyval.list) = node;
	}
    break;

  case 88:
#line 736 "go.y"
    {
		markdcl();
	}
    break;

  case 89:
#line 740 "go.y"
    {
		Node *n;
		n = (yyvsp[(3) - (3)].node)->ntest;
		if(n != N && n->op != OTYPESW)
			n = N;
		typesw = nod(OXXX, typesw, n);
	}
    break;

  case 90:
#line 748 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (7)].node);
		(yyval.node)->op = OSWITCH;
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
		typesw = typesw->left;
		popdcl();
	}
    break;

  case 91:
#line 758 "go.y"
    {
		typesw = nod(OXXX, typesw, N);
	}
    break;

  case 92:
#line 762 "go.y"
    {
		(yyval.node) = nod(OSELECT, N, N);
		(yyval.node)->lineno = typesw->lineno;
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		typesw = typesw->left;
	}
    break;

  case 94:
#line 775 "go.y"
    {
		(yyval.node) = nod(OOROR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 95:
#line 779 "go.y"
    {
		(yyval.node) = nod(OANDAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 96:
#line 783 "go.y"
    {
		(yyval.node) = nod(OEQ, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 97:
#line 787 "go.y"
    {
		(yyval.node) = nod(ONE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 98:
#line 791 "go.y"
    {
		(yyval.node) = nod(OLT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 99:
#line 795 "go.y"
    {
		(yyval.node) = nod(OLE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 100:
#line 799 "go.y"
    {
		(yyval.node) = nod(OGE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 101:
#line 803 "go.y"
    {
		(yyval.node) = nod(OGT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 102:
#line 807 "go.y"
    {
		(yyval.node) = nod(OADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 103:
#line 811 "go.y"
    {
		(yyval.node) = nod(OSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 104:
#line 815 "go.y"
    {
		(yyval.node) = nod(OOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 105:
#line 819 "go.y"
    {
		(yyval.node) = nod(OXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 106:
#line 823 "go.y"
    {
		(yyval.node) = nod(OMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 107:
#line 827 "go.y"
    {
		(yyval.node) = nod(ODIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 108:
#line 831 "go.y"
    {
		(yyval.node) = nod(OMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 109:
#line 835 "go.y"
    {
		(yyval.node) = nod(OAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 110:
#line 839 "go.y"
    {
		(yyval.node) = nod(OANDNOT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 111:
#line 843 "go.y"
    {
		(yyval.node) = nod(OLSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 112:
#line 847 "go.y"
    {
		(yyval.node) = nod(ORSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 113:
#line 852 "go.y"
    {
		(yyval.node) = nod(OSEND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 115:
#line 859 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 116:
#line 863 "go.y"
    {
		if((yyvsp[(2) - (2)].node)->op == OCOMPLIT) {
			// Special case for &T{...}: turn into (*T){...}.
			(yyval.node) = (yyvsp[(2) - (2)].node);
			(yyval.node)->right = nod(OIND, (yyval.node)->right, N);
			(yyval.node)->right->implicit = 1;
		} else {
			(yyval.node) = nod(OADDR, (yyvsp[(2) - (2)].node), N);
		}
	}
    break;

  case 117:
#line 874 "go.y"
    {
		(yyval.node) = nod(OPLUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 118:
#line 878 "go.y"
    {
		(yyval.node) = nod(OMINUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 119:
#line 882 "go.y"
    {
		(yyval.node) = nod(ONOT, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 120:
#line 886 "go.y"
    {
		yyerror("the bitwise complement operator is ^");
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 121:
#line 891 "go.y"
    {
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 122:
#line 895 "go.y"
    {
		(yyval.node) = nod(ORECV, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 123:
#line 905 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (3)].node), N);
	}
    break;

  case 124:
#line 909 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (5)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
	}
    break;

  case 125:
#line 914 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (6)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (6)].list);
		(yyval.node)->isddd = 1;
	}
    break;

  case 126:
#line 922 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 128:
#line 927 "go.y"
    {
		if((yyvsp[(1) - (3)].node)->op == OPACK) {
			Sym *s;
			s = restrictlookup((yyvsp[(3) - (3)].sym)->name, (yyvsp[(1) - (3)].node)->pkg);
			(yyvsp[(1) - (3)].node)->used = 1;
			(yyval.node) = oldname(s);
			break;
		}
		(yyval.node) = nod(OXDOT, (yyvsp[(1) - (3)].node), newname((yyvsp[(3) - (3)].sym)));
	}
    break;

  case 129:
#line 938 "go.y"
    {
		(yyval.node) = nod(ODOTTYPE, (yyvsp[(1) - (5)].node), (yyvsp[(4) - (5)].node));
	}
    break;

  case 130:
#line 942 "go.y"
    {
		(yyval.node) = nod(OTYPESW, N, (yyvsp[(1) - (5)].node));
	}
    break;

  case 131:
#line 946 "go.y"
    {
		(yyval.node) = nod(OINDEX, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 132:
#line 950 "go.y"
    {
		(yyval.node) = nod(OSLICE, (yyvsp[(1) - (6)].node), nod(OKEY, (yyvsp[(3) - (6)].node), (yyvsp[(5) - (6)].node)));
	}
    break;

  case 134:
#line 955 "go.y"
    {
		// conversion
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (5)].node), N);
		(yyval.node)->list = list1((yyvsp[(3) - (5)].node));
	}
    break;

  case 135:
#line 961 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 136:
#line 968 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
	}
    break;

  case 137:
#line 974 "go.y"
    {
		yyerror("cannot parenthesize type in composite literal");
		(yyval.node) = (yyvsp[(5) - (7)].node);
		(yyval.node)->right = (yyvsp[(2) - (7)].node);
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
	}
    break;

  case 139:
#line 983 "go.y"
    {
		// composite expression.
		// make node early so we get the right line number.
		(yyval.node) = nod(OCOMPLIT, N, N);
	}
    break;

  case 140:
#line 991 "go.y"
    {
		(yyval.node) = nod(OKEY, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 141:
#line 997 "go.y"
    {
		// These nodes do not carry line numbers.
		// Since a composite literal commonly spans several lines,
		// the line number on errors may be misleading.
		// Introduce a wrapper node to give the correct line.
		(yyval.node) = (yyvsp[(1) - (1)].node);
		switch((yyval.node)->op) {
		case ONAME:
		case ONONAME:
		case OTYPE:
		case OPACK:
		case OLITERAL:
			(yyval.node) = nod(OPAREN, (yyval.node), N);
			(yyval.node)->implicit = 1;
		}
	}
    break;

  case 142:
#line 1014 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (4)].node);
		(yyval.node)->list = (yyvsp[(3) - (4)].list);
	}
    break;

  case 144:
#line 1022 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (4)].node);
		(yyval.node)->list = (yyvsp[(3) - (4)].list);
	}
    break;

  case 146:
#line 1030 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
		
		// Need to know on lhs of := whether there are ( ).
		// Don't bother with the OPAREN in other cases:
		// it's just a waste of memory and time.
		switch((yyval.node)->op) {
		case ONAME:
		case ONONAME:
		case OPACK:
		case OTYPE:
		case OLITERAL:
		case OTYPESW:
			(yyval.node) = nod(OPAREN, (yyval.node), N);
		}
	}
    break;

  case 150:
#line 1056 "go.y"
    {
		(yyval.i) = LBODY;
	}
    break;

  case 151:
#line 1060 "go.y"
    {
		(yyval.i) = '{';
	}
    break;

  case 152:
#line 1071 "go.y"
    {
		if((yyvsp[(1) - (1)].sym) == S)
			(yyval.node) = N;
		else
			(yyval.node) = newname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 153:
#line 1080 "go.y"
    {
		(yyval.node) = dclname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 154:
#line 1085 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 156:
#line 1092 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if(importpkg != nil && !exportname((yyvsp[(1) - (1)].sym)->name))
			(yyval.sym) = pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg);
	}
    break;

  case 158:
#line 1100 "go.y"
    {
		(yyval.sym) = S;
	}
    break;

  case 159:
#line 1106 "go.y"
    {
		Pkg *p;

		if((yyvsp[(2) - (4)].val).u.sval->len == 0)
			p = importpkg;
		else {
			if(isbadimport((yyvsp[(2) - (4)].val).u.sval))
				errorexit();
			p = mkpkg((yyvsp[(2) - (4)].val).u.sval);
		}
		(yyval.sym) = pkglookup((yyvsp[(4) - (4)].sym)->name, p);
	}
    break;

  case 160:
#line 1121 "go.y"
    {
		(yyval.node) = oldname((yyvsp[(1) - (1)].sym));
		if((yyval.node)->pack != N)
			(yyval.node)->pack->used = 1;
	}
    break;

  case 162:
#line 1141 "go.y"
    {
		yyerror("final argument in variadic function missing type");
		(yyval.node) = nod(ODDD, typenod(typ(TINTER)), N);
	}
    break;

  case 163:
#line 1146 "go.y"
    {
		(yyval.node) = nod(ODDD, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 169:
#line 1157 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 173:
#line 1166 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 178:
#line 1176 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 188:
#line 1197 "go.y"
    {
		if((yyvsp[(1) - (3)].node)->op == OPACK) {
			Sym *s;
			s = restrictlookup((yyvsp[(3) - (3)].sym)->name, (yyvsp[(1) - (3)].node)->pkg);
			(yyvsp[(1) - (3)].node)->used = 1;
			(yyval.node) = oldname(s);
			break;
		}
		(yyval.node) = nod(OXDOT, (yyvsp[(1) - (3)].node), newname((yyvsp[(3) - (3)].sym)));
	}
    break;

  case 189:
#line 1210 "go.y"
    {
		(yyval.node) = nod(OTARRAY, (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].node));
	}
    break;

  case 190:
#line 1214 "go.y"
    {
		// array literal of nelem
		(yyval.node) = nod(OTARRAY, nod(ODDD, N, N), (yyvsp[(4) - (4)].node));
	}
    break;

  case 191:
#line 1219 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->etype = Cboth;
	}
    break;

  case 192:
#line 1224 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Csend;
	}
    break;

  case 193:
#line 1229 "go.y"
    {
		(yyval.node) = nod(OTMAP, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node));
	}
    break;

  case 196:
#line 1237 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 197:
#line 1243 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Crecv;
	}
    break;

  case 198:
#line 1250 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 199:
#line 1256 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 200:
#line 1263 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 201:
#line 1269 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 202:
#line 1280 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
		if((yyval.node) == N)
			break;
		if(noescape && (yyvsp[(3) - (3)].list) != nil)
			yyerror("can only use //go:noescape with external func implementations");
		(yyval.node)->nbody = (yyvsp[(3) - (3)].list);
		(yyval.node)->endlineno = lineno;
		(yyval.node)->noescape = noescape;
		funcbody((yyval.node));
	}
    break;

  case 203:
#line 1294 "go.y"
    {
		Node *t;

		(yyval.node) = N;
		(yyvsp[(3) - (5)].list) = checkarglist((yyvsp[(3) - (5)].list), 1);

		if(strcmp((yyvsp[(1) - (5)].sym)->name, "init") == 0) {
			(yyvsp[(1) - (5)].sym) = renameinit();
			if((yyvsp[(3) - (5)].list) != nil || (yyvsp[(5) - (5)].list) != nil)
				yyerror("func init must have no arguments and no return values");
		}
		if(strcmp(localpkg->name, "main") == 0 && strcmp((yyvsp[(1) - (5)].sym)->name, "main") == 0) {
			if((yyvsp[(3) - (5)].list) != nil || (yyvsp[(5) - (5)].list) != nil)
				yyerror("func main must have no arguments and no return values");
		}

		t = nod(OTFUNC, N, N);
		t->list = (yyvsp[(3) - (5)].list);
		t->rlist = (yyvsp[(5) - (5)].list);

		(yyval.node) = nod(ODCLFUNC, N, N);
		(yyval.node)->nname = newname((yyvsp[(1) - (5)].sym));
		(yyval.node)->nname->defn = (yyval.node);
		(yyval.node)->nname->ntype = t;		// TODO: check if nname already has an ntype
		declare((yyval.node)->nname, PFUNC);

		funchdr((yyval.node));
	}
    break;

  case 204:
#line 1323 "go.y"
    {
		Node *rcvr, *t;

		(yyval.node) = N;
		(yyvsp[(2) - (8)].list) = checkarglist((yyvsp[(2) - (8)].list), 0);
		(yyvsp[(6) - (8)].list) = checkarglist((yyvsp[(6) - (8)].list), 1);

		if((yyvsp[(2) - (8)].list) == nil) {
			yyerror("method has no receiver");
			break;
		}
		if((yyvsp[(2) - (8)].list)->next != nil) {
			yyerror("method has multiple receivers");
			break;
		}
		rcvr = (yyvsp[(2) - (8)].list)->n;
		if(rcvr->op != ODCLFIELD) {
			yyerror("bad receiver in method");
			break;
		}
		if(rcvr->right->op == OTPAREN || (rcvr->right->op == OIND && rcvr->right->left->op == OTPAREN))
			yyerror("cannot parenthesize receiver type");

		t = nod(OTFUNC, rcvr, N);
		t->list = (yyvsp[(6) - (8)].list);
		t->rlist = (yyvsp[(8) - (8)].list);

		(yyval.node) = nod(ODCLFUNC, N, N);
		(yyval.node)->shortname = newname((yyvsp[(4) - (8)].sym));
		(yyval.node)->nname = methodname1((yyval.node)->shortname, rcvr->right);
		(yyval.node)->nname->defn = (yyval.node);
		(yyval.node)->nname->ntype = t;
		(yyval.node)->nname->nointerface = nointerface;
		declare((yyval.node)->nname, PFUNC);

		funchdr((yyval.node));
	}
    break;

  case 205:
#line 1363 "go.y"
    {
		Sym *s;
		Type *t;

		(yyval.node) = N;

		s = (yyvsp[(1) - (5)].sym);
		t = functype(N, (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list));

		importsym(s, ONAME);
		if(s->def != N && s->def->op == ONAME) {
			if(eqtype(t, s->def->type)) {
				dclcontext = PDISCARD;  // since we skip funchdr below
				break;
			}
			yyerror("inconsistent definition for func %S during import\n\t%T\n\t%T", s, s->def->type, t);
		}

		(yyval.node) = newname(s);
		(yyval.node)->type = t;
		declare((yyval.node), PFUNC);

		funchdr((yyval.node));
	}
    break;

  case 206:
#line 1388 "go.y"
    {
		(yyval.node) = methodname1(newname((yyvsp[(4) - (8)].sym)), (yyvsp[(2) - (8)].list)->n->right); 
		(yyval.node)->type = functype((yyvsp[(2) - (8)].list)->n, (yyvsp[(6) - (8)].list), (yyvsp[(8) - (8)].list));

		checkwidth((yyval.node)->type);
		addmethod((yyvsp[(4) - (8)].sym), (yyval.node)->type, 0, nointerface);
		nointerface = 0;
		funchdr((yyval.node));
		
		// inl.c's inlnode in on a dotmeth node expects to find the inlineable body as
		// (dotmeth's type)->nname->inl, and dotmeth's type has been pulled
		// out by typecheck's lookdot as this $$->ttype.  So by providing
		// this back link here we avoid special casing there.
		(yyval.node)->type->nname = (yyval.node);
	}
    break;

  case 207:
#line 1406 "go.y"
    {
		(yyvsp[(3) - (5)].list) = checkarglist((yyvsp[(3) - (5)].list), 1);
		(yyval.node) = nod(OTFUNC, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		(yyval.node)->rlist = (yyvsp[(5) - (5)].list);
	}
    break;

  case 208:
#line 1414 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 209:
#line 1418 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
		if((yyval.list) == nil)
			(yyval.list) = list1(nod(OEMPTY, N, N));
	}
    break;

  case 210:
#line 1426 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 211:
#line 1430 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, (yyvsp[(1) - (1)].node)));
	}
    break;

  case 212:
#line 1434 "go.y"
    {
		(yyvsp[(2) - (3)].list) = checkarglist((yyvsp[(2) - (3)].list), 0);
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 213:
#line 1441 "go.y"
    {
		closurehdr((yyvsp[(1) - (1)].node));
	}
    break;

  case 214:
#line 1447 "go.y"
    {
		(yyval.node) = closurebody((yyvsp[(3) - (4)].list));
		fixlbrace((yyvsp[(2) - (4)].i));
	}
    break;

  case 215:
#line 1452 "go.y"
    {
		(yyval.node) = closurebody(nil);
	}
    break;

  case 216:
#line 1463 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 217:
#line 1467 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(2) - (3)].list));
		if(nsyntaxerrors == 0)
			testdclstack();
		nointerface = 0;
		noescape = 0;
	}
    break;

  case 219:
#line 1478 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 221:
#line 1485 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 222:
#line 1491 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 223:
#line 1495 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 225:
#line 1502 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 226:
#line 1508 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 227:
#line 1512 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 228:
#line 1518 "go.y"
    {
		NodeList *l;

		Node *n;
		l = (yyvsp[(1) - (3)].list);
		if(l != nil && l->next == nil && l->n == nil) {
			// ? symbol, during import
			n = (yyvsp[(2) - (3)].node);
			if(n->op == OIND)
				n = n->left;
			n = embedded(n->sym);
			n->right = (yyvsp[(2) - (3)].node);
			n->val = (yyvsp[(3) - (3)].val);
			(yyval.list) = list1(n);
			break;
		}

		for(l=(yyvsp[(1) - (3)].list); l; l=l->next) {
			l->n = nod(ODCLFIELD, l->n, (yyvsp[(2) - (3)].node));
			l->n->val = (yyvsp[(3) - (3)].val);
		}
	}
    break;

  case 229:
#line 1541 "go.y"
    {
		(yyvsp[(1) - (2)].node)->val = (yyvsp[(2) - (2)].val);
		(yyval.list) = list1((yyvsp[(1) - (2)].node));
	}
    break;

  case 230:
#line 1546 "go.y"
    {
		(yyvsp[(2) - (4)].node)->val = (yyvsp[(4) - (4)].val);
		(yyval.list) = list1((yyvsp[(2) - (4)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 231:
#line 1552 "go.y"
    {
		(yyvsp[(2) - (3)].node)->right = nod(OIND, (yyvsp[(2) - (3)].node)->right, N);
		(yyvsp[(2) - (3)].node)->val = (yyvsp[(3) - (3)].val);
		(yyval.list) = list1((yyvsp[(2) - (3)].node));
	}
    break;

  case 232:
#line 1558 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 233:
#line 1565 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 234:
#line 1574 "go.y"
    {
		Node *n;

		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		n = oldname((yyvsp[(1) - (1)].sym));
		if(n->pack != N)
			n->pack->used = 1;
	}
    break;

  case 235:
#line 1583 "go.y"
    {
		Pkg *pkg;

		if((yyvsp[(1) - (3)].sym)->def == N || (yyvsp[(1) - (3)].sym)->def->op != OPACK) {
			yyerror("%S is not a package", (yyvsp[(1) - (3)].sym));
			pkg = localpkg;
		} else {
			(yyvsp[(1) - (3)].sym)->def->used = 1;
			pkg = (yyvsp[(1) - (3)].sym)->def->pkg;
		}
		(yyval.sym) = restrictlookup((yyvsp[(3) - (3)].sym)->name, pkg);
	}
    break;

  case 236:
#line 1598 "go.y"
    {
		(yyval.node) = embedded((yyvsp[(1) - (1)].sym));
	}
    break;

  case 237:
#line 1604 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
		ifacedcl((yyval.node));
	}
    break;

  case 238:
#line 1609 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(1) - (1)].sym)));
	}
    break;

  case 239:
#line 1613 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(2) - (3)].sym)));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 240:
#line 1620 "go.y"
    {
		// without func keyword
		(yyvsp[(2) - (4)].list) = checkarglist((yyvsp[(2) - (4)].list), 1);
		(yyval.node) = nod(OTFUNC, fakethis(), N);
		(yyval.node)->list = (yyvsp[(2) - (4)].list);
		(yyval.node)->rlist = (yyvsp[(4) - (4)].list);
	}
    break;

  case 242:
#line 1634 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 243:
#line 1640 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 245:
#line 1649 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 246:
#line 1653 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 247:
#line 1658 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 248:
#line 1662 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 249:
#line 1670 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 251:
#line 1675 "go.y"
    {
		(yyval.node) = liststmt((yyvsp[(1) - (1)].list));
	}
    break;

  case 253:
#line 1680 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 259:
#line 1691 "go.y"
    {
		(yyvsp[(1) - (2)].node) = nod(OLABEL, (yyvsp[(1) - (2)].node), N);
		(yyvsp[(1) - (2)].node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 260:
#line 1696 "go.y"
    {
		NodeList *l;

		(yyvsp[(1) - (4)].node)->defn = (yyvsp[(4) - (4)].node);
		l = list1((yyvsp[(1) - (4)].node));
		if((yyvsp[(4) - (4)].node))
			l = list(l, (yyvsp[(4) - (4)].node));
		(yyval.node) = liststmt(l);
	}
    break;

  case 261:
#line 1706 "go.y"
    {
		// will be converted to OFALL
		(yyval.node) = nod(OXFALL, N, N);
	}
    break;

  case 262:
#line 1711 "go.y"
    {
		(yyval.node) = nod(OBREAK, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 263:
#line 1715 "go.y"
    {
		(yyval.node) = nod(OCONTINUE, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 264:
#line 1719 "go.y"
    {
		(yyval.node) = nod(OPROC, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 265:
#line 1723 "go.y"
    {
		(yyval.node) = nod(ODEFER, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 266:
#line 1727 "go.y"
    {
		(yyval.node) = nod(OGOTO, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 267:
#line 1732 "go.y"
    {
		(yyval.node) = nod(ORETURN, N, N);
		(yyval.node)->list = (yyvsp[(2) - (2)].list);
		if((yyval.node)->list == nil && curfn != N) {
			NodeList *l;

			for(l=curfn->dcl; l; l=l->next) {
				if(l->n->class == PPARAM)
					continue;
				if(l->n->class != PPARAMOUT)
					break;
				if(l->n->sym->def != l->n)
					yyerror("%s is shadowed during return", l->n->sym->name);
			}
		}
	}
    break;

  case 268:
#line 1751 "go.y"
    {
		(yyval.list) = nil;
		if((yyvsp[(1) - (1)].node) != N)
			(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 269:
#line 1757 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (3)].list);
		if((yyvsp[(3) - (3)].node) != N)
			(yyval.list) = list((yyval.list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 270:
#line 1765 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 271:
#line 1769 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 272:
#line 1775 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 273:
#line 1779 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 274:
#line 1785 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 275:
#line 1789 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 276:
#line 1795 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 277:
#line 1799 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 278:
#line 1808 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 279:
#line 1812 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 280:
#line 1816 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 281:
#line 1820 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 282:
#line 1825 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 283:
#line 1829 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 288:
#line 1843 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 290:
#line 1849 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 292:
#line 1855 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 294:
#line 1861 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 296:
#line 1867 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 298:
#line 1873 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 300:
#line 1879 "go.y"
    {
		(yyval.val).ctype = CTxxx;
	}
    break;

  case 302:
#line 1889 "go.y"
    {
		importimport((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].val).u.sval);
	}
    break;

  case 303:
#line 1893 "go.y"
    {
		importvar((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].type));
	}
    break;

  case 304:
#line 1897 "go.y"
    {
		importconst((yyvsp[(2) - (5)].sym), types[TIDEAL], (yyvsp[(4) - (5)].node));
	}
    break;

  case 305:
#line 1901 "go.y"
    {
		importconst((yyvsp[(2) - (6)].sym), (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].node));
	}
    break;

  case 306:
#line 1905 "go.y"
    {
		importtype((yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].type));
	}
    break;

  case 307:
#line 1909 "go.y"
    {
		if((yyvsp[(2) - (4)].node) == N) {
			dclcontext = PEXTERN;  // since we skip the funcbody below
			break;
		}

		(yyvsp[(2) - (4)].node)->inl = (yyvsp[(3) - (4)].list);

		funcbody((yyvsp[(2) - (4)].node));
		importlist = list(importlist, (yyvsp[(2) - (4)].node));

		if(debug['E']) {
			print("import [%Z] func %lN \n", importpkg->path, (yyvsp[(2) - (4)].node));
			if(debug['m'] > 2 && (yyvsp[(2) - (4)].node)->inl)
				print("inl body:%+H\n", (yyvsp[(2) - (4)].node)->inl);
		}
	}
    break;

  case 308:
#line 1929 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		structpkg = (yyval.sym)->pkg;
	}
    break;

  case 309:
#line 1936 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
		importsym((yyvsp[(1) - (1)].sym), OTYPE);
	}
    break;

  case 315:
#line 1956 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
	}
    break;

  case 316:
#line 1960 "go.y"
    {
		// predefined name like uint8
		(yyvsp[(1) - (1)].sym) = pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg);
		if((yyvsp[(1) - (1)].sym)->def == N || (yyvsp[(1) - (1)].sym)->def->op != OTYPE) {
			yyerror("%s is not a type", (yyvsp[(1) - (1)].sym)->name);
			(yyval.type) = T;
		} else
			(yyval.type) = (yyvsp[(1) - (1)].sym)->def->type;
	}
    break;

  case 317:
#line 1970 "go.y"
    {
		(yyval.type) = aindex(N, (yyvsp[(3) - (3)].type));
	}
    break;

  case 318:
#line 1974 "go.y"
    {
		(yyval.type) = aindex(nodlit((yyvsp[(2) - (4)].val)), (yyvsp[(4) - (4)].type));
	}
    break;

  case 319:
#line 1978 "go.y"
    {
		(yyval.type) = maptype((yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].type));
	}
    break;

  case 320:
#line 1982 "go.y"
    {
		(yyval.type) = tostruct((yyvsp[(3) - (4)].list));
	}
    break;

  case 321:
#line 1986 "go.y"
    {
		(yyval.type) = tointerface((yyvsp[(3) - (4)].list));
	}
    break;

  case 322:
#line 1990 "go.y"
    {
		(yyval.type) = ptrto((yyvsp[(2) - (2)].type));
	}
    break;

  case 323:
#line 1994 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(2) - (2)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 324:
#line 2000 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (4)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 325:
#line 2006 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Csend;
	}
    break;

  case 326:
#line 2014 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Crecv;
	}
    break;

  case 327:
#line 2022 "go.y"
    {
		(yyval.type) = functype(nil, (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list));
	}
    break;

  case 328:
#line 2028 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(2) - (3)].type)));
		if((yyvsp[(1) - (3)].sym))
			(yyval.node)->left = newname((yyvsp[(1) - (3)].sym));
		(yyval.node)->val = (yyvsp[(3) - (3)].val);
	}
    break;

  case 329:
#line 2035 "go.y"
    {
		Type *t;
	
		t = typ(TARRAY);
		t->bound = -1;
		t->type = (yyvsp[(3) - (4)].type);

		(yyval.node) = nod(ODCLFIELD, N, typenod(t));
		if((yyvsp[(1) - (4)].sym))
			(yyval.node)->left = newname((yyvsp[(1) - (4)].sym));
		(yyval.node)->isddd = 1;
		(yyval.node)->val = (yyvsp[(4) - (4)].val);
	}
    break;

  case 330:
#line 2051 "go.y"
    {
		Sym *s;

		if((yyvsp[(1) - (3)].sym) != S) {
			(yyval.node) = nod(ODCLFIELD, newname((yyvsp[(1) - (3)].sym)), typenod((yyvsp[(2) - (3)].type)));
			(yyval.node)->val = (yyvsp[(3) - (3)].val);
		} else {
			s = (yyvsp[(2) - (3)].type)->sym;
			if(s == S && isptr[(yyvsp[(2) - (3)].type)->etype])
				s = (yyvsp[(2) - (3)].type)->type->sym;
			(yyval.node) = embedded(s);
			(yyval.node)->right = typenod((yyvsp[(2) - (3)].type));
			(yyval.node)->val = (yyvsp[(3) - (3)].val);
		}
	}
    break;

  case 331:
#line 2069 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, newname((yyvsp[(1) - (5)].sym)), typenod(functype(fakethis(), (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list))));
	}
    break;

  case 332:
#line 2073 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type)));
	}
    break;

  case 333:
#line 2078 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 335:
#line 2085 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 336:
#line 2089 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type))));
	}
    break;

  case 337:
#line 2099 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 338:
#line 2103 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(2) - (2)].val));
		switch((yyval.node)->val.ctype){
		case CTINT:
		case CTRUNE:
			mpnegfix((yyval.node)->val.u.xval);
			break;
		case CTFLT:
			mpnegflt((yyval.node)->val.u.fval);
			break;
		default:
			yyerror("bad negated constant");
		}
	}
    break;

  case 339:
#line 2118 "go.y"
    {
		(yyval.node) = oldname(pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg));
		if((yyval.node)->op != OLITERAL)
			yyerror("bad constant %S", (yyval.node)->sym);
	}
    break;

  case 341:
#line 2127 "go.y"
    {
		if((yyvsp[(2) - (5)].node)->val.ctype == CTRUNE && (yyvsp[(4) - (5)].node)->val.ctype == CTINT) {
			(yyval.node) = (yyvsp[(2) - (5)].node);
			mpaddfixfix((yyvsp[(2) - (5)].node)->val.u.xval, (yyvsp[(4) - (5)].node)->val.u.xval, 0);
			break;
		}
		(yyvsp[(4) - (5)].node)->val.u.cval->real = (yyvsp[(4) - (5)].node)->val.u.cval->imag;
		mpmovecflt(&(yyvsp[(4) - (5)].node)->val.u.cval->imag, 0.0);
		(yyval.node) = nodcplxlit((yyvsp[(2) - (5)].node)->val, (yyvsp[(4) - (5)].node)->val);
	}
    break;

  case 344:
#line 2143 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 345:
#line 2147 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 346:
#line 2153 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 347:
#line 2157 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 348:
#line 2163 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 349:
#line 2167 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;


/* Line 1267 of yacc.c.  */
#line 4850 "y.tab.c"
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


#line 2171 "go.y"


static void
fixlbrace(int lbr)
{
	// If the opening brace was an LBODY,
	// set up for another one now that we're done.
	// See comment in lex.c about loophack.
	if(lbr == LBODY)
		loophack = 1;
}


