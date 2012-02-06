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
     LBREAK = 260,
     LCASE = 261,
     LCHAN = 262,
     LCOLAS = 263,
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
#define LBREAK 260
#define LCASE 261
#define LCHAN 262
#define LCOLAS 263
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
#define YYLAST   2157

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  76
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  138
/* YYNRULES -- Number of rules.  */
#define YYNRULES  343
/* YYNRULES -- Number of states.  */
#define YYNSTATES  652

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
      26,    32,    36,    40,    42,    46,    48,    51,    54,    59,
      60,    62,    63,    68,    69,    71,    73,    75,    77,    80,
      86,    90,    93,    99,   107,   111,   114,   120,   124,   126,
     129,   134,   138,   143,   147,   149,   152,   154,   156,   159,
     161,   165,   169,   173,   176,   179,   183,   189,   195,   198,
     199,   204,   205,   209,   210,   213,   214,   219,   224,   229,
     235,   237,   239,   242,   243,   247,   249,   253,   254,   255,
     256,   264,   265,   268,   271,   272,   273,   281,   282,   288,
     290,   294,   298,   302,   306,   310,   314,   318,   322,   326,
     330,   334,   338,   342,   346,   350,   354,   358,   362,   366,
     370,   372,   375,   378,   381,   384,   387,   390,   393,   396,
     400,   406,   413,   415,   417,   421,   427,   433,   438,   445,
     447,   452,   458,   464,   472,   474,   475,   479,   481,   486,
     488,   492,   494,   496,   498,   500,   502,   504,   506,   507,
     509,   511,   513,   515,   520,   522,   524,   526,   529,   531,
     533,   535,   537,   539,   543,   545,   547,   549,   552,   554,
     556,   558,   560,   564,   566,   568,   570,   572,   574,   576,
     578,   580,   582,   586,   591,   596,   599,   603,   609,   611,
     613,   616,   620,   626,   630,   636,   640,   644,   650,   659,
     665,   674,   680,   681,   685,   686,   688,   692,   694,   699,
     702,   703,   707,   709,   713,   715,   719,   721,   725,   727,
     731,   733,   737,   741,   744,   749,   753,   759,   765,   767,
     771,   773,   776,   778,   782,   787,   789,   792,   795,   797,
     799,   803,   804,   807,   808,   810,   812,   814,   816,   818,
     820,   822,   824,   826,   827,   832,   834,   837,   840,   843,
     846,   849,   852,   854,   858,   860,   864,   866,   870,   872,
     876,   878,   882,   884,   886,   890,   894,   895,   898,   899,
     901,   902,   904,   905,   907,   908,   910,   911,   913,   914,
     916,   917,   919,   920,   922,   923,   925,   930,   935,   941,
     948,   953,   958,   960,   962,   964,   966,   968,   970,   972,
     974,   976,   980,   985,   991,   996,  1001,  1004,  1007,  1012,
    1016,  1020,  1026,  1030,  1035,  1039,  1045,  1047,  1048,  1050,
    1054,  1056,  1058,  1061,  1063,  1065,  1071,  1072,  1075,  1077,
    1081,  1083,  1087,  1089
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      77,     0,    -1,    79,    78,    81,   162,    -1,    -1,    25,
     137,    62,    -1,    -1,    80,    86,    88,    -1,    -1,    81,
      82,    62,    -1,    21,    83,    -1,    21,    59,    84,   186,
      60,    -1,    21,    59,    60,    -1,    85,    86,    88,    -1,
      83,    -1,    84,    62,    83,    -1,     3,    -1,   137,     3,
      -1,    63,     3,    -1,    25,    24,    87,    62,    -1,    -1,
      24,    -1,    -1,    89,   210,    64,    64,    -1,    -1,    91,
      -1,   154,    -1,   177,    -1,     1,    -1,    32,    93,    -1,
      32,    59,   163,   186,    60,    -1,    32,    59,    60,    -1,
      92,    94,    -1,    92,    59,    94,   186,    60,    -1,    92,
      59,    94,    62,   164,   186,    60,    -1,    92,    59,    60,
      -1,    31,    97,    -1,    31,    59,   165,   186,    60,    -1,
      31,    59,    60,    -1,     9,    -1,   181,   142,    -1,   181,
     142,    65,   182,    -1,   181,    65,   182,    -1,   181,   142,
      65,   182,    -1,   181,    65,   182,    -1,    94,    -1,   181,
     142,    -1,   181,    -1,   137,    -1,    96,   142,    -1,   123,
      -1,   123,     4,   123,    -1,   182,    65,   182,    -1,   182,
       8,   182,    -1,   123,    42,    -1,   123,    37,    -1,     6,
     183,    66,    -1,     6,   183,    65,   123,    66,    -1,     6,
     183,     8,   123,    66,    -1,    12,    66,    -1,    -1,    67,
     101,   179,    68,    -1,    -1,    99,   103,   179,    -1,    -1,
     104,   102,    -1,    -1,    35,   106,   179,    68,    -1,   182,
      65,    26,   123,    -1,   182,     8,    26,   123,    -1,   190,
      62,   190,    62,   190,    -1,   190,    -1,   107,    -1,   108,
     105,    -1,    -1,    16,   111,   109,    -1,   190,    -1,   190,
      62,   190,    -1,    -1,    -1,    -1,    20,   114,   112,   115,
     105,   116,   117,    -1,    -1,    14,   113,    -1,    14,   100,
      -1,    -1,    -1,    30,   119,   112,   120,    35,   104,    68,
      -1,    -1,    28,   122,    35,   104,    68,    -1,   124,    -1,
     123,    47,   123,    -1,   123,    33,   123,    -1,   123,    38,
     123,    -1,   123,    46,   123,    -1,   123,    45,   123,    -1,
     123,    43,   123,    -1,   123,    39,   123,    -1,   123,    40,
     123,    -1,   123,    49,   123,    -1,   123,    50,   123,    -1,
     123,    51,   123,    -1,   123,    52,   123,    -1,   123,    53,
     123,    -1,   123,    54,   123,    -1,   123,    55,   123,    -1,
     123,    56,   123,    -1,   123,    34,   123,    -1,   123,    44,
     123,    -1,   123,    48,   123,    -1,   123,    36,   123,    -1,
     130,    -1,    53,   124,    -1,    56,   124,    -1,    49,   124,
      -1,    50,   124,    -1,    69,   124,    -1,    70,   124,    -1,
      52,   124,    -1,    36,   124,    -1,   130,    59,    60,    -1,
     130,    59,   183,   187,    60,    -1,   130,    59,   183,    11,
     187,    60,    -1,     3,    -1,   139,    -1,   130,    63,   137,
      -1,   130,    63,    59,   131,    60,    -1,   130,    63,    59,
      31,    60,    -1,   130,    71,   123,    72,    -1,   130,    71,
     188,    66,   188,    72,    -1,   125,    -1,   145,    59,   123,
      60,    -1,   146,   133,   127,   185,    68,    -1,   126,    67,
     127,   185,    68,    -1,    59,   131,    60,    67,   127,   185,
      68,    -1,   161,    -1,    -1,   123,    66,   129,    -1,   123,
      -1,    67,   127,   185,    68,    -1,   126,    -1,    59,   131,
      60,    -1,   123,    -1,   143,    -1,   142,    -1,    35,    -1,
      67,    -1,   137,    -1,   137,    -1,    -1,   134,    -1,    24,
      -1,   138,    -1,    73,    -1,    74,     3,    63,    24,    -1,
     137,    -1,   134,    -1,    11,    -1,    11,   142,    -1,   151,
      -1,   157,    -1,   149,    -1,   150,    -1,   148,    -1,    59,
     142,    60,    -1,   151,    -1,   157,    -1,   149,    -1,    53,
     143,    -1,   157,    -1,   149,    -1,   150,    -1,   148,    -1,
      59,   142,    60,    -1,   157,    -1,   149,    -1,   149,    -1,
     151,    -1,   157,    -1,   149,    -1,   150,    -1,   148,    -1,
     139,    -1,   139,    63,   137,    -1,    71,   188,    72,   142,
      -1,    71,    11,    72,   142,    -1,     7,   144,    -1,     7,
      36,   142,    -1,    23,    71,   142,    72,   142,    -1,   152,
      -1,   153,    -1,    53,   142,    -1,    36,     7,   142,    -1,
      29,   133,   166,   186,    68,    -1,    29,   133,    68,    -1,
      22,   133,   167,   186,    68,    -1,    22,   133,    68,    -1,
      17,   155,   158,    -1,   137,    59,   175,    60,   159,    -1,
      59,   175,    60,   137,    59,   175,    60,   159,    -1,   196,
      59,   191,    60,   206,    -1,    59,   211,    60,   137,    59,
     191,    60,   206,    -1,    17,    59,   175,    60,   159,    -1,
      -1,    67,   179,    68,    -1,    -1,   147,    -1,    59,   175,
      60,    -1,   157,    -1,   160,   133,   179,    68,    -1,   160,
       1,    -1,    -1,   162,    90,    62,    -1,    93,    -1,   163,
      62,    93,    -1,    95,    -1,   164,    62,    95,    -1,    97,
      -1,   165,    62,    97,    -1,   168,    -1,   166,    62,   168,
      -1,   171,    -1,   167,    62,   171,    -1,   180,   142,   194,
      -1,   170,   194,    -1,    59,   170,    60,   194,    -1,    53,
     170,   194,    -1,    59,    53,   170,    60,   194,    -1,    53,
      59,   170,    60,   194,    -1,    24,    -1,    24,    63,   137,
      -1,   169,    -1,   134,   172,    -1,   169,    -1,    59,   169,
      60,    -1,    59,   175,    60,   159,    -1,   132,    -1,   137,
     132,    -1,   137,   141,    -1,   141,    -1,   173,    -1,   174,
      75,   173,    -1,    -1,   174,   187,    -1,    -1,   100,    -1,
      91,    -1,   177,    -1,     1,    -1,    98,    -1,   110,    -1,
     118,    -1,   121,    -1,   113,    -1,    -1,   140,    66,   178,
     176,    -1,    15,    -1,     5,   136,    -1,    10,   136,    -1,
      18,   125,    -1,    13,   125,    -1,    19,   134,    -1,    27,
     189,    -1,   176,    -1,   179,    62,   176,    -1,   134,    -1,
     180,    75,   134,    -1,   135,    -1,   181,    75,   135,    -1,
     123,    -1,   182,    75,   123,    -1,   131,    -1,   183,    75,
     131,    -1,   128,    -1,   129,    -1,   184,    75,   128,    -1,
     184,    75,   129,    -1,    -1,   184,   187,    -1,    -1,    62,
      -1,    -1,    75,    -1,    -1,   123,    -1,    -1,   182,    -1,
      -1,    98,    -1,    -1,   211,    -1,    -1,   212,    -1,    -1,
     213,    -1,    -1,     3,    -1,    21,    24,     3,    62,    -1,
      32,   196,   198,    62,    -1,     9,   196,    65,   209,    62,
      -1,     9,   196,   198,    65,   209,    62,    -1,    31,   197,
     198,    62,    -1,    17,   156,   158,    62,    -1,   138,    -1,
     196,    -1,   200,    -1,   201,    -1,   202,    -1,   200,    -1,
     202,    -1,   138,    -1,    24,    -1,    71,    72,   198,    -1,
      71,     3,    72,   198,    -1,    23,    71,   198,    72,   198,
      -1,    29,    67,   192,    68,    -1,    22,    67,   193,    68,
      -1,    53,   198,    -1,     7,   199,    -1,     7,    59,   201,
      60,    -1,     7,    36,   198,    -1,    36,     7,   198,    -1,
      17,    59,   191,    60,   206,    -1,   137,   198,   194,    -1,
     137,    11,   198,   194,    -1,   137,   198,   194,    -1,   137,
      59,   191,    60,   206,    -1,   198,    -1,    -1,   207,    -1,
      59,   191,    60,    -1,   198,    -1,     3,    -1,    50,     3,
      -1,   137,    -1,   208,    -1,    59,   208,    49,   208,    60,
      -1,    -1,   210,   195,    -1,   203,    -1,   211,    75,   203,
      -1,   204,    -1,   212,    62,   204,    -1,   205,    -1,   213,
      62,   205,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   124,   124,   133,   140,   151,   151,   166,   167,   170,
     171,   172,   175,   211,   212,   215,   222,   229,   238,   251,
     252,   259,   259,   272,   276,   277,   281,   286,   292,   296,
     300,   304,   310,   316,   322,   327,   331,   335,   341,   347,
     351,   355,   361,   365,   371,   372,   376,   382,   391,   397,
     401,   406,   418,   434,   439,   446,   466,   484,   493,   512,
     511,   523,   522,   553,   556,   563,   562,   573,   579,   588,
     599,   605,   608,   616,   615,   626,   632,   644,   648,   653,
     643,   665,   668,   672,   679,   683,   678,   701,   700,   716,
     717,   721,   725,   729,   733,   737,   741,   745,   749,   753,
     757,   761,   765,   769,   773,   777,   781,   785,   789,   794,
     800,   801,   805,   816,   820,   824,   828,   833,   837,   847,
     851,   856,   864,   868,   869,   880,   884,   888,   892,   896,
     897,   903,   910,   916,   923,   926,   933,   939,   940,   947,
     948,   966,   967,   970,   973,   977,   988,   994,  1000,  1003,
    1006,  1013,  1014,  1020,  1029,  1037,  1049,  1054,  1060,  1061,
    1062,  1063,  1064,  1065,  1071,  1072,  1073,  1074,  1080,  1081,
    1082,  1083,  1084,  1090,  1091,  1094,  1097,  1098,  1099,  1100,
    1101,  1104,  1105,  1118,  1122,  1127,  1132,  1137,  1141,  1142,
    1145,  1151,  1158,  1164,  1171,  1177,  1188,  1199,  1228,  1267,
    1290,  1307,  1316,  1319,  1327,  1331,  1335,  1342,  1348,  1353,
    1365,  1368,  1376,  1377,  1383,  1384,  1390,  1394,  1400,  1401,
    1407,  1411,  1417,  1426,  1431,  1437,  1443,  1450,  1459,  1468,
    1483,  1489,  1494,  1498,  1505,  1518,  1519,  1525,  1531,  1534,
    1538,  1544,  1547,  1556,  1559,  1560,  1564,  1565,  1571,  1572,
    1573,  1574,  1575,  1577,  1576,  1591,  1596,  1600,  1604,  1608,
    1612,  1617,  1636,  1642,  1650,  1654,  1660,  1664,  1670,  1674,
    1680,  1684,  1693,  1697,  1701,  1705,  1711,  1714,  1722,  1723,
    1725,  1726,  1729,  1732,  1735,  1738,  1741,  1744,  1747,  1750,
    1753,  1756,  1759,  1762,  1765,  1768,  1774,  1778,  1782,  1786,
    1790,  1794,  1812,  1819,  1830,  1831,  1832,  1835,  1836,  1839,
    1843,  1853,  1857,  1861,  1865,  1869,  1873,  1877,  1883,  1889,
    1897,  1905,  1911,  1918,  1934,  1952,  1956,  1962,  1965,  1968,
    1972,  1982,  1986,  2001,  2009,  2010,  2020,  2021,  2024,  2028,
    2034,  2038,  2044,  2048
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
const char *yytname[] =
{
  "$end", "error", "$undefined", "LLITERAL", "LASOP", "LBREAK", "LCASE",
  "LCHAN", "LCOLAS", "LCONST", "LCONTINUE", "LDDD", "LDEFAULT", "LDEFER",
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
  "if_header", "if_stmt", "@7", "@8", "@9", "else", "switch_stmt", "@10",
  "@11", "select_stmt", "@12", "expr", "uexpr", "pseudocall",
  "pexpr_no_paren", "start_complit", "keyval", "complitexpr", "pexpr",
  "expr_or_type", "name_or_type", "lbrace", "new_name", "dcl_name",
  "onew_name", "sym", "hidden_importsym", "name", "labelname", "dotdotdot",
  "ntype", "non_expr_type", "non_recvchantype", "convtype", "comptype",
  "fnret_type", "dotname", "othertype", "ptrtype", "recvchantype",
  "structtype", "interfacetype", "xfndcl", "fndcl", "hidden_fndcl",
  "fntype", "fnbody", "fnres", "fnlitdcl", "fnliteral", "xdcl_list",
  "vardcl_list", "constdcl_list", "typedcl_list", "structdcl_list",
  "interfacedcl_list", "structdcl", "packname", "embed", "interfacedcl",
  "indcl", "arg_type", "arg_type_list", "oarg_type_list_ocomma", "stmt",
  "non_dcl_stmt", "@13", "stmt_list", "new_name_list", "dcl_name_list",
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
      82,    82,    83,    84,    84,    85,    85,    85,    86,    87,
      87,    89,    88,    90,    90,    90,    90,    90,    91,    91,
      91,    91,    91,    91,    91,    91,    91,    91,    92,    93,
      93,    93,    94,    94,    95,    95,    95,    96,    97,    98,
      98,    98,    98,    98,    98,    99,    99,    99,    99,   101,
     100,   103,   102,   104,   104,   106,   105,   107,   107,   108,
     108,   108,   109,   111,   110,   112,   112,   114,   115,   116,
     113,   117,   117,   117,   119,   120,   118,   122,   121,   123,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     124,   124,   124,   124,   124,   124,   124,   124,   124,   125,
     125,   125,   126,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   126,   126,   127,   128,   129,   129,   130,
     130,   131,   131,   132,   133,   133,   134,   135,   136,   136,
     137,   137,   137,   138,   139,   140,   141,   141,   142,   142,
     142,   142,   142,   142,   143,   143,   143,   143,   144,   144,
     144,   144,   144,   145,   145,   146,   147,   147,   147,   147,
     147,   148,   148,   149,   149,   149,   149,   149,   149,   149,
     150,   151,   152,   152,   153,   153,   154,   155,   155,   156,
     156,   157,   158,   158,   159,   159,   159,   160,   161,   161,
     162,   162,   163,   163,   164,   164,   165,   165,   166,   166,
     167,   167,   168,   168,   168,   168,   168,   168,   169,   169,
     170,   171,   171,   171,   172,   173,   173,   173,   173,   174,
     174,   175,   175,   176,   176,   176,   176,   176,   177,   177,
     177,   177,   177,   178,   177,   177,   177,   177,   177,   177,
     177,   177,   179,   179,   180,   180,   181,   181,   182,   182,
     183,   183,   184,   184,   184,   184,   185,   185,   186,   186,
     187,   187,   188,   188,   189,   189,   190,   190,   191,   191,
     192,   192,   193,   193,   194,   194,   195,   195,   195,   195,
     195,   195,   196,   197,   198,   198,   198,   199,   199,   200,
     200,   200,   200,   200,   200,   200,   200,   200,   200,   200,
     201,   202,   203,   203,   204,   205,   205,   206,   206,   207,
     207,   208,   208,   208,   209,   209,   210,   210,   211,   211,
     212,   212,   213,   213
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     4,     0,     3,     0,     3,     0,     3,     2,
       5,     3,     3,     1,     3,     1,     2,     2,     4,     0,
       1,     0,     4,     0,     1,     1,     1,     1,     2,     5,
       3,     2,     5,     7,     3,     2,     5,     3,     1,     2,
       4,     3,     4,     3,     1,     2,     1,     1,     2,     1,
       3,     3,     3,     2,     2,     3,     5,     5,     2,     0,
       4,     0,     3,     0,     2,     0,     4,     4,     4,     5,
       1,     1,     2,     0,     3,     1,     3,     0,     0,     0,
       7,     0,     2,     2,     0,     0,     7,     0,     5,     1,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       1,     2,     2,     2,     2,     2,     2,     2,     2,     3,
       5,     6,     1,     1,     3,     5,     5,     4,     6,     1,
       4,     5,     5,     7,     1,     0,     3,     1,     4,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       1,     1,     1,     4,     1,     1,     1,     2,     1,     1,
       1,     1,     1,     3,     1,     1,     1,     2,     1,     1,
       1,     1,     3,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     4,     4,     2,     3,     5,     1,     1,
       2,     3,     5,     3,     5,     3,     3,     5,     8,     5,
       8,     5,     0,     3,     0,     1,     3,     1,     4,     2,
       0,     3,     1,     3,     1,     3,     1,     3,     1,     3,
       1,     3,     3,     2,     4,     3,     5,     5,     1,     3,
       1,     2,     1,     3,     4,     1,     2,     2,     1,     1,
       3,     0,     2,     0,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     0,     4,     1,     2,     2,     2,     2,
       2,     2,     1,     3,     1,     3,     1,     3,     1,     3,
       1,     3,     1,     1,     3,     3,     0,     2,     0,     1,
       0,     1,     0,     1,     0,     1,     0,     1,     0,     1,
       0,     1,     0,     1,     0,     1,     4,     4,     5,     6,
       4,     4,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     4,     5,     4,     4,     2,     2,     4,     3,
       3,     5,     3,     4,     3,     5,     1,     0,     1,     3,
       1,     1,     2,     1,     1,     5,     0,     2,     1,     3,
       1,     3,     1,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       5,     0,     3,     0,     1,     0,     7,     0,    21,   150,
     152,     0,     0,   151,   210,    19,     6,   336,     0,     4,
       0,     0,     0,    20,     0,     0,     0,    15,     0,     0,
       9,     0,     0,     8,    27,   122,   148,     0,    38,   148,
       0,   255,    73,     0,     0,     0,    77,     0,     0,   284,
      87,     0,    84,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   282,     0,    24,     0,   248,   249,
     252,   250,   251,    49,    89,   129,   139,   110,   155,   154,
     123,     0,     0,     0,   175,   188,   189,    25,   207,     0,
     134,    26,     0,    18,     0,     0,     0,     0,     0,     0,
     337,   153,    11,    13,   278,    17,    21,    16,   149,   256,
     146,     0,     0,     0,     0,   154,   181,   185,   171,   169,
     170,   168,   257,   129,     0,   286,   241,     0,   202,   129,
     260,   286,   144,   145,     0,     0,   268,   285,   261,     0,
       0,   286,     0,     0,    35,    47,     0,    28,   266,   147,
       0,   118,   113,   114,   117,   111,   112,     0,     0,   141,
       0,   142,   166,   164,   165,   115,   116,     0,   283,     0,
     211,     0,    31,     0,     0,     0,     0,     0,    54,     0,
       0,     0,    53,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   135,     0,     0,
     282,   253,     0,   135,   209,     0,     0,     0,     0,   302,
       0,     0,   202,     0,     0,   303,     0,     0,    22,   279,
       0,    12,   241,     0,     0,   186,   162,   160,   161,   158,
     159,   190,     0,     0,   287,    71,     0,    74,     0,    70,
     156,   235,   154,   238,   143,   239,   280,     0,   241,     0,
     196,    78,    75,   150,     0,   195,     0,   278,   232,   220,
       0,    63,     0,     0,   193,   264,   278,   218,   230,   294,
       0,    85,    37,   216,   278,    48,    30,   212,   278,     0,
       0,    39,     0,   167,   140,     0,     0,    34,   278,     0,
       0,    50,    91,   106,   109,    92,    96,    97,    95,   107,
      94,    93,    90,   108,    98,    99,   100,   101,   102,   103,
     104,   105,   276,   119,   270,   280,     0,   124,   283,     0,
       0,     0,   276,   247,    59,   245,   244,   262,   246,     0,
      52,    51,   269,     0,     0,     0,     0,   310,     0,     0,
       0,     0,     0,   309,     0,   304,   305,   306,     0,   338,
       0,     0,   288,     0,     0,     0,    14,    10,     0,     0,
       0,   172,   182,    65,    72,     0,     0,   286,   157,   236,
     237,   281,   242,   204,     0,     0,     0,   286,     0,   228,
       0,   241,   231,   279,     0,     0,     0,     0,   294,     0,
       0,   279,     0,   295,   223,     0,   294,     0,   279,     0,
     279,     0,    41,   267,     0,     0,     0,   191,   162,   160,
     161,   159,   135,   184,   183,   279,     0,    43,     0,   135,
     137,   272,   273,   280,     0,   280,   281,     0,     0,     0,
     127,   282,   254,   130,     0,     0,     0,   208,     0,     0,
     317,   307,   308,   288,   292,     0,   290,     0,   316,   331,
       0,     0,   333,   334,     0,     0,     0,     0,     0,   294,
       0,     0,   301,     0,   289,   296,   300,   297,   204,   163,
       0,     0,     0,     0,   240,   241,   154,   205,   180,   178,
     179,   176,   177,   201,   204,   203,    79,    76,   229,   233,
       0,   221,   194,   187,     0,     0,    88,    61,    64,     0,
     225,     0,   294,   219,   192,   265,   222,    63,   217,    36,
     213,    29,    40,     0,   276,    44,   214,   278,    46,    32,
      42,   276,     0,   281,   277,   132,   281,     0,   271,   120,
     126,   125,     0,   131,     0,   263,   319,     0,     0,   310,
       0,   309,     0,   326,   342,   293,     0,     0,     0,   340,
     291,   320,   332,     0,   298,     0,   311,     0,   294,   322,
       0,   339,   327,     0,    68,    67,   286,     0,   241,   197,
      81,   204,     0,    58,     0,   294,   294,   224,     0,   163,
       0,   279,     0,    45,     0,   137,   136,   274,   275,   121,
     128,    60,   318,   327,   288,   315,     0,     0,   294,   314,
       0,     0,   312,   299,   323,   288,   288,   330,   199,   328,
      66,    69,   206,     0,     0,    80,   234,     0,     0,    55,
       0,    62,   227,   226,    86,   133,   215,    33,   138,   321,
       0,   343,   313,   324,   341,     0,     0,     0,   204,    83,
      82,     0,     0,   327,   335,   327,   329,   198,    57,    56,
     325,   200
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     6,     2,     3,    14,    21,    30,   104,    31,
       8,    24,    16,    17,    65,   325,    67,   147,   515,   516,
     143,   144,    68,   497,   326,   435,   498,   574,   386,   364,
     470,   235,   236,   237,    69,   125,   251,    70,   131,   376,
     570,   615,    71,   141,   397,    72,   139,    73,    74,    75,
      76,   312,   421,   422,    77,   314,   241,   134,    78,   148,
     109,   115,    13,    80,    81,   243,   244,   161,   117,    82,
      83,   477,   226,    84,   228,   229,    85,    86,    87,   128,
     212,    88,   250,   483,    89,    90,    22,   278,   517,   274,
     266,   257,   267,   268,   269,   259,   382,   245,   246,   247,
     327,   328,   320,   329,   270,   150,    92,   315,   423,   424,
     220,   372,   169,   138,   252,   463,   548,   542,   394,   100,
     210,   216,   607,   440,   345,   346,   347,   349,   549,   544,
     608,   609,   453,   454,    25,   464,   550,   545
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -516
static const yytype_int16 yypact[] =
{
    -516,    69,    27,    75,  -516,   161,  -516,   125,  -516,  -516,
    -516,   164,   113,  -516,   156,   166,  -516,  -516,   142,  -516,
      44,   145,  1036,  -516,   165,   275,   206,  -516,   139,   243,
    -516,    75,   244,  -516,  -516,  -516,   161,    60,  -516,   161,
     771,  -516,  -516,   150,   771,   161,  -516,   137,   177,  1473,
    -516,   137,  -516,   434,   474,  1473,  1473,  1473,  1473,  1473,
    1473,  1528,  1473,  1473,   710,   192,  -516,   486,  -516,  -516,
    -516,  -516,  -516,   847,  -516,  -516,   190,   174,  -516,   200,
    -516,   211,   223,   137,   238,  -516,  -516,  -516,   241,   186,
    -516,  -516,    45,  -516,   224,    11,   279,   224,   224,   240,
    -516,  -516,  -516,  -516,   252,  -516,  -516,  -516,  -516,  -516,
    -516,   246,  1736,  1736,  1736,  -516,   255,  -516,  -516,  -516,
    -516,  -516,  -516,    33,   174,  1473,  1697,   263,   265,   219,
    -516,  1473,  -516,  -516,   347,  1736,  2030,   259,  -516,   302,
     423,  1473,   228,  1736,  -516,  -516,   331,  -516,  -516,  -516,
    1638,  -516,  -516,  -516,  -516,  -516,  -516,  1583,  1528,  2030,
     278,  -516,    31,  -516,   183,  -516,  -516,   288,  2030,   292,
    -516,   334,  -516,  1663,  1473,  1473,  1473,  1473,  -516,  1473,
    1473,  1473,  -516,  1473,  1473,  1473,  1473,  1473,  1473,  1473,
    1473,  1473,  1473,  1473,  1473,  1473,  1473,  -516,   934,   529,
    1473,  -516,  1473,  -516,  -516,  1195,  1473,  1473,  1473,  -516,
     782,   161,   265,   285,   364,  -516,  1264,  1264,  -516,   216,
     290,  -516,  1697,   363,  1736,  -516,  -516,  -516,  -516,  -516,
    -516,  -516,   312,   161,  -516,  -516,   340,  -516,    79,   323,
    1736,  -516,  1697,  -516,  -516,  -516,   326,   327,  1697,  1195,
    -516,  -516,   336,    85,   379,  -516,   354,   355,  -516,  -516,
     353,  -516,    50,   112,  -516,  -516,   365,  -516,  -516,   426,
    1671,  -516,  -516,  -516,   372,  -516,  -516,  -516,   378,  1473,
     161,   361,  1740,  -516,   387,  1736,  1736,  -516,   397,  1473,
     380,  2030,  2101,  -516,  2054,   616,   616,   616,   616,  -516,
     616,   616,  2078,  -516,   585,   585,   585,   585,  -516,  -516,
    -516,  -516,  1253,  -516,  -516,    40,  1308,  -516,  1903,   396,
    1121,  2005,  1253,  -516,  -516,  -516,  -516,  -516,  -516,    29,
     259,   259,  2030,  1811,   407,   400,   406,  -516,   401,   473,
    1264,    52,    34,  -516,   418,  -516,  -516,  -516,   925,  -516,
      19,   422,   161,   424,   428,   430,  -516,  -516,   435,  1736,
     445,  -516,  -516,  -516,  -516,  1363,  1418,  1473,  -516,  -516,
    -516,  1697,  -516,  1768,   452,   127,   340,  1473,   161,   425,
     454,  1697,  -516,   554,   448,  1736,   102,   379,   426,   379,
     457,   477,   455,  -516,  -516,   161,   426,   485,   161,   466,
     161,   468,   259,  -516,  1473,  1779,  1736,  -516,   260,   274,
     276,   310,  -516,  -516,  -516,   161,   469,   259,  1473,  -516,
    1933,  -516,  -516,   464,   475,   467,  1528,   481,   484,   489,
    -516,  1473,  -516,  -516,   478,  1195,  1121,  -516,  1264,   518,
    -516,  -516,  -516,   161,  1837,  1264,   161,  1264,  -516,  -516,
     552,   307,  -516,  -516,   495,   490,  1264,    52,  1264,   426,
     161,   161,  -516,   498,   491,  -516,  -516,  -516,  1768,  -516,
    1195,  1473,  1473,   506,  -516,  1697,   511,  -516,  -516,  -516,
    -516,  -516,  -516,  -516,  1768,  -516,  -516,  -516,  -516,  -516,
     505,  -516,  -516,  -516,  1528,   508,  -516,  -516,  -516,   512,
    -516,   515,   426,  -516,  -516,  -516,  -516,  -516,  -516,  -516,
    -516,  -516,   259,   517,  1253,  -516,  -516,   509,  1663,  -516,
     259,  1253,  1253,  1253,  -516,  -516,  -516,   519,  -516,  -516,
    -516,  -516,   510,  -516,   202,  -516,  -516,   520,   523,   525,
     526,   528,   522,  -516,  -516,   532,   527,  1264,   536,  -516,
     535,  -516,  -516,   567,  -516,  1264,  -516,   555,   426,  -516,
     559,  -516,  1864,   231,  2030,  2030,  1473,   563,  1697,  -516,
     611,  1768,    57,  -516,  1121,   426,   426,  -516,   226,   383,
     558,   161,   571,   380,   564,  2030,  -516,  -516,  -516,  -516,
    -516,  -516,  -516,  1864,   161,  -516,  1837,  1264,   426,  -516,
     161,   307,  -516,  -516,  -516,   161,   161,  -516,  -516,  -516,
    -516,  -516,  -516,   575,    26,  -516,  -516,  1473,  1473,  -516,
    1528,   574,  -516,  -516,  -516,  -516,  -516,  -516,  -516,  -516,
     577,  -516,  -516,  -516,  -516,   582,   583,   586,  1768,  -516,
    -516,  1957,  1981,  1864,  -516,  1864,  -516,  -516,  -516,  -516,
    -516,  -516
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -516,  -516,  -516,  -516,  -516,  -516,  -516,    -4,  -516,  -516,
     618,  -516,   541,  -516,  -516,   629,  -516,  -137,   -25,    71,
    -516,  -139,  -109,  -516,    39,  -516,  -516,  -516,   149,   281,
    -516,  -516,  -516,  -516,  -516,  -516,   521,    47,  -516,  -516,
    -516,  -516,  -516,  -516,  -516,  -516,  -516,   503,     1,   272,
    -516,  -190,   136,  -450,   277,   -47,   431,     3,     5,   394,
     636,    -5,   405,   239,  -516,   443,   249,   542,  -516,  -516,
    -516,  -516,   -33,    38,   -31,    10,  -516,  -516,  -516,  -516,
    -516,    43,   492,  -459,  -516,  -516,  -516,  -516,  -516,  -516,
    -516,  -516,   311,  -127,  -227,   325,  -516,   335,  -516,  -220,
    -293,   690,  -516,  -244,  -516,   -66,    67,   221,  -516,  -311,
    -245,  -285,  -192,  -516,  -106,  -423,  -516,  -516,  -378,  -516,
     377,  -516,   376,  -516,   385,   280,   389,   264,   120,   130,
    -515,  -516,  -425,   271,  -516,   524,  -516,  -516
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -269
static const yytype_int16 yytable[] =
{
      12,   173,   358,   273,   118,   375,   120,   258,   319,   277,
     500,   434,   384,   322,   160,    32,   234,    79,   506,   239,
     538,   392,   234,    32,   103,   569,   553,   432,   374,   399,
     427,   110,   234,   401,   110,   388,   390,   455,   127,  -259,
     110,   108,   172,   416,   108,  -259,    46,    27,   145,   149,
     130,   425,     5,   206,   140,   449,   151,   152,   153,   154,
     155,   156,   149,   165,   166,   617,  -175,    37,     9,     4,
     211,   163,   586,   588,   379,   119,     9,   111,   629,   460,
     121,   559,    47,    48,     9,    11,   203,   365,  -228,    51,
    -174,   436,   205,   324,   461,  -259,   112,   437,  -175,   162,
       7,  -259,   450,    28,   164,   173,   456,    29,   494,   387,
     207,   451,   616,   113,   495,   426,   137,    10,    11,   114,
     208,   242,   618,   619,   577,    10,    11,   380,   650,   110,
     651,    64,   620,    10,    11,   110,   379,   145,   524,   256,
     527,   149,    27,   535,   366,   265,   288,  -228,   378,    15,
     227,   227,   227,  -228,   208,   230,   230,   230,   151,   155,
     499,   490,   501,     9,   227,   389,   149,    18,   163,   230,
     496,   630,   132,   227,     9,    19,   635,    20,   230,   647,
     604,   227,   636,   637,  -207,     9,   230,   204,   227,   436,
      23,   534,   238,   230,   317,   485,   162,   622,   623,   102,
      79,   164,    29,   580,   133,    26,   348,    33,   163,   126,
     584,   227,    10,    11,    32,   356,   230,   242,  -207,    27,
     633,   132,   514,    10,    11,  -258,   563,    93,   362,   521,
     101,  -258,   494,   198,    10,    11,   162,   199,   495,   532,
       9,   164,  -173,   242,    79,   200,   105,   107,   135,   408,
    -207,   410,     9,   133,   170,   567,   258,   197,   234,   508,
     227,   473,   227,   510,   436,   230,  -146,   230,   234,   429,
     591,   487,   582,   330,   331,   149,   116,   201,   227,    29,
     227,  -258,   202,   230,    94,   230,   227,  -258,   272,    10,
      11,   230,    95,   436,   624,  -171,    96,  -174,    11,   610,
    -173,    10,    11,   214,   218,   222,    97,    98,   227,  -169,
     449,  -170,   123,   230,   219,    79,   129,   124,   233,  -171,
     409,   124,   248,   227,   227,   411,   163,  -171,   230,   230,
     621,     9,   249,  -169,   208,  -170,   452,   261,   284,    99,
     478,  -169,   480,  -170,   352,  -168,   402,   348,   613,   518,
     357,   116,   116,   116,   162,     9,   417,   450,     9,   164,
     285,   225,   231,   232,   286,   116,   242,   353,   476,  -168,
     359,   253,   361,   488,   116,   363,   242,  -168,   110,   528,
      10,    11,   116,   481,   260,   367,   110,   373,   256,   116,
     110,   276,   275,   145,   287,   149,   265,   227,   377,   281,
     505,   371,   230,   379,    10,    11,   254,    10,    11,   227,
     149,   479,   116,   381,   230,   255,   482,   383,  -172,   227,
      10,    11,   290,   227,   230,   385,   404,   391,   230,   393,
      79,    79,   330,   331,   398,   478,   163,   480,   348,   540,
     400,   547,  -172,   227,   227,   418,   452,   253,   230,   230,
    -172,   478,   452,   480,   412,   560,   348,   234,     9,   415,
     611,   116,   431,   116,   162,    79,   443,   444,   446,   164,
     242,   512,   213,   360,   215,   217,   262,   445,   481,   116,
     447,   116,   263,   457,   462,   520,   465,   116,   378,   368,
     466,   264,   467,   142,   481,   468,    10,    11,     9,   209,
     209,   253,   209,   209,   163,   469,   479,    10,    11,   116,
       9,   482,   484,   227,   489,   518,   492,   502,   230,   396,
     507,   116,   479,   504,   116,   116,   509,   482,   511,   519,
     262,   407,   162,   146,   413,   414,   263,   164,   478,   523,
     480,   529,   526,   525,   530,   171,   533,    10,    11,   531,
      10,    11,   136,     9,   339,   552,   227,   554,   562,    10,
      11,   230,   555,   242,   159,   571,   461,   168,   566,    79,
     568,   581,   575,   528,   573,   576,   149,   579,   253,   589,
     592,   481,   590,   593,  -150,   594,   344,  -151,   316,   348,
     595,   540,   354,   355,   596,   547,   452,   600,   116,   597,
     348,   348,    10,    11,   599,   478,   227,   480,   407,   479,
     116,   230,   116,   254,   482,   343,   601,   603,   605,   176,
     116,   343,   343,   612,   116,   614,   625,    10,    11,   184,
     163,   627,   628,   188,   493,   638,   436,   643,   193,   194,
     195,   196,   644,   645,   116,   116,   646,   221,   481,   106,
     176,    66,   626,   639,   225,   513,   578,   486,   162,   587,
     184,   640,   271,   164,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   369,   403,   122,   479,   291,   292,   293,
     294,   482,   295,   296,   297,   370,   298,   299,   300,   301,
     302,   303,   304,   305,   306,   307,   308,   309,   310,   311,
     283,   159,   503,   318,   351,   321,   474,   116,   491,   136,
     136,   332,    91,    35,   116,   572,   448,    37,   441,   537,
     634,   167,   442,   116,   459,   561,   631,   111,   557,     0,
       0,     0,    47,    48,     9,   350,     0,     0,   343,    51,
       0,     0,     0,     0,     0,   343,    55,     0,     0,     0,
       0,     0,     0,   343,     0,     0,     0,   116,     0,    56,
      57,     0,    58,    59,     0,     0,    60,   583,     0,    61,
       0,     0,     0,     0,    35,     0,     0,     0,    37,    62,
      63,    64,   136,    10,    11,     0,     0,     0,   111,   333,
       0,     0,   136,    47,    48,     9,     0,     0,     0,   334,
      51,     0,     0,     0,   335,   336,   337,   116,     0,     0,
     116,   338,     0,     0,   536,   420,     0,     0,   339,   159,
     543,   546,     0,   551,     0,   420,     0,     0,     0,     0,
      61,     0,   556,     0,   558,   340,     0,     0,     0,     0,
       0,     0,    64,   343,    10,    11,     0,   341,     0,   541,
     343,   174,   343,   342,     0,  -268,    11,     0,     0,     0,
       0,   343,     0,   343,     0,     0,     0,     0,   136,   136,
       0,     0,     0,     0,     0,     0,     0,   116,     0,     0,
     175,   176,     0,   177,   178,   179,   180,   181,     0,   182,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,     0,     0,     0,   136,     0,     0,
       0,     0,  -268,     0,     0,     0,     0,     0,     0,     0,
       0,   136,  -268,   598,     0,     0,     0,     0,     0,   159,
       0,   602,   333,     0,   168,     0,   458,    35,     0,     0,
       0,    37,   334,     0,     0,     0,     0,   335,   336,   337,
       0,   111,   343,     0,   338,     0,    47,    48,     9,     0,
     343,   339,     0,    51,     0,     0,     0,   343,     0,     0,
     157,     0,   543,   632,   564,   565,     0,     0,   340,     0,
       0,     0,     0,    56,    57,     0,    58,   158,     0,     0,
      60,     0,     0,    61,   313,     0,   342,   159,   343,    11,
       0,   541,   343,    62,    63,    64,     0,    10,    11,     0,
       0,     0,     0,     0,     0,     0,     0,   420,     0,     0,
       0,     0,     0,     0,   420,   585,   420,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    -2,    34,     0,    35,
       0,    36,     0,    37,     0,    38,    39,     0,   343,    40,
     343,    41,    42,    43,    44,    45,    46,     0,    47,    48,
       9,     0,     0,    49,    50,    51,    52,    53,    54,     0,
       0,     0,    55,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    56,    57,     0,    58,    59,
       0,     0,    60,     0,     0,    61,     0,     0,   -23,     0,
       0,     0,     0,     0,     0,    62,    63,    64,     0,    10,
      11,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     641,   642,   323,   159,    35,     0,    36,  -243,    37,     0,
      38,    39,     0,  -243,    40,     0,    41,    42,   111,    44,
      45,    46,     0,    47,    48,     9,     0,     0,    49,    50,
      51,    52,    53,    54,     0,     0,     0,    55,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      56,    57,     0,    58,    59,     0,     0,    60,     0,     0,
      61,     0,     0,  -243,     0,     0,     0,     0,   324,  -243,
      62,    63,    64,     0,    10,    11,   323,     0,    35,     0,
      36,     0,    37,     0,    38,    39,     0,     0,    40,     0,
      41,    42,   111,    44,    45,    46,     0,    47,    48,     9,
       0,     0,    49,    50,    51,    52,    53,    54,     0,     0,
       0,    55,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    56,    57,     0,    58,    59,     0,
       0,    60,     0,     0,    61,     0,    35,  -243,     0,     0,
      37,     0,   324,  -243,    62,    63,    64,     0,    10,    11,
     111,   333,     0,     0,     0,    47,    48,     9,     0,     0,
       0,   334,    51,     0,     0,     0,   335,   336,   337,    55,
       0,     0,     0,   338,     0,     0,     0,     0,     0,     0,
     339,     0,    56,    57,     0,    58,    59,     0,     0,    60,
       0,    35,    61,     0,     0,    37,     0,   340,     0,     0,
     419,     0,    62,    63,    64,   111,    10,    11,     0,     0,
      47,    48,     9,     0,     0,   342,     0,    51,    11,   428,
       0,     0,     0,     0,   157,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    56,    57,     0,
      58,   158,     0,     0,    60,     0,    35,    61,     0,     0,
      37,     0,     0,     0,     0,     0,     0,    62,    63,    64,
     111,    10,    11,     0,     0,    47,    48,     9,     0,   471,
       0,     0,    51,     0,     0,     0,     0,     0,     0,    55,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    56,    57,     0,    58,    59,     0,     0,    60,
       0,    35,    61,     0,     0,    37,     0,     0,     0,     0,
       0,     0,    62,    63,    64,   111,    10,    11,     0,     0,
      47,    48,     9,     0,   472,     0,     0,    51,     0,     0,
       0,     0,     0,     0,    55,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    56,    57,     0,
      58,    59,     0,     0,    60,     0,    35,    61,     0,     0,
      37,     0,     0,     0,     0,     0,     0,    62,    63,    64,
     111,    10,    11,     0,     0,    47,    48,     9,     0,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,    55,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    56,    57,     0,    58,    59,     0,     0,    60,
       0,    35,    61,     0,     0,    37,     0,     0,     0,     0,
       0,     0,    62,    63,    64,   111,    10,    11,     0,     0,
      47,    48,     9,     0,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,   157,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    56,    57,     0,
      58,   158,     0,     0,    60,     0,    35,    61,     0,     0,
     282,     0,     0,     0,     0,     0,     0,    62,    63,    64,
     111,    10,    11,     0,     0,    47,    48,     9,     0,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,    55,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    56,    57,     0,    58,    59,     0,     0,    60,
       0,     0,    61,     0,     0,    37,     0,     0,     0,     0,
       0,     0,    62,    63,    64,   111,    10,    11,     0,     0,
      47,    48,     9,     0,     0,     0,     0,    51,     0,     0,
      37,     0,     0,     0,   223,     0,     0,     0,    37,     0,
     111,     0,     0,     0,     0,    47,    48,     9,   111,     0,
       0,   113,    51,    47,    48,     9,     0,   224,     0,   223,
      51,     0,     0,   279,    37,     0,     0,   223,   240,    64,
       0,    10,    11,   280,   111,     0,   113,     0,     0,    47,
      48,     9,   224,     0,   113,     0,    51,     0,   289,     0,
     224,     0,     0,   223,    64,     0,    10,    11,   280,     0,
       0,     0,    64,    37,    10,    11,   395,    37,     0,     0,
     113,     0,     0,   111,     0,     0,   224,   111,    47,    48,
       9,     0,    47,    48,     9,    51,     0,     0,    64,    51,
      10,    11,   223,     0,     0,    37,   405,     0,     0,     0,
       0,     0,     0,     0,     0,   111,   282,     0,     0,   113,
      47,    48,     9,   113,     0,   224,   111,    51,     0,   406,
       0,    47,    48,     9,   223,     0,     0,    64,    51,    10,
      11,    64,     0,    10,    11,   223,     0,     0,   333,     0,
       0,   113,     0,     0,     0,     0,     0,   475,   334,     0,
       0,     0,   113,   335,   336,   337,     0,     0,   224,    64,
     338,    10,    11,     0,   333,     0,     0,   438,     0,     0,
      64,     0,    10,    11,   334,     0,     0,     0,     0,   335,
     336,   539,     0,     0,   340,     0,   338,     0,     0,     0,
     439,   333,     0,   339,     0,     0,     0,     0,     0,     0,
       0,   334,   342,     0,     0,    11,   335,   336,   337,     0,
     340,     0,     0,   338,     0,     0,     0,     0,     0,     0,
     339,     0,     0,     0,     0,     0,     0,     0,   342,     0,
      10,    11,     0,     0,     0,     0,     0,   340,     0,     0,
       0,     0,     0,   606,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   342,   175,   176,    11,   177,
       0,   179,   180,   181,     0,     0,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
       0,     0,     0,     0,     0,     0,   175,   176,     0,   177,
       0,   179,   180,   181,     0,   430,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     175,   176,     0,   177,     0,   179,   180,   181,     0,   522,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   175,   176,     0,   177,     0,   179,
     180,   181,     0,   648,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   175,   176,
       0,   177,     0,   179,   180,   181,     0,   649,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,     0,   175,   176,   433,   177,     0,   179,   180,
     181,     0,     0,   183,   184,   185,   186,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,   175,   176,     0,
       0,     0,   179,   180,   181,     0,     0,   183,   184,   185,
     186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,   175,   176,     0,     0,     0,   179,   180,   181,     0,
       0,   183,   184,   185,   186,     0,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   176,     0,     0,     0,   179,
     180,   181,     0,     0,   183,   184,   185,   186,     0,   188,
     189,   190,   191,   192,   193,   194,   195,   196
};

static const yytype_int16 yycheck[] =
{
       5,    67,   222,   142,    37,   249,    37,   134,   200,   146,
     388,   322,   257,   203,    61,    20,   125,    22,   396,   125,
     443,   266,   131,    28,    28,   484,   451,   320,   248,   274,
     315,    36,   141,   278,    39,   262,   263,     3,    43,     6,
      45,    36,    67,   288,    39,    12,    20,     3,    53,    54,
      45,    11,    25,     8,    51,     3,    55,    56,    57,    58,
      59,    60,    67,    62,    63,     8,    35,     7,    24,     0,
      59,    61,   522,   523,    24,    37,    24,    17,   593,    60,
      37,   459,    22,    23,    24,    74,    83,     8,     3,    29,
      59,    62,    89,    67,    75,    62,    36,    68,    67,    61,
      25,    68,    50,    59,    61,   171,    72,    63,     6,    59,
      65,    59,   571,    53,    12,    75,    49,    73,    74,    59,
      75,   126,    65,    66,   502,    73,    74,   254,   643,   134,
     645,    71,    75,    73,    74,   140,    24,   142,   423,   134,
     425,   146,     3,   436,    65,   140,   171,    62,    63,    24,
     112,   113,   114,    68,    75,   112,   113,   114,   157,   158,
     387,   381,   389,    24,   126,    53,   171,     3,   158,   126,
      68,   594,    35,   135,    24,    62,   601,    21,   135,   638,
     558,   143,   605,   606,     1,    24,   143,     1,   150,    62,
      24,   435,   125,   150,   199,    68,   158,   575,   576,    60,
     205,   158,    63,   514,    67,    63,   211,    62,   198,    59,
     521,   173,    73,    74,   219,   219,   173,   222,    35,     3,
     598,    35,   412,    73,    74,     6,   470,    62,   233,   419,
      24,    12,     6,    59,    73,    74,   198,    63,    12,   431,
      24,   198,    59,   248,   249,    71,     3,     3,    71,   282,
      67,   282,    24,    67,    62,   475,   383,    67,   367,   398,
     222,   367,   224,   400,    62,   222,    66,   224,   377,   316,
      68,   377,   517,   206,   207,   280,    37,    66,   240,    63,
     242,    62,    59,   240,     9,   242,   248,    68,    60,    73,
      74,   248,    17,    62,    68,    35,    21,    59,    74,    68,
      59,    73,    74,    24,    64,    59,    31,    32,   270,    35,
       3,    35,    40,   270,    62,   320,    44,    40,    63,    59,
     282,    44,    59,   285,   286,   282,   316,    67,   285,   286,
     574,    24,    67,    59,    75,    59,   341,    35,    60,    64,
     373,    67,   373,    67,    59,    35,   279,   352,   568,   415,
      60,   112,   113,   114,   316,    24,   289,    50,    24,   316,
      72,   112,   113,   114,    72,   126,   371,     3,   373,    59,
       7,    24,    60,   378,   135,    35,   381,    67,   383,   426,
      73,    74,   143,   373,   135,    62,   391,    60,   383,   150,
     395,    60,   143,   398,    60,   400,   391,   359,    62,   150,
     395,    75,   359,    24,    73,    74,    59,    73,    74,   371,
     415,   373,   173,    59,   371,    68,   373,    62,    35,   381,
      73,    74,   173,   385,   381,    72,    65,    62,   385,     3,
     435,   436,   365,   366,    62,   468,   426,   468,   443,   444,
      62,   446,    59,   405,   406,    65,   451,    24,   405,   406,
      67,   484,   457,   484,    67,   460,   461,   566,    24,    62,
     566,   222,    66,   224,   426,   470,    59,    67,    67,   426,
     475,   404,    95,   224,    97,    98,    53,    71,   468,   240,
       7,   242,    59,    65,    62,   418,    62,   248,    63,   240,
      62,    68,    62,    59,   484,    60,    73,    74,    24,    94,
      95,    24,    97,    98,   494,    60,   468,    73,    74,   270,
      24,   468,    60,   475,    60,   581,    68,    60,   475,   270,
      35,   282,   484,    68,   285,   286,    60,   484,    60,    60,
      53,   282,   494,    59,   285,   286,    59,   494,   571,    75,
     571,    60,    75,    68,    60,    59,    68,    73,    74,    60,
      73,    74,    49,    24,    36,     3,   518,    62,    60,    73,
      74,   518,    72,   568,    61,    60,    75,    64,    62,   574,
      59,    62,    60,   620,    66,    60,   581,    60,    24,    60,
      60,   571,    72,    60,    59,    59,   210,    59,    59,   594,
      68,   596,   216,   217,    62,   600,   601,    62,   359,    72,
     605,   606,    73,    74,    68,   638,   568,   638,   359,   571,
     371,   568,   373,    59,   571,   210,    49,    62,    59,    34,
     381,   216,   217,    60,   385,    14,    68,    73,    74,    44,
     620,    60,    68,    48,   385,    60,    62,    60,    53,    54,
      55,    56,    60,    60,   405,   406,    60,   106,   638,    31,
      34,    22,   581,   614,   405,   406,   507,   376,   620,   523,
      44,   614,   141,   620,    48,    49,    50,    51,    52,    53,
      54,    55,    56,   242,   280,    39,   638,   174,   175,   176,
     177,   638,   179,   180,   181,   242,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     158,   198,   391,   200,   212,   202,   371,   468,   383,   206,
     207,   208,    22,     3,   475,   494,   340,     7,   333,   439,
     600,    11,   333,   484,   348,   461,   596,    17,   457,    -1,
      -1,    -1,    22,    23,    24,   211,    -1,    -1,   333,    29,
      -1,    -1,    -1,    -1,    -1,   340,    36,    -1,    -1,    -1,
      -1,    -1,    -1,   348,    -1,    -1,    -1,   518,    -1,    49,
      50,    -1,    52,    53,    -1,    -1,    56,   518,    -1,    59,
      -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,     7,    69,
      70,    71,   279,    73,    74,    -1,    -1,    -1,    17,     7,
      -1,    -1,   289,    22,    23,    24,    -1,    -1,    -1,    17,
      29,    -1,    -1,    -1,    22,    23,    24,   568,    -1,    -1,
     571,    29,    -1,    -1,   438,   312,    -1,    -1,    36,   316,
     444,   445,    -1,   447,    -1,   322,    -1,    -1,    -1,    -1,
      59,    -1,   456,    -1,   458,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    71,   438,    73,    74,    -1,    65,    -1,   444,
     445,     4,   447,    71,    -1,     8,    74,    -1,    -1,    -1,
      -1,   456,    -1,   458,    -1,    -1,    -1,    -1,   365,   366,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   638,    -1,    -1,
      33,    34,    -1,    36,    37,    38,    39,    40,    -1,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    -1,    -1,    -1,   404,    -1,    -1,
      -1,    -1,    65,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   418,    75,   547,    -1,    -1,    -1,    -1,    -1,   426,
      -1,   555,     7,    -1,   431,    -1,    11,     3,    -1,    -1,
      -1,     7,    17,    -1,    -1,    -1,    -1,    22,    23,    24,
      -1,    17,   547,    -1,    29,    -1,    22,    23,    24,    -1,
     555,    36,    -1,    29,    -1,    -1,    -1,   562,    -1,    -1,
      36,    -1,   596,   597,   471,   472,    -1,    -1,    53,    -1,
      -1,    -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,
      56,    -1,    -1,    59,    60,    -1,    71,   494,   593,    74,
      -1,   596,   597,    69,    70,    71,    -1,    73,    74,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   514,    -1,    -1,
      -1,    -1,    -1,    -1,   521,   522,   523,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     0,     1,    -1,     3,
      -1,     5,    -1,     7,    -1,     9,    10,    -1,   643,    13,
     645,    15,    16,    17,    18,    19,    20,    -1,    22,    23,
      24,    -1,    -1,    27,    28,    29,    30,    31,    32,    -1,
      -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    49,    50,    -1,    52,    53,
      -1,    -1,    56,    -1,    -1,    59,    -1,    -1,    62,    -1,
      -1,    -1,    -1,    -1,    -1,    69,    70,    71,    -1,    73,
      74,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     617,   618,     1,   620,     3,    -1,     5,     6,     7,    -1,
       9,    10,    -1,    12,    13,    -1,    15,    16,    17,    18,
      19,    20,    -1,    22,    23,    24,    -1,    -1,    27,    28,
      29,    30,    31,    32,    -1,    -1,    -1,    36,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      49,    50,    -1,    52,    53,    -1,    -1,    56,    -1,    -1,
      59,    -1,    -1,    62,    -1,    -1,    -1,    -1,    67,    68,
      69,    70,    71,    -1,    73,    74,     1,    -1,     3,    -1,
       5,    -1,     7,    -1,     9,    10,    -1,    -1,    13,    -1,
      15,    16,    17,    18,    19,    20,    -1,    22,    23,    24,
      -1,    -1,    27,    28,    29,    30,    31,    32,    -1,    -1,
      -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    50,    -1,    52,    53,    -1,
      -1,    56,    -1,    -1,    59,    -1,     3,    62,    -1,    -1,
       7,    -1,    67,    68,    69,    70,    71,    -1,    73,    74,
      17,     7,    -1,    -1,    -1,    22,    23,    24,    -1,    -1,
      -1,    17,    29,    -1,    -1,    -1,    22,    23,    24,    36,
      -1,    -1,    -1,    29,    -1,    -1,    -1,    -1,    -1,    -1,
      36,    -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,     3,    59,    -1,    -1,     7,    -1,    53,    -1,    -1,
      67,    -1,    69,    70,    71,    17,    73,    74,    -1,    -1,
      22,    23,    24,    -1,    -1,    71,    -1,    29,    74,    31,
      -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,
      52,    53,    -1,    -1,    56,    -1,     3,    59,    -1,    -1,
       7,    -1,    -1,    -1,    -1,    -1,    -1,    69,    70,    71,
      17,    73,    74,    -1,    -1,    22,    23,    24,    -1,    26,
      -1,    -1,    29,    -1,    -1,    -1,    -1,    -1,    -1,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,     3,    59,    -1,    -1,     7,    -1,    -1,    -1,    -1,
      -1,    -1,    69,    70,    71,    17,    73,    74,    -1,    -1,
      22,    23,    24,    -1,    26,    -1,    -1,    29,    -1,    -1,
      -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,
      52,    53,    -1,    -1,    56,    -1,     3,    59,    -1,    -1,
       7,    -1,    -1,    -1,    -1,    -1,    -1,    69,    70,    71,
      17,    73,    74,    -1,    -1,    22,    23,    24,    -1,    -1,
      -1,    -1,    29,    -1,    -1,    -1,    -1,    -1,    -1,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,     3,    59,    -1,    -1,     7,    -1,    -1,    -1,    -1,
      -1,    -1,    69,    70,    71,    17,    73,    74,    -1,    -1,
      22,    23,    24,    -1,    -1,    -1,    -1,    29,    -1,    -1,
      -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,
      52,    53,    -1,    -1,    56,    -1,     3,    59,    -1,    -1,
       7,    -1,    -1,    -1,    -1,    -1,    -1,    69,    70,    71,
      17,    73,    74,    -1,    -1,    22,    23,    24,    -1,    -1,
      -1,    -1,    29,    -1,    -1,    -1,    -1,    -1,    -1,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,    -1,    59,    -1,    -1,     7,    -1,    -1,    -1,    -1,
      -1,    -1,    69,    70,    71,    17,    73,    74,    -1,    -1,
      22,    23,    24,    -1,    -1,    -1,    -1,    29,    -1,    -1,
       7,    -1,    -1,    -1,    36,    -1,    -1,    -1,     7,    -1,
      17,    -1,    -1,    -1,    -1,    22,    23,    24,    17,    -1,
      -1,    53,    29,    22,    23,    24,    -1,    59,    -1,    36,
      29,    -1,    -1,    65,     7,    -1,    -1,    36,    11,    71,
      -1,    73,    74,    75,    17,    -1,    53,    -1,    -1,    22,
      23,    24,    59,    -1,    53,    -1,    29,    -1,    65,    -1,
      59,    -1,    -1,    36,    71,    -1,    73,    74,    75,    -1,
      -1,    -1,    71,     7,    73,    74,    75,     7,    -1,    -1,
      53,    -1,    -1,    17,    -1,    -1,    59,    17,    22,    23,
      24,    -1,    22,    23,    24,    29,    -1,    -1,    71,    29,
      73,    74,    36,    -1,    -1,     7,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    17,     7,    -1,    -1,    53,
      22,    23,    24,    53,    -1,    59,    17,    29,    -1,    59,
      -1,    22,    23,    24,    36,    -1,    -1,    71,    29,    73,
      74,    71,    -1,    73,    74,    36,    -1,    -1,     7,    -1,
      -1,    53,    -1,    -1,    -1,    -1,    -1,    59,    17,    -1,
      -1,    -1,    53,    22,    23,    24,    -1,    -1,    59,    71,
      29,    73,    74,    -1,     7,    -1,    -1,    36,    -1,    -1,
      71,    -1,    73,    74,    17,    -1,    -1,    -1,    -1,    22,
      23,    24,    -1,    -1,    53,    -1,    29,    -1,    -1,    -1,
      59,     7,    -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    17,    71,    -1,    -1,    74,    22,    23,    24,    -1,
      53,    -1,    -1,    29,    -1,    -1,    -1,    -1,    -1,    -1,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    -1,
      73,    74,    -1,    -1,    -1,    -1,    -1,    53,    -1,    -1,
      -1,    -1,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    71,    33,    34,    74,    36,
      -1,    38,    39,    40,    -1,    -1,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      -1,    -1,    -1,    -1,    -1,    -1,    33,    34,    -1,    36,
      -1,    38,    39,    40,    -1,    72,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      33,    34,    -1,    36,    -1,    38,    39,    40,    -1,    66,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    33,    34,    -1,    36,    -1,    38,
      39,    40,    -1,    66,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    33,    34,
      -1,    36,    -1,    38,    39,    40,    -1,    66,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    -1,    33,    34,    60,    36,    -1,    38,    39,
      40,    -1,    -1,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    33,    34,    -1,
      -1,    -1,    38,    39,    40,    -1,    -1,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    33,    34,    -1,    -1,    -1,    38,    39,    40,    -1,
      -1,    43,    44,    45,    46,    -1,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    34,    -1,    -1,    -1,    38,
      39,    40,    -1,    -1,    43,    44,    45,    46,    -1,    48,
      49,    50,    51,    52,    53,    54,    55,    56
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    77,    79,    80,     0,    25,    78,    25,    86,    24,
      73,    74,   137,   138,    81,    24,    88,    89,     3,    62,
      21,    82,   162,    24,    87,   210,    63,     3,    59,    63,
      83,    85,   137,    62,     1,     3,     5,     7,     9,    10,
      13,    15,    16,    17,    18,    19,    20,    22,    23,    27,
      28,    29,    30,    31,    32,    36,    49,    50,    52,    53,
      56,    59,    69,    70,    71,    90,    91,    92,    98,   110,
     113,   118,   121,   123,   124,   125,   126,   130,   134,   137,
     139,   140,   145,   146,   149,   152,   153,   154,   157,   160,
     161,   177,   182,    62,     9,    17,    21,    31,    32,    64,
     195,    24,    60,    83,    84,     3,    86,     3,   134,   136,
     137,    17,    36,    53,    59,   137,   139,   144,   148,   149,
     150,   157,   136,   125,   130,   111,    59,   137,   155,   125,
     134,   114,    35,    67,   133,    71,   123,   182,   189,   122,
     133,   119,    59,    96,    97,   137,    59,    93,   135,   137,
     181,   124,   124,   124,   124,   124,   124,    36,    53,   123,
     131,   143,   149,   151,   157,   124,   124,    11,   123,   188,
      62,    59,    94,   181,     4,    33,    34,    36,    37,    38,
      39,    40,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    67,    59,    63,
      71,    66,    59,   133,     1,   133,     8,    65,    75,   138,
     196,    59,   156,   196,    24,   196,   197,   196,    64,    62,
     186,    88,    59,    36,    59,   142,   148,   149,   150,   151,
     157,   142,   142,    63,    98,   107,   108,   109,   182,   190,
      11,   132,   137,   141,   142,   173,   174,   175,    59,    67,
     158,   112,   190,    24,    59,    68,   134,   167,   169,   171,
     142,    35,    53,    59,    68,   134,   166,   168,   169,   170,
     180,   112,    60,    97,   165,   142,    60,    93,   163,    65,
      75,   142,     7,   143,    60,    72,    72,    60,    94,    65,
     142,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   123,   127,    60,   131,   183,    59,   137,   123,   188,
     178,   123,   127,     1,    67,    91,   100,   176,   177,   179,
     182,   182,   123,     7,    17,    22,    23,    24,    29,    36,
      53,    65,    71,   138,   198,   200,   201,   202,   137,   203,
     211,   158,    59,     3,   198,   198,    83,    60,   175,     7,
     142,    60,   137,    35,   105,     8,    65,    62,   142,   132,
     141,    75,   187,    60,   175,   179,   115,    62,    63,    24,
     169,    59,   172,    62,   186,    72,   104,    59,   170,    53,
     170,    62,   186,     3,   194,    75,   142,   120,    62,   186,
      62,   186,   182,   135,    65,    36,    59,   142,   148,   149,
     150,   157,    67,   142,   142,    62,   186,   182,    65,    67,
     123,   128,   129,   184,   185,    11,    75,   187,    31,   131,
      72,    66,   176,    60,   185,   101,    62,    68,    36,    59,
     199,   200,   202,    59,    67,    71,    67,     7,   198,     3,
      50,    59,   137,   208,   209,     3,    72,    65,    11,   198,
      60,    75,    62,   191,   211,    62,    62,    62,    60,    60,
     106,    26,    26,   190,   173,    59,   137,   147,   148,   149,
     150,   151,   157,   159,    60,    68,   105,   190,   137,    60,
     175,   171,    68,   142,     6,    12,    68,    99,   102,   170,
     194,   170,    60,   168,    68,   134,   194,    35,    97,    60,
      93,    60,   182,   142,   127,    94,    95,   164,   181,    60,
     182,   127,    66,    75,   187,    68,    75,   187,   131,    60,
      60,    60,   188,    68,   179,   176,   198,   201,   191,    24,
     137,   138,   193,   198,   205,   213,   198,   137,   192,   204,
     212,   198,     3,   208,    62,    72,   198,   209,   198,   194,
     137,   203,    60,   179,   123,   123,    62,   175,    59,   159,
     116,    60,   183,    66,   103,    60,    60,   194,   104,    60,
     185,    62,   186,   142,   185,   123,   129,   128,   129,    60,
      72,    68,    60,    60,    59,    68,    62,    72,   198,    68,
      62,    49,   198,    62,   194,    59,    59,   198,   206,   207,
      68,   190,    60,   175,    14,   117,   159,     8,    65,    66,
      75,   179,   194,   194,    68,    68,    95,    60,    68,   206,
     191,   205,   198,   194,   204,   208,   191,   191,    60,   100,
     113,   123,   123,    60,    60,    60,    60,   159,    66,    66,
     206,   206
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

  case 15:
#line 216 "go.y"
    {
		// import with original name
		(yyval.i) = parserline();
		importmyname = S;
		importfile(&(yyvsp[(1) - (1)].val), (yyval.i));
	}
    break;

  case 16:
#line 223 "go.y"
    {
		// import with given name
		(yyval.i) = parserline();
		importmyname = (yyvsp[(1) - (2)].sym);
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 17:
#line 230 "go.y"
    {
		// import into my name space
		(yyval.i) = parserline();
		importmyname = lookup(".");
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 18:
#line 239 "go.y"
    {
		if(importpkg->name == nil) {
			importpkg->name = (yyvsp[(2) - (4)].sym)->name;
			pkglookup((yyvsp[(2) - (4)].sym)->name, nil)->npkg++;
		} else if(strcmp(importpkg->name, (yyvsp[(2) - (4)].sym)->name) != 0)
			yyerror("conflicting names %s and %s for package \"%Z\"", importpkg->name, (yyvsp[(2) - (4)].sym)->name, importpkg->path);
		importpkg->direct = 1;
		
		if(safemode && !curio.importsafe)
			yyerror("cannot import unsafe package \"%Z\"", importpkg->path);
	}
    break;

  case 20:
#line 253 "go.y"
    {
		if(strcmp((yyvsp[(1) - (1)].sym)->name, "safe") == 0)
			curio.importsafe = 1;
	}
    break;

  case 21:
#line 259 "go.y"
    {
		defercheckwidth();
	}
    break;

  case 22:
#line 263 "go.y"
    {
		resumecheckwidth();
		unimportfile();
	}
    break;

  case 23:
#line 272 "go.y"
    {
		yyerror("empty top-level declaration");
		(yyval.list) = nil;
	}
    break;

  case 25:
#line 278 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 26:
#line 282 "go.y"
    {
		yyerror("non-declaration statement outside function body");
		(yyval.list) = nil;
	}
    break;

  case 27:
#line 287 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 28:
#line 293 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
	}
    break;

  case 29:
#line 297 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 30:
#line 301 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 31:
#line 305 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 32:
#line 311 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 33:
#line 317 "go.y"
    {
		(yyval.list) = concat((yyvsp[(3) - (7)].list), (yyvsp[(5) - (7)].list));
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 34:
#line 323 "go.y"
    {
		(yyval.list) = nil;
		iota = -100000;
	}
    break;

  case 35:
#line 328 "go.y"
    {
		(yyval.list) = list1((yyvsp[(2) - (2)].node));
	}
    break;

  case 36:
#line 332 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 37:
#line 336 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 38:
#line 342 "go.y"
    {
		iota = 0;
	}
    break;

  case 39:
#line 348 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 40:
#line 352 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 41:
#line 356 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (3)].list), nil, (yyvsp[(3) - (3)].list));
	}
    break;

  case 42:
#line 362 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 43:
#line 366 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (3)].list), N, (yyvsp[(3) - (3)].list));
	}
    break;

  case 45:
#line 373 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 46:
#line 377 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (1)].list), N, nil);
	}
    break;

  case 47:
#line 383 "go.y"
    {
		// different from dclname because the name
		// becomes visible right here, not at the end
		// of the declaration.
		(yyval.node) = typedcl0((yyvsp[(1) - (1)].sym));
	}
    break;

  case 48:
#line 392 "go.y"
    {
		(yyval.node) = typedcl1((yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node), 1);
	}
    break;

  case 49:
#line 398 "go.y"
    {
		(yyval.node) = (yyvsp[(1) - (1)].node);
	}
    break;

  case 50:
#line 402 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
		(yyval.node)->etype = (yyvsp[(2) - (3)].i);			// rathole to pass opcode
	}
    break;

  case 51:
#line 407 "go.y"
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

  case 52:
#line 419 "go.y"
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
		(yyval.node) = colas((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 53:
#line 435 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OADD;
	}
    break;

  case 54:
#line 440 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OSUB;
	}
    break;

  case 55:
#line 447 "go.y"
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

  case 56:
#line 467 "go.y"
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

  case 57:
#line 485 "go.y"
    {
		// will be converted to OCASE
		// right will point to next case
		// done in casebody()
		markdcl();
		(yyval.node) = nod(OXCASE, N, N);
		(yyval.node)->list = list1(colas((yyvsp[(2) - (5)].list), list1((yyvsp[(4) - (5)].node))));
	}
    break;

  case 58:
#line 494 "go.y"
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

  case 59:
#line 512 "go.y"
    {
		markdcl();
	}
    break;

  case 60:
#line 516 "go.y"
    {
		(yyval.node) = liststmt((yyvsp[(3) - (4)].list));
		popdcl();
	}
    break;

  case 61:
#line 523 "go.y"
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

  case 62:
#line 533 "go.y"
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

  case 63:
#line 553 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 64:
#line 557 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node));
	}
    break;

  case 65:
#line 563 "go.y"
    {
		markdcl();
	}
    break;

  case 66:
#line 567 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (4)].list);
		popdcl();
	}
    break;

  case 67:
#line 574 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->etype = 0;	// := flag
	}
    break;

  case 68:
#line 580 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->colas = 1;
		colasdefn((yyvsp[(1) - (4)].list), (yyval.node));
	}
    break;

  case 69:
#line 589 "go.y"
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

  case 70:
#line 600 "go.y"
    {
		// normal test
		(yyval.node) = nod(OFOR, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 72:
#line 609 "go.y"
    {
		(yyval.node) = (yyvsp[(1) - (2)].node);
		(yyval.node)->nbody = concat((yyval.node)->nbody, (yyvsp[(2) - (2)].list));
	}
    break;

  case 73:
#line 616 "go.y"
    {
		markdcl();
	}
    break;

  case 74:
#line 620 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (3)].node);
		popdcl();
	}
    break;

  case 75:
#line 627 "go.y"
    {
		// test
		(yyval.node) = nod(OIF, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 76:
#line 633 "go.y"
    {
		// init ; test
		(yyval.node) = nod(OIF, N, N);
		if((yyvsp[(1) - (3)].node) != N)
			(yyval.node)->ninit = list1((yyvsp[(1) - (3)].node));
		(yyval.node)->ntest = (yyvsp[(3) - (3)].node);
	}
    break;

  case 77:
#line 644 "go.y"
    {
		markdcl();
	}
    break;

  case 78:
#line 648 "go.y"
    {
		if((yyvsp[(3) - (3)].node)->ntest == N)
			yyerror("missing condition in if statement");
	}
    break;

  case 79:
#line 653 "go.y"
    {
		(yyvsp[(3) - (5)].node)->nbody = (yyvsp[(5) - (5)].list);
	}
    break;

  case 80:
#line 657 "go.y"
    {
		popdcl();
		(yyval.node) = (yyvsp[(3) - (7)].node);
		if((yyvsp[(7) - (7)].node) != N)
			(yyval.node)->nelse = list1((yyvsp[(7) - (7)].node));
	}
    break;

  case 81:
#line 665 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 82:
#line 669 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (2)].node);
	}
    break;

  case 83:
#line 673 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (2)].node);
	}
    break;

  case 84:
#line 679 "go.y"
    {
		markdcl();
	}
    break;

  case 85:
#line 683 "go.y"
    {
		Node *n;
		n = (yyvsp[(3) - (3)].node)->ntest;
		if(n != N && n->op != OTYPESW)
			n = N;
		typesw = nod(OXXX, typesw, n);
	}
    break;

  case 86:
#line 691 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (7)].node);
		(yyval.node)->op = OSWITCH;
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
		typesw = typesw->left;
		popdcl();
	}
    break;

  case 87:
#line 701 "go.y"
    {
		typesw = nod(OXXX, typesw, N);
	}
    break;

  case 88:
#line 705 "go.y"
    {
		(yyval.node) = nod(OSELECT, N, N);
		(yyval.node)->lineno = typesw->lineno;
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		typesw = typesw->left;
	}
    break;

  case 90:
#line 718 "go.y"
    {
		(yyval.node) = nod(OOROR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 91:
#line 722 "go.y"
    {
		(yyval.node) = nod(OANDAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 92:
#line 726 "go.y"
    {
		(yyval.node) = nod(OEQ, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 93:
#line 730 "go.y"
    {
		(yyval.node) = nod(ONE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 94:
#line 734 "go.y"
    {
		(yyval.node) = nod(OLT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 95:
#line 738 "go.y"
    {
		(yyval.node) = nod(OLE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 96:
#line 742 "go.y"
    {
		(yyval.node) = nod(OGE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 97:
#line 746 "go.y"
    {
		(yyval.node) = nod(OGT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 98:
#line 750 "go.y"
    {
		(yyval.node) = nod(OADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 99:
#line 754 "go.y"
    {
		(yyval.node) = nod(OSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 100:
#line 758 "go.y"
    {
		(yyval.node) = nod(OOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 101:
#line 762 "go.y"
    {
		(yyval.node) = nod(OXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 102:
#line 766 "go.y"
    {
		(yyval.node) = nod(OMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 103:
#line 770 "go.y"
    {
		(yyval.node) = nod(ODIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 104:
#line 774 "go.y"
    {
		(yyval.node) = nod(OMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 105:
#line 778 "go.y"
    {
		(yyval.node) = nod(OAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 106:
#line 782 "go.y"
    {
		(yyval.node) = nod(OANDNOT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 107:
#line 786 "go.y"
    {
		(yyval.node) = nod(OLSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 108:
#line 790 "go.y"
    {
		(yyval.node) = nod(ORSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 109:
#line 795 "go.y"
    {
		(yyval.node) = nod(OSEND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 111:
#line 802 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 112:
#line 806 "go.y"
    {
		if((yyvsp[(2) - (2)].node)->op == OCOMPLIT) {
			// Special case for &T{...}: turn into (*T){...}.
			(yyval.node) = (yyvsp[(2) - (2)].node);
			(yyval.node)->right = nod(OIND, (yyval.node)->right, N);
			(yyval.node)->right->implicit = ImplPtr;
		} else {
			(yyval.node) = nod(OADDR, (yyvsp[(2) - (2)].node), N);
		}
	}
    break;

  case 113:
#line 817 "go.y"
    {
		(yyval.node) = nod(OPLUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 114:
#line 821 "go.y"
    {
		(yyval.node) = nod(OMINUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 115:
#line 825 "go.y"
    {
		(yyval.node) = nod(ONOT, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 116:
#line 829 "go.y"
    {
		yyerror("the bitwise complement operator is ^");
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 117:
#line 834 "go.y"
    {
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 118:
#line 838 "go.y"
    {
		(yyval.node) = nod(ORECV, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 119:
#line 848 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (3)].node), N);
	}
    break;

  case 120:
#line 852 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (5)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
	}
    break;

  case 121:
#line 857 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (6)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (6)].list);
		(yyval.node)->isddd = 1;
	}
    break;

  case 122:
#line 865 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 124:
#line 870 "go.y"
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

  case 125:
#line 881 "go.y"
    {
		(yyval.node) = nod(ODOTTYPE, (yyvsp[(1) - (5)].node), (yyvsp[(4) - (5)].node));
	}
    break;

  case 126:
#line 885 "go.y"
    {
		(yyval.node) = nod(OTYPESW, N, (yyvsp[(1) - (5)].node));
	}
    break;

  case 127:
#line 889 "go.y"
    {
		(yyval.node) = nod(OINDEX, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 128:
#line 893 "go.y"
    {
		(yyval.node) = nod(OSLICE, (yyvsp[(1) - (6)].node), nod(OKEY, (yyvsp[(3) - (6)].node), (yyvsp[(5) - (6)].node)));
	}
    break;

  case 130:
#line 898 "go.y"
    {
		// conversion
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (4)].node), N);
		(yyval.node)->list = list1((yyvsp[(3) - (4)].node));
	}
    break;

  case 131:
#line 904 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 132:
#line 911 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
	}
    break;

  case 133:
#line 917 "go.y"
    {
		yyerror("cannot parenthesize type in composite literal");
		(yyval.node) = (yyvsp[(5) - (7)].node);
		(yyval.node)->right = (yyvsp[(2) - (7)].node);
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
	}
    break;

  case 135:
#line 926 "go.y"
    {
		// composite expression.
		// make node early so we get the right line number.
		(yyval.node) = nod(OCOMPLIT, N, N);
	}
    break;

  case 136:
#line 934 "go.y"
    {
		(yyval.node) = nod(OKEY, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 138:
#line 941 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (4)].node);
		(yyval.node)->list = (yyvsp[(3) - (4)].list);
	}
    break;

  case 140:
#line 949 "go.y"
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
			(yyval.node) = nod(OPAREN, (yyval.node), N);
		}
	}
    break;

  case 144:
#line 974 "go.y"
    {
		(yyval.i) = LBODY;
	}
    break;

  case 145:
#line 978 "go.y"
    {
		(yyval.i) = '{';
	}
    break;

  case 146:
#line 989 "go.y"
    {
		(yyval.node) = newname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 147:
#line 995 "go.y"
    {
		(yyval.node) = dclname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 148:
#line 1000 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 150:
#line 1007 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if(importpkg != nil && !exportname((yyvsp[(1) - (1)].sym)->name))
			(yyval.sym) = pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg);
	}
    break;

  case 152:
#line 1015 "go.y"
    {
		(yyval.sym) = S;
	}
    break;

  case 153:
#line 1021 "go.y"
    {
		if((yyvsp[(2) - (4)].val).u.sval->len == 0)
			(yyval.sym) = pkglookup((yyvsp[(4) - (4)].sym)->name, importpkg);
		else
			(yyval.sym) = pkglookup((yyvsp[(4) - (4)].sym)->name, mkpkg((yyvsp[(2) - (4)].val).u.sval));
	}
    break;

  case 154:
#line 1030 "go.y"
    {
		(yyval.node) = oldname((yyvsp[(1) - (1)].sym));
		if((yyval.node)->pack != N)
			(yyval.node)->pack->used = 1;
	}
    break;

  case 156:
#line 1050 "go.y"
    {
		yyerror("final argument in variadic function missing type");
		(yyval.node) = nod(ODDD, typenod(typ(TINTER)), N);
	}
    break;

  case 157:
#line 1055 "go.y"
    {
		(yyval.node) = nod(ODDD, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 163:
#line 1066 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 167:
#line 1075 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 172:
#line 1085 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 182:
#line 1106 "go.y"
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

  case 183:
#line 1119 "go.y"
    {
		(yyval.node) = nod(OTARRAY, (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].node));
	}
    break;

  case 184:
#line 1123 "go.y"
    {
		// array literal of nelem
		(yyval.node) = nod(OTARRAY, nod(ODDD, N, N), (yyvsp[(4) - (4)].node));
	}
    break;

  case 185:
#line 1128 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->etype = Cboth;
	}
    break;

  case 186:
#line 1133 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Csend;
	}
    break;

  case 187:
#line 1138 "go.y"
    {
		(yyval.node) = nod(OTMAP, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node));
	}
    break;

  case 190:
#line 1146 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 191:
#line 1152 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Crecv;
	}
    break;

  case 192:
#line 1159 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 193:
#line 1165 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 194:
#line 1172 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 195:
#line 1178 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 196:
#line 1189 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
		if((yyval.node) == N)
			break;
		(yyval.node)->nbody = (yyvsp[(3) - (3)].list);
		(yyval.node)->endlineno = lineno;
		funcbody((yyval.node));
	}
    break;

  case 197:
#line 1200 "go.y"
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

  case 198:
#line 1229 "go.y"
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
		declare((yyval.node)->nname, PFUNC);

		funchdr((yyval.node));
	}
    break;

  case 199:
#line 1268 "go.y"
    {
		Sym *s;
		Type *t;

		(yyval.node) = N;

		s = (yyvsp[(1) - (5)].sym);
		t = functype(N, (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list));

		importsym(s, ONAME);
		if(s->def != N && s->def->op == ONAME) {
			if(eqtype(t, s->def->type))
				break;
			yyerror("inconsistent definition for func %S during import\n\t%T\n\t%T", s, s->def->type, t);
		}

		(yyval.node) = newname(s);
		(yyval.node)->type = t;
		declare((yyval.node), PFUNC);

		funchdr((yyval.node));
	}
    break;

  case 200:
#line 1291 "go.y"
    {
		(yyval.node) = methodname1(newname((yyvsp[(4) - (8)].sym)), (yyvsp[(2) - (8)].list)->n->right); 
		(yyval.node)->type = functype((yyvsp[(2) - (8)].list)->n, (yyvsp[(6) - (8)].list), (yyvsp[(8) - (8)].list));

		checkwidth((yyval.node)->type);
		addmethod((yyvsp[(4) - (8)].sym), (yyval.node)->type, 0);
		funchdr((yyval.node));
		
		// inl.c's inlnode in on a dotmeth node expects to find the inlineable body as
		// (dotmeth's type)->nname->inl, and dotmeth's type has been pulled
		// out by typecheck's lookdot as this $$->ttype.  So by providing
		// this back link here we avoid special casing there.
		(yyval.node)->type->nname = (yyval.node);
	}
    break;

  case 201:
#line 1308 "go.y"
    {
		(yyvsp[(3) - (5)].list) = checkarglist((yyvsp[(3) - (5)].list), 1);
		(yyval.node) = nod(OTFUNC, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		(yyval.node)->rlist = (yyvsp[(5) - (5)].list);
	}
    break;

  case 202:
#line 1316 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 203:
#line 1320 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
		if((yyval.list) == nil)
			(yyval.list) = list1(nod(OEMPTY, N, N));
	}
    break;

  case 204:
#line 1328 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 205:
#line 1332 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, (yyvsp[(1) - (1)].node)));
	}
    break;

  case 206:
#line 1336 "go.y"
    {
		(yyvsp[(2) - (3)].list) = checkarglist((yyvsp[(2) - (3)].list), 0);
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 207:
#line 1343 "go.y"
    {
		closurehdr((yyvsp[(1) - (1)].node));
	}
    break;

  case 208:
#line 1349 "go.y"
    {
		(yyval.node) = closurebody((yyvsp[(3) - (4)].list));
		fixlbrace((yyvsp[(2) - (4)].i));
	}
    break;

  case 209:
#line 1354 "go.y"
    {
		(yyval.node) = closurebody(nil);
	}
    break;

  case 210:
#line 1365 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 211:
#line 1369 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(2) - (3)].list));
		if(nsyntaxerrors == 0)
			testdclstack();
	}
    break;

  case 213:
#line 1378 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 215:
#line 1385 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 216:
#line 1391 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 217:
#line 1395 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 219:
#line 1402 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 220:
#line 1408 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 221:
#line 1412 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 222:
#line 1418 "go.y"
    {
		NodeList *l;

		for(l=(yyvsp[(1) - (3)].list); l; l=l->next) {
			l->n = nod(ODCLFIELD, l->n, (yyvsp[(2) - (3)].node));
			l->n->val = (yyvsp[(3) - (3)].val);
		}
	}
    break;

  case 223:
#line 1427 "go.y"
    {
		(yyvsp[(1) - (2)].node)->val = (yyvsp[(2) - (2)].val);
		(yyval.list) = list1((yyvsp[(1) - (2)].node));
	}
    break;

  case 224:
#line 1432 "go.y"
    {
		(yyvsp[(2) - (4)].node)->val = (yyvsp[(4) - (4)].val);
		(yyval.list) = list1((yyvsp[(2) - (4)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 225:
#line 1438 "go.y"
    {
		(yyvsp[(2) - (3)].node)->right = nod(OIND, (yyvsp[(2) - (3)].node)->right, N);
		(yyvsp[(2) - (3)].node)->val = (yyvsp[(3) - (3)].val);
		(yyval.list) = list1((yyvsp[(2) - (3)].node));
	}
    break;

  case 226:
#line 1444 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 227:
#line 1451 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 228:
#line 1460 "go.y"
    {
		Node *n;

		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		n = oldname((yyvsp[(1) - (1)].sym));
		if(n->pack != N)
			n->pack->used = 1;
	}
    break;

  case 229:
#line 1469 "go.y"
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

  case 230:
#line 1484 "go.y"
    {
		(yyval.node) = embedded((yyvsp[(1) - (1)].sym));
	}
    break;

  case 231:
#line 1490 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
		ifacedcl((yyval.node));
	}
    break;

  case 232:
#line 1495 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(1) - (1)].sym)));
	}
    break;

  case 233:
#line 1499 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(2) - (3)].sym)));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 234:
#line 1506 "go.y"
    {
		// without func keyword
		(yyvsp[(2) - (4)].list) = checkarglist((yyvsp[(2) - (4)].list), 1);
		(yyval.node) = nod(OTFUNC, fakethis(), N);
		(yyval.node)->list = (yyvsp[(2) - (4)].list);
		(yyval.node)->rlist = (yyvsp[(4) - (4)].list);
	}
    break;

  case 236:
#line 1520 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 237:
#line 1526 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 239:
#line 1535 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 240:
#line 1539 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 241:
#line 1544 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 242:
#line 1548 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 243:
#line 1556 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 245:
#line 1561 "go.y"
    {
		(yyval.node) = liststmt((yyvsp[(1) - (1)].list));
	}
    break;

  case 247:
#line 1566 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 253:
#line 1577 "go.y"
    {
		(yyvsp[(1) - (2)].node) = nod(OLABEL, (yyvsp[(1) - (2)].node), N);
		(yyvsp[(1) - (2)].node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 254:
#line 1582 "go.y"
    {
		NodeList *l;

		(yyvsp[(1) - (4)].node)->defn = (yyvsp[(4) - (4)].node);
		l = list1((yyvsp[(1) - (4)].node));
		if((yyvsp[(4) - (4)].node))
			l = list(l, (yyvsp[(4) - (4)].node));
		(yyval.node) = liststmt(l);
	}
    break;

  case 255:
#line 1592 "go.y"
    {
		// will be converted to OFALL
		(yyval.node) = nod(OXFALL, N, N);
	}
    break;

  case 256:
#line 1597 "go.y"
    {
		(yyval.node) = nod(OBREAK, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 257:
#line 1601 "go.y"
    {
		(yyval.node) = nod(OCONTINUE, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 258:
#line 1605 "go.y"
    {
		(yyval.node) = nod(OPROC, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 259:
#line 1609 "go.y"
    {
		(yyval.node) = nod(ODEFER, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 260:
#line 1613 "go.y"
    {
		(yyval.node) = nod(OGOTO, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 261:
#line 1618 "go.y"
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

  case 262:
#line 1637 "go.y"
    {
		(yyval.list) = nil;
		if((yyvsp[(1) - (1)].node) != N)
			(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 263:
#line 1643 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (3)].list);
		if((yyvsp[(3) - (3)].node) != N)
			(yyval.list) = list((yyval.list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 264:
#line 1651 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 265:
#line 1655 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 266:
#line 1661 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 267:
#line 1665 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 268:
#line 1671 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 269:
#line 1675 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 270:
#line 1681 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 271:
#line 1685 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 272:
#line 1694 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 273:
#line 1698 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 274:
#line 1702 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 275:
#line 1706 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 276:
#line 1711 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 277:
#line 1715 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 282:
#line 1729 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 284:
#line 1735 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 286:
#line 1741 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 288:
#line 1747 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 290:
#line 1753 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 292:
#line 1759 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 294:
#line 1765 "go.y"
    {
		(yyval.val).ctype = CTxxx;
	}
    break;

  case 296:
#line 1775 "go.y"
    {
		importimport((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].val).u.sval);
	}
    break;

  case 297:
#line 1779 "go.y"
    {
		importvar((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].type));
	}
    break;

  case 298:
#line 1783 "go.y"
    {
		importconst((yyvsp[(2) - (5)].sym), types[TIDEAL], (yyvsp[(4) - (5)].node));
	}
    break;

  case 299:
#line 1787 "go.y"
    {
		importconst((yyvsp[(2) - (6)].sym), (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].node));
	}
    break;

  case 300:
#line 1791 "go.y"
    {
		importtype((yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].type));
	}
    break;

  case 301:
#line 1795 "go.y"
    {
		if((yyvsp[(2) - (4)].node) == N)
			break;

		(yyvsp[(2) - (4)].node)->inl = (yyvsp[(3) - (4)].list);

		funcbody((yyvsp[(2) - (4)].node));
		importlist = list(importlist, (yyvsp[(2) - (4)].node));

		if(debug['E']) {
			print("import [%Z] func %lN \n", importpkg->path, (yyvsp[(2) - (4)].node));
			if(debug['l'] > 2 && (yyvsp[(2) - (4)].node)->inl)
				print("inl body:%+H\n", (yyvsp[(2) - (4)].node)->inl);
		}
	}
    break;

  case 302:
#line 1813 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		structpkg = (yyval.sym)->pkg;
	}
    break;

  case 303:
#line 1820 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
		importsym((yyvsp[(1) - (1)].sym), OTYPE);
	}
    break;

  case 309:
#line 1840 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
	}
    break;

  case 310:
#line 1844 "go.y"
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

  case 311:
#line 1854 "go.y"
    {
		(yyval.type) = aindex(N, (yyvsp[(3) - (3)].type));
	}
    break;

  case 312:
#line 1858 "go.y"
    {
		(yyval.type) = aindex(nodlit((yyvsp[(2) - (4)].val)), (yyvsp[(4) - (4)].type));
	}
    break;

  case 313:
#line 1862 "go.y"
    {
		(yyval.type) = maptype((yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].type));
	}
    break;

  case 314:
#line 1866 "go.y"
    {
		(yyval.type) = tostruct((yyvsp[(3) - (4)].list));
	}
    break;

  case 315:
#line 1870 "go.y"
    {
		(yyval.type) = tointerface((yyvsp[(3) - (4)].list));
	}
    break;

  case 316:
#line 1874 "go.y"
    {
		(yyval.type) = ptrto((yyvsp[(2) - (2)].type));
	}
    break;

  case 317:
#line 1878 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(2) - (2)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 318:
#line 1884 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (4)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 319:
#line 1890 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Csend;
	}
    break;

  case 320:
#line 1898 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Crecv;
	}
    break;

  case 321:
#line 1906 "go.y"
    {
		(yyval.type) = functype(nil, (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list));
	}
    break;

  case 322:
#line 1912 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(2) - (3)].type)));
		if((yyvsp[(1) - (3)].sym))
			(yyval.node)->left = newname((yyvsp[(1) - (3)].sym));
		(yyval.node)->val = (yyvsp[(3) - (3)].val);
	}
    break;

  case 323:
#line 1919 "go.y"
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

  case 324:
#line 1935 "go.y"
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

  case 325:
#line 1953 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, newname((yyvsp[(1) - (5)].sym)), typenod(functype(fakethis(), (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list))));
	}
    break;

  case 326:
#line 1957 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type)));
	}
    break;

  case 327:
#line 1962 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 329:
#line 1969 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 330:
#line 1973 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type))));
	}
    break;

  case 331:
#line 1983 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 332:
#line 1987 "go.y"
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

  case 333:
#line 2002 "go.y"
    {
		(yyval.node) = oldname(pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg));
		if((yyval.node)->op != OLITERAL)
			yyerror("bad constant %S", (yyval.node)->sym);
	}
    break;

  case 335:
#line 2011 "go.y"
    {
		if((yyvsp[(2) - (5)].node)->val.ctype == CTRUNE && (yyvsp[(4) - (5)].node)->val.ctype == CTINT) {
			(yyval.node) = (yyvsp[(2) - (5)].node);
			mpaddfixfix((yyvsp[(2) - (5)].node)->val.u.xval, (yyvsp[(4) - (5)].node)->val.u.xval);
			break;
		}
		(yyval.node) = nodcplxlit((yyvsp[(2) - (5)].node)->val, (yyvsp[(4) - (5)].node)->val);
	}
    break;

  case 338:
#line 2025 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 339:
#line 2029 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 340:
#line 2035 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 341:
#line 2039 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 342:
#line 2045 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 343:
#line 2049 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;


/* Line 1267 of yacc.c.  */
#line 4702 "y.tab.c"
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


#line 2053 "go.y"


static void
fixlbrace(int lbr)
{
	// If the opening brace was an LBODY,
	// set up for another one now that we're done.
	// See comment in lex.c about loophack.
	if(lbr == LBODY)
		loophack = 1;
}


