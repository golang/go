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
#define YYLAST   2270

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  76
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  142
/* YYNRULES -- Number of rules.  */
#define YYNRULES  351
/* YYNRULES -- Number of states.  */
#define YYNSTATES  667

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
     439,   445,   450,   457,   466,   468,   474,   480,   486,   494,
     496,   497,   501,   503,   508,   510,   515,   517,   521,   523,
     525,   527,   529,   531,   533,   535,   536,   538,   540,   542,
     544,   549,   554,   556,   558,   560,   563,   565,   567,   569,
     571,   573,   577,   579,   581,   583,   586,   588,   590,   592,
     594,   598,   600,   602,   604,   606,   608,   610,   612,   614,
     616,   620,   625,   630,   633,   637,   643,   645,   647,   650,
     654,   660,   664,   670,   674,   678,   684,   693,   699,   708,
     714,   715,   719,   720,   722,   726,   728,   733,   736,   737,
     741,   743,   747,   749,   753,   755,   759,   761,   765,   767,
     771,   775,   778,   783,   787,   793,   799,   801,   805,   807,
     810,   812,   816,   821,   823,   826,   829,   831,   833,   837,
     838,   841,   842,   844,   846,   848,   850,   852,   854,   856,
     858,   860,   861,   866,   868,   871,   874,   877,   880,   883,
     886,   888,   892,   894,   898,   900,   904,   906,   910,   912,
     916,   918,   920,   924,   928,   929,   932,   933,   935,   936,
     938,   939,   941,   942,   944,   945,   947,   948,   950,   951,
     953,   954,   956,   957,   959,   964,   969,   975,   982,   987,
     992,   994,   996,   998,  1000,  1002,  1004,  1006,  1008,  1010,
    1014,  1019,  1025,  1030,  1035,  1038,  1041,  1046,  1050,  1054,
    1060,  1064,  1069,  1073,  1079,  1081,  1082,  1084,  1088,  1090,
    1092,  1095,  1097,  1099,  1105,  1106,  1109,  1111,  1115,  1117,
    1121,  1123
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
     134,    71,   192,    66,   192,    72,    -1,   134,    71,   192,
      66,   192,    66,   192,    72,    -1,   128,    -1,   149,    59,
     126,   191,    60,    -1,   150,   137,   130,   189,    68,    -1,
     129,    67,   130,   189,    68,    -1,    59,   135,    60,    67,
     130,   189,    68,    -1,   165,    -1,    -1,   126,    66,   133,
      -1,   126,    -1,    67,   130,   189,    68,    -1,   126,    -1,
      67,   130,   189,    68,    -1,   129,    -1,    59,   135,    60,
      -1,   126,    -1,   147,    -1,   146,    -1,    35,    -1,    67,
      -1,   141,    -1,   141,    -1,    -1,   138,    -1,    24,    -1,
     142,    -1,    73,    -1,    74,     3,    63,    24,    -1,    74,
       3,    63,    73,    -1,   141,    -1,   138,    -1,    11,    -1,
      11,   146,    -1,   155,    -1,   161,    -1,   153,    -1,   154,
      -1,   152,    -1,    59,   146,    60,    -1,   155,    -1,   161,
      -1,   153,    -1,    53,   147,    -1,   161,    -1,   153,    -1,
     154,    -1,   152,    -1,    59,   146,    60,    -1,   161,    -1,
     153,    -1,   153,    -1,   155,    -1,   161,    -1,   153,    -1,
     154,    -1,   152,    -1,   143,    -1,   143,    63,   141,    -1,
      71,   192,    72,   146,    -1,    71,    11,    72,   146,    -1,
       8,   148,    -1,     8,    36,   146,    -1,    23,    71,   146,
      72,   146,    -1,   156,    -1,   157,    -1,    53,   146,    -1,
      36,     8,   146,    -1,    29,   137,   170,   190,    68,    -1,
      29,   137,    68,    -1,    22,   137,   171,   190,    68,    -1,
      22,   137,    68,    -1,    17,   159,   162,    -1,   141,    59,
     179,    60,   163,    -1,    59,   179,    60,   141,    59,   179,
      60,   163,    -1,   200,    59,   195,    60,   210,    -1,    59,
     215,    60,   141,    59,   195,    60,   210,    -1,    17,    59,
     179,    60,   163,    -1,    -1,    67,   183,    68,    -1,    -1,
     151,    -1,    59,   179,    60,    -1,   161,    -1,   164,   137,
     183,    68,    -1,   164,     1,    -1,    -1,   166,    90,    62,
      -1,    93,    -1,   167,    62,    93,    -1,    95,    -1,   168,
      62,    95,    -1,    97,    -1,   169,    62,    97,    -1,   172,
      -1,   170,    62,   172,    -1,   175,    -1,   171,    62,   175,
      -1,   184,   146,   198,    -1,   174,   198,    -1,    59,   174,
      60,   198,    -1,    53,   174,   198,    -1,    59,    53,   174,
      60,   198,    -1,    53,    59,   174,    60,   198,    -1,    24,
      -1,    24,    63,   141,    -1,   173,    -1,   138,   176,    -1,
     173,    -1,    59,   173,    60,    -1,    59,   179,    60,   163,
      -1,   136,    -1,   141,   136,    -1,   141,   145,    -1,   145,
      -1,   177,    -1,   178,    75,   177,    -1,    -1,   178,   191,
      -1,    -1,   100,    -1,    91,    -1,   181,    -1,     1,    -1,
      98,    -1,   110,    -1,   121,    -1,   124,    -1,   113,    -1,
      -1,   144,    66,   182,   180,    -1,    15,    -1,     6,   140,
      -1,    10,   140,    -1,    18,   128,    -1,    13,   128,    -1,
      19,   138,    -1,    27,   193,    -1,   180,    -1,   183,    62,
     180,    -1,   138,    -1,   184,    75,   138,    -1,   139,    -1,
     185,    75,   139,    -1,   126,    -1,   186,    75,   126,    -1,
     135,    -1,   187,    75,   135,    -1,   131,    -1,   132,    -1,
     188,    75,   131,    -1,   188,    75,   132,    -1,    -1,   188,
     191,    -1,    -1,    62,    -1,    -1,    75,    -1,    -1,   126,
      -1,    -1,   186,    -1,    -1,    98,    -1,    -1,   215,    -1,
      -1,   216,    -1,    -1,   217,    -1,    -1,     3,    -1,    21,
      24,     3,    62,    -1,    32,   200,   202,    62,    -1,     9,
     200,    65,   213,    62,    -1,     9,   200,   202,    65,   213,
      62,    -1,    31,   201,   202,    62,    -1,    17,   160,   162,
      62,    -1,   142,    -1,   200,    -1,   204,    -1,   205,    -1,
     206,    -1,   204,    -1,   206,    -1,   142,    -1,    24,    -1,
      71,    72,   202,    -1,    71,     3,    72,   202,    -1,    23,
      71,   202,    72,   202,    -1,    29,    67,   196,    68,    -1,
      22,    67,   197,    68,    -1,    53,   202,    -1,     8,   203,
      -1,     8,    59,   205,    60,    -1,     8,    36,   202,    -1,
      36,     8,   202,    -1,    17,    59,   195,    60,   210,    -1,
     141,   202,   198,    -1,   141,    11,   202,   198,    -1,   141,
     202,   198,    -1,   141,    59,   195,    60,   210,    -1,   202,
      -1,    -1,   211,    -1,    59,   195,    60,    -1,   202,    -1,
       3,    -1,    50,     3,    -1,   141,    -1,   212,    -1,    59,
     212,    49,   212,    60,    -1,    -1,   214,   199,    -1,   207,
      -1,   215,    75,   207,    -1,   208,    -1,   216,    62,   208,
      -1,   209,    -1,   217,    62,   209,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   124,   124,   133,   139,   150,   150,   165,   166,   169,
     170,   171,   174,   211,   222,   223,   226,   233,   240,   249,
     263,   264,   271,   271,   284,   288,   289,   293,   298,   304,
     308,   312,   316,   322,   328,   334,   339,   343,   347,   353,
     359,   363,   367,   373,   377,   383,   384,   388,   394,   403,
     409,   427,   432,   444,   460,   465,   472,   492,   510,   519,
     538,   537,   552,   551,   583,   586,   593,   592,   603,   609,
     618,   629,   635,   638,   646,   645,   656,   662,   674,   678,
     683,   673,   704,   703,   716,   719,   725,   728,   740,   744,
     739,   762,   761,   777,   778,   782,   786,   790,   794,   798,
     802,   806,   810,   814,   818,   822,   826,   830,   834,   838,
     842,   846,   850,   855,   861,   862,   866,   877,   881,   885,
     889,   894,   898,   908,   912,   917,   925,   929,   930,   941,
     945,   949,   953,   957,   965,   966,   972,   979,   985,   992,
     995,  1002,  1008,  1025,  1032,  1033,  1040,  1041,  1060,  1061,
    1064,  1067,  1071,  1082,  1091,  1097,  1100,  1103,  1110,  1111,
    1117,  1130,  1145,  1153,  1165,  1170,  1176,  1177,  1178,  1179,
    1180,  1181,  1187,  1188,  1189,  1190,  1196,  1197,  1198,  1199,
    1200,  1206,  1207,  1210,  1213,  1214,  1215,  1216,  1217,  1220,
    1221,  1234,  1238,  1243,  1248,  1253,  1257,  1258,  1261,  1267,
    1274,  1280,  1287,  1293,  1304,  1318,  1347,  1387,  1412,  1430,
    1439,  1442,  1450,  1454,  1458,  1465,  1471,  1476,  1488,  1491,
    1501,  1502,  1508,  1509,  1515,  1519,  1525,  1526,  1532,  1536,
    1542,  1565,  1570,  1576,  1582,  1589,  1598,  1607,  1622,  1628,
    1633,  1637,  1644,  1657,  1658,  1664,  1670,  1673,  1677,  1683,
    1686,  1695,  1698,  1699,  1703,  1704,  1710,  1711,  1712,  1713,
    1714,  1716,  1715,  1730,  1736,  1740,  1744,  1748,  1752,  1757,
    1776,  1782,  1790,  1794,  1800,  1804,  1810,  1814,  1820,  1824,
    1833,  1837,  1841,  1845,  1851,  1854,  1862,  1863,  1865,  1866,
    1869,  1872,  1875,  1878,  1881,  1884,  1887,  1890,  1893,  1896,
    1899,  1902,  1905,  1908,  1914,  1918,  1922,  1926,  1930,  1934,
    1954,  1961,  1972,  1973,  1974,  1977,  1978,  1981,  1985,  1995,
    1999,  2003,  2007,  2011,  2015,  2019,  2025,  2031,  2039,  2047,
    2053,  2060,  2076,  2098,  2102,  2108,  2111,  2114,  2118,  2128,
    2132,  2151,  2159,  2160,  2172,  2173,  2176,  2180,  2186,  2190,
    2196,  2200
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
     129,   129,   129,   129,   129,   129,   129,   129,   129,   129,
     130,   131,   132,   132,   133,   133,   134,   134,   135,   135,
     136,   137,   137,   138,   139,   140,   140,   141,   141,   141,
     142,   142,   143,   144,   145,   145,   146,   146,   146,   146,
     146,   146,   147,   147,   147,   147,   148,   148,   148,   148,
     148,   149,   149,   150,   151,   151,   151,   151,   151,   152,
     152,   153,   153,   153,   153,   153,   153,   153,   154,   155,
     156,   156,   157,   157,   158,   159,   159,   160,   160,   161,
     162,   162,   163,   163,   163,   164,   165,   165,   166,   166,
     167,   167,   168,   168,   169,   169,   170,   170,   171,   171,
     172,   172,   172,   172,   172,   172,   173,   173,   174,   175,
     175,   175,   176,   177,   177,   177,   177,   178,   178,   179,
     179,   180,   180,   180,   180,   180,   181,   181,   181,   181,
     181,   182,   181,   181,   181,   181,   181,   181,   181,   181,
     183,   183,   184,   184,   185,   185,   186,   186,   187,   187,
     188,   188,   188,   188,   189,   189,   190,   190,   191,   191,
     192,   192,   193,   193,   194,   194,   195,   195,   196,   196,
     197,   197,   198,   198,   199,   199,   199,   199,   199,   199,
     200,   201,   202,   202,   202,   203,   203,   204,   204,   204,
     204,   204,   204,   204,   204,   204,   204,   204,   205,   206,
     207,   207,   208,   209,   209,   210,   210,   211,   211,   212,
     212,   212,   213,   213,   214,   214,   215,   215,   216,   216,
     217,   217
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
       5,     4,     6,     8,     1,     5,     5,     5,     7,     1,
       0,     3,     1,     4,     1,     4,     1,     3,     1,     1,
       1,     1,     1,     1,     1,     0,     1,     1,     1,     1,
       4,     4,     1,     1,     1,     2,     1,     1,     1,     1,
       1,     3,     1,     1,     1,     2,     1,     1,     1,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       3,     4,     4,     2,     3,     5,     1,     1,     2,     3,
       5,     3,     5,     3,     3,     5,     8,     5,     8,     5,
       0,     3,     0,     1,     3,     1,     4,     2,     0,     3,
       1,     3,     1,     3,     1,     3,     1,     3,     1,     3,
       3,     2,     4,     3,     5,     5,     1,     3,     1,     2,
       1,     3,     4,     1,     2,     2,     1,     1,     3,     0,
       2,     0,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     0,     4,     1,     2,     2,     2,     2,     2,     2,
       1,     3,     1,     3,     1,     3,     1,     3,     1,     3,
       1,     1,     3,     3,     0,     2,     0,     1,     0,     1,
       0,     1,     0,     1,     0,     1,     0,     1,     0,     1,
       0,     1,     0,     1,     4,     4,     5,     6,     4,     4,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       4,     5,     4,     4,     2,     2,     4,     3,     3,     5,
       3,     4,     3,     5,     1,     0,     1,     3,     1,     1,
       2,     1,     1,     5,     0,     2,     1,     3,     1,     3,
       1,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       5,     0,     3,     0,     1,     0,     7,     0,    22,   157,
     159,     0,     0,   158,   218,    20,     6,   344,     0,     4,
       0,     0,     0,    21,     0,     0,     0,    16,     0,     0,
       9,    22,     0,     8,    28,   126,   155,     0,    39,   155,
       0,   263,    74,     0,     0,     0,    78,     0,     0,   292,
      91,     0,    88,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   290,     0,    25,     0,   256,   257,
     260,   258,   259,    50,    93,   134,   146,   114,   163,   162,
     127,     0,     0,     0,   183,   196,   197,    26,   215,     0,
     139,    27,     0,    19,     0,     0,     0,     0,     0,     0,
     345,   160,   161,    11,    14,   286,    18,    22,    13,    17,
     156,   264,   153,     0,     0,     0,     0,   162,   189,   193,
     179,   177,   178,   176,   265,   134,     0,   294,   249,     0,
     210,   134,   268,   294,   151,   152,     0,     0,   276,   293,
     269,     0,     0,   294,     0,     0,    36,    48,     0,    29,
     274,   154,     0,   122,   117,   118,   121,   115,   116,     0,
       0,   148,     0,   149,   174,   172,   173,   119,   120,     0,
     291,     0,   219,     0,    32,     0,     0,     0,     0,     0,
      55,     0,     0,     0,    54,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   140,
       0,     0,   290,   261,     0,   140,   217,     0,     0,     0,
       0,   310,     0,     0,   210,     0,     0,   311,     0,     0,
      23,   287,     0,    12,   249,     0,     0,   194,   170,   168,
     169,   166,   167,   198,     0,     0,   295,    72,     0,    75,
       0,    71,   164,   243,   162,   246,   150,   247,   288,     0,
     249,     0,   204,    79,    76,   157,     0,   203,     0,   286,
     240,   228,     0,    64,     0,     0,   201,   272,   286,   226,
     238,   302,     0,    89,    38,   224,   286,    49,    31,   220,
     286,     0,     0,    40,     0,   175,   147,     0,     0,    35,
     286,     0,     0,    51,    95,   110,   113,    96,   100,   101,
      99,   111,    98,    97,    94,   112,   102,   103,   104,   105,
     106,   107,   108,   109,   284,   123,   278,   288,     0,   128,
     291,     0,     0,   288,   284,   255,    60,   253,   252,   270,
     254,     0,    53,    52,   277,     0,     0,     0,     0,   318,
       0,     0,     0,     0,     0,   317,     0,   312,   313,   314,
       0,   346,     0,     0,   296,     0,     0,     0,    15,    10,
       0,     0,     0,   180,   190,    66,    73,     0,     0,   294,
     165,   244,   245,   289,   250,   212,     0,     0,     0,   294,
       0,   236,     0,   249,   239,   287,     0,     0,     0,     0,
     302,     0,     0,   287,     0,   303,   231,     0,   302,     0,
     287,     0,   287,     0,    42,   275,     0,     0,     0,   199,
     170,   168,   169,   167,   140,   192,   191,   287,     0,    44,
       0,   140,   142,   280,   281,   288,     0,   288,   289,     0,
       0,     0,   131,   290,   262,   289,     0,     0,     0,     0,
     216,     0,     0,   325,   315,   316,   296,   300,     0,   298,
       0,   324,   339,     0,     0,   341,   342,     0,     0,     0,
       0,     0,   302,     0,     0,   309,     0,   297,   304,   308,
     305,   212,   171,     0,     0,     0,     0,   248,   249,   162,
     213,   188,   186,   187,   184,   185,   209,   212,   211,    80,
      77,   237,   241,     0,   229,   202,   195,     0,     0,    92,
      62,    65,     0,   233,     0,   302,   227,   200,   273,   230,
      64,   225,    37,   221,    30,    41,     0,   284,    45,   222,
     286,    47,    33,    43,   284,     0,   289,   285,   137,     0,
     279,   124,   130,   129,     0,   135,   136,     0,   271,   327,
       0,     0,   318,     0,   317,     0,   334,   350,   301,     0,
       0,     0,   348,   299,   328,   340,     0,   306,     0,   319,
       0,   302,   330,     0,   347,   335,     0,    69,    68,   294,
       0,   249,   205,    84,   212,     0,    59,     0,   302,   302,
     232,     0,   171,     0,   287,     0,    46,     0,   140,   144,
     141,   282,   283,   125,   290,   132,    61,   326,   335,   296,
     323,     0,     0,   302,   322,     0,     0,   320,   307,   331,
     296,   296,   338,   207,   336,    67,    70,   214,     0,    86,
     242,     0,     0,    56,     0,    63,   235,   234,    90,   138,
     223,    34,   143,   284,     0,   329,     0,   351,   321,   332,
     349,     0,     0,     0,   212,     0,    85,    81,     0,     0,
       0,   133,   335,   343,   335,   337,   206,    82,    87,    58,
      57,   145,   333,   208,   294,     0,    83
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     6,     2,     3,    14,    21,    30,   105,    31,
       8,    24,    16,    17,    65,   327,    67,   149,   518,   519,
     145,   146,    68,   500,   328,   438,   501,   577,   388,   366,
     473,   237,   238,   239,    69,   127,   253,    70,   133,   378,
     573,   646,   664,   619,   647,    71,   143,   399,    72,   141,
      73,    74,    75,    76,   314,   423,   424,   590,    77,   316,
     243,   136,    78,   150,   111,   117,    13,    80,    81,   245,
     246,   163,   119,    82,    83,   480,   228,    84,   230,   231,
      85,    86,    87,   130,   214,    88,   252,   486,    89,    90,
      22,   280,   520,   276,   268,   259,   269,   270,   271,   261,
     384,   247,   248,   249,   329,   330,   322,   331,   272,   152,
      92,   317,   425,   426,   222,   374,   171,   140,   254,   466,
     551,   545,   396,   100,   212,   218,   612,   443,   347,   348,
     349,   351,   552,   547,   613,   614,   456,   457,    25,   467,
     553,   548
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -474
static const yytype_int16 yypact[] =
{
    -474,    48,    28,    35,  -474,   258,  -474,    37,  -474,  -474,
    -474,    61,    12,  -474,    85,   107,  -474,  -474,    70,  -474,
     156,    82,  1059,  -474,   122,   328,    22,  -474,    56,   199,
    -474,    35,   211,  -474,  -474,  -474,   258,   767,  -474,   258,
     459,  -474,  -474,   152,   459,   258,  -474,    23,   145,  1650,
    -474,    23,  -474,   294,   359,  1650,  1650,  1650,  1650,  1650,
    1650,  1693,  1650,  1650,  1289,   159,  -474,   412,  -474,  -474,
    -474,  -474,  -474,   939,  -474,  -474,   157,   302,  -474,   168,
    -474,   175,   184,    23,   204,  -474,  -474,  -474,   219,    54,
    -474,  -474,    47,  -474,   227,   -12,   269,   227,   227,   239,
    -474,  -474,  -474,  -474,  -474,   240,  -474,  -474,  -474,  -474,
    -474,  -474,  -474,   250,  1813,  1813,  1813,  -474,   259,  -474,
    -474,  -474,  -474,  -474,  -474,    64,   302,  1650,  1805,   262,
     260,   174,  -474,  1650,  -474,  -474,   221,  1813,  2166,   255,
    -474,   290,   237,  1650,   304,  1813,  -474,  -474,   420,  -474,
    -474,  -474,   580,  -474,  -474,  -474,  -474,  -474,  -474,  1736,
    1693,  2166,   280,  -474,   253,  -474,    50,  -474,  -474,   275,
    2166,   285,  -474,   430,  -474,   612,  1650,  1650,  1650,  1650,
    -474,  1650,  1650,  1650,  -474,  1650,  1650,  1650,  1650,  1650,
    1650,  1650,  1650,  1650,  1650,  1650,  1650,  1650,  1650,  -474,
    1332,   428,  1650,  -474,  1650,  -474,  -474,  1234,  1650,  1650,
    1650,  -474,   763,   258,   260,   293,   369,  -474,  1992,  1992,
    -474,    51,   326,  -474,  1805,   392,  1813,  -474,  -474,  -474,
    -474,  -474,  -474,  -474,   341,   258,  -474,  -474,   371,  -474,
      89,   342,  1813,  -474,  1805,  -474,  -474,  -474,   335,   360,
    1805,  1234,  -474,  -474,   357,    99,   399,  -474,   365,   380,
    -474,  -474,   377,  -474,   173,   151,  -474,  -474,   381,  -474,
    -474,   456,  1779,  -474,  -474,  -474,   401,  -474,  -474,  -474,
     404,  1650,   258,   366,  1838,  -474,   405,  1813,  1813,  -474,
     407,  1650,   410,  2166,   650,  -474,  2190,   877,   877,   877,
     877,  -474,   877,   877,  2214,  -474,   461,   461,   461,   461,
    -474,  -474,  -474,  -474,  1387,  -474,  -474,    52,  1442,  -474,
    2064,   411,  1160,  2031,  1387,  -474,  -474,  -474,  -474,  -474,
    -474,    19,   255,   255,  2166,  1905,   447,   441,   439,  -474,
     444,   505,  1992,   225,    27,  -474,   454,  -474,  -474,  -474,
    1931,  -474,   125,   458,   258,   460,   465,   466,  -474,  -474,
     463,  1813,   480,  -474,  -474,  -474,  -474,  1497,  1552,  1650,
    -474,  -474,  -474,  1805,  -474,  1872,   484,    24,   371,  1650,
     258,   485,   487,  1805,  -474,   472,   481,  1813,    81,   399,
     456,   399,   490,   289,   483,  -474,  -474,   258,   456,   519,
     258,   495,   258,   496,   255,  -474,  1650,  1897,  1813,  -474,
     321,   349,   350,   354,  -474,  -474,  -474,   258,   497,   255,
    1650,  -474,  2094,  -474,  -474,   488,   491,   489,  1693,   498,
     500,   502,  -474,  1650,  -474,  -474,   506,   503,  1234,  1160,
    -474,  1992,   534,  -474,  -474,  -474,   258,  1958,  1992,   258,
    1992,  -474,  -474,   565,   149,  -474,  -474,   510,   504,  1992,
     225,  1992,   456,   258,   258,  -474,   514,   507,  -474,  -474,
    -474,  1872,  -474,  1234,  1650,  1650,   515,  -474,  1805,   520,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,  1872,  -474,  -474,
    -474,  -474,  -474,   518,  -474,  -474,  -474,  1693,   517,  -474,
    -474,  -474,   524,  -474,   525,   456,  -474,  -474,  -474,  -474,
    -474,  -474,  -474,  -474,  -474,   255,   526,  1387,  -474,  -474,
     527,   612,  -474,   255,  1387,  1595,  1387,  -474,  -474,   530,
    -474,  -474,  -474,  -474,   116,  -474,  -474,   141,  -474,  -474,
     539,   540,   521,   542,   546,   538,  -474,  -474,   548,   543,
    1992,   549,  -474,   552,  -474,  -474,   562,  -474,  1992,  -474,
     556,   456,  -474,   560,  -474,  1984,   238,  2166,  2166,  1650,
     561,  1805,  -474,  -474,  1872,    32,  -474,  1160,   456,   456,
    -474,   186,   370,   554,   258,   563,   410,   557,  -474,  2166,
    -474,  -474,  -474,  -474,  1650,  -474,  -474,  -474,  1984,   258,
    -474,  1958,  1992,   456,  -474,   258,   149,  -474,  -474,  -474,
     258,   258,  -474,  -474,  -474,  -474,  -474,  -474,   564,   613,
    -474,  1650,  1650,  -474,  1693,   566,  -474,  -474,  -474,  -474,
    -474,  -474,  -474,  1387,   558,  -474,   571,  -474,  -474,  -474,
    -474,   577,   582,   583,  1872,    36,  -474,  -474,  2118,  2142,
     572,  -474,  1984,  -474,  1984,  -474,  -474,  -474,  -474,  -474,
    -474,  -474,  -474,  -474,  1650,   371,  -474
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -474,  -474,  -474,  -474,  -474,  -474,  -474,   -15,  -474,  -474,
     616,  -474,    -3,  -474,  -474,   622,  -474,  -125,   -27,    66,
    -474,  -124,  -112,  -474,    11,  -474,  -474,  -474,   147,  -368,
    -474,  -474,  -474,  -474,  -474,  -474,  -140,  -474,  -474,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,
     532,    10,   247,  -474,  -194,   132,   135,  -474,   279,   -59,
     418,    67,     5,   384,   624,   425,   317,    20,  -474,   424,
     636,   509,  -474,  -474,  -474,  -474,   -36,   -37,   -31,   -49,
    -474,  -474,  -474,  -474,  -474,   -32,   464,  -473,  -474,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,   277,  -119,  -231,   287,
    -474,   300,  -474,  -205,  -300,   652,  -474,  -242,  -474,   -63,
     106,   182,  -474,  -316,  -241,  -285,  -195,  -474,  -111,  -420,
    -474,  -474,  -245,  -474,   402,  -474,  -176,  -474,   345,   249,
     346,   218,    87,    96,  -415,  -474,  -429,   252,  -474,   522,
    -474,  -474
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -277
static const yytype_int16 yytable[] =
{
     121,   120,   162,   273,   175,   123,   122,   321,   437,   377,
     489,   324,   165,   104,   572,   236,   241,   260,   386,   360,
     275,   236,   434,   279,   164,   556,   541,   394,   108,   166,
     458,   236,   429,   390,   392,   401,   346,   621,   436,   403,
     174,   110,   356,   357,   110,   376,   101,   213,     4,   418,
     132,  -215,   208,     5,    27,   206,   657,   118,   134,    27,
       7,    15,    11,   427,    18,   153,   154,   155,   156,   157,
     158,  -267,   167,   168,    19,     9,  -267,   229,   229,   229,
       9,   439,   232,   232,   232,  -215,   439,   440,   497,   134,
     135,   229,   488,   498,   367,   102,   232,   622,   623,   459,
     229,   620,  -236,   326,   223,   232,    20,   624,   229,  -181,
     175,   165,   209,   232,    29,   229,   103,  -215,   142,    29,
     232,   135,   210,   164,    10,    11,  -267,   428,   166,    10,
      11,    23,  -267,    26,   118,   118,   118,   382,   229,   538,
     527,   258,   529,   232,    33,   503,   290,   267,   118,   499,
     205,   165,   452,   509,   368,   139,   207,   118,   502,    27,
     504,  -236,   380,   164,   210,   118,   451,  -236,   166,   153,
     157,   656,   118,     9,   462,   381,     9,   641,   493,   636,
       9,  -266,   594,   635,    93,   463,  -266,   229,   595,   229,
     642,   643,   232,   497,   232,   118,   537,   381,   498,   453,
     464,   583,   106,   439,   391,   229,   358,   229,   587,   596,
     232,   128,   232,   229,   109,    28,   137,   562,   232,    29,
     517,   172,    10,    11,   199,    10,    11,   524,   452,    10,
      11,   566,   389,   240,  -153,   229,  -266,   662,   534,   663,
     232,   203,  -266,   204,   118,   255,   118,   411,   410,     9,
     229,   229,   413,   412,   628,   232,   232,   236,   476,   431,
     580,   255,   118,  -182,   118,   539,   260,   236,   490,   165,
     118,   546,   549,   570,   554,   453,   511,   513,  -181,   585,
     256,   164,     9,   559,   454,   561,   166,   125,  -183,   257,
     264,   131,   118,   216,    10,    11,   265,   666,    10,    11,
     439,    11,   221,   220,   118,   266,   615,   118,   118,   224,
      10,    11,  -182,   255,   332,   333,   609,   650,     9,   126,
    -183,   250,   235,   126,   229,   263,   484,   251,     9,   232,
     210,    10,    11,   626,   627,   625,   229,    94,   482,   481,
     286,   232,   264,   485,   483,    95,   229,   287,   265,    96,
     229,   232,   354,   144,   521,   232,  -179,   288,   639,    97,
      98,   200,    10,    11,   274,   201,   618,    10,    11,   530,
     229,   229,   355,   202,   603,   232,   232,    10,    11,   165,
    -179,   118,   607,     9,  -177,  -178,   359,   404,  -179,  -176,
     258,   164,    99,   118,   633,   118,   166,   419,   267,   634,
     361,   363,   508,   118,   369,  -180,   365,   118,  -177,  -178,
     373,   211,   211,  -176,   211,   211,  -177,  -178,   148,   379,
     375,  -176,   484,   381,   383,   546,   638,   118,   118,  -180,
      12,   406,    10,    11,   482,   481,     9,  -180,   484,   485,
     483,   229,   385,   393,     9,    32,   232,    79,   165,   387,
     482,   481,     9,    32,     9,   485,   483,   236,   616,   395,
     164,   112,    35,   400,   112,   166,   402,    37,   129,   417,
     112,   173,   414,   332,   333,   420,   113,   433,   147,   151,
     278,    47,    48,     9,   229,    10,    11,   318,    51,   232,
     289,   118,   151,    10,    11,   178,   255,   215,   118,   217,
     219,    10,    11,    10,    11,   186,   446,   118,   447,   190,
     448,   449,   515,   450,   195,   196,   197,   198,    61,   460,
     465,   521,   468,   471,   665,   484,   523,   469,   470,   345,
      64,   256,    10,    11,   229,   345,   345,   482,   481,   232,
     472,   118,   485,   483,   487,    10,    11,   492,   380,   495,
     505,   507,   236,   244,   510,   512,   514,   522,   531,   528,
     532,   112,   533,   526,   435,   530,   535,   112,   555,   147,
     341,   536,   557,   151,   565,   165,   558,   569,   574,   571,
    -157,   138,   464,   576,   578,   579,   582,   164,    37,   584,
     593,   118,   166,   161,   118,   484,   170,   113,   151,   597,
     598,   599,    47,    48,     9,  -158,   600,   482,   481,    51,
     601,   606,   485,   483,   605,   602,   225,   604,   608,   610,
      37,   617,   629,   631,   644,   632,   319,   645,   439,   113,
     651,   652,    79,   115,    47,    48,     9,   653,   350,   226,
     661,    51,   654,   655,    66,   281,    32,   107,   225,   244,
     630,    64,   345,    10,    11,   282,   658,   581,   591,   345,
     364,   592,   371,   124,   118,   115,   405,   345,   372,   285,
     506,   226,   494,   477,    91,   244,    79,   291,   353,   575,
     444,   445,   564,    64,   178,    10,    11,   282,   181,   182,
     183,   540,   640,   185,   186,   187,   188,   637,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   151,   293,   294,
     295,   296,   560,   297,   298,   299,     0,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
     313,     0,   161,     0,   320,   352,   323,     0,     0,     0,
     138,   138,   334,     0,     0,     0,     0,    79,     0,     0,
     227,   233,   234,     0,     0,     0,     0,     0,   345,     0,
       0,     0,     0,     0,   544,   345,     0,   345,   455,     0,
       0,   335,     0,   262,     0,    37,   345,     0,   345,   350,
     336,   277,     0,     0,   113,   337,   338,   339,   283,    47,
      48,     9,   340,     0,     0,     0,    51,     0,   244,   341,
     479,     0,     0,   114,     0,   491,     0,     0,   244,     0,
     112,   292,     0,   138,     0,     0,   342,     0,   112,     0,
     115,     0,   112,   138,     0,   147,   116,   151,   343,     0,
       0,     0,     0,     0,   344,     0,     0,    11,    64,     0,
      10,    11,   151,     0,     0,     0,   422,     0,     0,     0,
     161,     0,     0,     0,     0,     0,   422,     0,     0,     0,
       0,     0,   362,    79,    79,     0,     0,   345,     0,     0,
       0,   350,   543,     0,   550,   345,     0,     0,   370,   455,
       0,     0,   345,     0,     0,   455,     0,     0,   563,   350,
       0,     0,     0,     0,     0,     0,     0,     0,    79,   138,
     138,     0,     0,   244,     0,     0,     0,     0,   398,     0,
       0,   178,     0,     0,     0,   345,     0,     0,   544,   345,
     409,   186,     0,   415,   416,   190,   191,   192,   193,   194,
     195,   196,   197,   198,     0,     0,     0,     0,   138,     0,
       0,     0,     0,   176,  -276,     0,     0,     0,     0,     0,
       0,     0,   138,     0,     0,     0,     0,     0,     0,     0,
     161,     0,     0,     0,     0,   170,     0,     0,     0,   345,
       0,   345,   177,   178,     0,   179,   180,   181,   182,   183,
       0,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   244,   409,     0,     0,
       0,     0,    79,     0,  -276,     0,   567,   568,     0,   151,
       0,     0,     0,     0,  -276,     0,     0,     0,     0,     0,
       0,     0,     0,   496,   350,     0,   543,     0,     0,   161,
     550,   455,     0,     0,     0,   350,   350,     0,     0,     0,
       0,     0,     0,   227,   516,     0,     0,     0,     0,   422,
       0,     0,     0,     0,     0,     0,   422,   589,   422,    -2,
      34,     0,    35,     0,     0,    36,     0,    37,    38,    39,
       0,     0,    40,     0,    41,    42,    43,    44,    45,    46,
       0,    47,    48,     9,     0,     0,    49,    50,    51,    52,
      53,    54,     0,     0,     0,    55,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    56,    57,
       0,    58,    59,     0,     0,    60,     0,     0,    61,     0,
       0,   -24,     0,     0,     0,     0,   170,     0,    62,    63,
      64,     0,    10,    11,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   648,   649,     0,   161,   586,     0,     0,
       0,   325,     0,    35,     0,   422,    36,  -251,    37,    38,
      39,     0,  -251,    40,     0,    41,    42,   113,    44,    45,
      46,     0,    47,    48,     9,     0,     0,    49,    50,    51,
      52,    53,    54,     0,     0,     0,    55,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    56,
      57,     0,    58,    59,     0,     0,    60,     0,     0,    61,
       0,     0,  -251,     0,     0,     0,     0,   326,  -251,    62,
      63,    64,     0,    10,    11,   325,     0,    35,     0,     0,
      36,     0,    37,    38,    39,     0,     0,    40,     0,    41,
      42,   113,    44,    45,    46,     0,    47,    48,     9,     0,
       0,    49,    50,    51,    52,    53,    54,     0,     0,     0,
      55,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    56,    57,     0,    58,    59,     0,     0,
      60,     0,    35,    61,     0,     0,  -251,    37,     0,     0,
     169,   326,  -251,    62,    63,    64,   113,    10,    11,     0,
       0,    47,    48,     9,     0,     0,     0,     0,    51,     0,
       0,     0,     0,     0,     0,    55,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    35,     0,     0,    56,    57,
      37,    58,    59,     0,     0,    60,     0,     0,    61,   113,
       0,     0,     0,     0,    47,    48,     9,     0,    62,    63,
      64,    51,    10,    11,     0,     0,     0,     0,   159,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    56,    57,     0,    58,   160,     0,     0,    60,     0,
      35,    61,   315,     0,     0,    37,     0,     0,     0,     0,
       0,    62,    63,    64,   113,    10,    11,     0,     0,    47,
      48,     9,     0,     0,     0,     0,    51,     0,     0,     0,
       0,     0,     0,    55,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    56,    57,     0,    58,
      59,     0,     0,    60,     0,    35,    61,     0,     0,     0,
      37,     0,     0,     0,   421,     0,    62,    63,    64,   113,
      10,    11,     0,     0,    47,    48,     9,     0,     0,     0,
       0,    51,     0,   430,     0,     0,     0,     0,   159,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    56,    57,     0,    58,   160,     0,     0,    60,     0,
      35,    61,     0,     0,     0,    37,     0,     0,     0,     0,
       0,    62,    63,    64,   113,    10,    11,     0,     0,    47,
      48,     9,     0,   474,     0,     0,    51,     0,     0,     0,
       0,     0,     0,    55,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    56,    57,     0,    58,
      59,     0,     0,    60,     0,    35,    61,     0,     0,     0,
      37,     0,     0,     0,     0,     0,    62,    63,    64,   113,
      10,    11,     0,     0,    47,    48,     9,     0,   475,     0,
       0,    51,     0,     0,     0,     0,     0,     0,    55,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    35,     0,
       0,    56,    57,    37,    58,    59,     0,     0,    60,     0,
       0,    61,   113,     0,     0,     0,     0,    47,    48,     9,
       0,    62,    63,    64,    51,    10,    11,     0,     0,     0,
       0,    55,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    56,    57,     0,    58,    59,     0,
       0,    60,     0,    35,    61,     0,     0,     0,    37,     0,
       0,     0,   588,     0,    62,    63,    64,   113,    10,    11,
       0,     0,    47,    48,     9,     0,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,    55,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    35,     0,     0,    56,
      57,    37,    58,    59,     0,     0,    60,     0,     0,    61,
     113,     0,     0,     0,     0,    47,    48,     9,     0,    62,
      63,    64,    51,    10,    11,     0,     0,     0,     0,   159,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    35,
       0,     0,    56,    57,   284,    58,   160,     0,     0,    60,
       0,     0,    61,   113,     0,     0,     0,     0,    47,    48,
       9,     0,    62,    63,    64,    51,    10,    11,     0,     0,
       0,     0,    55,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    56,    57,    37,    58,    59,
       0,     0,    60,     0,     0,    61,   113,     0,     0,     0,
       0,    47,    48,     9,     0,    62,    63,    64,    51,    10,
      11,     0,     0,    37,     0,   225,   242,     0,     0,     0,
       0,    37,   113,     0,     0,     0,     0,    47,    48,     9,
     113,     0,   115,     0,    51,    47,    48,     9,   226,     0,
       0,   225,    51,     0,     0,     0,    37,     0,     0,   225,
      64,     0,    10,    11,   397,   113,     0,     0,   115,     0,
      47,    48,     9,     0,   226,     0,   115,    51,     0,     0,
       0,     0,   226,     0,   407,     0,    64,     0,    10,    11,
      37,     0,     0,     0,    64,     0,    10,    11,     0,   113,
       0,   115,     0,     0,    47,    48,     9,   408,     0,     0,
       0,    51,     0,     0,     0,   284,     0,     0,   225,    64,
       0,    10,    11,   335,   113,     0,     0,     0,     0,    47,
      48,     9,   336,     0,     0,   115,    51,   337,   338,   339,
       0,   478,     0,   225,   340,     0,     0,     0,     0,   335,
       0,   441,   461,    64,     0,    10,    11,     0,   336,     0,
     115,     0,     0,   337,   338,   339,   226,     0,   342,     0,
     340,     0,     0,     0,   442,     0,   335,   341,    64,     0,
      10,    11,     0,     0,     0,   336,   344,     0,     0,    11,
     337,   338,   542,     0,   342,     0,     0,   340,     0,     0,
       0,     0,   335,     0,   341,     0,     0,     0,     0,     0,
     335,   336,   344,     0,     0,    11,   337,   338,   339,   336,
       0,   342,     0,   340,   337,   338,   339,     0,     0,     0,
     341,   340,     0,     0,     0,     0,     0,     0,   341,   344,
       0,    10,    11,     0,     0,     0,     0,   342,     0,     0,
       0,     0,     0,   611,     0,   342,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   344,     0,     0,    11,     0,
       0,     0,     0,   344,   177,   178,    11,   179,     0,   181,
     182,   183,     0,     0,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   177,   178,     0,
     179,     0,   181,   182,   183,     0,   435,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     198,     0,     0,     0,     0,     0,     0,   177,   178,     0,
     179,     0,   181,   182,   183,     0,   432,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     198,   177,   178,     0,   179,     0,   181,   182,   183,     0,
     525,   185,   186,   187,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   197,   198,   177,   178,     0,   179,     0,
     181,   182,   183,     0,   659,   185,   186,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   198,   177,
     178,     0,   179,     0,   181,   182,   183,     0,   660,   185,
     186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,   197,   198,   177,   178,     0,     0,     0,   181,   182,
     183,     0,     0,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   177,   178,     0,
       0,     0,   181,   182,   183,     0,     0,   185,   186,   187,
     188,     0,   190,   191,   192,   193,   194,   195,   196,   197,
     198
};

static const yytype_int16 yycheck[] =
{
      37,    37,    61,   143,    67,    37,    37,   202,   324,   251,
     378,   205,    61,    28,   487,   127,   127,   136,   259,   224,
     144,   133,   322,   148,    61,   454,   446,   268,    31,    61,
       3,   143,   317,   264,   265,   276,   212,     5,   323,   280,
      67,    36,   218,   219,    39,   250,    24,    59,     0,   290,
      45,     1,     5,    25,     3,     1,    20,    37,    35,     3,
      25,    24,    74,    11,     3,    55,    56,    57,    58,    59,
      60,     7,    62,    63,    62,    24,    12,   114,   115,   116,
      24,    62,   114,   115,   116,    35,    62,    68,     7,    35,
      67,   128,    68,    12,     5,    73,   128,    65,    66,    72,
     137,   574,     3,    67,   107,   137,    21,    75,   145,    59,
     173,   160,    65,   145,    63,   152,    60,    67,    51,    63,
     152,    67,    75,   160,    73,    74,    62,    75,   160,    73,
      74,    24,    68,    63,   114,   115,   116,   256,   175,   439,
     425,   136,   427,   175,    62,   390,   173,   142,   128,    68,
      83,   200,     3,   398,    65,    49,    89,   137,   389,     3,
     391,    62,    63,   200,    75,   145,   342,    68,   200,   159,
     160,   644,   152,    24,   350,    24,    24,   606,   383,   599,
      24,     7,    66,   598,    62,    60,    12,   224,    72,   226,
     610,   611,   224,     7,   226,   175,   438,    24,    12,    50,
      75,   517,     3,    62,    53,   242,   221,   244,   524,    68,
     242,    59,   244,   250,     3,    59,    71,   462,   250,    63,
     414,    62,    73,    74,    67,    73,    74,   421,     3,    73,
      74,   473,    59,   127,    66,   272,    62,   652,   433,   654,
     272,    66,    68,    59,   224,    24,   226,   284,   284,    24,
     287,   288,   284,   284,    68,   287,   288,   369,   369,   318,
     505,    24,   242,    59,   244,   441,   385,   379,   379,   318,
     250,   447,   448,   478,   450,    50,   400,   402,    59,   520,
      59,   318,    24,   459,    59,   461,   318,    40,    35,    68,
      53,    44,   272,    24,    73,    74,    59,   665,    73,    74,
      62,    74,    62,    64,   284,    68,    68,   287,   288,    59,
      73,    74,    59,    24,   208,   209,   561,   633,    24,    40,
      67,    59,    63,    44,   361,    35,   375,    67,    24,   361,
      75,    73,    74,   578,   579,   577,   373,     9,   375,   375,
      60,   373,    53,   375,   375,    17,   383,    72,    59,    21,
     387,   383,    59,    59,   417,   387,    35,    72,   603,    31,
      32,    59,    73,    74,    60,    63,   571,    73,    74,   428,
     407,   408,     3,    71,   550,   407,   408,    73,    74,   428,
      59,   361,   558,    24,    35,    35,    60,   281,    67,    35,
     385,   428,    64,   373,   588,   375,   428,   291,   393,   594,
       8,    60,   397,   383,    62,    35,    35,   387,    59,    59,
      75,    94,    95,    59,    97,    98,    67,    67,    59,    62,
      60,    67,   471,    24,    59,   601,   602,   407,   408,    59,
       5,    65,    73,    74,   471,   471,    24,    67,   487,   471,
     471,   478,    62,    62,    24,    20,   478,    22,   497,    72,
     487,   487,    24,    28,    24,   487,   487,   569,   569,     3,
     497,    36,     3,    62,    39,   497,    62,     8,    43,    62,
      45,    59,    67,   367,   368,    65,    17,    66,    53,    54,
      60,    22,    23,    24,   521,    73,    74,    59,    29,   521,
      60,   471,    67,    73,    74,    34,    24,    95,   478,    97,
      98,    73,    74,    73,    74,    44,    59,   487,    67,    48,
      71,    67,   406,     8,    53,    54,    55,    56,    59,    65,
      62,   584,    62,    60,   664,   574,   420,    62,    62,   212,
      71,    59,    73,    74,   571,   218,   219,   574,   574,   571,
      60,   521,   574,   574,    60,    73,    74,    60,    63,    68,
      60,    68,   664,   128,    35,    60,    60,    60,    60,    68,
      60,   136,    60,    75,    75,   624,    60,   142,     3,   144,
      36,    68,    62,   148,    60,   624,    72,    62,    60,    59,
      59,    49,    75,    66,    60,    60,    60,   624,     8,    62,
      60,   571,   624,    61,   574,   644,    64,    17,   173,    60,
      60,    59,    22,    23,    24,    59,    68,   644,   644,    29,
      62,    49,   644,   644,    62,    72,    36,    68,    62,    59,
       8,    60,    68,    60,    60,    68,   201,    14,    62,    17,
      72,    60,   207,    53,    22,    23,    24,    60,   213,    59,
      68,    29,    60,    60,    22,    65,   221,    31,    36,   224,
     584,    71,   335,    73,    74,    75,   645,   510,   526,   342,
     235,   526,   244,    39,   644,    53,   282,   350,   244,   160,
     393,    59,   385,   373,    22,   250,   251,    65,   214,   497,
     335,   335,   464,    71,    34,    73,    74,    75,    38,    39,
      40,   442,   605,    43,    44,    45,    46,   601,    48,    49,
      50,    51,    52,    53,    54,    55,    56,   282,   176,   177,
     178,   179,   460,   181,   182,   183,    -1,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     198,    -1,   200,    -1,   202,   213,   204,    -1,    -1,    -1,
     208,   209,   210,    -1,    -1,    -1,    -1,   322,    -1,    -1,
     114,   115,   116,    -1,    -1,    -1,    -1,    -1,   441,    -1,
      -1,    -1,    -1,    -1,   447,   448,    -1,   450,   343,    -1,
      -1,     8,    -1,   137,    -1,     8,   459,    -1,   461,   354,
      17,   145,    -1,    -1,    17,    22,    23,    24,   152,    22,
      23,    24,    29,    -1,    -1,    -1,    29,    -1,   373,    36,
     375,    -1,    -1,    36,    -1,   380,    -1,    -1,   383,    -1,
     385,   175,    -1,   281,    -1,    -1,    53,    -1,   393,    -1,
      53,    -1,   397,   291,    -1,   400,    59,   402,    65,    -1,
      -1,    -1,    -1,    -1,    71,    -1,    -1,    74,    71,    -1,
      73,    74,   417,    -1,    -1,    -1,   314,    -1,    -1,    -1,
     318,    -1,    -1,    -1,    -1,    -1,   324,    -1,    -1,    -1,
      -1,    -1,   226,   438,   439,    -1,    -1,   550,    -1,    -1,
      -1,   446,   447,    -1,   449,   558,    -1,    -1,   242,   454,
      -1,    -1,   565,    -1,    -1,   460,    -1,    -1,   463,   464,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   473,   367,
     368,    -1,    -1,   478,    -1,    -1,    -1,    -1,   272,    -1,
      -1,    34,    -1,    -1,    -1,   598,    -1,    -1,   601,   602,
     284,    44,    -1,   287,   288,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    -1,    -1,    -1,    -1,   406,    -1,
      -1,    -1,    -1,     4,     5,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   420,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     428,    -1,    -1,    -1,    -1,   433,    -1,    -1,    -1,   652,
      -1,   654,    33,    34,    -1,    36,    37,    38,    39,    40,
      -1,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,   571,   361,    -1,    -1,
      -1,    -1,   577,    -1,    65,    -1,   474,   475,    -1,   584,
      -1,    -1,    -1,    -1,    75,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   387,   599,    -1,   601,    -1,    -1,   497,
     605,   606,    -1,    -1,    -1,   610,   611,    -1,    -1,    -1,
      -1,    -1,    -1,   407,   408,    -1,    -1,    -1,    -1,   517,
      -1,    -1,    -1,    -1,    -1,    -1,   524,   525,   526,     0,
       1,    -1,     3,    -1,    -1,     6,    -1,     8,     9,    10,
      -1,    -1,    13,    -1,    15,    16,    17,    18,    19,    20,
      -1,    22,    23,    24,    -1,    -1,    27,    28,    29,    30,
      31,    32,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,
      -1,    52,    53,    -1,    -1,    56,    -1,    -1,    59,    -1,
      -1,    62,    -1,    -1,    -1,    -1,   594,    -1,    69,    70,
      71,    -1,    73,    74,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   621,   622,    -1,   624,   521,    -1,    -1,
      -1,     1,    -1,     3,    -1,   633,     6,     7,     8,     9,
      10,    -1,    12,    13,    -1,    15,    16,    17,    18,    19,
      20,    -1,    22,    23,    24,    -1,    -1,    27,    28,    29,
      30,    31,    32,    -1,    -1,    -1,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,
      50,    -1,    52,    53,    -1,    -1,    56,    -1,    -1,    59,
      -1,    -1,    62,    -1,    -1,    -1,    -1,    67,    68,    69,
      70,    71,    -1,    73,    74,     1,    -1,     3,    -1,    -1,
       6,    -1,     8,     9,    10,    -1,    -1,    13,    -1,    15,
      16,    17,    18,    19,    20,    -1,    22,    23,    24,    -1,
      -1,    27,    28,    29,    30,    31,    32,    -1,    -1,    -1,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,
      56,    -1,     3,    59,    -1,    -1,    62,     8,    -1,    -1,
      11,    67,    68,    69,    70,    71,    17,    73,    74,    -1,
      -1,    22,    23,    24,    -1,    -1,    -1,    -1,    29,    -1,
      -1,    -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    49,    50,
       8,    52,    53,    -1,    -1,    56,    -1,    -1,    59,    17,
      -1,    -1,    -1,    -1,    22,    23,    24,    -1,    69,    70,
      71,    29,    73,    74,    -1,    -1,    -1,    -1,    36,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,    -1,
       3,    59,    60,    -1,    -1,     8,    -1,    -1,    -1,    -1,
      -1,    69,    70,    71,    17,    73,    74,    -1,    -1,    22,
      23,    24,    -1,    -1,    -1,    -1,    29,    -1,    -1,    -1,
      -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,    52,
      53,    -1,    -1,    56,    -1,     3,    59,    -1,    -1,    -1,
       8,    -1,    -1,    -1,    67,    -1,    69,    70,    71,    17,
      73,    74,    -1,    -1,    22,    23,    24,    -1,    -1,    -1,
      -1,    29,    -1,    31,    -1,    -1,    -1,    -1,    36,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,    -1,
       3,    59,    -1,    -1,    -1,     8,    -1,    -1,    -1,    -1,
      -1,    69,    70,    71,    17,    73,    74,    -1,    -1,    22,
      23,    24,    -1,    26,    -1,    -1,    29,    -1,    -1,    -1,
      -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,    52,
      53,    -1,    -1,    56,    -1,     3,    59,    -1,    -1,    -1,
       8,    -1,    -1,    -1,    -1,    -1,    69,    70,    71,    17,
      73,    74,    -1,    -1,    22,    23,    24,    -1,    26,    -1,
      -1,    29,    -1,    -1,    -1,    -1,    -1,    -1,    36,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,
      -1,    49,    50,     8,    52,    53,    -1,    -1,    56,    -1,
      -1,    59,    17,    -1,    -1,    -1,    -1,    22,    23,    24,
      -1,    69,    70,    71,    29,    73,    74,    -1,    -1,    -1,
      -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    50,    -1,    52,    53,    -1,
      -1,    56,    -1,     3,    59,    -1,    -1,    -1,     8,    -1,
      -1,    -1,    67,    -1,    69,    70,    71,    17,    73,    74,
      -1,    -1,    22,    23,    24,    -1,    -1,    -1,    -1,    29,
      -1,    -1,    -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    49,
      50,     8,    52,    53,    -1,    -1,    56,    -1,    -1,    59,
      17,    -1,    -1,    -1,    -1,    22,    23,    24,    -1,    69,
      70,    71,    29,    73,    74,    -1,    -1,    -1,    -1,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,
      -1,    -1,    49,    50,     8,    52,    53,    -1,    -1,    56,
      -1,    -1,    59,    17,    -1,    -1,    -1,    -1,    22,    23,
      24,    -1,    69,    70,    71,    29,    73,    74,    -1,    -1,
      -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    49,    50,     8,    52,    53,
      -1,    -1,    56,    -1,    -1,    59,    17,    -1,    -1,    -1,
      -1,    22,    23,    24,    -1,    69,    70,    71,    29,    73,
      74,    -1,    -1,     8,    -1,    36,    11,    -1,    -1,    -1,
      -1,     8,    17,    -1,    -1,    -1,    -1,    22,    23,    24,
      17,    -1,    53,    -1,    29,    22,    23,    24,    59,    -1,
      -1,    36,    29,    -1,    -1,    -1,     8,    -1,    -1,    36,
      71,    -1,    73,    74,    75,    17,    -1,    -1,    53,    -1,
      22,    23,    24,    -1,    59,    -1,    53,    29,    -1,    -1,
      -1,    -1,    59,    -1,    36,    -1,    71,    -1,    73,    74,
       8,    -1,    -1,    -1,    71,    -1,    73,    74,    -1,    17,
      -1,    53,    -1,    -1,    22,    23,    24,    59,    -1,    -1,
      -1,    29,    -1,    -1,    -1,     8,    -1,    -1,    36,    71,
      -1,    73,    74,     8,    17,    -1,    -1,    -1,    -1,    22,
      23,    24,    17,    -1,    -1,    53,    29,    22,    23,    24,
      -1,    59,    -1,    36,    29,    -1,    -1,    -1,    -1,     8,
      -1,    36,    11,    71,    -1,    73,    74,    -1,    17,    -1,
      53,    -1,    -1,    22,    23,    24,    59,    -1,    53,    -1,
      29,    -1,    -1,    -1,    59,    -1,     8,    36,    71,    -1,
      73,    74,    -1,    -1,    -1,    17,    71,    -1,    -1,    74,
      22,    23,    24,    -1,    53,    -1,    -1,    29,    -1,    -1,
      -1,    -1,     8,    -1,    36,    -1,    -1,    -1,    -1,    -1,
       8,    17,    71,    -1,    -1,    74,    22,    23,    24,    17,
      -1,    53,    -1,    29,    22,    23,    24,    -1,    -1,    -1,
      36,    29,    -1,    -1,    -1,    -1,    -1,    -1,    36,    71,
      -1,    73,    74,    -1,    -1,    -1,    -1,    53,    -1,    -1,
      -1,    -1,    -1,    59,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    71,    -1,    -1,    74,    -1,
      -1,    -1,    -1,    71,    33,    34,    74,    36,    -1,    38,
      39,    40,    -1,    -1,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    33,    34,    -1,
      36,    -1,    38,    39,    40,    -1,    75,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    -1,    -1,    -1,    -1,    -1,    -1,    33,    34,    -1,
      36,    -1,    38,    39,    40,    -1,    72,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    33,    34,    -1,    36,    -1,    38,    39,    40,    -1,
      66,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    33,    34,    -1,    36,    -1,
      38,    39,    40,    -1,    66,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    33,
      34,    -1,    36,    -1,    38,    39,    40,    -1,    66,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    33,    34,    -1,    -1,    -1,    38,    39,
      40,    -1,    -1,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    33,    34,    -1,
      -1,    -1,    38,    39,    40,    -1,    -1,    43,    44,    45,
      46,    -1,    48,    49,    50,    51,    52,    53,    54,    55,
      56
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
     199,    24,    73,    60,    83,    84,     3,    86,    88,     3,
     138,   140,   141,    17,    36,    53,    59,   141,   143,   148,
     152,   153,   154,   161,   140,   128,   134,   111,    59,   141,
     159,   128,   138,   114,    35,    67,   137,    71,   126,   186,
     193,   125,   137,   122,    59,    96,    97,   141,    59,    93,
     139,   141,   185,   127,   127,   127,   127,   127,   127,    36,
      53,   126,   135,   147,   153,   155,   161,   127,   127,    11,
     126,   192,    62,    59,    94,   185,     4,    33,    34,    36,
      37,    38,    39,    40,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    67,
      59,    63,    71,    66,    59,   137,     1,   137,     5,    65,
      75,   142,   200,    59,   160,   200,    24,   200,   201,   200,
      64,    62,   190,    88,    59,    36,    59,   146,   152,   153,
     154,   155,   161,   146,   146,    63,    98,   107,   108,   109,
     186,   194,    11,   136,   141,   145,   146,   177,   178,   179,
      59,    67,   162,   112,   194,    24,    59,    68,   138,   171,
     173,   175,   146,    35,    53,    59,    68,   138,   170,   172,
     173,   174,   184,   112,    60,    97,   169,   146,    60,    93,
     167,    65,    75,   146,     8,   147,    60,    72,    72,    60,
      94,    65,   146,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   126,   130,    60,   135,   187,    59,   141,
     126,   192,   182,   126,   130,     1,    67,    91,   100,   180,
     181,   183,   186,   186,   126,     8,    17,    22,    23,    24,
      29,    36,    53,    65,    71,   142,   202,   204,   205,   206,
     141,   207,   215,   162,    59,     3,   202,   202,    83,    60,
     179,     8,   146,    60,   141,    35,   105,     5,    65,    62,
     146,   136,   145,    75,   191,    60,   179,   183,   115,    62,
      63,    24,   173,    59,   176,    62,   190,    72,   104,    59,
     174,    53,   174,    62,   190,     3,   198,    75,   146,   123,
      62,   190,    62,   190,   186,   139,    65,    36,    59,   146,
     152,   153,   154,   161,    67,   146,   146,    62,   190,   186,
      65,    67,   126,   131,   132,   188,   189,    11,    75,   191,
      31,   135,    72,    66,   180,    75,   191,   189,   101,    62,
      68,    36,    59,   203,   204,   206,    59,    67,    71,    67,
       8,   202,     3,    50,    59,   141,   212,   213,     3,    72,
      65,    11,   202,    60,    75,    62,   195,   215,    62,    62,
      62,    60,    60,   106,    26,    26,   194,   177,    59,   141,
     151,   152,   153,   154,   155,   161,   163,    60,    68,   105,
     194,   141,    60,   179,   175,    68,   146,     7,    12,    68,
      99,   102,   174,   198,   174,    60,   172,    68,   138,   198,
      35,    97,    60,    93,    60,   186,   146,   130,    94,    95,
     168,   185,    60,   186,   130,    66,    75,   191,    68,   191,
     135,    60,    60,    60,   192,    60,    68,   183,   180,   202,
     205,   195,    24,   141,   142,   197,   202,   209,   217,   202,
     141,   196,   208,   216,   202,     3,   212,    62,    72,   202,
     213,   202,   198,   141,   207,    60,   183,   126,   126,    62,
     179,    59,   163,   116,    60,   187,    66,   103,    60,    60,
     198,   104,    60,   189,    62,   190,   146,   189,    67,   126,
     133,   131,   132,    60,    66,    72,    68,    60,    60,    59,
      68,    62,    72,   202,    68,    62,    49,   202,    62,   198,
      59,    59,   202,   210,   211,    68,   194,    60,   179,   119,
     163,     5,    65,    66,    75,   183,   198,   198,    68,    68,
      95,    60,    68,   130,   192,   210,   195,   209,   202,   198,
     208,   212,   195,   195,    60,    14,   117,   120,   126,   126,
     189,    72,    60,    60,    60,    60,   163,    20,   100,    66,
      66,    68,   210,   210,   118,   112,   105
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
		errorexit();
	}
    break;

  case 4:
#line 140 "go.y"
    {
		mkpackage((yyvsp[(2) - (3)].sym)->name);
	}
    break;

  case 5:
#line 150 "go.y"
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
#line 161 "go.y"
    {
		importpkg = nil;
	}
    break;

  case 12:
#line 175 "go.y"
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
		if(strcmp(my->name, "init") == 0) {
			yyerror("cannot import package as init - init must be a func");
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
#line 212 "go.y"
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
#line 227 "go.y"
    {
		// import with original name
		(yyval.i) = parserline();
		importmyname = S;
		importfile(&(yyvsp[(1) - (1)].val), (yyval.i));
	}
    break;

  case 17:
#line 234 "go.y"
    {
		// import with given name
		(yyval.i) = parserline();
		importmyname = (yyvsp[(1) - (2)].sym);
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 18:
#line 241 "go.y"
    {
		// import into my name space
		(yyval.i) = parserline();
		importmyname = lookup(".");
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 19:
#line 250 "go.y"
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
#line 265 "go.y"
    {
		if(strcmp((yyvsp[(1) - (1)].sym)->name, "safe") == 0)
			curio.importsafe = 1;
	}
    break;

  case 22:
#line 271 "go.y"
    {
		defercheckwidth();
	}
    break;

  case 23:
#line 275 "go.y"
    {
		resumecheckwidth();
		unimportfile();
	}
    break;

  case 24:
#line 284 "go.y"
    {
		yyerror("empty top-level declaration");
		(yyval.list) = nil;
	}
    break;

  case 26:
#line 290 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 27:
#line 294 "go.y"
    {
		yyerror("non-declaration statement outside function body");
		(yyval.list) = nil;
	}
    break;

  case 28:
#line 299 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 29:
#line 305 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
	}
    break;

  case 30:
#line 309 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 31:
#line 313 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 32:
#line 317 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 33:
#line 323 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 34:
#line 329 "go.y"
    {
		(yyval.list) = concat((yyvsp[(3) - (7)].list), (yyvsp[(5) - (7)].list));
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 35:
#line 335 "go.y"
    {
		(yyval.list) = nil;
		iota = -100000;
	}
    break;

  case 36:
#line 340 "go.y"
    {
		(yyval.list) = list1((yyvsp[(2) - (2)].node));
	}
    break;

  case 37:
#line 344 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 38:
#line 348 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 39:
#line 354 "go.y"
    {
		iota = 0;
	}
    break;

  case 40:
#line 360 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 41:
#line 364 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 42:
#line 368 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (3)].list), nil, (yyvsp[(3) - (3)].list));
	}
    break;

  case 43:
#line 374 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 44:
#line 378 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (3)].list), N, (yyvsp[(3) - (3)].list));
	}
    break;

  case 46:
#line 385 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 47:
#line 389 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (1)].list), N, nil);
	}
    break;

  case 48:
#line 395 "go.y"
    {
		// different from dclname because the name
		// becomes visible right here, not at the end
		// of the declaration.
		(yyval.node) = typedcl0((yyvsp[(1) - (1)].sym));
	}
    break;

  case 49:
#line 404 "go.y"
    {
		(yyval.node) = typedcl1((yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node), 1);
	}
    break;

  case 50:
#line 410 "go.y"
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
#line 428 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
		(yyval.node)->etype = (yyvsp[(2) - (3)].i);			// rathole to pass opcode
	}
    break;

  case 52:
#line 433 "go.y"
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
#line 445 "go.y"
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
#line 461 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OADD;
	}
    break;

  case 55:
#line 466 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OSUB;
	}
    break;

  case 56:
#line 473 "go.y"
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
#line 493 "go.y"
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
#line 511 "go.y"
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
#line 520 "go.y"
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
#line 538 "go.y"
    {
		markdcl();
	}
    break;

  case 61:
#line 542 "go.y"
    {
		if((yyvsp[(3) - (4)].list) == nil)
			(yyval.node) = nod(OEMPTY, N, N);
		else
			(yyval.node) = liststmt((yyvsp[(3) - (4)].list));
		popdcl();
	}
    break;

  case 62:
#line 552 "go.y"
    {
		// If the last token read by the lexer was consumed
		// as part of the case, clear it (parser has cleared yychar).
		// If the last token read by the lexer was the lookahead
		// leave it alone (parser has it cached in yychar).
		// This is so that the stmt_list action doesn't look at
		// the case tokens if the stmt_list is empty.
		yylast = yychar;
		(yyvsp[(1) - (1)].node)->xoffset = block;
	}
    break;

  case 63:
#line 563 "go.y"
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
#line 583 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 65:
#line 587 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node));
	}
    break;

  case 66:
#line 593 "go.y"
    {
		markdcl();
	}
    break;

  case 67:
#line 597 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (4)].list);
		popdcl();
	}
    break;

  case 68:
#line 604 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->etype = 0;	// := flag
	}
    break;

  case 69:
#line 610 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->colas = 1;
		colasdefn((yyvsp[(1) - (4)].list), (yyval.node));
	}
    break;

  case 70:
#line 619 "go.y"
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
#line 630 "go.y"
    {
		// normal test
		(yyval.node) = nod(OFOR, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 73:
#line 639 "go.y"
    {
		(yyval.node) = (yyvsp[(1) - (2)].node);
		(yyval.node)->nbody = concat((yyval.node)->nbody, (yyvsp[(2) - (2)].list));
	}
    break;

  case 74:
#line 646 "go.y"
    {
		markdcl();
	}
    break;

  case 75:
#line 650 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (3)].node);
		popdcl();
	}
    break;

  case 76:
#line 657 "go.y"
    {
		// test
		(yyval.node) = nod(OIF, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 77:
#line 663 "go.y"
    {
		// init ; test
		(yyval.node) = nod(OIF, N, N);
		if((yyvsp[(1) - (3)].node) != N)
			(yyval.node)->ninit = list1((yyvsp[(1) - (3)].node));
		(yyval.node)->ntest = (yyvsp[(3) - (3)].node);
	}
    break;

  case 78:
#line 674 "go.y"
    {
		markdcl();
	}
    break;

  case 79:
#line 678 "go.y"
    {
		if((yyvsp[(3) - (3)].node)->ntest == N)
			yyerror("missing condition in if statement");
	}
    break;

  case 80:
#line 683 "go.y"
    {
		(yyvsp[(3) - (5)].node)->nbody = (yyvsp[(5) - (5)].list);
	}
    break;

  case 81:
#line 687 "go.y"
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
#line 704 "go.y"
    {
		markdcl();
	}
    break;

  case 83:
#line 708 "go.y"
    {
		if((yyvsp[(4) - (5)].node)->ntest == N)
			yyerror("missing condition in if statement");
		(yyvsp[(4) - (5)].node)->nbody = (yyvsp[(5) - (5)].list);
		(yyval.list) = list1((yyvsp[(4) - (5)].node));
	}
    break;

  case 84:
#line 716 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 85:
#line 720 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].list));
	}
    break;

  case 86:
#line 725 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 87:
#line 729 "go.y"
    {
		NodeList *node;
		
		node = mal(sizeof *node);
		node->n = (yyvsp[(2) - (2)].node);
		node->end = node;
		(yyval.list) = node;
	}
    break;

  case 88:
#line 740 "go.y"
    {
		markdcl();
	}
    break;

  case 89:
#line 744 "go.y"
    {
		Node *n;
		n = (yyvsp[(3) - (3)].node)->ntest;
		if(n != N && n->op != OTYPESW)
			n = N;
		typesw = nod(OXXX, typesw, n);
	}
    break;

  case 90:
#line 752 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (7)].node);
		(yyval.node)->op = OSWITCH;
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
		typesw = typesw->left;
		popdcl();
	}
    break;

  case 91:
#line 762 "go.y"
    {
		typesw = nod(OXXX, typesw, N);
	}
    break;

  case 92:
#line 766 "go.y"
    {
		(yyval.node) = nod(OSELECT, N, N);
		(yyval.node)->lineno = typesw->lineno;
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		typesw = typesw->left;
	}
    break;

  case 94:
#line 779 "go.y"
    {
		(yyval.node) = nod(OOROR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 95:
#line 783 "go.y"
    {
		(yyval.node) = nod(OANDAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 96:
#line 787 "go.y"
    {
		(yyval.node) = nod(OEQ, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 97:
#line 791 "go.y"
    {
		(yyval.node) = nod(ONE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 98:
#line 795 "go.y"
    {
		(yyval.node) = nod(OLT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 99:
#line 799 "go.y"
    {
		(yyval.node) = nod(OLE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 100:
#line 803 "go.y"
    {
		(yyval.node) = nod(OGE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 101:
#line 807 "go.y"
    {
		(yyval.node) = nod(OGT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 102:
#line 811 "go.y"
    {
		(yyval.node) = nod(OADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 103:
#line 815 "go.y"
    {
		(yyval.node) = nod(OSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 104:
#line 819 "go.y"
    {
		(yyval.node) = nod(OOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 105:
#line 823 "go.y"
    {
		(yyval.node) = nod(OXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 106:
#line 827 "go.y"
    {
		(yyval.node) = nod(OMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 107:
#line 831 "go.y"
    {
		(yyval.node) = nod(ODIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 108:
#line 835 "go.y"
    {
		(yyval.node) = nod(OMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 109:
#line 839 "go.y"
    {
		(yyval.node) = nod(OAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 110:
#line 843 "go.y"
    {
		(yyval.node) = nod(OANDNOT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 111:
#line 847 "go.y"
    {
		(yyval.node) = nod(OLSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 112:
#line 851 "go.y"
    {
		(yyval.node) = nod(ORSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 113:
#line 856 "go.y"
    {
		(yyval.node) = nod(OSEND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 115:
#line 863 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 116:
#line 867 "go.y"
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
#line 878 "go.y"
    {
		(yyval.node) = nod(OPLUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 118:
#line 882 "go.y"
    {
		(yyval.node) = nod(OMINUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 119:
#line 886 "go.y"
    {
		(yyval.node) = nod(ONOT, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 120:
#line 890 "go.y"
    {
		yyerror("the bitwise complement operator is ^");
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 121:
#line 895 "go.y"
    {
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 122:
#line 899 "go.y"
    {
		(yyval.node) = nod(ORECV, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 123:
#line 909 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (3)].node), N);
	}
    break;

  case 124:
#line 913 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (5)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
	}
    break;

  case 125:
#line 918 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (6)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (6)].list);
		(yyval.node)->isddd = 1;
	}
    break;

  case 126:
#line 926 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 128:
#line 931 "go.y"
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
#line 942 "go.y"
    {
		(yyval.node) = nod(ODOTTYPE, (yyvsp[(1) - (5)].node), (yyvsp[(4) - (5)].node));
	}
    break;

  case 130:
#line 946 "go.y"
    {
		(yyval.node) = nod(OTYPESW, N, (yyvsp[(1) - (5)].node));
	}
    break;

  case 131:
#line 950 "go.y"
    {
		(yyval.node) = nod(OINDEX, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 132:
#line 954 "go.y"
    {
		(yyval.node) = nod(OSLICE, (yyvsp[(1) - (6)].node), nod(OKEY, (yyvsp[(3) - (6)].node), (yyvsp[(5) - (6)].node)));
	}
    break;

  case 133:
#line 958 "go.y"
    {
		if((yyvsp[(5) - (8)].node) == N)
			yyerror("middle index required in 3-index slice");
		if((yyvsp[(7) - (8)].node) == N)
			yyerror("final index required in 3-index slice");
		(yyval.node) = nod(OSLICE3, (yyvsp[(1) - (8)].node), nod(OKEY, (yyvsp[(3) - (8)].node), nod(OKEY, (yyvsp[(5) - (8)].node), (yyvsp[(7) - (8)].node))));
	}
    break;

  case 135:
#line 967 "go.y"
    {
		// conversion
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (5)].node), N);
		(yyval.node)->list = list1((yyvsp[(3) - (5)].node));
	}
    break;

  case 136:
#line 973 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 137:
#line 980 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
	}
    break;

  case 138:
#line 986 "go.y"
    {
		yyerror("cannot parenthesize type in composite literal");
		(yyval.node) = (yyvsp[(5) - (7)].node);
		(yyval.node)->right = (yyvsp[(2) - (7)].node);
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
	}
    break;

  case 140:
#line 995 "go.y"
    {
		// composite expression.
		// make node early so we get the right line number.
		(yyval.node) = nod(OCOMPLIT, N, N);
	}
    break;

  case 141:
#line 1003 "go.y"
    {
		(yyval.node) = nod(OKEY, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 142:
#line 1009 "go.y"
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

  case 143:
#line 1026 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (4)].node);
		(yyval.node)->list = (yyvsp[(3) - (4)].list);
	}
    break;

  case 145:
#line 1034 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (4)].node);
		(yyval.node)->list = (yyvsp[(3) - (4)].list);
	}
    break;

  case 147:
#line 1042 "go.y"
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

  case 151:
#line 1068 "go.y"
    {
		(yyval.i) = LBODY;
	}
    break;

  case 152:
#line 1072 "go.y"
    {
		(yyval.i) = '{';
	}
    break;

  case 153:
#line 1083 "go.y"
    {
		if((yyvsp[(1) - (1)].sym) == S)
			(yyval.node) = N;
		else
			(yyval.node) = newname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 154:
#line 1092 "go.y"
    {
		(yyval.node) = dclname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 155:
#line 1097 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 157:
#line 1104 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if(importpkg != nil && !exportname((yyvsp[(1) - (1)].sym)->name))
			(yyval.sym) = pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg);
	}
    break;

  case 159:
#line 1112 "go.y"
    {
		(yyval.sym) = S;
	}
    break;

  case 160:
#line 1118 "go.y"
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

  case 161:
#line 1131 "go.y"
    {
		Pkg *p;

		if((yyvsp[(2) - (4)].val).u.sval->len == 0)
			p = importpkg;
		else {
			if(isbadimport((yyvsp[(2) - (4)].val).u.sval))
				errorexit();
			p = mkpkg((yyvsp[(2) - (4)].val).u.sval);
		}
		(yyval.sym) = pkglookup("?", p);
	}
    break;

  case 162:
#line 1146 "go.y"
    {
		(yyval.node) = oldname((yyvsp[(1) - (1)].sym));
		if((yyval.node)->pack != N)
			(yyval.node)->pack->used = 1;
	}
    break;

  case 164:
#line 1166 "go.y"
    {
		yyerror("final argument in variadic function missing type");
		(yyval.node) = nod(ODDD, typenod(typ(TINTER)), N);
	}
    break;

  case 165:
#line 1171 "go.y"
    {
		(yyval.node) = nod(ODDD, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 171:
#line 1182 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 175:
#line 1191 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 180:
#line 1201 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 190:
#line 1222 "go.y"
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

  case 191:
#line 1235 "go.y"
    {
		(yyval.node) = nod(OTARRAY, (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].node));
	}
    break;

  case 192:
#line 1239 "go.y"
    {
		// array literal of nelem
		(yyval.node) = nod(OTARRAY, nod(ODDD, N, N), (yyvsp[(4) - (4)].node));
	}
    break;

  case 193:
#line 1244 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->etype = Cboth;
	}
    break;

  case 194:
#line 1249 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Csend;
	}
    break;

  case 195:
#line 1254 "go.y"
    {
		(yyval.node) = nod(OTMAP, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node));
	}
    break;

  case 198:
#line 1262 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 199:
#line 1268 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Crecv;
	}
    break;

  case 200:
#line 1275 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 201:
#line 1281 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 202:
#line 1288 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 203:
#line 1294 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 204:
#line 1305 "go.y"
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

  case 205:
#line 1319 "go.y"
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

  case 206:
#line 1348 "go.y"
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

  case 207:
#line 1388 "go.y"
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

  case 208:
#line 1413 "go.y"
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

  case 209:
#line 1431 "go.y"
    {
		(yyvsp[(3) - (5)].list) = checkarglist((yyvsp[(3) - (5)].list), 1);
		(yyval.node) = nod(OTFUNC, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		(yyval.node)->rlist = (yyvsp[(5) - (5)].list);
	}
    break;

  case 210:
#line 1439 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 211:
#line 1443 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
		if((yyval.list) == nil)
			(yyval.list) = list1(nod(OEMPTY, N, N));
	}
    break;

  case 212:
#line 1451 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 213:
#line 1455 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, (yyvsp[(1) - (1)].node)));
	}
    break;

  case 214:
#line 1459 "go.y"
    {
		(yyvsp[(2) - (3)].list) = checkarglist((yyvsp[(2) - (3)].list), 0);
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 215:
#line 1466 "go.y"
    {
		closurehdr((yyvsp[(1) - (1)].node));
	}
    break;

  case 216:
#line 1472 "go.y"
    {
		(yyval.node) = closurebody((yyvsp[(3) - (4)].list));
		fixlbrace((yyvsp[(2) - (4)].i));
	}
    break;

  case 217:
#line 1477 "go.y"
    {
		(yyval.node) = closurebody(nil);
	}
    break;

  case 218:
#line 1488 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 219:
#line 1492 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(2) - (3)].list));
		if(nsyntaxerrors == 0)
			testdclstack();
		nointerface = 0;
		noescape = 0;
	}
    break;

  case 221:
#line 1503 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 223:
#line 1510 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 224:
#line 1516 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 225:
#line 1520 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 227:
#line 1527 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 228:
#line 1533 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 229:
#line 1537 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 230:
#line 1543 "go.y"
    {
		NodeList *l;

		Node *n;
		l = (yyvsp[(1) - (3)].list);
		if(l == nil) {
			// ? symbol, during import (list1(N) == nil)
			n = (yyvsp[(2) - (3)].node);
			if(n->op == OIND)
				n = n->left;
			n = embedded(n->sym, importpkg);
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

  case 231:
#line 1566 "go.y"
    {
		(yyvsp[(1) - (2)].node)->val = (yyvsp[(2) - (2)].val);
		(yyval.list) = list1((yyvsp[(1) - (2)].node));
	}
    break;

  case 232:
#line 1571 "go.y"
    {
		(yyvsp[(2) - (4)].node)->val = (yyvsp[(4) - (4)].val);
		(yyval.list) = list1((yyvsp[(2) - (4)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 233:
#line 1577 "go.y"
    {
		(yyvsp[(2) - (3)].node)->right = nod(OIND, (yyvsp[(2) - (3)].node)->right, N);
		(yyvsp[(2) - (3)].node)->val = (yyvsp[(3) - (3)].val);
		(yyval.list) = list1((yyvsp[(2) - (3)].node));
	}
    break;

  case 234:
#line 1583 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 235:
#line 1590 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 236:
#line 1599 "go.y"
    {
		Node *n;

		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		n = oldname((yyvsp[(1) - (1)].sym));
		if(n->pack != N)
			n->pack->used = 1;
	}
    break;

  case 237:
#line 1608 "go.y"
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

  case 238:
#line 1623 "go.y"
    {
		(yyval.node) = embedded((yyvsp[(1) - (1)].sym), localpkg);
	}
    break;

  case 239:
#line 1629 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
		ifacedcl((yyval.node));
	}
    break;

  case 240:
#line 1634 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(1) - (1)].sym)));
	}
    break;

  case 241:
#line 1638 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(2) - (3)].sym)));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 242:
#line 1645 "go.y"
    {
		// without func keyword
		(yyvsp[(2) - (4)].list) = checkarglist((yyvsp[(2) - (4)].list), 1);
		(yyval.node) = nod(OTFUNC, fakethis(), N);
		(yyval.node)->list = (yyvsp[(2) - (4)].list);
		(yyval.node)->rlist = (yyvsp[(4) - (4)].list);
	}
    break;

  case 244:
#line 1659 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 245:
#line 1665 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 247:
#line 1674 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 248:
#line 1678 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 249:
#line 1683 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 250:
#line 1687 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 251:
#line 1695 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 253:
#line 1700 "go.y"
    {
		(yyval.node) = liststmt((yyvsp[(1) - (1)].list));
	}
    break;

  case 255:
#line 1705 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 261:
#line 1716 "go.y"
    {
		(yyvsp[(1) - (2)].node) = nod(OLABEL, (yyvsp[(1) - (2)].node), N);
		(yyvsp[(1) - (2)].node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 262:
#line 1721 "go.y"
    {
		NodeList *l;

		(yyvsp[(1) - (4)].node)->defn = (yyvsp[(4) - (4)].node);
		l = list1((yyvsp[(1) - (4)].node));
		if((yyvsp[(4) - (4)].node))
			l = list(l, (yyvsp[(4) - (4)].node));
		(yyval.node) = liststmt(l);
	}
    break;

  case 263:
#line 1731 "go.y"
    {
		// will be converted to OFALL
		(yyval.node) = nod(OXFALL, N, N);
		(yyval.node)->xoffset = block;
	}
    break;

  case 264:
#line 1737 "go.y"
    {
		(yyval.node) = nod(OBREAK, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 265:
#line 1741 "go.y"
    {
		(yyval.node) = nod(OCONTINUE, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 266:
#line 1745 "go.y"
    {
		(yyval.node) = nod(OPROC, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 267:
#line 1749 "go.y"
    {
		(yyval.node) = nod(ODEFER, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 268:
#line 1753 "go.y"
    {
		(yyval.node) = nod(OGOTO, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 269:
#line 1758 "go.y"
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

  case 270:
#line 1777 "go.y"
    {
		(yyval.list) = nil;
		if((yyvsp[(1) - (1)].node) != N)
			(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 271:
#line 1783 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (3)].list);
		if((yyvsp[(3) - (3)].node) != N)
			(yyval.list) = list((yyval.list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 272:
#line 1791 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 273:
#line 1795 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 274:
#line 1801 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 275:
#line 1805 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 276:
#line 1811 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 277:
#line 1815 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 278:
#line 1821 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 279:
#line 1825 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 280:
#line 1834 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 281:
#line 1838 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 282:
#line 1842 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 283:
#line 1846 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 284:
#line 1851 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 285:
#line 1855 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 290:
#line 1869 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 292:
#line 1875 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 294:
#line 1881 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 296:
#line 1887 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 298:
#line 1893 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 300:
#line 1899 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 302:
#line 1905 "go.y"
    {
		(yyval.val).ctype = CTxxx;
	}
    break;

  case 304:
#line 1915 "go.y"
    {
		importimport((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].val).u.sval);
	}
    break;

  case 305:
#line 1919 "go.y"
    {
		importvar((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].type));
	}
    break;

  case 306:
#line 1923 "go.y"
    {
		importconst((yyvsp[(2) - (5)].sym), types[TIDEAL], (yyvsp[(4) - (5)].node));
	}
    break;

  case 307:
#line 1927 "go.y"
    {
		importconst((yyvsp[(2) - (6)].sym), (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].node));
	}
    break;

  case 308:
#line 1931 "go.y"
    {
		importtype((yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].type));
	}
    break;

  case 309:
#line 1935 "go.y"
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

  case 310:
#line 1955 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		structpkg = (yyval.sym)->pkg;
	}
    break;

  case 311:
#line 1962 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
		importsym((yyvsp[(1) - (1)].sym), OTYPE);
	}
    break;

  case 317:
#line 1982 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
	}
    break;

  case 318:
#line 1986 "go.y"
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

  case 319:
#line 1996 "go.y"
    {
		(yyval.type) = aindex(N, (yyvsp[(3) - (3)].type));
	}
    break;

  case 320:
#line 2000 "go.y"
    {
		(yyval.type) = aindex(nodlit((yyvsp[(2) - (4)].val)), (yyvsp[(4) - (4)].type));
	}
    break;

  case 321:
#line 2004 "go.y"
    {
		(yyval.type) = maptype((yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].type));
	}
    break;

  case 322:
#line 2008 "go.y"
    {
		(yyval.type) = tostruct((yyvsp[(3) - (4)].list));
	}
    break;

  case 323:
#line 2012 "go.y"
    {
		(yyval.type) = tointerface((yyvsp[(3) - (4)].list));
	}
    break;

  case 324:
#line 2016 "go.y"
    {
		(yyval.type) = ptrto((yyvsp[(2) - (2)].type));
	}
    break;

  case 325:
#line 2020 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(2) - (2)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 326:
#line 2026 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (4)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 327:
#line 2032 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Csend;
	}
    break;

  case 328:
#line 2040 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Crecv;
	}
    break;

  case 329:
#line 2048 "go.y"
    {
		(yyval.type) = functype(nil, (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list));
	}
    break;

  case 330:
#line 2054 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(2) - (3)].type)));
		if((yyvsp[(1) - (3)].sym))
			(yyval.node)->left = newname((yyvsp[(1) - (3)].sym));
		(yyval.node)->val = (yyvsp[(3) - (3)].val);
	}
    break;

  case 331:
#line 2061 "go.y"
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

  case 332:
#line 2077 "go.y"
    {
		Sym *s;
		Pkg *p;

		if((yyvsp[(1) - (3)].sym) != S && strcmp((yyvsp[(1) - (3)].sym)->name, "?") != 0) {
			(yyval.node) = nod(ODCLFIELD, newname((yyvsp[(1) - (3)].sym)), typenod((yyvsp[(2) - (3)].type)));
			(yyval.node)->val = (yyvsp[(3) - (3)].val);
		} else {
			s = (yyvsp[(2) - (3)].type)->sym;
			if(s == S && isptr[(yyvsp[(2) - (3)].type)->etype])
				s = (yyvsp[(2) - (3)].type)->type->sym;
			p = importpkg;
			if((yyvsp[(1) - (3)].sym) != S)
				p = (yyvsp[(1) - (3)].sym)->pkg;
			(yyval.node) = embedded(s, p);
			(yyval.node)->right = typenod((yyvsp[(2) - (3)].type));
			(yyval.node)->val = (yyvsp[(3) - (3)].val);
		}
	}
    break;

  case 333:
#line 2099 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, newname((yyvsp[(1) - (5)].sym)), typenod(functype(fakethis(), (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list))));
	}
    break;

  case 334:
#line 2103 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type)));
	}
    break;

  case 335:
#line 2108 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 337:
#line 2115 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 338:
#line 2119 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type))));
	}
    break;

  case 339:
#line 2129 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 340:
#line 2133 "go.y"
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
		case CTCPLX:
			mpnegflt(&(yyval.node)->val.u.cval->real);
			mpnegflt(&(yyval.node)->val.u.cval->imag);
			break;
		default:
			yyerror("bad negated constant");
		}
	}
    break;

  case 341:
#line 2152 "go.y"
    {
		(yyval.node) = oldname(pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg));
		if((yyval.node)->op != OLITERAL)
			yyerror("bad constant %S", (yyval.node)->sym);
	}
    break;

  case 343:
#line 2161 "go.y"
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

  case 346:
#line 2177 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 347:
#line 2181 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 348:
#line 2187 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 349:
#line 2191 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 350:
#line 2197 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 351:
#line 2201 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;


/* Line 1267 of yacc.c.  */
#line 4911 "y.tab.c"
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


#line 2205 "go.y"


static void
fixlbrace(int lbr)
{
	// If the opening brace was an LBODY,
	// set up for another one now that we're done.
	// See comment in lex.c about loophack.
	if(lbr == LBODY)
		loophack = 1;
}


