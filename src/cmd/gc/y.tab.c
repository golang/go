/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison implementation for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

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
#define YYBISON_VERSION "2.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Copy the first part of user declarations.  */

/* Line 268 of yacc.c  */
#line 20 "go.y"

#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and go.h re-#defines getc */
#include <libc.h>
#include "go.h"

static void fixlbrace(int);


/* Line 268 of yacc.c  */
#line 81 "y.tab.c"

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




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 293 of yacc.c  */
#line 28 "go.y"

	Node*		node;
	NodeList*		list;
	Type*		type;
	Sym*		sym;
	struct	Val	val;
	int		i;



/* Line 293 of yacc.c  */
#line 230 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 242 "y.tab.c"

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
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
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
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
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
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
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
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
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
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  4
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   2097

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  76
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  138
/* YYNRULES -- Number of rules.  */
#define YYNRULES  344
/* YYNRULES -- Number of states.  */
#define YYNSTATES  653

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
     258,   259,   267,   268,   271,   274,   275,   276,   284,   285,
     291,   293,   297,   301,   305,   309,   313,   317,   321,   325,
     329,   333,   337,   341,   345,   349,   353,   357,   361,   365,
     369,   373,   375,   378,   381,   384,   387,   390,   393,   396,
     399,   403,   409,   416,   418,   420,   424,   430,   436,   441,
     448,   450,   455,   461,   467,   475,   477,   478,   482,   484,
     489,   491,   495,   497,   499,   501,   503,   505,   507,   509,
     510,   512,   514,   516,   518,   523,   525,   527,   529,   532,
     534,   536,   538,   540,   542,   546,   548,   550,   552,   555,
     557,   559,   561,   563,   567,   569,   571,   573,   575,   577,
     579,   581,   583,   585,   589,   594,   599,   602,   606,   612,
     614,   616,   619,   623,   629,   633,   639,   643,   647,   653,
     662,   668,   677,   683,   684,   688,   689,   691,   695,   697,
     702,   705,   706,   710,   712,   716,   718,   722,   724,   728,
     730,   734,   736,   740,   744,   747,   752,   756,   762,   768,
     770,   774,   776,   779,   781,   785,   790,   792,   795,   798,
     800,   802,   806,   807,   810,   811,   813,   815,   817,   819,
     821,   823,   825,   827,   829,   830,   835,   837,   840,   843,
     846,   849,   852,   855,   857,   861,   863,   867,   869,   873,
     875,   879,   881,   885,   887,   889,   893,   897,   898,   901,
     902,   904,   905,   907,   908,   910,   911,   913,   914,   916,
     917,   919,   920,   922,   923,   925,   926,   928,   933,   938,
     944,   951,   956,   961,   963,   965,   967,   969,   971,   973,
     975,   977,   979,   983,   988,   994,   999,  1004,  1007,  1010,
    1015,  1019,  1023,  1029,  1033,  1038,  1042,  1048,  1050,  1051,
    1053,  1057,  1059,  1061,  1064,  1066,  1068,  1074,  1075,  1078,
    1080,  1084,  1086,  1090,  1092
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      77,     0,    -1,    79,    78,    81,   162,    -1,    -1,    25,
     137,    62,    -1,    -1,    80,    86,    88,    -1,    -1,    81,
      82,    62,    -1,    21,    83,    -1,    21,    59,    84,   186,
      60,    -1,    21,    59,    60,    -1,    85,    86,    88,    -1,
      85,    88,    -1,    83,    -1,    84,    62,    83,    -1,     3,
      -1,   137,     3,    -1,    63,     3,    -1,    25,    24,    87,
      62,    -1,    -1,    24,    -1,    -1,    89,   210,    64,    64,
      -1,    -1,    91,    -1,   154,    -1,   177,    -1,     1,    -1,
      32,    93,    -1,    32,    59,   163,   186,    60,    -1,    32,
      59,    60,    -1,    92,    94,    -1,    92,    59,    94,   186,
      60,    -1,    92,    59,    94,    62,   164,   186,    60,    -1,
      92,    59,    60,    -1,    31,    97,    -1,    31,    59,   165,
     186,    60,    -1,    31,    59,    60,    -1,     9,    -1,   181,
     142,    -1,   181,   142,    65,   182,    -1,   181,    65,   182,
      -1,   181,   142,    65,   182,    -1,   181,    65,   182,    -1,
      94,    -1,   181,   142,    -1,   181,    -1,   137,    -1,    96,
     142,    -1,   123,    -1,   123,     4,   123,    -1,   182,    65,
     182,    -1,   182,     5,   182,    -1,   123,    42,    -1,   123,
      37,    -1,     7,   183,    66,    -1,     7,   183,    65,   123,
      66,    -1,     7,   183,     5,   123,    66,    -1,    12,    66,
      -1,    -1,    67,   101,   179,    68,    -1,    -1,    99,   103,
     179,    -1,    -1,   104,   102,    -1,    -1,    35,   106,   179,
      68,    -1,   182,    65,    26,   123,    -1,   182,     5,    26,
     123,    -1,   190,    62,   190,    62,   190,    -1,   190,    -1,
     107,    -1,   108,   105,    -1,    -1,    16,   111,   109,    -1,
     190,    -1,   190,    62,   190,    -1,    -1,    -1,    -1,    20,
     114,   112,   115,   105,   116,   117,    -1,    -1,    14,   113,
      -1,    14,   100,    -1,    -1,    -1,    30,   119,   112,   120,
      35,   104,    68,    -1,    -1,    28,   122,    35,   104,    68,
      -1,   124,    -1,   123,    47,   123,    -1,   123,    33,   123,
      -1,   123,    38,   123,    -1,   123,    46,   123,    -1,   123,
      45,   123,    -1,   123,    43,   123,    -1,   123,    39,   123,
      -1,   123,    40,   123,    -1,   123,    49,   123,    -1,   123,
      50,   123,    -1,   123,    51,   123,    -1,   123,    52,   123,
      -1,   123,    53,   123,    -1,   123,    54,   123,    -1,   123,
      55,   123,    -1,   123,    56,   123,    -1,   123,    34,   123,
      -1,   123,    44,   123,    -1,   123,    48,   123,    -1,   123,
      36,   123,    -1,   130,    -1,    53,   124,    -1,    56,   124,
      -1,    49,   124,    -1,    50,   124,    -1,    69,   124,    -1,
      70,   124,    -1,    52,   124,    -1,    36,   124,    -1,   130,
      59,    60,    -1,   130,    59,   183,   187,    60,    -1,   130,
      59,   183,    11,   187,    60,    -1,     3,    -1,   139,    -1,
     130,    63,   137,    -1,   130,    63,    59,   131,    60,    -1,
     130,    63,    59,    31,    60,    -1,   130,    71,   123,    72,
      -1,   130,    71,   188,    66,   188,    72,    -1,   125,    -1,
     145,    59,   123,    60,    -1,   146,   133,   127,   185,    68,
      -1,   126,    67,   127,   185,    68,    -1,    59,   131,    60,
      67,   127,   185,    68,    -1,   161,    -1,    -1,   123,    66,
     129,    -1,   123,    -1,    67,   127,   185,    68,    -1,   126,
      -1,    59,   131,    60,    -1,   123,    -1,   143,    -1,   142,
      -1,    35,    -1,    67,    -1,   137,    -1,   137,    -1,    -1,
     134,    -1,    24,    -1,   138,    -1,    73,    -1,    74,     3,
      63,    24,    -1,   137,    -1,   134,    -1,    11,    -1,    11,
     142,    -1,   151,    -1,   157,    -1,   149,    -1,   150,    -1,
     148,    -1,    59,   142,    60,    -1,   151,    -1,   157,    -1,
     149,    -1,    53,   143,    -1,   157,    -1,   149,    -1,   150,
      -1,   148,    -1,    59,   142,    60,    -1,   157,    -1,   149,
      -1,   149,    -1,   151,    -1,   157,    -1,   149,    -1,   150,
      -1,   148,    -1,   139,    -1,   139,    63,   137,    -1,    71,
     188,    72,   142,    -1,    71,    11,    72,   142,    -1,     8,
     144,    -1,     8,    36,   142,    -1,    23,    71,   142,    72,
     142,    -1,   152,    -1,   153,    -1,    53,   142,    -1,    36,
       8,   142,    -1,    29,   133,   166,   186,    68,    -1,    29,
     133,    68,    -1,    22,   133,   167,   186,    68,    -1,    22,
     133,    68,    -1,    17,   155,   158,    -1,   137,    59,   175,
      60,   159,    -1,    59,   175,    60,   137,    59,   175,    60,
     159,    -1,   196,    59,   191,    60,   206,    -1,    59,   211,
      60,   137,    59,   191,    60,   206,    -1,    17,    59,   175,
      60,   159,    -1,    -1,    67,   179,    68,    -1,    -1,   147,
      -1,    59,   175,    60,    -1,   157,    -1,   160,   133,   179,
      68,    -1,   160,     1,    -1,    -1,   162,    90,    62,    -1,
      93,    -1,   163,    62,    93,    -1,    95,    -1,   164,    62,
      95,    -1,    97,    -1,   165,    62,    97,    -1,   168,    -1,
     166,    62,   168,    -1,   171,    -1,   167,    62,   171,    -1,
     180,   142,   194,    -1,   170,   194,    -1,    59,   170,    60,
     194,    -1,    53,   170,   194,    -1,    59,    53,   170,    60,
     194,    -1,    53,    59,   170,    60,   194,    -1,    24,    -1,
      24,    63,   137,    -1,   169,    -1,   134,   172,    -1,   169,
      -1,    59,   169,    60,    -1,    59,   175,    60,   159,    -1,
     132,    -1,   137,   132,    -1,   137,   141,    -1,   141,    -1,
     173,    -1,   174,    75,   173,    -1,    -1,   174,   187,    -1,
      -1,   100,    -1,    91,    -1,   177,    -1,     1,    -1,    98,
      -1,   110,    -1,   118,    -1,   121,    -1,   113,    -1,    -1,
     140,    66,   178,   176,    -1,    15,    -1,     6,   136,    -1,
      10,   136,    -1,    18,   125,    -1,    13,   125,    -1,    19,
     134,    -1,    27,   189,    -1,   176,    -1,   179,    62,   176,
      -1,   134,    -1,   180,    75,   134,    -1,   135,    -1,   181,
      75,   135,    -1,   123,    -1,   182,    75,   123,    -1,   131,
      -1,   183,    75,   131,    -1,   128,    -1,   129,    -1,   184,
      75,   128,    -1,   184,    75,   129,    -1,    -1,   184,   187,
      -1,    -1,    62,    -1,    -1,    75,    -1,    -1,   123,    -1,
      -1,   182,    -1,    -1,    98,    -1,    -1,   211,    -1,    -1,
     212,    -1,    -1,   213,    -1,    -1,     3,    -1,    21,    24,
       3,    62,    -1,    32,   196,   198,    62,    -1,     9,   196,
      65,   209,    62,    -1,     9,   196,   198,    65,   209,    62,
      -1,    31,   197,   198,    62,    -1,    17,   156,   158,    62,
      -1,   138,    -1,   196,    -1,   200,    -1,   201,    -1,   202,
      -1,   200,    -1,   202,    -1,   138,    -1,    24,    -1,    71,
      72,   198,    -1,    71,     3,    72,   198,    -1,    23,    71,
     198,    72,   198,    -1,    29,    67,   192,    68,    -1,    22,
      67,   193,    68,    -1,    53,   198,    -1,     8,   199,    -1,
       8,    59,   201,    60,    -1,     8,    36,   198,    -1,    36,
       8,   198,    -1,    17,    59,   191,    60,   206,    -1,   137,
     198,   194,    -1,   137,    11,   198,   194,    -1,   137,   198,
     194,    -1,   137,    59,   191,    60,   206,    -1,   198,    -1,
      -1,   207,    -1,    59,   191,    60,    -1,   198,    -1,     3,
      -1,    50,     3,    -1,   137,    -1,   208,    -1,    59,   208,
      49,   208,    60,    -1,    -1,   210,   195,    -1,   203,    -1,
     211,    75,   203,    -1,   204,    -1,   212,    62,   204,    -1,
     205,    -1,   213,    62,   205,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   124,   124,   133,   140,   151,   151,   166,   167,   170,
     171,   172,   175,   208,   219,   220,   223,   230,   237,   246,
     259,   260,   267,   267,   280,   284,   285,   289,   294,   300,
     304,   308,   312,   318,   324,   330,   335,   339,   343,   349,
     355,   359,   363,   369,   373,   379,   380,   384,   390,   399,
     405,   409,   414,   426,   442,   447,   454,   474,   492,   501,
     520,   519,   531,   530,   561,   564,   571,   570,   581,   587,
     596,   607,   613,   616,   624,   623,   634,   640,   652,   656,
     661,   651,   673,   676,   680,   687,   691,   686,   709,   708,
     724,   725,   729,   733,   737,   741,   745,   749,   753,   757,
     761,   765,   769,   773,   777,   781,   785,   789,   793,   797,
     802,   808,   809,   813,   824,   828,   832,   836,   841,   845,
     855,   859,   864,   872,   876,   877,   888,   892,   896,   900,
     904,   905,   911,   918,   924,   931,   934,   941,   947,   948,
     955,   956,   974,   975,   978,   981,   985,   996,  1005,  1011,
    1014,  1017,  1024,  1025,  1031,  1040,  1048,  1060,  1065,  1071,
    1072,  1073,  1074,  1075,  1076,  1082,  1083,  1084,  1085,  1091,
    1092,  1093,  1094,  1095,  1101,  1102,  1105,  1108,  1109,  1110,
    1111,  1112,  1115,  1116,  1129,  1133,  1138,  1143,  1148,  1152,
    1153,  1156,  1162,  1169,  1175,  1182,  1188,  1199,  1210,  1239,
    1278,  1301,  1318,  1327,  1330,  1338,  1342,  1346,  1353,  1359,
    1364,  1376,  1379,  1387,  1388,  1394,  1395,  1401,  1405,  1411,
    1412,  1418,  1422,  1428,  1451,  1456,  1462,  1468,  1475,  1484,
    1493,  1508,  1514,  1519,  1523,  1530,  1543,  1544,  1550,  1556,
    1559,  1563,  1569,  1572,  1581,  1584,  1585,  1589,  1590,  1596,
    1597,  1598,  1599,  1600,  1602,  1601,  1616,  1621,  1625,  1629,
    1633,  1637,  1642,  1661,  1667,  1675,  1679,  1685,  1689,  1695,
    1699,  1705,  1709,  1718,  1722,  1726,  1730,  1736,  1739,  1747,
    1748,  1750,  1751,  1754,  1757,  1760,  1763,  1766,  1769,  1772,
    1775,  1778,  1781,  1784,  1787,  1790,  1793,  1799,  1803,  1807,
    1811,  1815,  1819,  1837,  1844,  1855,  1856,  1857,  1860,  1861,
    1864,  1868,  1878,  1882,  1886,  1890,  1894,  1898,  1902,  1908,
    1914,  1922,  1930,  1936,  1943,  1959,  1977,  1981,  1987,  1990,
    1993,  1997,  2007,  2011,  2026,  2034,  2035,  2045,  2046,  2049,
    2053,  2059,  2063,  2069,  2073
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
  "package", "loadsys", "$@1", "imports", "import", "import_stmt",
  "import_stmt_list", "import_here", "import_package", "import_safety",
  "import_there", "$@2", "xdcl", "common_dcl", "lconst", "vardcl",
  "constdcl", "constdcl1", "typedclname", "typedcl", "simple_stmt", "case",
  "compound_stmt", "$@3", "caseblock", "$@4", "caseblock_list",
  "loop_body", "$@5", "range_stmt", "for_header", "for_body", "for_stmt",
  "$@6", "if_header", "if_stmt", "$@7", "$@8", "$@9", "else",
  "switch_stmt", "$@10", "$@11", "select_stmt", "$@12", "expr", "uexpr",
  "pseudocall", "pexpr_no_paren", "start_complit", "keyval", "complitexpr",
  "pexpr", "expr_or_type", "name_or_type", "lbrace", "new_name",
  "dcl_name", "onew_name", "sym", "hidden_importsym", "name", "labelname",
  "dotdotdot", "ntype", "non_expr_type", "non_recvchantype", "convtype",
  "comptype", "fnret_type", "dotname", "othertype", "ptrtype",
  "recvchantype", "structtype", "interfacetype", "xfndcl", "fndcl",
  "hidden_fndcl", "fntype", "fnbody", "fnres", "fnlitdcl", "fnliteral",
  "xdcl_list", "vardcl_list", "constdcl_list", "typedcl_list",
  "structdcl_list", "interfacedcl_list", "structdcl", "packname", "embed",
  "interfacedcl", "indcl", "arg_type", "arg_type_list",
  "oarg_type_list_ocomma", "stmt", "non_dcl_stmt", "$@13", "stmt_list",
  "new_name_list", "dcl_name_list", "expr_list", "expr_or_type_list",
  "keyval_list", "braced_keyval_list", "osemi", "ocomma", "oexpr",
  "oexpr_list", "osimple_stmt", "ohidden_funarg_list",
  "ohidden_structdcl_list", "ohidden_interfacedcl_list", "oliteral",
  "hidden_import", "hidden_pkg_importsym", "hidden_pkgtype", "hidden_type",
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
     116,   113,   117,   117,   117,   119,   120,   118,   122,   121,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   124,   124,   124,   124,   124,   124,   124,   124,   124,
     125,   125,   125,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   126,   126,   126,   127,   128,   129,   129,
     130,   130,   131,   131,   132,   133,   133,   134,   135,   136,
     136,   137,   137,   137,   138,   139,   140,   141,   141,   142,
     142,   142,   142,   142,   142,   143,   143,   143,   143,   144,
     144,   144,   144,   144,   145,   145,   146,   147,   147,   147,
     147,   147,   148,   148,   149,   149,   149,   149,   149,   149,
     149,   150,   151,   152,   152,   153,   153,   154,   155,   155,
     156,   156,   157,   158,   158,   159,   159,   159,   160,   161,
     161,   162,   162,   163,   163,   164,   164,   165,   165,   166,
     166,   167,   167,   168,   168,   168,   168,   168,   168,   169,
     169,   170,   171,   171,   171,   172,   173,   173,   173,   173,
     174,   174,   175,   175,   176,   176,   176,   176,   176,   177,
     177,   177,   177,   177,   178,   177,   177,   177,   177,   177,
     177,   177,   177,   179,   179,   180,   180,   181,   181,   182,
     182,   183,   183,   184,   184,   184,   184,   185,   185,   186,
     186,   187,   187,   188,   188,   189,   189,   190,   190,   191,
     191,   192,   192,   193,   193,   194,   194,   195,   195,   195,
     195,   195,   195,   196,   197,   198,   198,   198,   199,   199,
     200,   200,   200,   200,   200,   200,   200,   200,   200,   200,
     200,   201,   202,   203,   203,   204,   205,   205,   206,   206,
     207,   207,   208,   208,   208,   209,   209,   210,   210,   211,
     211,   212,   212,   213,   213
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
       0,     7,     0,     2,     2,     0,     0,     7,     0,     5,
       1,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     1,     2,     2,     2,     2,     2,     2,     2,     2,
       3,     5,     6,     1,     1,     3,     5,     5,     4,     6,
       1,     4,     5,     5,     7,     1,     0,     3,     1,     4,
       1,     3,     1,     1,     1,     1,     1,     1,     1,     0,
       1,     1,     1,     1,     4,     1,     1,     1,     2,     1,
       1,     1,     1,     1,     3,     1,     1,     1,     2,     1,
       1,     1,     1,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     3,     4,     4,     2,     3,     5,     1,
       1,     2,     3,     5,     3,     5,     3,     3,     5,     8,
       5,     8,     5,     0,     3,     0,     1,     3,     1,     4,
       2,     0,     3,     1,     3,     1,     3,     1,     3,     1,
       3,     1,     3,     3,     2,     4,     3,     5,     5,     1,
       3,     1,     2,     1,     3,     4,     1,     2,     2,     1,
       1,     3,     0,     2,     0,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     0,     4,     1,     2,     2,     2,
       2,     2,     2,     1,     3,     1,     3,     1,     3,     1,
       3,     1,     3,     1,     1,     3,     3,     0,     2,     0,
       1,     0,     1,     0,     1,     0,     1,     0,     1,     0,
       1,     0,     1,     0,     1,     0,     1,     4,     4,     5,
       6,     4,     4,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     4,     5,     4,     4,     2,     2,     4,
       3,     3,     5,     3,     4,     3,     5,     1,     0,     1,
       3,     1,     1,     2,     1,     1,     5,     0,     2,     1,
       3,     1,     3,     1,     3
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       5,     0,     3,     0,     1,     0,     7,     0,    22,   151,
     153,     0,     0,   152,   211,    20,     6,   337,     0,     4,
       0,     0,     0,    21,     0,     0,     0,    16,     0,     0,
       9,    22,     0,     8,    28,   123,   149,     0,    39,   149,
       0,   256,    74,     0,     0,     0,    78,     0,     0,   285,
      88,     0,    85,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   283,     0,    25,     0,   249,   250,
     253,   251,   252,    50,    90,   130,   140,   111,   156,   155,
     124,     0,     0,     0,   176,   189,   190,    26,   208,     0,
     135,    27,     0,    19,     0,     0,     0,     0,     0,     0,
     338,   154,    11,    14,   279,    18,    22,    13,    17,   150,
     257,   147,     0,     0,     0,     0,   155,   182,   186,   172,
     170,   171,   169,   258,   130,     0,   287,   242,     0,   203,
     130,   261,   287,   145,   146,     0,     0,   269,   286,   262,
       0,     0,   287,     0,     0,    36,    48,     0,    29,   267,
     148,     0,   119,   114,   115,   118,   112,   113,     0,     0,
     142,     0,   143,   167,   165,   166,   116,   117,     0,   284,
       0,   212,     0,    32,     0,     0,     0,     0,     0,    55,
       0,     0,     0,    54,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   136,     0,
       0,   283,   254,     0,   136,   210,     0,     0,     0,     0,
     303,     0,     0,   203,     0,     0,   304,     0,     0,    23,
     280,     0,    12,   242,     0,     0,   187,   163,   161,   162,
     159,   160,   191,     0,     0,   288,    72,     0,    75,     0,
      71,   157,   236,   155,   239,   144,   240,   281,     0,   242,
       0,   197,    79,    76,   151,     0,   196,     0,   279,   233,
     221,     0,    64,     0,     0,   194,   265,   279,   219,   231,
     295,     0,    86,    38,   217,   279,    49,    31,   213,   279,
       0,     0,    40,     0,   168,   141,     0,     0,    35,   279,
       0,     0,    51,    92,   107,   110,    93,    97,    98,    96,
     108,    95,    94,    91,   109,    99,   100,   101,   102,   103,
     104,   105,   106,   277,   120,   271,   281,     0,   125,   284,
       0,     0,     0,   277,   248,    60,   246,   245,   263,   247,
       0,    53,    52,   270,     0,     0,     0,     0,   311,     0,
       0,     0,     0,     0,   310,     0,   305,   306,   307,     0,
     339,     0,     0,   289,     0,     0,     0,    15,    10,     0,
       0,     0,   173,   183,    66,    73,     0,     0,   287,   158,
     237,   238,   282,   243,   205,     0,     0,     0,   287,     0,
     229,     0,   242,   232,   280,     0,     0,     0,     0,   295,
       0,     0,   280,     0,   296,   224,     0,   295,     0,   280,
       0,   280,     0,    42,   268,     0,     0,     0,   192,   163,
     161,   162,   160,   136,   185,   184,   280,     0,    44,     0,
     136,   138,   273,   274,   281,     0,   281,   282,     0,     0,
       0,   128,   283,   255,   131,     0,     0,     0,   209,     0,
       0,   318,   308,   309,   289,   293,     0,   291,     0,   317,
     332,     0,     0,   334,   335,     0,     0,     0,     0,     0,
     295,     0,     0,   302,     0,   290,   297,   301,   298,   205,
     164,     0,     0,     0,     0,   241,   242,   155,   206,   181,
     179,   180,   177,   178,   202,   205,   204,    80,    77,   230,
     234,     0,   222,   195,   188,     0,     0,    89,    62,    65,
       0,   226,     0,   295,   220,   193,   266,   223,    64,   218,
      37,   214,    30,    41,     0,   277,    45,   215,   279,    47,
      33,    43,   277,     0,   282,   278,   133,   282,     0,   272,
     121,   127,   126,     0,   132,     0,   264,   320,     0,     0,
     311,     0,   310,     0,   327,   343,   294,     0,     0,     0,
     341,   292,   321,   333,     0,   299,     0,   312,     0,   295,
     323,     0,   340,   328,     0,    69,    68,   287,     0,   242,
     198,    82,   205,     0,    59,     0,   295,   295,   225,     0,
     164,     0,   280,     0,    46,     0,   138,   137,   275,   276,
     122,   129,    61,   319,   328,   289,   316,     0,     0,   295,
     315,     0,     0,   313,   300,   324,   289,   289,   331,   200,
     329,    67,    70,   207,     0,     0,    81,   235,     0,     0,
      56,     0,    63,   228,   227,    87,   134,   216,    34,   139,
     322,     0,   344,   314,   325,   342,     0,     0,     0,   205,
      84,    83,     0,     0,   328,   336,   328,   330,   199,    58,
      57,   326,   201
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     6,     2,     3,    14,    21,    30,   104,    31,
       8,    24,    16,    17,    65,   326,    67,   148,   516,   517,
     144,   145,    68,   498,   327,   436,   499,   575,   387,   365,
     471,   236,   237,   238,    69,   126,   252,    70,   132,   377,
     571,   616,    71,   142,   398,    72,   140,    73,    74,    75,
      76,   313,   422,   423,    77,   315,   242,   135,    78,   149,
     110,   116,    13,    80,    81,   244,   245,   162,   118,    82,
      83,   478,   227,    84,   229,   230,    85,    86,    87,   129,
     213,    88,   251,   484,    89,    90,    22,   279,   518,   275,
     267,   258,   268,   269,   270,   260,   383,   246,   247,   248,
     328,   329,   321,   330,   271,   151,    92,   316,   424,   425,
     221,   373,   170,   139,   253,   464,   549,   543,   395,   100,
     211,   217,   608,   441,   346,   347,   348,   350,   550,   545,
     609,   610,   454,   455,    25,   465,   551,   546
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -549
static const yytype_int16 yypact[] =
{
    -549,    63,    58,    91,  -549,   227,  -549,    44,  -549,  -549,
    -549,    67,    57,  -549,   111,   135,  -549,  -549,   123,  -549,
      50,   151,   904,  -549,   157,   463,   149,  -549,    54,   219,
    -549,    91,   230,  -549,  -549,  -549,   227,  1654,  -549,   227,
     288,  -549,  -549,   316,   288,   227,  -549,    21,   167,  1466,
    -549,    21,  -549,   327,   333,  1466,  1466,  1466,  1466,  1466,
    1466,  1509,  1466,  1466,   985,   193,  -549,   419,  -549,  -549,
    -549,  -549,  -549,   796,  -549,  -549,   176,     1,  -549,   194,
    -549,   196,   206,    21,   215,  -549,  -549,  -549,   216,    51,
    -549,  -549,    45,  -549,   203,     7,   256,   203,   203,   231,
    -549,  -549,  -549,  -549,   244,  -549,  -549,  -549,  -549,  -549,
    -549,  -549,   234,  1681,  1681,  1681,  -549,   250,  -549,  -549,
    -549,  -549,  -549,  -549,   168,     1,  1466,  1628,   260,   264,
     266,  -549,  1466,  -549,  -549,   110,  1681,  1993,   257,  -549,
     300,   229,  1466,   451,  1681,  -549,  -549,   461,  -549,  -549,
    -549,   662,  -549,  -549,  -549,  -549,  -549,  -549,  1552,  1509,
    1993,   263,  -549,    10,  -549,   165,  -549,  -549,   270,  1993,
     274,  -549,   485,  -549,  1595,  1466,  1466,  1466,  1466,  -549,
    1466,  1466,  1466,  -549,  1466,  1466,  1466,  1466,  1466,  1466,
    1466,  1466,  1466,  1466,  1466,  1466,  1466,  1466,  -549,  1203,
     506,  1466,  -549,  1466,  -549,  -549,  1134,  1466,  1466,  1466,
    -549,   673,   227,   264,   293,   355,  -549,  1214,  1214,  -549,
      76,   303,  -549,  1628,   357,  1681,  -549,  -549,  -549,  -549,
    -549,  -549,  -549,   328,   227,  -549,  -549,   358,  -549,    68,
     337,  1681,  -549,  1628,  -549,  -549,  -549,   329,   348,  1628,
    1134,  -549,  -549,   351,   128,   390,  -549,   359,   354,  -549,
    -549,   349,  -549,    30,    34,  -549,  -549,   366,  -549,  -549,
     427,  1620,  -549,  -549,  -549,   371,  -549,  -549,  -549,   373,
    1466,   227,   376,  1707,  -549,   391,  1681,  1681,  -549,   400,
    1466,   398,  1993,  1839,  -549,  2017,   755,   755,   755,   755,
    -549,   755,   755,  2041,  -549,   582,   582,   582,   582,  -549,
    -549,  -549,  -549,  1258,  -549,  -549,    33,  1313,  -549,  1866,
     356,  1060,  1968,  1258,  -549,  -549,  -549,  -549,  -549,  -549,
       3,   257,   257,  1993,  1748,   405,   402,   403,  -549,   412,
     475,  1214,    52,    29,  -549,   421,  -549,  -549,  -549,  1774,
    -549,    85,   425,   227,   426,   429,   434,  -549,  -549,   438,
    1681,   439,  -549,  -549,  -549,  -549,  1368,  1423,  1466,  -549,
    -549,  -549,  1628,  -549,  1715,   440,    86,   358,  1466,   227,
     441,   443,  1628,  -549,   508,   437,  1681,    78,   390,   427,
     390,   446,   280,   442,  -549,  -549,   227,   427,   454,   227,
     448,   227,   453,   257,  -549,  1466,  1740,  1681,  -549,   181,
     248,   338,   360,  -549,  -549,  -549,   227,   455,   257,  1466,
    -549,  1896,  -549,  -549,   447,   450,   456,  1509,   457,   466,
     469,  -549,  1466,  -549,  -549,   468,  1134,  1060,  -549,  1214,
     501,  -549,  -549,  -549,   227,  1801,  1214,   227,  1214,  -549,
    -549,   537,   161,  -549,  -549,   480,   488,  1214,    52,  1214,
     427,   227,   227,  -549,   503,   486,  -549,  -549,  -549,  1715,
    -549,  1134,  1466,  1466,   514,  -549,  1628,   509,  -549,  -549,
    -549,  -549,  -549,  -549,  -549,  1715,  -549,  -549,  -549,  -549,
    -549,   518,  -549,  -549,  -549,  1509,   517,  -549,  -549,  -549,
     525,  -549,   528,   427,  -549,  -549,  -549,  -549,  -549,  -549,
    -549,  -549,  -549,   257,   531,  1258,  -549,  -549,   532,  1595,
    -549,   257,  1258,  1258,  1258,  -549,  -549,  -549,   533,  -549,
    -549,  -549,  -549,   526,  -549,   109,  -549,  -549,   539,   540,
     545,   546,   550,   543,  -549,  -549,   551,   547,  1214,   552,
    -549,   556,  -549,  -549,   573,  -549,  1214,  -549,   561,   427,
    -549,   565,  -549,  1827,   131,  1993,  1993,  1466,   567,  1628,
    -549,   611,  1715,    28,  -549,  1060,   427,   427,  -549,   100,
     367,   563,   227,   574,   398,   571,  1993,  -549,  -549,  -549,
    -549,  -549,  -549,  -549,  1827,   227,  -549,  1801,  1214,   427,
    -549,   227,   161,  -549,  -549,  -549,   227,   227,  -549,  -549,
    -549,  -549,  -549,  -549,   581,    15,  -549,  -549,  1466,  1466,
    -549,  1509,   580,  -549,  -549,  -549,  -549,  -549,  -549,  -549,
    -549,   584,  -549,  -549,  -549,  -549,   585,   586,   587,  1715,
    -549,  -549,  1920,  1944,  1827,  -549,  1827,  -549,  -549,  -549,
    -549,  -549,  -549
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -549,  -549,  -549,  -549,  -549,  -549,  -549,    -6,  -549,  -549,
     617,  -549,   -11,  -549,  -549,   627,  -549,  -134,   -25,    71,
    -549,  -135,  -121,  -549,    35,  -549,  -549,  -549,   146,   279,
    -549,  -549,  -549,  -549,  -549,  -549,   513,    42,  -549,  -549,
    -549,  -549,  -549,  -549,  -549,  -549,  -549,   579,   493,   245,
    -549,  -192,   134,  -318,   278,   -47,   418,     8,   -20,   381,
     624,    -5,   449,   346,  -549,   422,    95,   510,  -549,  -549,
    -549,  -549,   -33,    38,   -31,   -18,  -549,  -549,  -549,  -549,
    -549,    43,   458,  -467,  -549,  -549,  -549,  -549,  -549,  -549,
    -549,  -549,   276,  -126,  -227,   289,  -549,   302,  -549,  -220,
    -297,   650,  -549,  -248,  -549,   -66,    18,   183,  -549,  -295,
    -228,  -289,  -191,  -549,  -119,  -403,  -549,  -549,  -305,  -549,
     273,  -549,   127,  -549,   342,   240,   353,   226,    88,    96,
    -548,  -549,  -426,   236,  -549,   487,  -549,  -549
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -270
static const yytype_int16 yytable[] =
{
      12,   174,   376,   359,   119,   235,   121,   240,   274,   259,
     320,   235,   323,   278,   161,    32,   109,    79,   570,   109,
     107,   235,   103,    32,   433,   131,   554,   428,   435,   375,
     385,   111,   456,   618,   111,    46,   389,   391,   128,   393,
     111,   539,   173,   164,   426,  -176,   630,   400,   146,   150,
     207,   402,   205,    27,   380,   450,   133,    27,   380,   141,
     199,   417,   150,     4,   200,   437,   212,   138,    15,  -175,
      18,   438,   201,   366,     9,   120,     9,  -176,     9,    27,
     122,    11,   325,     5,   501,   495,   133,   390,   134,   388,
     496,   204,   507,   619,   620,   222,   651,   206,   652,   163,
       9,   457,   451,   621,   165,   617,   174,   495,   427,    28,
     208,   452,   496,    29,   102,   257,     7,    29,   134,    19,
     209,   266,   243,    10,    11,    10,    11,    10,    11,   381,
     111,  -229,    20,   367,   254,   525,   111,   528,   146,    29,
     536,   164,   150,   209,   239,   461,   497,   289,   437,    10,
      11,   228,   228,   228,   486,   560,   231,   231,   231,    23,
     462,   500,   491,   502,   450,   228,  -208,   150,   625,   255,
     231,   437,   648,   101,   228,  -260,   636,   592,   256,   231,
    -260,   164,   228,    10,    11,     9,    26,   231,   535,   228,
    -229,   379,   631,   437,   231,   318,  -229,   163,   578,   611,
    -208,    79,   165,   637,   638,   587,   589,   349,   226,   232,
     233,   451,   228,    33,   357,    32,  -172,   231,   243,    93,
     581,   515,   105,   564,  -174,   331,   332,   585,   522,   363,
    -260,   261,  -208,   108,    10,    11,  -260,   163,   136,   276,
    -172,   533,   165,   198,   243,    79,   282,   235,  -172,   474,
     409,     9,   411,   254,   605,   171,   568,   235,   259,   488,
    -147,   228,   202,   228,   509,   203,   231,   511,   231,   291,
     430,   623,   624,  -259,  -175,  -174,   150,    11,  -259,   228,
     215,   228,   263,  -170,   231,   124,   231,   228,   264,   130,
     583,    35,   231,   223,   634,   219,    37,   265,   403,   164,
      10,    11,    10,    11,   254,   112,   220,  -170,   418,   228,
      47,    48,     9,   234,   231,  -170,    79,    51,   125,   249,
     361,   410,   125,   285,   228,   228,   412,   622,  -259,   231,
     231,   250,   209,   263,  -259,   262,   369,   453,   345,   264,
       9,   479,   286,   481,   355,   356,   287,    61,   349,   614,
     519,     9,   353,    10,    11,   163,   482,     9,   354,    64,
     165,    10,    11,   358,   257,   360,   397,   243,   214,   477,
     216,   218,   266,  -171,   489,   127,   506,   243,   408,   111,
     529,   414,   415,   117,   331,   332,   143,   111,   362,    10,
      11,   111,   147,   364,   146,  -169,   150,  -171,   228,   368,
      10,    11,  -173,   231,   372,  -171,    10,    11,   374,   164,
     228,   150,   480,   378,   380,   231,   384,   483,   382,  -169,
     228,   386,   432,   513,   228,   231,  -173,  -169,   392,   231,
     394,    79,    79,   399,  -173,   401,   479,   521,   481,   349,
     541,   405,   548,     9,   228,   228,   235,   453,   612,   231,
     231,   482,   479,   453,   481,   408,   561,   349,   413,   117,
     117,   117,   416,   419,   444,   163,    79,   482,   449,   445,
     165,   243,    94,   117,   446,     9,   460,   164,   172,   447,
      95,   494,   117,   448,    96,     9,   458,   463,   466,   508,
     117,   467,    10,    11,    97,    98,   468,   117,   469,   470,
     485,   226,   514,   490,   379,   493,   503,   480,   510,     9,
     505,   273,   483,   512,   228,   520,   519,   530,   526,   231,
     117,   277,   524,   480,    10,    11,   531,    99,   483,   532,
       9,   527,   254,   163,    10,    11,   534,   340,   165,   479,
     553,   481,   555,   210,   210,   288,   210,   210,   152,   153,
     154,   155,   156,   157,   482,   166,   167,   228,    10,    11,
     556,   462,   231,   563,   243,   317,   537,   255,   569,   117,
      79,   117,   544,   547,   529,   552,   567,   150,   572,    10,
      11,    10,    11,   574,   557,   576,   559,   117,   577,   117,
     349,   580,   541,   590,   582,   117,   548,   453,   591,   593,
     594,   349,   349,   164,  -151,   595,   479,   228,   481,  -152,
     480,   596,   231,   597,   584,   483,   177,   117,   601,   598,
     600,   482,   602,   604,   606,   615,   185,   613,   137,   117,
     189,   626,   117,   117,   628,   194,   195,   196,   197,   629,
     160,   639,   437,   169,   644,   645,   646,   647,   106,    66,
     640,   152,   156,   627,   579,   272,   487,   641,   588,   163,
     344,   370,   404,   123,   165,   371,   344,   344,   504,   284,
      37,   352,    91,   492,   475,   599,   442,   480,   573,   112,
     538,   334,   483,   603,    47,    48,     9,   443,   562,   635,
     335,    51,     0,   632,   558,   336,   337,   338,   224,   351,
       0,     0,   339,     0,     0,     0,   117,     0,     0,   340,
       0,     0,     0,     0,     0,   114,     0,     0,   117,     0,
     117,   225,     0,     0,   544,   633,   341,   280,   117,     0,
       0,     0,   117,    64,     0,    10,    11,   281,   342,     0,
       0,     0,     0,     0,   343,     0,     0,    11,     0,     0,
       0,     0,   117,   117,   292,   293,   294,   295,     0,   296,
     297,   298,     0,   299,   300,   301,   302,   303,   304,   305,
     306,   307,   308,   309,   310,   311,   312,     0,   160,     0,
     319,     0,   322,   344,     0,     0,   137,   137,   333,   177,
     344,     0,     0,     0,     0,     0,     0,     0,   344,   185,
     175,  -269,     0,   189,   190,   191,   192,   193,   194,   195,
     196,   197,     0,     0,     0,   117,     0,     0,     0,     0,
       0,     0,   117,     0,     0,     0,     0,     0,     0,   176,
     177,   117,   178,   179,   180,   181,   182,     0,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,     0,     0,     0,     0,     0,     0,   137,
       0,  -269,     0,     0,     0,   117,     0,     0,     0,   137,
       0,  -269,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   344,     0,
       0,     0,   421,     0,   542,   344,   160,   344,     0,     0,
       0,     0,   421,     0,    -2,    34,   344,    35,   344,     0,
      36,     0,    37,    38,    39,   117,     0,    40,   117,    41,
      42,    43,    44,    45,    46,     0,    47,    48,     9,     0,
       0,    49,    50,    51,    52,    53,    54,     0,     0,     0,
      55,     0,     0,     0,     0,   137,   137,     0,     0,     0,
       0,     0,     0,    56,    57,     0,    58,    59,     0,     0,
      60,     0,     0,    61,     0,     0,   -24,     0,     0,     0,
       0,     0,     0,    62,    63,    64,     0,    10,    11,     0,
       0,     0,     0,     0,   137,   117,     0,     0,    35,     0,
       0,     0,     0,    37,     0,     0,   168,   344,   137,     0,
       0,     0,   112,     0,     0,   344,   160,    47,    48,     9,
       0,   169,   344,     0,    51,     0,     0,     0,     0,     0,
       0,    55,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    56,    57,     0,    58,    59,     0,
       0,    60,     0,   344,    61,     0,   542,   344,     0,     0,
       0,   565,   566,     0,    62,    63,    64,     0,    10,    11,
       0,   324,     0,    35,     0,     0,    36,  -244,    37,    38,
      39,     0,  -244,    40,   160,    41,    42,   112,    44,    45,
      46,     0,    47,    48,     9,     0,     0,    49,    50,    51,
      52,    53,    54,   344,   421,   344,    55,     0,     0,     0,
       0,   421,   586,   421,     0,     0,     0,     0,     0,    56,
      57,     0,    58,    59,     0,     0,    60,     0,     0,    61,
       0,     0,  -244,     0,     0,     0,     0,   325,  -244,    62,
      63,    64,     0,    10,    11,   324,     0,    35,     0,     0,
      36,     0,    37,    38,    39,     0,     0,    40,     0,    41,
      42,   112,    44,    45,    46,     0,    47,    48,     9,     0,
       0,    49,    50,    51,    52,    53,    54,     0,     0,     0,
      55,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    56,    57,     0,    58,    59,     0,     0,
      60,     0,     0,    61,     0,     0,  -244,   642,   643,     0,
     160,   325,  -244,    62,    63,    64,    35,    10,    11,     0,
       0,    37,     0,     0,     0,     0,     0,     0,     0,     0,
     112,     0,   334,     0,     0,    47,    48,     9,     0,     0,
       0,   335,    51,     0,     0,     0,   336,   337,   338,   158,
       0,     0,     0,   339,     0,     0,     0,     0,     0,     0,
     340,     0,    56,    57,     0,    58,   159,     0,     0,    60,
       0,    35,    61,   314,     0,     0,    37,   341,     0,     0,
       0,     0,    62,    63,    64,   112,    10,    11,     0,     0,
      47,    48,     9,     0,     0,   343,     0,    51,    11,     0,
       0,     0,     0,     0,    55,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    56,    57,     0,
      58,    59,     0,     0,    60,     0,    35,    61,     0,     0,
       0,    37,     0,     0,     0,   420,     0,    62,    63,    64,
     112,    10,    11,     0,     0,    47,    48,     9,     0,     0,
       0,     0,    51,     0,   429,     0,     0,     0,     0,   158,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    56,    57,     0,    58,   159,     0,     0,    60,
       0,    35,    61,     0,     0,     0,    37,     0,     0,     0,
       0,     0,    62,    63,    64,   112,    10,    11,     0,     0,
      47,    48,     9,     0,   472,     0,     0,    51,     0,     0,
       0,     0,     0,     0,    55,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    56,    57,     0,
      58,    59,     0,     0,    60,     0,    35,    61,     0,     0,
       0,    37,     0,     0,     0,     0,     0,    62,    63,    64,
     112,    10,    11,     0,     0,    47,    48,     9,     0,   473,
       0,     0,    51,     0,     0,     0,     0,     0,     0,    55,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    35,
       0,     0,    56,    57,    37,    58,    59,     0,     0,    60,
       0,     0,    61,   112,     0,     0,     0,     0,    47,    48,
       9,     0,    62,    63,    64,    51,    10,    11,     0,     0,
       0,     0,    55,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    35,     0,     0,    56,    57,    37,    58,    59,
       0,     0,    60,     0,     0,    61,   112,     0,     0,     0,
       0,    47,    48,     9,     0,    62,    63,    64,    51,    10,
      11,     0,     0,     0,     0,   158,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    35,     0,     0,    56,    57,
     283,    58,   159,     0,     0,    60,     0,     0,    61,   112,
       0,     0,     0,     0,    47,    48,     9,     0,    62,    63,
      64,    51,    10,    11,     0,     0,     0,     0,    55,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    56,    57,    37,    58,    59,     0,     0,    60,     0,
       0,    61,   112,     0,     0,     0,     0,    47,    48,     9,
       0,    62,    63,    64,    51,    10,    11,     0,    37,     0,
       0,   224,     0,     0,     0,     0,    37,   112,     0,   241,
       0,     0,    47,    48,     9,   112,     0,     0,   114,    51,
      47,    48,     9,     0,   225,     0,   224,    51,     0,     0,
     290,     0,    37,     0,   224,     0,    64,     0,    10,    11,
     281,   112,     0,   114,     0,     0,    47,    48,     9,   225,
       0,   114,     0,    51,     0,     0,     0,   225,     0,    37,
     113,    64,     0,    10,    11,   396,     0,     0,   112,    64,
       0,    10,    11,    47,    48,     9,     0,   114,     0,     0,
      51,     0,     0,   115,     0,    37,     0,   224,     0,     0,
       0,     0,     0,    37,   112,    64,     0,    10,    11,    47,
      48,     9,   112,     0,   114,     0,    51,    47,    48,     9,
     225,     0,     0,   406,    51,     0,     0,     0,   283,     0,
       0,   224,    64,     0,    10,    11,   334,   112,     0,     0,
     114,     0,    47,    48,     9,   335,   407,     0,   114,    51,
     336,   337,   338,     0,   476,     0,   224,   339,    64,     0,
      10,    11,   334,     0,   439,   459,    64,     0,    10,    11,
       0,   335,     0,   114,     0,     0,   336,   337,   338,   225,
       0,   341,     0,   339,     0,     0,     0,   440,     0,   334,
     340,    64,     0,    10,    11,     0,     0,     0,   335,   343,
       0,     0,    11,   336,   337,   540,     0,   341,     0,     0,
     339,     0,     0,     0,     0,   334,     0,   340,     0,     0,
       0,     0,     0,     0,   335,   343,     0,     0,    11,   336,
     337,   338,     0,     0,   341,     0,   339,     0,     0,     0,
       0,     0,     0,   340,     0,     0,     0,     0,     0,     0,
       0,     0,   343,   177,    10,    11,     0,   180,   181,   182,
     341,     0,   184,   185,   186,   187,   607,   189,   190,   191,
     192,   193,   194,   195,   196,   197,     0,     0,   343,   176,
     177,    11,   178,     0,   180,   181,   182,     0,     0,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,     0,     0,     0,     0,     0,     0,   176,
     177,     0,   178,     0,   180,   181,   182,     0,   431,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   176,   177,     0,   178,     0,   180,   181,
     182,     0,   523,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   176,   177,     0,
     178,     0,   180,   181,   182,     0,   649,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   176,   177,     0,   178,     0,   180,   181,   182,     0,
     650,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,     0,   176,   177,   434,   178,
       0,   180,   181,   182,     0,     0,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     176,   177,     0,     0,     0,   180,   181,   182,     0,     0,
     184,   185,   186,   187,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   197,   176,   177,     0,     0,     0,   180,
     181,   182,     0,     0,   184,   185,   186,   187,     0,   189,
     190,   191,   192,   193,   194,   195,   196,   197
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-549))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       5,    67,   250,   223,    37,   126,    37,   126,   143,   135,
     201,   132,   204,   147,    61,    20,    36,    22,   485,    39,
      31,   142,    28,    28,   321,    45,   452,   316,   323,   249,
     258,    36,     3,     5,    39,    20,   263,   264,    43,   267,
      45,   444,    67,    61,    11,    35,   594,   275,    53,    54,
       5,   279,     1,     3,    24,     3,    35,     3,    24,    51,
      59,   289,    67,     0,    63,    62,    59,    49,    24,    59,
       3,    68,    71,     5,    24,    37,    24,    67,    24,     3,
      37,    74,    67,    25,   389,     7,    35,    53,    67,    59,
      12,    83,   397,    65,    66,   106,   644,    89,   646,    61,
      24,    72,    50,    75,    61,   572,   172,     7,    75,    59,
      65,    59,    12,    63,    60,   135,    25,    63,    67,    62,
      75,   141,   127,    73,    74,    73,    74,    73,    74,   255,
     135,     3,    21,    65,    24,   424,   141,   426,   143,    63,
     437,   159,   147,    75,   126,    60,    68,   172,    62,    73,
      74,   113,   114,   115,    68,   460,   113,   114,   115,    24,
      75,   388,   382,   390,     3,   127,     1,   172,    68,    59,
     127,    62,   639,    24,   136,     7,   602,    68,    68,   136,
      12,   199,   144,    73,    74,    24,    63,   144,   436,   151,
      62,    63,   595,    62,   151,   200,    68,   159,   503,    68,
      35,   206,   159,   606,   607,   523,   524,   212,   113,   114,
     115,    50,   174,    62,   220,   220,    35,   174,   223,    62,
     515,   413,     3,   471,    59,   207,   208,   522,   420,   234,
      62,   136,    67,     3,    73,    74,    68,   199,    71,   144,
      59,   432,   199,    67,   249,   250,   151,   368,    67,   368,
     283,    24,   283,    24,   559,    62,   476,   378,   384,   378,
      66,   223,    66,   225,   399,    59,   223,   401,   225,   174,
     317,   576,   577,     7,    59,    59,   281,    74,    12,   241,
      24,   243,    53,    35,   241,    40,   243,   249,    59,    44,
     518,     3,   249,    59,   599,    64,     8,    68,   280,   317,
      73,    74,    73,    74,    24,    17,    62,    59,   290,   271,
      22,    23,    24,    63,   271,    67,   321,    29,    40,    59,
     225,   283,    44,    60,   286,   287,   283,   575,    62,   286,
     287,    67,    75,    53,    68,    35,   241,   342,   211,    59,
      24,   374,    72,   374,   217,   218,    72,    59,   353,   569,
     416,    24,    59,    73,    74,   317,   374,    24,     3,    71,
     317,    73,    74,    60,   384,     8,   271,   372,    95,   374,
      97,    98,   392,    35,   379,    59,   396,   382,   283,   384,
     427,   286,   287,    37,   366,   367,    59,   392,    60,    73,
      74,   396,    59,    35,   399,    35,   401,    59,   360,    62,
      73,    74,    35,   360,    75,    67,    73,    74,    60,   427,
     372,   416,   374,    62,    24,   372,    62,   374,    59,    59,
     382,    72,    66,   405,   386,   382,    59,    67,    62,   386,
       3,   436,   437,    62,    67,    62,   469,   419,   469,   444,
     445,    65,   447,    24,   406,   407,   567,   452,   567,   406,
     407,   469,   485,   458,   485,   360,   461,   462,    67,   113,
     114,   115,    62,    65,    59,   427,   471,   485,   341,    67,
     427,   476,     9,   127,    71,    24,   349,   495,    59,    67,
      17,   386,   136,     8,    21,    24,    65,    62,    62,    35,
     144,    62,    73,    74,    31,    32,    62,   151,    60,    60,
      60,   406,   407,    60,    63,    68,    60,   469,    60,    24,
      68,    60,   469,    60,   476,    60,   582,    60,    68,   476,
     174,    60,    75,   485,    73,    74,    60,    64,   485,    60,
      24,    75,    24,   495,    73,    74,    68,    36,   495,   572,
       3,   572,    62,    94,    95,    60,    97,    98,    55,    56,
      57,    58,    59,    60,   572,    62,    63,   519,    73,    74,
      72,    75,   519,    60,   569,    59,   439,    59,    59,   223,
     575,   225,   445,   446,   621,   448,    62,   582,    60,    73,
      74,    73,    74,    66,   457,    60,   459,   241,    60,   243,
     595,    60,   597,    60,    62,   249,   601,   602,    72,    60,
      60,   606,   607,   621,    59,    59,   639,   569,   639,    59,
     572,    68,   569,    62,   519,   572,    34,   271,    62,    72,
      68,   639,    49,    62,    59,    14,    44,    60,    49,   283,
      48,    68,   286,   287,    60,    53,    54,    55,    56,    68,
      61,    60,    62,    64,    60,    60,    60,    60,    31,    22,
     615,   158,   159,   582,   508,   142,   377,   615,   524,   621,
     211,   243,   281,    39,   621,   243,   217,   218,   392,   159,
       8,   213,    22,   384,   372,   548,   334,   639,   495,    17,
     440,     8,   639,   556,    22,    23,    24,   334,   462,   601,
      17,    29,    -1,   597,   458,    22,    23,    24,    36,   212,
      -1,    -1,    29,    -1,    -1,    -1,   360,    -1,    -1,    36,
      -1,    -1,    -1,    -1,    -1,    53,    -1,    -1,   372,    -1,
     374,    59,    -1,    -1,   597,   598,    53,    65,   382,    -1,
      -1,    -1,   386,    71,    -1,    73,    74,    75,    65,    -1,
      -1,    -1,    -1,    -1,    71,    -1,    -1,    74,    -1,    -1,
      -1,    -1,   406,   407,   175,   176,   177,   178,    -1,   180,
     181,   182,    -1,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,    -1,   199,    -1,
     201,    -1,   203,   334,    -1,    -1,   207,   208,   209,    34,
     341,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   349,    44,
       4,     5,    -1,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    -1,    -1,    -1,   469,    -1,    -1,    -1,    -1,
      -1,    -1,   476,    -1,    -1,    -1,    -1,    -1,    -1,    33,
      34,   485,    36,    37,    38,    39,    40,    -1,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    -1,    -1,    -1,    -1,    -1,    -1,   280,
      -1,    65,    -1,    -1,    -1,   519,    -1,    -1,    -1,   290,
      -1,    75,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   439,    -1,
      -1,    -1,   313,    -1,   445,   446,   317,   448,    -1,    -1,
      -1,    -1,   323,    -1,     0,     1,   457,     3,   459,    -1,
       6,    -1,     8,     9,    10,   569,    -1,    13,   572,    15,
      16,    17,    18,    19,    20,    -1,    22,    23,    24,    -1,
      -1,    27,    28,    29,    30,    31,    32,    -1,    -1,    -1,
      36,    -1,    -1,    -1,    -1,   366,   367,    -1,    -1,    -1,
      -1,    -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,
      56,    -1,    -1,    59,    -1,    -1,    62,    -1,    -1,    -1,
      -1,    -1,    -1,    69,    70,    71,    -1,    73,    74,    -1,
      -1,    -1,    -1,    -1,   405,   639,    -1,    -1,     3,    -1,
      -1,    -1,    -1,     8,    -1,    -1,    11,   548,   419,    -1,
      -1,    -1,    17,    -1,    -1,   556,   427,    22,    23,    24,
      -1,   432,   563,    -1,    29,    -1,    -1,    -1,    -1,    -1,
      -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    50,    -1,    52,    53,    -1,
      -1,    56,    -1,   594,    59,    -1,   597,   598,    -1,    -1,
      -1,   472,   473,    -1,    69,    70,    71,    -1,    73,    74,
      -1,     1,    -1,     3,    -1,    -1,     6,     7,     8,     9,
      10,    -1,    12,    13,   495,    15,    16,    17,    18,    19,
      20,    -1,    22,    23,    24,    -1,    -1,    27,    28,    29,
      30,    31,    32,   644,   515,   646,    36,    -1,    -1,    -1,
      -1,   522,   523,   524,    -1,    -1,    -1,    -1,    -1,    49,
      50,    -1,    52,    53,    -1,    -1,    56,    -1,    -1,    59,
      -1,    -1,    62,    -1,    -1,    -1,    -1,    67,    68,    69,
      70,    71,    -1,    73,    74,     1,    -1,     3,    -1,    -1,
       6,    -1,     8,     9,    10,    -1,    -1,    13,    -1,    15,
      16,    17,    18,    19,    20,    -1,    22,    23,    24,    -1,
      -1,    27,    28,    29,    30,    31,    32,    -1,    -1,    -1,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,
      56,    -1,    -1,    59,    -1,    -1,    62,   618,   619,    -1,
     621,    67,    68,    69,    70,    71,     3,    73,    74,    -1,
      -1,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      17,    -1,     8,    -1,    -1,    22,    23,    24,    -1,    -1,
      -1,    17,    29,    -1,    -1,    -1,    22,    23,    24,    36,
      -1,    -1,    -1,    29,    -1,    -1,    -1,    -1,    -1,    -1,
      36,    -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,     3,    59,    60,    -1,    -1,     8,    53,    -1,    -1,
      -1,    -1,    69,    70,    71,    17,    73,    74,    -1,    -1,
      22,    23,    24,    -1,    -1,    71,    -1,    29,    74,    -1,
      -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,
      52,    53,    -1,    -1,    56,    -1,     3,    59,    -1,    -1,
      -1,     8,    -1,    -1,    -1,    67,    -1,    69,    70,    71,
      17,    73,    74,    -1,    -1,    22,    23,    24,    -1,    -1,
      -1,    -1,    29,    -1,    31,    -1,    -1,    -1,    -1,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,     3,    59,    -1,    -1,    -1,     8,    -1,    -1,    -1,
      -1,    -1,    69,    70,    71,    17,    73,    74,    -1,    -1,
      22,    23,    24,    -1,    26,    -1,    -1,    29,    -1,    -1,
      -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,
      52,    53,    -1,    -1,    56,    -1,     3,    59,    -1,    -1,
      -1,     8,    -1,    -1,    -1,    -1,    -1,    69,    70,    71,
      17,    73,    74,    -1,    -1,    22,    23,    24,    -1,    26,
      -1,    -1,    29,    -1,    -1,    -1,    -1,    -1,    -1,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,
      -1,    -1,    49,    50,     8,    52,    53,    -1,    -1,    56,
      -1,    -1,    59,    17,    -1,    -1,    -1,    -1,    22,    23,
      24,    -1,    69,    70,    71,    29,    73,    74,    -1,    -1,
      -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     3,    -1,    -1,    49,    50,     8,    52,    53,
      -1,    -1,    56,    -1,    -1,    59,    17,    -1,    -1,    -1,
      -1,    22,    23,    24,    -1,    69,    70,    71,    29,    73,
      74,    -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    49,    50,
       8,    52,    53,    -1,    -1,    56,    -1,    -1,    59,    17,
      -1,    -1,    -1,    -1,    22,    23,    24,    -1,    69,    70,
      71,    29,    73,    74,    -1,    -1,    -1,    -1,    36,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    49,    50,     8,    52,    53,    -1,    -1,    56,    -1,
      -1,    59,    17,    -1,    -1,    -1,    -1,    22,    23,    24,
      -1,    69,    70,    71,    29,    73,    74,    -1,     8,    -1,
      -1,    36,    -1,    -1,    -1,    -1,     8,    17,    -1,    11,
      -1,    -1,    22,    23,    24,    17,    -1,    -1,    53,    29,
      22,    23,    24,    -1,    59,    -1,    36,    29,    -1,    -1,
      65,    -1,     8,    -1,    36,    -1,    71,    -1,    73,    74,
      75,    17,    -1,    53,    -1,    -1,    22,    23,    24,    59,
      -1,    53,    -1,    29,    -1,    -1,    -1,    59,    -1,     8,
      36,    71,    -1,    73,    74,    75,    -1,    -1,    17,    71,
      -1,    73,    74,    22,    23,    24,    -1,    53,    -1,    -1,
      29,    -1,    -1,    59,    -1,     8,    -1,    36,    -1,    -1,
      -1,    -1,    -1,     8,    17,    71,    -1,    73,    74,    22,
      23,    24,    17,    -1,    53,    -1,    29,    22,    23,    24,
      59,    -1,    -1,    36,    29,    -1,    -1,    -1,     8,    -1,
      -1,    36,    71,    -1,    73,    74,     8,    17,    -1,    -1,
      53,    -1,    22,    23,    24,    17,    59,    -1,    53,    29,
      22,    23,    24,    -1,    59,    -1,    36,    29,    71,    -1,
      73,    74,     8,    -1,    36,    11,    71,    -1,    73,    74,
      -1,    17,    -1,    53,    -1,    -1,    22,    23,    24,    59,
      -1,    53,    -1,    29,    -1,    -1,    -1,    59,    -1,     8,
      36,    71,    -1,    73,    74,    -1,    -1,    -1,    17,    71,
      -1,    -1,    74,    22,    23,    24,    -1,    53,    -1,    -1,
      29,    -1,    -1,    -1,    -1,     8,    -1,    36,    -1,    -1,
      -1,    -1,    -1,    -1,    17,    71,    -1,    -1,    74,    22,
      23,    24,    -1,    -1,    53,    -1,    29,    -1,    -1,    -1,
      -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    71,    34,    73,    74,    -1,    38,    39,    40,
      53,    -1,    43,    44,    45,    46,    59,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    -1,    -1,    71,    33,
      34,    74,    36,    -1,    38,    39,    40,    -1,    -1,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    -1,    -1,    -1,    -1,    -1,    -1,    33,
      34,    -1,    36,    -1,    38,    39,    40,    -1,    72,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    33,    34,    -1,    36,    -1,    38,    39,
      40,    -1,    66,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    33,    34,    -1,
      36,    -1,    38,    39,    40,    -1,    66,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    33,    34,    -1,    36,    -1,    38,    39,    40,    -1,
      66,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    -1,    33,    34,    60,    36,
      -1,    38,    39,    40,    -1,    -1,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      33,    34,    -1,    -1,    -1,    38,    39,    40,    -1,    -1,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    33,    34,    -1,    -1,    -1,    38,
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
      83,    85,   137,    62,     1,     3,     6,     8,     9,    10,
      13,    15,    16,    17,    18,    19,    20,    22,    23,    27,
      28,    29,    30,    31,    32,    36,    49,    50,    52,    53,
      56,    59,    69,    70,    71,    90,    91,    92,    98,   110,
     113,   118,   121,   123,   124,   125,   126,   130,   134,   137,
     139,   140,   145,   146,   149,   152,   153,   154,   157,   160,
     161,   177,   182,    62,     9,    17,    21,    31,    32,    64,
     195,    24,    60,    83,    84,     3,    86,    88,     3,   134,
     136,   137,    17,    36,    53,    59,   137,   139,   144,   148,
     149,   150,   157,   136,   125,   130,   111,    59,   137,   155,
     125,   134,   114,    35,    67,   133,    71,   123,   182,   189,
     122,   133,   119,    59,    96,    97,   137,    59,    93,   135,
     137,   181,   124,   124,   124,   124,   124,   124,    36,    53,
     123,   131,   143,   149,   151,   157,   124,   124,    11,   123,
     188,    62,    59,    94,   181,     4,    33,    34,    36,    37,
      38,    39,    40,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    67,    59,
      63,    71,    66,    59,   133,     1,   133,     5,    65,    75,
     138,   196,    59,   156,   196,    24,   196,   197,   196,    64,
      62,   186,    88,    59,    36,    59,   142,   148,   149,   150,
     151,   157,   142,   142,    63,    98,   107,   108,   109,   182,
     190,    11,   132,   137,   141,   142,   173,   174,   175,    59,
      67,   158,   112,   190,    24,    59,    68,   134,   167,   169,
     171,   142,    35,    53,    59,    68,   134,   166,   168,   169,
     170,   180,   112,    60,    97,   165,   142,    60,    93,   163,
      65,    75,   142,     8,   143,    60,    72,    72,    60,    94,
      65,   142,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   123,   123,   127,    60,   131,   183,    59,   137,   123,
     188,   178,   123,   127,     1,    67,    91,   100,   176,   177,
     179,   182,   182,   123,     8,    17,    22,    23,    24,    29,
      36,    53,    65,    71,   138,   198,   200,   201,   202,   137,
     203,   211,   158,    59,     3,   198,   198,    83,    60,   175,
       8,   142,    60,   137,    35,   105,     5,    65,    62,   142,
     132,   141,    75,   187,    60,   175,   179,   115,    62,    63,
      24,   169,    59,   172,    62,   186,    72,   104,    59,   170,
      53,   170,    62,   186,     3,   194,    75,   142,   120,    62,
     186,    62,   186,   182,   135,    65,    36,    59,   142,   148,
     149,   150,   157,    67,   142,   142,    62,   186,   182,    65,
      67,   123,   128,   129,   184,   185,    11,    75,   187,    31,
     131,    72,    66,   176,    60,   185,   101,    62,    68,    36,
      59,   199,   200,   202,    59,    67,    71,    67,     8,   198,
       3,    50,    59,   137,   208,   209,     3,    72,    65,    11,
     198,    60,    75,    62,   191,   211,    62,    62,    62,    60,
      60,   106,    26,    26,   190,   173,    59,   137,   147,   148,
     149,   150,   151,   157,   159,    60,    68,   105,   190,   137,
      60,   175,   171,    68,   142,     7,    12,    68,    99,   102,
     170,   194,   170,    60,   168,    68,   134,   194,    35,    97,
      60,    93,    60,   182,   142,   127,    94,    95,   164,   181,
      60,   182,   127,    66,    75,   187,    68,    75,   187,   131,
      60,    60,    60,   188,    68,   179,   176,   198,   201,   191,
      24,   137,   138,   193,   198,   205,   213,   198,   137,   192,
     204,   212,   198,     3,   208,    62,    72,   198,   209,   198,
     194,   137,   203,    60,   179,   123,   123,    62,   175,    59,
     159,   116,    60,   183,    66,   103,    60,    60,   194,   104,
      60,   185,    62,   186,   142,   185,   123,   129,   128,   129,
      60,    72,    68,    60,    60,    59,    68,    62,    72,   198,
      68,    62,    49,   198,    62,   194,    59,    59,   198,   206,
     207,    68,   190,    60,   175,    14,   117,   159,     5,    65,
      66,    75,   179,   194,   194,    68,    68,    95,    60,    68,
     206,   191,   205,   198,   194,   204,   208,   191,   191,    60,
     100,   113,   123,   123,    60,    60,    60,    60,   159,    66,
      66,   206,   206
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
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL		goto yyerrlab
#if defined YYFAIL
  /* This is here to suppress warnings from the GCC cpp's
     -Wunused-macros.  Normally we don't worry about that warning, but
     some users do, and we want to make it easy for users to remove
     YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
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


/* This macro is provided for backward compatibility. */

#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
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
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
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
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
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

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = 0;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                yysize1 = yysize + yytnamerr (0, yytname[yyx]);
                if (! (yysize <= yysize1
                       && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                  return 2;
                yysize = yysize1;
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  yysize1 = yysize + yystrlen (yyformat);
  if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
    return 2;
  yysize = yysize1;

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
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


/* The lookahead symbol.  */
int yychar, yystate;

/* The semantic value of the lookahead symbol.  */
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
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

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
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
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

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
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
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
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

/* Line 1806 of yacc.c  */
#line 128 "go.y"
    {
		xtop = concat(xtop, (yyvsp[(4) - (4)].list));
	}
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 134 "go.y"
    {
		prevlineno = lineno;
		yyerror("package statement must be first");
		flusherrors();
		mkpackage("main");
	}
    break;

  case 4:

/* Line 1806 of yacc.c  */
#line 141 "go.y"
    {
		mkpackage((yyvsp[(2) - (3)].sym)->name);
	}
    break;

  case 5:

/* Line 1806 of yacc.c  */
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

/* Line 1806 of yacc.c  */
#line 162 "go.y"
    {
		importpkg = nil;
	}
    break;

  case 12:

/* Line 1806 of yacc.c  */
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

/* Line 1806 of yacc.c  */
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

/* Line 1806 of yacc.c  */
#line 224 "go.y"
    {
		// import with original name
		(yyval.i) = parserline();
		importmyname = S;
		importfile(&(yyvsp[(1) - (1)].val), (yyval.i));
	}
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 231 "go.y"
    {
		// import with given name
		(yyval.i) = parserline();
		importmyname = (yyvsp[(1) - (2)].sym);
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 238 "go.y"
    {
		// import into my name space
		(yyval.i) = parserline();
		importmyname = lookup(".");
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 247 "go.y"
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

  case 21:

/* Line 1806 of yacc.c  */
#line 261 "go.y"
    {
		if(strcmp((yyvsp[(1) - (1)].sym)->name, "safe") == 0)
			curio.importsafe = 1;
	}
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 267 "go.y"
    {
		defercheckwidth();
	}
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 271 "go.y"
    {
		resumecheckwidth();
		unimportfile();
	}
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 280 "go.y"
    {
		yyerror("empty top-level declaration");
		(yyval.list) = nil;
	}
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 286 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 290 "go.y"
    {
		yyerror("non-declaration statement outside function body");
		(yyval.list) = nil;
	}
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 295 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 301 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
	}
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 305 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 309 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 313 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 319 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 325 "go.y"
    {
		(yyval.list) = concat((yyvsp[(3) - (7)].list), (yyvsp[(5) - (7)].list));
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 331 "go.y"
    {
		(yyval.list) = nil;
		iota = -100000;
	}
    break;

  case 36:

/* Line 1806 of yacc.c  */
#line 336 "go.y"
    {
		(yyval.list) = list1((yyvsp[(2) - (2)].node));
	}
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 340 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 344 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 39:

/* Line 1806 of yacc.c  */
#line 350 "go.y"
    {
		iota = 0;
	}
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 356 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 360 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 364 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (3)].list), nil, (yyvsp[(3) - (3)].list));
	}
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 370 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 374 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (3)].list), N, (yyvsp[(3) - (3)].list));
	}
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 381 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 385 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (1)].list), N, nil);
	}
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 391 "go.y"
    {
		// different from dclname because the name
		// becomes visible right here, not at the end
		// of the declaration.
		(yyval.node) = typedcl0((yyvsp[(1) - (1)].sym));
	}
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 400 "go.y"
    {
		(yyval.node) = typedcl1((yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node), 1);
	}
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 406 "go.y"
    {
		(yyval.node) = (yyvsp[(1) - (1)].node);
	}
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 410 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
		(yyval.node)->etype = (yyvsp[(2) - (3)].i);			// rathole to pass opcode
	}
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 415 "go.y"
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

/* Line 1806 of yacc.c  */
#line 427 "go.y"
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

/* Line 1806 of yacc.c  */
#line 443 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OADD;
	}
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 448 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OSUB;
	}
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 455 "go.y"
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

/* Line 1806 of yacc.c  */
#line 475 "go.y"
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

/* Line 1806 of yacc.c  */
#line 493 "go.y"
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

/* Line 1806 of yacc.c  */
#line 502 "go.y"
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

/* Line 1806 of yacc.c  */
#line 520 "go.y"
    {
		markdcl();
	}
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 524 "go.y"
    {
		(yyval.node) = liststmt((yyvsp[(3) - (4)].list));
		popdcl();
	}
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 531 "go.y"
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

/* Line 1806 of yacc.c  */
#line 541 "go.y"
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

/* Line 1806 of yacc.c  */
#line 561 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 565 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node));
	}
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 571 "go.y"
    {
		markdcl();
	}
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 575 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (4)].list);
		popdcl();
	}
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 582 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->etype = 0;	// := flag
	}
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 588 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->colas = 1;
		colasdefn((yyvsp[(1) - (4)].list), (yyval.node));
	}
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 597 "go.y"
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

/* Line 1806 of yacc.c  */
#line 608 "go.y"
    {
		// normal test
		(yyval.node) = nod(OFOR, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 73:

/* Line 1806 of yacc.c  */
#line 617 "go.y"
    {
		(yyval.node) = (yyvsp[(1) - (2)].node);
		(yyval.node)->nbody = concat((yyval.node)->nbody, (yyvsp[(2) - (2)].list));
	}
    break;

  case 74:

/* Line 1806 of yacc.c  */
#line 624 "go.y"
    {
		markdcl();
	}
    break;

  case 75:

/* Line 1806 of yacc.c  */
#line 628 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (3)].node);
		popdcl();
	}
    break;

  case 76:

/* Line 1806 of yacc.c  */
#line 635 "go.y"
    {
		// test
		(yyval.node) = nod(OIF, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 77:

/* Line 1806 of yacc.c  */
#line 641 "go.y"
    {
		// init ; test
		(yyval.node) = nod(OIF, N, N);
		if((yyvsp[(1) - (3)].node) != N)
			(yyval.node)->ninit = list1((yyvsp[(1) - (3)].node));
		(yyval.node)->ntest = (yyvsp[(3) - (3)].node);
	}
    break;

  case 78:

/* Line 1806 of yacc.c  */
#line 652 "go.y"
    {
		markdcl();
	}
    break;

  case 79:

/* Line 1806 of yacc.c  */
#line 656 "go.y"
    {
		if((yyvsp[(3) - (3)].node)->ntest == N)
			yyerror("missing condition in if statement");
	}
    break;

  case 80:

/* Line 1806 of yacc.c  */
#line 661 "go.y"
    {
		(yyvsp[(3) - (5)].node)->nbody = (yyvsp[(5) - (5)].list);
	}
    break;

  case 81:

/* Line 1806 of yacc.c  */
#line 665 "go.y"
    {
		popdcl();
		(yyval.node) = (yyvsp[(3) - (7)].node);
		if((yyvsp[(7) - (7)].node) != N)
			(yyval.node)->nelse = list1((yyvsp[(7) - (7)].node));
	}
    break;

  case 82:

/* Line 1806 of yacc.c  */
#line 673 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 677 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (2)].node);
	}
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 681 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (2)].node);
	}
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 687 "go.y"
    {
		markdcl();
	}
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 691 "go.y"
    {
		Node *n;
		n = (yyvsp[(3) - (3)].node)->ntest;
		if(n != N && n->op != OTYPESW)
			n = N;
		typesw = nod(OXXX, typesw, n);
	}
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 699 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (7)].node);
		(yyval.node)->op = OSWITCH;
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
		typesw = typesw->left;
		popdcl();
	}
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 709 "go.y"
    {
		typesw = nod(OXXX, typesw, N);
	}
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 713 "go.y"
    {
		(yyval.node) = nod(OSELECT, N, N);
		(yyval.node)->lineno = typesw->lineno;
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		typesw = typesw->left;
	}
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 726 "go.y"
    {
		(yyval.node) = nod(OOROR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 92:

/* Line 1806 of yacc.c  */
#line 730 "go.y"
    {
		(yyval.node) = nod(OANDAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 93:

/* Line 1806 of yacc.c  */
#line 734 "go.y"
    {
		(yyval.node) = nod(OEQ, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 738 "go.y"
    {
		(yyval.node) = nod(ONE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 742 "go.y"
    {
		(yyval.node) = nod(OLT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 746 "go.y"
    {
		(yyval.node) = nod(OLE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 750 "go.y"
    {
		(yyval.node) = nod(OGE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 754 "go.y"
    {
		(yyval.node) = nod(OGT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 758 "go.y"
    {
		(yyval.node) = nod(OADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 100:

/* Line 1806 of yacc.c  */
#line 762 "go.y"
    {
		(yyval.node) = nod(OSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 766 "go.y"
    {
		(yyval.node) = nod(OOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 102:

/* Line 1806 of yacc.c  */
#line 770 "go.y"
    {
		(yyval.node) = nod(OXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 103:

/* Line 1806 of yacc.c  */
#line 774 "go.y"
    {
		(yyval.node) = nod(OMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 104:

/* Line 1806 of yacc.c  */
#line 778 "go.y"
    {
		(yyval.node) = nod(ODIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 105:

/* Line 1806 of yacc.c  */
#line 782 "go.y"
    {
		(yyval.node) = nod(OMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 106:

/* Line 1806 of yacc.c  */
#line 786 "go.y"
    {
		(yyval.node) = nod(OAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 107:

/* Line 1806 of yacc.c  */
#line 790 "go.y"
    {
		(yyval.node) = nod(OANDNOT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 108:

/* Line 1806 of yacc.c  */
#line 794 "go.y"
    {
		(yyval.node) = nod(OLSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 109:

/* Line 1806 of yacc.c  */
#line 798 "go.y"
    {
		(yyval.node) = nod(ORSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 110:

/* Line 1806 of yacc.c  */
#line 803 "go.y"
    {
		(yyval.node) = nod(OSEND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 112:

/* Line 1806 of yacc.c  */
#line 810 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 113:

/* Line 1806 of yacc.c  */
#line 814 "go.y"
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

  case 114:

/* Line 1806 of yacc.c  */
#line 825 "go.y"
    {
		(yyval.node) = nod(OPLUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 115:

/* Line 1806 of yacc.c  */
#line 829 "go.y"
    {
		(yyval.node) = nod(OMINUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 116:

/* Line 1806 of yacc.c  */
#line 833 "go.y"
    {
		(yyval.node) = nod(ONOT, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 117:

/* Line 1806 of yacc.c  */
#line 837 "go.y"
    {
		yyerror("the bitwise complement operator is ^");
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 118:

/* Line 1806 of yacc.c  */
#line 842 "go.y"
    {
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 119:

/* Line 1806 of yacc.c  */
#line 846 "go.y"
    {
		(yyval.node) = nod(ORECV, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 120:

/* Line 1806 of yacc.c  */
#line 856 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (3)].node), N);
	}
    break;

  case 121:

/* Line 1806 of yacc.c  */
#line 860 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (5)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
	}
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 865 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (6)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (6)].list);
		(yyval.node)->isddd = 1;
	}
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 873 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 125:

/* Line 1806 of yacc.c  */
#line 878 "go.y"
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

  case 126:

/* Line 1806 of yacc.c  */
#line 889 "go.y"
    {
		(yyval.node) = nod(ODOTTYPE, (yyvsp[(1) - (5)].node), (yyvsp[(4) - (5)].node));
	}
    break;

  case 127:

/* Line 1806 of yacc.c  */
#line 893 "go.y"
    {
		(yyval.node) = nod(OTYPESW, N, (yyvsp[(1) - (5)].node));
	}
    break;

  case 128:

/* Line 1806 of yacc.c  */
#line 897 "go.y"
    {
		(yyval.node) = nod(OINDEX, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 129:

/* Line 1806 of yacc.c  */
#line 901 "go.y"
    {
		(yyval.node) = nod(OSLICE, (yyvsp[(1) - (6)].node), nod(OKEY, (yyvsp[(3) - (6)].node), (yyvsp[(5) - (6)].node)));
	}
    break;

  case 131:

/* Line 1806 of yacc.c  */
#line 906 "go.y"
    {
		// conversion
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (4)].node), N);
		(yyval.node)->list = list1((yyvsp[(3) - (4)].node));
	}
    break;

  case 132:

/* Line 1806 of yacc.c  */
#line 912 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 133:

/* Line 1806 of yacc.c  */
#line 919 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
	}
    break;

  case 134:

/* Line 1806 of yacc.c  */
#line 925 "go.y"
    {
		yyerror("cannot parenthesize type in composite literal");
		(yyval.node) = (yyvsp[(5) - (7)].node);
		(yyval.node)->right = (yyvsp[(2) - (7)].node);
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
	}
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 934 "go.y"
    {
		// composite expression.
		// make node early so we get the right line number.
		(yyval.node) = nod(OCOMPLIT, N, N);
	}
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 942 "go.y"
    {
		(yyval.node) = nod(OKEY, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 139:

/* Line 1806 of yacc.c  */
#line 949 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (4)].node);
		(yyval.node)->list = (yyvsp[(3) - (4)].list);
	}
    break;

  case 141:

/* Line 1806 of yacc.c  */
#line 957 "go.y"
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

  case 145:

/* Line 1806 of yacc.c  */
#line 982 "go.y"
    {
		(yyval.i) = LBODY;
	}
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 986 "go.y"
    {
		(yyval.i) = '{';
	}
    break;

  case 147:

/* Line 1806 of yacc.c  */
#line 997 "go.y"
    {
		if((yyvsp[(1) - (1)].sym) == S)
			(yyval.node) = N;
		else
			(yyval.node) = newname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 148:

/* Line 1806 of yacc.c  */
#line 1006 "go.y"
    {
		(yyval.node) = dclname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 149:

/* Line 1806 of yacc.c  */
#line 1011 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 151:

/* Line 1806 of yacc.c  */
#line 1018 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if(importpkg != nil && !exportname((yyvsp[(1) - (1)].sym)->name))
			(yyval.sym) = pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg);
	}
    break;

  case 153:

/* Line 1806 of yacc.c  */
#line 1026 "go.y"
    {
		(yyval.sym) = S;
	}
    break;

  case 154:

/* Line 1806 of yacc.c  */
#line 1032 "go.y"
    {
		if((yyvsp[(2) - (4)].val).u.sval->len == 0)
			(yyval.sym) = pkglookup((yyvsp[(4) - (4)].sym)->name, importpkg);
		else
			(yyval.sym) = pkglookup((yyvsp[(4) - (4)].sym)->name, mkpkg((yyvsp[(2) - (4)].val).u.sval));
	}
    break;

  case 155:

/* Line 1806 of yacc.c  */
#line 1041 "go.y"
    {
		(yyval.node) = oldname((yyvsp[(1) - (1)].sym));
		if((yyval.node)->pack != N)
			(yyval.node)->pack->used = 1;
	}
    break;

  case 157:

/* Line 1806 of yacc.c  */
#line 1061 "go.y"
    {
		yyerror("final argument in variadic function missing type");
		(yyval.node) = nod(ODDD, typenod(typ(TINTER)), N);
	}
    break;

  case 158:

/* Line 1806 of yacc.c  */
#line 1066 "go.y"
    {
		(yyval.node) = nod(ODDD, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 164:

/* Line 1806 of yacc.c  */
#line 1077 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 168:

/* Line 1806 of yacc.c  */
#line 1086 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 173:

/* Line 1806 of yacc.c  */
#line 1096 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 183:

/* Line 1806 of yacc.c  */
#line 1117 "go.y"
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

  case 184:

/* Line 1806 of yacc.c  */
#line 1130 "go.y"
    {
		(yyval.node) = nod(OTARRAY, (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].node));
	}
    break;

  case 185:

/* Line 1806 of yacc.c  */
#line 1134 "go.y"
    {
		// array literal of nelem
		(yyval.node) = nod(OTARRAY, nod(ODDD, N, N), (yyvsp[(4) - (4)].node));
	}
    break;

  case 186:

/* Line 1806 of yacc.c  */
#line 1139 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->etype = Cboth;
	}
    break;

  case 187:

/* Line 1806 of yacc.c  */
#line 1144 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Csend;
	}
    break;

  case 188:

/* Line 1806 of yacc.c  */
#line 1149 "go.y"
    {
		(yyval.node) = nod(OTMAP, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node));
	}
    break;

  case 191:

/* Line 1806 of yacc.c  */
#line 1157 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 192:

/* Line 1806 of yacc.c  */
#line 1163 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Crecv;
	}
    break;

  case 193:

/* Line 1806 of yacc.c  */
#line 1170 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 194:

/* Line 1806 of yacc.c  */
#line 1176 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 195:

/* Line 1806 of yacc.c  */
#line 1183 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 196:

/* Line 1806 of yacc.c  */
#line 1189 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 197:

/* Line 1806 of yacc.c  */
#line 1200 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
		if((yyval.node) == N)
			break;
		(yyval.node)->nbody = (yyvsp[(3) - (3)].list);
		(yyval.node)->endlineno = lineno;
		funcbody((yyval.node));
	}
    break;

  case 198:

/* Line 1806 of yacc.c  */
#line 1211 "go.y"
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

  case 199:

/* Line 1806 of yacc.c  */
#line 1240 "go.y"
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

  case 200:

/* Line 1806 of yacc.c  */
#line 1279 "go.y"
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

  case 201:

/* Line 1806 of yacc.c  */
#line 1302 "go.y"
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

  case 202:

/* Line 1806 of yacc.c  */
#line 1319 "go.y"
    {
		(yyvsp[(3) - (5)].list) = checkarglist((yyvsp[(3) - (5)].list), 1);
		(yyval.node) = nod(OTFUNC, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		(yyval.node)->rlist = (yyvsp[(5) - (5)].list);
	}
    break;

  case 203:

/* Line 1806 of yacc.c  */
#line 1327 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 204:

/* Line 1806 of yacc.c  */
#line 1331 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
		if((yyval.list) == nil)
			(yyval.list) = list1(nod(OEMPTY, N, N));
	}
    break;

  case 205:

/* Line 1806 of yacc.c  */
#line 1339 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 206:

/* Line 1806 of yacc.c  */
#line 1343 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, (yyvsp[(1) - (1)].node)));
	}
    break;

  case 207:

/* Line 1806 of yacc.c  */
#line 1347 "go.y"
    {
		(yyvsp[(2) - (3)].list) = checkarglist((yyvsp[(2) - (3)].list), 0);
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 208:

/* Line 1806 of yacc.c  */
#line 1354 "go.y"
    {
		closurehdr((yyvsp[(1) - (1)].node));
	}
    break;

  case 209:

/* Line 1806 of yacc.c  */
#line 1360 "go.y"
    {
		(yyval.node) = closurebody((yyvsp[(3) - (4)].list));
		fixlbrace((yyvsp[(2) - (4)].i));
	}
    break;

  case 210:

/* Line 1806 of yacc.c  */
#line 1365 "go.y"
    {
		(yyval.node) = closurebody(nil);
	}
    break;

  case 211:

/* Line 1806 of yacc.c  */
#line 1376 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 212:

/* Line 1806 of yacc.c  */
#line 1380 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(2) - (3)].list));
		if(nsyntaxerrors == 0)
			testdclstack();
	}
    break;

  case 214:

/* Line 1806 of yacc.c  */
#line 1389 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 216:

/* Line 1806 of yacc.c  */
#line 1396 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 217:

/* Line 1806 of yacc.c  */
#line 1402 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 218:

/* Line 1806 of yacc.c  */
#line 1406 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 220:

/* Line 1806 of yacc.c  */
#line 1413 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 221:

/* Line 1806 of yacc.c  */
#line 1419 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 222:

/* Line 1806 of yacc.c  */
#line 1423 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 223:

/* Line 1806 of yacc.c  */
#line 1429 "go.y"
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

  case 224:

/* Line 1806 of yacc.c  */
#line 1452 "go.y"
    {
		(yyvsp[(1) - (2)].node)->val = (yyvsp[(2) - (2)].val);
		(yyval.list) = list1((yyvsp[(1) - (2)].node));
	}
    break;

  case 225:

/* Line 1806 of yacc.c  */
#line 1457 "go.y"
    {
		(yyvsp[(2) - (4)].node)->val = (yyvsp[(4) - (4)].val);
		(yyval.list) = list1((yyvsp[(2) - (4)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 226:

/* Line 1806 of yacc.c  */
#line 1463 "go.y"
    {
		(yyvsp[(2) - (3)].node)->right = nod(OIND, (yyvsp[(2) - (3)].node)->right, N);
		(yyvsp[(2) - (3)].node)->val = (yyvsp[(3) - (3)].val);
		(yyval.list) = list1((yyvsp[(2) - (3)].node));
	}
    break;

  case 227:

/* Line 1806 of yacc.c  */
#line 1469 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 228:

/* Line 1806 of yacc.c  */
#line 1476 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 229:

/* Line 1806 of yacc.c  */
#line 1485 "go.y"
    {
		Node *n;

		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		n = oldname((yyvsp[(1) - (1)].sym));
		if(n->pack != N)
			n->pack->used = 1;
	}
    break;

  case 230:

/* Line 1806 of yacc.c  */
#line 1494 "go.y"
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

  case 231:

/* Line 1806 of yacc.c  */
#line 1509 "go.y"
    {
		(yyval.node) = embedded((yyvsp[(1) - (1)].sym));
	}
    break;

  case 232:

/* Line 1806 of yacc.c  */
#line 1515 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
		ifacedcl((yyval.node));
	}
    break;

  case 233:

/* Line 1806 of yacc.c  */
#line 1520 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(1) - (1)].sym)));
	}
    break;

  case 234:

/* Line 1806 of yacc.c  */
#line 1524 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(2) - (3)].sym)));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 235:

/* Line 1806 of yacc.c  */
#line 1531 "go.y"
    {
		// without func keyword
		(yyvsp[(2) - (4)].list) = checkarglist((yyvsp[(2) - (4)].list), 1);
		(yyval.node) = nod(OTFUNC, fakethis(), N);
		(yyval.node)->list = (yyvsp[(2) - (4)].list);
		(yyval.node)->rlist = (yyvsp[(4) - (4)].list);
	}
    break;

  case 237:

/* Line 1806 of yacc.c  */
#line 1545 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 238:

/* Line 1806 of yacc.c  */
#line 1551 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 240:

/* Line 1806 of yacc.c  */
#line 1560 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 241:

/* Line 1806 of yacc.c  */
#line 1564 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 242:

/* Line 1806 of yacc.c  */
#line 1569 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 243:

/* Line 1806 of yacc.c  */
#line 1573 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 244:

/* Line 1806 of yacc.c  */
#line 1581 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 246:

/* Line 1806 of yacc.c  */
#line 1586 "go.y"
    {
		(yyval.node) = liststmt((yyvsp[(1) - (1)].list));
	}
    break;

  case 248:

/* Line 1806 of yacc.c  */
#line 1591 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 254:

/* Line 1806 of yacc.c  */
#line 1602 "go.y"
    {
		(yyvsp[(1) - (2)].node) = nod(OLABEL, (yyvsp[(1) - (2)].node), N);
		(yyvsp[(1) - (2)].node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 255:

/* Line 1806 of yacc.c  */
#line 1607 "go.y"
    {
		NodeList *l;

		(yyvsp[(1) - (4)].node)->defn = (yyvsp[(4) - (4)].node);
		l = list1((yyvsp[(1) - (4)].node));
		if((yyvsp[(4) - (4)].node))
			l = list(l, (yyvsp[(4) - (4)].node));
		(yyval.node) = liststmt(l);
	}
    break;

  case 256:

/* Line 1806 of yacc.c  */
#line 1617 "go.y"
    {
		// will be converted to OFALL
		(yyval.node) = nod(OXFALL, N, N);
	}
    break;

  case 257:

/* Line 1806 of yacc.c  */
#line 1622 "go.y"
    {
		(yyval.node) = nod(OBREAK, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 258:

/* Line 1806 of yacc.c  */
#line 1626 "go.y"
    {
		(yyval.node) = nod(OCONTINUE, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 259:

/* Line 1806 of yacc.c  */
#line 1630 "go.y"
    {
		(yyval.node) = nod(OPROC, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 260:

/* Line 1806 of yacc.c  */
#line 1634 "go.y"
    {
		(yyval.node) = nod(ODEFER, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 261:

/* Line 1806 of yacc.c  */
#line 1638 "go.y"
    {
		(yyval.node) = nod(OGOTO, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 262:

/* Line 1806 of yacc.c  */
#line 1643 "go.y"
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

  case 263:

/* Line 1806 of yacc.c  */
#line 1662 "go.y"
    {
		(yyval.list) = nil;
		if((yyvsp[(1) - (1)].node) != N)
			(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 264:

/* Line 1806 of yacc.c  */
#line 1668 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (3)].list);
		if((yyvsp[(3) - (3)].node) != N)
			(yyval.list) = list((yyval.list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 265:

/* Line 1806 of yacc.c  */
#line 1676 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 266:

/* Line 1806 of yacc.c  */
#line 1680 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 267:

/* Line 1806 of yacc.c  */
#line 1686 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 268:

/* Line 1806 of yacc.c  */
#line 1690 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 269:

/* Line 1806 of yacc.c  */
#line 1696 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 270:

/* Line 1806 of yacc.c  */
#line 1700 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 271:

/* Line 1806 of yacc.c  */
#line 1706 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 272:

/* Line 1806 of yacc.c  */
#line 1710 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 273:

/* Line 1806 of yacc.c  */
#line 1719 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 274:

/* Line 1806 of yacc.c  */
#line 1723 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 275:

/* Line 1806 of yacc.c  */
#line 1727 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 276:

/* Line 1806 of yacc.c  */
#line 1731 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 277:

/* Line 1806 of yacc.c  */
#line 1736 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 278:

/* Line 1806 of yacc.c  */
#line 1740 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 283:

/* Line 1806 of yacc.c  */
#line 1754 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 285:

/* Line 1806 of yacc.c  */
#line 1760 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 287:

/* Line 1806 of yacc.c  */
#line 1766 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 289:

/* Line 1806 of yacc.c  */
#line 1772 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 291:

/* Line 1806 of yacc.c  */
#line 1778 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 293:

/* Line 1806 of yacc.c  */
#line 1784 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 295:

/* Line 1806 of yacc.c  */
#line 1790 "go.y"
    {
		(yyval.val).ctype = CTxxx;
	}
    break;

  case 297:

/* Line 1806 of yacc.c  */
#line 1800 "go.y"
    {
		importimport((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].val).u.sval);
	}
    break;

  case 298:

/* Line 1806 of yacc.c  */
#line 1804 "go.y"
    {
		importvar((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].type));
	}
    break;

  case 299:

/* Line 1806 of yacc.c  */
#line 1808 "go.y"
    {
		importconst((yyvsp[(2) - (5)].sym), types[TIDEAL], (yyvsp[(4) - (5)].node));
	}
    break;

  case 300:

/* Line 1806 of yacc.c  */
#line 1812 "go.y"
    {
		importconst((yyvsp[(2) - (6)].sym), (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].node));
	}
    break;

  case 301:

/* Line 1806 of yacc.c  */
#line 1816 "go.y"
    {
		importtype((yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].type));
	}
    break;

  case 302:

/* Line 1806 of yacc.c  */
#line 1820 "go.y"
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

  case 303:

/* Line 1806 of yacc.c  */
#line 1838 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		structpkg = (yyval.sym)->pkg;
	}
    break;

  case 304:

/* Line 1806 of yacc.c  */
#line 1845 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
		importsym((yyvsp[(1) - (1)].sym), OTYPE);
	}
    break;

  case 310:

/* Line 1806 of yacc.c  */
#line 1865 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
	}
    break;

  case 311:

/* Line 1806 of yacc.c  */
#line 1869 "go.y"
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

  case 312:

/* Line 1806 of yacc.c  */
#line 1879 "go.y"
    {
		(yyval.type) = aindex(N, (yyvsp[(3) - (3)].type));
	}
    break;

  case 313:

/* Line 1806 of yacc.c  */
#line 1883 "go.y"
    {
		(yyval.type) = aindex(nodlit((yyvsp[(2) - (4)].val)), (yyvsp[(4) - (4)].type));
	}
    break;

  case 314:

/* Line 1806 of yacc.c  */
#line 1887 "go.y"
    {
		(yyval.type) = maptype((yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].type));
	}
    break;

  case 315:

/* Line 1806 of yacc.c  */
#line 1891 "go.y"
    {
		(yyval.type) = tostruct((yyvsp[(3) - (4)].list));
	}
    break;

  case 316:

/* Line 1806 of yacc.c  */
#line 1895 "go.y"
    {
		(yyval.type) = tointerface((yyvsp[(3) - (4)].list));
	}
    break;

  case 317:

/* Line 1806 of yacc.c  */
#line 1899 "go.y"
    {
		(yyval.type) = ptrto((yyvsp[(2) - (2)].type));
	}
    break;

  case 318:

/* Line 1806 of yacc.c  */
#line 1903 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(2) - (2)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 319:

/* Line 1806 of yacc.c  */
#line 1909 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (4)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 320:

/* Line 1806 of yacc.c  */
#line 1915 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Csend;
	}
    break;

  case 321:

/* Line 1806 of yacc.c  */
#line 1923 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Crecv;
	}
    break;

  case 322:

/* Line 1806 of yacc.c  */
#line 1931 "go.y"
    {
		(yyval.type) = functype(nil, (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list));
	}
    break;

  case 323:

/* Line 1806 of yacc.c  */
#line 1937 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(2) - (3)].type)));
		if((yyvsp[(1) - (3)].sym))
			(yyval.node)->left = newname((yyvsp[(1) - (3)].sym));
		(yyval.node)->val = (yyvsp[(3) - (3)].val);
	}
    break;

  case 324:

/* Line 1806 of yacc.c  */
#line 1944 "go.y"
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

  case 325:

/* Line 1806 of yacc.c  */
#line 1960 "go.y"
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

  case 326:

/* Line 1806 of yacc.c  */
#line 1978 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, newname((yyvsp[(1) - (5)].sym)), typenod(functype(fakethis(), (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list))));
	}
    break;

  case 327:

/* Line 1806 of yacc.c  */
#line 1982 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type)));
	}
    break;

  case 328:

/* Line 1806 of yacc.c  */
#line 1987 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 330:

/* Line 1806 of yacc.c  */
#line 1994 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 331:

/* Line 1806 of yacc.c  */
#line 1998 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type))));
	}
    break;

  case 332:

/* Line 1806 of yacc.c  */
#line 2008 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 333:

/* Line 1806 of yacc.c  */
#line 2012 "go.y"
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

  case 334:

/* Line 1806 of yacc.c  */
#line 2027 "go.y"
    {
		(yyval.node) = oldname(pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg));
		if((yyval.node)->op != OLITERAL)
			yyerror("bad constant %S", (yyval.node)->sym);
	}
    break;

  case 336:

/* Line 1806 of yacc.c  */
#line 2036 "go.y"
    {
		if((yyvsp[(2) - (5)].node)->val.ctype == CTRUNE && (yyvsp[(4) - (5)].node)->val.ctype == CTINT) {
			(yyval.node) = (yyvsp[(2) - (5)].node);
			mpaddfixfix((yyvsp[(2) - (5)].node)->val.u.xval, (yyvsp[(4) - (5)].node)->val.u.xval, 0);
			break;
		}
		(yyval.node) = nodcplxlit((yyvsp[(2) - (5)].node)->val, (yyvsp[(4) - (5)].node)->val);
	}
    break;

  case 339:

/* Line 1806 of yacc.c  */
#line 2050 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 340:

/* Line 1806 of yacc.c  */
#line 2054 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 341:

/* Line 1806 of yacc.c  */
#line 2060 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 342:

/* Line 1806 of yacc.c  */
#line 2064 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 343:

/* Line 1806 of yacc.c  */
#line 2070 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 344:

/* Line 1806 of yacc.c  */
#line 2074 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;



/* Line 1806 of yacc.c  */
#line 5290 "y.tab.c"
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
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
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
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

  /* Else will try to reuse lookahead token after shifting the error
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
      if (!yypact_value_is_default (yyn))
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

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
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



/* Line 2067 of yacc.c  */
#line 2078 "go.y"


static void
fixlbrace(int lbr)
{
	// If the opening brace was an LBODY,
	// set up for another one now that we're done.
	// See comment in lex.c about loophack.
	if(lbr == LBODY)
		loophack = 1;
}


