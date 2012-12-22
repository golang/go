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
#define YYLAST   2144

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  76
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  142
/* YYNRULES -- Number of rules.  */
#define YYNRULES  349
/* YYNRULES -- Number of states.  */
#define YYNSTATES  662

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
     439,   445,   450,   457,   459,   464,   470,   476,   484,   486,
     487,   491,   493,   498,   500,   505,   507,   511,   513,   515,
     517,   519,   521,   523,   525,   526,   528,   530,   532,   534,
     539,   541,   543,   545,   548,   550,   552,   554,   556,   558,
     562,   564,   566,   568,   571,   573,   575,   577,   579,   583,
     585,   587,   589,   591,   593,   595,   597,   599,   601,   605,
     610,   615,   618,   622,   628,   630,   632,   635,   639,   645,
     649,   655,   659,   663,   669,   678,   684,   693,   699,   700,
     704,   705,   707,   711,   713,   718,   721,   722,   726,   728,
     732,   734,   738,   740,   744,   746,   750,   752,   756,   760,
     763,   768,   772,   778,   784,   786,   790,   792,   795,   797,
     801,   806,   808,   811,   814,   816,   818,   822,   823,   826,
     827,   829,   831,   833,   835,   837,   839,   841,   843,   845,
     846,   851,   853,   856,   859,   862,   865,   868,   871,   873,
     877,   879,   883,   885,   889,   891,   895,   897,   901,   903,
     905,   909,   913,   914,   917,   918,   920,   921,   923,   924,
     926,   927,   929,   930,   932,   933,   935,   936,   938,   939,
     941,   942,   944,   949,   954,   960,   967,   972,   977,   979,
     981,   983,   985,   987,   989,   991,   993,   995,   999,  1004,
    1010,  1015,  1020,  1023,  1026,  1031,  1035,  1039,  1045,  1049,
    1054,  1058,  1064,  1066,  1067,  1069,  1073,  1075,  1077,  1080,
    1082,  1084,  1090,  1091,  1094,  1096,  1100,  1102,  1106,  1108
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
      59,   126,    60,    -1,   150,   137,   130,   189,    68,    -1,
     129,    67,   130,   189,    68,    -1,    59,   135,    60,    67,
     130,   189,    68,    -1,   165,    -1,    -1,   126,    66,   133,
      -1,   126,    -1,    67,   130,   189,    68,    -1,   126,    -1,
      67,   130,   189,    68,    -1,   129,    -1,    59,   135,    60,
      -1,   126,    -1,   147,    -1,   146,    -1,    35,    -1,    67,
      -1,   141,    -1,   141,    -1,    -1,   138,    -1,    24,    -1,
     142,    -1,    73,    -1,    74,     3,    63,    24,    -1,   141,
      -1,   138,    -1,    11,    -1,    11,   146,    -1,   155,    -1,
     161,    -1,   153,    -1,   154,    -1,   152,    -1,    59,   146,
      60,    -1,   155,    -1,   161,    -1,   153,    -1,    53,   147,
      -1,   161,    -1,   153,    -1,   154,    -1,   152,    -1,    59,
     146,    60,    -1,   161,    -1,   153,    -1,   153,    -1,   155,
      -1,   161,    -1,   153,    -1,   154,    -1,   152,    -1,   143,
      -1,   143,    63,   141,    -1,    71,   192,    72,   146,    -1,
      71,    11,    72,   146,    -1,     8,   148,    -1,     8,    36,
     146,    -1,    23,    71,   146,    72,   146,    -1,   156,    -1,
     157,    -1,    53,   146,    -1,    36,     8,   146,    -1,    29,
     137,   170,   190,    68,    -1,    29,   137,    68,    -1,    22,
     137,   171,   190,    68,    -1,    22,   137,    68,    -1,    17,
     159,   162,    -1,   141,    59,   179,    60,   163,    -1,    59,
     179,    60,   141,    59,   179,    60,   163,    -1,   200,    59,
     195,    60,   210,    -1,    59,   215,    60,   141,    59,   195,
      60,   210,    -1,    17,    59,   179,    60,   163,    -1,    -1,
      67,   183,    68,    -1,    -1,   151,    -1,    59,   179,    60,
      -1,   161,    -1,   164,   137,   183,    68,    -1,   164,     1,
      -1,    -1,   166,    90,    62,    -1,    93,    -1,   167,    62,
      93,    -1,    95,    -1,   168,    62,    95,    -1,    97,    -1,
     169,    62,    97,    -1,   172,    -1,   170,    62,   172,    -1,
     175,    -1,   171,    62,   175,    -1,   184,   146,   198,    -1,
     174,   198,    -1,    59,   174,    60,   198,    -1,    53,   174,
     198,    -1,    59,    53,   174,    60,   198,    -1,    53,    59,
     174,    60,   198,    -1,    24,    -1,    24,    63,   141,    -1,
     173,    -1,   138,   176,    -1,   173,    -1,    59,   173,    60,
      -1,    59,   179,    60,   163,    -1,   136,    -1,   141,   136,
      -1,   141,   145,    -1,   145,    -1,   177,    -1,   178,    75,
     177,    -1,    -1,   178,   191,    -1,    -1,   100,    -1,    91,
      -1,   181,    -1,     1,    -1,    98,    -1,   110,    -1,   121,
      -1,   124,    -1,   113,    -1,    -1,   144,    66,   182,   180,
      -1,    15,    -1,     6,   140,    -1,    10,   140,    -1,    18,
     128,    -1,    13,   128,    -1,    19,   138,    -1,    27,   193,
      -1,   180,    -1,   183,    62,   180,    -1,   138,    -1,   184,
      75,   138,    -1,   139,    -1,   185,    75,   139,    -1,   126,
      -1,   186,    75,   126,    -1,   135,    -1,   187,    75,   135,
      -1,   131,    -1,   132,    -1,   188,    75,   131,    -1,   188,
      75,   132,    -1,    -1,   188,   191,    -1,    -1,    62,    -1,
      -1,    75,    -1,    -1,   126,    -1,    -1,   186,    -1,    -1,
      98,    -1,    -1,   215,    -1,    -1,   216,    -1,    -1,   217,
      -1,    -1,     3,    -1,    21,    24,     3,    62,    -1,    32,
     200,   202,    62,    -1,     9,   200,    65,   213,    62,    -1,
       9,   200,   202,    65,   213,    62,    -1,    31,   201,   202,
      62,    -1,    17,   160,   162,    62,    -1,   142,    -1,   200,
      -1,   204,    -1,   205,    -1,   206,    -1,   204,    -1,   206,
      -1,   142,    -1,    24,    -1,    71,    72,   202,    -1,    71,
       3,    72,   202,    -1,    23,    71,   202,    72,   202,    -1,
      29,    67,   196,    68,    -1,    22,    67,   197,    68,    -1,
      53,   202,    -1,     8,   203,    -1,     8,    59,   205,    60,
      -1,     8,    36,   202,    -1,    36,     8,   202,    -1,    17,
      59,   195,    60,   210,    -1,   141,   202,   198,    -1,   141,
      11,   202,   198,    -1,   141,   202,   198,    -1,   141,    59,
     195,    60,   210,    -1,   202,    -1,    -1,   211,    -1,    59,
     195,    60,    -1,   202,    -1,     3,    -1,    50,     3,    -1,
     141,    -1,   212,    -1,    59,   212,    49,   212,    60,    -1,
      -1,   214,   199,    -1,   207,    -1,   215,    75,   207,    -1,
     208,    -1,   216,    62,   208,    -1,   209,    -1,   217,    62,
     209,    -1
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
     661,   651,   682,   681,   694,   697,   703,   706,   718,   722,
     717,   740,   739,   755,   756,   760,   764,   768,   772,   776,
     780,   784,   788,   792,   796,   800,   804,   808,   812,   816,
     820,   824,   828,   833,   839,   840,   844,   855,   859,   863,
     867,   872,   876,   886,   890,   895,   903,   907,   908,   919,
     923,   927,   931,   935,   936,   942,   949,   955,   962,   965,
     972,   978,   994,  1001,  1002,  1009,  1010,  1029,  1030,  1033,
    1036,  1040,  1051,  1060,  1066,  1069,  1072,  1079,  1080,  1086,
    1101,  1109,  1121,  1126,  1132,  1133,  1134,  1135,  1136,  1137,
    1143,  1144,  1145,  1146,  1152,  1153,  1154,  1155,  1156,  1162,
    1163,  1166,  1169,  1170,  1171,  1172,  1173,  1176,  1177,  1190,
    1194,  1199,  1204,  1209,  1213,  1214,  1217,  1223,  1230,  1236,
    1243,  1249,  1260,  1271,  1300,  1340,  1365,  1383,  1392,  1395,
    1403,  1407,  1411,  1418,  1424,  1429,  1441,  1444,  1453,  1454,
    1460,  1461,  1467,  1471,  1477,  1478,  1484,  1488,  1494,  1517,
    1522,  1528,  1534,  1541,  1550,  1559,  1574,  1580,  1585,  1589,
    1596,  1609,  1610,  1616,  1622,  1625,  1629,  1635,  1638,  1647,
    1650,  1651,  1655,  1656,  1662,  1663,  1664,  1665,  1666,  1668,
    1667,  1682,  1687,  1691,  1695,  1699,  1703,  1708,  1727,  1733,
    1741,  1745,  1751,  1755,  1761,  1765,  1771,  1775,  1784,  1788,
    1792,  1796,  1802,  1805,  1813,  1814,  1816,  1817,  1820,  1823,
    1826,  1829,  1832,  1835,  1838,  1841,  1844,  1847,  1850,  1853,
    1856,  1859,  1865,  1869,  1873,  1877,  1881,  1885,  1905,  1912,
    1923,  1924,  1925,  1928,  1929,  1932,  1936,  1946,  1950,  1954,
    1958,  1962,  1966,  1970,  1976,  1982,  1990,  1998,  2004,  2011,
    2027,  2045,  2049,  2055,  2058,  2061,  2065,  2075,  2079,  2094,
    2102,  2103,  2115,  2116,  2119,  2123,  2129,  2133,  2139,  2143
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
  "$@6", "if_header", "if_stmt", "$@7", "$@8", "$@9", "elseif", "$@10",
  "elseif_list", "else", "switch_stmt", "$@11", "$@12", "select_stmt",
  "$@13", "expr", "uexpr", "pseudocall", "pexpr_no_paren", "start_complit",
  "keyval", "bare_complitexpr", "complitexpr", "pexpr", "expr_or_type",
  "name_or_type", "lbrace", "new_name", "dcl_name", "onew_name", "sym",
  "hidden_importsym", "name", "labelname", "dotdotdot", "ntype",
  "non_expr_type", "non_recvchantype", "convtype", "comptype",
  "fnret_type", "dotname", "othertype", "ptrtype", "recvchantype",
  "structtype", "interfacetype", "xfndcl", "fndcl", "hidden_fndcl",
  "fntype", "fnbody", "fnres", "fnlitdcl", "fnliteral", "xdcl_list",
  "vardcl_list", "constdcl_list", "typedcl_list", "structdcl_list",
  "interfacedcl_list", "structdcl", "packname", "embed", "interfacedcl",
  "indcl", "arg_type", "arg_type_list", "oarg_type_list_ocomma", "stmt",
  "non_dcl_stmt", "$@14", "stmt_list", "new_name_list", "dcl_name_list",
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
       5,     4,     6,     1,     4,     5,     5,     7,     1,     0,
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

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
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
       0,     0,     0,   282,   253,    60,   251,   250,   268,   252,
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
       0,   131,   288,   260,   134,     0,     0,     0,   214,     0,
       0,   323,   313,   314,   294,   298,     0,   296,     0,   322,
     337,     0,     0,   339,   340,     0,     0,     0,     0,     0,
     300,     0,     0,   307,     0,   295,   302,   306,   303,   210,
     169,     0,     0,     0,     0,   246,   247,   160,   211,   186,
     184,   185,   182,   183,   207,   210,   209,    80,    77,   235,
     239,     0,   227,   200,   193,     0,     0,    92,    62,    65,
       0,   231,     0,   300,   225,   198,   271,   228,    64,   223,
      37,   219,    30,    41,     0,   282,    45,   220,   284,    47,
      33,    43,   282,     0,   287,   283,   136,   287,     0,   277,
     124,   130,   129,     0,   135,     0,   269,   325,     0,     0,
     316,     0,   315,     0,   332,   348,   299,     0,     0,     0,
     346,   297,   326,   338,     0,   304,     0,   317,     0,   300,
     328,     0,   345,   333,     0,    69,    68,   292,     0,   247,
     203,    84,   210,     0,    59,     0,   300,   300,   230,     0,
     169,     0,   285,     0,    46,     0,   139,   143,   140,   280,
     281,   125,   132,    61,   324,   333,   294,   321,     0,     0,
     300,   320,     0,     0,   318,   305,   329,   294,   294,   336,
     205,   334,    67,    70,   212,     0,    86,   240,     0,     0,
      56,     0,    63,   233,   232,    90,   137,   221,    34,   142,
     282,   327,     0,   349,   319,   330,   347,     0,     0,     0,
     210,     0,    85,    81,     0,     0,     0,   333,   341,   333,
     335,   204,    82,    87,    58,    57,   144,   331,   206,   292,
       0,    83
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     6,     2,     3,    14,    21,    30,   104,    31,
       8,    24,    16,    17,    65,   326,    67,   148,   516,   517,
     144,   145,    68,   498,   327,   436,   499,   575,   387,   365,
     471,   236,   237,   238,    69,   126,   252,    70,   132,   377,
     571,   642,   659,   616,   643,    71,   142,   398,    72,   140,
      73,    74,    75,    76,   313,   422,   423,   588,    77,   315,
     242,   135,    78,   149,   110,   116,    13,    80,    81,   244,
     245,   162,   118,    82,    83,   478,   227,    84,   229,   230,
      85,    86,    87,   129,   213,    88,   251,   484,    89,    90,
      22,   279,   518,   275,   267,   258,   268,   269,   270,   260,
     383,   246,   247,   248,   328,   329,   321,   330,   271,   151,
      92,   316,   424,   425,   221,   373,   170,   139,   253,   464,
     549,   543,   395,   100,   211,   217,   609,   441,   346,   347,
     348,   350,   550,   545,   610,   611,   454,   455,    25,   465,
     551,   546
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -467
static const yytype_int16 yypact[] =
{
    -467,    53,    59,    64,  -467,    34,  -467,   120,  -467,  -467,
    -467,   144,    94,  -467,   177,   181,  -467,  -467,   145,  -467,
      54,   160,  1026,  -467,   163,   397,   215,  -467,   113,   237,
    -467,    64,   239,  -467,  -467,  -467,    34,  1711,  -467,    34,
    1567,  -467,  -467,   336,  1567,    34,  -467,   161,   180,  1464,
    -467,   161,  -467,   389,   409,  1464,  1464,  1464,  1464,  1464,
    1464,  1507,  1464,  1464,   840,   191,  -467,   461,  -467,  -467,
    -467,  -467,  -467,   672,  -467,  -467,   207,   227,  -467,   213,
    -467,   228,   199,   161,   218,  -467,  -467,  -467,   236,    91,
    -467,  -467,    76,  -467,   223,   -13,   276,   223,   223,   247,
    -467,  -467,  -467,  -467,   255,  -467,  -467,  -467,  -467,  -467,
    -467,  -467,   263,  1722,  1722,  1722,  -467,   262,  -467,  -467,
    -467,  -467,  -467,  -467,   261,   227,  1464,  1679,   265,   259,
     335,  -467,  1464,  -467,  -467,   399,  1722,  2040,   269,  -467,
     293,    23,  1464,   211,  1722,  -467,  -467,   291,  -467,  -467,
    -467,  1593,  -467,  -467,  -467,  -467,  -467,  -467,  1550,  1507,
    2040,   280,  -467,    31,  -467,   171,  -467,  -467,   300,  2040,
     304,  -467,   347,  -467,  1620,  1464,  1464,  1464,  1464,  -467,
    1464,  1464,  1464,  -467,  1464,  1464,  1464,  1464,  1464,  1464,
    1464,  1464,  1464,  1464,  1464,  1464,  1464,  1464,  -467,   726,
     484,  1464,  -467,  1464,  -467,  -467,  1187,  1464,  1464,  1464,
    -467,   310,    34,   259,   295,   377,  -467,  1267,  1267,  -467,
     115,   322,  -467,  1679,   375,  1722,  -467,  -467,  -467,  -467,
    -467,  -467,  -467,   326,    34,  -467,  -467,   353,  -467,    78,
     327,  1722,  -467,  1679,  -467,  -467,  -467,   323,   339,  1679,
    1187,  -467,  -467,   343,   122,   388,  -467,   342,   354,  -467,
    -467,   358,  -467,    21,    32,  -467,  -467,   379,  -467,  -467,
     412,  1652,  -467,  -467,  -467,   387,  -467,  -467,  -467,   413,
    1464,    34,   411,  1754,  -467,   383,  1722,  1722,  -467,   416,
    1464,   414,  2040,   881,  -467,  2064,  1178,  1178,  1178,  1178,
    -467,  1178,  1178,  2088,  -467,   598,   598,   598,   598,  -467,
    -467,  -467,  -467,   936,  -467,  -467,    48,  1256,  -467,  1913,
     408,  1113,  2015,   936,  -467,  -467,  -467,  -467,  -467,  -467,
      -8,   269,   269,  2040,  1833,   425,   419,   421,  -467,   424,
     485,  1267,    90,    49,  -467,   430,  -467,  -467,  -467,  1841,
    -467,    27,   435,    34,   439,   440,   442,  -467,  -467,   445,
    1722,   446,  -467,  -467,  -467,  -467,  1311,  1366,  1464,  -467,
    -467,  -467,  1679,  -467,  1780,   447,   148,   353,  1464,    34,
     459,   449,  1679,  -467,   501,   455,  1722,    92,   388,   412,
     388,   450,   240,   458,  -467,  -467,    34,   412,   492,    34,
     468,    34,   472,   269,  -467,  1464,  1807,  1722,  -467,   165,
     167,   253,   392,  -467,  -467,  -467,    34,   480,   269,  1464,
    -467,  1943,  -467,  -467,   470,   478,   473,  1507,   489,   490,
     491,  -467,  1464,  -467,  -467,   487,  1187,  1113,  -467,  1267,
     516,  -467,  -467,  -467,    34,  1866,  1267,    34,  1267,  -467,
    -467,   550,   196,  -467,  -467,   494,   493,  1267,    90,  1267,
     412,    34,    34,  -467,   503,   486,  -467,  -467,  -467,  1780,
    -467,  1187,  1464,  1464,   507,  -467,  1679,   512,  -467,  -467,
    -467,  -467,  -467,  -467,  -467,  1780,  -467,  -467,  -467,  -467,
    -467,   513,  -467,  -467,  -467,  1507,   515,  -467,  -467,  -467,
     518,  -467,   519,   412,  -467,  -467,  -467,  -467,  -467,  -467,
    -467,  -467,  -467,   269,   522,   936,  -467,  -467,   521,  1620,
    -467,   269,   936,  1409,   936,  -467,  -467,  -467,   524,  -467,
    -467,  -467,  -467,   514,  -467,   192,  -467,  -467,   525,   528,
     530,   533,   535,   527,  -467,  -467,   537,   532,  1267,   538,
    -467,   543,  -467,  -467,   562,  -467,  1267,  -467,   551,   412,
    -467,   553,  -467,  1874,   219,  2040,  2040,  1464,   554,  1679,
    -467,  -467,  1780,    45,  -467,  1113,   412,   412,  -467,    93,
     429,   548,    34,   557,   414,   552,  -467,  2040,  -467,  -467,
    -467,  -467,  -467,  -467,  -467,  1874,    34,  -467,  1866,  1267,
     412,  -467,    34,   196,  -467,  -467,  -467,    34,    34,  -467,
    -467,  -467,  -467,  -467,  -467,   558,   605,  -467,  1464,  1464,
    -467,  1507,   560,  -467,  -467,  -467,  -467,  -467,  -467,  -467,
     936,  -467,   563,  -467,  -467,  -467,  -467,   564,   565,   567,
    1780,    68,  -467,  -467,  1967,  1991,   561,  1874,  -467,  1874,
    -467,  -467,  -467,  -467,  -467,  -467,  -467,  -467,  -467,  1464,
     353,  -467
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -467,  -467,  -467,  -467,  -467,  -467,  -467,    -6,  -467,  -467,
     599,  -467,    13,  -467,  -467,   613,  -467,  -123,   -24,    58,
    -467,  -127,  -121,  -467,     2,  -467,  -467,  -467,   128,  -369,
    -467,  -467,  -467,  -467,  -467,  -467,  -140,  -467,  -467,  -467,
    -467,  -467,  -467,  -467,  -467,  -467,  -467,  -467,  -467,  -467,
     632,    12,    33,  -467,  -201,   121,   123,  -467,   183,   -54,
     401,   114,   -26,   369,   616,    -5,   420,   385,  -467,   417,
     -50,   498,  -467,  -467,  -467,  -467,   -33,    18,   -31,   -25,
    -467,  -467,  -467,  -467,  -467,   194,   448,  -460,  -467,  -467,
    -467,  -467,  -467,  -467,  -467,  -467,   267,  -109,  -231,   278,
    -467,   292,  -467,  -214,  -282,   643,  -467,  -223,  -467,   -66,
     282,   172,  -467,  -311,  -238,  -274,  -183,  -467,  -112,  -414,
    -467,  -467,  -294,  -467,   264,  -467,    85,  -467,   332,   229,
     340,   216,    71,    77,  -466,  -467,  -424,   222,  -467,   474,
    -467,  -467
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -275
static const yytype_int16 yytable[] =
{
      12,   174,   272,   323,   119,   235,   121,   161,   487,   359,
     109,   235,   435,   109,   240,    32,   274,    79,   320,   131,
     385,   235,   103,    32,   278,   570,   259,   376,   554,   393,
     539,   111,   389,   391,   111,   375,   164,   400,   128,   433,
     111,   402,   428,   173,   107,   380,   212,   254,   146,   150,
     618,   417,   456,     4,   437,   120,   380,    27,     9,   426,
     438,    11,   150,   226,   232,   233,  -181,   152,   153,   154,
     155,   156,   157,   124,   166,   167,   263,   130,     9,   163,
     388,   207,   264,   366,     5,   390,   261,   461,   652,     7,
    -180,   265,   205,   450,   276,   501,    10,    11,  -181,   495,
     495,   282,   462,   507,   496,   496,   174,    10,    11,   257,
     619,   620,   617,    28,     9,   266,    27,    29,    27,   222,
     621,   457,   243,   427,   291,  -234,   133,    10,    11,   631,
     111,   228,   228,   228,   164,   325,   111,     9,   146,     9,
     451,   208,   150,   367,    15,   228,   381,    18,   289,   452,
     525,   209,   528,   209,   228,   536,    19,   500,   134,   502,
     497,   625,   228,    10,    11,   141,   560,   150,   491,   228,
     152,   156,  -213,   102,   164,   361,    29,   163,    29,   637,
     651,   657,   632,   658,  -234,   379,    10,    11,    10,    11,
    -234,   369,   228,   638,   639,   318,   133,   204,    20,   450,
    -177,    79,  -175,   206,   581,    23,  -213,   349,    26,   578,
     437,   585,   515,   535,   357,    32,   486,   163,   243,   522,
       9,   397,    33,   125,  -177,    93,  -175,   125,   134,   363,
    -179,   122,  -177,   408,  -175,     9,   414,   415,  -213,   101,
     105,   228,   108,   228,   243,    79,   451,   235,   564,   533,
     409,   136,   411,   171,   437,   165,   474,   235,   203,   228,
     593,   228,   568,   430,   254,   606,   488,   228,  -265,    10,
      11,   273,   509,  -265,   198,   259,   150,  -180,   511,  -152,
     583,   437,   623,   624,    10,    11,   199,   612,  -176,   228,
     200,   661,   164,   263,   202,  -179,   345,    11,   201,   264,
     215,   410,   355,   356,   228,   228,   635,   231,   231,   231,
     408,   219,  -176,    10,    11,     9,    79,   220,   334,   646,
    -176,   231,   223,  -265,   249,   234,   250,   335,   262,  -265,
     231,   138,   336,   337,   338,   163,   494,   453,   231,   339,
     285,   479,  -264,   481,   209,   231,   340,  -264,   349,   482,
     519,   277,   622,   165,   353,   615,   226,   514,   257,   214,
       9,   216,   218,   341,    10,    11,   266,   243,   231,   477,
     506,     9,   286,   529,   489,   342,   287,   243,   228,   111,
     354,   343,   358,   360,    11,   630,   362,   111,   364,   368,
     228,   111,   480,   165,   146,   127,   150,  -264,   372,   374,
     228,   382,   164,  -264,   228,   378,    94,   288,   239,    10,
      11,   150,   380,     9,    95,   394,   384,   231,    96,   231,
      10,    11,   117,   254,   228,   228,   449,  -174,    97,    98,
     386,    79,    79,     9,   460,   231,   479,   231,   481,   349,
     541,   392,   548,   231,   482,   163,   235,   453,   143,   399,
     413,  -174,   479,   453,   481,   613,   561,   349,   255,  -174,
     482,    99,    10,    11,  -178,   231,    79,   256,   147,   584,
     164,   243,    10,    11,   432,   401,   405,   412,   416,   419,
     231,   231,    10,    11,   444,     9,   445,   480,  -178,   331,
     332,   447,   446,   448,   228,   458,  -178,   463,   117,   117,
     117,   466,   467,   480,   468,   469,   470,   485,     9,   490,
     503,   165,   117,   163,   210,   210,   519,   210,   210,   660,
     172,   117,   379,   493,   537,   254,   505,   508,   510,   117,
     544,   547,   512,   552,    10,    11,   117,   228,   235,   479,
     520,   481,   557,   317,   559,   524,   526,   482,   527,   530,
     531,   532,   340,   553,   231,   534,   555,    10,    11,   117,
     255,   462,   403,   563,   243,   556,   231,   529,   483,   567,
      79,   569,   418,   572,    10,    11,   231,   150,   576,   577,
     231,   574,   580,   582,   591,   594,   592,   228,   595,  -156,
     480,   349,   596,   541,  -157,   597,   164,   548,   453,   598,
     231,   231,   349,   349,   599,   602,   601,   479,   117,   481,
     117,   603,   607,   605,   614,   482,   626,   628,   640,   641,
     629,   165,   437,   647,   648,   649,   117,   650,   117,   656,
     106,   344,   177,   600,   117,    66,   579,   344,   344,   163,
     627,   604,   185,   653,   370,   589,   189,   590,   331,   332,
     404,   194,   195,   196,   197,   123,   117,   284,   480,   504,
     371,   352,   492,   483,   475,    91,   442,   573,   117,   538,
     231,   117,   117,   636,   443,   633,   175,  -274,   562,   483,
     558,   137,     0,   544,   634,     0,   351,   513,     0,   165,
       0,     0,     0,   160,     0,     0,   169,     0,     0,     0,
       0,   521,     0,     0,     0,   176,   177,     0,   178,   179,
     180,   181,   182,   231,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,    35,
       0,     0,     0,     0,    37,     0,     0,  -274,     0,     0,
       0,     0,     0,   112,     0,   117,     0,  -274,    47,    48,
       9,     0,     0,     0,   344,    51,     0,   117,     0,   117,
       0,   344,   158,   231,     0,     0,   483,   117,     0,   344,
       0,   117,     0,     0,     0,    56,    57,     0,    58,   159,
       0,     0,    60,     0,     0,    61,   314,     0,     0,     0,
       0,   117,   117,     0,     0,    62,    63,    64,     0,    10,
      11,     0,     0,     0,     0,     0,     0,   292,   293,   294,
     295,     0,   296,   297,   298,   165,   299,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
       0,   160,     0,   319,   483,   322,     0,     0,     0,   137,
     137,   333,     0,    35,     0,     0,     0,     0,    37,     0,
       0,   168,     0,     0,   117,     0,     0,   112,     0,   344,
       0,   117,    47,    48,     9,   542,   344,     0,   344,    51,
     117,     0,     0,     0,     0,     0,    55,   344,     0,   344,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    56,
      57,     0,    58,    59,     0,     0,    60,     0,     0,    61,
       0,     0,     0,     0,   117,     0,     0,     0,     0,    62,
      63,    64,   137,    10,    11,   177,     0,     0,     0,   180,
     181,   182,   137,     0,   184,   185,   186,   187,     0,   189,
     190,   191,   192,   193,   194,   195,   196,   197,     0,    35,
       0,     0,     0,     0,    37,   421,     0,     0,     0,   160,
       0,     0,     0,   112,   117,   421,     0,   117,    47,    48,
       9,     0,     0,     0,     0,    51,     0,     0,   344,     0,
       0,     0,    55,     0,     0,     0,   344,     0,     0,     0,
       0,     0,     0,   344,     0,    56,    57,     0,    58,    59,
       0,     0,    60,     0,     0,    61,     0,     0,   137,   137,
       0,     0,     0,   420,     0,    62,    63,    64,     0,    10,
      11,     0,     0,     0,     0,   344,     0,     0,   542,   344,
       0,     0,     0,     0,     0,   117,    -2,    34,     0,    35,
       0,     0,    36,     0,    37,    38,    39,   137,     0,    40,
       0,    41,    42,    43,    44,    45,    46,     0,    47,    48,
       9,   137,     0,    49,    50,    51,    52,    53,    54,   160,
       0,     0,    55,     0,   169,     0,     0,   344,     0,   344,
       0,     0,     0,     0,     0,    56,    57,     0,    58,    59,
       0,     0,    60,     0,     0,    61,     0,     0,   -24,     0,
       0,     0,     0,     0,     0,    62,    63,    64,     0,    10,
      11,     0,     0,     0,   565,   566,     0,     0,     0,     0,
       0,     0,     0,     0,   324,     0,    35,     0,     0,    36,
    -249,    37,    38,    39,     0,  -249,    40,   160,    41,    42,
     112,    44,    45,    46,     0,    47,    48,     9,     0,     0,
      49,    50,    51,    52,    53,    54,     0,   421,     0,    55,
       0,     0,     0,     0,   421,   587,   421,     0,     0,     0,
       0,     0,    56,    57,     0,    58,    59,     0,     0,    60,
       0,     0,    61,     0,     0,  -249,     0,     0,     0,     0,
     325,  -249,    62,    63,    64,     0,    10,    11,   324,     0,
      35,     0,     0,    36,     0,    37,    38,    39,     0,     0,
      40,     0,    41,    42,   112,    44,    45,    46,     0,    47,
      48,     9,   177,     0,    49,    50,    51,    52,    53,    54,
       0,     0,   185,    55,     0,     0,   189,   190,   191,   192,
     193,   194,   195,   196,   197,     0,    56,    57,     0,    58,
      59,     0,     0,    60,     0,     0,    61,     0,     0,  -249,
     644,   645,     0,   160,   325,  -249,    62,    63,    64,    35,
      10,    11,   421,     0,    37,     0,     0,     0,     0,     0,
       0,     0,     0,   112,     0,   334,     0,     0,    47,    48,
       9,     0,     0,     0,   335,    51,     0,   429,     0,   336,
     337,   338,   158,     0,     0,     0,   339,     0,     0,     0,
       0,     0,     0,   340,     0,    56,    57,     0,    58,   159,
       0,     0,    60,     0,    35,    61,     0,     0,     0,    37,
     341,     0,     0,     0,     0,    62,    63,    64,   112,    10,
      11,     0,     0,    47,    48,     9,     0,   472,   343,     0,
      51,    11,     0,     0,     0,     0,     0,    55,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      56,    57,     0,    58,    59,     0,     0,    60,     0,    35,
      61,     0,     0,     0,    37,     0,     0,     0,     0,     0,
      62,    63,    64,   112,    10,    11,     0,     0,    47,    48,
       9,     0,   473,     0,     0,    51,     0,     0,     0,     0,
       0,     0,    55,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    35,     0,     0,    56,    57,    37,    58,    59,
       0,     0,    60,     0,     0,    61,   112,     0,     0,     0,
       0,    47,    48,     9,     0,    62,    63,    64,    51,    10,
      11,     0,     0,     0,     0,    55,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    56,    57,
       0,    58,    59,     0,     0,    60,     0,    35,    61,     0,
       0,     0,    37,     0,     0,     0,   586,     0,    62,    63,
      64,   112,    10,    11,     0,     0,    47,    48,     9,     0,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
      55,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      35,     0,     0,    56,    57,    37,    58,    59,     0,     0,
      60,     0,     0,    61,   112,     0,     0,     0,     0,    47,
      48,     9,     0,    62,    63,    64,    51,    10,    11,     0,
       0,     0,     0,   158,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    35,     0,     0,    56,    57,   283,    58,
     159,     0,     0,    60,     0,     0,    61,   112,     0,     0,
      35,     0,    47,    48,     9,    37,    62,    63,    64,    51,
      10,    11,     0,     0,   112,     0,    55,     0,     0,    47,
      48,     9,     0,     0,     0,     0,    51,     0,     0,    56,
      57,    37,    58,    59,     0,     0,    60,     0,     0,    61,
     112,     0,     0,     0,     0,    47,    48,     9,     0,    62,
      63,    64,    51,    10,    11,     0,    61,     0,    37,   224,
       0,     0,     0,     0,     0,     0,     0,   112,    64,     0,
      10,    11,    47,    48,     9,     0,   114,     0,     0,    51,
       0,     0,   225,     0,     0,     0,   224,     0,   280,     0,
      37,     0,     0,     0,    64,     0,    10,    11,   281,   112,
       0,     0,     0,   114,    47,    48,     9,     0,     0,   225,
       0,    51,     0,     0,     0,   290,     0,    37,   224,     0,
     241,    64,     0,    10,    11,   281,   112,     0,     0,     0,
       0,    47,    48,     9,     0,   114,     0,     0,    51,     0,
       0,   225,     0,     0,     0,   224,     0,     0,     0,    37,
       0,     0,     0,    64,     0,    10,    11,   396,   112,     0,
      37,     0,   114,    47,    48,     9,     0,     0,   225,   112,
      51,     0,     0,     0,    47,    48,     9,   113,     0,     0,
      64,    51,    10,    11,     0,     0,     0,     0,   224,     0,
       0,     0,    37,     0,   114,     0,     0,     0,     0,     0,
     115,   112,     0,     0,     0,   114,    47,    48,     9,     0,
       0,   225,    64,    51,    10,    11,     0,     0,    37,     0,
     406,     0,     0,    64,     0,    10,    11,   112,     0,     0,
       0,     0,    47,    48,     9,     0,     0,   114,     0,    51,
       0,     0,     0,   407,     0,   283,   224,     0,     0,     0,
       0,     0,     0,     0,   112,    64,     0,    10,    11,    47,
      48,     9,     0,   114,     0,     0,    51,     0,     0,   476,
       0,   334,     0,   224,     0,     0,     0,     0,     0,   334,
     335,    64,   459,    10,    11,   336,   337,   338,   335,     0,
     114,     0,   339,   336,   337,   338,   225,     0,     0,   439,
     339,     0,     0,     0,   334,     0,     0,   340,    64,     0,
      10,    11,   334,   335,     0,     0,   341,     0,   336,   337,
     540,   335,   440,     0,   341,   339,   336,   337,   338,     0,
       0,     0,   340,   339,   343,     0,     0,    11,     0,     0,
     340,     0,   343,     0,     0,    11,     0,     0,     0,   341,
       0,     0,     0,     0,     0,     0,     0,   341,     0,     0,
       0,     0,     0,   608,     0,     0,     0,   343,     0,    10,
      11,     0,     0,     0,     0,   343,   176,   177,    11,   178,
       0,   180,   181,   182,     0,     0,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
       0,     0,     0,     0,     0,     0,   176,   177,     0,   178,
       0,   180,   181,   182,     0,   431,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     176,   177,     0,   178,     0,   180,   181,   182,     0,   523,
     184,   185,   186,   187,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   197,   176,   177,     0,   178,     0,   180,
     181,   182,     0,   654,   184,   185,   186,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   176,   177,
       0,   178,     0,   180,   181,   182,     0,   655,   184,   185,
     186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,   197,     0,   176,   177,   434,   178,     0,   180,   181,
     182,     0,     0,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   176,   177,     0,
       0,     0,   180,   181,   182,     0,     0,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   176,   177,     0,     0,     0,   180,   181,   182,     0,
       0,   184,   185,   186,   187,     0,   189,   190,   191,   192,
     193,   194,   195,   196,   197
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-467))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       5,    67,   142,   204,    37,   126,    37,    61,   377,   223,
      36,   132,   323,    39,   126,    20,   143,    22,   201,    45,
     258,   142,    28,    28,   147,   485,   135,   250,   452,   267,
     444,    36,   263,   264,    39,   249,    61,   275,    43,   321,
      45,   279,   316,    67,    31,    24,    59,    24,    53,    54,
       5,   289,     3,     0,    62,    37,    24,     3,    24,    11,
      68,    74,    67,   113,   114,   115,    35,    55,    56,    57,
      58,    59,    60,    40,    62,    63,    53,    44,    24,    61,
      59,     5,    59,     5,    25,    53,   136,    60,    20,    25,
      59,    68,     1,     3,   144,   389,    73,    74,    67,     7,
       7,   151,    75,   397,    12,    12,   172,    73,    74,   135,
      65,    66,   572,    59,    24,   141,     3,    63,     3,   106,
      75,    72,   127,    75,   174,     3,    35,    73,    74,   595,
     135,   113,   114,   115,   159,    67,   141,    24,   143,    24,
      50,    65,   147,    65,    24,   127,   255,     3,   172,    59,
     424,    75,   426,    75,   136,   437,    62,   388,    67,   390,
      68,    68,   144,    73,    74,    51,   460,   172,   382,   151,
     158,   159,     1,    60,   199,   225,    63,   159,    63,   603,
     640,   647,   596,   649,    62,    63,    73,    74,    73,    74,
      68,   241,   174,   607,   608,   200,    35,    83,    21,     3,
      35,   206,    35,    89,   515,    24,    35,   212,    63,   503,
      62,   522,   413,   436,   220,   220,    68,   199,   223,   420,
      24,   271,    62,    40,    59,    62,    59,    44,    67,   234,
      59,    37,    67,   283,    67,    24,   286,   287,    67,    24,
       3,   223,     3,   225,   249,   250,    50,   368,   471,   432,
     283,    71,   283,    62,    62,    61,   368,   378,    59,   241,
      68,   243,   476,   317,    24,   559,   378,   249,     7,    73,
      74,    60,   399,    12,    67,   384,   281,    59,   401,    66,
     518,    62,   576,   577,    73,    74,    59,    68,    35,   271,
      63,   660,   317,    53,    66,    59,   211,    74,    71,    59,
      24,   283,   217,   218,   286,   287,   600,   113,   114,   115,
     360,    64,    59,    73,    74,    24,   321,    62,     8,   630,
      67,   127,    59,    62,    59,    63,    67,    17,    35,    68,
     136,    49,    22,    23,    24,   317,   386,   342,   144,    29,
      60,   374,     7,   374,    75,   151,    36,    12,   353,   374,
     416,    60,   575,   159,    59,   569,   406,   407,   384,    95,
      24,    97,    98,    53,    73,    74,   392,   372,   174,   374,
     396,    24,    72,   427,   379,    65,    72,   382,   360,   384,
       3,    71,    60,     8,    74,   586,    60,   392,    35,    62,
     372,   396,   374,   199,   399,    59,   401,    62,    75,    60,
     382,    59,   427,    68,   386,    62,     9,    60,   126,    73,
      74,   416,    24,    24,    17,     3,    62,   223,    21,   225,
      73,    74,    37,    24,   406,   407,   341,    35,    31,    32,
      72,   436,   437,    24,   349,   241,   469,   243,   469,   444,
     445,    62,   447,   249,   469,   427,   567,   452,    59,    62,
      67,    59,   485,   458,   485,   567,   461,   462,    59,    67,
     485,    64,    73,    74,    35,   271,   471,    68,    59,   519,
     495,   476,    73,    74,    66,    62,    65,   283,    62,    65,
     286,   287,    73,    74,    59,    24,    67,   469,    59,   207,
     208,    67,    71,     8,   476,    65,    67,    62,   113,   114,
     115,    62,    62,   485,    62,    60,    60,    60,    24,    60,
      60,   317,   127,   495,    94,    95,   582,    97,    98,   659,
      59,   136,    63,    68,   439,    24,    68,    35,    60,   144,
     445,   446,    60,   448,    73,    74,   151,   519,   659,   572,
      60,   572,   457,    59,   459,    75,    68,   572,    75,    60,
      60,    60,    36,     3,   360,    68,    62,    73,    74,   174,
      59,    75,   280,    60,   569,    72,   372,   621,   374,    62,
     575,    59,   290,    60,    73,    74,   382,   582,    60,    60,
     386,    66,    60,    62,    60,    60,    72,   569,    60,    59,
     572,   596,    59,   598,    59,    68,   621,   602,   603,    62,
     406,   407,   607,   608,    72,    62,    68,   640,   223,   640,
     225,    49,    59,    62,    60,   640,    68,    60,    60,    14,
      68,   427,    62,    60,    60,    60,   241,    60,   243,    68,
      31,   211,    34,   548,   249,    22,   508,   217,   218,   621,
     582,   556,    44,   641,   243,   524,    48,   524,   366,   367,
     281,    53,    54,    55,    56,    39,   271,   159,   640,   392,
     243,   213,   384,   469,   372,    22,   334,   495,   283,   440,
     476,   286,   287,   602,   334,   598,     4,     5,   462,   485,
     458,    49,    -1,   598,   599,    -1,   212,   405,    -1,   495,
      -1,    -1,    -1,    61,    -1,    -1,    64,    -1,    -1,    -1,
      -1,   419,    -1,    -1,    -1,    33,    34,    -1,    36,    37,
      38,    39,    40,   519,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,     3,
      -1,    -1,    -1,    -1,     8,    -1,    -1,    65,    -1,    -1,
      -1,    -1,    -1,    17,    -1,   360,    -1,    75,    22,    23,
      24,    -1,    -1,    -1,   334,    29,    -1,   372,    -1,   374,
      -1,   341,    36,   569,    -1,    -1,   572,   382,    -1,   349,
      -1,   386,    -1,    -1,    -1,    49,    50,    -1,    52,    53,
      -1,    -1,    56,    -1,    -1,    59,    60,    -1,    -1,    -1,
      -1,   406,   407,    -1,    -1,    69,    70,    71,    -1,    73,
      74,    -1,    -1,    -1,    -1,    -1,    -1,   175,   176,   177,
     178,    -1,   180,   181,   182,   621,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
      -1,   199,    -1,   201,   640,   203,    -1,    -1,    -1,   207,
     208,   209,    -1,     3,    -1,    -1,    -1,    -1,     8,    -1,
      -1,    11,    -1,    -1,   469,    -1,    -1,    17,    -1,   439,
      -1,   476,    22,    23,    24,   445,   446,    -1,   448,    29,
     485,    -1,    -1,    -1,    -1,    -1,    36,   457,    -1,   459,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,
      50,    -1,    52,    53,    -1,    -1,    56,    -1,    -1,    59,
      -1,    -1,    -1,    -1,   519,    -1,    -1,    -1,    -1,    69,
      70,    71,   280,    73,    74,    34,    -1,    -1,    -1,    38,
      39,    40,   290,    -1,    43,    44,    45,    46,    -1,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    -1,     3,
      -1,    -1,    -1,    -1,     8,   313,    -1,    -1,    -1,   317,
      -1,    -1,    -1,    17,   569,   323,    -1,   572,    22,    23,
      24,    -1,    -1,    -1,    -1,    29,    -1,    -1,   548,    -1,
      -1,    -1,    36,    -1,    -1,    -1,   556,    -1,    -1,    -1,
      -1,    -1,    -1,   563,    -1,    49,    50,    -1,    52,    53,
      -1,    -1,    56,    -1,    -1,    59,    -1,    -1,   366,   367,
      -1,    -1,    -1,    67,    -1,    69,    70,    71,    -1,    73,
      74,    -1,    -1,    -1,    -1,   595,    -1,    -1,   598,   599,
      -1,    -1,    -1,    -1,    -1,   640,     0,     1,    -1,     3,
      -1,    -1,     6,    -1,     8,     9,    10,   405,    -1,    13,
      -1,    15,    16,    17,    18,    19,    20,    -1,    22,    23,
      24,   419,    -1,    27,    28,    29,    30,    31,    32,   427,
      -1,    -1,    36,    -1,   432,    -1,    -1,   647,    -1,   649,
      -1,    -1,    -1,    -1,    -1,    49,    50,    -1,    52,    53,
      -1,    -1,    56,    -1,    -1,    59,    -1,    -1,    62,    -1,
      -1,    -1,    -1,    -1,    -1,    69,    70,    71,    -1,    73,
      74,    -1,    -1,    -1,   472,   473,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,     3,    -1,    -1,     6,
       7,     8,     9,    10,    -1,    12,    13,   495,    15,    16,
      17,    18,    19,    20,    -1,    22,    23,    24,    -1,    -1,
      27,    28,    29,    30,    31,    32,    -1,   515,    -1,    36,
      -1,    -1,    -1,    -1,   522,   523,   524,    -1,    -1,    -1,
      -1,    -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,    -1,    59,    -1,    -1,    62,    -1,    -1,    -1,    -1,
      67,    68,    69,    70,    71,    -1,    73,    74,     1,    -1,
       3,    -1,    -1,     6,    -1,     8,     9,    10,    -1,    -1,
      13,    -1,    15,    16,    17,    18,    19,    20,    -1,    22,
      23,    24,    34,    -1,    27,    28,    29,    30,    31,    32,
      -1,    -1,    44,    36,    -1,    -1,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    -1,    49,    50,    -1,    52,
      53,    -1,    -1,    56,    -1,    -1,    59,    -1,    -1,    62,
     618,   619,    -1,   621,    67,    68,    69,    70,    71,     3,
      73,    74,   630,    -1,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    17,    -1,     8,    -1,    -1,    22,    23,
      24,    -1,    -1,    -1,    17,    29,    -1,    31,    -1,    22,
      23,    24,    36,    -1,    -1,    -1,    29,    -1,    -1,    -1,
      -1,    -1,    -1,    36,    -1,    49,    50,    -1,    52,    53,
      -1,    -1,    56,    -1,     3,    59,    -1,    -1,    -1,     8,
      53,    -1,    -1,    -1,    -1,    69,    70,    71,    17,    73,
      74,    -1,    -1,    22,    23,    24,    -1,    26,    71,    -1,
      29,    74,    -1,    -1,    -1,    -1,    -1,    36,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      49,    50,    -1,    52,    53,    -1,    -1,    56,    -1,     3,
      59,    -1,    -1,    -1,     8,    -1,    -1,    -1,    -1,    -1,
      69,    70,    71,    17,    73,    74,    -1,    -1,    22,    23,
      24,    -1,    26,    -1,    -1,    29,    -1,    -1,    -1,    -1,
      -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     3,    -1,    -1,    49,    50,     8,    52,    53,
      -1,    -1,    56,    -1,    -1,    59,    17,    -1,    -1,    -1,
      -1,    22,    23,    24,    -1,    69,    70,    71,    29,    73,
      74,    -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,
      -1,    52,    53,    -1,    -1,    56,    -1,     3,    59,    -1,
      -1,    -1,     8,    -1,    -1,    -1,    67,    -1,    69,    70,
      71,    17,    73,    74,    -1,    -1,    22,    23,    24,    -1,
      -1,    -1,    -1,    29,    -1,    -1,    -1,    -1,    -1,    -1,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       3,    -1,    -1,    49,    50,     8,    52,    53,    -1,    -1,
      56,    -1,    -1,    59,    17,    -1,    -1,    -1,    -1,    22,
      23,    24,    -1,    69,    70,    71,    29,    73,    74,    -1,
      -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     3,    -1,    -1,    49,    50,     8,    52,
      53,    -1,    -1,    56,    -1,    -1,    59,    17,    -1,    -1,
       3,    -1,    22,    23,    24,     8,    69,    70,    71,    29,
      73,    74,    -1,    -1,    17,    -1,    36,    -1,    -1,    22,
      23,    24,    -1,    -1,    -1,    -1,    29,    -1,    -1,    49,
      50,     8,    52,    53,    -1,    -1,    56,    -1,    -1,    59,
      17,    -1,    -1,    -1,    -1,    22,    23,    24,    -1,    69,
      70,    71,    29,    73,    74,    -1,    59,    -1,     8,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    17,    71,    -1,
      73,    74,    22,    23,    24,    -1,    53,    -1,    -1,    29,
      -1,    -1,    59,    -1,    -1,    -1,    36,    -1,    65,    -1,
       8,    -1,    -1,    -1,    71,    -1,    73,    74,    75,    17,
      -1,    -1,    -1,    53,    22,    23,    24,    -1,    -1,    59,
      -1,    29,    -1,    -1,    -1,    65,    -1,     8,    36,    -1,
      11,    71,    -1,    73,    74,    75,    17,    -1,    -1,    -1,
      -1,    22,    23,    24,    -1,    53,    -1,    -1,    29,    -1,
      -1,    59,    -1,    -1,    -1,    36,    -1,    -1,    -1,     8,
      -1,    -1,    -1,    71,    -1,    73,    74,    75,    17,    -1,
       8,    -1,    53,    22,    23,    24,    -1,    -1,    59,    17,
      29,    -1,    -1,    -1,    22,    23,    24,    36,    -1,    -1,
      71,    29,    73,    74,    -1,    -1,    -1,    -1,    36,    -1,
      -1,    -1,     8,    -1,    53,    -1,    -1,    -1,    -1,    -1,
      59,    17,    -1,    -1,    -1,    53,    22,    23,    24,    -1,
      -1,    59,    71,    29,    73,    74,    -1,    -1,     8,    -1,
      36,    -1,    -1,    71,    -1,    73,    74,    17,    -1,    -1,
      -1,    -1,    22,    23,    24,    -1,    -1,    53,    -1,    29,
      -1,    -1,    -1,    59,    -1,     8,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    17,    71,    -1,    73,    74,    22,
      23,    24,    -1,    53,    -1,    -1,    29,    -1,    -1,    59,
      -1,     8,    -1,    36,    -1,    -1,    -1,    -1,    -1,     8,
      17,    71,    11,    73,    74,    22,    23,    24,    17,    -1,
      53,    -1,    29,    22,    23,    24,    59,    -1,    -1,    36,
      29,    -1,    -1,    -1,     8,    -1,    -1,    36,    71,    -1,
      73,    74,     8,    17,    -1,    -1,    53,    -1,    22,    23,
      24,    17,    59,    -1,    53,    29,    22,    23,    24,    -1,
      -1,    -1,    36,    29,    71,    -1,    -1,    74,    -1,    -1,
      36,    -1,    71,    -1,    -1,    74,    -1,    -1,    -1,    53,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    53,    -1,    -1,
      -1,    -1,    -1,    59,    -1,    -1,    -1,    71,    -1,    73,
      74,    -1,    -1,    -1,    -1,    71,    33,    34,    74,    36,
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
     135,    72,    66,   180,    60,   189,   101,    62,    68,    36,
      59,   203,   204,   206,    59,    67,    71,    67,     8,   202,
       3,    50,    59,   141,   212,   213,     3,    72,    65,    11,
     202,    60,    75,    62,   195,   215,    62,    62,    62,    60,
      60,   106,    26,    26,   194,   177,    59,   141,   151,   152,
     153,   154,   155,   161,   163,    60,    68,   105,   194,   141,
      60,   179,   175,    68,   146,     7,    12,    68,    99,   102,
     174,   198,   174,    60,   172,    68,   138,   198,    35,    97,
      60,    93,    60,   186,   146,   130,    94,    95,   168,   185,
      60,   186,   130,    66,    75,   191,    68,    75,   191,   135,
      60,    60,    60,   192,    68,   183,   180,   202,   205,   195,
      24,   141,   142,   197,   202,   209,   217,   202,   141,   196,
     208,   216,   202,     3,   212,    62,    72,   202,   213,   202,
     198,   141,   207,    60,   183,   126,   126,    62,   179,    59,
     163,   116,    60,   187,    66,   103,    60,    60,   198,   104,
      60,   189,    62,   190,   146,   189,    67,   126,   133,   131,
     132,    60,    72,    68,    60,    60,    59,    68,    62,    72,
     202,    68,    62,    49,   202,    62,   198,    59,    59,   202,
     210,   211,    68,   194,    60,   179,   119,   163,     5,    65,
      66,    75,   183,   198,   198,    68,    68,    95,    60,    68,
     130,   210,   195,   209,   202,   198,   208,   212,   195,   195,
      60,    14,   117,   120,   126,   126,   189,    60,    60,    60,
      60,   163,    20,   100,    66,    66,    68,   210,   210,   118,
     112,   105
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

/* Line 1806 of yacc.c  */
#line 682 "go.y"
    {
		markdcl();
	}
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 686 "go.y"
    {
		if((yyvsp[(4) - (5)].node)->ntest == N)
			yyerror("missing condition in if statement");
		(yyvsp[(4) - (5)].node)->nbody = (yyvsp[(5) - (5)].list);
		(yyval.list) = list1((yyvsp[(4) - (5)].node));
	}
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 694 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 698 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].list));
	}
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 703 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 707 "go.y"
    {
		NodeList *node;
		
		node = mal(sizeof *node);
		node->n = (yyvsp[(2) - (2)].node);
		node->end = node;
		(yyval.list) = node;
	}
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 718 "go.y"
    {
		markdcl();
	}
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 722 "go.y"
    {
		Node *n;
		n = (yyvsp[(3) - (3)].node)->ntest;
		if(n != N && n->op != OTYPESW)
			n = N;
		typesw = nod(OXXX, typesw, n);
	}
    break;

  case 90:

/* Line 1806 of yacc.c  */
#line 730 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (7)].node);
		(yyval.node)->op = OSWITCH;
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
		typesw = typesw->left;
		popdcl();
	}
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 740 "go.y"
    {
		typesw = nod(OXXX, typesw, N);
	}
    break;

  case 92:

/* Line 1806 of yacc.c  */
#line 744 "go.y"
    {
		(yyval.node) = nod(OSELECT, N, N);
		(yyval.node)->lineno = typesw->lineno;
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		typesw = typesw->left;
	}
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 757 "go.y"
    {
		(yyval.node) = nod(OOROR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 761 "go.y"
    {
		(yyval.node) = nod(OANDAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 765 "go.y"
    {
		(yyval.node) = nod(OEQ, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 769 "go.y"
    {
		(yyval.node) = nod(ONE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 773 "go.y"
    {
		(yyval.node) = nod(OLT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 777 "go.y"
    {
		(yyval.node) = nod(OLE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 100:

/* Line 1806 of yacc.c  */
#line 781 "go.y"
    {
		(yyval.node) = nod(OGE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 785 "go.y"
    {
		(yyval.node) = nod(OGT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 102:

/* Line 1806 of yacc.c  */
#line 789 "go.y"
    {
		(yyval.node) = nod(OADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 103:

/* Line 1806 of yacc.c  */
#line 793 "go.y"
    {
		(yyval.node) = nod(OSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 104:

/* Line 1806 of yacc.c  */
#line 797 "go.y"
    {
		(yyval.node) = nod(OOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 105:

/* Line 1806 of yacc.c  */
#line 801 "go.y"
    {
		(yyval.node) = nod(OXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 106:

/* Line 1806 of yacc.c  */
#line 805 "go.y"
    {
		(yyval.node) = nod(OMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 107:

/* Line 1806 of yacc.c  */
#line 809 "go.y"
    {
		(yyval.node) = nod(ODIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 108:

/* Line 1806 of yacc.c  */
#line 813 "go.y"
    {
		(yyval.node) = nod(OMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 109:

/* Line 1806 of yacc.c  */
#line 817 "go.y"
    {
		(yyval.node) = nod(OAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 110:

/* Line 1806 of yacc.c  */
#line 821 "go.y"
    {
		(yyval.node) = nod(OANDNOT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 111:

/* Line 1806 of yacc.c  */
#line 825 "go.y"
    {
		(yyval.node) = nod(OLSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 112:

/* Line 1806 of yacc.c  */
#line 829 "go.y"
    {
		(yyval.node) = nod(ORSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 113:

/* Line 1806 of yacc.c  */
#line 834 "go.y"
    {
		(yyval.node) = nod(OSEND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 115:

/* Line 1806 of yacc.c  */
#line 841 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 116:

/* Line 1806 of yacc.c  */
#line 845 "go.y"
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

/* Line 1806 of yacc.c  */
#line 856 "go.y"
    {
		(yyval.node) = nod(OPLUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 118:

/* Line 1806 of yacc.c  */
#line 860 "go.y"
    {
		(yyval.node) = nod(OMINUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 119:

/* Line 1806 of yacc.c  */
#line 864 "go.y"
    {
		(yyval.node) = nod(ONOT, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 120:

/* Line 1806 of yacc.c  */
#line 868 "go.y"
    {
		yyerror("the bitwise complement operator is ^");
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 121:

/* Line 1806 of yacc.c  */
#line 873 "go.y"
    {
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 877 "go.y"
    {
		(yyval.node) = nod(ORECV, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 887 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (3)].node), N);
	}
    break;

  case 124:

/* Line 1806 of yacc.c  */
#line 891 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (5)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
	}
    break;

  case 125:

/* Line 1806 of yacc.c  */
#line 896 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (6)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (6)].list);
		(yyval.node)->isddd = 1;
	}
    break;

  case 126:

/* Line 1806 of yacc.c  */
#line 904 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 128:

/* Line 1806 of yacc.c  */
#line 909 "go.y"
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

/* Line 1806 of yacc.c  */
#line 920 "go.y"
    {
		(yyval.node) = nod(ODOTTYPE, (yyvsp[(1) - (5)].node), (yyvsp[(4) - (5)].node));
	}
    break;

  case 130:

/* Line 1806 of yacc.c  */
#line 924 "go.y"
    {
		(yyval.node) = nod(OTYPESW, N, (yyvsp[(1) - (5)].node));
	}
    break;

  case 131:

/* Line 1806 of yacc.c  */
#line 928 "go.y"
    {
		(yyval.node) = nod(OINDEX, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 132:

/* Line 1806 of yacc.c  */
#line 932 "go.y"
    {
		(yyval.node) = nod(OSLICE, (yyvsp[(1) - (6)].node), nod(OKEY, (yyvsp[(3) - (6)].node), (yyvsp[(5) - (6)].node)));
	}
    break;

  case 134:

/* Line 1806 of yacc.c  */
#line 937 "go.y"
    {
		// conversion
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (4)].node), N);
		(yyval.node)->list = list1((yyvsp[(3) - (4)].node));
	}
    break;

  case 135:

/* Line 1806 of yacc.c  */
#line 943 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 950 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
	}
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 956 "go.y"
    {
		yyerror("cannot parenthesize type in composite literal");
		(yyval.node) = (yyvsp[(5) - (7)].node);
		(yyval.node)->right = (yyvsp[(2) - (7)].node);
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
	}
    break;

  case 139:

/* Line 1806 of yacc.c  */
#line 965 "go.y"
    {
		// composite expression.
		// make node early so we get the right line number.
		(yyval.node) = nod(OCOMPLIT, N, N);
	}
    break;

  case 140:

/* Line 1806 of yacc.c  */
#line 973 "go.y"
    {
		(yyval.node) = nod(OKEY, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 141:

/* Line 1806 of yacc.c  */
#line 979 "go.y"
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
		}
	}
    break;

  case 142:

/* Line 1806 of yacc.c  */
#line 995 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (4)].node);
		(yyval.node)->list = (yyvsp[(3) - (4)].list);
	}
    break;

  case 144:

/* Line 1806 of yacc.c  */
#line 1003 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (4)].node);
		(yyval.node)->list = (yyvsp[(3) - (4)].list);
	}
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 1011 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1037 "go.y"
    {
		(yyval.i) = LBODY;
	}
    break;

  case 151:

/* Line 1806 of yacc.c  */
#line 1041 "go.y"
    {
		(yyval.i) = '{';
	}
    break;

  case 152:

/* Line 1806 of yacc.c  */
#line 1052 "go.y"
    {
		if((yyvsp[(1) - (1)].sym) == S)
			(yyval.node) = N;
		else
			(yyval.node) = newname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 153:

/* Line 1806 of yacc.c  */
#line 1061 "go.y"
    {
		(yyval.node) = dclname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 154:

/* Line 1806 of yacc.c  */
#line 1066 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 156:

/* Line 1806 of yacc.c  */
#line 1073 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if(importpkg != nil && !exportname((yyvsp[(1) - (1)].sym)->name))
			(yyval.sym) = pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg);
	}
    break;

  case 158:

/* Line 1806 of yacc.c  */
#line 1081 "go.y"
    {
		(yyval.sym) = S;
	}
    break;

  case 159:

/* Line 1806 of yacc.c  */
#line 1087 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1102 "go.y"
    {
		(yyval.node) = oldname((yyvsp[(1) - (1)].sym));
		if((yyval.node)->pack != N)
			(yyval.node)->pack->used = 1;
	}
    break;

  case 162:

/* Line 1806 of yacc.c  */
#line 1122 "go.y"
    {
		yyerror("final argument in variadic function missing type");
		(yyval.node) = nod(ODDD, typenod(typ(TINTER)), N);
	}
    break;

  case 163:

/* Line 1806 of yacc.c  */
#line 1127 "go.y"
    {
		(yyval.node) = nod(ODDD, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 169:

/* Line 1806 of yacc.c  */
#line 1138 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 173:

/* Line 1806 of yacc.c  */
#line 1147 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 178:

/* Line 1806 of yacc.c  */
#line 1157 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 188:

/* Line 1806 of yacc.c  */
#line 1178 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1191 "go.y"
    {
		(yyval.node) = nod(OTARRAY, (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].node));
	}
    break;

  case 190:

/* Line 1806 of yacc.c  */
#line 1195 "go.y"
    {
		// array literal of nelem
		(yyval.node) = nod(OTARRAY, nod(ODDD, N, N), (yyvsp[(4) - (4)].node));
	}
    break;

  case 191:

/* Line 1806 of yacc.c  */
#line 1200 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->etype = Cboth;
	}
    break;

  case 192:

/* Line 1806 of yacc.c  */
#line 1205 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Csend;
	}
    break;

  case 193:

/* Line 1806 of yacc.c  */
#line 1210 "go.y"
    {
		(yyval.node) = nod(OTMAP, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node));
	}
    break;

  case 196:

/* Line 1806 of yacc.c  */
#line 1218 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 197:

/* Line 1806 of yacc.c  */
#line 1224 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Crecv;
	}
    break;

  case 198:

/* Line 1806 of yacc.c  */
#line 1231 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 199:

/* Line 1806 of yacc.c  */
#line 1237 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 200:

/* Line 1806 of yacc.c  */
#line 1244 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 201:

/* Line 1806 of yacc.c  */
#line 1250 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 202:

/* Line 1806 of yacc.c  */
#line 1261 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
		if((yyval.node) == N)
			break;
		(yyval.node)->nbody = (yyvsp[(3) - (3)].list);
		(yyval.node)->endlineno = lineno;
		funcbody((yyval.node));
	}
    break;

  case 203:

/* Line 1806 of yacc.c  */
#line 1272 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1301 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1341 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1366 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1384 "go.y"
    {
		(yyvsp[(3) - (5)].list) = checkarglist((yyvsp[(3) - (5)].list), 1);
		(yyval.node) = nod(OTFUNC, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		(yyval.node)->rlist = (yyvsp[(5) - (5)].list);
	}
    break;

  case 208:

/* Line 1806 of yacc.c  */
#line 1392 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 209:

/* Line 1806 of yacc.c  */
#line 1396 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
		if((yyval.list) == nil)
			(yyval.list) = list1(nod(OEMPTY, N, N));
	}
    break;

  case 210:

/* Line 1806 of yacc.c  */
#line 1404 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 211:

/* Line 1806 of yacc.c  */
#line 1408 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, (yyvsp[(1) - (1)].node)));
	}
    break;

  case 212:

/* Line 1806 of yacc.c  */
#line 1412 "go.y"
    {
		(yyvsp[(2) - (3)].list) = checkarglist((yyvsp[(2) - (3)].list), 0);
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 213:

/* Line 1806 of yacc.c  */
#line 1419 "go.y"
    {
		closurehdr((yyvsp[(1) - (1)].node));
	}
    break;

  case 214:

/* Line 1806 of yacc.c  */
#line 1425 "go.y"
    {
		(yyval.node) = closurebody((yyvsp[(3) - (4)].list));
		fixlbrace((yyvsp[(2) - (4)].i));
	}
    break;

  case 215:

/* Line 1806 of yacc.c  */
#line 1430 "go.y"
    {
		(yyval.node) = closurebody(nil);
	}
    break;

  case 216:

/* Line 1806 of yacc.c  */
#line 1441 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 217:

/* Line 1806 of yacc.c  */
#line 1445 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(2) - (3)].list));
		if(nsyntaxerrors == 0)
			testdclstack();
		nointerface = 0;
	}
    break;

  case 219:

/* Line 1806 of yacc.c  */
#line 1455 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 221:

/* Line 1806 of yacc.c  */
#line 1462 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 222:

/* Line 1806 of yacc.c  */
#line 1468 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 223:

/* Line 1806 of yacc.c  */
#line 1472 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 225:

/* Line 1806 of yacc.c  */
#line 1479 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 226:

/* Line 1806 of yacc.c  */
#line 1485 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 227:

/* Line 1806 of yacc.c  */
#line 1489 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 228:

/* Line 1806 of yacc.c  */
#line 1495 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1518 "go.y"
    {
		(yyvsp[(1) - (2)].node)->val = (yyvsp[(2) - (2)].val);
		(yyval.list) = list1((yyvsp[(1) - (2)].node));
	}
    break;

  case 230:

/* Line 1806 of yacc.c  */
#line 1523 "go.y"
    {
		(yyvsp[(2) - (4)].node)->val = (yyvsp[(4) - (4)].val);
		(yyval.list) = list1((yyvsp[(2) - (4)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 231:

/* Line 1806 of yacc.c  */
#line 1529 "go.y"
    {
		(yyvsp[(2) - (3)].node)->right = nod(OIND, (yyvsp[(2) - (3)].node)->right, N);
		(yyvsp[(2) - (3)].node)->val = (yyvsp[(3) - (3)].val);
		(yyval.list) = list1((yyvsp[(2) - (3)].node));
	}
    break;

  case 232:

/* Line 1806 of yacc.c  */
#line 1535 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 233:

/* Line 1806 of yacc.c  */
#line 1542 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 234:

/* Line 1806 of yacc.c  */
#line 1551 "go.y"
    {
		Node *n;

		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		n = oldname((yyvsp[(1) - (1)].sym));
		if(n->pack != N)
			n->pack->used = 1;
	}
    break;

  case 235:

/* Line 1806 of yacc.c  */
#line 1560 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1575 "go.y"
    {
		(yyval.node) = embedded((yyvsp[(1) - (1)].sym));
	}
    break;

  case 237:

/* Line 1806 of yacc.c  */
#line 1581 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
		ifacedcl((yyval.node));
	}
    break;

  case 238:

/* Line 1806 of yacc.c  */
#line 1586 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(1) - (1)].sym)));
	}
    break;

  case 239:

/* Line 1806 of yacc.c  */
#line 1590 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(2) - (3)].sym)));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 240:

/* Line 1806 of yacc.c  */
#line 1597 "go.y"
    {
		// without func keyword
		(yyvsp[(2) - (4)].list) = checkarglist((yyvsp[(2) - (4)].list), 1);
		(yyval.node) = nod(OTFUNC, fakethis(), N);
		(yyval.node)->list = (yyvsp[(2) - (4)].list);
		(yyval.node)->rlist = (yyvsp[(4) - (4)].list);
	}
    break;

  case 242:

/* Line 1806 of yacc.c  */
#line 1611 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 243:

/* Line 1806 of yacc.c  */
#line 1617 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 245:

/* Line 1806 of yacc.c  */
#line 1626 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 246:

/* Line 1806 of yacc.c  */
#line 1630 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 247:

/* Line 1806 of yacc.c  */
#line 1635 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 248:

/* Line 1806 of yacc.c  */
#line 1639 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 249:

/* Line 1806 of yacc.c  */
#line 1647 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 251:

/* Line 1806 of yacc.c  */
#line 1652 "go.y"
    {
		(yyval.node) = liststmt((yyvsp[(1) - (1)].list));
	}
    break;

  case 253:

/* Line 1806 of yacc.c  */
#line 1657 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 259:

/* Line 1806 of yacc.c  */
#line 1668 "go.y"
    {
		(yyvsp[(1) - (2)].node) = nod(OLABEL, (yyvsp[(1) - (2)].node), N);
		(yyvsp[(1) - (2)].node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 260:

/* Line 1806 of yacc.c  */
#line 1673 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1683 "go.y"
    {
		// will be converted to OFALL
		(yyval.node) = nod(OXFALL, N, N);
	}
    break;

  case 262:

/* Line 1806 of yacc.c  */
#line 1688 "go.y"
    {
		(yyval.node) = nod(OBREAK, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 263:

/* Line 1806 of yacc.c  */
#line 1692 "go.y"
    {
		(yyval.node) = nod(OCONTINUE, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 264:

/* Line 1806 of yacc.c  */
#line 1696 "go.y"
    {
		(yyval.node) = nod(OPROC, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 265:

/* Line 1806 of yacc.c  */
#line 1700 "go.y"
    {
		(yyval.node) = nod(ODEFER, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 266:

/* Line 1806 of yacc.c  */
#line 1704 "go.y"
    {
		(yyval.node) = nod(OGOTO, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 267:

/* Line 1806 of yacc.c  */
#line 1709 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1728 "go.y"
    {
		(yyval.list) = nil;
		if((yyvsp[(1) - (1)].node) != N)
			(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 269:

/* Line 1806 of yacc.c  */
#line 1734 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (3)].list);
		if((yyvsp[(3) - (3)].node) != N)
			(yyval.list) = list((yyval.list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 270:

/* Line 1806 of yacc.c  */
#line 1742 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 271:

/* Line 1806 of yacc.c  */
#line 1746 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 272:

/* Line 1806 of yacc.c  */
#line 1752 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 273:

/* Line 1806 of yacc.c  */
#line 1756 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 274:

/* Line 1806 of yacc.c  */
#line 1762 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 275:

/* Line 1806 of yacc.c  */
#line 1766 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 276:

/* Line 1806 of yacc.c  */
#line 1772 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 277:

/* Line 1806 of yacc.c  */
#line 1776 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 278:

/* Line 1806 of yacc.c  */
#line 1785 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 279:

/* Line 1806 of yacc.c  */
#line 1789 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 280:

/* Line 1806 of yacc.c  */
#line 1793 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 281:

/* Line 1806 of yacc.c  */
#line 1797 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 282:

/* Line 1806 of yacc.c  */
#line 1802 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 283:

/* Line 1806 of yacc.c  */
#line 1806 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 288:

/* Line 1806 of yacc.c  */
#line 1820 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 290:

/* Line 1806 of yacc.c  */
#line 1826 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 292:

/* Line 1806 of yacc.c  */
#line 1832 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 294:

/* Line 1806 of yacc.c  */
#line 1838 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 296:

/* Line 1806 of yacc.c  */
#line 1844 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 298:

/* Line 1806 of yacc.c  */
#line 1850 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 300:

/* Line 1806 of yacc.c  */
#line 1856 "go.y"
    {
		(yyval.val).ctype = CTxxx;
	}
    break;

  case 302:

/* Line 1806 of yacc.c  */
#line 1866 "go.y"
    {
		importimport((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].val).u.sval);
	}
    break;

  case 303:

/* Line 1806 of yacc.c  */
#line 1870 "go.y"
    {
		importvar((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].type));
	}
    break;

  case 304:

/* Line 1806 of yacc.c  */
#line 1874 "go.y"
    {
		importconst((yyvsp[(2) - (5)].sym), types[TIDEAL], (yyvsp[(4) - (5)].node));
	}
    break;

  case 305:

/* Line 1806 of yacc.c  */
#line 1878 "go.y"
    {
		importconst((yyvsp[(2) - (6)].sym), (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].node));
	}
    break;

  case 306:

/* Line 1806 of yacc.c  */
#line 1882 "go.y"
    {
		importtype((yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].type));
	}
    break;

  case 307:

/* Line 1806 of yacc.c  */
#line 1886 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1906 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		structpkg = (yyval.sym)->pkg;
	}
    break;

  case 309:

/* Line 1806 of yacc.c  */
#line 1913 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
		importsym((yyvsp[(1) - (1)].sym), OTYPE);
	}
    break;

  case 315:

/* Line 1806 of yacc.c  */
#line 1933 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
	}
    break;

  case 316:

/* Line 1806 of yacc.c  */
#line 1937 "go.y"
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

/* Line 1806 of yacc.c  */
#line 1947 "go.y"
    {
		(yyval.type) = aindex(N, (yyvsp[(3) - (3)].type));
	}
    break;

  case 318:

/* Line 1806 of yacc.c  */
#line 1951 "go.y"
    {
		(yyval.type) = aindex(nodlit((yyvsp[(2) - (4)].val)), (yyvsp[(4) - (4)].type));
	}
    break;

  case 319:

/* Line 1806 of yacc.c  */
#line 1955 "go.y"
    {
		(yyval.type) = maptype((yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].type));
	}
    break;

  case 320:

/* Line 1806 of yacc.c  */
#line 1959 "go.y"
    {
		(yyval.type) = tostruct((yyvsp[(3) - (4)].list));
	}
    break;

  case 321:

/* Line 1806 of yacc.c  */
#line 1963 "go.y"
    {
		(yyval.type) = tointerface((yyvsp[(3) - (4)].list));
	}
    break;

  case 322:

/* Line 1806 of yacc.c  */
#line 1967 "go.y"
    {
		(yyval.type) = ptrto((yyvsp[(2) - (2)].type));
	}
    break;

  case 323:

/* Line 1806 of yacc.c  */
#line 1971 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(2) - (2)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 324:

/* Line 1806 of yacc.c  */
#line 1977 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (4)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 325:

/* Line 1806 of yacc.c  */
#line 1983 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Csend;
	}
    break;

  case 326:

/* Line 1806 of yacc.c  */
#line 1991 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Crecv;
	}
    break;

  case 327:

/* Line 1806 of yacc.c  */
#line 1999 "go.y"
    {
		(yyval.type) = functype(nil, (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list));
	}
    break;

  case 328:

/* Line 1806 of yacc.c  */
#line 2005 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(2) - (3)].type)));
		if((yyvsp[(1) - (3)].sym))
			(yyval.node)->left = newname((yyvsp[(1) - (3)].sym));
		(yyval.node)->val = (yyvsp[(3) - (3)].val);
	}
    break;

  case 329:

/* Line 1806 of yacc.c  */
#line 2012 "go.y"
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

/* Line 1806 of yacc.c  */
#line 2028 "go.y"
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

/* Line 1806 of yacc.c  */
#line 2046 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, newname((yyvsp[(1) - (5)].sym)), typenod(functype(fakethis(), (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list))));
	}
    break;

  case 332:

/* Line 1806 of yacc.c  */
#line 2050 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type)));
	}
    break;

  case 333:

/* Line 1806 of yacc.c  */
#line 2055 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 335:

/* Line 1806 of yacc.c  */
#line 2062 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 336:

/* Line 1806 of yacc.c  */
#line 2066 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type))));
	}
    break;

  case 337:

/* Line 1806 of yacc.c  */
#line 2076 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 338:

/* Line 1806 of yacc.c  */
#line 2080 "go.y"
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

/* Line 1806 of yacc.c  */
#line 2095 "go.y"
    {
		(yyval.node) = oldname(pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg));
		if((yyval.node)->op != OLITERAL)
			yyerror("bad constant %S", (yyval.node)->sym);
	}
    break;

  case 341:

/* Line 1806 of yacc.c  */
#line 2104 "go.y"
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

/* Line 1806 of yacc.c  */
#line 2120 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 345:

/* Line 1806 of yacc.c  */
#line 2124 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 346:

/* Line 1806 of yacc.c  */
#line 2130 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 347:

/* Line 1806 of yacc.c  */
#line 2134 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 348:

/* Line 1806 of yacc.c  */
#line 2140 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 349:

/* Line 1806 of yacc.c  */
#line 2144 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;



/* Line 1806 of yacc.c  */
#line 5398 "y.tab.c"
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
#line 2148 "go.y"


static void
fixlbrace(int lbr)
{
	// If the opening brace was an LBODY,
	// set up for another one now that we're done.
	// See comment in lex.c about loophack.
	if(lbr == LBODY)
		loophack = 1;
}


