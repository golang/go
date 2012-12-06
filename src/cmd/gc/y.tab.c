/* A Bison parser, made by GNU Bison 2.6.5.  */

/* Bison implementation for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2012 Free Software Foundation, Inc.
   
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
#define YYBISON_VERSION "2.6.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
/* Line 360 of yacc.c  */
#line 20 "go.y"

#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and go.h re-#defines getc */
#include <libc.h>
#include "go.h"

static void fixlbrace(int);

/* Line 360 of yacc.c  */
#line 77 "y.tab.c"

# ifndef YY_NULL
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULL nullptr
#  else
#   define YY_NULL 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 1
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
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
/* Line 376 of yacc.c  */
#line 28 "go.y"

	Node*		node;
	NodeList*		list;
	Type*		type;
	Sym*		sym;
	struct	Val	val;
	int		i;


/* Line 376 of yacc.c  */
#line 232 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;

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

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

/* Line 379 of yacc.c  */
#line 260 "y.tab.c"

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
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(N) (N)
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
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
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
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (YYID (0))
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  4
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   2049

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  76
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  141
/* YYNRULES -- Number of rules.  */
#define YYNRULES  347
/* YYNRULES -- Number of states.  */
#define YYNSTATES  658

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
     487,   491,   493,   498,   500,   504,   506,   508,   510,   512,
     514,   516,   518,   519,   521,   523,   525,   527,   532,   534,
     536,   538,   541,   543,   545,   547,   549,   551,   555,   557,
     559,   561,   564,   566,   568,   570,   572,   576,   578,   580,
     582,   584,   586,   588,   590,   592,   594,   598,   603,   608,
     611,   615,   621,   623,   625,   628,   632,   638,   642,   648,
     652,   656,   662,   671,   677,   686,   692,   693,   697,   698,
     700,   704,   706,   711,   714,   715,   719,   721,   725,   727,
     731,   733,   737,   739,   743,   745,   749,   753,   756,   761,
     765,   771,   777,   779,   783,   785,   788,   790,   794,   799,
     801,   804,   807,   809,   811,   815,   816,   819,   820,   822,
     824,   826,   828,   830,   832,   834,   836,   838,   839,   844,
     846,   849,   852,   855,   858,   861,   864,   866,   870,   872,
     876,   878,   882,   884,   888,   890,   894,   896,   898,   902,
     906,   907,   910,   911,   913,   914,   916,   917,   919,   920,
     922,   923,   925,   926,   928,   929,   931,   932,   934,   935,
     937,   942,   947,   953,   960,   965,   970,   972,   974,   976,
     978,   980,   982,   984,   986,   988,   992,   997,  1003,  1008,
    1013,  1016,  1019,  1024,  1028,  1032,  1038,  1042,  1047,  1051,
    1057,  1059,  1060,  1062,  1066,  1068,  1070,  1073,  1075,  1077,
    1083,  1084,  1087,  1089,  1093,  1095,  1099,  1101
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      77,     0,    -1,    79,    78,    81,   165,    -1,    -1,    25,
     140,    62,    -1,    -1,    80,    86,    88,    -1,    -1,    81,
      82,    62,    -1,    21,    83,    -1,    21,    59,    84,   189,
      60,    -1,    21,    59,    60,    -1,    85,    86,    88,    -1,
      85,    88,    -1,    83,    -1,    84,    62,    83,    -1,     3,
      -1,   140,     3,    -1,    63,     3,    -1,    25,    24,    87,
      62,    -1,    -1,    24,    -1,    -1,    89,   213,    64,    64,
      -1,    -1,    91,    -1,   157,    -1,   180,    -1,     1,    -1,
      32,    93,    -1,    32,    59,   166,   189,    60,    -1,    32,
      59,    60,    -1,    92,    94,    -1,    92,    59,    94,   189,
      60,    -1,    92,    59,    94,    62,   167,   189,    60,    -1,
      92,    59,    60,    -1,    31,    97,    -1,    31,    59,   168,
     189,    60,    -1,    31,    59,    60,    -1,     9,    -1,   184,
     145,    -1,   184,   145,    65,   185,    -1,   184,    65,   185,
      -1,   184,   145,    65,   185,    -1,   184,    65,   185,    -1,
      94,    -1,   184,   145,    -1,   184,    -1,   140,    -1,    96,
     145,    -1,   126,    -1,   126,     4,   126,    -1,   185,    65,
     185,    -1,   185,     5,   185,    -1,   126,    42,    -1,   126,
      37,    -1,     7,   186,    66,    -1,     7,   186,    65,   126,
      66,    -1,     7,   186,     5,   126,    66,    -1,    12,    66,
      -1,    -1,    67,   101,   182,    68,    -1,    -1,    99,   103,
     182,    -1,    -1,   104,   102,    -1,    -1,    35,   106,   182,
      68,    -1,   185,    65,    26,   126,    -1,   185,     5,    26,
     126,    -1,   193,    62,   193,    62,   193,    -1,   193,    -1,
     107,    -1,   108,   105,    -1,    -1,    16,   111,   109,    -1,
     193,    -1,   193,    62,   193,    -1,    -1,    -1,    -1,    20,
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
     126,    -1,   133,    -1,    53,   127,    -1,    56,   127,    -1,
      49,   127,    -1,    50,   127,    -1,    69,   127,    -1,    70,
     127,    -1,    52,   127,    -1,    36,   127,    -1,   133,    59,
      60,    -1,   133,    59,   186,   190,    60,    -1,   133,    59,
     186,    11,   190,    60,    -1,     3,    -1,   142,    -1,   133,
      63,   140,    -1,   133,    63,    59,   134,    60,    -1,   133,
      63,    59,    31,    60,    -1,   133,    71,   126,    72,    -1,
     133,    71,   191,    66,   191,    72,    -1,   128,    -1,   148,
      59,   126,    60,    -1,   149,   136,   130,   188,    68,    -1,
     129,    67,   130,   188,    68,    -1,    59,   134,    60,    67,
     130,   188,    68,    -1,   164,    -1,    -1,   126,    66,   132,
      -1,   126,    -1,    67,   130,   188,    68,    -1,   129,    -1,
      59,   134,    60,    -1,   126,    -1,   146,    -1,   145,    -1,
      35,    -1,    67,    -1,   140,    -1,   140,    -1,    -1,   137,
      -1,    24,    -1,   141,    -1,    73,    -1,    74,     3,    63,
      24,    -1,   140,    -1,   137,    -1,    11,    -1,    11,   145,
      -1,   154,    -1,   160,    -1,   152,    -1,   153,    -1,   151,
      -1,    59,   145,    60,    -1,   154,    -1,   160,    -1,   152,
      -1,    53,   146,    -1,   160,    -1,   152,    -1,   153,    -1,
     151,    -1,    59,   145,    60,    -1,   160,    -1,   152,    -1,
     152,    -1,   154,    -1,   160,    -1,   152,    -1,   153,    -1,
     151,    -1,   142,    -1,   142,    63,   140,    -1,    71,   191,
      72,   145,    -1,    71,    11,    72,   145,    -1,     8,   147,
      -1,     8,    36,   145,    -1,    23,    71,   145,    72,   145,
      -1,   155,    -1,   156,    -1,    53,   145,    -1,    36,     8,
     145,    -1,    29,   136,   169,   189,    68,    -1,    29,   136,
      68,    -1,    22,   136,   170,   189,    68,    -1,    22,   136,
      68,    -1,    17,   158,   161,    -1,   140,    59,   178,    60,
     162,    -1,    59,   178,    60,   140,    59,   178,    60,   162,
      -1,   199,    59,   194,    60,   209,    -1,    59,   214,    60,
     140,    59,   194,    60,   209,    -1,    17,    59,   178,    60,
     162,    -1,    -1,    67,   182,    68,    -1,    -1,   150,    -1,
      59,   178,    60,    -1,   160,    -1,   163,   136,   182,    68,
      -1,   163,     1,    -1,    -1,   165,    90,    62,    -1,    93,
      -1,   166,    62,    93,    -1,    95,    -1,   167,    62,    95,
      -1,    97,    -1,   168,    62,    97,    -1,   171,    -1,   169,
      62,   171,    -1,   174,    -1,   170,    62,   174,    -1,   183,
     145,   197,    -1,   173,   197,    -1,    59,   173,    60,   197,
      -1,    53,   173,   197,    -1,    59,    53,   173,    60,   197,
      -1,    53,    59,   173,    60,   197,    -1,    24,    -1,    24,
      63,   140,    -1,   172,    -1,   137,   175,    -1,   172,    -1,
      59,   172,    60,    -1,    59,   178,    60,   162,    -1,   135,
      -1,   140,   135,    -1,   140,   144,    -1,   144,    -1,   176,
      -1,   177,    75,   176,    -1,    -1,   177,   190,    -1,    -1,
     100,    -1,    91,    -1,   180,    -1,     1,    -1,    98,    -1,
     110,    -1,   121,    -1,   124,    -1,   113,    -1,    -1,   143,
      66,   181,   179,    -1,    15,    -1,     6,   139,    -1,    10,
     139,    -1,    18,   128,    -1,    13,   128,    -1,    19,   137,
      -1,    27,   192,    -1,   179,    -1,   182,    62,   179,    -1,
     137,    -1,   183,    75,   137,    -1,   138,    -1,   184,    75,
     138,    -1,   126,    -1,   185,    75,   126,    -1,   134,    -1,
     186,    75,   134,    -1,   131,    -1,   132,    -1,   187,    75,
     131,    -1,   187,    75,   132,    -1,    -1,   187,   190,    -1,
      -1,    62,    -1,    -1,    75,    -1,    -1,   126,    -1,    -1,
     185,    -1,    -1,    98,    -1,    -1,   214,    -1,    -1,   215,
      -1,    -1,   216,    -1,    -1,     3,    -1,    21,    24,     3,
      62,    -1,    32,   199,   201,    62,    -1,     9,   199,    65,
     212,    62,    -1,     9,   199,   201,    65,   212,    62,    -1,
      31,   200,   201,    62,    -1,    17,   159,   161,    62,    -1,
     141,    -1,   199,    -1,   203,    -1,   204,    -1,   205,    -1,
     203,    -1,   205,    -1,   141,    -1,    24,    -1,    71,    72,
     201,    -1,    71,     3,    72,   201,    -1,    23,    71,   201,
      72,   201,    -1,    29,    67,   195,    68,    -1,    22,    67,
     196,    68,    -1,    53,   201,    -1,     8,   202,    -1,     8,
      59,   204,    60,    -1,     8,    36,   201,    -1,    36,     8,
     201,    -1,    17,    59,   194,    60,   209,    -1,   140,   201,
     197,    -1,   140,    11,   201,   197,    -1,   140,   201,   197,
      -1,   140,    59,   194,    60,   209,    -1,   201,    -1,    -1,
     210,    -1,    59,   194,    60,    -1,   201,    -1,     3,    -1,
      50,     3,    -1,   140,    -1,   211,    -1,    59,   211,    49,
     211,    60,    -1,    -1,   213,   198,    -1,   206,    -1,   214,
      75,   206,    -1,   207,    -1,   215,    62,   207,    -1,   208,
      -1,   216,    62,   208,    -1
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
     972,   978,   979,   986,   987,  1005,  1006,  1009,  1012,  1016,
    1027,  1036,  1042,  1045,  1048,  1055,  1056,  1062,  1077,  1085,
    1097,  1102,  1108,  1109,  1110,  1111,  1112,  1113,  1119,  1120,
    1121,  1122,  1128,  1129,  1130,  1131,  1132,  1138,  1139,  1142,
    1145,  1146,  1147,  1148,  1149,  1152,  1153,  1166,  1170,  1175,
    1180,  1185,  1189,  1190,  1193,  1199,  1206,  1212,  1219,  1225,
    1236,  1247,  1276,  1316,  1341,  1359,  1368,  1371,  1379,  1383,
    1387,  1394,  1400,  1405,  1417,  1420,  1429,  1430,  1436,  1437,
    1443,  1447,  1453,  1454,  1460,  1464,  1470,  1493,  1498,  1504,
    1510,  1517,  1526,  1535,  1550,  1556,  1561,  1565,  1572,  1585,
    1586,  1592,  1598,  1601,  1605,  1611,  1614,  1623,  1626,  1627,
    1631,  1632,  1638,  1639,  1640,  1641,  1642,  1644,  1643,  1658,
    1663,  1667,  1671,  1675,  1679,  1684,  1703,  1709,  1717,  1721,
    1727,  1731,  1737,  1741,  1747,  1751,  1760,  1764,  1768,  1772,
    1778,  1781,  1789,  1790,  1792,  1793,  1796,  1799,  1802,  1805,
    1808,  1811,  1814,  1817,  1820,  1823,  1826,  1829,  1832,  1835,
    1841,  1845,  1849,  1853,  1857,  1861,  1881,  1888,  1899,  1900,
    1901,  1904,  1905,  1908,  1912,  1922,  1926,  1930,  1934,  1938,
    1942,  1946,  1952,  1958,  1966,  1974,  1980,  1987,  2003,  2021,
    2025,  2031,  2034,  2037,  2041,  2051,  2055,  2070,  2078,  2079,
    2091,  2092,  2095,  2099,  2105,  2109,  2115,  2119
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 1
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
  "keyval", "complitexpr", "pexpr", "expr_or_type", "name_or_type",
  "lbrace", "new_name", "dcl_name", "onew_name", "sym", "hidden_importsym",
  "name", "labelname", "dotdotdot", "ntype", "non_expr_type",
  "non_recvchantype", "convtype", "comptype", "fnret_type", "dotname",
  "othertype", "ptrtype", "recvchantype", "structtype", "interfacetype",
  "xfndcl", "fndcl", "hidden_fndcl", "fntype", "fnbody", "fnres",
  "fnlitdcl", "fnliteral", "xdcl_list", "vardcl_list", "constdcl_list",
  "typedcl_list", "structdcl_list", "interfacedcl_list", "structdcl",
  "packname", "embed", "interfacedcl", "indcl", "arg_type",
  "arg_type_list", "oarg_type_list_ocomma", "stmt", "non_dcl_stmt", "$@14",
  "stmt_list", "new_name_list", "dcl_name_list", "expr_list",
  "expr_or_type_list", "keyval_list", "braced_keyval_list", "osemi",
  "ocomma", "oexpr", "oexpr_list", "osimple_stmt", "ohidden_funarg_list",
  "ohidden_structdcl_list", "ohidden_interfacedcl_list", "oliteral",
  "hidden_import", "hidden_pkg_importsym", "hidden_pkgtype", "hidden_type",
  "hidden_type_non_recv_chan", "hidden_type_misc", "hidden_type_recv_chan",
  "hidden_type_func", "hidden_funarg", "hidden_structdcl",
  "hidden_interfacedcl", "ohidden_funres", "hidden_funres",
  "hidden_literal", "hidden_constant", "hidden_import_list",
  "hidden_funarg_list", "hidden_structdcl_list",
  "hidden_interfacedcl_list", YY_NULL
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
     131,   132,   132,   133,   133,   134,   134,   135,   136,   136,
     137,   138,   139,   139,   140,   140,   140,   141,   142,   143,
     144,   144,   145,   145,   145,   145,   145,   145,   146,   146,
     146,   146,   147,   147,   147,   147,   147,   148,   148,   149,
     150,   150,   150,   150,   150,   151,   151,   152,   152,   152,
     152,   152,   152,   152,   153,   154,   155,   155,   156,   156,
     157,   158,   158,   159,   159,   160,   161,   161,   162,   162,
     162,   163,   164,   164,   165,   165,   166,   166,   167,   167,
     168,   168,   169,   169,   170,   170,   171,   171,   171,   171,
     171,   171,   172,   172,   173,   174,   174,   174,   175,   176,
     176,   176,   176,   177,   177,   178,   178,   179,   179,   179,
     179,   179,   180,   180,   180,   180,   180,   181,   180,   180,
     180,   180,   180,   180,   180,   180,   182,   182,   183,   183,
     184,   184,   185,   185,   186,   186,   187,   187,   187,   187,
     188,   188,   189,   189,   190,   190,   191,   191,   192,   192,
     193,   193,   194,   194,   195,   195,   196,   196,   197,   197,
     198,   198,   198,   198,   198,   198,   199,   200,   201,   201,
     201,   202,   202,   203,   203,   203,   203,   203,   203,   203,
     203,   203,   203,   203,   204,   205,   206,   206,   207,   208,
     208,   209,   209,   210,   210,   211,   211,   211,   212,   212,
     213,   213,   214,   214,   215,   215,   216,   216
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
       3,     1,     4,     1,     3,     1,     1,     1,     1,     1,
       1,     1,     0,     1,     1,     1,     1,     4,     1,     1,
       1,     2,     1,     1,     1,     1,     1,     3,     1,     1,
       1,     2,     1,     1,     1,     1,     3,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     4,     4,     2,
       3,     5,     1,     1,     2,     3,     5,     3,     5,     3,
       3,     5,     8,     5,     8,     5,     0,     3,     0,     1,
       3,     1,     4,     2,     0,     3,     1,     3,     1,     3,
       1,     3,     1,     3,     1,     3,     3,     2,     4,     3,
       5,     5,     1,     3,     1,     2,     1,     3,     4,     1,
       2,     2,     1,     1,     3,     0,     2,     0,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     0,     4,     1,
       2,     2,     2,     2,     2,     2,     1,     3,     1,     3,
       1,     3,     1,     3,     1,     3,     1,     1,     3,     3,
       0,     2,     0,     1,     0,     1,     0,     1,     0,     1,
       0,     1,     0,     1,     0,     1,     0,     1,     0,     1,
       4,     4,     5,     6,     4,     4,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     3,     4,     5,     4,     4,
       2,     2,     4,     3,     3,     5,     3,     4,     3,     5,
       1,     0,     1,     3,     1,     1,     2,     1,     1,     5,
       0,     2,     1,     3,     1,     3,     1,     3
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       5,     0,     3,     0,     1,     0,     7,     0,    22,   154,
     156,     0,     0,   155,   214,    20,     6,   340,     0,     4,
       0,     0,     0,    21,     0,     0,     0,    16,     0,     0,
       9,    22,     0,     8,    28,   126,   152,     0,    39,   152,
       0,   259,    74,     0,     0,     0,    78,     0,     0,   288,
      91,     0,    88,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   286,     0,    25,     0,   252,   253,
     256,   254,   255,    50,    93,   133,   143,   114,   159,   158,
     127,     0,     0,     0,   179,   192,   193,    26,   211,     0,
     138,    27,     0,    19,     0,     0,     0,     0,     0,     0,
     341,   157,    11,    14,   282,    18,    22,    13,    17,   153,
     260,   150,     0,     0,     0,     0,   158,   185,   189,   175,
     173,   174,   172,   261,   133,     0,   290,   245,     0,   206,
     133,   264,   290,   148,   149,     0,     0,   272,   289,   265,
       0,     0,   290,     0,     0,    36,    48,     0,    29,   270,
     151,     0,   122,   117,   118,   121,   115,   116,     0,     0,
     145,     0,   146,   170,   168,   169,   119,   120,     0,   287,
       0,   215,     0,    32,     0,     0,     0,     0,     0,    55,
       0,     0,     0,    54,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   139,     0,
       0,   286,   257,     0,   139,   213,     0,     0,     0,     0,
     306,     0,     0,   206,     0,     0,   307,     0,     0,    23,
     283,     0,    12,   245,     0,     0,   190,   166,   164,   165,
     162,   163,   194,     0,     0,   291,    72,     0,    75,     0,
      71,   160,   239,   158,   242,   147,   243,   284,     0,   245,
       0,   200,    79,    76,   154,     0,   199,     0,   282,   236,
     224,     0,    64,     0,     0,   197,   268,   282,   222,   234,
     298,     0,    89,    38,   220,   282,    49,    31,   216,   282,
       0,     0,    40,     0,   171,   144,     0,     0,    35,   282,
       0,     0,    51,    95,   110,   113,    96,   100,   101,    99,
     111,    98,    97,    94,   112,   102,   103,   104,   105,   106,
     107,   108,   109,   280,   123,   274,   284,     0,   128,   287,
       0,     0,     0,   280,   251,    60,   249,   248,   266,   250,
       0,    53,    52,   273,     0,     0,     0,     0,   314,     0,
       0,     0,     0,     0,   313,     0,   308,   309,   310,     0,
     342,     0,     0,   292,     0,     0,     0,    15,    10,     0,
       0,     0,   176,   186,    66,    73,     0,     0,   290,   161,
     240,   241,   285,   246,   208,     0,     0,     0,   290,     0,
     232,     0,   245,   235,   283,     0,     0,     0,     0,   298,
       0,     0,   283,     0,   299,   227,     0,   298,     0,   283,
       0,   283,     0,    42,   271,     0,     0,     0,   195,   166,
     164,   165,   163,   139,   188,   187,   283,     0,    44,     0,
     139,   141,   276,   277,   284,     0,   284,   285,     0,     0,
       0,   131,   286,   258,   134,     0,     0,     0,   212,     0,
       0,   321,   311,   312,   292,   296,     0,   294,     0,   320,
     335,     0,     0,   337,   338,     0,     0,     0,     0,     0,
     298,     0,     0,   305,     0,   293,   300,   304,   301,   208,
     167,     0,     0,     0,     0,   244,   245,   158,   209,   184,
     182,   183,   180,   181,   205,   208,   207,    80,    77,   233,
     237,     0,   225,   198,   191,     0,     0,    92,    62,    65,
       0,   229,     0,   298,   223,   196,   269,   226,    64,   221,
      37,   217,    30,    41,     0,   280,    45,   218,   282,    47,
      33,    43,   280,     0,   285,   281,   136,   285,     0,   275,
     124,   130,   129,     0,   135,     0,   267,   323,     0,     0,
     314,     0,   313,     0,   330,   346,   297,     0,     0,     0,
     344,   295,   324,   336,     0,   302,     0,   315,     0,   298,
     326,     0,   343,   331,     0,    69,    68,   290,     0,   245,
     201,    84,   208,     0,    59,     0,   298,   298,   228,     0,
     167,     0,   283,     0,    46,     0,   141,   140,   278,   279,
     125,   132,    61,   322,   331,   292,   319,     0,     0,   298,
     318,     0,     0,   316,   303,   327,   292,   292,   334,   203,
     332,    67,    70,   210,     0,    86,   238,     0,     0,    56,
       0,    63,   231,   230,    90,   137,   219,    34,   142,   325,
       0,   347,   317,   328,   345,     0,     0,     0,   208,     0,
      85,    81,     0,     0,   331,   339,   331,   333,   202,    82,
      87,    58,    57,   329,   204,   290,     0,    83
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     6,     2,     3,    14,    21,    30,   104,    31,
       8,    24,    16,    17,    65,   326,    67,   148,   516,   517,
     144,   145,    68,   498,   327,   436,   499,   575,   387,   365,
     471,   236,   237,   238,    69,   126,   252,    70,   132,   377,
     571,   640,   655,   615,   641,    71,   142,   398,    72,   140,
      73,    74,    75,    76,   313,   422,   423,    77,   315,   242,
     135,    78,   149,   110,   116,    13,    80,    81,   244,   245,
     162,   118,    82,    83,   478,   227,    84,   229,   230,    85,
      86,    87,   129,   213,    88,   251,   484,    89,    90,    22,
     279,   518,   275,   267,   258,   268,   269,   270,   260,   383,
     246,   247,   248,   328,   329,   321,   330,   271,   151,    92,
     316,   424,   425,   221,   373,   170,   139,   253,   464,   549,
     543,   395,   100,   211,   217,   608,   441,   346,   347,   348,
     350,   550,   545,   609,   610,   454,   455,    25,   465,   551,
     546
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -459
static const yytype_int16 yypact[] =
{
    -459,    55,    46,    53,  -459,   181,  -459,    44,  -459,  -459,
    -459,    91,    20,  -459,    75,   122,  -459,  -459,    97,  -459,
     472,   124,  1062,  -459,   129,   276,   153,  -459,   228,   190,
    -459,    53,   200,  -459,  -459,  -459,   181,  1638,  -459,   181,
      66,  -459,  -459,   197,    66,   181,  -459,    30,   142,  1475,
    -459,    30,  -459,   444,   467,  1475,  1475,  1475,  1475,  1475,
    1475,  1518,  1475,  1475,   638,   163,  -459,   485,  -459,  -459,
    -459,  -459,  -459,   830,  -459,  -459,   172,   240,  -459,   187,
    -459,   192,   201,    30,   213,  -459,  -459,  -459,   218,    65,
    -459,  -459,    40,  -459,   204,     5,   271,   204,   204,   241,
    -459,  -459,  -459,  -459,   238,  -459,  -459,  -459,  -459,  -459,
    -459,  -459,   251,  1663,  1663,  1663,  -459,   254,  -459,  -459,
    -459,  -459,  -459,  -459,    79,   240,  1475,  1630,   261,   246,
     236,  -459,  1475,  -459,  -459,   413,  1663,   804,   248,  -459,
     301,   662,  1475,   508,  1663,  -459,  -459,   551,  -459,  -459,
    -459,   477,  -459,  -459,  -459,  -459,  -459,  -459,  1561,  1518,
     804,   278,  -459,    17,  -459,    42,  -459,  -459,   270,   804,
     279,  -459,   561,  -459,  1604,  1475,  1475,  1475,  1475,  -459,
    1475,  1475,  1475,  -459,  1475,  1475,  1475,  1475,  1475,  1475,
    1475,  1475,  1475,  1475,  1475,  1475,  1475,  1475,  -459,   946,
     564,  1475,  -459,  1475,  -459,  -459,  1212,  1475,  1475,  1475,
    -459,   335,   181,   246,   295,   367,  -459,  1832,  1832,  -459,
      29,   322,  -459,  1630,   376,  1663,  -459,  -459,  -459,  -459,
    -459,  -459,  -459,   332,   181,  -459,  -459,   360,  -459,   106,
     339,  1663,  -459,  1630,  -459,  -459,  -459,   338,   354,  1630,
    1212,  -459,  -459,   356,   170,   395,  -459,   362,   361,  -459,
    -459,   358,  -459,    22,   125,  -459,  -459,   364,  -459,  -459,
     424,   884,  -459,  -459,  -459,   366,  -459,  -459,  -459,   371,
    1475,   181,   370,  1697,  -459,   374,  1663,  1663,  -459,   400,
    1475,   373,   804,  1993,  -459,  1946,   690,   690,   690,   690,
    -459,   690,   690,  1970,  -459,   751,   751,   751,   751,  -459,
    -459,  -459,  -459,  1267,  -459,  -459,    52,  1322,  -459,  1843,
     397,  1138,  1922,  1267,  -459,  -459,  -459,  -459,  -459,  -459,
      -8,   248,   248,   804,  1755,   405,   416,   417,  -459,   422,
     487,  1832,   419,    54,  -459,   433,  -459,  -459,  -459,   648,
    -459,    10,   440,   181,   442,   443,   446,  -459,  -459,   450,
    1663,   451,  -459,  -459,  -459,  -459,  1377,  1432,  1475,  -459,
    -459,  -459,  1630,  -459,  1722,   461,   128,   360,  1475,   181,
     459,   464,  1630,  -459,   580,   457,  1663,    60,   395,   424,
     395,   469,   259,   475,  -459,  -459,   181,   424,   492,   181,
     489,   181,   493,   248,  -459,  1475,  1730,  1663,  -459,   309,
     314,   326,   330,  -459,  -459,  -459,   181,   495,   248,  1475,
    -459,   991,  -459,  -459,   462,   479,   481,  1518,   500,   503,
     514,  -459,  1475,  -459,  -459,   510,  1212,  1138,  -459,  1832,
     540,  -459,  -459,  -459,   181,  1789,  1832,   181,  1832,  -459,
    -459,   576,   272,  -459,  -459,   518,   511,  1832,   419,  1832,
     424,   181,   181,  -459,   524,   516,  -459,  -459,  -459,  1722,
    -459,  1212,  1475,  1475,   532,  -459,  1630,   527,  -459,  -459,
    -459,  -459,  -459,  -459,  -459,  1722,  -459,  -459,  -459,  -459,
    -459,   538,  -459,  -459,  -459,  1518,   537,  -459,  -459,  -459,
     546,  -459,   548,   424,  -459,  -459,  -459,  -459,  -459,  -459,
    -459,  -459,  -459,   248,   553,  1267,  -459,  -459,   552,  1604,
    -459,   248,  1267,  1267,  1267,  -459,  -459,  -459,   556,  -459,
    -459,  -459,  -459,   547,  -459,   146,  -459,  -459,   558,   562,
     567,   569,   572,   568,  -459,  -459,   578,   571,  1832,   579,
    -459,   582,  -459,  -459,   599,  -459,  1832,  -459,   588,   424,
    -459,   592,  -459,  1799,   168,   804,   804,  1475,   597,  1630,
    -459,  -459,  1722,    68,  -459,  1138,   424,   424,  -459,    86,
     340,   584,   181,   604,   373,   598,   804,  -459,  -459,  -459,
    -459,  -459,  -459,  -459,  1799,   181,  -459,  1789,  1832,   424,
    -459,   181,   272,  -459,  -459,  -459,   181,   181,  -459,  -459,
    -459,  -459,  -459,  -459,   608,   655,  -459,  1475,  1475,  -459,
    1518,   611,  -459,  -459,  -459,  -459,  -459,  -459,  -459,  -459,
     615,  -459,  -459,  -459,  -459,   618,   619,   620,  1722,    41,
    -459,  -459,  1874,  1898,  1799,  -459,  1799,  -459,  -459,  -459,
    -459,  -459,  -459,  -459,  -459,  1475,   360,  -459
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -459,  -459,  -459,  -459,  -459,  -459,  -459,    -4,  -459,  -459,
     651,  -459,     4,  -459,  -459,   661,  -459,  -134,   -41,   103,
    -459,  -135,  -121,  -459,    50,  -459,  -459,  -459,   184,  -374,
    -459,  -459,  -459,  -459,  -459,  -459,  -140,  -459,  -459,  -459,
    -459,  -459,  -459,  -459,  -459,  -459,  -459,  -459,  -459,  -459,
     581,    61,   115,  -459,  -186,   169,  -411,   250,   -55,   452,
     264,    -6,   418,   659,    -5,   382,   346,  -459,   460,    48,
     541,  -459,  -459,  -459,  -459,   -33,    38,   -18,   -11,  -459,
    -459,  -459,  -459,  -459,    43,   497,  -458,  -459,  -459,  -459,
    -459,  -459,  -459,  -459,  -459,   310,  -110,  -205,   321,  -459,
     341,  -459,  -207,  -293,   692,  -459,  -236,  -459,   -66,   -39,
     222,  -459,  -311,  -238,  -260,  -192,  -459,  -119,  -397,  -459,
    -459,  -353,  -459,   307,  -459,   355,  -459,   391,   286,   393,
     267,   132,   140,  -418,  -459,  -430,   289,  -459,   536,  -459,
    -459
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -273
static const yytype_int16 yytable[] =
{
      12,   174,   272,   487,   119,   235,   161,   240,   274,   320,
     138,   235,   435,   278,   376,    32,   359,    79,   323,   121,
     385,   235,   554,    32,   103,   259,   173,   570,   433,   393,
     109,   111,    27,   109,   111,   107,   501,   400,   128,   131,
     111,   402,   375,  -211,   507,   207,   380,   539,   146,   150,
     164,   417,  -179,     9,   437,     4,   428,   456,   389,   391,
     438,   649,   150,   426,   212,   133,   205,   495,    15,    35,
     461,     5,   496,   617,    37,   120,  -178,  -211,     7,    11,
     122,   388,    19,   112,  -179,   462,  -263,   239,    47,    48,
       9,  -263,    29,   495,    18,    51,    20,   134,   496,   163,
     133,  -177,    10,    11,   165,   208,   174,   560,   325,  -211,
     222,   366,   587,   589,   616,   209,   152,   153,   154,   155,
     156,   157,   243,   166,   167,    61,   457,   427,   497,   257,
     111,   289,   134,   618,   619,   266,   111,    64,   146,    10,
      11,  -263,   150,   620,   536,   381,    23,  -263,   164,   380,
     578,   228,   228,   228,   624,   124,   231,   231,   231,   130,
      26,   226,   232,   233,   525,   228,   528,   150,   331,   332,
     231,   367,   635,  -232,   228,   491,   629,   101,   390,   231,
     648,   209,   228,   500,   261,   502,    33,   231,   164,   228,
     437,    93,   276,   105,   231,   318,   486,   163,   630,   282,
     535,    79,   165,   108,   581,     9,   605,   349,   437,   636,
     637,   585,   228,   136,   592,    32,   357,   231,   243,   152,
     156,     9,   291,   622,   623,   171,   653,   515,   654,   363,
     437,    27,  -232,   379,   522,   564,   611,   163,  -232,   198,
     533,   403,   165,  -262,   243,    79,   633,   235,  -262,   474,
     409,   418,     9,  -150,    10,    11,   127,   235,   202,   488,
     203,   228,   430,   228,   509,   411,   231,   511,   231,   568,
      10,    11,  -178,   361,   259,   450,   150,  -177,    11,   228,
     583,   228,   657,   254,   231,    94,   231,   228,   102,   369,
     125,    29,   231,    95,   125,   215,     9,    96,  -262,   199,
     220,    10,    11,   200,  -262,   219,   164,    97,    98,   228,
     223,   201,   263,   250,   231,   141,    79,   234,   264,   397,
     249,   410,   451,   209,   228,   228,   412,   331,   332,   231,
     231,   408,    10,    11,   414,   415,   262,   453,   285,   621,
      99,   479,   286,   334,  -175,    10,    11,   204,   349,  -173,
     519,   287,   335,   206,   353,   163,   481,   336,   337,   338,
     165,  -174,   614,   482,   339,  -172,   513,   243,  -175,   477,
     354,   340,   529,  -173,   489,  -176,  -175,   243,   257,   111,
     521,  -173,   358,   117,   360,  -174,   266,   111,   341,  -172,
     506,   111,   362,  -174,   146,   364,   150,  -172,   228,  -176,
     342,   368,   214,   231,   216,   218,   343,  -176,   408,    11,
     228,   150,   480,   372,   374,   231,   164,   483,   378,   380,
     228,   382,   450,   384,   228,   231,   392,   394,   399,   231,
     386,    79,    79,   401,   494,   405,   479,   254,   419,   349,
     541,   413,   548,     9,   228,   228,   235,   453,   612,   231,
     231,   481,   479,   453,   226,   514,   561,   349,   482,   117,
     117,   117,   416,   432,   444,   163,    79,   481,     9,   451,
     165,   243,   255,   117,   482,    27,   210,   210,   452,   210,
     210,   256,   117,   445,   164,    37,    10,    11,   446,   447,
     117,     9,    10,    11,   112,   448,     9,   117,   458,    47,
      48,     9,   463,   143,   466,   467,    51,   480,   468,     9,
     469,   470,   483,   224,   228,   656,   519,    10,    11,   231,
     117,   485,   379,   480,   490,   493,   147,   508,   483,   503,
     114,    28,     9,   163,   235,    29,   225,   524,   165,   479,
      10,    11,   280,   505,   172,    10,    11,   526,    64,   510,
      10,    11,   281,   512,   481,   520,   527,   228,    10,    11,
     530,   482,   231,   531,   243,   529,   345,   584,   273,   117,
      79,   117,   355,   356,   532,     9,   340,   150,   534,   553,
     555,    10,    11,   556,   563,     9,   569,   117,     9,   117,
     349,   462,   541,   344,   567,   117,   548,   453,   572,   344,
     344,   349,   349,   574,   254,   479,   576,   228,   577,   164,
     480,   277,   231,   580,   582,   483,   590,   117,   593,   591,
     481,   288,   594,   317,    10,    11,  -154,   482,   595,   117,
     137,  -155,   117,   117,    10,    11,   596,    10,    11,   255,
     597,    35,   160,   598,   601,   169,    37,   600,   602,   168,
     604,   606,   625,    10,    11,   112,   334,   613,   163,   459,
      47,    48,     9,   165,   627,   335,   628,    51,   638,   639,
     336,   337,   338,   437,    55,   644,   480,   339,   645,   646,
     647,   483,   106,    66,   340,   626,   254,    56,    57,   650,
      58,    59,   579,   588,    60,   370,   449,    61,   123,   404,
     284,   341,   504,   371,   460,   492,   117,    62,    63,    64,
     352,    10,    11,   475,    91,   263,   344,   573,   117,   343,
     117,   264,    11,   344,   177,   442,   538,   443,   117,   562,
     265,   344,   117,   634,   185,    10,    11,   631,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   558,   351,     0,
       0,     0,   117,   117,     0,     0,   292,   293,   294,   295,
       0,   296,   297,   298,     0,   299,   300,   301,   302,   303,
     304,   305,   306,   307,   308,   309,   310,   311,   312,     0,
     160,     0,   319,     0,   322,   177,     0,     0,   137,   137,
     333,     0,     0,     0,   537,   185,     0,     0,     0,   189,
     544,   547,     0,   552,   194,   195,   196,   197,     0,     0,
       0,     0,   557,     0,   559,   117,     0,     0,     0,     0,
       0,   344,   117,     0,     0,     0,     0,   542,   344,     0,
     344,   117,     0,     0,   175,  -272,     0,   176,   177,   344,
     178,   344,   180,   181,   182,     0,     0,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   137,     0,   176,   177,   117,   178,   179,   180,   181,
     182,   137,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,     0,     0,     0,
       0,     0,    37,     0,   421,  -272,     0,     0,   160,     0,
       0,   112,     0,   599,   421,  -272,    47,    48,     9,     0,
       0,   603,     0,    51,     0,   117,     0,     0,   117,     0,
     224,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     344,     0,     0,     0,     0,     0,     0,   114,   344,     0,
       0,     0,     0,   225,     0,   344,     0,   137,   137,    35,
       0,     0,   544,   632,    37,    64,     0,    10,    11,   396,
       0,     0,     0,   112,     0,     0,     0,     0,    47,    48,
       9,     0,     0,     0,     0,    51,   344,     0,     0,   542,
     344,     0,   158,     0,   117,     0,   137,     0,     0,     0,
       0,     0,     0,     0,     0,    56,    57,     0,    58,   159,
     137,     0,    60,     0,     0,    61,   314,     0,   160,     0,
       0,     0,     0,   169,     0,    62,    63,    64,     0,    10,
      11,     0,     0,     0,   176,   177,   344,   178,   344,   180,
     181,   182,     0,     0,   184,   185,   186,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,   197,     0,     0,
       0,     0,     0,   565,   566,     0,     0,   523,     0,     0,
       0,     0,    -2,    34,     0,    35,     0,     0,    36,     0,
      37,    38,    39,     0,     0,    40,   160,    41,    42,    43,
      44,    45,    46,     0,    47,    48,     9,     0,     0,    49,
      50,    51,    52,    53,    54,     0,   421,     0,    55,     0,
       0,     0,     0,   421,   586,   421,     0,     0,     0,     0,
       0,    56,    57,     0,    58,    59,     0,     0,    60,     0,
       0,    61,     0,     0,   -24,     0,     0,     0,     0,     0,
       0,    62,    63,    64,     0,    10,    11,     0,     0,   324,
       0,    35,     0,     0,    36,  -247,    37,    38,    39,     0,
    -247,    40,     0,    41,    42,   112,    44,    45,    46,     0,
      47,    48,     9,     0,     0,    49,    50,    51,    52,    53,
      54,     0,     0,     0,    55,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    56,    57,     0,
      58,    59,     0,     0,    60,     0,     0,    61,   642,   643,
    -247,   160,     0,     0,     0,   325,  -247,    62,    63,    64,
       0,    10,    11,   324,     0,    35,     0,     0,    36,     0,
      37,    38,    39,     0,     0,    40,     0,    41,    42,   112,
      44,    45,    46,     0,    47,    48,     9,     0,     0,    49,
      50,    51,    52,    53,    54,     0,     0,     0,    55,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    56,    57,     0,    58,    59,     0,     0,    60,     0,
      35,    61,     0,     0,  -247,    37,     0,     0,     0,   325,
    -247,    62,    63,    64,   112,    10,    11,     0,     0,    47,
      48,     9,     0,     0,     0,     0,    51,     0,     0,     0,
       0,     0,     0,    55,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    56,    57,     0,    58,
      59,     0,     0,    60,     0,    35,    61,     0,     0,     0,
      37,     0,     0,     0,   420,     0,    62,    63,    64,   112,
      10,    11,     0,     0,    47,    48,     9,     0,     0,     0,
       0,    51,     0,   429,     0,     0,     0,     0,   158,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    56,    57,     0,    58,   159,     0,     0,    60,     0,
      35,    61,     0,     0,     0,    37,     0,     0,     0,     0,
       0,    62,    63,    64,   112,    10,    11,     0,     0,    47,
      48,     9,     0,   472,     0,     0,    51,     0,     0,     0,
       0,     0,     0,    55,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    56,    57,     0,    58,
      59,     0,     0,    60,     0,    35,    61,     0,     0,     0,
      37,     0,     0,     0,     0,     0,    62,    63,    64,   112,
      10,    11,     0,     0,    47,    48,     9,     0,   473,     0,
       0,    51,     0,     0,     0,     0,     0,     0,    55,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    35,     0,
       0,    56,    57,    37,    58,    59,     0,     0,    60,     0,
       0,    61,   112,     0,     0,     0,     0,    47,    48,     9,
       0,    62,    63,    64,    51,    10,    11,     0,     0,     0,
       0,    55,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    35,     0,     0,    56,    57,    37,    58,    59,     0,
       0,    60,     0,     0,    61,   112,     0,     0,     0,     0,
      47,    48,     9,     0,    62,    63,    64,    51,    10,    11,
       0,     0,     0,     0,   158,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    35,     0,     0,    56,    57,   283,
      58,   159,     0,     0,    60,     0,     0,    61,   112,     0,
       0,     0,     0,    47,    48,     9,     0,    62,    63,    64,
      51,    10,    11,     0,     0,     0,     0,    55,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      56,    57,    37,    58,    59,     0,     0,    60,     0,     0,
      61,   112,     0,     0,     0,     0,    47,    48,     9,     0,
      62,    63,    64,    51,    10,    11,     0,     0,    37,     0,
     224,   241,     0,     0,     0,     0,    37,   112,     0,     0,
       0,     0,    47,    48,     9,   112,     0,   114,     0,    51,
      47,    48,     9,   225,     0,     0,   224,    51,     0,   290,
       0,    37,     0,     0,   113,    64,     0,    10,    11,   281,
     112,     0,     0,   114,     0,    47,    48,     9,     0,   225,
       0,   114,    51,     0,     0,     0,     0,   115,     0,   224,
       0,    64,     0,    10,    11,    37,     0,     0,     0,    64,
       0,    10,    11,     0,   112,     0,   114,     0,     0,    47,
      48,     9,   225,     0,     0,     0,    51,     0,     0,     0,
      37,     0,     0,   406,    64,     0,    10,    11,   283,   112,
       0,     0,     0,     0,    47,    48,     9,   112,     0,     0,
     114,    51,    47,    48,     9,     0,   407,     0,   224,    51,
       0,     0,     0,   334,     0,     0,   224,     0,    64,     0,
      10,    11,   335,     0,     0,   114,     0,   336,   337,   338,
       0,   476,     0,   114,   339,     0,     0,     0,     0,   225,
       0,   439,     0,    64,     0,    10,    11,   334,     0,     0,
       0,    64,     0,    10,    11,     0,   335,   334,   341,     0,
       0,   336,   337,   540,   440,     0,   335,     0,   339,     0,
       0,   336,   337,   338,     0,   340,   343,     0,   339,    11,
       0,     0,     0,     0,     0,   340,     0,     0,     0,     0,
     334,     0,   341,     0,     0,     0,     0,     0,     0,   335,
       0,     0,   341,     0,   336,   337,   338,     0,   607,     0,
     343,   339,    10,    11,     0,     0,     0,     0,   340,     0,
     343,     0,     0,    11,     0,     0,   176,   177,     0,   178,
       0,   180,   181,   182,     0,   341,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
       0,     0,     0,   343,     0,     0,    11,   176,   177,     0,
     178,     0,   180,   181,   182,   431,     0,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   176,   177,     0,   178,     0,   180,   181,   182,     0,
     651,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   176,   177,     0,   178,     0,
     180,   181,   182,     0,   652,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   176,
     177,     0,   434,     0,   180,   181,   182,     0,     0,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   176,   177,     0,     0,     0,   180,   181,
     182,     0,     0,   184,   185,   186,   187,     0,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   177,     0,     0,
       0,   180,   181,   182,     0,     0,   184,   185,   186,   187,
       0,   189,   190,   191,   192,   193,   194,   195,   196,   197
};

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-459)))

#define yytable_value_is_error(Yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       5,    67,   142,   377,    37,   126,    61,   126,   143,   201,
      49,   132,   323,   147,   250,    20,   223,    22,   204,    37,
     258,   142,   452,    28,    28,   135,    67,   485,   321,   267,
      36,    36,     3,    39,    39,    31,   389,   275,    43,    45,
      45,   279,   249,     1,   397,     5,    24,   444,    53,    54,
      61,   289,    35,    24,    62,     0,   316,     3,   263,   264,
      68,    20,    67,    11,    59,    35,     1,     7,    24,     3,
      60,    25,    12,     5,     8,    37,    59,    35,    25,    74,
      37,    59,    62,    17,    67,    75,     7,   126,    22,    23,
      24,    12,    63,     7,     3,    29,    21,    67,    12,    61,
      35,    59,    73,    74,    61,    65,   172,   460,    67,    67,
     106,     5,   523,   524,   572,    75,    55,    56,    57,    58,
      59,    60,   127,    62,    63,    59,    72,    75,    68,   135,
     135,   172,    67,    65,    66,   141,   141,    71,   143,    73,
      74,    62,   147,    75,   437,   255,    24,    68,   159,    24,
     503,   113,   114,   115,    68,    40,   113,   114,   115,    44,
      63,   113,   114,   115,   424,   127,   426,   172,   207,   208,
     127,    65,   602,     3,   136,   382,   594,    24,    53,   136,
     638,    75,   144,   388,   136,   390,    62,   144,   199,   151,
      62,    62,   144,     3,   151,   200,    68,   159,   595,   151,
     436,   206,   159,     3,   515,    24,   559,   212,    62,   606,
     607,   522,   174,    71,    68,   220,   220,   174,   223,   158,
     159,    24,   174,   576,   577,    62,   644,   413,   646,   234,
      62,     3,    62,    63,   420,   471,    68,   199,    68,    67,
     432,   280,   199,     7,   249,   250,   599,   368,    12,   368,
     283,   290,    24,    66,    73,    74,    59,   378,    66,   378,
      59,   223,   317,   225,   399,   283,   223,   401,   225,   476,
      73,    74,    59,   225,   384,     3,   281,    59,    74,   241,
     518,   243,   656,    24,   241,     9,   243,   249,    60,   241,
      40,    63,   249,    17,    44,    24,    24,    21,    62,    59,
      62,    73,    74,    63,    68,    64,   317,    31,    32,   271,
      59,    71,    53,    67,   271,    51,   321,    63,    59,   271,
      59,   283,    50,    75,   286,   287,   283,   366,   367,   286,
     287,   283,    73,    74,   286,   287,    35,   342,    60,   575,
      64,   374,    72,     8,    35,    73,    74,    83,   353,    35,
     416,    72,    17,    89,    59,   317,   374,    22,    23,    24,
     317,    35,   569,   374,    29,    35,   405,   372,    59,   374,
       3,    36,   427,    59,   379,    35,    67,   382,   384,   384,
     419,    67,    60,    37,     8,    59,   392,   392,    53,    59,
     396,   396,    60,    67,   399,    35,   401,    67,   360,    59,
      65,    62,    95,   360,    97,    98,    71,    67,   360,    74,
     372,   416,   374,    75,    60,   372,   427,   374,    62,    24,
     382,    59,     3,    62,   386,   382,    62,     3,    62,   386,
      72,   436,   437,    62,   386,    65,   469,    24,    65,   444,
     445,    67,   447,    24,   406,   407,   567,   452,   567,   406,
     407,   469,   485,   458,   406,   407,   461,   462,   469,   113,
     114,   115,    62,    66,    59,   427,   471,   485,    24,    50,
     427,   476,    59,   127,   485,     3,    94,    95,    59,    97,
      98,    68,   136,    67,   495,     8,    73,    74,    71,    67,
     144,    24,    73,    74,    17,     8,    24,   151,    65,    22,
      23,    24,    62,    59,    62,    62,    29,   469,    62,    24,
      60,    60,   469,    36,   476,   655,   582,    73,    74,   476,
     174,    60,    63,   485,    60,    68,    59,    35,   485,    60,
      53,    59,    24,   495,   655,    63,    59,    75,   495,   572,
      73,    74,    65,    68,    59,    73,    74,    68,    71,    60,
      73,    74,    75,    60,   572,    60,    75,   519,    73,    74,
      60,   572,   519,    60,   569,   620,   211,   519,    60,   223,
     575,   225,   217,   218,    60,    24,    36,   582,    68,     3,
      62,    73,    74,    72,    60,    24,    59,   241,    24,   243,
     595,    75,   597,   211,    62,   249,   601,   602,    60,   217,
     218,   606,   607,    66,    24,   638,    60,   569,    60,   620,
     572,    60,   569,    60,    62,   572,    60,   271,    60,    72,
     638,    60,    60,    59,    73,    74,    59,   638,    59,   283,
      49,    59,   286,   287,    73,    74,    68,    73,    74,    59,
      62,     3,    61,    72,    62,    64,     8,    68,    49,    11,
      62,    59,    68,    73,    74,    17,     8,    60,   620,    11,
      22,    23,    24,   620,    60,    17,    68,    29,    60,    14,
      22,    23,    24,    62,    36,    60,   638,    29,    60,    60,
      60,   638,    31,    22,    36,   582,    24,    49,    50,   639,
      52,    53,   508,   524,    56,   243,   341,    59,    39,   281,
     159,    53,   392,   243,   349,   384,   360,    69,    70,    71,
     213,    73,    74,   372,    22,    53,   334,   495,   372,    71,
     374,    59,    74,   341,    34,   334,   440,   334,   382,   462,
      68,   349,   386,   601,    44,    73,    74,   597,    48,    49,
      50,    51,    52,    53,    54,    55,    56,   458,   212,    -1,
      -1,    -1,   406,   407,    -1,    -1,   175,   176,   177,   178,
      -1,   180,   181,   182,    -1,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,    -1,
     199,    -1,   201,    -1,   203,    34,    -1,    -1,   207,   208,
     209,    -1,    -1,    -1,   439,    44,    -1,    -1,    -1,    48,
     445,   446,    -1,   448,    53,    54,    55,    56,    -1,    -1,
      -1,    -1,   457,    -1,   459,   469,    -1,    -1,    -1,    -1,
      -1,   439,   476,    -1,    -1,    -1,    -1,   445,   446,    -1,
     448,   485,    -1,    -1,     4,     5,    -1,    33,    34,   457,
      36,   459,    38,    39,    40,    -1,    -1,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,   280,    -1,    33,    34,   519,    36,    37,    38,    39,
      40,   290,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    -1,    -1,    -1,
      -1,    -1,     8,    -1,   313,    65,    -1,    -1,   317,    -1,
      -1,    17,    -1,   548,   323,    75,    22,    23,    24,    -1,
      -1,   556,    -1,    29,    -1,   569,    -1,    -1,   572,    -1,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     548,    -1,    -1,    -1,    -1,    -1,    -1,    53,   556,    -1,
      -1,    -1,    -1,    59,    -1,   563,    -1,   366,   367,     3,
      -1,    -1,   597,   598,     8,    71,    -1,    73,    74,    75,
      -1,    -1,    -1,    17,    -1,    -1,    -1,    -1,    22,    23,
      24,    -1,    -1,    -1,    -1,    29,   594,    -1,    -1,   597,
     598,    -1,    36,    -1,   638,    -1,   405,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    49,    50,    -1,    52,    53,
     419,    -1,    56,    -1,    -1,    59,    60,    -1,   427,    -1,
      -1,    -1,    -1,   432,    -1,    69,    70,    71,    -1,    73,
      74,    -1,    -1,    -1,    33,    34,   644,    36,   646,    38,
      39,    40,    -1,    -1,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    -1,    -1,
      -1,    -1,    -1,   472,   473,    -1,    -1,    66,    -1,    -1,
      -1,    -1,     0,     1,    -1,     3,    -1,    -1,     6,    -1,
       8,     9,    10,    -1,    -1,    13,   495,    15,    16,    17,
      18,    19,    20,    -1,    22,    23,    24,    -1,    -1,    27,
      28,    29,    30,    31,    32,    -1,   515,    -1,    36,    -1,
      -1,    -1,    -1,   522,   523,   524,    -1,    -1,    -1,    -1,
      -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,    -1,
      -1,    59,    -1,    -1,    62,    -1,    -1,    -1,    -1,    -1,
      -1,    69,    70,    71,    -1,    73,    74,    -1,    -1,     1,
      -1,     3,    -1,    -1,     6,     7,     8,     9,    10,    -1,
      12,    13,    -1,    15,    16,    17,    18,    19,    20,    -1,
      22,    23,    24,    -1,    -1,    27,    28,    29,    30,    31,
      32,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,
      52,    53,    -1,    -1,    56,    -1,    -1,    59,   617,   618,
      62,   620,    -1,    -1,    -1,    67,    68,    69,    70,    71,
      -1,    73,    74,     1,    -1,     3,    -1,    -1,     6,    -1,
       8,     9,    10,    -1,    -1,    13,    -1,    15,    16,    17,
      18,    19,    20,    -1,    22,    23,    24,    -1,    -1,    27,
      28,    29,    30,    31,    32,    -1,    -1,    -1,    36,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    49,    50,    -1,    52,    53,    -1,    -1,    56,    -1,
       3,    59,    -1,    -1,    62,     8,    -1,    -1,    -1,    67,
      68,    69,    70,    71,    17,    73,    74,    -1,    -1,    22,
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
      -1,     3,    -1,    -1,    49,    50,     8,    52,    53,    -1,
      -1,    56,    -1,    -1,    59,    17,    -1,    -1,    -1,    -1,
      22,    23,    24,    -1,    69,    70,    71,    29,    73,    74,
      -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     3,    -1,    -1,    49,    50,     8,
      52,    53,    -1,    -1,    56,    -1,    -1,    59,    17,    -1,
      -1,    -1,    -1,    22,    23,    24,    -1,    69,    70,    71,
      29,    73,    74,    -1,    -1,    -1,    -1,    36,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      49,    50,     8,    52,    53,    -1,    -1,    56,    -1,    -1,
      59,    17,    -1,    -1,    -1,    -1,    22,    23,    24,    -1,
      69,    70,    71,    29,    73,    74,    -1,    -1,     8,    -1,
      36,    11,    -1,    -1,    -1,    -1,     8,    17,    -1,    -1,
      -1,    -1,    22,    23,    24,    17,    -1,    53,    -1,    29,
      22,    23,    24,    59,    -1,    -1,    36,    29,    -1,    65,
      -1,     8,    -1,    -1,    36,    71,    -1,    73,    74,    75,
      17,    -1,    -1,    53,    -1,    22,    23,    24,    -1,    59,
      -1,    53,    29,    -1,    -1,    -1,    -1,    59,    -1,    36,
      -1,    71,    -1,    73,    74,     8,    -1,    -1,    -1,    71,
      -1,    73,    74,    -1,    17,    -1,    53,    -1,    -1,    22,
      23,    24,    59,    -1,    -1,    -1,    29,    -1,    -1,    -1,
       8,    -1,    -1,    36,    71,    -1,    73,    74,     8,    17,
      -1,    -1,    -1,    -1,    22,    23,    24,    17,    -1,    -1,
      53,    29,    22,    23,    24,    -1,    59,    -1,    36,    29,
      -1,    -1,    -1,     8,    -1,    -1,    36,    -1,    71,    -1,
      73,    74,    17,    -1,    -1,    53,    -1,    22,    23,    24,
      -1,    59,    -1,    53,    29,    -1,    -1,    -1,    -1,    59,
      -1,    36,    -1,    71,    -1,    73,    74,     8,    -1,    -1,
      -1,    71,    -1,    73,    74,    -1,    17,     8,    53,    -1,
      -1,    22,    23,    24,    59,    -1,    17,    -1,    29,    -1,
      -1,    22,    23,    24,    -1,    36,    71,    -1,    29,    74,
      -1,    -1,    -1,    -1,    -1,    36,    -1,    -1,    -1,    -1,
       8,    -1,    53,    -1,    -1,    -1,    -1,    -1,    -1,    17,
      -1,    -1,    53,    -1,    22,    23,    24,    -1,    59,    -1,
      71,    29,    73,    74,    -1,    -1,    -1,    -1,    36,    -1,
      71,    -1,    -1,    74,    -1,    -1,    33,    34,    -1,    36,
      -1,    38,    39,    40,    -1,    53,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      -1,    -1,    -1,    71,    -1,    -1,    74,    33,    34,    -1,
      36,    -1,    38,    39,    40,    72,    -1,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    33,    34,    -1,    36,    -1,    38,    39,    40,    -1,
      66,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    33,    34,    -1,    36,    -1,
      38,    39,    40,    -1,    66,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    33,
      34,    -1,    60,    -1,    38,    39,    40,    -1,    -1,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    33,    34,    -1,    -1,    -1,    38,    39,
      40,    -1,    -1,    43,    44,    45,    46,    -1,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    34,    -1,    -1,
      -1,    38,    39,    40,    -1,    -1,    43,    44,    45,    46,
      -1,    48,    49,    50,    51,    52,    53,    54,    55,    56
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    77,    79,    80,     0,    25,    78,    25,    86,    24,
      73,    74,   140,   141,    81,    24,    88,    89,     3,    62,
      21,    82,   165,    24,    87,   213,    63,     3,    59,    63,
      83,    85,   140,    62,     1,     3,     6,     8,     9,    10,
      13,    15,    16,    17,    18,    19,    20,    22,    23,    27,
      28,    29,    30,    31,    32,    36,    49,    50,    52,    53,
      56,    59,    69,    70,    71,    90,    91,    92,    98,   110,
     113,   121,   124,   126,   127,   128,   129,   133,   137,   140,
     142,   143,   148,   149,   152,   155,   156,   157,   160,   163,
     164,   180,   185,    62,     9,    17,    21,    31,    32,    64,
     198,    24,    60,    83,    84,     3,    86,    88,     3,   137,
     139,   140,    17,    36,    53,    59,   140,   142,   147,   151,
     152,   153,   160,   139,   128,   133,   111,    59,   140,   158,
     128,   137,   114,    35,    67,   136,    71,   126,   185,   192,
     125,   136,   122,    59,    96,    97,   140,    59,    93,   138,
     140,   184,   127,   127,   127,   127,   127,   127,    36,    53,
     126,   134,   146,   152,   154,   160,   127,   127,    11,   126,
     191,    62,    59,    94,   184,     4,    33,    34,    36,    37,
      38,    39,    40,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    67,    59,
      63,    71,    66,    59,   136,     1,   136,     5,    65,    75,
     141,   199,    59,   159,   199,    24,   199,   200,   199,    64,
      62,   189,    88,    59,    36,    59,   145,   151,   152,   153,
     154,   160,   145,   145,    63,    98,   107,   108,   109,   185,
     193,    11,   135,   140,   144,   145,   176,   177,   178,    59,
      67,   161,   112,   193,    24,    59,    68,   137,   170,   172,
     174,   145,    35,    53,    59,    68,   137,   169,   171,   172,
     173,   183,   112,    60,    97,   168,   145,    60,    93,   166,
      65,    75,   145,     8,   146,    60,    72,    72,    60,    94,
      65,   145,   126,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   126,   126,   126,   126,   126,   126,   126,
     126,   126,   126,   130,    60,   134,   186,    59,   140,   126,
     191,   181,   126,   130,     1,    67,    91,   100,   179,   180,
     182,   185,   185,   126,     8,    17,    22,    23,    24,    29,
      36,    53,    65,    71,   141,   201,   203,   204,   205,   140,
     206,   214,   161,    59,     3,   201,   201,    83,    60,   178,
       8,   145,    60,   140,    35,   105,     5,    65,    62,   145,
     135,   144,    75,   190,    60,   178,   182,   115,    62,    63,
      24,   172,    59,   175,    62,   189,    72,   104,    59,   173,
      53,   173,    62,   189,     3,   197,    75,   145,   123,    62,
     189,    62,   189,   185,   138,    65,    36,    59,   145,   151,
     152,   153,   160,    67,   145,   145,    62,   189,   185,    65,
      67,   126,   131,   132,   187,   188,    11,    75,   190,    31,
     134,    72,    66,   179,    60,   188,   101,    62,    68,    36,
      59,   202,   203,   205,    59,    67,    71,    67,     8,   201,
       3,    50,    59,   140,   211,   212,     3,    72,    65,    11,
     201,    60,    75,    62,   194,   214,    62,    62,    62,    60,
      60,   106,    26,    26,   193,   176,    59,   140,   150,   151,
     152,   153,   154,   160,   162,    60,    68,   105,   193,   140,
      60,   178,   174,    68,   145,     7,    12,    68,    99,   102,
     173,   197,   173,    60,   171,    68,   137,   197,    35,    97,
      60,    93,    60,   185,   145,   130,    94,    95,   167,   184,
      60,   185,   130,    66,    75,   190,    68,    75,   190,   134,
      60,    60,    60,   191,    68,   182,   179,   201,   204,   194,
      24,   140,   141,   196,   201,   208,   216,   201,   140,   195,
     207,   215,   201,     3,   211,    62,    72,   201,   212,   201,
     197,   140,   206,    60,   182,   126,   126,    62,   178,    59,
     162,   116,    60,   186,    66,   103,    60,    60,   197,   104,
      60,   188,    62,   189,   145,   188,   126,   132,   131,   132,
      60,    72,    68,    60,    60,    59,    68,    62,    72,   201,
      68,    62,    49,   201,    62,   197,    59,    59,   201,   209,
     210,    68,   193,    60,   178,   119,   162,     5,    65,    66,
      75,   182,   197,   197,    68,    68,    95,    60,    68,   209,
     194,   208,   201,   197,   207,   211,   194,   194,    60,    14,
     117,   120,   126,   126,    60,    60,    60,    60,   162,    20,
     100,    66,    66,   209,   209,   118,   112,   105
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

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))

/* Error token number */
#define YYTERROR	1
#define YYERRCODE	256


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
  FILE *yyo = yyoutput;
  YYUSE (yyo);
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
  YYSIZE_T yysize0 = yytnamerr (YY_NULL, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULL;
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
                yysize1 = yysize + yytnamerr (YY_NULL, yytname[yyx]);
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




/* The lookahead symbol.  */
int yychar, yystate;


#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval YY_INITIAL_VALUE(yyval_default);

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

       Refer to the stacks through separate pointers, to allow yyoverflow
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
  int yytoken = 0;
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

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
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
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

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
/* Line 1778 of yacc.c  */
#line 128 "go.y"
    {
		xtop = concat(xtop, (yyvsp[(4) - (4)].list));
	}
    break;

  case 3:
/* Line 1778 of yacc.c  */
#line 134 "go.y"
    {
		prevlineno = lineno;
		yyerror("package statement must be first");
		flusherrors();
		mkpackage("main");
	}
    break;

  case 4:
/* Line 1778 of yacc.c  */
#line 141 "go.y"
    {
		mkpackage((yyvsp[(2) - (3)].sym)->name);
	}
    break;

  case 5:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 162 "go.y"
    {
		importpkg = nil;
	}
    break;

  case 12:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 224 "go.y"
    {
		// import with original name
		(yyval.i) = parserline();
		importmyname = S;
		importfile(&(yyvsp[(1) - (1)].val), (yyval.i));
	}
    break;

  case 17:
/* Line 1778 of yacc.c  */
#line 231 "go.y"
    {
		// import with given name
		(yyval.i) = parserline();
		importmyname = (yyvsp[(1) - (2)].sym);
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 18:
/* Line 1778 of yacc.c  */
#line 238 "go.y"
    {
		// import into my name space
		(yyval.i) = parserline();
		importmyname = lookup(".");
		importfile(&(yyvsp[(2) - (2)].val), (yyval.i));
	}
    break;

  case 19:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 261 "go.y"
    {
		if(strcmp((yyvsp[(1) - (1)].sym)->name, "safe") == 0)
			curio.importsafe = 1;
	}
    break;

  case 22:
/* Line 1778 of yacc.c  */
#line 267 "go.y"
    {
		defercheckwidth();
	}
    break;

  case 23:
/* Line 1778 of yacc.c  */
#line 271 "go.y"
    {
		resumecheckwidth();
		unimportfile();
	}
    break;

  case 24:
/* Line 1778 of yacc.c  */
#line 280 "go.y"
    {
		yyerror("empty top-level declaration");
		(yyval.list) = nil;
	}
    break;

  case 26:
/* Line 1778 of yacc.c  */
#line 286 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 27:
/* Line 1778 of yacc.c  */
#line 290 "go.y"
    {
		yyerror("non-declaration statement outside function body");
		(yyval.list) = nil;
	}
    break;

  case 28:
/* Line 1778 of yacc.c  */
#line 295 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 29:
/* Line 1778 of yacc.c  */
#line 301 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
	}
    break;

  case 30:
/* Line 1778 of yacc.c  */
#line 305 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 31:
/* Line 1778 of yacc.c  */
#line 309 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 32:
/* Line 1778 of yacc.c  */
#line 313 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (2)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 33:
/* Line 1778 of yacc.c  */
#line 319 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 34:
/* Line 1778 of yacc.c  */
#line 325 "go.y"
    {
		(yyval.list) = concat((yyvsp[(3) - (7)].list), (yyvsp[(5) - (7)].list));
		iota = -100000;
		lastconst = nil;
	}
    break;

  case 35:
/* Line 1778 of yacc.c  */
#line 331 "go.y"
    {
		(yyval.list) = nil;
		iota = -100000;
	}
    break;

  case 36:
/* Line 1778 of yacc.c  */
#line 336 "go.y"
    {
		(yyval.list) = list1((yyvsp[(2) - (2)].node));
	}
    break;

  case 37:
/* Line 1778 of yacc.c  */
#line 340 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (5)].list);
	}
    break;

  case 38:
/* Line 1778 of yacc.c  */
#line 344 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 39:
/* Line 1778 of yacc.c  */
#line 350 "go.y"
    {
		iota = 0;
	}
    break;

  case 40:
/* Line 1778 of yacc.c  */
#line 356 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 41:
/* Line 1778 of yacc.c  */
#line 360 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 42:
/* Line 1778 of yacc.c  */
#line 364 "go.y"
    {
		(yyval.list) = variter((yyvsp[(1) - (3)].list), nil, (yyvsp[(3) - (3)].list));
	}
    break;

  case 43:
/* Line 1778 of yacc.c  */
#line 370 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (4)].list), (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].list));
	}
    break;

  case 44:
/* Line 1778 of yacc.c  */
#line 374 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (3)].list), N, (yyvsp[(3) - (3)].list));
	}
    break;

  case 46:
/* Line 1778 of yacc.c  */
#line 381 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node), nil);
	}
    break;

  case 47:
/* Line 1778 of yacc.c  */
#line 385 "go.y"
    {
		(yyval.list) = constiter((yyvsp[(1) - (1)].list), N, nil);
	}
    break;

  case 48:
/* Line 1778 of yacc.c  */
#line 391 "go.y"
    {
		// different from dclname because the name
		// becomes visible right here, not at the end
		// of the declaration.
		(yyval.node) = typedcl0((yyvsp[(1) - (1)].sym));
	}
    break;

  case 49:
/* Line 1778 of yacc.c  */
#line 400 "go.y"
    {
		(yyval.node) = typedcl1((yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node), 1);
	}
    break;

  case 50:
/* Line 1778 of yacc.c  */
#line 406 "go.y"
    {
		(yyval.node) = (yyvsp[(1) - (1)].node);
	}
    break;

  case 51:
/* Line 1778 of yacc.c  */
#line 410 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
		(yyval.node)->etype = (yyvsp[(2) - (3)].i);			// rathole to pass opcode
	}
    break;

  case 52:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 443 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OADD;
	}
    break;

  case 55:
/* Line 1778 of yacc.c  */
#line 448 "go.y"
    {
		(yyval.node) = nod(OASOP, (yyvsp[(1) - (2)].node), nodintconst(1));
		(yyval.node)->etype = OSUB;
	}
    break;

  case 56:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 520 "go.y"
    {
		markdcl();
	}
    break;

  case 61:
/* Line 1778 of yacc.c  */
#line 524 "go.y"
    {
		(yyval.node) = liststmt((yyvsp[(3) - (4)].list));
		popdcl();
	}
    break;

  case 62:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 561 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 65:
/* Line 1778 of yacc.c  */
#line 565 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].node));
	}
    break;

  case 66:
/* Line 1778 of yacc.c  */
#line 571 "go.y"
    {
		markdcl();
	}
    break;

  case 67:
/* Line 1778 of yacc.c  */
#line 575 "go.y"
    {
		(yyval.list) = (yyvsp[(3) - (4)].list);
		popdcl();
	}
    break;

  case 68:
/* Line 1778 of yacc.c  */
#line 582 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->etype = 0;	// := flag
	}
    break;

  case 69:
/* Line 1778 of yacc.c  */
#line 588 "go.y"
    {
		(yyval.node) = nod(ORANGE, N, (yyvsp[(4) - (4)].node));
		(yyval.node)->list = (yyvsp[(1) - (4)].list);
		(yyval.node)->colas = 1;
		colasdefn((yyvsp[(1) - (4)].list), (yyval.node));
	}
    break;

  case 70:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 608 "go.y"
    {
		// normal test
		(yyval.node) = nod(OFOR, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 73:
/* Line 1778 of yacc.c  */
#line 617 "go.y"
    {
		(yyval.node) = (yyvsp[(1) - (2)].node);
		(yyval.node)->nbody = concat((yyval.node)->nbody, (yyvsp[(2) - (2)].list));
	}
    break;

  case 74:
/* Line 1778 of yacc.c  */
#line 624 "go.y"
    {
		markdcl();
	}
    break;

  case 75:
/* Line 1778 of yacc.c  */
#line 628 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (3)].node);
		popdcl();
	}
    break;

  case 76:
/* Line 1778 of yacc.c  */
#line 635 "go.y"
    {
		// test
		(yyval.node) = nod(OIF, N, N);
		(yyval.node)->ntest = (yyvsp[(1) - (1)].node);
	}
    break;

  case 77:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 652 "go.y"
    {
		markdcl();
	}
    break;

  case 79:
/* Line 1778 of yacc.c  */
#line 656 "go.y"
    {
		if((yyvsp[(3) - (3)].node)->ntest == N)
			yyerror("missing condition in if statement");
	}
    break;

  case 80:
/* Line 1778 of yacc.c  */
#line 661 "go.y"
    {
		(yyvsp[(3) - (5)].node)->nbody = (yyvsp[(5) - (5)].list);
	}
    break;

  case 81:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 682 "go.y"
    {
		markdcl();
	}
    break;

  case 83:
/* Line 1778 of yacc.c  */
#line 686 "go.y"
    {
		if((yyvsp[(4) - (5)].node)->ntest == N)
			yyerror("missing condition in if statement");
		(yyvsp[(4) - (5)].node)->nbody = (yyvsp[(5) - (5)].list);
		(yyval.list) = list1((yyvsp[(4) - (5)].node));
	}
    break;

  case 84:
/* Line 1778 of yacc.c  */
#line 694 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 85:
/* Line 1778 of yacc.c  */
#line 698 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (2)].list), (yyvsp[(2) - (2)].list));
	}
    break;

  case 86:
/* Line 1778 of yacc.c  */
#line 703 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 87:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 718 "go.y"
    {
		markdcl();
	}
    break;

  case 89:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 740 "go.y"
    {
		typesw = nod(OXXX, typesw, N);
	}
    break;

  case 92:
/* Line 1778 of yacc.c  */
#line 744 "go.y"
    {
		(yyval.node) = nod(OSELECT, N, N);
		(yyval.node)->lineno = typesw->lineno;
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		typesw = typesw->left;
	}
    break;

  case 94:
/* Line 1778 of yacc.c  */
#line 757 "go.y"
    {
		(yyval.node) = nod(OOROR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 95:
/* Line 1778 of yacc.c  */
#line 761 "go.y"
    {
		(yyval.node) = nod(OANDAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 96:
/* Line 1778 of yacc.c  */
#line 765 "go.y"
    {
		(yyval.node) = nod(OEQ, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 97:
/* Line 1778 of yacc.c  */
#line 769 "go.y"
    {
		(yyval.node) = nod(ONE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 98:
/* Line 1778 of yacc.c  */
#line 773 "go.y"
    {
		(yyval.node) = nod(OLT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 99:
/* Line 1778 of yacc.c  */
#line 777 "go.y"
    {
		(yyval.node) = nod(OLE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 100:
/* Line 1778 of yacc.c  */
#line 781 "go.y"
    {
		(yyval.node) = nod(OGE, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 101:
/* Line 1778 of yacc.c  */
#line 785 "go.y"
    {
		(yyval.node) = nod(OGT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 102:
/* Line 1778 of yacc.c  */
#line 789 "go.y"
    {
		(yyval.node) = nod(OADD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 103:
/* Line 1778 of yacc.c  */
#line 793 "go.y"
    {
		(yyval.node) = nod(OSUB, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 104:
/* Line 1778 of yacc.c  */
#line 797 "go.y"
    {
		(yyval.node) = nod(OOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 105:
/* Line 1778 of yacc.c  */
#line 801 "go.y"
    {
		(yyval.node) = nod(OXOR, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 106:
/* Line 1778 of yacc.c  */
#line 805 "go.y"
    {
		(yyval.node) = nod(OMUL, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 107:
/* Line 1778 of yacc.c  */
#line 809 "go.y"
    {
		(yyval.node) = nod(ODIV, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 108:
/* Line 1778 of yacc.c  */
#line 813 "go.y"
    {
		(yyval.node) = nod(OMOD, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 109:
/* Line 1778 of yacc.c  */
#line 817 "go.y"
    {
		(yyval.node) = nod(OAND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 110:
/* Line 1778 of yacc.c  */
#line 821 "go.y"
    {
		(yyval.node) = nod(OANDNOT, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 111:
/* Line 1778 of yacc.c  */
#line 825 "go.y"
    {
		(yyval.node) = nod(OLSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 112:
/* Line 1778 of yacc.c  */
#line 829 "go.y"
    {
		(yyval.node) = nod(ORSH, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 113:
/* Line 1778 of yacc.c  */
#line 834 "go.y"
    {
		(yyval.node) = nod(OSEND, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 115:
/* Line 1778 of yacc.c  */
#line 841 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 116:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 856 "go.y"
    {
		(yyval.node) = nod(OPLUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 118:
/* Line 1778 of yacc.c  */
#line 860 "go.y"
    {
		(yyval.node) = nod(OMINUS, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 119:
/* Line 1778 of yacc.c  */
#line 864 "go.y"
    {
		(yyval.node) = nod(ONOT, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 120:
/* Line 1778 of yacc.c  */
#line 868 "go.y"
    {
		yyerror("the bitwise complement operator is ^");
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 121:
/* Line 1778 of yacc.c  */
#line 873 "go.y"
    {
		(yyval.node) = nod(OCOM, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 122:
/* Line 1778 of yacc.c  */
#line 877 "go.y"
    {
		(yyval.node) = nod(ORECV, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 123:
/* Line 1778 of yacc.c  */
#line 887 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (3)].node), N);
	}
    break;

  case 124:
/* Line 1778 of yacc.c  */
#line 891 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (5)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
	}
    break;

  case 125:
/* Line 1778 of yacc.c  */
#line 896 "go.y"
    {
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (6)].node), N);
		(yyval.node)->list = (yyvsp[(3) - (6)].list);
		(yyval.node)->isddd = 1;
	}
    break;

  case 126:
/* Line 1778 of yacc.c  */
#line 904 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 128:
/* Line 1778 of yacc.c  */
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
/* Line 1778 of yacc.c  */
#line 920 "go.y"
    {
		(yyval.node) = nod(ODOTTYPE, (yyvsp[(1) - (5)].node), (yyvsp[(4) - (5)].node));
	}
    break;

  case 130:
/* Line 1778 of yacc.c  */
#line 924 "go.y"
    {
		(yyval.node) = nod(OTYPESW, N, (yyvsp[(1) - (5)].node));
	}
    break;

  case 131:
/* Line 1778 of yacc.c  */
#line 928 "go.y"
    {
		(yyval.node) = nod(OINDEX, (yyvsp[(1) - (4)].node), (yyvsp[(3) - (4)].node));
	}
    break;

  case 132:
/* Line 1778 of yacc.c  */
#line 932 "go.y"
    {
		(yyval.node) = nod(OSLICE, (yyvsp[(1) - (6)].node), nod(OKEY, (yyvsp[(3) - (6)].node), (yyvsp[(5) - (6)].node)));
	}
    break;

  case 134:
/* Line 1778 of yacc.c  */
#line 937 "go.y"
    {
		// conversion
		(yyval.node) = nod(OCALL, (yyvsp[(1) - (4)].node), N);
		(yyval.node)->list = list1((yyvsp[(3) - (4)].node));
	}
    break;

  case 135:
/* Line 1778 of yacc.c  */
#line 943 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 136:
/* Line 1778 of yacc.c  */
#line 950 "go.y"
    {
		(yyval.node) = (yyvsp[(3) - (5)].node);
		(yyval.node)->right = (yyvsp[(1) - (5)].node);
		(yyval.node)->list = (yyvsp[(4) - (5)].list);
	}
    break;

  case 137:
/* Line 1778 of yacc.c  */
#line 956 "go.y"
    {
		yyerror("cannot parenthesize type in composite literal");
		(yyval.node) = (yyvsp[(5) - (7)].node);
		(yyval.node)->right = (yyvsp[(2) - (7)].node);
		(yyval.node)->list = (yyvsp[(6) - (7)].list);
	}
    break;

  case 139:
/* Line 1778 of yacc.c  */
#line 965 "go.y"
    {
		// composite expression.
		// make node early so we get the right line number.
		(yyval.node) = nod(OCOMPLIT, N, N);
	}
    break;

  case 140:
/* Line 1778 of yacc.c  */
#line 973 "go.y"
    {
		(yyval.node) = nod(OKEY, (yyvsp[(1) - (3)].node), (yyvsp[(3) - (3)].node));
	}
    break;

  case 142:
/* Line 1778 of yacc.c  */
#line 980 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (4)].node);
		(yyval.node)->list = (yyvsp[(3) - (4)].list);
	}
    break;

  case 144:
/* Line 1778 of yacc.c  */
#line 988 "go.y"
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

  case 148:
/* Line 1778 of yacc.c  */
#line 1013 "go.y"
    {
		(yyval.i) = LBODY;
	}
    break;

  case 149:
/* Line 1778 of yacc.c  */
#line 1017 "go.y"
    {
		(yyval.i) = '{';
	}
    break;

  case 150:
/* Line 1778 of yacc.c  */
#line 1028 "go.y"
    {
		if((yyvsp[(1) - (1)].sym) == S)
			(yyval.node) = N;
		else
			(yyval.node) = newname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 151:
/* Line 1778 of yacc.c  */
#line 1037 "go.y"
    {
		(yyval.node) = dclname((yyvsp[(1) - (1)].sym));
	}
    break;

  case 152:
/* Line 1778 of yacc.c  */
#line 1042 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 154:
/* Line 1778 of yacc.c  */
#line 1049 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if(importpkg != nil && !exportname((yyvsp[(1) - (1)].sym)->name))
			(yyval.sym) = pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg);
	}
    break;

  case 156:
/* Line 1778 of yacc.c  */
#line 1057 "go.y"
    {
		(yyval.sym) = S;
	}
    break;

  case 157:
/* Line 1778 of yacc.c  */
#line 1063 "go.y"
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

  case 158:
/* Line 1778 of yacc.c  */
#line 1078 "go.y"
    {
		(yyval.node) = oldname((yyvsp[(1) - (1)].sym));
		if((yyval.node)->pack != N)
			(yyval.node)->pack->used = 1;
	}
    break;

  case 160:
/* Line 1778 of yacc.c  */
#line 1098 "go.y"
    {
		yyerror("final argument in variadic function missing type");
		(yyval.node) = nod(ODDD, typenod(typ(TINTER)), N);
	}
    break;

  case 161:
/* Line 1778 of yacc.c  */
#line 1103 "go.y"
    {
		(yyval.node) = nod(ODDD, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 167:
/* Line 1778 of yacc.c  */
#line 1114 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 171:
/* Line 1778 of yacc.c  */
#line 1123 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 176:
/* Line 1778 of yacc.c  */
#line 1133 "go.y"
    {
		(yyval.node) = nod(OTPAREN, (yyvsp[(2) - (3)].node), N);
	}
    break;

  case 186:
/* Line 1778 of yacc.c  */
#line 1154 "go.y"
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

  case 187:
/* Line 1778 of yacc.c  */
#line 1167 "go.y"
    {
		(yyval.node) = nod(OTARRAY, (yyvsp[(2) - (4)].node), (yyvsp[(4) - (4)].node));
	}
    break;

  case 188:
/* Line 1778 of yacc.c  */
#line 1171 "go.y"
    {
		// array literal of nelem
		(yyval.node) = nod(OTARRAY, nod(ODDD, N, N), (yyvsp[(4) - (4)].node));
	}
    break;

  case 189:
/* Line 1778 of yacc.c  */
#line 1176 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->etype = Cboth;
	}
    break;

  case 190:
/* Line 1778 of yacc.c  */
#line 1181 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Csend;
	}
    break;

  case 191:
/* Line 1778 of yacc.c  */
#line 1186 "go.y"
    {
		(yyval.node) = nod(OTMAP, (yyvsp[(3) - (5)].node), (yyvsp[(5) - (5)].node));
	}
    break;

  case 194:
/* Line 1778 of yacc.c  */
#line 1194 "go.y"
    {
		(yyval.node) = nod(OIND, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 195:
/* Line 1778 of yacc.c  */
#line 1200 "go.y"
    {
		(yyval.node) = nod(OTCHAN, (yyvsp[(3) - (3)].node), N);
		(yyval.node)->etype = Crecv;
	}
    break;

  case 196:
/* Line 1778 of yacc.c  */
#line 1207 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 197:
/* Line 1778 of yacc.c  */
#line 1213 "go.y"
    {
		(yyval.node) = nod(OTSTRUCT, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 198:
/* Line 1778 of yacc.c  */
#line 1220 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		fixlbrace((yyvsp[(2) - (5)].i));
	}
    break;

  case 199:
/* Line 1778 of yacc.c  */
#line 1226 "go.y"
    {
		(yyval.node) = nod(OTINTER, N, N);
		fixlbrace((yyvsp[(2) - (3)].i));
	}
    break;

  case 200:
/* Line 1778 of yacc.c  */
#line 1237 "go.y"
    {
		(yyval.node) = (yyvsp[(2) - (3)].node);
		if((yyval.node) == N)
			break;
		(yyval.node)->nbody = (yyvsp[(3) - (3)].list);
		(yyval.node)->endlineno = lineno;
		funcbody((yyval.node));
	}
    break;

  case 201:
/* Line 1778 of yacc.c  */
#line 1248 "go.y"
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

  case 202:
/* Line 1778 of yacc.c  */
#line 1277 "go.y"
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

  case 203:
/* Line 1778 of yacc.c  */
#line 1317 "go.y"
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

  case 204:
/* Line 1778 of yacc.c  */
#line 1342 "go.y"
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

  case 205:
/* Line 1778 of yacc.c  */
#line 1360 "go.y"
    {
		(yyvsp[(3) - (5)].list) = checkarglist((yyvsp[(3) - (5)].list), 1);
		(yyval.node) = nod(OTFUNC, N, N);
		(yyval.node)->list = (yyvsp[(3) - (5)].list);
		(yyval.node)->rlist = (yyvsp[(5) - (5)].list);
	}
    break;

  case 206:
/* Line 1778 of yacc.c  */
#line 1368 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 207:
/* Line 1778 of yacc.c  */
#line 1372 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
		if((yyval.list) == nil)
			(yyval.list) = list1(nod(OEMPTY, N, N));
	}
    break;

  case 208:
/* Line 1778 of yacc.c  */
#line 1380 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 209:
/* Line 1778 of yacc.c  */
#line 1384 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, (yyvsp[(1) - (1)].node)));
	}
    break;

  case 210:
/* Line 1778 of yacc.c  */
#line 1388 "go.y"
    {
		(yyvsp[(2) - (3)].list) = checkarglist((yyvsp[(2) - (3)].list), 0);
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 211:
/* Line 1778 of yacc.c  */
#line 1395 "go.y"
    {
		closurehdr((yyvsp[(1) - (1)].node));
	}
    break;

  case 212:
/* Line 1778 of yacc.c  */
#line 1401 "go.y"
    {
		(yyval.node) = closurebody((yyvsp[(3) - (4)].list));
		fixlbrace((yyvsp[(2) - (4)].i));
	}
    break;

  case 213:
/* Line 1778 of yacc.c  */
#line 1406 "go.y"
    {
		(yyval.node) = closurebody(nil);
	}
    break;

  case 214:
/* Line 1778 of yacc.c  */
#line 1417 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 215:
/* Line 1778 of yacc.c  */
#line 1421 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(2) - (3)].list));
		if(nsyntaxerrors == 0)
			testdclstack();
		nointerface = 0;
	}
    break;

  case 217:
/* Line 1778 of yacc.c  */
#line 1431 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 219:
/* Line 1778 of yacc.c  */
#line 1438 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 220:
/* Line 1778 of yacc.c  */
#line 1444 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 221:
/* Line 1778 of yacc.c  */
#line 1448 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 223:
/* Line 1778 of yacc.c  */
#line 1455 "go.y"
    {
		(yyval.list) = concat((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].list));
	}
    break;

  case 224:
/* Line 1778 of yacc.c  */
#line 1461 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 225:
/* Line 1778 of yacc.c  */
#line 1465 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 226:
/* Line 1778 of yacc.c  */
#line 1471 "go.y"
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

  case 227:
/* Line 1778 of yacc.c  */
#line 1494 "go.y"
    {
		(yyvsp[(1) - (2)].node)->val = (yyvsp[(2) - (2)].val);
		(yyval.list) = list1((yyvsp[(1) - (2)].node));
	}
    break;

  case 228:
/* Line 1778 of yacc.c  */
#line 1499 "go.y"
    {
		(yyvsp[(2) - (4)].node)->val = (yyvsp[(4) - (4)].val);
		(yyval.list) = list1((yyvsp[(2) - (4)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 229:
/* Line 1778 of yacc.c  */
#line 1505 "go.y"
    {
		(yyvsp[(2) - (3)].node)->right = nod(OIND, (yyvsp[(2) - (3)].node)->right, N);
		(yyvsp[(2) - (3)].node)->val = (yyvsp[(3) - (3)].val);
		(yyval.list) = list1((yyvsp[(2) - (3)].node));
	}
    break;

  case 230:
/* Line 1778 of yacc.c  */
#line 1511 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 231:
/* Line 1778 of yacc.c  */
#line 1518 "go.y"
    {
		(yyvsp[(3) - (5)].node)->right = nod(OIND, (yyvsp[(3) - (5)].node)->right, N);
		(yyvsp[(3) - (5)].node)->val = (yyvsp[(5) - (5)].val);
		(yyval.list) = list1((yyvsp[(3) - (5)].node));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 232:
/* Line 1778 of yacc.c  */
#line 1527 "go.y"
    {
		Node *n;

		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		n = oldname((yyvsp[(1) - (1)].sym));
		if(n->pack != N)
			n->pack->used = 1;
	}
    break;

  case 233:
/* Line 1778 of yacc.c  */
#line 1536 "go.y"
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

  case 234:
/* Line 1778 of yacc.c  */
#line 1551 "go.y"
    {
		(yyval.node) = embedded((yyvsp[(1) - (1)].sym));
	}
    break;

  case 235:
/* Line 1778 of yacc.c  */
#line 1557 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, (yyvsp[(1) - (2)].node), (yyvsp[(2) - (2)].node));
		ifacedcl((yyval.node));
	}
    break;

  case 236:
/* Line 1778 of yacc.c  */
#line 1562 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(1) - (1)].sym)));
	}
    break;

  case 237:
/* Line 1778 of yacc.c  */
#line 1566 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, oldname((yyvsp[(2) - (3)].sym)));
		yyerror("cannot parenthesize embedded type");
	}
    break;

  case 238:
/* Line 1778 of yacc.c  */
#line 1573 "go.y"
    {
		// without func keyword
		(yyvsp[(2) - (4)].list) = checkarglist((yyvsp[(2) - (4)].list), 1);
		(yyval.node) = nod(OTFUNC, fakethis(), N);
		(yyval.node)->list = (yyvsp[(2) - (4)].list);
		(yyval.node)->rlist = (yyvsp[(4) - (4)].list);
	}
    break;

  case 240:
/* Line 1778 of yacc.c  */
#line 1587 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 241:
/* Line 1778 of yacc.c  */
#line 1593 "go.y"
    {
		(yyval.node) = nod(ONONAME, N, N);
		(yyval.node)->sym = (yyvsp[(1) - (2)].sym);
		(yyval.node) = nod(OKEY, (yyval.node), (yyvsp[(2) - (2)].node));
	}
    break;

  case 243:
/* Line 1778 of yacc.c  */
#line 1602 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 244:
/* Line 1778 of yacc.c  */
#line 1606 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 245:
/* Line 1778 of yacc.c  */
#line 1611 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 246:
/* Line 1778 of yacc.c  */
#line 1615 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 247:
/* Line 1778 of yacc.c  */
#line 1623 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 249:
/* Line 1778 of yacc.c  */
#line 1628 "go.y"
    {
		(yyval.node) = liststmt((yyvsp[(1) - (1)].list));
	}
    break;

  case 251:
/* Line 1778 of yacc.c  */
#line 1633 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 257:
/* Line 1778 of yacc.c  */
#line 1644 "go.y"
    {
		(yyvsp[(1) - (2)].node) = nod(OLABEL, (yyvsp[(1) - (2)].node), N);
		(yyvsp[(1) - (2)].node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 258:
/* Line 1778 of yacc.c  */
#line 1649 "go.y"
    {
		NodeList *l;

		(yyvsp[(1) - (4)].node)->defn = (yyvsp[(4) - (4)].node);
		l = list1((yyvsp[(1) - (4)].node));
		if((yyvsp[(4) - (4)].node))
			l = list(l, (yyvsp[(4) - (4)].node));
		(yyval.node) = liststmt(l);
	}
    break;

  case 259:
/* Line 1778 of yacc.c  */
#line 1659 "go.y"
    {
		// will be converted to OFALL
		(yyval.node) = nod(OXFALL, N, N);
	}
    break;

  case 260:
/* Line 1778 of yacc.c  */
#line 1664 "go.y"
    {
		(yyval.node) = nod(OBREAK, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 261:
/* Line 1778 of yacc.c  */
#line 1668 "go.y"
    {
		(yyval.node) = nod(OCONTINUE, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 262:
/* Line 1778 of yacc.c  */
#line 1672 "go.y"
    {
		(yyval.node) = nod(OPROC, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 263:
/* Line 1778 of yacc.c  */
#line 1676 "go.y"
    {
		(yyval.node) = nod(ODEFER, (yyvsp[(2) - (2)].node), N);
	}
    break;

  case 264:
/* Line 1778 of yacc.c  */
#line 1680 "go.y"
    {
		(yyval.node) = nod(OGOTO, (yyvsp[(2) - (2)].node), N);
		(yyval.node)->sym = dclstack;  // context, for goto restrictions
	}
    break;

  case 265:
/* Line 1778 of yacc.c  */
#line 1685 "go.y"
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

  case 266:
/* Line 1778 of yacc.c  */
#line 1704 "go.y"
    {
		(yyval.list) = nil;
		if((yyvsp[(1) - (1)].node) != N)
			(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 267:
/* Line 1778 of yacc.c  */
#line 1710 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (3)].list);
		if((yyvsp[(3) - (3)].node) != N)
			(yyval.list) = list((yyval.list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 268:
/* Line 1778 of yacc.c  */
#line 1718 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 269:
/* Line 1778 of yacc.c  */
#line 1722 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 270:
/* Line 1778 of yacc.c  */
#line 1728 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 271:
/* Line 1778 of yacc.c  */
#line 1732 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 272:
/* Line 1778 of yacc.c  */
#line 1738 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 273:
/* Line 1778 of yacc.c  */
#line 1742 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 274:
/* Line 1778 of yacc.c  */
#line 1748 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 275:
/* Line 1778 of yacc.c  */
#line 1752 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 276:
/* Line 1778 of yacc.c  */
#line 1761 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 277:
/* Line 1778 of yacc.c  */
#line 1765 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 278:
/* Line 1778 of yacc.c  */
#line 1769 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 279:
/* Line 1778 of yacc.c  */
#line 1773 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 280:
/* Line 1778 of yacc.c  */
#line 1778 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 281:
/* Line 1778 of yacc.c  */
#line 1782 "go.y"
    {
		(yyval.list) = (yyvsp[(1) - (2)].list);
	}
    break;

  case 286:
/* Line 1778 of yacc.c  */
#line 1796 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 288:
/* Line 1778 of yacc.c  */
#line 1802 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 290:
/* Line 1778 of yacc.c  */
#line 1808 "go.y"
    {
		(yyval.node) = N;
	}
    break;

  case 292:
/* Line 1778 of yacc.c  */
#line 1814 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 294:
/* Line 1778 of yacc.c  */
#line 1820 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 296:
/* Line 1778 of yacc.c  */
#line 1826 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 298:
/* Line 1778 of yacc.c  */
#line 1832 "go.y"
    {
		(yyval.val).ctype = CTxxx;
	}
    break;

  case 300:
/* Line 1778 of yacc.c  */
#line 1842 "go.y"
    {
		importimport((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].val).u.sval);
	}
    break;

  case 301:
/* Line 1778 of yacc.c  */
#line 1846 "go.y"
    {
		importvar((yyvsp[(2) - (4)].sym), (yyvsp[(3) - (4)].type));
	}
    break;

  case 302:
/* Line 1778 of yacc.c  */
#line 1850 "go.y"
    {
		importconst((yyvsp[(2) - (5)].sym), types[TIDEAL], (yyvsp[(4) - (5)].node));
	}
    break;

  case 303:
/* Line 1778 of yacc.c  */
#line 1854 "go.y"
    {
		importconst((yyvsp[(2) - (6)].sym), (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].node));
	}
    break;

  case 304:
/* Line 1778 of yacc.c  */
#line 1858 "go.y"
    {
		importtype((yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].type));
	}
    break;

  case 305:
/* Line 1778 of yacc.c  */
#line 1862 "go.y"
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

  case 306:
/* Line 1778 of yacc.c  */
#line 1882 "go.y"
    {
		(yyval.sym) = (yyvsp[(1) - (1)].sym);
		structpkg = (yyval.sym)->pkg;
	}
    break;

  case 307:
/* Line 1778 of yacc.c  */
#line 1889 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
		importsym((yyvsp[(1) - (1)].sym), OTYPE);
	}
    break;

  case 313:
/* Line 1778 of yacc.c  */
#line 1909 "go.y"
    {
		(yyval.type) = pkgtype((yyvsp[(1) - (1)].sym));
	}
    break;

  case 314:
/* Line 1778 of yacc.c  */
#line 1913 "go.y"
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

  case 315:
/* Line 1778 of yacc.c  */
#line 1923 "go.y"
    {
		(yyval.type) = aindex(N, (yyvsp[(3) - (3)].type));
	}
    break;

  case 316:
/* Line 1778 of yacc.c  */
#line 1927 "go.y"
    {
		(yyval.type) = aindex(nodlit((yyvsp[(2) - (4)].val)), (yyvsp[(4) - (4)].type));
	}
    break;

  case 317:
/* Line 1778 of yacc.c  */
#line 1931 "go.y"
    {
		(yyval.type) = maptype((yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].type));
	}
    break;

  case 318:
/* Line 1778 of yacc.c  */
#line 1935 "go.y"
    {
		(yyval.type) = tostruct((yyvsp[(3) - (4)].list));
	}
    break;

  case 319:
/* Line 1778 of yacc.c  */
#line 1939 "go.y"
    {
		(yyval.type) = tointerface((yyvsp[(3) - (4)].list));
	}
    break;

  case 320:
/* Line 1778 of yacc.c  */
#line 1943 "go.y"
    {
		(yyval.type) = ptrto((yyvsp[(2) - (2)].type));
	}
    break;

  case 321:
/* Line 1778 of yacc.c  */
#line 1947 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(2) - (2)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 322:
/* Line 1778 of yacc.c  */
#line 1953 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (4)].type);
		(yyval.type)->chan = Cboth;
	}
    break;

  case 323:
/* Line 1778 of yacc.c  */
#line 1959 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Csend;
	}
    break;

  case 324:
/* Line 1778 of yacc.c  */
#line 1967 "go.y"
    {
		(yyval.type) = typ(TCHAN);
		(yyval.type)->type = (yyvsp[(3) - (3)].type);
		(yyval.type)->chan = Crecv;
	}
    break;

  case 325:
/* Line 1778 of yacc.c  */
#line 1975 "go.y"
    {
		(yyval.type) = functype(nil, (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list));
	}
    break;

  case 326:
/* Line 1778 of yacc.c  */
#line 1981 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(2) - (3)].type)));
		if((yyvsp[(1) - (3)].sym))
			(yyval.node)->left = newname((yyvsp[(1) - (3)].sym));
		(yyval.node)->val = (yyvsp[(3) - (3)].val);
	}
    break;

  case 327:
/* Line 1778 of yacc.c  */
#line 1988 "go.y"
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

  case 328:
/* Line 1778 of yacc.c  */
#line 2004 "go.y"
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

  case 329:
/* Line 1778 of yacc.c  */
#line 2022 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, newname((yyvsp[(1) - (5)].sym)), typenod(functype(fakethis(), (yyvsp[(3) - (5)].list), (yyvsp[(5) - (5)].list))));
	}
    break;

  case 330:
/* Line 1778 of yacc.c  */
#line 2026 "go.y"
    {
		(yyval.node) = nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type)));
	}
    break;

  case 331:
/* Line 1778 of yacc.c  */
#line 2031 "go.y"
    {
		(yyval.list) = nil;
	}
    break;

  case 333:
/* Line 1778 of yacc.c  */
#line 2038 "go.y"
    {
		(yyval.list) = (yyvsp[(2) - (3)].list);
	}
    break;

  case 334:
/* Line 1778 of yacc.c  */
#line 2042 "go.y"
    {
		(yyval.list) = list1(nod(ODCLFIELD, N, typenod((yyvsp[(1) - (1)].type))));
	}
    break;

  case 335:
/* Line 1778 of yacc.c  */
#line 2052 "go.y"
    {
		(yyval.node) = nodlit((yyvsp[(1) - (1)].val));
	}
    break;

  case 336:
/* Line 1778 of yacc.c  */
#line 2056 "go.y"
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

  case 337:
/* Line 1778 of yacc.c  */
#line 2071 "go.y"
    {
		(yyval.node) = oldname(pkglookup((yyvsp[(1) - (1)].sym)->name, builtinpkg));
		if((yyval.node)->op != OLITERAL)
			yyerror("bad constant %S", (yyval.node)->sym);
	}
    break;

  case 339:
/* Line 1778 of yacc.c  */
#line 2080 "go.y"
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

  case 342:
/* Line 1778 of yacc.c  */
#line 2096 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 343:
/* Line 1778 of yacc.c  */
#line 2100 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 344:
/* Line 1778 of yacc.c  */
#line 2106 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 345:
/* Line 1778 of yacc.c  */
#line 2110 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;

  case 346:
/* Line 1778 of yacc.c  */
#line 2116 "go.y"
    {
		(yyval.list) = list1((yyvsp[(1) - (1)].node));
	}
    break;

  case 347:
/* Line 1778 of yacc.c  */
#line 2120 "go.y"
    {
		(yyval.list) = list((yyvsp[(1) - (3)].list), (yyvsp[(3) - (3)].node));
	}
    break;


/* Line 1778 of yacc.c  */
#line 5055 "y.tab.c"
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

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


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

#if !defined yyoverflow || YYERROR_VERBOSE
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


/* Line 2041 of yacc.c  */
#line 2124 "go.y"


static void
fixlbrace(int lbr)
{
	// If the opening brace was an LBODY,
	// set up for another one now that we're done.
	// See comment in lex.c about loophack.
	if(lbr == LBODY)
		loophack = 1;
}

