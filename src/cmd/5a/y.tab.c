
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
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
#define YYBISON_VERSION "2.4.1"

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

/* Line 189 of yacc.c  */
#line 31 "a.y"

#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and a.h re-#defines getc */
#include <libc.h>
#include "a.h"


/* Line 189 of yacc.c  */
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
# define YYERROR_VERBOSE 0
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
     LTYPEF = 272,
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
     LR = 289,
     LREG = 290,
     LF = 291,
     LFREG = 292,
     LC = 293,
     LCREG = 294,
     LPSR = 295,
     LFCR = 296,
     LCOND = 297,
     LS = 298,
     LAT = 299,
     LFCONST = 300,
     LSCONST = 301,
     LNAME = 302,
     LLAB = 303,
     LVAR = 304
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
#define LTYPEF 272
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
#define LR 289
#define LREG 290
#define LF 291
#define LFREG 292
#define LC 293
#define LCREG 294
#define LPSR 295
#define LFCR 296
#define LCOND 297
#define LS 298
#define LAT 299
#define LFCONST 300
#define LSCONST 301
#define LNAME 302
#define LLAB 303
#define LVAR 304




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 214 of yacc.c  */
#line 38 "a.y"

	Sym	*sym;
	int32	lval;
	double	dval;
	char	sval[8];
	Gen	gen;



/* Line 214 of yacc.c  */
#line 225 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 237 "y.tab.c"

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
# if YYENABLE_NLS
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

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   603

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  70
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  35
/* YYNRULES -- Number of rules.  */
#define YYNRULES  130
/* YYNRULES -- Number of states.  */
#define YYNSTATES  329

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   304

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    68,    12,     5,     2,
      66,    67,    10,     8,    63,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    60,    62,
       6,    61,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    64,     2,    65,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    69,     2,     2,     2,
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
      55,    56,    57,    58,    59
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     5,     9,    10,    15,    16,    21,
      26,    31,    33,    36,    39,    47,    54,    60,    66,    72,
      77,    82,    86,    90,    95,   102,   110,   118,   126,   133,
     140,   144,   149,   156,   163,   168,   172,   178,   184,   192,
     199,   212,   220,   230,   233,   236,   237,   240,   243,   244,
     247,   252,   255,   258,   261,   264,   269,   272,   274,   277,
     281,   283,   287,   291,   293,   295,   297,   302,   304,   306,
     308,   310,   312,   314,   316,   320,   322,   327,   329,   334,
     336,   338,   340,   342,   345,   347,   353,   358,   363,   368,
     373,   375,   377,   379,   381,   386,   388,   390,   392,   397,
     399,   401,   403,   408,   413,   419,   427,   428,   431,   434,
     436,   438,   440,   442,   444,   447,   450,   453,   457,   458,
     461,   463,   467,   471,   475,   479,   483,   488,   493,   497,
     501
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      71,     0,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,
      58,    60,    74,    73,    -1,    -1,    57,    60,    75,    73,
      -1,    57,    61,   104,    62,    -1,    59,    61,   104,    62,
      -1,    62,    -1,    76,    62,    -1,     1,    62,    -1,    13,
      77,    88,    63,    95,    63,    90,    -1,    13,    77,    88,
      63,    95,    63,    -1,    13,    77,    88,    63,    90,    -1,
      14,    77,    88,    63,    90,    -1,    15,    77,    83,    63,
      83,    -1,    16,    77,    78,    79,    -1,    16,    77,    78,
      84,    -1,    36,    78,    85,    -1,    17,    78,    79,    -1,
      18,    77,    78,    83,    -1,    19,    77,    88,    63,    95,
      78,    -1,    20,    77,    86,    63,    64,    82,    65,    -1,
      20,    77,    64,    82,    65,    63,    86,    -1,    21,    77,
      90,    63,    85,    63,    90,    -1,    21,    77,    90,    63,
      85,    78,    -1,    21,    77,    78,    85,    63,    90,    -1,
      22,    77,    78,    -1,    23,    99,    63,    89,    -1,    23,
      99,    63,   102,    63,    89,    -1,    24,    99,    11,   102,
      63,    80,    -1,    25,    77,    90,    78,    -1,    29,    78,
      80,    -1,    30,    77,    98,    63,    98,    -1,    32,    77,
      97,    63,    98,    -1,    32,    77,    97,    63,    47,    63,
      98,    -1,    33,    77,    98,    63,    98,    78,    -1,    31,
      77,   102,    63,   104,    63,    95,    63,    96,    63,    96,
     103,    -1,    34,    77,    90,    63,    90,    63,    91,    -1,
      35,    77,    90,    63,    90,    63,    90,    63,    95,    -1,
      37,    87,    -1,    26,    78,    -1,    -1,    77,    52,    -1,
      77,    53,    -1,    -1,    63,    78,    -1,   102,    66,    42,
      67,    -1,    57,   100,    -1,    58,   100,    -1,    68,   102,
      -1,    68,    87,    -1,    68,    10,    68,    87,    -1,    68,
      56,    -1,    81,    -1,    68,    55,    -1,    68,     9,    55,
      -1,    95,    -1,    95,     9,    95,    -1,    95,    78,    82,
      -1,    90,    -1,    80,    -1,    92,    -1,    92,    66,    95,
      67,    -1,    50,    -1,    51,    -1,   102,    -1,    87,    -1,
      98,    -1,    85,    -1,    99,    -1,    66,    95,    67,    -1,
      85,    -1,   102,    66,    94,    67,    -1,    99,    -1,    99,
      66,    94,    67,    -1,    86,    -1,    90,    -1,    89,    -1,
      92,    -1,    68,   102,    -1,    95,    -1,    66,    95,    63,
      95,    67,    -1,    95,     6,     6,    93,    -1,    95,     7,
       7,    93,    -1,    95,     9,     7,    93,    -1,    95,    54,
       7,    93,    -1,    95,    -1,   102,    -1,    45,    -1,    42,
      -1,    44,    66,   104,    67,    -1,    94,    -1,    39,    -1,
      49,    -1,    48,    66,   104,    67,    -1,    98,    -1,    81,
      -1,    47,    -1,    46,    66,   102,    67,    -1,   102,    66,
     101,    67,    -1,    57,   100,    66,   101,    67,    -1,    57,
       6,     7,   100,    66,    40,    67,    -1,    -1,     8,   102,
      -1,     9,   102,    -1,    40,    -1,    39,    -1,    41,    -1,
      38,    -1,    59,    -1,     9,   102,    -1,     8,   102,    -1,
      69,   102,    -1,    66,   104,    67,    -1,    -1,    63,   104,
      -1,   102,    -1,   104,     8,   104,    -1,   104,     9,   104,
      -1,   104,    10,   104,    -1,   104,    11,   104,    -1,   104,
      12,   104,    -1,   104,     6,     6,   104,    -1,   104,     7,
       7,   104,    -1,   104,     5,   104,    -1,   104,     4,   104,
      -1,   104,     3,   104,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    67,    67,    69,    68,    76,    75,    83,    82,    88,
      93,    99,   100,   101,   107,   111,   115,   122,   129,   136,
     140,   147,   154,   161,   168,   175,   184,   196,   200,   204,
     211,   218,   222,   229,   236,   243,   250,   254,   258,   262,
     269,   291,   299,   308,   315,   321,   324,   328,   333,   334,
     337,   343,   352,   360,   366,   371,   376,   382,   385,   391,
     399,   403,   412,   418,   419,   420,   421,   426,   432,   438,
     444,   445,   448,   449,   457,   466,   467,   476,   477,   483,
     486,   487,   488,   490,   498,   506,   515,   521,   527,   533,
     541,   547,   555,   556,   560,   568,   569,   575,   576,   584,
     585,   588,   594,   602,   610,   618,   628,   631,   635,   641,
     642,   643,   646,   647,   651,   655,   659,   663,   669,   672,
     678,   679,   683,   687,   691,   695,   699,   703,   707,   711,
     715
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
  "LTYPEC", "LTYPED", "LTYPEE", "LTYPEF", "LTYPEG", "LTYPEH", "LTYPEI",
  "LTYPEJ", "LTYPEK", "LTYPEL", "LTYPEM", "LTYPEN", "LTYPEBX", "LTYPEPLD",
  "LCONST", "LSP", "LSB", "LFP", "LPC", "LTYPEX", "LR", "LREG", "LF",
  "LFREG", "LC", "LCREG", "LPSR", "LFCR", "LCOND", "LS", "LAT", "LFCONST",
  "LSCONST", "LNAME", "LLAB", "LVAR", "':'", "'='", "';'", "','", "'['",
  "']'", "'('", "')'", "'$'", "'~'", "$accept", "prog", "$@1", "line",
  "$@2", "$@3", "inst", "cond", "comma", "rel", "ximm", "fcon", "reglist",
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
      58,    61,    59,    44,    91,    93,    40,    41,    36,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    70,    71,    72,    71,    74,    73,    75,    73,    73,
      73,    73,    73,    73,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    77,    77,    77,    78,    78,
      79,    79,    79,    80,    80,    80,    80,    80,    81,    81,
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
       0,     2,     0,     0,     3,     0,     4,     0,     4,     4,
       4,     1,     2,     2,     7,     6,     5,     5,     5,     4,
       4,     3,     3,     4,     6,     7,     7,     7,     6,     6,
       3,     4,     6,     6,     4,     3,     5,     5,     7,     6,
      12,     7,     9,     2,     2,     0,     2,     2,     0,     2,
       4,     2,     2,     2,     2,     4,     2,     1,     2,     3,
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
       2,     3,     1,     0,     0,    45,    45,    45,    45,    48,
      45,    45,    45,    45,    45,     0,     0,    45,    48,    48,
      45,    45,    45,    45,    45,    45,    48,     0,     0,     0,
       0,    11,     4,     0,    13,     0,     0,     0,    48,    48,
       0,    48,     0,     0,    48,    48,     0,     0,   112,   106,
     113,     0,     0,     0,     0,     0,     0,    44,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    75,    79,    43,
      77,     0,     7,     0,     5,     0,    12,    96,    93,     0,
      92,    46,    47,     0,     0,    81,    80,    82,    95,    84,
       0,     0,   101,    67,    68,     0,    64,    57,     0,    70,
      63,    65,    71,    69,     0,    49,   106,   106,    22,     0,
       0,     0,     0,     0,     0,     0,     0,    84,    30,   115,
     114,     0,     0,     0,     0,   120,     0,   116,     0,     0,
       0,    48,    35,     0,     0,     0,   100,     0,    99,     0,
       0,     0,     0,    21,     0,     0,     0,     0,     0,     0,
       0,     0,    83,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    58,    56,    54,    53,     0,     0,   106,    19,
      20,    72,    73,     0,    51,    52,     0,    23,     0,     0,
      48,     0,     0,     0,     0,   106,   107,   108,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   117,
      31,     0,   110,   109,   111,     0,     0,    34,     0,     0,
       0,     0,     0,     0,     0,    74,     0,     0,     8,     9,
       6,    10,     0,    16,    84,     0,     0,     0,     0,    17,
       0,    59,     0,    18,     0,    51,     0,     0,    48,     0,
       0,     0,     0,     0,    48,     0,     0,   130,   129,   128,
       0,     0,   121,   122,   123,   124,   125,     0,   103,     0,
      36,     0,   101,    37,    48,     0,     0,    78,    76,    94,
      15,    86,    90,    91,    87,    88,    89,   102,    55,    66,
      50,    24,     0,    61,    62,     0,    29,    48,    28,     0,
     104,   126,   127,    32,    33,     0,     0,    39,     0,     0,
      14,    26,    25,    27,     0,     0,    38,     0,    41,     0,
     105,     0,     0,     0,     0,    97,     0,     0,    42,     0,
       0,     0,     0,   118,    85,    98,     0,    40,   119
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    32,   149,   147,    33,    35,   105,   108,
      96,    97,   179,    98,   170,    67,    68,    99,    84,    85,
      86,   308,    87,   271,    88,   117,   316,   137,   102,    70,
     124,   205,   125,   327,   126
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -114
static const yytype_int16 yypact[] =
{
    -114,    19,  -114,   311,   -20,  -114,  -114,  -114,  -114,   -36,
    -114,  -114,  -114,  -114,  -114,   413,   413,  -114,   -36,   -36,
    -114,  -114,  -114,  -114,  -114,  -114,   -36,   427,   -22,    -9,
      -7,  -114,  -114,    11,  -114,   480,   480,   341,    45,   -36,
     229,    45,   480,   210,   436,    45,   435,   435,  -114,   152,
    -114,   435,   435,    25,    15,    79,   517,  -114,    24,   134,
     393,   197,   134,   517,   517,    28,    71,  -114,  -114,  -114,
      35,    58,  -114,   435,  -114,   435,  -114,  -114,  -114,    69,
    -114,  -114,  -114,   435,    51,  -114,  -114,  -114,  -114,    46,
      57,    75,  -114,  -114,  -114,   119,  -114,  -114,    82,  -114,
    -114,    88,  -114,    58,   368,  -114,   159,   159,  -114,   113,
     373,   114,   411,   120,   123,    28,   145,  -114,  -114,  -114,
    -114,   193,   435,   435,   173,  -114,    54,  -114,   395,   111,
     435,   -36,  -114,   178,   182,    21,  -114,   198,  -114,   201,
     207,   212,   411,  -114,   206,   168,   558,   311,   310,   311,
     506,   435,  -114,   411,   271,   274,   285,   286,   411,   435,
      87,   228,  -114,  -114,  -114,    58,   373,   411,   152,  -114,
    -114,  -114,  -114,   231,  -114,  -114,   257,  -114,   411,   235,
       5,   259,   168,   275,    28,   159,  -114,  -114,   111,   435,
     435,   435,   333,   344,   435,   435,   435,   435,   435,  -114,
    -114,   289,  -114,  -114,  -114,   300,   308,  -114,   151,   435,
     319,   176,   151,   411,   411,  -114,   317,   322,  -114,  -114,
    -114,  -114,   224,  -114,   312,    71,    71,    71,    71,  -114,
     323,  -114,   427,  -114,   328,   173,    64,   329,   -36,   315,
     411,   411,   411,   411,   334,   339,   332,   577,   247,   584,
     435,   435,   273,   273,  -114,  -114,  -114,   340,  -114,    24,
    -114,   350,   351,  -114,   -36,   353,   365,  -114,  -114,  -114,
     411,  -114,  -114,  -114,  -114,  -114,  -114,  -114,  -114,  -114,
    -114,  -114,   439,  -114,  -114,   364,  -114,   157,  -114,   398,
    -114,   518,   518,  -114,  -114,   411,   151,  -114,   374,   411,
    -114,  -114,  -114,  -114,   382,   394,  -114,   411,  -114,   397,
    -114,   155,   403,   411,   392,  -114,   404,   411,  -114,   435,
     155,   401,   299,   406,  -114,  -114,   435,  -114,   568
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -114,  -114,  -114,   -78,  -114,  -114,  -114,   529,     2,   367,
     -50,   415,    48,   -88,  -114,   -48,   -40,   -21,    47,  -113,
     -19,  -114,   -28,   137,   -59,   -35,   154,  -114,   -49,    18,
     -83,   295,   -11,  -114,   -25
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -61
static const yytype_int16 yytable[] =
{
      89,    89,    89,   113,    54,    54,    69,    89,   132,   101,
     133,    40,   138,   139,   240,   200,    71,   143,   100,     2,
      57,    58,   177,   174,   175,   116,   103,    39,    65,   109,
     210,   144,   114,    53,    55,   119,   120,   131,    72,    73,
     104,   127,    34,   110,   140,   141,   115,   118,   148,   134,
     150,    74,   154,   155,    75,   156,   171,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   183,    39,   218,
     -60,   220,   152,    76,   164,    89,   162,   180,   233,    46,
      47,   129,   101,    90,   165,   235,   216,   217,   128,   111,
     130,   100,    95,   173,   142,    46,    47,    81,    82,   103,
     157,   145,   245,   202,   203,   204,   237,   144,    39,    48,
      77,   186,   187,    78,   153,    79,    80,   201,   224,   206,
     158,   199,   172,   217,   146,    48,   222,    46,   160,   161,
      50,    89,   234,   207,   223,   151,   244,    51,   101,   229,
      52,   159,   231,   238,   293,   166,    50,   100,   230,   120,
     202,   203,   204,    51,   167,   103,    52,    48,   121,   260,
     122,   123,   263,   264,   247,   248,   249,   122,   123,   252,
     253,   254,   255,   256,   162,   163,    49,   178,    50,   176,
      91,    92,   241,   181,   261,    66,    81,    82,    52,   182,
     272,   272,   272,   272,   265,   266,    77,    91,    92,    78,
     185,    79,    80,   314,   315,   283,   180,   180,   184,   294,
      78,   278,    79,    80,   273,   273,   273,   273,    46,    47,
      39,    71,    91,   262,   286,   291,   292,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,    46,    47,   188,
     281,   208,   301,    91,    92,   209,   288,   306,    48,    81,
      82,   300,   191,   192,   193,   194,   195,   196,   197,   198,
     305,   211,    81,    82,   212,   135,   297,    48,   303,    50,
     213,   114,   312,   215,   112,   214,    66,   225,   318,    52,
     309,   226,   321,   196,   197,   198,   106,   107,    50,   284,
     285,   269,   227,   228,   322,    51,   232,   236,    52,   237,
     239,   328,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,     4,   189,   190,   191,   192,   193,   194,   195,
     196,   197,   198,   242,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,   243,   250,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    46,
      47,   251,   257,   189,   190,   191,   192,   193,   194,   195,
     196,   197,   198,   274,   275,   276,   325,   258,    28,    29,
      30,   259,   219,    31,   231,   270,    46,    47,   282,    48,
      77,    46,    47,    78,   267,    79,    80,    91,    92,   268,
     277,    93,    94,    81,    82,   279,   280,   287,    49,   290,
      50,    46,    47,    46,    47,   289,    48,    66,    83,    95,
      52,    48,    77,   295,   296,    78,   298,    79,    80,    91,
      92,    46,    47,    93,    94,   168,   107,    50,   299,   302,
      49,    48,    50,    48,    66,    46,    47,    52,   304,    66,
     307,    95,    52,    46,    47,    81,    82,    46,    47,   310,
      77,    48,    50,    78,    50,    79,    80,   311,   319,    51,
     313,    51,    52,    83,    52,    48,   317,   320,   324,   326,
      49,   169,    50,    48,   323,    77,   136,    48,    78,    51,
      79,    80,    52,   246,    49,     0,    50,     0,    81,    82,
       0,     0,     0,    66,    50,     0,    52,     0,    50,    39,
       0,    51,     0,     0,    52,    66,     0,     0,    52,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   198,    77,
       0,     0,    78,     0,    79,    80,   194,   195,   196,   197,
     198,     0,    81,    82,     0,    36,    37,    38,     0,    41,
      42,    43,    44,    45,     0,     0,    56,     0,    83,    59,
      60,    61,    62,    63,    64,     0,    77,     0,     0,    78,
       0,    79,    80,     0,     0,     0,     0,     0,   221,    81,
      82,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     198,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     192,   193,   194,   195,   196,   197,   198,   202,   203,   204,
      78,     0,    79,    80
};

static const yytype_int16 yycheck[] =
{
      35,    36,    37,    43,    15,    16,    27,    42,    58,    37,
      59,     9,    61,    62,     9,   128,    27,    65,    37,     0,
      18,    19,   110,   106,   107,    44,    37,    63,    26,    40,
       9,    66,    43,    15,    16,    46,    47,    56,    60,    61,
      38,    52,    62,    41,    63,    64,    44,    45,    73,    60,
      75,    60,     6,     7,    61,     9,   104,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,   115,    63,   147,
      65,   149,    83,    62,    95,   110,    55,   112,   166,     8,
       9,    66,   110,    36,    95,   168,   145,   146,    63,    42,
      11,   110,    68,   104,    66,     8,     9,    52,    53,   110,
      54,    66,   185,    39,    40,    41,    42,   142,    63,    38,
      39,   122,   123,    42,    63,    44,    45,   128,   153,   130,
      63,    67,   104,   182,    66,    38,   151,     8,     9,    10,
      59,   166,   167,   131,   153,    66,   184,    66,   166,   158,
      69,    66,    55,   178,   257,    63,    59,   166,   159,   160,
      39,    40,    41,    66,    66,   166,    69,    38,     6,   208,
       8,     9,   211,   212,   189,   190,   191,     8,     9,   194,
     195,   196,   197,   198,    55,    56,    57,    63,    59,    66,
      46,    47,   180,    63,   209,    66,    52,    53,    69,    66,
     225,   226,   227,   228,   213,   214,    39,    46,    47,    42,
       7,    44,    45,    48,    49,   240,   241,   242,    63,   259,
      42,   232,    44,    45,   225,   226,   227,   228,     8,     9,
      63,   232,    46,    47,   243,   250,   251,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,     8,     9,    66,
     238,    63,   282,    46,    47,    63,   244,   296,    38,    52,
      53,   270,     5,     6,     7,     8,     9,    10,    11,    12,
     295,    63,    52,    53,    63,    68,   264,    38,   287,    59,
      63,   282,   307,    67,    64,    63,    66,     6,   313,    69,
     299,     7,   317,    10,    11,    12,    57,    58,    59,   241,
     242,    67,     7,     7,   319,    66,    68,    66,    69,    42,
      65,   326,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,     1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    64,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    63,     6,
      29,    30,    31,    32,    33,    34,    35,    36,    37,     8,
       9,     7,    63,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,   226,   227,   228,    67,    67,    57,    58,
      59,    63,    62,    62,    55,    63,     8,     9,    63,    38,
      39,     8,     9,    42,    67,    44,    45,    46,    47,    67,
      67,    50,    51,    52,    53,    67,    67,    63,    57,    67,
      59,     8,     9,     8,     9,    66,    38,    66,    68,    68,
      69,    38,    39,    63,    63,    42,    63,    44,    45,    46,
      47,     8,     9,    50,    51,    57,    58,    59,    63,    65,
      57,    38,    59,    38,    66,     8,     9,    69,    40,    66,
      66,    68,    69,     8,     9,    52,    53,     8,     9,    67,
      39,    38,    59,    42,    59,    44,    45,    63,    66,    66,
      63,    66,    69,    68,    69,    38,    63,    63,    67,    63,
      57,   104,    59,    38,   320,    39,    61,    38,    42,    66,
      44,    45,    69,   188,    57,    -1,    59,    -1,    52,    53,
      -1,    -1,    -1,    66,    59,    -1,    69,    -1,    59,    63,
      -1,    66,    -1,    -1,    69,    66,    -1,    -1,    69,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    39,
      -1,    -1,    42,    -1,    44,    45,     8,     9,    10,    11,
      12,    -1,    52,    53,    -1,     6,     7,     8,    -1,    10,
      11,    12,    13,    14,    -1,    -1,    17,    -1,    68,    20,
      21,    22,    23,    24,    25,    -1,    39,    -1,    -1,    42,
      -1,    44,    45,    -1,    -1,    -1,    -1,    -1,    62,    52,
      53,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,     4,     5,     6,     7,     8,     9,    10,    11,    12,
       6,     7,     8,     9,    10,    11,    12,    39,    40,    41,
      42,    -1,    44,    45
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    71,     0,    72,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    57,    58,
      59,    62,    73,    76,    62,    77,    77,    77,    77,    63,
      78,    77,    77,    77,    77,    77,     8,     9,    38,    57,
      59,    66,    69,    99,   102,    99,    77,    78,    78,    77,
      77,    77,    77,    77,    77,    78,    66,    85,    86,    87,
      99,   102,    60,    61,    60,    61,    62,    39,    42,    44,
      45,    52,    53,    68,    88,    89,    90,    92,    94,    95,
      88,    46,    47,    50,    51,    68,    80,    81,    83,    87,
      90,    92,    98,   102,    78,    78,    57,    58,    79,   102,
      78,    88,    64,    86,   102,    78,    90,    95,    78,   102,
     102,     6,     8,     9,   100,   102,   104,   102,    63,    66,
      11,    90,    80,    98,   102,    68,    81,    97,    98,    98,
      90,    90,    66,    85,    95,    66,    66,    75,   104,    74,
     104,    66,   102,    63,     6,     7,     9,    54,    63,    66,
       9,    10,    55,    56,    87,   102,    63,    66,    57,    79,
      84,    85,    99,   102,   100,   100,    66,    83,    63,    82,
      95,    63,    66,    85,    63,     7,   102,   102,    66,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    67,
      89,   102,    39,    40,    41,   101,   102,    78,    63,    63,
       9,    63,    63,    63,    63,    67,    94,    94,    73,    62,
      73,    62,   104,    90,    95,     6,     7,     7,     7,    90,
     102,    55,    68,    83,    95,   100,    66,    42,    95,    65,
       9,    78,    64,    63,    85,   100,   101,   104,   104,   104,
       6,     7,   104,   104,   104,   104,   104,    63,    67,    63,
      98,   104,    47,    98,    98,    90,    90,    67,    67,    67,
      63,    93,    95,   102,    93,    93,    93,    67,    87,    67,
      67,    78,    63,    95,    82,    82,    90,    63,    78,    66,
      67,   104,   104,    89,    80,    63,    63,    78,    63,    63,
      90,    86,    65,    90,    40,    95,    98,    66,    91,    90,
      67,    63,    95,    63,    48,    49,    96,    63,    95,    66,
      63,    95,   104,    96,    67,    67,    63,   103,   104
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
# if YYLTYPE_IS_TRIVIAL
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


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

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
  if (yyn == YYPACT_NINF)
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
      if (yyn == 0 || yyn == YYTABLE_NINF)
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
        case 3:

/* Line 1455 of yacc.c  */
#line 69 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:

/* Line 1455 of yacc.c  */
#line 76 "a.y"
    {
		if((yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->name);
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 7:

/* Line 1455 of yacc.c  */
#line 83 "a.y"
    {
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 9:

/* Line 1455 of yacc.c  */
#line 89 "a.y"
    {
		(yyvsp[(1) - (4)].sym)->type = LVAR;
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 10:

/* Line 1455 of yacc.c  */
#line 94 "a.y"
    {
		if((yyvsp[(1) - (4)].sym)->value != (yyvsp[(3) - (4)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (4)].sym)->name);
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 14:

/* Line 1455 of yacc.c  */
#line 108 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].gen), (yyvsp[(5) - (7)].lval), &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 15:

/* Line 1455 of yacc.c  */
#line 112 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].gen), (yyvsp[(5) - (6)].lval), &nullgen);
	}
    break;

  case 16:

/* Line 1455 of yacc.c  */
#line 116 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 17:

/* Line 1455 of yacc.c  */
#line 123 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 18:

/* Line 1455 of yacc.c  */
#line 130 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 137 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].gen));
	}
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 141 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].gen));
	}
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 148 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].gen));
	}
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 155 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].gen));
	}
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 162 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].gen));
	}
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 169 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].gen), (yyvsp[(5) - (6)].lval), &nullgen);
	}
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 176 "a.y"
    {
		Gen g;

		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(6) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].gen), NREG, &g);
	}
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 185 "a.y"
    {
		Gen g;

		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(4) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &g, NREG, &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 197 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(5) - (7)].gen), (yyvsp[(3) - (7)].gen).reg, &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 201 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(5) - (6)].gen), (yyvsp[(3) - (6)].gen).reg, &(yyvsp[(3) - (6)].gen));
	}
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 205 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(4) - (6)].gen), (yyvsp[(6) - (6)].gen).reg, &(yyvsp[(6) - (6)].gen));
	}
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 212 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), (yyvsp[(2) - (3)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 219 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].gen), 0, &(yyvsp[(4) - (4)].gen));
	}
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 223 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].gen), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].gen));
	}
    break;

  case 33:

/* Line 1455 of yacc.c  */
#line 230 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].gen), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].gen));
	}
    break;

  case 34:

/* Line 1455 of yacc.c  */
#line 237 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &(yyvsp[(3) - (4)].gen), NREG, &nullgen);
	}
    break;

  case 35:

/* Line 1455 of yacc.c  */
#line 244 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].gen));
	}
    break;

  case 36:

/* Line 1455 of yacc.c  */
#line 251 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 37:

/* Line 1455 of yacc.c  */
#line 255 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].gen), NREG, &(yyvsp[(5) - (5)].gen));
	}
    break;

  case 38:

/* Line 1455 of yacc.c  */
#line 259 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].gen), (yyvsp[(5) - (7)].lval), &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 39:

/* Line 1455 of yacc.c  */
#line 263 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].gen), (yyvsp[(5) - (6)].gen).reg, &nullgen);
	}
    break;

  case 40:

/* Line 1455 of yacc.c  */
#line 270 "a.y"
    {
		Gen g;

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
		outcode(AWORD, Always, &nullgen, NREG, &g);
	}
    break;

  case 41:

/* Line 1455 of yacc.c  */
#line 292 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].gen), (yyvsp[(5) - (7)].gen).reg, &(yyvsp[(7) - (7)].gen));
	}
    break;

  case 42:

/* Line 1455 of yacc.c  */
#line 300 "a.y"
    {
		(yyvsp[(7) - (9)].gen).type = D_REGREG2;
		(yyvsp[(7) - (9)].gen).offset = (yyvsp[(9) - (9)].lval);
		outcode((yyvsp[(1) - (9)].lval), (yyvsp[(2) - (9)].lval), &(yyvsp[(3) - (9)].gen), (yyvsp[(5) - (9)].gen).reg, &(yyvsp[(7) - (9)].gen));
	}
    break;

  case 43:

/* Line 1455 of yacc.c  */
#line 309 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &(yyvsp[(2) - (2)].gen), NREG, &nullgen);
	}
    break;

  case 44:

/* Line 1455 of yacc.c  */
#line 316 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &nullgen, NREG, &nullgen);
	}
    break;

  case 45:

/* Line 1455 of yacc.c  */
#line 321 "a.y"
    {
		(yyval.lval) = Always;
	}
    break;

  case 46:

/* Line 1455 of yacc.c  */
#line 325 "a.y"
    {
		(yyval.lval) = ((yyvsp[(1) - (2)].lval) & ~C_SCOND) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 47:

/* Line 1455 of yacc.c  */
#line 329 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (2)].lval) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 338 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 344 "a.y"
    {
		(yyval.gen) = nullgen;
		if(pass == 2)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->name);
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 353 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 361 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST;
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 367 "a.y"
    {
		(yyval.gen) = (yyvsp[(2) - (2)].gen);
		(yyval.gen).type = D_CONST;
	}
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 372 "a.y"
    {
		(yyval.gen) = (yyvsp[(4) - (4)].gen);
		(yyval.gen).type = D_OCONST;
	}
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 377 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SCONST;
		memcpy((yyval.gen).sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.gen).sval));
	}
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 386 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 392 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 400 "a.y"
    {
		(yyval.lval) = 1 << (yyvsp[(1) - (1)].lval);
	}
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 404 "a.y"
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

/* Line 1455 of yacc.c  */
#line 413 "a.y"
    {
		(yyval.lval) = (1<<(yyvsp[(1) - (3)].lval)) | (yyvsp[(3) - (3)].lval);
	}
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 422 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (4)].gen);
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 427 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_PSR;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 433 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FPCR;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 439 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 450 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (1)].gen);
		if((yyvsp[(1) - (1)].gen).name != D_EXTERN && (yyvsp[(1) - (1)].gen).name != D_STATIC) {
		}
	}
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 458 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).reg = (yyvsp[(2) - (3)].lval);
		(yyval.gen).offset = 0;
	}
    break;

  case 76:

/* Line 1455 of yacc.c  */
#line 468 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 78:

/* Line 1455 of yacc.c  */
#line 478 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (4)].gen);
		(yyval.gen).type = D_OREG;
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 491 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST;
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 499 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_REG;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 85:

/* Line 1455 of yacc.c  */
#line 507 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_REGREG;
		(yyval.gen).reg = (yyvsp[(2) - (5)].lval);
		(yyval.gen).offset = (yyvsp[(4) - (5)].lval);
	}
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 516 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (0 << 5);
	}
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 522 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (1 << 5);
	}
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 528 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (2 << 5);
	}
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 534 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (3 << 5);
	}
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 542 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 16)
			print("register value out of range\n");
		(yyval.lval) = (((yyvsp[(1) - (1)].lval)&15) << 8) | (1 << 4);
	}
    break;

  case 91:

/* Line 1455 of yacc.c  */
#line 548 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 32)
			print("shift value out of range\n");
		(yyval.lval) = ((yyvsp[(1) - (1)].lval)&31) << 7;
	}
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 557 "a.y"
    {
		(yyval.lval) = REGPC;
	}
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 561 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 570 "a.y"
    {
		(yyval.lval) = REGSP;
	}
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 577 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 589 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FREG;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 595 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FREG;
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 603 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).name = (yyvsp[(3) - (4)].lval);
		(yyval.gen).sym = S;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 611 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).name = (yyvsp[(4) - (5)].lval);
		(yyval.gen).sym = (yyvsp[(1) - (5)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 105:

/* Line 1455 of yacc.c  */
#line 619 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).name = D_STATIC;
		(yyval.gen).sym = (yyvsp[(1) - (7)].sym);
		(yyval.gen).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 106:

/* Line 1455 of yacc.c  */
#line 628 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 107:

/* Line 1455 of yacc.c  */
#line 632 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 108:

/* Line 1455 of yacc.c  */
#line 636 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 648 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 652 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 656 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 660 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 664 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 669 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 673 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 680 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 684 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 688 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 692 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 696 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 700 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 704 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 708 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 712 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 716 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;



/* Line 1455 of yacc.c  */
#line 2747 "y.tab.c"
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



