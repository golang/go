
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
#define YYLAST   609

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
      37,    90,    -1,    26,    78,    -1,    -1,    77,    52,    -1,
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
     269,   291,   298,   307,   315,   321,   324,   328,   333,   334,
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
       0,     0,     0,     0,     0,     0,    96,    93,     0,    92,
      43,    95,    84,     7,     0,     5,     0,    12,    46,    47,
       0,     0,    81,    80,    82,    84,     0,     0,   101,    67,
      68,     0,     0,    64,    57,     0,    75,    79,    70,    63,
      65,    71,    77,    69,     0,    49,   106,   106,    22,     0,
       0,     0,     0,     0,     0,     0,     0,    30,   115,   114,
       0,     0,     0,     0,   120,     0,   116,     0,     0,     0,
      48,    35,     0,     0,     0,   100,     0,    99,     0,     0,
       0,     0,    21,     0,     0,     0,     0,     0,    83,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    58,
      56,    54,    53,     0,     0,     0,     0,   106,    19,    20,
      72,    73,     0,    51,    52,     0,    23,     0,     0,    48,
       0,     0,     0,     0,   106,   107,   108,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   117,    31,
       0,   110,   109,   111,     0,     0,    34,     0,     0,     0,
       0,     0,     0,     0,     0,     8,     9,     6,    10,    16,
      84,     0,     0,     0,     0,    17,     0,    74,    59,     0,
      18,     0,     0,     0,    51,     0,     0,    48,     0,     0,
       0,     0,     0,    48,     0,     0,   130,   129,   128,     0,
       0,   121,   122,   123,   124,   125,     0,   103,     0,    36,
       0,   101,    37,    48,     0,     0,    94,    15,    86,    90,
      91,    87,    88,    89,   102,    55,     0,    66,    78,    76,
      50,    24,     0,    61,    62,     0,    29,    48,    28,     0,
     104,   126,   127,    32,    33,     0,     0,    39,     0,     0,
      14,    26,    25,    27,     0,     0,    38,     0,    41,     0,
     105,     0,     0,     0,     0,    97,     0,     0,    42,     0,
       0,     0,     0,   118,    85,    98,     0,    40,   119
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    32,   146,   144,    33,    35,   105,   108,
      93,    94,   178,    95,   169,    96,    97,    98,    81,    82,
      83,   308,    84,   268,    71,    72,   316,   136,   101,   102,
     123,   204,   124,   327,   125
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -119
static const yytype_int16 yypact[] =
{
    -119,    29,  -119,   283,    -9,  -119,  -119,  -119,  -119,   -23,
    -119,  -119,  -119,  -119,  -119,   407,   407,  -119,   -23,   -23,
    -119,  -119,  -119,  -119,  -119,  -119,   -23,   383,   -24,    16,
      18,  -119,  -119,    25,  -119,   196,   196,   320,    52,   -23,
     374,    52,   196,   342,   294,    52,   447,   447,  -119,   125,
    -119,   447,   447,    45,    47,   109,   174,  -119,    62,    98,
     382,   130,    98,   174,   174,    69,  -119,  -119,    80,  -119,
    -119,  -119,  -119,  -119,   447,  -119,   447,  -119,  -119,  -119,
     447,    96,  -119,  -119,  -119,    24,   105,    95,  -119,  -119,
    -119,   360,   101,  -119,  -119,   108,  -119,  -119,  -119,  -119,
     119,  -119,   127,   131,   401,  -119,    76,    76,  -119,   141,
     224,   111,   383,   154,   149,    69,   162,  -119,  -119,  -119,
     221,   447,   447,   163,  -119,    40,  -119,   409,   170,   447,
     -23,  -119,   167,   171,     2,  -119,   176,  -119,   181,   183,
     184,   383,  -119,   447,   283,   530,   283,   540,  -119,   383,
     244,   245,   251,   260,   383,   447,   206,   441,   208,  -119,
    -119,  -119,   131,   224,   383,   110,   282,   125,  -119,  -119,
    -119,  -119,   213,  -119,  -119,   238,  -119,   383,   220,     9,
     222,   110,   225,    69,    76,  -119,  -119,   170,   447,   447,
     447,   285,   287,   447,   447,   447,   447,   447,  -119,  -119,
     247,  -119,  -119,  -119,   263,   248,  -119,    71,   447,   276,
      79,    71,   383,   383,    56,  -119,  -119,  -119,  -119,  -119,
     271,   360,   360,   360,   360,  -119,   268,  -119,  -119,   415,
    -119,   270,   277,   281,   163,    60,   293,   -23,   280,   383,
     383,   383,   383,   286,   297,   311,   589,   579,   597,   447,
     447,   211,   211,  -119,  -119,  -119,   284,  -119,    62,  -119,
     520,   318,  -119,   -23,   321,   322,  -119,   383,  -119,  -119,
    -119,  -119,  -119,  -119,  -119,  -119,   131,  -119,  -119,  -119,
    -119,  -119,   453,  -119,  -119,   327,  -119,    77,  -119,   347,
    -119,   478,   478,  -119,  -119,   383,    71,  -119,   330,   383,
    -119,  -119,  -119,  -119,   326,   334,  -119,   383,  -119,   337,
    -119,   115,   340,   383,   341,  -119,   350,   383,  -119,   447,
     115,   354,    86,   367,  -119,  -119,   447,  -119,   570
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -119,  -119,  -119,   -73,  -119,  -119,  -119,   547,    -6,   332,
     -50,   376,   -60,   -82,  -119,   -46,   -39,   -86,   -15,  -118,
     -22,  -119,   -27,   152,  -111,   -35,   118,  -119,   -45,     8,
     -81,   255,   132,  -119,     6
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -61
static const yytype_int16 yytable[] =
{
      85,    85,    85,    40,   113,    70,   161,    85,   131,   199,
     100,   209,    57,    58,   132,    99,   137,   138,   239,   142,
      65,    86,   116,    53,    55,   173,   174,   111,   176,     2,
     150,   151,   104,   152,   130,   110,    73,    74,   115,   117,
      39,   139,   140,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,    34,   232,   233,   156,   159,   170,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   182,
     233,   215,    39,   217,   -60,    85,    75,   179,   153,    76,
     145,   230,   147,   100,   121,   122,   234,    77,    99,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   201,
     202,   203,   236,   244,    78,    79,   156,   198,   127,    46,
     157,   158,   171,   128,   220,    39,    66,    87,    88,    67,
     129,    68,    69,   266,   206,    87,   261,   219,    85,   231,
      92,   120,   225,   121,   122,   141,   100,   243,   293,    48,
      39,    99,   237,   275,    87,    88,   143,    54,    54,   214,
      78,    79,    67,   325,    68,    69,   159,   160,    49,   149,
      50,   155,   259,   314,   315,   262,   263,    91,   154,   103,
      52,   163,   109,   240,   177,   114,    87,    88,   118,   119,
     284,   285,    78,    79,   126,   164,   269,   269,   269,   269,
     264,   265,   133,   165,   246,   247,   248,   166,   134,   251,
     252,   253,   254,   255,   283,   179,   179,   175,   294,   201,
     202,   203,   148,    66,   260,   181,    67,   180,    68,    69,
     286,   195,   196,   197,   162,   183,    78,    79,   184,   187,
     207,   281,    46,    47,   208,    66,   172,   288,    67,   210,
      68,    69,   103,   301,   211,   300,   212,   213,    78,    79,
     221,   306,   222,   185,   186,   291,   292,   297,   223,   200,
     305,   205,    48,    66,    80,   303,    67,   224,    68,    69,
      87,    88,   312,   227,    89,    90,   229,   309,   318,   235,
     236,    49,   321,    50,     4,   238,   241,   226,   242,   119,
      91,   249,    92,    52,   250,   103,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
     256,   258,    19,    20,    21,    22,    23,    24,    25,    26,
      27,   201,   202,   203,    67,   322,    68,    69,    46,    47,
     257,   228,   328,    66,   267,   274,    67,   277,    68,    69,
      28,    29,    30,   282,   278,    31,    78,    79,   279,   287,
      46,    47,    80,   270,   270,   270,   270,    39,    48,    66,
     280,   276,    67,   289,    68,    69,    87,    88,    46,    47,
      89,    90,    78,    79,   271,   272,   273,    49,   290,    50,
      48,   296,    46,    47,   298,   299,    91,   304,    92,    52,
      46,    47,   302,   310,    78,    79,   307,   311,    48,    66,
     313,    50,    67,   317,    68,    69,   112,   319,    91,    46,
      47,    52,    48,   320,   114,    46,    47,    46,    47,    50,
      48,   324,    66,    46,    47,    67,    51,    68,    69,    52,
     326,   106,   107,    50,    78,    79,   168,   135,   323,    48,
      51,    50,   245,    52,     0,    48,     0,    48,    51,    46,
      47,    52,     0,    48,     0,    46,    47,     0,   167,   107,
      50,    46,    47,     0,    49,     0,    50,    91,    50,     0,
      52,     0,    49,    51,    50,    51,    52,    80,    52,    48,
       0,    91,     0,     0,    52,    48,   193,   194,   195,   196,
     197,    48,     0,     0,     0,     0,   228,     0,     0,     0,
      50,     0,     0,     0,     0,     0,    50,    51,     0,     0,
      52,     0,    50,    51,     0,     0,    52,     0,     0,    91,
       0,     0,    52,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,    36,    37,    38,     0,    41,    42,    43,
      44,    45,     0,     0,    56,     0,     0,    59,    60,    61,
      62,    63,    64,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   295,   190,   191,   192,   193,   194,   195,
     196,   197,   216,   189,   190,   191,   192,   193,   194,   195,
     196,   197,   218,   191,   192,   193,   194,   195,   196,   197
};

static const yytype_int16 yycheck[] =
{
      35,    36,    37,     9,    43,    27,    92,    42,    58,   127,
      37,     9,    18,    19,    59,    37,    61,    62,     9,    65,
      26,    36,    44,    15,    16,   106,   107,    42,   110,     0,
       6,     7,    38,     9,    56,    41,    60,    61,    44,    45,
      63,    63,    64,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    62,   165,   166,    91,    55,   104,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,   115,
     181,   144,    63,   146,    65,   110,    60,   112,    54,    61,
      74,   163,    76,   110,     8,     9,   167,    62,   110,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    39,
      40,    41,    42,   184,    52,    53,   141,    67,    63,     8,
       9,    10,   104,    66,   149,    63,    39,    46,    47,    42,
      11,    44,    45,    67,   130,    46,    47,   149,   163,   164,
      68,     6,   154,     8,     9,    66,   163,   183,   256,    38,
      63,   163,   177,   229,    46,    47,    66,    15,    16,   143,
      52,    53,    42,    67,    44,    45,    55,    56,    57,    63,
      59,    66,   207,    48,    49,   210,   211,    66,    63,    37,
      69,    63,    40,   179,    63,    43,    46,    47,    46,    47,
     240,   241,    52,    53,    52,    66,   221,   222,   223,   224,
     212,   213,    60,    66,   188,   189,   190,    66,    68,   193,
     194,   195,   196,   197,   239,   240,   241,    66,   258,    39,
      40,    41,    80,    39,   208,    66,    42,    63,    44,    45,
     242,    10,    11,    12,    92,    63,    52,    53,     7,    66,
      63,   237,     8,     9,    63,    39,   104,   243,    42,    63,
      44,    45,   110,   282,    63,   267,    63,    63,    52,    53,
       6,   296,     7,   121,   122,   249,   250,   263,     7,   127,
     295,   129,    38,    39,    68,   287,    42,     7,    44,    45,
      46,    47,   307,    67,    50,    51,    68,   299,   313,    66,
      42,    57,   317,    59,     1,    65,    64,   155,    63,   157,
      66,     6,    68,    69,     7,   163,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      63,    63,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    39,    40,    41,    42,   319,    44,    45,     8,     9,
      67,    55,   326,    39,    63,    67,    42,    67,    44,    45,
      57,    58,    59,    63,    67,    62,    52,    53,    67,    63,
       8,     9,    68,   221,   222,   223,   224,    63,    38,    39,
      67,   229,    42,    66,    44,    45,    46,    47,     8,     9,
      50,    51,    52,    53,   222,   223,   224,    57,    67,    59,
      38,    63,     8,     9,    63,    63,    66,    40,    68,    69,
       8,     9,    65,    67,    52,    53,    66,    63,    38,    39,
      63,    59,    42,    63,    44,    45,    64,    66,    66,     8,
       9,    69,    38,    63,   282,     8,     9,     8,     9,    59,
      38,    67,    39,     8,     9,    42,    66,    44,    45,    69,
      63,    57,    58,    59,    52,    53,   104,    61,   320,    38,
      66,    59,   187,    69,    -1,    38,    -1,    38,    66,     8,
       9,    69,    -1,    38,    -1,     8,     9,    -1,    57,    58,
      59,     8,     9,    -1,    57,    -1,    59,    66,    59,    -1,
      69,    -1,    57,    66,    59,    66,    69,    68,    69,    38,
      -1,    66,    -1,    -1,    69,    38,     8,     9,    10,    11,
      12,    38,    -1,    -1,    -1,    -1,    55,    -1,    -1,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    59,    66,    -1,    -1,
      69,    -1,    59,    66,    -1,    -1,    69,    -1,    -1,    66,
      -1,    -1,    69,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,     6,     7,     8,    -1,    10,    11,    12,
      13,    14,    -1,    -1,    17,    -1,    -1,    20,    21,    22,
      23,    24,    25,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    63,     5,     6,     7,     8,     9,    10,
      11,    12,    62,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    62,     6,     7,     8,     9,    10,    11,    12
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
      77,    77,    77,    77,    77,    78,    39,    42,    44,    45,
      90,    94,    95,    60,    61,    60,    61,    62,    52,    53,
      68,    88,    89,    90,    92,    95,    88,    46,    47,    50,
      51,    66,    68,    80,    81,    83,    85,    86,    87,    90,
      92,    98,    99,   102,    78,    78,    57,    58,    79,   102,
      78,    88,    64,    86,   102,    78,    90,    78,   102,   102,
       6,     8,     9,   100,   102,   104,   102,    63,    66,    11,
      90,    80,    98,   102,    68,    81,    97,    98,    98,    90,
      90,    66,    85,    66,    75,   104,    74,   104,   102,    63,
       6,     7,     9,    54,    63,    66,    95,     9,    10,    55,
      56,    87,   102,    63,    66,    66,    66,    57,    79,    84,
      85,    99,   102,   100,   100,    66,    83,    63,    82,    95,
      63,    66,    85,    63,     7,   102,   102,    66,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    67,    89,
     102,    39,    40,    41,   101,   102,    78,    63,    63,     9,
      63,    63,    63,    63,   104,    73,    62,    73,    62,    90,
      95,     6,     7,     7,     7,    90,   102,    67,    55,    68,
      83,    95,    94,    94,   100,    66,    42,    95,    65,     9,
      78,    64,    63,    85,   100,   101,   104,   104,   104,     6,
       7,   104,   104,   104,   104,   104,    63,    67,    63,    98,
     104,    47,    98,    98,    90,    90,    67,    63,    93,    95,
     102,    93,    93,    93,    67,    87,   102,    67,    67,    67,
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
#line 299 "a.y"
    {
		(yyvsp[(7) - (9)].gen).type = D_REGREG;
		(yyvsp[(7) - (9)].gen).offset = (yyvsp[(9) - (9)].lval);
		outcode((yyvsp[(1) - (9)].lval), (yyvsp[(2) - (9)].lval), &(yyvsp[(3) - (9)].gen), (yyvsp[(5) - (9)].gen).reg, &(yyvsp[(7) - (9)].gen));
	}
    break;

  case 43:

/* Line 1455 of yacc.c  */
#line 308 "a.y"
    {
		// TODO
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
#line 2748 "y.tab.c"
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



