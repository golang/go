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
     LTYPE0 = 258,
     LTYPE1 = 259,
     LTYPE2 = 260,
     LTYPE3 = 261,
     LTYPE4 = 262,
     LTYPEC = 263,
     LTYPED = 264,
     LTYPEN = 265,
     LTYPER = 266,
     LTYPET = 267,
     LTYPES = 268,
     LTYPEM = 269,
     LTYPEI = 270,
     LTYPEG = 271,
     LTYPEXC = 272,
     LTYPEX = 273,
     LTYPEPC = 274,
     LTYPEF = 275,
     LCONST = 276,
     LFP = 277,
     LPC = 278,
     LSB = 279,
     LBREG = 280,
     LLREG = 281,
     LSREG = 282,
     LFREG = 283,
     LXREG = 284,
     LFCONST = 285,
     LSCONST = 286,
     LSP = 287,
     LNAME = 288,
     LLAB = 289,
     LVAR = 290
   };
#endif
/* Tokens.  */
#define LTYPE0 258
#define LTYPE1 259
#define LTYPE2 260
#define LTYPE3 261
#define LTYPE4 262
#define LTYPEC 263
#define LTYPED 264
#define LTYPEN 265
#define LTYPER 266
#define LTYPET 267
#define LTYPES 268
#define LTYPEM 269
#define LTYPEI 270
#define LTYPEG 271
#define LTYPEXC 272
#define LTYPEX 273
#define LTYPEPC 274
#define LTYPEF 275
#define LCONST 276
#define LFP 277
#define LPC 278
#define LSB 279
#define LBREG 280
#define LLREG 281
#define LSREG 282
#define LFREG 283
#define LXREG 284
#define LFCONST 285
#define LSCONST 286
#define LSP 287
#define LNAME 288
#define LLAB 289
#define LVAR 290




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
#line 38 "a.y"
{
	Sym	*sym;
	int32	lval;
	struct {
		int32 v1;
		int32 v2;
	} con2;
	double	dval;
	char	sval[8];
	Addr	addr;
	Addr2	addr2;
}
/* Line 193 of yacc.c.  */
#line 187 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 200 "y.tab.c"

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
#define YYLAST   546

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  54
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  41
/* YYNRULES -- Number of rules.  */
#define YYNRULES  135
/* YYNRULES -- Number of states.  */
#define YYNSTATES  276

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   290

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    52,    12,     5,     2,
      50,    51,    10,     8,    49,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    46,    47,
       6,    48,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    53,     2,     2,     2,
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
      45
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     5,     9,    10,    15,    16,    21,
      23,    26,    29,    33,    37,    40,    43,    46,    49,    52,
      55,    58,    61,    64,    67,    70,    73,    76,    79,    82,
      85,    88,    91,    92,    94,    98,   102,   105,   107,   110,
     112,   115,   117,   121,   127,   131,   137,   140,   142,   145,
     147,   149,   153,   159,   163,   169,   172,   174,   178,   182,
     188,   194,   200,   204,   208,   210,   212,   214,   216,   219,
     222,   224,   226,   228,   230,   232,   237,   240,   243,   245,
     247,   249,   251,   253,   255,   258,   261,   264,   267,   272,
     278,   282,   285,   287,   290,   294,   299,   301,   303,   305,
     310,   315,   322,   332,   342,   346,   350,   355,   361,   370,
     372,   379,   385,   393,   394,   397,   400,   402,   404,   406,
     408,   410,   413,   416,   419,   423,   425,   429,   433,   437,
     441,   445,   450,   455,   459,   463
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      55,     0,    -1,    -1,    -1,    55,    56,    57,    -1,    -1,
      44,    46,    58,    57,    -1,    -1,    43,    46,    59,    57,
      -1,    47,    -1,    60,    47,    -1,     1,    47,    -1,    43,
      48,    94,    -1,    45,    48,    94,    -1,    13,    61,    -1,
      14,    65,    -1,    15,    64,    -1,    16,    62,    -1,    17,
      63,    -1,    21,    66,    -1,    19,    67,    -1,    22,    68,
      -1,    18,    69,    -1,    20,    70,    -1,    23,    71,    -1,
      24,    72,    -1,    25,    73,    -1,    26,    74,    -1,    27,
      75,    -1,    28,    76,    -1,    29,    77,    -1,    30,    78,
      -1,    -1,    49,    -1,    81,    49,    79,    -1,    79,    49,
      81,    -1,    81,    49,    -1,    81,    -1,    49,    79,    -1,
      79,    -1,    49,    82,    -1,    82,    -1,    84,    49,    82,
      -1,    90,    11,    93,    49,    84,    -1,    87,    49,    85,
      -1,    87,    49,    93,    49,    85,    -1,    49,    80,    -1,
      80,    -1,    10,    90,    -1,    61,    -1,    65,    -1,    81,
      49,    79,    -1,    81,    49,    79,    46,    36,    -1,    81,
      49,    79,    -1,    81,    49,    79,    46,    37,    -1,    81,
      49,    -1,    81,    -1,    81,    49,    79,    -1,    87,    49,
      84,    -1,    87,    49,    93,    49,    84,    -1,    83,    49,
      79,    49,    93,    -1,    84,    49,    79,    49,    83,    -1,
      81,    49,    81,    -1,    81,    49,    81,    -1,    83,    -1,
      87,    -1,    82,    -1,    89,    -1,    10,    83,    -1,    10,
      88,    -1,    83,    -1,    88,    -1,    84,    -1,    79,    -1,
      84,    -1,    93,    50,    33,    51,    -1,    43,    91,    -1,
      44,    91,    -1,    35,    -1,    38,    -1,    36,    -1,    39,
      -1,    42,    -1,    37,    -1,    52,    93,    -1,    52,    90,
      -1,    52,    41,    -1,    52,    40,    -1,    52,    50,    40,
      51,    -1,    52,    50,     9,    40,    51,    -1,    52,     9,
      40,    -1,    52,    86,    -1,    31,    -1,     9,    31,    -1,
      31,     9,    31,    -1,     9,    31,     9,    31,    -1,    88,
      -1,    89,    -1,    93,    -1,    93,    50,    36,    51,    -1,
      93,    50,    42,    51,    -1,    93,    50,    36,    10,    93,
      51,    -1,    93,    50,    36,    51,    50,    36,    10,    93,
      51,    -1,    93,    50,    36,    51,    50,    37,    10,    93,
      51,    -1,    50,    36,    51,    -1,    50,    42,    51,    -1,
      93,    50,    37,    51,    -1,    50,    36,    10,    93,    51,
      -1,    50,    36,    51,    50,    36,    10,    93,    51,    -1,
      90,    -1,    90,    50,    36,    10,    93,    51,    -1,    43,
      91,    50,    92,    51,    -1,    43,     6,     7,    91,    50,
      34,    51,    -1,    -1,     8,    93,    -1,     9,    93,    -1,
      34,    -1,    42,    -1,    32,    -1,    31,    -1,    45,    -1,
       9,    93,    -1,     8,    93,    -1,    53,    93,    -1,    50,
      94,    51,    -1,    93,    -1,    94,     8,    94,    -1,    94,
       9,    94,    -1,    94,    10,    94,    -1,    94,    11,    94,
      -1,    94,    12,    94,    -1,    94,     6,     6,    94,    -1,
      94,     7,     7,    94,    -1,    94,     5,    94,    -1,    94,
       4,    94,    -1,    94,     3,    94,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    69,    69,    71,    70,    78,    77,    85,    84,    90,
      91,    92,    95,   100,   106,   107,   108,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   126,   130,   137,   144,   151,   156,   163,   168,
     175,   180,   185,   192,   200,   205,   213,   218,   223,   232,
     233,   236,   241,   251,   256,   266,   271,   276,   283,   288,
     296,   304,   314,   323,   334,   335,   338,   339,   340,   344,
     348,   349,   350,   353,   354,   357,   363,   371,   379,   384,
     389,   394,   399,   404,   411,   417,   428,   434,   440,   446,
     452,   460,   469,   474,   479,   484,   491,   492,   495,   501,
     507,   513,   522,   531,   540,   545,   550,   556,   564,   574,
     578,   587,   594,   603,   606,   610,   616,   617,   621,   624,
     625,   629,   633,   637,   641,   647,   648,   652,   656,   660,
     664,   668,   672,   676,   680,   684
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'|'", "'^'", "'&'", "'<'", "'>'", "'+'",
  "'-'", "'*'", "'/'", "'%'", "LTYPE0", "LTYPE1", "LTYPE2", "LTYPE3",
  "LTYPE4", "LTYPEC", "LTYPED", "LTYPEN", "LTYPER", "LTYPET", "LTYPES",
  "LTYPEM", "LTYPEI", "LTYPEG", "LTYPEXC", "LTYPEX", "LTYPEPC", "LTYPEF",
  "LCONST", "LFP", "LPC", "LSB", "LBREG", "LLREG", "LSREG", "LFREG",
  "LXREG", "LFCONST", "LSCONST", "LSP", "LNAME", "LLAB", "LVAR", "':'",
  "';'", "'='", "','", "'('", "')'", "'$'", "'~'", "$accept", "prog", "@1",
  "line", "@2", "@3", "inst", "nonnon", "rimrem", "remrim", "rimnon",
  "nonrem", "nonrel", "spec1", "spec2", "spec3", "spec4", "spec5", "spec6",
  "spec7", "spec8", "spec9", "spec10", "spec11", "spec12", "rem", "rom",
  "rim", "rel", "reg", "imm", "imm2", "con2", "mem", "omem", "nmem", "nam",
  "offset", "pointer", "con", "expr", 0
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
     285,   286,   287,   288,   289,   290,    58,    59,    61,    44,
      40,    41,    36,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    54,    55,    56,    55,    58,    57,    59,    57,    57,
      57,    57,    60,    60,    60,    60,    60,    60,    60,    60,
      60,    60,    60,    60,    60,    60,    60,    60,    60,    60,
      60,    60,    61,    61,    62,    63,    64,    64,    65,    65,
      66,    66,    66,    67,    68,    68,    69,    69,    69,    70,
      70,    71,    71,    72,    72,    73,    73,    73,    74,    74,
      75,    76,    77,    78,    79,    79,    80,    80,    80,    80,
      80,    80,    80,    81,    81,    82,    82,    82,    83,    83,
      83,    83,    83,    83,    84,    84,    84,    84,    84,    84,
      84,    85,    86,    86,    86,    86,    87,    87,    88,    88,
      88,    88,    88,    88,    88,    88,    88,    88,    88,    89,
      89,    90,    90,    91,    91,    91,    92,    92,    92,    93,
      93,    93,    93,    93,    93,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     0,     4,     1,
       2,     2,     3,     3,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     0,     1,     3,     3,     2,     1,     2,     1,
       2,     1,     3,     5,     3,     5,     2,     1,     2,     1,
       1,     3,     5,     3,     5,     2,     1,     3,     3,     5,
       5,     5,     3,     3,     1,     1,     1,     1,     2,     2,
       1,     1,     1,     1,     1,     4,     2,     2,     1,     1,
       1,     1,     1,     1,     2,     2,     2,     2,     4,     5,
       3,     2,     1,     2,     3,     4,     1,     1,     1,     4,
       4,     6,     9,     9,     3,     3,     4,     5,     8,     1,
       6,     5,     7,     0,     2,     2,     1,     1,     1,     1,
       1,     2,     2,     2,     3,     1,     3,     3,     3,     3,
       3,     4,     4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    32,     0,     0,     0,     0,
       0,     0,    32,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     9,     4,     0,    11,
      33,    14,     0,     0,   119,    78,    80,    83,    79,    81,
      82,   113,   120,     0,     0,     0,    15,    39,    64,    65,
      96,    97,   109,    98,     0,    16,    73,    37,    74,    17,
       0,    18,     0,     0,   113,   113,     0,    22,    47,    66,
      70,    72,    71,    67,    98,    20,     0,    33,    49,    50,
      23,   113,     0,     0,    19,    41,     0,     0,    21,     0,
      24,     0,    25,     0,    26,    56,    27,     0,    28,     0,
      29,     0,    30,     0,    31,     0,     7,     0,     5,     0,
      10,   122,   121,     0,     0,     0,     0,    38,     0,     0,
     125,     0,   123,     0,     0,     0,    87,    86,     0,    85,
      84,    36,     0,     0,    68,    69,    48,    76,    77,     0,
      46,     0,     0,    76,    40,     0,     0,     0,     0,     0,
      55,     0,     0,     0,     0,     0,     0,    12,     0,    13,
     113,   114,   115,     0,     0,   104,   105,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   124,     0,     0,
       0,     0,    90,     0,     0,    34,    35,     0,     0,    42,
       0,    44,     0,    51,    53,    57,    58,     0,     0,     0,
      62,    63,     8,     6,     0,   118,   116,   117,     0,     0,
       0,   135,   134,   133,     0,     0,   126,   127,   128,   129,
     130,     0,     0,    99,   106,   100,     0,    88,    75,     0,
       0,    92,    91,     0,     0,     0,     0,     0,     0,     0,
     111,   107,     0,   131,   132,     0,     0,     0,    89,    43,
      93,     0,    45,    52,    54,    59,    60,    61,     0,     0,
     110,   101,     0,     0,     0,    94,   112,     0,     0,     0,
      95,   108,     0,     0,   102,   103
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    27,   158,   156,    28,    31,    59,    61,
      55,    46,    84,    75,    88,    67,    80,    90,    92,    94,
      96,    98,   100,   102,   104,    56,    68,    57,    69,    48,
      58,   191,   232,    49,    50,    51,    52,   116,   208,    53,
     121
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -104
static const yytype_int16 yypact[] =
{
    -104,     4,  -104,   173,   -26,   -25,   277,   297,   297,   349,
     225,   -14,   329,   396,    18,   297,   297,   297,    18,   171,
     -20,   297,   297,     2,    -4,    26,  -104,  -104,    43,  -104,
    -104,  -104,   478,   478,  -104,  -104,  -104,  -104,  -104,  -104,
    -104,   111,  -104,   349,   402,   478,  -104,  -104,  -104,  -104,
    -104,  -104,   -12,    -5,    83,  -104,  -104,    44,  -104,  -104,
      46,  -104,    49,   349,   111,   113,   245,  -104,  -104,  -104,
    -104,  -104,  -104,  -104,    50,  -104,   100,   349,  -104,  -104,
    -104,   113,   420,   478,  -104,  -104,    64,    66,  -104,    78,
    -104,    80,  -104,    85,  -104,    89,  -104,    93,  -104,    98,
    -104,   101,  -104,   112,  -104,   121,  -104,   478,  -104,   478,
    -104,  -104,  -104,   153,   478,   478,   135,  -104,     8,   163,
    -104,    74,  -104,   179,    52,   427,  -104,  -104,   445,  -104,
    -104,  -104,   349,   297,  -104,  -104,  -104,   135,  -104,   381,
    -104,    33,   478,  -104,  -104,   420,   186,   451,   349,   349,
     349,   460,   349,   349,   297,   297,   173,   172,   173,   172,
     113,  -104,  -104,     5,   478,   180,  -104,   478,   478,   478,
     226,   224,   478,   478,   478,   478,   478,  -104,   235,    36,
     195,   196,  -104,   466,   197,  -104,  -104,   199,   202,  -104,
      21,  -104,   203,   211,   219,  -104,  -104,   217,   222,   223,
    -104,  -104,  -104,  -104,   229,  -104,  -104,  -104,   240,   241,
     237,   232,   527,   534,   478,   478,   134,   134,  -104,  -104,
    -104,   478,   478,   243,  -104,  -104,   248,  -104,  -104,   -20,
     263,   287,  -104,   249,   264,   265,   -20,   478,   171,   269,
    -104,  -104,   294,   214,   214,   256,   258,   119,  -104,  -104,
     301,   280,  -104,  -104,  -104,  -104,  -104,  -104,   266,   478,
    -104,  -104,   308,   311,   292,  -104,  -104,   273,   478,   478,
    -104,  -104,   274,   278,  -104,  -104
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -104,  -104,  -104,  -103,  -104,  -104,  -104,   319,  -104,  -104,
    -104,   331,  -104,  -104,  -104,  -104,  -104,  -104,  -104,  -104,
    -104,  -104,  -104,  -104,  -104,    19,   275,    -2,    -6,    -9,
      -8,   115,  -104,    22,     1,    -1,    -3,   -48,  -104,   -10,
     -66
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      74,    70,    71,    87,     2,    86,    60,    85,    76,    73,
      99,    72,   101,    91,    93,    95,   137,   138,   164,   103,
     105,    29,   111,   112,    30,    47,    32,    33,    62,    41,
     230,    47,    54,   143,   120,   122,    89,   205,   123,   206,
      97,   157,   108,   159,   130,   124,   222,   207,   106,    34,
     107,   129,   231,   202,   134,   203,    74,    70,    71,   165,
     136,    41,   117,    42,   135,    73,   187,    72,    44,   179,
     180,    45,    87,   120,   109,   181,   144,   167,   168,   169,
     170,   171,   172,   173,   174,   175,   176,   223,   179,   180,
     110,    32,   125,   131,   181,   132,   117,   120,   133,   120,
     141,   211,   212,   213,   161,   162,   216,   217,   218,   219,
     220,   142,   204,   145,    34,   112,   146,   113,   120,   114,
     115,   114,   115,   126,   127,   177,    41,   147,    42,   148,
     134,   186,   188,   128,   149,    87,    45,   192,   150,   189,
     135,   197,   151,   196,   174,   175,   176,   152,   243,   244,
     153,   185,   200,   201,   209,   262,   263,   120,   120,   120,
     160,   154,   120,   120,   120,   120,   120,   193,   194,   195,
     155,   198,   199,   112,     4,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   163,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,   120,   120,    35,    36,    37,    38,
      39,   245,   246,    40,   166,   178,    23,    24,    25,   187,
      26,   249,   172,   173,   174,   175,   176,   256,   255,   257,
     210,   215,   214,    32,    33,    63,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   221,   224,   225,   227,   267,
     228,   229,   233,    32,    33,   139,    34,   234,   272,   273,
      35,    36,    37,    38,    39,   235,   236,    40,    64,    65,
      42,   237,   238,   242,    66,    44,    34,    54,    45,   239,
      35,    36,    37,    38,    39,    32,    33,    40,    64,    65,
      42,   240,   241,   247,   250,    44,   251,    54,    45,   248,
     253,   190,   254,   258,   259,    32,    33,   260,    34,   261,
     264,   265,    35,    36,    37,    38,    39,   266,   268,    40,
      41,   269,    42,   270,   271,   274,    43,    44,    34,   275,
      45,    78,    35,    36,    37,    38,    39,    32,    33,    40,
      41,   140,    42,    79,     0,     0,     0,    44,   252,    54,
      45,     0,     0,     0,     0,     0,     0,    32,    33,     0,
      34,     0,     0,     0,    35,    36,    37,    38,    39,     0,
       0,    40,    41,     0,    42,     0,     0,     0,    77,    44,
      34,     0,    45,     0,    35,    36,    37,    38,    39,    32,
      33,    40,    41,     0,    42,     0,     0,     0,     0,    44,
       0,     0,    45,     0,    32,    33,     0,     0,     0,     0,
      32,    33,    34,     0,     0,     0,    35,    36,    37,    38,
      39,     0,     0,    40,     0,     0,    42,    34,    32,    33,
       0,    44,     0,    34,    45,    32,    33,     0,   118,    81,
      65,    42,     0,     0,   119,    82,    83,    42,    54,    45,
       0,    34,    83,    32,   183,    45,     0,     0,    34,    32,
      33,     0,     0,    81,    65,    42,     0,   182,    32,    33,
      83,     0,    42,    45,    32,    33,    34,    83,     0,     0,
      45,     0,    34,     0,     0,   184,    32,    33,     0,     0,
      42,    34,     0,     0,     0,    83,    42,    34,    45,     0,
       0,    83,     0,   190,    45,    42,   226,     0,     0,    34,
      83,    42,    54,    45,     0,     0,    83,     0,     0,    45,
       0,     0,     0,    42,     0,     0,     0,     0,    83,     0,
       0,    45,   169,   170,   171,   172,   173,   174,   175,   176,
     170,   171,   172,   173,   174,   175,   176
};

static const yytype_int16 yycheck[] =
{
      10,    10,    10,    13,     0,    13,     8,    13,    11,    10,
      19,    10,    20,    15,    16,    17,    64,    65,    10,    21,
      22,    47,    32,    33,    49,     6,     8,     9,     9,    43,
       9,    12,    52,    81,    44,    45,    14,    32,    50,    34,
      18,   107,    46,   109,    54,    50,    10,    42,    46,    31,
      48,    54,    31,   156,    63,   158,    66,    66,    66,    51,
      63,    43,    43,    45,    63,    66,    33,    66,    50,    36,
      37,    53,    82,    83,    48,    42,    82,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    51,    36,    37,
      47,     8,     9,    49,    42,    49,    77,   107,    49,   109,
      50,   167,   168,   169,   114,   115,   172,   173,   174,   175,
     176,    11,   160,    49,    31,   125,    50,     6,   128,     8,
       9,     8,     9,    40,    41,    51,    43,    49,    45,    49,
     139,   133,   142,    50,    49,   145,    53,   147,    49,   145,
     139,   151,    49,   151,    10,    11,    12,    49,   214,   215,
      49,   132,   154,   155,   164,    36,    37,   167,   168,   169,
       7,    49,   172,   173,   174,   175,   176,   148,   149,   150,
      49,   152,   153,   183,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    50,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,   214,   215,    35,    36,    37,    38,
      39,   221,   222,    42,    51,    36,    43,    44,    45,    33,
      47,   229,     8,     9,    10,    11,    12,   237,   236,   238,
      50,     7,     6,     8,     9,    10,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    10,    51,    51,    51,   259,
      51,    49,    49,     8,     9,    10,    31,    46,   268,   269,
      35,    36,    37,    38,    39,    46,    49,    42,    43,    44,
      45,    49,    49,    36,    49,    50,    31,    52,    53,    50,
      35,    36,    37,    38,    39,     8,     9,    42,    43,    44,
      45,    51,    51,    50,    31,    50,     9,    52,    53,    51,
      36,    52,    37,    34,    10,     8,     9,    51,    31,    51,
       9,    31,    35,    36,    37,    38,    39,    51,    10,    42,
      43,    10,    45,    31,    51,    51,    49,    50,    31,    51,
      53,    12,    35,    36,    37,    38,    39,     8,     9,    42,
      43,    66,    45,    12,    -1,    -1,    -1,    50,   233,    52,
      53,    -1,    -1,    -1,    -1,    -1,    -1,     8,     9,    -1,
      31,    -1,    -1,    -1,    35,    36,    37,    38,    39,    -1,
      -1,    42,    43,    -1,    45,    -1,    -1,    -1,    49,    50,
      31,    -1,    53,    -1,    35,    36,    37,    38,    39,     8,
       9,    42,    43,    -1,    45,    -1,    -1,    -1,    -1,    50,
      -1,    -1,    53,    -1,     8,     9,    -1,    -1,    -1,    -1,
       8,     9,    31,    -1,    -1,    -1,    35,    36,    37,    38,
      39,    -1,    -1,    42,    -1,    -1,    45,    31,     8,     9,
      -1,    50,    -1,    31,    53,     8,     9,    -1,    36,    43,
      44,    45,    -1,    -1,    42,    49,    50,    45,    52,    53,
      -1,    31,    50,     8,     9,    53,    -1,    -1,    31,     8,
       9,    -1,    -1,    43,    44,    45,    -1,    40,     8,     9,
      50,    -1,    45,    53,     8,     9,    31,    50,    -1,    -1,
      53,    -1,    31,    -1,    -1,    40,     8,     9,    -1,    -1,
      45,    31,    -1,    -1,    -1,    50,    45,    31,    53,    -1,
      -1,    50,    -1,    52,    53,    45,    40,    -1,    -1,    31,
      50,    45,    52,    53,    -1,    -1,    50,    -1,    -1,    53,
      -1,    -1,    -1,    45,    -1,    -1,    -1,    -1,    50,    -1,
      -1,    53,     5,     6,     7,     8,     9,    10,    11,    12,
       6,     7,     8,     9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    55,     0,    56,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    43,    44,    45,    47,    57,    60,    47,
      49,    61,     8,     9,    31,    35,    36,    37,    38,    39,
      42,    43,    45,    49,    50,    53,    65,    79,    83,    87,
      88,    89,    90,    93,    52,    64,    79,    81,    84,    62,
      81,    63,    79,    10,    43,    44,    49,    69,    80,    82,
      83,    84,    88,    89,    93,    67,    90,    49,    61,    65,
      70,    43,    49,    50,    66,    82,    84,    93,    68,    87,
      71,    81,    72,    81,    73,    81,    74,    87,    75,    83,
      76,    84,    77,    81,    78,    81,    46,    48,    46,    48,
      47,    93,    93,     6,     8,     9,    91,    79,    36,    42,
      93,    94,    93,    50,    50,     9,    40,    41,    50,    90,
      93,    49,    49,    49,    83,    88,    90,    91,    91,    10,
      80,    50,    11,    91,    82,    49,    50,    49,    49,    49,
      49,    49,    49,    49,    49,    49,    59,    94,    58,    94,
       7,    93,    93,    50,    10,    51,    51,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    51,    36,    36,
      37,    42,    40,     9,    40,    79,    81,    33,    93,    82,
      52,    85,    93,    79,    79,    79,    84,    93,    79,    79,
      81,    81,    57,    57,    91,    32,    34,    42,    92,    93,
      50,    94,    94,    94,     6,     7,    94,    94,    94,    94,
      94,    10,    10,    51,    51,    51,    40,    51,    51,    49,
       9,    31,    86,    49,    46,    46,    49,    49,    49,    50,
      51,    51,    36,    94,    94,    93,    93,    50,    51,    84,
      31,     9,    85,    36,    37,    84,    93,    83,    34,    10,
      51,    51,    36,    37,     9,    31,    51,    93,    10,    10,
      31,    51,    93,    93,    51,    51
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
#line 71 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:
#line 78 "a.y"
    {
		if((yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->name);
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 7:
#line 85 "a.y"
    {
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 12:
#line 96 "a.y"
    {
		(yyvsp[(1) - (3)].sym)->type = LVAR;
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 13:
#line 101 "a.y"
    {
		if((yyvsp[(1) - (3)].sym)->value != (yyvsp[(3) - (3)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (3)].sym)->name);
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 14:
#line 106 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 15:
#line 107 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 16:
#line 108 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 17:
#line 109 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 18:
#line 110 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 19:
#line 111 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 20:
#line 112 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 21:
#line 113 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 22:
#line 114 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 23:
#line 115 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 24:
#line 116 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 25:
#line 117 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 26:
#line 118 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 27:
#line 119 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 28:
#line 120 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 29:
#line 121 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 30:
#line 122 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 31:
#line 123 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 32:
#line 126 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 33:
#line 131 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 34:
#line 138 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 35:
#line 145 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 36:
#line 152 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (2)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 37:
#line 157 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 38:
#line 164 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 39:
#line 169 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 40:
#line 176 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 41:
#line 181 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 42:
#line 186 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 43:
#line 193 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
	}
    break;

  case 44:
#line 201 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 45:
#line 206 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
	}
    break;

  case 46:
#line 214 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 47:
#line 219 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 48:
#line 224 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
		(yyval.addr2).to.index = (yyvsp[(2) - (2)].addr).type;
		(yyval.addr2).to.type = D_INDIR+D_ADDR;
	}
    break;

  case 51:
#line 237 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 52:
#line 242 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).from.index != D_NONE)
			yyerror("dp shift with lhs index");
		(yyval.addr2).from.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 53:
#line 252 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 54:
#line 257 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).to.index != D_NONE)
			yyerror("dp move with lhs index");
		(yyval.addr2).to.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 55:
#line 267 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (2)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 56:
#line 272 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 57:
#line 277 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 58:
#line 284 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 59:
#line 289 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
	}
    break;

  case 60:
#line 297 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to.offset = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 61:
#line 305 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
		if((yyvsp[(1) - (5)].addr).type != D_CONST)
			yyerror("illegal constant");
		(yyval.addr2).to.offset = (yyvsp[(1) - (5)].addr).offset;
	}
    break;

  case 62:
#line 315 "a.y"
    {
		if((yyvsp[(1) - (3)].addr).type != D_CONST || (yyvsp[(3) - (3)].addr).type != D_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 63:
#line 324 "a.y"
    {
		if((yyvsp[(1) - (3)].addr).type != D_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(3) - (3)].addr).type != D_EXTERN && (yyvsp[(3) - (3)].addr).type != D_STATIC)
			yyerror("value for FUNCDATA must be symbol reference");
 		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
 		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
 	}
    break;

  case 68:
#line 341 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 69:
#line 345 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 75:
#line 358 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 76:
#line 364 "a.y"
    {
		(yyval.addr) = nullgen;
		if(pass == 2)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->name);
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 77:
#line 372 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 78:
#line 380 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 79:
#line 385 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 80:
#line 390 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 81:
#line 395 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 82:
#line 400 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SP;
	}
    break;

  case 83:
#line 405 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 84:
#line 412 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 85:
#line 418 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).index = (yyvsp[(2) - (2)].addr).type;
		(yyval.addr).type = D_ADDR;
		/*
		if($2.type == D_AUTO || $2.type == D_PARAM)
			yyerror("constant cannot be automatic: %s",
				$2.sym->name);
		 */
	}
    break;

  case 86:
#line 429 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 87:
#line 435 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 88:
#line 441 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(3) - (4)].dval);
	}
    break;

  case 89:
#line 447 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(4) - (5)].dval);
	}
    break;

  case 90:
#line 453 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 91:
#line 461 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST2;
		(yyval.addr).offset = (yyvsp[(2) - (2)].con2).v1;
		(yyval.addr).offset2 = (yyvsp[(2) - (2)].con2).v2;
	}
    break;

  case 92:
#line 470 "a.y"
    {
		(yyval.con2).v1 = (yyvsp[(1) - (1)].lval);
		(yyval.con2).v2 = ArgsSizeUnknown;
	}
    break;

  case 93:
#line 475 "a.y"
    {
		(yyval.con2).v1 = -(yyvsp[(2) - (2)].lval);
		(yyval.con2).v2 = ArgsSizeUnknown;
	}
    break;

  case 94:
#line 480 "a.y"
    {
		(yyval.con2).v1 = (yyvsp[(1) - (3)].lval);
		(yyval.con2).v2 = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 95:
#line 485 "a.y"
    {
		(yyval.con2).v1 = -(yyvsp[(2) - (4)].lval);
		(yyval.con2).v2 = (yyvsp[(4) - (4)].lval);
	}
    break;

  case 98:
#line 496 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_NONE;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 99:
#line 502 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 100:
#line 508 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_SP;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 101:
#line 514 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_NONE;
		(yyval.addr).offset = (yyvsp[(1) - (6)].lval);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 102:
#line 523 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (9)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (9)].lval);
		(yyval.addr).index = (yyvsp[(6) - (9)].lval);
		(yyval.addr).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 103:
#line 532 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (9)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (9)].lval);
		(yyval.addr).index = (yyvsp[(6) - (9)].lval);
		(yyval.addr).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 104:
#line 541 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(2) - (3)].lval);
	}
    break;

  case 105:
#line 546 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_SP;
	}
    break;

  case 106:
#line 551 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 107:
#line 557 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_NONE;
		(yyval.addr).index = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 108:
#line 565 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(2) - (8)].lval);
		(yyval.addr).index = (yyvsp[(5) - (8)].lval);
		(yyval.addr).scale = (yyvsp[(7) - (8)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 109:
#line 575 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 110:
#line 579 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (6)].addr);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 111:
#line 588 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 112:
#line 595 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 1);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 113:
#line 603 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 114:
#line 607 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 115:
#line 611 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 117:
#line 618 "a.y"
    {
		(yyval.lval) = D_AUTO;
	}
    break;

  case 120:
#line 626 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 121:
#line 630 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 122:
#line 634 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 123:
#line 638 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 124:
#line 642 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 126:
#line 649 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 127:
#line 653 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:
#line 657 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:
#line 661 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 130:
#line 665 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 131:
#line 669 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 132:
#line 673 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 133:
#line 677 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 134:
#line 681 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 135:
#line 685 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2566 "y.tab.c"
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



