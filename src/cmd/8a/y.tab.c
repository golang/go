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
#define YYLAST   561

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  54
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  40
/* YYNRULES -- Number of rules.  */
#define YYNRULES  132
/* YYNRULES -- Number of states.  */
#define YYNSTATES  270

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
       0,     0,     3,     4,     5,     9,    10,    15,    17,    20,
      23,    27,    31,    34,    37,    40,    43,    46,    49,    52,
      55,    58,    61,    64,    67,    70,    73,    76,    79,    82,
      85,    86,    88,    92,    96,    99,   101,   104,   106,   109,
     111,   115,   121,   125,   131,   134,   136,   139,   141,   143,
     147,   153,   157,   163,   166,   168,   172,   176,   182,   188,
     194,   198,   202,   204,   206,   208,   210,   213,   216,   218,
     220,   222,   224,   226,   231,   234,   236,   238,   240,   242,
     244,   246,   249,   252,   255,   258,   263,   269,   273,   276,
     278,   281,   285,   290,   292,   294,   296,   301,   306,   313,
     323,   333,   337,   341,   346,   352,   361,   363,   370,   376,
     384,   385,   388,   391,   393,   395,   397,   399,   401,   404,
     407,   410,   414,   416,   420,   424,   428,   432,   436,   441,
     446,   450,   454
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      55,     0,    -1,    -1,    -1,    55,    56,    57,    -1,    -1,
      43,    46,    58,    57,    -1,    47,    -1,    59,    47,    -1,
       1,    47,    -1,    43,    48,    93,    -1,    45,    48,    93,
      -1,    13,    60,    -1,    14,    64,    -1,    15,    63,    -1,
      16,    61,    -1,    17,    62,    -1,    21,    65,    -1,    19,
      66,    -1,    22,    67,    -1,    18,    68,    -1,    20,    69,
      -1,    23,    70,    -1,    24,    71,    -1,    25,    72,    -1,
      26,    73,    -1,    27,    74,    -1,    28,    75,    -1,    29,
      76,    -1,    30,    77,    -1,    -1,    49,    -1,    80,    49,
      78,    -1,    78,    49,    80,    -1,    80,    49,    -1,    80,
      -1,    49,    78,    -1,    78,    -1,    49,    81,    -1,    81,
      -1,    83,    49,    81,    -1,    89,    11,    92,    49,    83,
      -1,    86,    49,    84,    -1,    86,    49,    92,    49,    84,
      -1,    49,    79,    -1,    79,    -1,    10,    89,    -1,    60,
      -1,    64,    -1,    80,    49,    78,    -1,    80,    49,    78,
      46,    36,    -1,    80,    49,    78,    -1,    80,    49,    78,
      46,    37,    -1,    80,    49,    -1,    80,    -1,    80,    49,
      78,    -1,    86,    49,    83,    -1,    86,    49,    92,    49,
      83,    -1,    82,    49,    78,    49,    92,    -1,    83,    49,
      78,    49,    82,    -1,    80,    49,    80,    -1,    80,    49,
      80,    -1,    82,    -1,    86,    -1,    81,    -1,    88,    -1,
      10,    82,    -1,    10,    87,    -1,    82,    -1,    87,    -1,
      83,    -1,    78,    -1,    83,    -1,    92,    50,    33,    51,
      -1,    43,    90,    -1,    35,    -1,    38,    -1,    36,    -1,
      39,    -1,    42,    -1,    37,    -1,    52,    92,    -1,    52,
      89,    -1,    52,    41,    -1,    52,    40,    -1,    52,    50,
      40,    51,    -1,    52,    50,     9,    40,    51,    -1,    52,
       9,    40,    -1,    52,    85,    -1,    31,    -1,     9,    31,
      -1,    31,     9,    31,    -1,     9,    31,     9,    31,    -1,
      87,    -1,    88,    -1,    92,    -1,    92,    50,    36,    51,
      -1,    92,    50,    42,    51,    -1,    92,    50,    36,    10,
      92,    51,    -1,    92,    50,    36,    51,    50,    36,    10,
      92,    51,    -1,    92,    50,    36,    51,    50,    37,    10,
      92,    51,    -1,    50,    36,    51,    -1,    50,    42,    51,
      -1,    92,    50,    37,    51,    -1,    50,    36,    10,    92,
      51,    -1,    50,    36,    51,    50,    36,    10,    92,    51,
      -1,    89,    -1,    89,    50,    36,    10,    92,    51,    -1,
      43,    90,    50,    91,    51,    -1,    43,     6,     7,    90,
      50,    34,    51,    -1,    -1,     8,    92,    -1,     9,    92,
      -1,    34,    -1,    42,    -1,    32,    -1,    31,    -1,    45,
      -1,     9,    92,    -1,     8,    92,    -1,    53,    92,    -1,
      50,    93,    51,    -1,    92,    -1,    93,     8,    93,    -1,
      93,     9,    93,    -1,    93,    10,    93,    -1,    93,    11,
      93,    -1,    93,    12,    93,    -1,    93,     6,     6,    93,
      -1,    93,     7,     7,    93,    -1,    93,     5,    93,    -1,
      93,     4,    93,    -1,    93,     3,    93,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    69,    69,    71,    70,    78,    77,    86,    87,    88,
      91,    96,   102,   103,   104,   105,   106,   107,   108,   109,
     110,   111,   112,   113,   114,   115,   116,   117,   118,   119,
     122,   126,   133,   140,   147,   152,   159,   164,   171,   176,
     181,   188,   196,   202,   211,   216,   221,   230,   231,   234,
     239,   249,   254,   264,   269,   274,   281,   286,   294,   302,
     312,   321,   332,   333,   336,   337,   338,   342,   346,   347,
     348,   351,   352,   355,   361,   372,   377,   382,   387,   392,
     397,   404,   410,   421,   427,   433,   439,   445,   453,   462,
     467,   472,   477,   484,   485,   488,   494,   500,   506,   515,
     524,   533,   538,   543,   549,   557,   567,   571,   580,   587,
     596,   599,   603,   609,   610,   614,   617,   618,   622,   626,
     630,   634,   640,   641,   645,   649,   653,   657,   661,   665,
     669,   673,   677
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
  "line", "@2", "inst", "nonnon", "rimrem", "remrim", "rimnon", "nonrem",
  "nonrel", "spec1", "spec2", "spec3", "spec4", "spec5", "spec6", "spec7",
  "spec8", "spec9", "spec10", "spec11", "spec12", "rem", "rom", "rim",
  "rel", "reg", "imm", "imm2", "con2", "mem", "omem", "nmem", "nam",
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
       0,    54,    55,    56,    55,    58,    57,    57,    57,    57,
      59,    59,    59,    59,    59,    59,    59,    59,    59,    59,
      59,    59,    59,    59,    59,    59,    59,    59,    59,    59,
      60,    60,    61,    62,    63,    63,    64,    64,    65,    65,
      65,    66,    67,    67,    68,    68,    68,    69,    69,    70,
      70,    71,    71,    72,    72,    72,    73,    73,    74,    75,
      76,    77,    78,    78,    79,    79,    79,    79,    79,    79,
      79,    80,    80,    81,    81,    82,    82,    82,    82,    82,
      82,    83,    83,    83,    83,    83,    83,    83,    84,    85,
      85,    85,    85,    86,    86,    87,    87,    87,    87,    87,
      87,    87,    87,    87,    87,    87,    88,    88,    89,    89,
      90,    90,    90,    91,    91,    91,    92,    92,    92,    92,
      92,    92,    93,    93,    93,    93,    93,    93,    93,    93,
      93,    93,    93
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     1,     2,     2,
       3,     3,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       0,     1,     3,     3,     2,     1,     2,     1,     2,     1,
       3,     5,     3,     5,     2,     1,     2,     1,     1,     3,
       5,     3,     5,     2,     1,     3,     3,     5,     5,     5,
       3,     3,     1,     1,     1,     1,     2,     2,     1,     1,
       1,     1,     1,     4,     2,     1,     1,     1,     1,     1,
       1,     2,     2,     2,     2,     4,     5,     3,     2,     1,
       2,     3,     4,     1,     1,     1,     4,     4,     6,     9,
       9,     3,     3,     4,     5,     8,     1,     6,     5,     7,
       0,     2,     2,     1,     1,     1,     1,     1,     2,     2,
       2,     3,     1,     3,     3,     3,     3,     3,     4,     4,
       3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    30,     0,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     7,     4,     0,     9,    31,
      12,     0,     0,   116,    75,    77,    80,    76,    78,    79,
     110,   117,     0,     0,     0,    13,    37,    62,    63,    93,
      94,   106,    95,     0,    14,    71,    35,    72,    15,     0,
      16,     0,     0,   110,     0,    20,    45,    64,    68,    70,
      69,    65,    95,    18,     0,    31,    47,    48,    21,   110,
       0,     0,    17,    39,     0,     0,    19,     0,    22,     0,
      23,     0,    24,    54,    25,     0,    26,     0,    27,     0,
      28,     0,    29,     0,     5,     0,     0,     8,   119,   118,
       0,     0,     0,     0,    36,     0,     0,   122,     0,   120,
       0,     0,     0,    84,    83,     0,    82,    81,    34,     0,
       0,    66,    67,    46,    74,     0,    44,     0,     0,    74,
      38,     0,     0,     0,     0,     0,    53,     0,     0,     0,
       0,     0,     0,    10,    11,   110,   111,   112,     0,     0,
     101,   102,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   121,     0,     0,     0,     0,    87,     0,     0,
      32,    33,     0,     0,    40,     0,    42,     0,    49,    51,
      55,    56,     0,     0,     0,    60,    61,     6,     0,   115,
     113,   114,     0,     0,     0,   132,   131,   130,     0,     0,
     123,   124,   125,   126,   127,     0,     0,    96,   103,    97,
       0,    85,    73,     0,     0,    89,    88,     0,     0,     0,
       0,     0,     0,     0,   108,   104,     0,   128,   129,     0,
       0,     0,    86,    41,    90,     0,    43,    50,    52,    57,
      58,    59,     0,     0,   107,    98,     0,     0,     0,    91,
     109,     0,     0,     0,    92,   105,     0,     0,    99,   100
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    26,   152,    27,    30,    58,    60,    54,
      45,    82,    73,    86,    65,    78,    88,    90,    92,    94,
      96,    98,   100,   102,    55,    66,    56,    67,    47,    57,
     186,   226,    48,    49,    50,    51,   113,   202,    52,   118
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -89
static const yytype_int16 yypact[] =
{
     -89,     8,   -89,   211,   -33,   -23,   288,   308,   308,   360,
     236,   -11,   340,    54,    41,   308,   308,   308,    41,   106,
     -13,   308,   308,    62,    -4,   -89,   -89,    45,   -89,   -89,
     -89,   484,   484,   -89,   -89,   -89,   -89,   -89,   -89,   -89,
      81,   -89,   360,   413,   484,   -89,   -89,   -89,   -89,   -89,
     -89,    38,    48,   407,   -89,   -89,    -2,   -89,   -89,    64,
     -89,    78,   360,    81,   256,   -89,   -89,   -89,   -89,   -89,
     -89,   -89,    61,   -89,   107,   360,   -89,   -89,   -89,    59,
     431,   484,   -89,   -89,    86,    79,   -89,    87,   -89,    89,
     -89,    91,   -89,    97,   -89,   102,   -89,   116,   -89,   120,
     -89,   148,   -89,   151,   -89,   484,   484,   -89,   -89,   -89,
     123,   484,   484,   105,   -89,     1,   150,   -89,   169,   -89,
     166,     9,    69,   -89,   -89,   456,   -89,   -89,   -89,   360,
     308,   -89,   -89,   -89,   105,   392,   -89,   -17,   484,   -89,
     -89,   431,   170,   460,   360,   360,   360,   469,   360,   360,
     308,   308,   211,   179,   179,    59,   -89,   -89,     6,   484,
     154,   -89,   484,   484,   484,   201,   149,   484,   484,   484,
     484,   484,   -89,   198,    13,   158,   159,   -89,   480,   160,
     -89,   -89,   162,   165,   -89,     0,   -89,   167,   171,   172,
     -89,   -89,   193,   199,   200,   -89,   -89,   -89,   197,   -89,
     -89,   -89,   168,   204,   214,   534,   542,   549,   484,   484,
     113,   113,   -89,   -89,   -89,   484,   484,   207,   -89,   -89,
     208,   -89,   -89,   -13,   220,   251,   -89,   209,   226,   231,
     -13,   484,   106,   229,   -89,   -89,   259,   184,   184,   219,
     225,    80,   -89,   -89,   268,   249,   -89,   -89,   -89,   -89,
     -89,   -89,   232,   484,   -89,   -89,   272,   274,   269,   -89,
     -89,   239,   484,   484,   -89,   -89,   252,   253,   -89,   -89
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -89,   -89,   -89,   153,   -89,   -89,   290,   -89,   -89,   -89,
     295,   -89,   -89,   -89,   -89,   -89,   -89,   -89,   -89,   -89,
     -89,   -89,   -89,   -89,    18,   246,    20,    -7,    -9,    -8,
      84,   -89,    51,    -3,    -6,     4,   -50,   -89,   -10,   -88
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      72,    68,    69,    85,    71,    84,    83,    70,     2,   224,
      97,   159,    99,   134,    28,    74,   182,   153,   154,   174,
     175,   108,   109,   216,    46,   176,    29,    61,    59,   139,
      46,   225,    40,   117,   119,    89,    91,    93,   199,    53,
     200,   101,   103,   127,   106,   174,   175,   128,   201,    31,
      32,   176,   160,   131,    72,    68,    69,   126,    71,   132,
     114,    70,    31,    32,   217,    87,   133,   111,   112,    95,
      85,   117,    33,   140,   205,   206,   207,    31,    32,   210,
     211,   212,   213,   214,    40,    33,    41,   110,   120,   111,
     112,    43,   107,   114,    44,   117,   117,    79,   121,    41,
      33,   156,   157,    80,    81,   198,    53,    44,   104,   177,
     105,   137,   109,   129,    41,   117,   256,   257,   138,    81,
     237,   238,    44,   169,   170,   171,   131,   130,   183,   142,
     155,    85,   132,   187,   184,   141,   143,   192,   144,   191,
     145,    34,    35,    36,    37,    38,   146,   180,    39,   203,
     181,   147,   117,   117,   117,   158,   209,   117,   117,   117,
     117,   117,   188,   189,   190,   148,   193,   194,   109,   149,
     195,   196,   162,   163,   164,   165,   166,   167,   168,   169,
     170,   171,   162,   163,   164,   165,   166,   167,   168,   169,
     170,   171,   167,   168,   169,   170,   171,   150,   117,   117,
     151,   161,   173,   182,   204,   239,   240,   208,   215,   218,
     219,   221,     4,   222,   223,   243,   227,   228,   229,   234,
     172,   250,   249,   251,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,   230,   261,    31,    32,    62,   233,   231,   232,
     236,   244,   266,   267,    23,   235,    24,   241,    25,   242,
     245,   185,   247,   252,    31,    32,   135,    33,   248,   253,
     254,    34,    35,    36,    37,    38,   255,   258,    39,    63,
     259,    41,   262,   260,   263,    64,    43,    33,    53,    44,
     265,    34,    35,    36,    37,    38,    31,    32,    39,    63,
     264,    41,    76,   268,   269,   197,    43,    77,    53,    44,
     136,   246,     0,     0,     0,     0,    31,    32,     0,    33,
       0,     0,     0,    34,    35,    36,    37,    38,     0,     0,
      39,    40,     0,    41,     0,     0,     0,    42,    43,    33,
       0,    44,     0,    34,    35,    36,    37,    38,    31,    32,
      39,    40,     0,    41,     0,     0,     0,     0,    43,     0,
      53,    44,     0,     0,     0,     0,     0,     0,    31,    32,
       0,    33,     0,     0,     0,    34,    35,    36,    37,    38,
       0,     0,    39,    40,     0,    41,     0,     0,     0,    75,
      43,    33,     0,    44,     0,    34,    35,    36,    37,    38,
      31,    32,    39,    40,     0,    41,     0,     0,     0,     0,
      43,     0,     0,    44,     0,    31,   122,     0,     0,     0,
       0,    31,    32,    33,     0,     0,     0,    34,    35,    36,
      37,    38,     0,     0,    39,     0,     0,    41,    33,    31,
      32,     0,    43,     0,    33,    44,     0,   123,   124,   115,
      40,     0,    41,     0,     0,   116,     0,   125,    41,     0,
      44,     0,    33,    81,    31,   178,    44,     0,    31,    32,
       0,     0,     0,     0,    79,     0,    41,    31,    32,     0,
       0,    81,     0,     0,    44,     0,     0,    33,    31,    32,
       0,    33,    31,    32,     0,     0,   179,     0,     0,     0,
      33,    41,     0,     0,     0,    41,    81,     0,     0,    44,
      81,    33,   185,    44,    41,    33,     0,     0,     0,    81,
     220,    53,    44,     0,     0,    41,     0,     0,     0,    41,
      81,     0,     0,    44,    81,     0,     0,    44,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   164,   165,   166,
     167,   168,   169,   170,   171,   165,   166,   167,   168,   169,
     170,   171
};

static const yytype_int16 yycheck[] =
{
      10,    10,    10,    13,    10,    13,    13,    10,     0,     9,
      19,    10,    20,    63,    47,    11,    33,   105,   106,    36,
      37,    31,    32,    10,     6,    42,    49,     9,     8,    79,
      12,    31,    43,    43,    44,    15,    16,    17,    32,    52,
      34,    21,    22,    53,    48,    36,    37,    49,    42,     8,
       9,    42,    51,    62,    64,    64,    64,    53,    64,    62,
      42,    64,     8,     9,    51,    14,    62,     8,     9,    18,
      80,    81,    31,    80,   162,   163,   164,     8,     9,   167,
     168,   169,   170,   171,    43,    31,    45,     6,    50,     8,
       9,    50,    47,    75,    53,   105,   106,    43,    50,    45,
      31,   111,   112,    49,    50,   155,    52,    53,    46,    40,
      48,    50,   122,    49,    45,   125,    36,    37,    11,    50,
     208,   209,    53,    10,    11,    12,   135,    49,   138,    50,
       7,   141,   135,   143,   141,    49,    49,   147,    49,   147,
      49,    35,    36,    37,    38,    39,    49,   129,    42,   159,
     130,    49,   162,   163,   164,    50,     7,   167,   168,   169,
     170,   171,   144,   145,   146,    49,   148,   149,   178,    49,
     150,   151,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,     8,     9,    10,    11,    12,    49,   208,   209,
      49,    51,    36,    33,    50,   215,   216,     6,    10,    51,
      51,    51,     1,    51,    49,   223,    49,    46,    46,    51,
      51,   231,   230,   232,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    49,   253,     8,     9,    10,    50,    49,    49,
      36,    31,   262,   263,    43,    51,    45,    50,    47,    51,
       9,    52,    36,    34,     8,     9,    10,    31,    37,    10,
      51,    35,    36,    37,    38,    39,    51,     9,    42,    43,
      31,    45,    10,    51,    10,    49,    50,    31,    52,    53,
      51,    35,    36,    37,    38,    39,     8,     9,    42,    43,
      31,    45,    12,    51,    51,   152,    50,    12,    52,    53,
      64,   227,    -1,    -1,    -1,    -1,     8,     9,    -1,    31,
      -1,    -1,    -1,    35,    36,    37,    38,    39,    -1,    -1,
      42,    43,    -1,    45,    -1,    -1,    -1,    49,    50,    31,
      -1,    53,    -1,    35,    36,    37,    38,    39,     8,     9,
      42,    43,    -1,    45,    -1,    -1,    -1,    -1,    50,    -1,
      52,    53,    -1,    -1,    -1,    -1,    -1,    -1,     8,     9,
      -1,    31,    -1,    -1,    -1,    35,    36,    37,    38,    39,
      -1,    -1,    42,    43,    -1,    45,    -1,    -1,    -1,    49,
      50,    31,    -1,    53,    -1,    35,    36,    37,    38,    39,
       8,     9,    42,    43,    -1,    45,    -1,    -1,    -1,    -1,
      50,    -1,    -1,    53,    -1,     8,     9,    -1,    -1,    -1,
      -1,     8,     9,    31,    -1,    -1,    -1,    35,    36,    37,
      38,    39,    -1,    -1,    42,    -1,    -1,    45,    31,     8,
       9,    -1,    50,    -1,    31,    53,    -1,    40,    41,    36,
      43,    -1,    45,    -1,    -1,    42,    -1,    50,    45,    -1,
      53,    -1,    31,    50,     8,     9,    53,    -1,     8,     9,
      -1,    -1,    -1,    -1,    43,    -1,    45,     8,     9,    -1,
      -1,    50,    -1,    -1,    53,    -1,    -1,    31,     8,     9,
      -1,    31,     8,     9,    -1,    -1,    40,    -1,    -1,    -1,
      31,    45,    -1,    -1,    -1,    45,    50,    -1,    -1,    53,
      50,    31,    52,    53,    45,    31,    -1,    -1,    -1,    50,
      40,    52,    53,    -1,    -1,    45,    -1,    -1,    -1,    45,
      50,    -1,    -1,    53,    50,    -1,    -1,    53,     4,     5,
       6,     7,     8,     9,    10,    11,    12,     5,     6,     7,
       8,     9,    10,    11,    12,     6,     7,     8,     9,    10,
      11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    55,     0,    56,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    43,    45,    47,    57,    59,    47,    49,
      60,     8,     9,    31,    35,    36,    37,    38,    39,    42,
      43,    45,    49,    50,    53,    64,    78,    82,    86,    87,
      88,    89,    92,    52,    63,    78,    80,    83,    61,    80,
      62,    78,    10,    43,    49,    68,    79,    81,    82,    83,
      87,    88,    92,    66,    89,    49,    60,    64,    69,    43,
      49,    50,    65,    81,    83,    92,    67,    86,    70,    80,
      71,    80,    72,    80,    73,    86,    74,    82,    75,    83,
      76,    80,    77,    80,    46,    48,    48,    47,    92,    92,
       6,     8,     9,    90,    78,    36,    42,    92,    93,    92,
      50,    50,     9,    40,    41,    50,    89,    92,    49,    49,
      49,    82,    87,    89,    90,    10,    79,    50,    11,    90,
      81,    49,    50,    49,    49,    49,    49,    49,    49,    49,
      49,    49,    58,    93,    93,     7,    92,    92,    50,    10,
      51,    51,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    51,    36,    36,    37,    42,    40,     9,    40,
      78,    80,    33,    92,    81,    52,    84,    92,    78,    78,
      78,    83,    92,    78,    78,    80,    80,    57,    90,    32,
      34,    42,    91,    92,    50,    93,    93,    93,     6,     7,
      93,    93,    93,    93,    93,    10,    10,    51,    51,    51,
      40,    51,    51,    49,     9,    31,    85,    49,    46,    46,
      49,    49,    49,    50,    51,    51,    36,    93,    93,    92,
      92,    50,    51,    83,    31,     9,    84,    36,    37,    83,
      92,    82,    34,    10,    51,    51,    36,    37,     9,    31,
      51,    92,    10,    10,    31,    51,    92,    92,    51,    51
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
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		if((yyvsp[(1) - (2)].sym)->type == LLAB && (yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 10:
#line 92 "a.y"
    {
		(yyvsp[(1) - (3)].sym)->type = LVAR;
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 11:
#line 97 "a.y"
    {
		if((yyvsp[(1) - (3)].sym)->value != (yyvsp[(3) - (3)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (3)].sym)->name);
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 12:
#line 102 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 13:
#line 103 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 14:
#line 104 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 15:
#line 105 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 16:
#line 106 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 17:
#line 107 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 18:
#line 108 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 19:
#line 109 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 20:
#line 110 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 21:
#line 111 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 22:
#line 112 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 23:
#line 113 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 24:
#line 114 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 25:
#line 115 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 26:
#line 116 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 27:
#line 117 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 28:
#line 118 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 29:
#line 119 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 30:
#line 122 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 31:
#line 127 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 32:
#line 134 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 33:
#line 141 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 34:
#line 148 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (2)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 35:
#line 153 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 36:
#line 160 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 37:
#line 165 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 38:
#line 172 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 39:
#line 177 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 40:
#line 182 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 41:
#line 189 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
	}
    break;

  case 42:
#line 197 "a.y"
    {
		settext((yyvsp[(1) - (3)].addr).sym);
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 43:
#line 203 "a.y"
    {
		settext((yyvsp[(1) - (5)].addr).sym);
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
	}
    break;

  case 44:
#line 212 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 45:
#line 217 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 46:
#line 222 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
		(yyval.addr2).to.index = (yyvsp[(2) - (2)].addr).type;
		(yyval.addr2).to.type = D_INDIR+D_ADDR;
	}
    break;

  case 49:
#line 235 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 50:
#line 240 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).from.index != D_NONE)
			yyerror("dp shift with lhs index");
		(yyval.addr2).from.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 51:
#line 250 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 52:
#line 255 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).to.index != D_NONE)
			yyerror("dp move with lhs index");
		(yyval.addr2).to.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 53:
#line 265 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (2)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 54:
#line 270 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 55:
#line 275 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 56:
#line 282 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 57:
#line 287 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
	}
    break;

  case 58:
#line 295 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to.offset = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 59:
#line 303 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
		if((yyvsp[(1) - (5)].addr).type != D_CONST)
			yyerror("illegal constant");
		(yyval.addr2).to.offset = (yyvsp[(1) - (5)].addr).offset;
	}
    break;

  case 60:
#line 313 "a.y"
    {
		if((yyvsp[(1) - (3)].addr).type != D_CONST || (yyvsp[(3) - (3)].addr).type != D_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 61:
#line 322 "a.y"
    {
		if((yyvsp[(1) - (3)].addr).type != D_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(3) - (3)].addr).type != D_EXTERN && (yyvsp[(3) - (3)].addr).type != D_STATIC)
			yyerror("value for FUNCDATA must be symbol reference");
 		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
 		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
 	}
    break;

  case 66:
#line 339 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 67:
#line 343 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 73:
#line 356 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 74:
#line 362 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		(yyval.addr) = nullgen;
		if(pass == 2 && (yyvsp[(1) - (2)].sym)->type != LLAB)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 75:
#line 373 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 76:
#line 378 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 77:
#line 383 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 78:
#line 388 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 79:
#line 393 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SP;
	}
    break;

  case 80:
#line 398 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 81:
#line 405 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 82:
#line 411 "a.y"
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

  case 83:
#line 422 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 84:
#line 428 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 85:
#line 434 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(3) - (4)].dval);
	}
    break;

  case 86:
#line 440 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(4) - (5)].dval);
	}
    break;

  case 87:
#line 446 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 88:
#line 454 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST2;
		(yyval.addr).offset = (yyvsp[(2) - (2)].con2).v1;
		(yyval.addr).offset2 = (yyvsp[(2) - (2)].con2).v2;
	}
    break;

  case 89:
#line 463 "a.y"
    {
		(yyval.con2).v1 = (yyvsp[(1) - (1)].lval);
		(yyval.con2).v2 = ArgsSizeUnknown;
	}
    break;

  case 90:
#line 468 "a.y"
    {
		(yyval.con2).v1 = -(yyvsp[(2) - (2)].lval);
		(yyval.con2).v2 = ArgsSizeUnknown;
	}
    break;

  case 91:
#line 473 "a.y"
    {
		(yyval.con2).v1 = (yyvsp[(1) - (3)].lval);
		(yyval.con2).v2 = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 92:
#line 478 "a.y"
    {
		(yyval.con2).v1 = -(yyvsp[(2) - (4)].lval);
		(yyval.con2).v2 = (yyvsp[(4) - (4)].lval);
	}
    break;

  case 95:
#line 489 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_NONE;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 96:
#line 495 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 97:
#line 501 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_SP;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 98:
#line 507 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_NONE;
		(yyval.addr).offset = (yyvsp[(1) - (6)].lval);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 99:
#line 516 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (9)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (9)].lval);
		(yyval.addr).index = (yyvsp[(6) - (9)].lval);
		(yyval.addr).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 100:
#line 525 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (9)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (9)].lval);
		(yyval.addr).index = (yyvsp[(6) - (9)].lval);
		(yyval.addr).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 101:
#line 534 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(2) - (3)].lval);
	}
    break;

  case 102:
#line 539 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_SP;
	}
    break;

  case 103:
#line 544 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 104:
#line 550 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_NONE;
		(yyval.addr).index = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 105:
#line 558 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(2) - (8)].lval);
		(yyval.addr).index = (yyvsp[(5) - (8)].lval);
		(yyval.addr).scale = (yyvsp[(7) - (8)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 106:
#line 568 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 107:
#line 572 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (6)].addr);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 108:
#line 581 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 109:
#line 588 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 1);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 110:
#line 596 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 111:
#line 600 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 112:
#line 604 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 114:
#line 611 "a.y"
    {
		(yyval.lval) = D_AUTO;
	}
    break;

  case 117:
#line 619 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 118:
#line 623 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 119:
#line 627 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 120:
#line 631 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 121:
#line 635 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 123:
#line 642 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 124:
#line 646 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:
#line 650 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:
#line 654 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 127:
#line 658 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:
#line 662 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 129:
#line 666 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 130:
#line 670 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 131:
#line 674 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 132:
#line 678 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2552 "y.tab.c"
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



