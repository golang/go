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
	double	dval;
	char	sval[8];
	Addr	addr;
	Addr2	addr2;
}
/* Line 193 of yacc.c.  */
#line 183 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 196 "y.tab.c"

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
#define YYLAST   544

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  54
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  39
/* YYNRULES -- Number of rules.  */
#define YYNRULES  131
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
       2,     2,     2,     2,     2,     2,    50,    12,     5,     2,
      51,    52,    10,     8,    49,     9,     2,    11,     2,     2,
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
      23,    27,    31,    34,    37,    40,    43,    46,    49,    51,
      53,    56,    59,    62,    65,    68,    70,    73,    76,    79,
      82,    83,    85,    89,    93,    96,    98,   101,   103,   106,
     108,   112,   119,   125,   133,   138,   145,   148,   150,   153,
     155,   157,   161,   167,   171,   177,   180,   182,   186,   192,
     198,   202,   206,   208,   210,   212,   214,   217,   220,   222,
     224,   226,   228,   230,   235,   238,   240,   242,   244,   246,
     248,   250,   253,   256,   259,   262,   267,   273,   277,   279,
     282,   286,   291,   293,   295,   297,   302,   307,   314,   324,
     334,   338,   342,   347,   353,   362,   364,   371,   377,   385,
     386,   389,   392,   394,   396,   398,   400,   402,   405,   408,
     411,   415,   417,   421,   425,   429,   433,   437,   442,   447,
     451,   455
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      55,     0,    -1,    -1,    -1,    55,    56,    57,    -1,    -1,
      43,    46,    58,    57,    -1,    47,    -1,    59,    47,    -1,
       1,    47,    -1,    43,    48,    92,    -1,    45,    48,    92,
      -1,    13,    60,    -1,    14,    64,    -1,    15,    63,    -1,
      16,    61,    -1,    17,    62,    -1,    21,    65,    -1,    66,
      -1,    67,    -1,    18,    69,    -1,    20,    70,    -1,    23,
      71,    -1,    24,    72,    -1,    25,    73,    -1,    68,    -1,
      27,    74,    -1,    28,    75,    -1,    29,    76,    -1,    30,
      77,    -1,    -1,    49,    -1,    80,    49,    78,    -1,    78,
      49,    80,    -1,    80,    49,    -1,    80,    -1,    49,    78,
      -1,    78,    -1,    49,    81,    -1,    81,    -1,    83,    49,
      81,    -1,    19,    88,    11,    91,    49,    83,    -1,    22,
      85,    49,    50,    84,    -1,    22,    85,    49,    91,    49,
      50,    84,    -1,    26,    85,    49,    83,    -1,    26,    85,
      49,    91,    49,    83,    -1,    49,    79,    -1,    79,    -1,
      10,    88,    -1,    60,    -1,    64,    -1,    80,    49,    78,
      -1,    80,    49,    78,    46,    36,    -1,    80,    49,    78,
      -1,    80,    49,    78,    46,    37,    -1,    80,    49,    -1,
      80,    -1,    80,    49,    78,    -1,    82,    49,    78,    49,
      91,    -1,    83,    49,    78,    49,    82,    -1,    80,    49,
      80,    -1,    80,    49,    80,    -1,    82,    -1,    85,    -1,
      81,    -1,    87,    -1,    10,    82,    -1,    10,    86,    -1,
      82,    -1,    86,    -1,    83,    -1,    78,    -1,    83,    -1,
      91,    51,    33,    52,    -1,    43,    89,    -1,    35,    -1,
      38,    -1,    36,    -1,    39,    -1,    42,    -1,    37,    -1,
      50,    91,    -1,    50,    88,    -1,    50,    41,    -1,    50,
      40,    -1,    50,    51,    40,    52,    -1,    50,    51,     9,
      40,    52,    -1,    50,     9,    40,    -1,    31,    -1,     9,
      31,    -1,    31,     9,    31,    -1,     9,    31,     9,    31,
      -1,    86,    -1,    87,    -1,    91,    -1,    91,    51,    36,
      52,    -1,    91,    51,    42,    52,    -1,    91,    51,    36,
      10,    91,    52,    -1,    91,    51,    36,    52,    51,    36,
      10,    91,    52,    -1,    91,    51,    36,    52,    51,    37,
      10,    91,    52,    -1,    51,    36,    52,    -1,    51,    42,
      52,    -1,    91,    51,    37,    52,    -1,    51,    36,    10,
      91,    52,    -1,    51,    36,    52,    51,    36,    10,    91,
      52,    -1,    88,    -1,    88,    51,    36,    10,    91,    52,
      -1,    43,    89,    51,    90,    52,    -1,    43,     6,     7,
      89,    51,    34,    52,    -1,    -1,     8,    91,    -1,     9,
      91,    -1,    34,    -1,    42,    -1,    32,    -1,    31,    -1,
      45,    -1,     9,    91,    -1,     8,    91,    -1,    53,    91,
      -1,    51,    92,    52,    -1,    91,    -1,    92,     8,    92,
      -1,    92,     9,    92,    -1,    92,    10,    92,    -1,    92,
      11,    92,    -1,    92,    12,    92,    -1,    92,     6,     6,
      92,    -1,    92,     7,     7,    92,    -1,    92,     5,    92,
      -1,    92,     4,    92,    -1,    92,     3,    92,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    64,    64,    66,    65,    73,    72,    81,    82,    83,
      86,    91,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     117,   121,   128,   135,   142,   147,   154,   159,   166,   171,
     176,   183,   196,   204,   218,   226,   240,   245,   250,   258,
     259,   262,   267,   277,   282,   292,   297,   302,   309,   317,
     327,   336,   347,   348,   351,   352,   353,   357,   361,   362,
     363,   366,   367,   370,   376,   387,   393,   399,   405,   411,
     417,   425,   431,   441,   447,   453,   459,   465,   473,   480,
     487,   494,   503,   504,   507,   514,   521,   528,   538,   548,
     558,   564,   570,   577,   586,   597,   601,   610,   618,   628,
     631,   635,   641,   642,   646,   649,   650,   654,   658,   662,
     666,   672,   673,   677,   681,   685,   689,   693,   697,   701,
     705,   709
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
  "';'", "'='", "','", "'$'", "'('", "')'", "'~'", "$accept", "prog", "@1",
  "line", "@2", "inst", "nonnon", "rimrem", "remrim", "rimnon", "nonrem",
  "nonrel", "spec1", "spec2", "spec8", "spec3", "spec4", "spec5", "spec6",
  "spec7", "spec9", "spec10", "spec11", "spec12", "rem", "rom", "rim",
  "rel", "reg", "imm", "textsize", "mem", "omem", "nmem", "nam", "offset",
  "pointer", "con", "expr", 0
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
      36,    40,    41,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    54,    55,    56,    55,    58,    57,    57,    57,    57,
      59,    59,    59,    59,    59,    59,    59,    59,    59,    59,
      59,    59,    59,    59,    59,    59,    59,    59,    59,    59,
      60,    60,    61,    62,    63,    63,    64,    64,    65,    65,
      65,    66,    67,    67,    68,    68,    69,    69,    69,    70,
      70,    71,    71,    72,    72,    73,    73,    73,    74,    75,
      76,    77,    78,    78,    79,    79,    79,    79,    79,    79,
      79,    80,    80,    81,    81,    82,    82,    82,    82,    82,
      82,    83,    83,    83,    83,    83,    83,    83,    84,    84,
      84,    84,    85,    85,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    87,    87,    88,    88,    89,
      89,    89,    90,    90,    90,    91,    91,    91,    91,    91,
      91,    92,    92,    92,    92,    92,    92,    92,    92,    92,
      92,    92
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     1,     2,     2,
       3,     3,     2,     2,     2,     2,     2,     2,     1,     1,
       2,     2,     2,     2,     2,     1,     2,     2,     2,     2,
       0,     1,     3,     3,     2,     1,     2,     1,     2,     1,
       3,     6,     5,     7,     4,     6,     2,     1,     2,     1,
       1,     3,     5,     3,     5,     2,     1,     3,     5,     5,
       3,     3,     1,     1,     1,     1,     2,     2,     1,     1,
       1,     1,     1,     4,     2,     1,     1,     1,     1,     1,
       1,     2,     2,     2,     2,     4,     5,     3,     1,     2,
       3,     4,     1,     1,     1,     4,     4,     6,     9,     9,
       3,     3,     4,     5,     8,     1,     6,     5,     7,     0,
       2,     2,     1,     1,     1,     1,     1,     2,     2,     2,
       3,     1,     3,     3,     3,     3,     3,     4,     4,     3,
       3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    30,     0,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     7,     4,     0,    18,    19,
      25,     9,    31,    12,     0,     0,   115,    75,    77,    80,
      76,    78,    79,   109,   116,     0,     0,     0,    13,    37,
      62,    63,    92,    93,   105,    94,     0,    14,    71,    35,
      72,    15,     0,    16,     0,     0,   109,     0,    20,    47,
      64,    68,    70,    69,    65,    94,     0,    31,    49,    50,
      21,   109,     0,     0,    17,    39,     0,     0,     0,    22,
       0,    23,     0,    24,    56,     0,    26,     0,    27,     0,
      28,     0,    29,     0,     5,     0,     0,     8,   118,   117,
       0,     0,     0,     0,    36,     0,     0,   121,     0,   119,
       0,     0,     0,    84,    83,     0,    82,    81,    34,     0,
       0,    66,    67,    48,    74,     0,    46,     0,     0,    74,
      38,     0,     0,     0,     0,     0,    55,     0,     0,     0,
       0,     0,     0,    10,    11,   109,   110,   111,     0,     0,
     100,   101,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   120,     0,     0,     0,     0,    87,     0,     0,
      32,    33,     0,     0,    40,     0,     0,    51,    53,    57,
      44,     0,     0,     0,    60,    61,     6,     0,   114,   112,
     113,     0,     0,     0,   131,   130,   129,     0,     0,   122,
     123,   124,   125,   126,     0,     0,    95,   102,    96,     0,
      85,    73,     0,     0,    88,    42,     0,     0,     0,     0,
       0,     0,     0,   107,   103,     0,   127,   128,     0,     0,
       0,    86,    41,    89,     0,     0,    52,    54,    45,    58,
      59,     0,     0,   106,    97,     0,     0,     0,    90,    43,
     108,     0,     0,     0,    91,   104,     0,     0,    98,    99
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    26,   152,    27,    33,    61,    63,    57,
      48,    84,    28,    29,    30,    68,    80,    89,    91,    93,
      96,    98,   100,   102,    58,    69,    59,    70,    50,    60,
     225,    51,    52,    53,    54,   113,   201,    55,   118
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -87
static const yytype_int16 yypact[] =
{
     -87,    35,   -87,   242,     2,    -4,   164,   313,   313,   361,
     265,     8,   337,    60,   410,   313,   313,   313,   410,   241,
     -11,   313,   313,   -31,    17,   -87,   -87,    23,   -87,   -87,
     -87,   -87,   -87,   -87,   474,   474,   -87,   -87,   -87,   -87,
     -87,   -87,   -87,    20,   -87,   361,   401,   474,   -87,   -87,
     -87,   -87,   -87,   -87,    16,    28,   185,   -87,   -87,    22,
     -87,   -87,    25,   -87,    31,   361,    20,   289,   -87,   -87,
     -87,   -87,   -87,   -87,   -87,    48,    29,   361,   -87,   -87,
     -87,    13,   417,   474,   -87,   -87,    51,    53,    57,   -87,
      58,   -87,    59,   -87,    70,    71,   -87,    75,   -87,    80,
     -87,    86,   -87,   102,   -87,   474,   474,   -87,   -87,   -87,
      37,   474,   474,    76,   -87,     1,   103,   -87,   175,   -87,
     126,    50,    85,   -87,   -87,   426,   -87,   -87,   -87,   361,
     313,   -87,   -87,   -87,    76,   385,   -87,    81,   474,   -87,
     -87,   417,   130,   436,   361,   361,   361,   464,   361,   361,
     313,   313,   242,   525,   525,    13,   -87,   -87,    18,   474,
     113,   -87,   474,   474,   474,   165,   167,   474,   474,   474,
     474,   474,   -87,   186,     3,   123,   156,   -87,   467,   158,
     -87,   -87,   159,   163,   -87,     7,   169,   173,   177,   -87,
     -87,   182,   183,   184,   -87,   -87,   -87,   178,   -87,   -87,
     -87,   172,   187,   198,   136,   239,   532,   474,   474,    78,
      78,   -87,   -87,   -87,   474,   474,   189,   -87,   -87,   202,
     -87,   -87,   -11,   204,   228,   -87,   191,   245,   247,   -11,
     474,   241,   248,   -87,   -87,   276,   180,   180,   236,   238,
      -5,   -87,   -87,   282,   261,     7,   -87,   -87,   -87,   -87,
     -87,   243,   474,   -87,   -87,   283,   284,   274,   -87,   -87,
     -87,   254,   474,   474,   -87,   -87,   257,   259,   -87,   -87
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -87,   -87,   -87,   160,   -87,   -87,   301,   -87,   -87,   -87,
     305,   -87,   -87,   -87,   -87,   -87,   -87,   -87,   -87,   -87,
     -87,   -87,   -87,   -87,    21,   252,    26,    -7,    -9,    -8,
      84,     0,    -3,    -6,    -2,   -58,   -87,   -10,   -86
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      75,    71,    72,    87,    74,    86,    85,    73,   134,    76,
      97,   159,    99,   215,    88,   104,   223,   105,    95,   153,
     154,   111,   112,   139,   108,   109,   110,    49,   111,   112,
      64,   255,   256,    49,    62,     2,   117,   119,   224,    56,
     138,    90,    92,    94,   155,    32,   127,   101,   103,    31,
     198,    43,   199,   160,   126,   216,   131,    75,    71,    72,
     200,    74,   132,   133,    73,   106,   114,   120,    34,    35,
     107,   128,    87,   117,   129,   140,   204,   205,   206,   121,
     130,   209,   210,   211,   212,   213,   174,   175,   169,   170,
     171,    36,   176,    34,    35,   117,   117,   197,   114,   137,
     141,   156,   157,    81,   142,    44,   143,   144,   145,    82,
      56,    83,   109,    47,   182,   117,    36,   174,   175,   146,
     147,   236,   237,   176,   148,   177,   131,   158,   183,   149,
      44,    87,   132,   186,   184,   150,    83,   191,    47,   190,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   202,
     180,   151,   117,   117,   117,   161,   181,   117,   117,   117,
     117,   117,   173,   182,   203,   187,   188,   189,   109,   192,
     193,   207,    34,    35,   208,   217,   194,   195,   162,   163,
     164,   165,   166,   167,   168,   169,   170,   171,   167,   168,
     169,   170,   171,    34,   122,    36,   214,   117,   117,    37,
      38,    39,    40,    41,   238,   239,    42,    43,   218,    44,
     220,   221,   222,    45,   242,    46,    36,    47,   226,   227,
     249,   248,   250,   228,   233,   123,   124,   172,    43,   232,
      44,   229,   230,   231,   235,   243,   125,   244,    47,   234,
     240,   245,   261,     4,   164,   165,   166,   167,   168,   169,
     170,   171,   266,   267,   241,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    34,    35,    65,    37,    38,    39,    40,
      41,   246,   251,    42,   247,    23,   252,    24,   253,    25,
     254,   257,   258,   262,   263,   260,    36,    34,    35,   135,
      37,    38,    39,    40,    41,   264,   265,    42,    66,   268,
      44,   269,   196,    78,    67,    56,    46,    79,    47,   136,
      36,    34,    35,     0,    37,    38,    39,    40,    41,   259,
       0,    42,    66,     0,    44,     0,     0,     0,     0,    56,
      46,     0,    47,     0,    36,    34,    35,     0,    37,    38,
      39,    40,    41,     0,     0,    42,    43,     0,    44,     0,
       0,     0,     0,    56,    46,     0,    47,     0,    36,    34,
      35,     0,    37,    38,    39,    40,    41,     0,     0,    42,
      43,     0,    44,     0,     0,     0,    77,     0,    46,     0,
      47,     0,    36,    34,    35,     0,    37,    38,    39,    40,
      41,     0,     0,    42,    43,     0,    44,     0,     0,    34,
      35,     0,    46,     0,    47,     0,    36,     0,    34,    35,
      37,    38,    39,    40,    41,    34,    35,    42,     0,     0,
      44,     0,    36,     0,    34,   178,    46,   115,    47,     0,
       0,    36,     0,   116,    34,    35,    44,     0,    36,     0,
       0,     0,    83,    43,    47,    44,     0,    36,     0,     0,
      81,    46,    44,    47,     0,     0,   179,    36,    83,     0,
      47,    44,    34,    35,     0,    34,    35,    83,     0,    47,
       0,    44,    34,    35,     0,     0,   185,    83,     0,    47,
       0,     0,     0,     0,     0,    36,     0,     0,    36,     0,
       0,     0,     0,     0,     0,    36,     0,   219,     0,    44,
       0,     0,    44,     0,    56,    83,     0,    47,    83,    44,
      47,     0,     0,     0,     0,    83,     0,    47,   162,   163,
     164,   165,   166,   167,   168,   169,   170,   171,   165,   166,
     167,   168,   169,   170,   171
};

static const yytype_int16 yycheck[] =
{
      10,    10,    10,    13,    10,    13,    13,    10,    66,    11,
      19,    10,    20,    10,    14,    46,     9,    48,    18,   105,
     106,     8,     9,    81,    34,    35,     6,     6,     8,     9,
       9,    36,    37,    12,     8,     0,    46,    47,    31,    50,
      11,    15,    16,    17,     7,    49,    56,    21,    22,    47,
      32,    43,    34,    52,    56,    52,    65,    67,    67,    67,
      42,    67,    65,    65,    67,    48,    45,    51,     8,     9,
      47,    49,    82,    83,    49,    82,   162,   163,   164,    51,
      49,   167,   168,   169,   170,   171,    36,    37,    10,    11,
      12,    31,    42,     8,     9,   105,   106,   155,    77,    51,
      49,   111,   112,    43,    51,    45,    49,    49,    49,    49,
      50,    51,   122,    53,    33,   125,    31,    36,    37,    49,
      49,   207,   208,    42,    49,    40,   135,    51,   138,    49,
      45,   141,   135,   143,   141,    49,    51,   147,    53,   147,
       4,     5,     6,     7,     8,     9,    10,    11,    12,   159,
     129,    49,   162,   163,   164,    52,   130,   167,   168,   169,
     170,   171,    36,    33,    51,   144,   145,   146,   178,   148,
     149,     6,     8,     9,     7,    52,   150,   151,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,     8,     9,
      10,    11,    12,     8,     9,    31,    10,   207,   208,    35,
      36,    37,    38,    39,   214,   215,    42,    43,    52,    45,
      52,    52,    49,    49,   222,    51,    31,    53,    49,    46,
     230,   229,   231,    46,    52,    40,    41,    52,    43,    51,
      45,    49,    49,    49,    36,    31,    51,     9,    53,    52,
      51,    50,   252,     1,     5,     6,     7,     8,     9,    10,
      11,    12,   262,   263,    52,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,     8,     9,    10,    35,    36,    37,    38,
      39,    36,    34,    42,    37,    43,    10,    45,    52,    47,
      52,     9,    31,    10,    10,    52,    31,     8,     9,    10,
      35,    36,    37,    38,    39,    31,    52,    42,    43,    52,
      45,    52,   152,    12,    49,    50,    51,    12,    53,    67,
      31,     8,     9,    -1,    35,    36,    37,    38,    39,   245,
      -1,    42,    43,    -1,    45,    -1,    -1,    -1,    -1,    50,
      51,    -1,    53,    -1,    31,     8,     9,    -1,    35,    36,
      37,    38,    39,    -1,    -1,    42,    43,    -1,    45,    -1,
      -1,    -1,    -1,    50,    51,    -1,    53,    -1,    31,     8,
       9,    -1,    35,    36,    37,    38,    39,    -1,    -1,    42,
      43,    -1,    45,    -1,    -1,    -1,    49,    -1,    51,    -1,
      53,    -1,    31,     8,     9,    -1,    35,    36,    37,    38,
      39,    -1,    -1,    42,    43,    -1,    45,    -1,    -1,     8,
       9,    -1,    51,    -1,    53,    -1,    31,    -1,     8,     9,
      35,    36,    37,    38,    39,     8,     9,    42,    -1,    -1,
      45,    -1,    31,    -1,     8,     9,    51,    36,    53,    -1,
      -1,    31,    -1,    42,     8,     9,    45,    -1,    31,    -1,
      -1,    -1,    51,    43,    53,    45,    -1,    31,    -1,    -1,
      43,    51,    45,    53,    -1,    -1,    40,    31,    51,    -1,
      53,    45,     8,     9,    -1,     8,     9,    51,    -1,    53,
      -1,    45,     8,     9,    -1,    -1,    50,    51,    -1,    53,
      -1,    -1,    -1,    -1,    -1,    31,    -1,    -1,    31,    -1,
      -1,    -1,    -1,    -1,    -1,    31,    -1,    40,    -1,    45,
      -1,    -1,    45,    -1,    50,    51,    -1,    53,    51,    45,
      53,    -1,    -1,    -1,    -1,    51,    -1,    53,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,     6,     7,
       8,     9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    55,     0,    56,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    43,    45,    47,    57,    59,    66,    67,
      68,    47,    49,    60,     8,     9,    31,    35,    36,    37,
      38,    39,    42,    43,    45,    49,    51,    53,    64,    78,
      82,    85,    86,    87,    88,    91,    50,    63,    78,    80,
      83,    61,    80,    62,    78,    10,    43,    49,    69,    79,
      81,    82,    83,    86,    87,    91,    88,    49,    60,    64,
      70,    43,    49,    51,    65,    81,    83,    91,    85,    71,
      80,    72,    80,    73,    80,    85,    74,    82,    75,    83,
      76,    80,    77,    80,    46,    48,    48,    47,    91,    91,
       6,     8,     9,    89,    78,    36,    42,    91,    92,    91,
      51,    51,     9,    40,    41,    51,    88,    91,    49,    49,
      49,    82,    86,    88,    89,    10,    79,    51,    11,    89,
      81,    49,    51,    49,    49,    49,    49,    49,    49,    49,
      49,    49,    58,    92,    92,     7,    91,    91,    51,    10,
      52,    52,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    52,    36,    36,    37,    42,    40,     9,    40,
      78,    80,    33,    91,    81,    50,    91,    78,    78,    78,
      83,    91,    78,    78,    80,    80,    57,    89,    32,    34,
      42,    90,    91,    51,    92,    92,    92,     6,     7,    92,
      92,    92,    92,    92,    10,    10,    52,    52,    52,    40,
      52,    52,    49,     9,    31,    84,    49,    46,    46,    49,
      49,    49,    51,    52,    52,    36,    92,    92,    91,    91,
      51,    52,    83,    31,     9,    50,    36,    37,    83,    91,
      82,    34,    10,    52,    52,    36,    37,     9,    31,    84,
      52,    91,    10,    10,    31,    52,    91,    91,    52,    52
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
#line 66 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:
#line 73 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		if((yyvsp[(1) - (2)].sym)->type == LLAB && (yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 10:
#line 87 "a.y"
    {
		(yyvsp[(1) - (3)].sym)->type = LVAR;
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 11:
#line 92 "a.y"
    {
		if((yyvsp[(1) - (3)].sym)->value != (yyvsp[(3) - (3)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (3)].sym)->name);
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 12:
#line 97 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 13:
#line 98 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 14:
#line 99 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 15:
#line 100 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 16:
#line 101 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 17:
#line 102 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 20:
#line 105 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 21:
#line 106 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 22:
#line 107 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 23:
#line 108 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 24:
#line 109 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 26:
#line 111 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 27:
#line 112 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 28:
#line 113 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 29:
#line 114 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 30:
#line 117 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 31:
#line 122 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 32:
#line 129 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 33:
#line 136 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 34:
#line 143 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (2)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 35:
#line 148 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 36:
#line 155 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 37:
#line 160 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 38:
#line 167 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 39:
#line 172 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 40:
#line 177 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 41:
#line 184 "a.y"
    {
		Addr2 a;
		a.from = (yyvsp[(2) - (6)].addr);
		a.to = (yyvsp[(6) - (6)].addr);
		outcode(ADATA, &a);
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 42:
#line 197 "a.y"
    {
		Addr2 a;
		settext((yyvsp[(2) - (5)].addr).sym);
		a.from = (yyvsp[(2) - (5)].addr);
		a.to = (yyvsp[(5) - (5)].addr);
		outcode(ATEXT, &a);
	}
    break;

  case 43:
#line 205 "a.y"
    {
		Addr2 a;
		settext((yyvsp[(2) - (7)].addr).sym);
		a.from = (yyvsp[(2) - (7)].addr);
		a.to = (yyvsp[(7) - (7)].addr);
		outcode(ATEXT, &a);
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (7)].lval);
		}
	}
    break;

  case 44:
#line 219 "a.y"
    {
		Addr2 a;
		settext((yyvsp[(2) - (4)].addr).sym);
		a.from = (yyvsp[(2) - (4)].addr);
		a.to = (yyvsp[(4) - (4)].addr);
		outcode(AGLOBL, &a);
	}
    break;

  case 45:
#line 227 "a.y"
    {
		Addr2 a;
		settext((yyvsp[(2) - (6)].addr).sym);
		a.from = (yyvsp[(2) - (6)].addr);
		a.to = (yyvsp[(6) - (6)].addr);
		outcode(AGLOBL, &a);
		if(pass > 1) {
			lastpc->from3.type = TYPE_CONST;
			lastpc->from3.offset = (yyvsp[(4) - (6)].lval);
		}
	}
    break;

  case 46:
#line 241 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 47:
#line 246 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 48:
#line 251 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
		(yyval.addr2).to.type = TYPE_INDIR;
	}
    break;

  case 51:
#line 263 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 52:
#line 268 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).from.index != TYPE_NONE)
			yyerror("dp shift with lhs index");
		(yyval.addr2).from.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 53:
#line 278 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 54:
#line 283 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).to.index != TYPE_NONE)
			yyerror("dp move with lhs index");
		(yyval.addr2).to.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 55:
#line 293 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (2)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 56:
#line 298 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 57:
#line 303 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 58:
#line 310 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to.offset = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 59:
#line 318 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
		if((yyvsp[(1) - (5)].addr).type != TYPE_CONST)
			yyerror("illegal constant");
		(yyval.addr2).to.offset = (yyvsp[(1) - (5)].addr).offset;
	}
    break;

  case 60:
#line 328 "a.y"
    {
		if((yyvsp[(1) - (3)].addr).type != TYPE_CONST || (yyvsp[(3) - (3)].addr).type != TYPE_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 61:
#line 337 "a.y"
    {
		if((yyvsp[(1) - (3)].addr).type != TYPE_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(3) - (3)].addr).type != TYPE_MEM || ((yyvsp[(3) - (3)].addr).name != NAME_EXTERN && (yyvsp[(3) - (3)].addr).name != NAME_STATIC))
			yyerror("value for FUNCDATA must be symbol reference");
 		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
 		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
 	}
    break;

  case 66:
#line 354 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 67:
#line 358 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 73:
#line 371 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 74:
#line 377 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		(yyval.addr) = nullgen;
		if(pass == 2 && (yyvsp[(1) - (2)].sym)->type != LLAB)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 75:
#line 388 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 76:
#line 394 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 77:
#line 400 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 78:
#line 406 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 79:
#line 412 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_SP;
	}
    break;

  case 80:
#line 418 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 81:
#line 426 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 82:
#line 432 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = TYPE_ADDR;
		/*
		if($2.name == NAME_AUTO || $2.name == NAME_PARAM)
			yyerror("constant cannot be automatic: %s",
				$2.sym->name);
		 */
	}
    break;

  case 83:
#line 442 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 84:
#line 448 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 85:
#line 454 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = (yyvsp[(3) - (4)].dval);
	}
    break;

  case 86:
#line 460 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(4) - (5)].dval);
	}
    break;

  case 87:
#line 466 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 88:
#line 474 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 89:
#line 481 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (2)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 90:
#line 488 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (3)].lval);
		(yyval.addr).u.argsize = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 91:
#line 495 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (4)].lval);
		(yyval.addr).u.argsize = (yyvsp[(4) - (4)].lval);
	}
    break;

  case 94:
#line 508 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = REG_NONE;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 95:
#line 515 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 96:
#line 522 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = REG_SP;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 97:
#line 529 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = REG_NONE;
		(yyval.addr).offset = (yyvsp[(1) - (6)].lval);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 98:
#line 539 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (9)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (9)].lval);
		(yyval.addr).index = (yyvsp[(6) - (9)].lval);
		(yyval.addr).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 99:
#line 549 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (9)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (9)].lval);
		(yyval.addr).index = (yyvsp[(6) - (9)].lval);
		(yyval.addr).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 100:
#line 559 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 101:
#line 565 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = REG_SP;
	}
    break;

  case 102:
#line 571 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 103:
#line 578 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = REG_NONE;
		(yyval.addr).index = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 104:
#line 587 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (8)].lval);
		(yyval.addr).index = (yyvsp[(5) - (8)].lval);
		(yyval.addr).scale = (yyvsp[(7) - (8)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 105:
#line 598 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 106:
#line 602 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (6)].addr);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 107:
#line 611 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 108:
#line 619 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = NAME_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 1);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 109:
#line 628 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 110:
#line 632 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 111:
#line 636 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 113:
#line 643 "a.y"
    {
		(yyval.lval) = NAME_AUTO;
	}
    break;

  case 116:
#line 651 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 117:
#line 655 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 118:
#line 659 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 119:
#line 663 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 120:
#line 667 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 122:
#line 674 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 123:
#line 678 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 124:
#line 682 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:
#line 686 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:
#line 690 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 127:
#line 694 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 128:
#line 698 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 129:
#line 702 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 130:
#line 706 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 131:
#line 710 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2565 "y.tab.c"
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



