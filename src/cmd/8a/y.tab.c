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
     LCONST = 274,
     LFP = 275,
     LPC = 276,
     LSB = 277,
     LBREG = 278,
     LLREG = 279,
     LSREG = 280,
     LFREG = 281,
     LXREG = 282,
     LFCONST = 283,
     LSCONST = 284,
     LSP = 285,
     LNAME = 286,
     LLAB = 287,
     LVAR = 288
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
#define LCONST 274
#define LFP 275
#define LPC 276
#define LSB 277
#define LBREG 278
#define LLREG 279
#define LSREG 280
#define LFREG 281
#define LXREG 282
#define LFCONST 283
#define LSCONST 284
#define LSP 285
#define LNAME 286
#define LLAB 287
#define LVAR 288




/* Copy the first part of user declarations.  */
#line 31 "a.y"

#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and a.h re-#defines getc */
#include <libc.h>
#include "a.h"


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
#line 37 "a.y"
{
	Sym	*sym;
	int32	lval;
	struct {
		int32 v1;
		int32 v2;
	} con2;
	double	dval;
	char	sval[8];
	Gen	gen;
	Gen2	gen2;
}
/* Line 193 of yacc.c.  */
#line 182 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 195 "y.tab.c"

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
#define YYLAST   537

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  52
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  39
/* YYNRULES -- Number of rules.  */
#define YYNRULES  131
/* YYNRULES -- Number of states.  */
#define YYNSTATES  266

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   288

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    50,    12,     5,     2,
      48,    49,    10,     8,    47,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    44,    45,
       6,    46,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    51,     2,     2,     2,
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
      35,    36,    37,    38,    39,    40,    41,    42,    43
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     5,     9,    10,    15,    16,    21,
      23,    26,    29,    33,    37,    40,    43,    46,    49,    52,
      55,    58,    61,    64,    67,    70,    73,    76,    79,    82,
      85,    86,    88,    92,    96,    99,   101,   104,   106,   109,
     111,   115,   121,   125,   131,   134,   136,   139,   141,   143,
     147,   153,   157,   163,   166,   168,   172,   176,   182,   188,
     194,   196,   198,   200,   202,   205,   208,   210,   212,   214,
     216,   218,   223,   226,   229,   231,   233,   235,   237,   239,
     241,   244,   247,   250,   253,   258,   264,   268,   271,   273,
     276,   280,   285,   287,   289,   291,   296,   301,   308,   318,
     328,   332,   336,   341,   347,   356,   358,   365,   371,   379,
     380,   383,   386,   388,   390,   392,   394,   396,   399,   402,
     405,   409,   411,   415,   419,   423,   427,   431,   436,   441,
     445,   449
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      53,     0,    -1,    -1,    -1,    53,    54,    55,    -1,    -1,
      42,    44,    56,    55,    -1,    -1,    41,    44,    57,    55,
      -1,    45,    -1,    58,    45,    -1,     1,    45,    -1,    41,
      46,    90,    -1,    43,    46,    90,    -1,    13,    59,    -1,
      14,    63,    -1,    15,    62,    -1,    16,    60,    -1,    17,
      61,    -1,    21,    64,    -1,    19,    65,    -1,    22,    66,
      -1,    18,    67,    -1,    20,    68,    -1,    23,    69,    -1,
      24,    70,    -1,    25,    71,    -1,    26,    72,    -1,    27,
      73,    -1,    28,    74,    -1,    -1,    47,    -1,    77,    47,
      75,    -1,    75,    47,    77,    -1,    77,    47,    -1,    77,
      -1,    47,    75,    -1,    75,    -1,    47,    78,    -1,    78,
      -1,    80,    47,    78,    -1,    86,    11,    89,    47,    80,
      -1,    83,    47,    81,    -1,    83,    47,    89,    47,    81,
      -1,    47,    76,    -1,    76,    -1,    10,    86,    -1,    59,
      -1,    63,    -1,    77,    47,    75,    -1,    77,    47,    75,
      44,    34,    -1,    77,    47,    75,    -1,    77,    47,    75,
      44,    35,    -1,    77,    47,    -1,    77,    -1,    77,    47,
      75,    -1,    83,    47,    80,    -1,    83,    47,    89,    47,
      80,    -1,    79,    47,    75,    47,    89,    -1,    80,    47,
      75,    47,    79,    -1,    79,    -1,    83,    -1,    78,    -1,
      85,    -1,    10,    79,    -1,    10,    84,    -1,    79,    -1,
      84,    -1,    80,    -1,    75,    -1,    80,    -1,    89,    48,
      31,    49,    -1,    41,    87,    -1,    42,    87,    -1,    33,
      -1,    36,    -1,    34,    -1,    37,    -1,    40,    -1,    35,
      -1,    50,    89,    -1,    50,    86,    -1,    50,    39,    -1,
      50,    38,    -1,    50,    48,    38,    49,    -1,    50,    48,
       9,    38,    49,    -1,    50,     9,    38,    -1,    50,    82,
      -1,    29,    -1,     9,    29,    -1,    29,     9,    29,    -1,
       9,    29,     9,    29,    -1,    84,    -1,    85,    -1,    89,
      -1,    89,    48,    34,    49,    -1,    89,    48,    40,    49,
      -1,    89,    48,    34,    10,    89,    49,    -1,    89,    48,
      34,    49,    48,    34,    10,    89,    49,    -1,    89,    48,
      34,    49,    48,    35,    10,    89,    49,    -1,    48,    34,
      49,    -1,    48,    40,    49,    -1,    89,    48,    35,    49,
      -1,    48,    34,    10,    89,    49,    -1,    48,    34,    49,
      48,    34,    10,    89,    49,    -1,    86,    -1,    86,    48,
      34,    10,    89,    49,    -1,    41,    87,    48,    88,    49,
      -1,    41,     6,     7,    87,    48,    32,    49,    -1,    -1,
       8,    89,    -1,     9,    89,    -1,    32,    -1,    40,    -1,
      30,    -1,    29,    -1,    43,    -1,     9,    89,    -1,     8,
      89,    -1,    51,    89,    -1,    48,    90,    49,    -1,    89,
      -1,    90,     8,    90,    -1,    90,     9,    90,    -1,    90,
      10,    90,    -1,    90,    11,    90,    -1,    90,    12,    90,
      -1,    90,     6,     6,    90,    -1,    90,     7,     7,    90,
      -1,    90,     5,    90,    -1,    90,     4,    90,    -1,    90,
       3,    90,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    68,    68,    70,    69,    77,    76,    84,    83,    89,
      90,    91,    94,    99,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     123,   127,   134,   141,   148,   153,   160,   165,   172,   177,
     182,   189,   197,   202,   210,   215,   220,   229,   230,   233,
     238,   248,   253,   263,   268,   273,   280,   285,   293,   301,
     311,   312,   315,   316,   317,   321,   325,   326,   327,   330,
     331,   334,   340,   349,   358,   363,   368,   373,   378,   383,
     390,   396,   407,   413,   419,   425,   431,   439,   448,   453,
     458,   463,   470,   471,   474,   480,   486,   492,   501,   510,
     519,   524,   529,   535,   543,   553,   557,   566,   573,   582,
     585,   589,   595,   596,   600,   603,   604,   608,   612,   616,
     620,   626,   627,   631,   635,   639,   643,   647,   651,   655,
     659,   663
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
  "LTYPEM", "LTYPEI", "LTYPEG", "LTYPEXC", "LTYPEX", "LCONST", "LFP",
  "LPC", "LSB", "LBREG", "LLREG", "LSREG", "LFREG", "LXREG", "LFCONST",
  "LSCONST", "LSP", "LNAME", "LLAB", "LVAR", "':'", "';'", "'='", "','",
  "'('", "')'", "'$'", "'~'", "$accept", "prog", "@1", "line", "@2", "@3",
  "inst", "nonnon", "rimrem", "remrim", "rimnon", "nonrem", "nonrel",
  "spec1", "spec2", "spec3", "spec4", "spec5", "spec6", "spec7", "spec8",
  "spec9", "spec10", "rem", "rom", "rim", "rel", "reg", "imm", "imm2",
  "con2", "mem", "omem", "nmem", "nam", "offset", "pointer", "con", "expr", 0
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
     285,   286,   287,   288,    58,    59,    61,    44,    40,    41,
      36,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    52,    53,    54,    53,    56,    55,    57,    55,    55,
      55,    55,    58,    58,    58,    58,    58,    58,    58,    58,
      58,    58,    58,    58,    58,    58,    58,    58,    58,    58,
      59,    59,    60,    61,    62,    62,    63,    63,    64,    64,
      64,    65,    66,    66,    67,    67,    67,    68,    68,    69,
      69,    70,    70,    71,    71,    71,    72,    72,    73,    74,
      75,    75,    76,    76,    76,    76,    76,    76,    76,    77,
      77,    78,    78,    78,    79,    79,    79,    79,    79,    79,
      80,    80,    80,    80,    80,    80,    80,    81,    82,    82,
      82,    82,    83,    83,    84,    84,    84,    84,    84,    84,
      84,    84,    84,    84,    84,    85,    85,    86,    86,    87,
      87,    87,    88,    88,    88,    89,    89,    89,    89,    89,
      89,    90,    90,    90,    90,    90,    90,    90,    90,    90,
      90,    90
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     0,     4,     1,
       2,     2,     3,     3,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       0,     1,     3,     3,     2,     1,     2,     1,     2,     1,
       3,     5,     3,     5,     2,     1,     2,     1,     1,     3,
       5,     3,     5,     2,     1,     3,     3,     5,     5,     5,
       1,     1,     1,     1,     2,     2,     1,     1,     1,     1,
       1,     4,     2,     2,     1,     1,     1,     1,     1,     1,
       2,     2,     2,     2,     4,     5,     3,     2,     1,     2,
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
       0,     0,     0,     0,     9,     4,     0,    11,    31,    14,
       0,     0,   115,    74,    76,    79,    75,    77,    78,   109,
     116,     0,     0,     0,    15,    37,    60,    61,    92,    93,
     105,    94,     0,    16,    69,    35,    70,    17,     0,    18,
       0,     0,   109,   109,     0,    22,    45,    62,    66,    68,
      67,    63,    94,    20,     0,    31,    47,    48,    23,   109,
       0,     0,    19,    39,     0,     0,    21,     0,    24,     0,
      25,     0,    26,    54,    27,     0,    28,     0,    29,     0,
       7,     0,     5,     0,    10,   118,   117,     0,     0,     0,
       0,    36,     0,     0,   121,     0,   119,     0,     0,     0,
      83,    82,     0,    81,    80,    34,     0,     0,    64,    65,
      46,    72,    73,     0,    44,     0,     0,    72,    38,     0,
       0,     0,     0,     0,    53,     0,     0,     0,     0,    12,
       0,    13,   109,   110,   111,     0,     0,   100,   101,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   120,
       0,     0,     0,     0,    86,     0,     0,    32,    33,     0,
       0,    40,     0,    42,     0,    49,    51,    55,    56,     0,
       0,     0,     8,     6,     0,   114,   112,   113,     0,     0,
       0,   131,   130,   129,     0,     0,   122,   123,   124,   125,
     126,     0,     0,    95,   102,    96,     0,    84,    71,     0,
       0,    88,    87,     0,     0,     0,     0,     0,     0,     0,
     107,   103,     0,   127,   128,     0,     0,     0,    85,    41,
      89,     0,    43,    50,    52,    57,    58,    59,     0,     0,
     106,    97,     0,     0,     0,    90,   108,     0,     0,     0,
      91,   104,     0,     0,    98,    99
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    25,   150,   148,    26,    29,    57,    59,
      53,    44,    82,    73,    86,    65,    78,    88,    90,    92,
      94,    96,    98,    54,    66,    55,    67,    46,    56,   183,
     222,    47,    48,    49,    50,   110,   198,    51,   115
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -100
static const yytype_int16 yypact[] =
{
    -100,    22,  -100,   165,   -32,   -22,   265,   288,   288,   334,
     195,    -1,   311,   212,   416,   288,   288,   288,   416,    81,
       7,   -16,    24,   -12,  -100,  -100,    -4,  -100,  -100,  -100,
     469,   469,  -100,  -100,  -100,  -100,  -100,  -100,  -100,    39,
    -100,   334,   387,   469,  -100,  -100,  -100,  -100,  -100,  -100,
      46,    65,   372,  -100,  -100,    72,  -100,  -100,    83,  -100,
      86,   334,    39,   102,   242,  -100,  -100,  -100,  -100,  -100,
    -100,  -100,    77,  -100,   117,   334,  -100,  -100,  -100,   102,
     410,   469,  -100,  -100,    89,    90,  -100,    92,  -100,    97,
    -100,    98,  -100,   100,  -100,   101,  -100,   105,  -100,   106,
    -100,   469,  -100,   469,  -100,  -100,  -100,   135,   469,   469,
     114,  -100,    -6,   128,  -100,    71,  -100,   175,    32,   218,
    -100,  -100,   425,  -100,  -100,  -100,   334,   288,  -100,  -100,
    -100,   114,  -100,   357,  -100,    29,   469,  -100,  -100,   410,
     181,   440,   334,   334,   334,   457,   334,   334,   165,   164,
     165,   164,   102,  -100,  -100,     6,   469,   166,  -100,   469,
     469,   469,   207,   208,   469,   469,   469,   469,   469,  -100,
     206,     4,   173,   174,  -100,   463,   176,  -100,  -100,   184,
     187,  -100,    15,  -100,   193,   200,   213,  -100,  -100,   211,
     217,   220,  -100,  -100,   222,  -100,  -100,  -100,   216,   219,
     238,   517,   525,    78,   469,   469,    95,    95,  -100,  -100,
    -100,   469,   469,   232,  -100,  -100,   237,  -100,  -100,     7,
     252,   278,  -100,   239,   254,   256,     7,   469,    81,   263,
    -100,  -100,   293,   188,   188,   255,   258,    88,  -100,  -100,
     300,   281,  -100,  -100,  -100,  -100,  -100,  -100,   262,   469,
    -100,  -100,   304,   305,   289,  -100,  -100,   277,   469,   469,
    -100,  -100,   283,   284,  -100,  -100
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -100,  -100,  -100,   -99,  -100,  -100,  -100,   315,  -100,  -100,
    -100,   318,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    -100,  -100,  -100,    17,   270,     0,    -7,    -9,    -8,   112,
    -100,    13,     1,    -3,    -2,   -44,  -100,   -10,   -64
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      72,    68,    69,    85,   156,    84,    83,    71,    58,    74,
      97,    70,    99,    27,   212,    89,    91,    93,   131,   132,
     105,   106,     2,    45,   220,    28,    60,    87,   100,    45,
     101,    95,   114,   116,   103,   137,   195,   149,   196,   151,
      39,   104,   124,   157,   221,   107,   197,   108,   109,   192,
     123,   193,   128,   213,    72,    68,    69,    52,   111,   130,
     179,    71,   129,   171,   172,    70,   171,   172,   102,   173,
      85,   114,   173,   138,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   162,   163,   164,   165,   166,   167,
     168,   114,   111,   114,   117,   201,   202,   203,   153,   154,
     206,   207,   208,   209,   210,   166,   167,   168,   194,   106,
     108,   109,   114,   118,    33,    34,    35,    36,    37,   125,
     169,    38,   252,   253,   128,   135,   180,   178,   136,    85,
     126,   184,   181,   127,   129,   189,   139,   188,   140,   141,
     233,   234,   152,   177,   142,   143,   199,   144,   145,   114,
     114,   114,   146,   147,   114,   114,   114,   114,   114,   185,
     186,   187,   155,   190,   191,   106,     4,   159,   160,   161,
     162,   163,   164,   165,   166,   167,   168,   158,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,   114,   114,   164,   165,   166,   167,
     168,   235,   236,    30,    31,    61,    21,    22,    23,   170,
      24,   239,   179,   204,   200,   205,   211,   246,   245,   247,
      30,    31,   214,   215,    32,   217,    30,    31,    33,    34,
      35,    36,    37,   218,   219,    38,    62,    63,    40,   257,
     223,    32,    64,    42,   224,    52,    43,    32,   262,   263,
      30,    31,   133,    79,    63,    40,   174,   225,   226,    80,
      81,    40,    52,    43,   227,   230,    81,   228,   231,    43,
     229,    32,   232,    30,    31,    33,    34,    35,    36,    37,
     237,   240,    38,    62,    63,    40,   238,   241,   243,   182,
      42,   244,    52,    43,    32,   248,    30,    31,    33,    34,
      35,    36,    37,   249,   250,    38,    39,   251,    40,   254,
     255,   256,    41,    42,   258,   259,    43,    32,   260,    30,
      31,    33,    34,    35,    36,    37,   261,    76,    38,    39,
      77,    40,   264,   265,   134,   242,    42,     0,    52,    43,
      32,     0,    30,    31,    33,    34,    35,    36,    37,     0,
       0,    38,    39,     0,    40,     0,     0,     0,    75,    42,
       0,     0,    43,    32,     0,    30,    31,    33,    34,    35,
      36,    37,     0,     0,    38,    39,     0,    40,     0,     0,
      30,   119,    42,     0,     0,    43,    32,     0,     0,     0,
      33,    34,    35,    36,    37,    30,    31,    38,     0,     0,
      40,    32,     0,     0,     0,    42,     0,     0,    43,     0,
     120,   121,     0,    39,     0,    40,    32,     0,    30,    31,
     122,   112,     0,    43,    30,    31,     0,   113,     0,     0,
      40,     0,     0,    30,   175,    81,     0,     0,    43,    32,
       0,     0,     0,     0,     0,    32,     0,     0,    30,    31,
       0,    79,    63,    40,    32,     0,     0,    39,    81,    40,
       0,    43,     0,   176,    42,    30,    31,    43,    40,    32,
       0,    30,    31,    81,     0,     0,    43,    30,    31,     0,
       0,     0,     0,    40,     0,     0,    32,     0,    81,     0,
     182,    43,    32,     0,     0,     0,     0,     0,    32,     0,
      40,   216,     0,     0,     0,    81,    40,    52,    43,     0,
       0,    81,    40,     0,    43,     0,     0,    81,     0,     0,
      43,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     161,   162,   163,   164,   165,   166,   167,   168
};

static const yytype_int16 yycheck[] =
{
      10,    10,    10,    13,    10,    13,    13,    10,     8,    11,
      19,    10,    20,    45,    10,    15,    16,    17,    62,    63,
      30,    31,     0,     6,     9,    47,     9,    14,    44,    12,
      46,    18,    42,    43,    46,    79,    30,   101,    32,   103,
      41,    45,    52,    49,    29,     6,    40,     8,     9,   148,
      52,   150,    61,    49,    64,    64,    64,    50,    41,    61,
      31,    64,    61,    34,    35,    64,    34,    35,    44,    40,
      80,    81,    40,    80,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,     6,     7,     8,     9,    10,    11,
      12,   101,    75,   103,    48,   159,   160,   161,   108,   109,
     164,   165,   166,   167,   168,    10,    11,    12,   152,   119,
       8,     9,   122,    48,    33,    34,    35,    36,    37,    47,
      49,    40,    34,    35,   133,    48,   136,   127,    11,   139,
      47,   141,   139,    47,   133,   145,    47,   145,    48,    47,
     204,   205,     7,   126,    47,    47,   156,    47,    47,   159,
     160,   161,    47,    47,   164,   165,   166,   167,   168,   142,
     143,   144,    48,   146,   147,   175,     1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    49,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,   204,   205,     8,     9,    10,    11,
      12,   211,   212,     8,     9,    10,    41,    42,    43,    34,
      45,   219,    31,     6,    48,     7,    10,   227,   226,   228,
       8,     9,    49,    49,    29,    49,     8,     9,    33,    34,
      35,    36,    37,    49,    47,    40,    41,    42,    43,   249,
      47,    29,    47,    48,    44,    50,    51,    29,   258,   259,
       8,     9,    10,    41,    42,    43,    38,    44,    47,    47,
      48,    43,    50,    51,    47,    49,    48,    47,    49,    51,
      48,    29,    34,     8,     9,    33,    34,    35,    36,    37,
      48,    29,    40,    41,    42,    43,    49,     9,    34,    50,
      48,    35,    50,    51,    29,    32,     8,     9,    33,    34,
      35,    36,    37,    10,    49,    40,    41,    49,    43,     9,
      29,    49,    47,    48,    10,    10,    51,    29,    29,     8,
       9,    33,    34,    35,    36,    37,    49,    12,    40,    41,
      12,    43,    49,    49,    64,   223,    48,    -1,    50,    51,
      29,    -1,     8,     9,    33,    34,    35,    36,    37,    -1,
      -1,    40,    41,    -1,    43,    -1,    -1,    -1,    47,    48,
      -1,    -1,    51,    29,    -1,     8,     9,    33,    34,    35,
      36,    37,    -1,    -1,    40,    41,    -1,    43,    -1,    -1,
       8,     9,    48,    -1,    -1,    51,    29,    -1,    -1,    -1,
      33,    34,    35,    36,    37,     8,     9,    40,    -1,    -1,
      43,    29,    -1,    -1,    -1,    48,    -1,    -1,    51,    -1,
      38,    39,    -1,    41,    -1,    43,    29,    -1,     8,     9,
      48,    34,    -1,    51,     8,     9,    -1,    40,    -1,    -1,
      43,    -1,    -1,     8,     9,    48,    -1,    -1,    51,    29,
      -1,    -1,    -1,    -1,    -1,    29,    -1,    -1,     8,     9,
      -1,    41,    42,    43,    29,    -1,    -1,    41,    48,    43,
      -1,    51,    -1,    38,    48,     8,     9,    51,    43,    29,
      -1,     8,     9,    48,    -1,    -1,    51,     8,     9,    -1,
      -1,    -1,    -1,    43,    -1,    -1,    29,    -1,    48,    -1,
      50,    51,    29,    -1,    -1,    -1,    -1,    -1,    29,    -1,
      43,    38,    -1,    -1,    -1,    48,    43,    50,    51,    -1,
      -1,    48,    43,    -1,    51,    -1,    -1,    48,    -1,    -1,
      51,     4,     5,     6,     7,     8,     9,    10,    11,    12,
       5,     6,     7,     8,     9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    53,     0,    54,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    41,    42,    43,    45,    55,    58,    45,    47,    59,
       8,     9,    29,    33,    34,    35,    36,    37,    40,    41,
      43,    47,    48,    51,    63,    75,    79,    83,    84,    85,
      86,    89,    50,    62,    75,    77,    80,    60,    77,    61,
      75,    10,    41,    42,    47,    67,    76,    78,    79,    80,
      84,    85,    89,    65,    86,    47,    59,    63,    68,    41,
      47,    48,    64,    78,    80,    89,    66,    83,    69,    77,
      70,    77,    71,    77,    72,    83,    73,    79,    74,    80,
      44,    46,    44,    46,    45,    89,    89,     6,     8,     9,
      87,    75,    34,    40,    89,    90,    89,    48,    48,     9,
      38,    39,    48,    86,    89,    47,    47,    47,    79,    84,
      86,    87,    87,    10,    76,    48,    11,    87,    78,    47,
      48,    47,    47,    47,    47,    47,    47,    47,    57,    90,
      56,    90,     7,    89,    89,    48,    10,    49,    49,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    49,
      34,    34,    35,    40,    38,     9,    38,    75,    77,    31,
      89,    78,    50,    81,    89,    75,    75,    75,    80,    89,
      75,    75,    55,    55,    87,    30,    32,    40,    88,    89,
      48,    90,    90,    90,     6,     7,    90,    90,    90,    90,
      90,    10,    10,    49,    49,    49,    38,    49,    49,    47,
       9,    29,    82,    47,    44,    44,    47,    47,    47,    48,
      49,    49,    34,    90,    90,    89,    89,    48,    49,    80,
      29,     9,    81,    34,    35,    80,    89,    79,    32,    10,
      49,    49,    34,    35,     9,    29,    49,    89,    10,    10,
      29,    49,    89,    89,    49,    49
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
#line 70 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:
#line 77 "a.y"
    {
		if((yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->name);
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 7:
#line 84 "a.y"
    {
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 12:
#line 95 "a.y"
    {
		(yyvsp[(1) - (3)].sym)->type = LVAR;
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 13:
#line 100 "a.y"
    {
		if((yyvsp[(1) - (3)].sym)->value != (yyvsp[(3) - (3)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (3)].sym)->name);
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 14:
#line 105 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 15:
#line 106 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 16:
#line 107 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 17:
#line 108 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 18:
#line 109 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 19:
#line 110 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 20:
#line 111 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 21:
#line 112 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 22:
#line 113 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 23:
#line 114 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 24:
#line 115 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 25:
#line 116 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 26:
#line 117 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 27:
#line 118 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 28:
#line 119 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 29:
#line 120 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].gen2)); }
    break;

  case 30:
#line 123 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = nullgen;
	}
    break;

  case 31:
#line 128 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = nullgen;
	}
    break;

  case 32:
#line 135 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 33:
#line 142 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 34:
#line 149 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (2)].gen);
		(yyval.gen2).to = nullgen;
	}
    break;

  case 35:
#line 154 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (1)].gen);
		(yyval.gen2).to = nullgen;
	}
    break;

  case 36:
#line 161 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 37:
#line 166 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(1) - (1)].gen);
	}
    break;

  case 38:
#line 173 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 39:
#line 178 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(1) - (1)].gen);
	}
    break;

  case 40:
#line 183 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 41:
#line 190 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.gen2).to = (yyvsp[(5) - (5)].gen);
	}
    break;

  case 42:
#line 198 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 43:
#line 203 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.gen2).to = (yyvsp[(5) - (5)].gen);
	}
    break;

  case 44:
#line 211 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 45:
#line 216 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(1) - (1)].gen);
	}
    break;

  case 46:
#line 221 "a.y"
    {
		(yyval.gen2).from = nullgen;
		(yyval.gen2).to = (yyvsp[(2) - (2)].gen);
		(yyval.gen2).to.index = (yyvsp[(2) - (2)].gen).type;
		(yyval.gen2).to.type = D_INDIR+D_ADDR;
	}
    break;

  case 49:
#line 234 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 50:
#line 239 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (5)].gen);
		if((yyval.gen2).from.index != D_NONE)
			yyerror("dp shift with lhs index");
		(yyval.gen2).from.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 51:
#line 249 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 52:
#line 254 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (5)].gen);
		if((yyval.gen2).to.index != D_NONE)
			yyerror("dp move with lhs index");
		(yyval.gen2).to.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 53:
#line 264 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (2)].gen);
		(yyval.gen2).to = nullgen;
	}
    break;

  case 54:
#line 269 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (1)].gen);
		(yyval.gen2).to = nullgen;
	}
    break;

  case 55:
#line 274 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 56:
#line 281 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (3)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (3)].gen);
	}
    break;

  case 57:
#line 286 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.gen2).to = (yyvsp[(5) - (5)].gen);
	}
    break;

  case 58:
#line 294 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(1) - (5)].gen);
		(yyval.gen2).to = (yyvsp[(3) - (5)].gen);
		(yyval.gen2).to.offset = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 59:
#line 302 "a.y"
    {
		(yyval.gen2).from = (yyvsp[(3) - (5)].gen);
		(yyval.gen2).to = (yyvsp[(5) - (5)].gen);
		if((yyvsp[(1) - (5)].gen).type != D_CONST)
			yyerror("illegal constant");
		(yyval.gen2).to.offset = (yyvsp[(1) - (5)].gen).offset;
	}
    break;

  case 64:
#line 318 "a.y"
    {
		(yyval.gen) = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 65:
#line 322 "a.y"
    {
		(yyval.gen) = (yyvsp[(2) - (2)].gen);
	}
    break;

  case 71:
#line 335 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 72:
#line 341 "a.y"
    {
		(yyval.gen) = nullgen;
		if(pass == 2)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->name);
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 73:
#line 350 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 74:
#line 359 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 75:
#line 364 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 76:
#line 369 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 77:
#line 374 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 78:
#line 379 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SP;
	}
    break;

  case 79:
#line 384 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 80:
#line 391 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST;
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 81:
#line 397 "a.y"
    {
		(yyval.gen) = (yyvsp[(2) - (2)].gen);
		(yyval.gen).index = (yyvsp[(2) - (2)].gen).type;
		(yyval.gen).type = D_ADDR;
		/*
		if($2.type == D_AUTO || $2.type == D_PARAM)
			yyerror("constant cannot be automatic: %s",
				$2.sym->name);
		 */
	}
    break;

  case 82:
#line 408 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SCONST;
		memcpy((yyval.gen).sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.gen).sval));
	}
    break;

  case 83:
#line 414 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 84:
#line 420 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = (yyvsp[(3) - (4)].dval);
	}
    break;

  case 85:
#line 426 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = -(yyvsp[(4) - (5)].dval);
	}
    break;

  case 86:
#line 432 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 87:
#line 440 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST2;
		(yyval.gen).offset = (yyvsp[(2) - (2)].con2).v1;
		(yyval.gen).offset2 = (yyvsp[(2) - (2)].con2).v2;
	}
    break;

  case 88:
#line 449 "a.y"
    {
		(yyval.con2).v1 = (yyvsp[(1) - (1)].lval);
		(yyval.con2).v2 = 0;
	}
    break;

  case 89:
#line 454 "a.y"
    {
		(yyval.con2).v1 = -(yyvsp[(2) - (2)].lval);
		(yyval.con2).v2 = 0;
	}
    break;

  case 90:
#line 459 "a.y"
    {
		(yyval.con2).v1 = (yyvsp[(1) - (3)].lval);
		(yyval.con2).v2 = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 91:
#line 464 "a.y"
    {
		(yyval.con2).v1 = -(yyvsp[(2) - (4)].lval);
		(yyval.con2).v2 = (yyvsp[(4) - (4)].lval);
	}
    break;

  case 94:
#line 475 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_NONE;
		(yyval.gen).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 95:
#line 481 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 96:
#line 487 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_SP;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 97:
#line 493 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_NONE;
		(yyval.gen).offset = (yyvsp[(1) - (6)].lval);
		(yyval.gen).index = (yyvsp[(3) - (6)].lval);
		(yyval.gen).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 98:
#line 502 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(3) - (9)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (9)].lval);
		(yyval.gen).index = (yyvsp[(6) - (9)].lval);
		(yyval.gen).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 99:
#line 511 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(3) - (9)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (9)].lval);
		(yyval.gen).index = (yyvsp[(6) - (9)].lval);
		(yyval.gen).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 100:
#line 520 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(2) - (3)].lval);
	}
    break;

  case 101:
#line 525 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_SP;
	}
    break;

  case 102:
#line 530 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 103:
#line 536 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+D_NONE;
		(yyval.gen).index = (yyvsp[(2) - (5)].lval);
		(yyval.gen).scale = (yyvsp[(4) - (5)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 104:
#line 544 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_INDIR+(yyvsp[(2) - (8)].lval);
		(yyval.gen).index = (yyvsp[(5) - (8)].lval);
		(yyval.gen).scale = (yyvsp[(7) - (8)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 105:
#line 554 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (1)].gen);
	}
    break;

  case 106:
#line 558 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (6)].gen);
		(yyval.gen).index = (yyvsp[(3) - (6)].lval);
		(yyval.gen).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.gen).scale);
	}
    break;

  case 107:
#line 567 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = (yyvsp[(4) - (5)].lval);
		(yyval.gen).sym = (yyvsp[(1) - (5)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 108:
#line 574 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_STATIC;
		(yyval.gen).sym = (yyvsp[(1) - (7)].sym);
		(yyval.gen).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 109:
#line 582 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 110:
#line 586 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 111:
#line 590 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 113:
#line 597 "a.y"
    {
		(yyval.lval) = D_AUTO;
	}
    break;

  case 116:
#line 605 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 117:
#line 609 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 118:
#line 613 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 119:
#line 617 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 120:
#line 621 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 122:
#line 628 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 123:
#line 632 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 124:
#line 636 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:
#line 640 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:
#line 644 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 127:
#line 648 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 128:
#line 652 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 129:
#line 656 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 130:
#line 660 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 131:
#line 664 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2521 "y.tab.c"
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



