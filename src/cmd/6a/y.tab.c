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
     LTYPEG = 268,
     LTYPEPC = 269,
     LTYPES = 270,
     LTYPEM = 271,
     LTYPEI = 272,
     LTYPEXC = 273,
     LTYPEX = 274,
     LTYPERT = 275,
     LTYPEF = 276,
     LCONST = 277,
     LFP = 278,
     LPC = 279,
     LSB = 280,
     LBREG = 281,
     LLREG = 282,
     LSREG = 283,
     LFREG = 284,
     LMREG = 285,
     LXREG = 286,
     LFCONST = 287,
     LSCONST = 288,
     LSP = 289,
     LNAME = 290,
     LLAB = 291,
     LVAR = 292
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
#define LTYPEG 268
#define LTYPEPC 269
#define LTYPES 270
#define LTYPEM 271
#define LTYPEI 272
#define LTYPEXC 273
#define LTYPEX 274
#define LTYPERT 275
#define LTYPEF 276
#define LCONST 277
#define LFP 278
#define LPC 279
#define LSB 280
#define LBREG 281
#define LLREG 282
#define LSREG 283
#define LFREG 284
#define LMREG 285
#define LXREG 286
#define LFCONST 287
#define LSCONST 288
#define LSP 289
#define LNAME 290
#define LLAB 291
#define LVAR 292




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
	vlong	lval;
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
#define YYLAST   524

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  56
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  40
/* YYNRULES -- Number of rules.  */
#define YYNRULES  133
/* YYNRULES -- Number of states.  */
#define YYNSTATES  271

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   292

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    52,    12,     5,     2,
      53,    54,    10,     8,    51,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    48,    49,
       6,    50,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    55,     2,     2,     2,
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
      45,    46,    47
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     5,     9,    10,    15,    17,    20,
      23,    27,    31,    34,    37,    40,    43,    46,    49,    51,
      53,    56,    59,    62,    65,    68,    71,    74,    77,    79,
      82,    85,    86,    88,    92,    96,    99,   101,   104,   106,
     109,   111,   115,   122,   128,   136,   141,   148,   151,   153,
     155,   157,   161,   167,   171,   177,   180,   182,   186,   192,
     198,   199,   201,   205,   209,   211,   213,   215,   217,   220,
     223,   225,   227,   229,   231,   236,   239,   241,   243,   245,
     247,   249,   251,   253,   256,   259,   262,   265,   270,   276,
     280,   282,   284,   286,   291,   296,   301,   308,   318,   328,
     332,   336,   342,   351,   353,   360,   366,   374,   375,   378,
     381,   383,   385,   387,   389,   391,   394,   397,   400,   404,
     406,   409,   413,   418,   420,   424,   428,   432,   436,   440,
     445,   450,   454,   458
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      57,     0,    -1,    -1,    -1,    57,    58,    59,    -1,    -1,
      45,    48,    60,    59,    -1,    49,    -1,    61,    49,    -1,
       1,    49,    -1,    45,    50,    95,    -1,    47,    50,    95,
      -1,    13,    62,    -1,    14,    66,    -1,    15,    65,    -1,
      16,    63,    -1,    17,    64,    -1,    21,    67,    -1,    68,
      -1,    69,    -1,    18,    71,    -1,    20,    72,    -1,    25,
      73,    -1,    26,    74,    -1,    27,    75,    -1,    28,    76,
      -1,    29,    77,    -1,    30,    78,    -1,    70,    -1,    24,
      79,    -1,    31,    80,    -1,    -1,    51,    -1,    83,    51,
      81,    -1,    81,    51,    83,    -1,    83,    51,    -1,    83,
      -1,    51,    81,    -1,    81,    -1,    51,    84,    -1,    84,
      -1,    86,    51,    84,    -1,    19,    90,    11,    93,    51,
      86,    -1,    22,    87,    51,    52,    94,    -1,    22,    87,
      51,    93,    51,    52,    94,    -1,    23,    87,    51,    86,
      -1,    23,    87,    51,    93,    51,    86,    -1,    51,    82,
      -1,    82,    -1,    62,    -1,    66,    -1,    83,    51,    81,
      -1,    83,    51,    81,    48,    37,    -1,    83,    51,    81,
      -1,    83,    51,    81,    48,    38,    -1,    83,    51,    -1,
      83,    -1,    83,    51,    81,    -1,    85,    51,    81,    51,
      93,    -1,    86,    51,    81,    51,    85,    -1,    -1,    86,
      -1,    83,    51,    83,    -1,    83,    51,    83,    -1,    85,
      -1,    87,    -1,    84,    -1,    89,    -1,    10,    85,    -1,
      10,    88,    -1,    85,    -1,    88,    -1,    81,    -1,    86,
      -1,    93,    53,    34,    54,    -1,    45,    91,    -1,    36,
      -1,    39,    -1,    37,    -1,    40,    -1,    44,    -1,    38,
      -1,    41,    -1,    52,    93,    -1,    52,    90,    -1,    52,
      43,    -1,    52,    42,    -1,    52,    53,    42,    54,    -1,
      52,    53,     9,    42,    54,    -1,    52,     9,    42,    -1,
      88,    -1,    89,    -1,    93,    -1,    93,    53,    37,    54,
      -1,    93,    53,    44,    54,    -1,    93,    53,    38,    54,
      -1,    93,    53,    37,    10,    93,    54,    -1,    93,    53,
      37,    54,    53,    37,    10,    93,    54,    -1,    93,    53,
      37,    54,    53,    38,    10,    93,    54,    -1,    53,    37,
      54,    -1,    53,    44,    54,    -1,    53,    37,    10,    93,
      54,    -1,    53,    37,    54,    53,    37,    10,    93,    54,
      -1,    90,    -1,    90,    53,    37,    10,    93,    54,    -1,
      45,    91,    53,    92,    54,    -1,    45,     6,     7,    91,
      53,    35,    54,    -1,    -1,     8,    93,    -1,     9,    93,
      -1,    35,    -1,    44,    -1,    33,    -1,    32,    -1,    47,
      -1,     9,    93,    -1,     8,    93,    -1,    55,    93,    -1,
      53,    95,    54,    -1,    32,    -1,     9,    32,    -1,    32,
       9,    32,    -1,     9,    32,     9,    32,    -1,    93,    -1,
      95,     8,    95,    -1,    95,     9,    95,    -1,    95,    10,
      95,    -1,    95,    11,    95,    -1,    95,    12,    95,    -1,
      95,     6,     6,    95,    -1,    95,     7,     7,    95,    -1,
      95,     5,    95,    -1,    95,     4,    95,    -1,    95,     3,
      95,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    66,    66,    68,    67,    75,    74,    83,    84,    85,
      88,    93,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   120,   124,   131,   138,   145,   150,   157,   162,   169,
     174,   179,   186,   199,   207,   221,   229,   243,   248,   255,
     256,   259,   264,   274,   279,   289,   294,   299,   306,   314,
     324,   328,   335,   344,   355,   356,   359,   360,   361,   365,
     369,   370,   373,   374,   377,   383,   394,   400,   406,   412,
     418,   424,   430,   438,   444,   454,   460,   466,   472,   478,
     486,   487,   490,   496,   503,   510,   517,   526,   536,   546,
     552,   558,   566,   577,   581,   590,   598,   608,   611,   615,
     621,   622,   626,   629,   630,   634,   638,   642,   646,   652,
     659,   666,   673,   682,   683,   687,   691,   695,   699,   703,
     707,   711,   715,   719
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'|'", "'^'", "'&'", "'<'", "'>'", "'+'",
  "'-'", "'*'", "'/'", "'%'", "LTYPE0", "LTYPE1", "LTYPE2", "LTYPE3",
  "LTYPE4", "LTYPEC", "LTYPED", "LTYPEN", "LTYPER", "LTYPET", "LTYPEG",
  "LTYPEPC", "LTYPES", "LTYPEM", "LTYPEI", "LTYPEXC", "LTYPEX", "LTYPERT",
  "LTYPEF", "LCONST", "LFP", "LPC", "LSB", "LBREG", "LLREG", "LSREG",
  "LFREG", "LMREG", "LXREG", "LFCONST", "LSCONST", "LSP", "LNAME", "LLAB",
  "LVAR", "':'", "';'", "'='", "','", "'$'", "'('", "')'", "'~'",
  "$accept", "prog", "@1", "line", "@2", "inst", "nonnon", "rimrem",
  "remrim", "rimnon", "nonrem", "nonrel", "spec1", "spec2", "spec11",
  "spec3", "spec4", "spec5", "spec6", "spec7", "spec8", "spec9", "spec10",
  "spec12", "spec13", "rem", "rom", "rim", "rel", "reg", "imm", "mem",
  "omem", "nmem", "nam", "offset", "pointer", "con", "textsize", "expr", 0
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
     285,   286,   287,   288,   289,   290,   291,   292,    58,    59,
      61,    44,    36,    40,    41,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    56,    57,    58,    57,    60,    59,    59,    59,    59,
      61,    61,    61,    61,    61,    61,    61,    61,    61,    61,
      61,    61,    61,    61,    61,    61,    61,    61,    61,    61,
      61,    62,    62,    63,    64,    65,    65,    66,    66,    67,
      67,    67,    68,    69,    69,    70,    70,    71,    71,    72,
      72,    73,    73,    74,    74,    75,    75,    75,    76,    77,
      78,    78,    79,    80,    81,    81,    82,    82,    82,    82,
      82,    82,    83,    83,    84,    84,    85,    85,    85,    85,
      85,    85,    85,    86,    86,    86,    86,    86,    86,    86,
      87,    87,    88,    88,    88,    88,    88,    88,    88,    88,
      88,    88,    88,    89,    89,    90,    90,    91,    91,    91,
      92,    92,    92,    93,    93,    93,    93,    93,    93,    94,
      94,    94,    94,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     1,     2,     2,
       3,     3,     2,     2,     2,     2,     2,     2,     1,     1,
       2,     2,     2,     2,     2,     2,     2,     2,     1,     2,
       2,     0,     1,     3,     3,     2,     1,     2,     1,     2,
       1,     3,     6,     5,     7,     4,     6,     2,     1,     1,
       1,     3,     5,     3,     5,     2,     1,     3,     5,     5,
       0,     1,     3,     3,     1,     1,     1,     1,     2,     2,
       1,     1,     1,     1,     4,     2,     1,     1,     1,     1,
       1,     1,     1,     2,     2,     2,     2,     4,     5,     3,
       1,     1,     1,     4,     4,     4,     6,     9,     9,     3,
       3,     5,     8,     1,     6,     5,     7,     0,     2,     2,
       1,     1,     1,     1,     1,     2,     2,     2,     3,     1,
       2,     3,     4,     1,     3,     3,     3,     3,     3,     4,
       4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    31,     0,     0,     0,     0,
       0,     0,    31,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    60,     0,     0,     0,     7,     4,     0,    18,
      19,    28,     9,    32,    12,     0,     0,   113,    76,    78,
      81,    77,    79,    82,    80,   107,   114,     0,     0,     0,
      13,    38,    64,    65,    90,    91,   103,    92,     0,    14,
      72,    36,    73,    15,     0,    16,     0,     0,   107,     0,
      20,    48,    66,    70,    71,    67,    92,     0,    32,    49,
      50,    21,   107,     0,     0,    17,    40,     0,     0,     0,
       0,    29,     0,    22,     0,    23,     0,    24,    56,    25,
       0,    26,     0,    27,    61,    30,     0,     5,     0,     0,
       8,   116,   115,     0,     0,     0,     0,    37,     0,     0,
     123,     0,   117,     0,     0,     0,    86,    85,     0,    84,
      83,    35,     0,     0,    68,    69,    75,    47,     0,     0,
      75,    39,     0,     0,     0,     0,     0,     0,     0,    55,
       0,     0,     0,     0,    10,    11,   107,   108,   109,     0,
       0,    99,   100,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   118,     0,     0,     0,     0,    89,     0,
       0,    33,    34,     0,     0,    41,     0,     0,    45,     0,
      62,    51,    53,    57,     0,     0,    63,     6,     0,   112,
     110,   111,     0,     0,     0,   133,   132,   131,     0,     0,
     124,   125,   126,   127,   128,     0,     0,    93,    95,    94,
       0,    87,    74,     0,     0,   119,    43,     0,     0,     0,
       0,     0,     0,     0,   105,   101,     0,   129,   130,     0,
       0,     0,    88,    42,   120,     0,     0,    46,    52,    54,
      58,    59,     0,     0,   104,    96,     0,     0,     0,   121,
      44,   106,     0,     0,     0,   122,   102,     0,     0,    97,
      98
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    27,   153,    28,    34,    63,    65,    59,
      50,    85,    29,    30,    31,    70,    81,    93,    95,    97,
      99,   101,   103,    91,   105,    60,    71,    61,    72,    52,
      62,    53,    54,    55,    56,   116,   202,    57,   226,   121
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -87
static const yytype_int16 yypact[] =
{
     -87,    24,   -87,   211,    20,    -5,   236,   256,   256,   330,
     156,    25,   290,    55,   364,   364,   256,   256,   256,   256,
     145,    29,    29,   256,    17,    46,   -87,   -87,    26,   -87,
     -87,   -87,   -87,   -87,   -87,   451,   451,   -87,   -87,   -87,
     -87,   -87,   -87,   -87,   -87,    27,   -87,   330,   270,   451,
     -87,   -87,   -87,   -87,   -87,   -87,    39,    44,    48,   -87,
     -87,    65,   -87,   -87,    66,   -87,    68,   350,    27,   310,
     -87,   -87,   -87,   -87,   -87,   -87,    71,   110,   330,   -87,
     -87,   -87,    23,   384,   451,   -87,   -87,    75,    72,    77,
      82,   -87,    85,   -87,    87,   -87,    88,   -87,    89,   -87,
      90,   -87,    91,   -87,   -87,   -87,    92,   -87,   451,   451,
     -87,   -87,   -87,   120,   451,   451,    98,   -87,     7,   113,
     -87,   168,   -87,   115,     5,   391,   -87,   -87,   398,   -87,
     -87,   -87,   330,   256,   -87,   -87,    98,   -87,     3,   451,
     -87,   -87,   384,   122,   416,   426,   256,   330,   330,   330,
     330,   330,   256,   211,   504,   504,    23,   -87,   -87,    76,
     451,   117,   -87,   451,   451,   451,   162,   180,   451,   451,
     451,   451,   451,   -87,   181,     8,   136,   148,   -87,   433,
     150,   -87,   -87,   154,   159,   -87,    12,   163,   -87,   165,
     -87,   169,   170,   -87,   204,   206,   -87,   -87,   160,   -87,
     -87,   -87,   205,   207,   182,   485,   512,   240,   451,   451,
     102,   102,   -87,   -87,   -87,   451,   451,   209,   -87,   -87,
     212,   -87,   -87,    29,   231,   258,   -87,   217,    29,   233,
     244,   451,   145,   249,   -87,   -87,   261,    42,    42,   232,
     250,   -22,   -87,   -87,   276,   273,    12,   -87,   -87,   -87,
     -87,   -87,   252,   451,   -87,   -87,   280,   300,   281,   -87,
     -87,   -87,   262,   451,   451,   -87,   -87,   267,   278,   -87,
     -87
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -87,   -87,   -87,   171,   -87,   -87,   303,   -87,   -87,   -87,
     321,   -87,   -87,   -87,   -87,   -87,   -87,   -87,   -87,   -87,
     -87,   -87,   -87,   -87,   -87,    -2,   243,    11,   -11,    -9,
      -8,    74,    -1,     2,    -3,   -62,   -87,   -10,    94,   -86
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      76,    73,    86,    88,    51,    87,   136,    66,    77,    74,
      51,   100,    75,   102,   104,   256,   257,   160,   216,    64,
     140,   224,   154,   155,     2,   111,   112,    92,    94,    96,
      98,   114,   115,   113,   106,   114,   115,   183,   120,   122,
     175,   176,   175,   176,   225,   117,    33,   177,   130,   177,
     168,   169,   170,   171,   172,   129,    35,   125,   134,    76,
      73,   161,   217,    35,    36,   107,   135,   108,    74,    32,
      45,    75,   141,    88,   120,   110,   117,   205,   206,   207,
      37,    58,   210,   211,   212,   213,   214,    37,    89,    90,
     126,   127,   123,    45,   198,    46,   109,   124,   120,   120,
      82,   128,    46,    49,   157,   158,    83,    58,    84,   199,
      49,   200,   170,   171,   172,   112,   131,   132,   120,   133,
     201,   139,   237,   238,   138,   143,   142,   156,   144,   184,
     181,   185,    88,   145,   187,   189,   146,   188,   147,   148,
     149,   150,   151,   152,   182,   191,   192,   193,   194,   195,
     203,   159,   174,   120,   120,   120,   183,   190,   120,   120,
     120,   120,   120,   196,    35,    36,    67,   162,   208,   112,
     204,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,    38,    39,    40,    41,    42,    43,   209,    37,    44,
     218,   215,    38,    39,    40,    41,    42,    43,   120,   120,
      44,    68,   219,    46,   221,   239,   240,    69,   222,    48,
     223,    49,     4,   233,   227,   243,   228,   229,   230,   236,
     247,   250,   173,   251,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,   262,    35,    36,   166,   167,   168,   169,
     170,   171,   172,   267,   268,   231,    24,   232,    25,   234,
      26,   235,   241,   244,    35,    36,   242,   245,    37,   246,
     248,   253,    38,    39,    40,    41,    42,    43,    35,    36,
      44,    45,   249,    46,   252,   258,   254,    47,    37,    48,
     263,    49,    38,    39,    40,    41,    42,    43,    35,    36,
      44,    45,    37,    46,   255,   259,   261,   118,    58,    48,
     264,    49,   137,   265,   119,    79,   266,    46,    35,    36,
      67,   269,    37,    84,   197,    49,    38,    39,    40,    41,
      42,    43,   270,    80,    44,    45,     0,    46,    35,    36,
     260,    78,    37,    48,     0,    49,    38,    39,    40,    41,
      42,    43,     0,     0,    44,    68,     0,    46,    35,    36,
       0,     0,    37,    48,     0,    49,    38,    39,    40,    41,
      42,    43,    35,    36,    44,    45,     0,    46,     0,     0,
       0,     0,    37,    48,     0,    49,    38,    39,    40,    41,
      42,    43,    35,    36,    44,     0,    37,    46,     0,    35,
      36,     0,     0,    48,     0,    49,    35,   179,     0,    45,
       0,    46,     0,     0,     0,     0,    37,    48,     0,    49,
       0,     0,     0,    37,    35,    36,     0,     0,     0,    82,
      37,    46,     0,   178,    35,    36,     0,    84,    46,    49,
     180,    35,    36,     0,    84,    46,    49,     0,    37,     0,
       0,    84,     0,    49,     0,     0,     0,     0,    37,    35,
      36,     0,     0,    46,     0,    37,     0,     0,   186,    84,
       0,    49,     0,    46,     0,   220,     0,     0,    58,    84,
      46,    49,     0,    37,     0,     0,    84,     0,    49,   164,
     165,   166,   167,   168,   169,   170,   171,   172,    46,     0,
       0,     0,     0,     0,    84,     0,    49,   163,   164,   165,
     166,   167,   168,   169,   170,   171,   172,   165,   166,   167,
     168,   169,   170,   171,   172
};

static const yytype_int16 yycheck[] =
{
      10,    10,    13,    13,     6,    13,    68,     9,    11,    10,
      12,    20,    10,    21,    22,    37,    38,    10,    10,     8,
      82,     9,   108,   109,     0,    35,    36,    16,    17,    18,
      19,     8,     9,     6,    23,     8,     9,    34,    48,    49,
      37,    38,    37,    38,    32,    47,    51,    44,    58,    44,
       8,     9,    10,    11,    12,    58,     8,     9,    67,    69,
      69,    54,    54,     8,     9,    48,    67,    50,    69,    49,
      45,    69,    83,    83,    84,    49,    78,   163,   164,   165,
      32,    52,   168,   169,   170,   171,   172,    32,    14,    15,
      42,    43,    53,    45,   156,    47,    50,    53,   108,   109,
      45,    53,    47,    55,   114,   115,    51,    52,    53,    33,
      55,    35,    10,    11,    12,   125,    51,    51,   128,    51,
      44,    11,   208,   209,    53,    53,    51,     7,    51,   139,
     132,   142,   142,    51,   144,   145,    51,   145,    51,    51,
      51,    51,    51,    51,   133,   147,   148,   149,   150,   151,
     160,    53,    37,   163,   164,   165,    34,   146,   168,   169,
     170,   171,   172,   152,     8,     9,    10,    54,     6,   179,
      53,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    36,    37,    38,    39,    40,    41,     7,    32,    44,
      54,    10,    36,    37,    38,    39,    40,    41,   208,   209,
      44,    45,    54,    47,    54,   215,   216,    51,    54,    53,
      51,    55,     1,    53,    51,   223,    51,    48,    48,    37,
     228,   231,    54,   232,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,   253,     8,     9,     6,     7,     8,     9,
      10,    11,    12,   263,   264,    51,    45,    51,    47,    54,
      49,    54,    53,    32,     8,     9,    54,     9,    32,    52,
      37,    10,    36,    37,    38,    39,    40,    41,     8,     9,
      44,    45,    38,    47,    35,     9,    54,    51,    32,    53,
      10,    55,    36,    37,    38,    39,    40,    41,     8,     9,
      44,    45,    32,    47,    54,    32,    54,    37,    52,    53,
      10,    55,    69,    32,    44,    12,    54,    47,     8,     9,
      10,    54,    32,    53,   153,    55,    36,    37,    38,    39,
      40,    41,    54,    12,    44,    45,    -1,    47,     8,     9,
     246,    51,    32,    53,    -1,    55,    36,    37,    38,    39,
      40,    41,    -1,    -1,    44,    45,    -1,    47,     8,     9,
      -1,    -1,    32,    53,    -1,    55,    36,    37,    38,    39,
      40,    41,     8,     9,    44,    45,    -1,    47,    -1,    -1,
      -1,    -1,    32,    53,    -1,    55,    36,    37,    38,    39,
      40,    41,     8,     9,    44,    -1,    32,    47,    -1,     8,
       9,    -1,    -1,    53,    -1,    55,     8,     9,    -1,    45,
      -1,    47,    -1,    -1,    -1,    -1,    32,    53,    -1,    55,
      -1,    -1,    -1,    32,     8,     9,    -1,    -1,    -1,    45,
      32,    47,    -1,    42,     8,     9,    -1,    53,    47,    55,
      42,     8,     9,    -1,    53,    47,    55,    -1,    32,    -1,
      -1,    53,    -1,    55,    -1,    -1,    -1,    -1,    32,     8,
       9,    -1,    -1,    47,    -1,    32,    -1,    -1,    52,    53,
      -1,    55,    -1,    47,    -1,    42,    -1,    -1,    52,    53,
      47,    55,    -1,    32,    -1,    -1,    53,    -1,    55,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    47,    -1,
      -1,    -1,    -1,    -1,    53,    -1,    55,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,     5,     6,     7,
       8,     9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    57,     0,    58,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    45,    47,    49,    59,    61,    68,
      69,    70,    49,    51,    62,     8,     9,    32,    36,    37,
      38,    39,    40,    41,    44,    45,    47,    51,    53,    55,
      66,    81,    85,    87,    88,    89,    90,    93,    52,    65,
      81,    83,    86,    63,    83,    64,    81,    10,    45,    51,
      71,    82,    84,    85,    88,    89,    93,    90,    51,    62,
      66,    72,    45,    51,    53,    67,    84,    86,    93,    87,
      87,    79,    83,    73,    83,    74,    83,    75,    83,    76,
      85,    77,    86,    78,    86,    80,    83,    48,    50,    50,
      49,    93,    93,     6,     8,     9,    91,    81,    37,    44,
      93,    95,    93,    53,    53,     9,    42,    43,    53,    90,
      93,    51,    51,    51,    85,    88,    91,    82,    53,    11,
      91,    84,    51,    53,    51,    51,    51,    51,    51,    51,
      51,    51,    51,    60,    95,    95,     7,    93,    93,    53,
      10,    54,    54,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    54,    37,    37,    38,    44,    42,     9,
      42,    81,    83,    34,    93,    84,    52,    93,    86,    93,
      83,    81,    81,    81,    81,    81,    83,    59,    91,    33,
      35,    44,    92,    93,    53,    95,    95,    95,     6,     7,
      95,    95,    95,    95,    95,    10,    10,    54,    54,    54,
      42,    54,    54,    51,     9,    32,    94,    51,    51,    48,
      48,    51,    51,    53,    54,    54,    37,    95,    95,    93,
      93,    53,    54,    86,    32,     9,    52,    86,    37,    38,
      93,    85,    35,    10,    54,    54,    37,    38,     9,    32,
      94,    54,    93,    10,    10,    32,    54,    93,    93,    54,
      54
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
#line 68 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:
#line 75 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		if((yyvsp[(1) - (2)].sym)->type == LLAB && (yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s (%s)", (yyvsp[(1) - (2)].sym)->labelname, (yyvsp[(1) - (2)].sym)->name);
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 10:
#line 89 "a.y"
    {
		(yyvsp[(1) - (3)].sym)->type = LVAR;
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 11:
#line 94 "a.y"
    {
		if((yyvsp[(1) - (3)].sym)->value != (yyvsp[(3) - (3)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (3)].sym)->name);
		(yyvsp[(1) - (3)].sym)->value = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 12:
#line 99 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 13:
#line 100 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 14:
#line 101 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 15:
#line 102 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 16:
#line 103 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 17:
#line 104 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 20:
#line 107 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 21:
#line 108 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 22:
#line 109 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 23:
#line 110 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 24:
#line 111 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 25:
#line 112 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 26:
#line 113 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 27:
#line 114 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 29:
#line 116 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 30:
#line 117 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 31:
#line 120 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 32:
#line 125 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 33:
#line 132 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 34:
#line 139 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 35:
#line 146 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (2)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 36:
#line 151 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 37:
#line 158 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 38:
#line 163 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 39:
#line 170 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 40:
#line 175 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 41:
#line 180 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 42:
#line 187 "a.y"
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

  case 43:
#line 200 "a.y"
    {
		Addr2 a;
		settext((yyvsp[(2) - (5)].addr).sym);
		a.from = (yyvsp[(2) - (5)].addr);
		a.to = (yyvsp[(5) - (5)].addr);
		outcode(ATEXT, &a);
	}
    break;

  case 44:
#line 208 "a.y"
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

  case 45:
#line 222 "a.y"
    {
		Addr2 a;
		settext((yyvsp[(2) - (4)].addr).sym);
		a.from = (yyvsp[(2) - (4)].addr);
		a.to = (yyvsp[(4) - (4)].addr);
		outcode(AGLOBL, &a);
	}
    break;

  case 46:
#line 230 "a.y"
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

  case 47:
#line 244 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 48:
#line 249 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 51:
#line 260 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 52:
#line 265 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).from.index != TYPE_NONE)
			yyerror("dp shift with lhs index");
		(yyval.addr2).from.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 53:
#line 275 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 54:
#line 280 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).to.index != TYPE_NONE)
			yyerror("dp move with lhs index");
		(yyval.addr2).to.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 55:
#line 290 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (2)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 56:
#line 295 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 57:
#line 300 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 58:
#line 307 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to.offset = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 59:
#line 315 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
		if((yyvsp[(1) - (5)].addr).type != TYPE_CONST)
			yyerror("illegal constant");
		(yyval.addr2).to.offset = (yyvsp[(1) - (5)].addr).offset;
	}
    break;

  case 60:
#line 324 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 61:
#line 329 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 62:
#line 336 "a.y"
    {
		if((yyvsp[(1) - (3)].addr).type != TYPE_CONST || (yyvsp[(3) - (3)].addr).type != TYPE_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 63:
#line 345 "a.y"
    {
		if((yyvsp[(1) - (3)].addr).type != TYPE_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(3) - (3)].addr).type != TYPE_MEM || ((yyvsp[(3) - (3)].addr).name != NAME_EXTERN && (yyvsp[(3) - (3)].addr).name != NAME_STATIC))
			yyerror("value for FUNCDATA must be symbol reference");
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 68:
#line 362 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 69:
#line 366 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 74:
#line 378 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 75:
#line 384 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		(yyval.addr) = nullgen;
		if(pass == 2 && (yyvsp[(1) - (2)].sym)->type != LLAB)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyval.addr).type = TYPE_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 76:
#line 395 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 77:
#line 401 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 78:
#line 407 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 79:
#line 413 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 80:
#line 419 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = REG_SP;
	}
    break;

  case 81:
#line 425 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 82:
#line 431 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 83:
#line 439 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 84:
#line 445 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = TYPE_ADDR;
		/*
		if($2.type == D_AUTO || $2.type == D_PARAM)
			yyerror("constant cannot be automatic: %s",
				$2.sym->name);
		 */
	}
    break;

  case 85:
#line 455 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 86:
#line 461 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 87:
#line 467 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = (yyvsp[(3) - (4)].dval);
	}
    break;

  case 88:
#line 473 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(4) - (5)].dval);
	}
    break;

  case 89:
#line 479 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 92:
#line 491 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 93:
#line 497 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 94:
#line 504 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = REG_SP;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 95:
#line 511 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 96:
#line 518 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).offset = (yyvsp[(1) - (6)].lval);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 97:
#line 527 "a.y"
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

  case 98:
#line 537 "a.y"
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
#line 547 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 100:
#line 553 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = REG_SP;
	}
    break;

  case 101:
#line 559 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).index = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 102:
#line 567 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).reg = (yyvsp[(2) - (8)].lval);
		(yyval.addr).index = (yyvsp[(5) - (8)].lval);
		(yyval.addr).scale = (yyvsp[(7) - (8)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 103:
#line 578 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 104:
#line 582 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (6)].addr);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 105:
#line 591 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 106:
#line 599 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_MEM;
		(yyval.addr).name = NAME_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 1);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 107:
#line 608 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 108:
#line 612 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 109:
#line 616 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 111:
#line 623 "a.y"
    {
		(yyval.lval) = NAME_AUTO;
	}
    break;

  case 114:
#line 631 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 115:
#line 635 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 116:
#line 639 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 117:
#line 643 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 118:
#line 647 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 119:
#line 653 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 120:
#line 660 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (2)].lval);
		(yyval.addr).u.argsize = ArgsSizeUnknown;
	}
    break;

  case 121:
#line 667 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = (yyvsp[(1) - (3)].lval);
		(yyval.addr).u.argsize = (yyvsp[(3) - (3)].lval);
	}
    break;

  case 122:
#line 674 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = TYPE_TEXTSIZE;
		(yyval.addr).offset = -(yyvsp[(2) - (4)].lval);
		(yyval.addr).u.argsize = (yyvsp[(4) - (4)].lval);
	}
    break;

  case 124:
#line 684 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:
#line 688 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:
#line 692 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 127:
#line 696 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:
#line 700 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:
#line 704 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 130:
#line 708 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 131:
#line 712 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 132:
#line 716 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 133:
#line 720 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2587 "y.tab.c"
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



