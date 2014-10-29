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
#define YYLAST   549

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  56
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  41
/* YYNRULES -- Number of rules.  */
#define YYNRULES  134
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
       2,     2,     2,     2,     2,     2,    54,    12,     5,     2,
      52,    53,    10,     8,    51,     9,     2,    11,     2,     2,
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
      23,    27,    31,    34,    37,    40,    43,    46,    49,    52,
      55,    58,    61,    64,    67,    70,    73,    76,    79,    82,
      85,    88,    89,    91,    95,    99,   102,   104,   107,   109,
     112,   114,   118,   124,   128,   134,   137,   139,   141,   143,
     147,   153,   157,   163,   166,   168,   172,   178,   184,   185,
     187,   191,   197,   201,   205,   207,   209,   211,   213,   216,
     219,   221,   223,   225,   227,   232,   235,   237,   239,   241,
     243,   245,   247,   249,   252,   255,   258,   261,   264,   269,
     275,   279,   281,   283,   285,   290,   295,   300,   307,   317,
     327,   331,   335,   341,   350,   352,   359,   365,   373,   374,
     377,   380,   382,   384,   386,   388,   390,   393,   396,   399,
     403,   405,   408,   412,   417,   419,   423,   427,   431,   435,
     439,   444,   449,   453,   457
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      57,     0,    -1,    -1,    -1,    57,    58,    59,    -1,    -1,
      45,    48,    60,    59,    -1,    49,    -1,    61,    49,    -1,
       1,    49,    -1,    45,    50,    96,    -1,    47,    50,    96,
      -1,    13,    62,    -1,    14,    66,    -1,    15,    65,    -1,
      16,    63,    -1,    17,    64,    -1,    21,    67,    -1,    19,
      68,    -1,    22,    69,    -1,    18,    70,    -1,    20,    71,
      -1,    25,    72,    -1,    26,    73,    -1,    27,    74,    -1,
      28,    75,    -1,    29,    76,    -1,    30,    77,    -1,    23,
      78,    -1,    24,    79,    -1,    31,    80,    -1,    -1,    51,
      -1,    83,    51,    81,    -1,    81,    51,    83,    -1,    83,
      51,    -1,    83,    -1,    51,    81,    -1,    81,    -1,    51,
      84,    -1,    84,    -1,    87,    51,    84,    -1,    91,    11,
      94,    51,    87,    -1,    88,    51,    86,    -1,    88,    51,
      94,    51,    86,    -1,    51,    82,    -1,    82,    -1,    62,
      -1,    66,    -1,    83,    51,    81,    -1,    83,    51,    81,
      48,    37,    -1,    83,    51,    81,    -1,    83,    51,    81,
      48,    38,    -1,    83,    51,    -1,    83,    -1,    83,    51,
      81,    -1,    85,    51,    81,    51,    94,    -1,    87,    51,
      81,    51,    85,    -1,    -1,    87,    -1,    88,    51,    87,
      -1,    88,    51,    94,    51,    87,    -1,    83,    51,    83,
      -1,    83,    51,    83,    -1,    85,    -1,    88,    -1,    84,
      -1,    90,    -1,    10,    85,    -1,    10,    89,    -1,    85,
      -1,    89,    -1,    81,    -1,    87,    -1,    94,    52,    34,
      53,    -1,    45,    92,    -1,    36,    -1,    39,    -1,    37,
      -1,    40,    -1,    44,    -1,    38,    -1,    41,    -1,    54,
      95,    -1,    54,    94,    -1,    54,    91,    -1,    54,    43,
      -1,    54,    42,    -1,    54,    52,    42,    53,    -1,    54,
      52,     9,    42,    53,    -1,    54,     9,    42,    -1,    89,
      -1,    90,    -1,    94,    -1,    94,    52,    37,    53,    -1,
      94,    52,    44,    53,    -1,    94,    52,    38,    53,    -1,
      94,    52,    37,    10,    94,    53,    -1,    94,    52,    37,
      53,    52,    37,    10,    94,    53,    -1,    94,    52,    37,
      53,    52,    38,    10,    94,    53,    -1,    52,    37,    53,
      -1,    52,    44,    53,    -1,    52,    37,    10,    94,    53,
      -1,    52,    37,    53,    52,    37,    10,    94,    53,    -1,
      91,    -1,    91,    52,    37,    10,    94,    53,    -1,    45,
      92,    52,    93,    53,    -1,    45,     6,     7,    92,    52,
      35,    53,    -1,    -1,     8,    94,    -1,     9,    94,    -1,
      35,    -1,    44,    -1,    33,    -1,    32,    -1,    47,    -1,
       9,    94,    -1,     8,    94,    -1,    55,    94,    -1,    52,
      96,    53,    -1,    32,    -1,     9,    32,    -1,    32,     9,
      32,    -1,     9,    32,     9,    32,    -1,    94,    -1,    96,
       8,    96,    -1,    96,     9,    96,    -1,    96,    10,    96,
      -1,    96,    11,    96,    -1,    96,    12,    96,    -1,    96,
       6,     6,    96,    -1,    96,     7,     7,    96,    -1,    96,
       5,    96,    -1,    96,     4,    96,    -1,    96,     3,    96,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    66,    66,    68,    67,    75,    74,    83,    84,    85,
      88,    93,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   120,   124,   131,   138,   145,   150,   157,   162,   169,
     174,   179,   186,   194,   200,   209,   214,   221,   222,   225,
     230,   240,   245,   255,   260,   265,   272,   280,   290,   294,
     301,   306,   314,   323,   334,   335,   338,   339,   340,   344,
     348,   349,   352,   353,   356,   362,   373,   378,   383,   388,
     393,   398,   403,   409,   417,   423,   434,   440,   446,   452,
     458,   466,   467,   470,   476,   482,   488,   494,   503,   512,
     521,   526,   531,   539,   549,   553,   562,   569,   578,   581,
     585,   591,   592,   596,   599,   600,   604,   608,   612,   616,
     622,   627,   632,   637,   644,   645,   649,   653,   657,   661,
     665,   669,   673,   677,   681
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
  "LVAR", "':'", "';'", "'='", "','", "'('", "')'", "'$'", "'~'",
  "$accept", "prog", "@1", "line", "@2", "inst", "nonnon", "rimrem",
  "remrim", "rimnon", "nonrem", "nonrel", "spec1", "spec2", "spec3",
  "spec4", "spec5", "spec6", "spec7", "spec8", "spec9", "spec10", "spec11",
  "spec12", "spec13", "rem", "rom", "rim", "rel", "reg", "imm2", "imm",
  "mem", "omem", "nmem", "nam", "offset", "pointer", "con", "con2", "expr", 0
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
      61,    44,    40,    41,    36,   126
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
      72,    73,    73,    74,    74,    74,    75,    76,    77,    77,
      78,    78,    79,    80,    81,    81,    82,    82,    82,    82,
      82,    82,    83,    83,    84,    84,    85,    85,    85,    85,
      85,    85,    85,    86,    87,    87,    87,    87,    87,    87,
      87,    88,    88,    89,    89,    89,    89,    89,    89,    89,
      89,    89,    89,    89,    90,    90,    91,    91,    92,    92,
      92,    93,    93,    93,    94,    94,    94,    94,    94,    94,
      95,    95,    95,    95,    96,    96,    96,    96,    96,    96,
      96,    96,    96,    96,    96
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     1,     2,     2,
       3,     3,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     0,     1,     3,     3,     2,     1,     2,     1,     2,
       1,     3,     5,     3,     5,     2,     1,     1,     1,     3,
       5,     3,     5,     2,     1,     3,     5,     5,     0,     1,
       3,     5,     3,     3,     1,     1,     1,     1,     2,     2,
       1,     1,     1,     1,     4,     2,     1,     1,     1,     1,
       1,     1,     1,     2,     2,     2,     2,     2,     4,     5,
       3,     1,     1,     1,     4,     4,     4,     6,     9,     9,
       3,     3,     5,     8,     1,     6,     5,     7,     0,     2,
       2,     1,     1,     1,     1,     1,     2,     2,     2,     3,
       1,     2,     3,     4,     1,     3,     3,     3,     3,     3,
       4,     4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    31,     0,     0,     0,     0,
       0,     0,    31,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    58,     0,     0,     0,     7,     4,     0,     9,
      32,    12,     0,     0,   114,    76,    78,    81,    77,    79,
      82,    80,   108,   115,     0,     0,     0,    13,    38,    64,
      65,    91,    92,   104,    93,     0,    14,    72,    36,    73,
      15,     0,    16,     0,     0,   108,     0,    20,    46,    66,
      70,    71,    67,    93,    18,     0,    32,    47,    48,    21,
     108,     0,     0,    17,    40,     0,     0,    19,     0,    28,
       0,    29,     0,    22,     0,    23,     0,    24,    54,    25,
       0,    26,     0,    27,    59,    30,     0,     5,     0,     0,
       8,   117,   116,     0,     0,     0,     0,    37,     0,     0,
     124,     0,   118,     0,     0,     0,    87,    86,     0,    85,
      84,    35,     0,     0,    68,    69,    75,    45,     0,     0,
      75,    39,     0,     0,     0,     0,     0,     0,     0,    53,
       0,     0,     0,     0,    10,    11,   108,   109,   110,     0,
       0,   100,   101,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   119,     0,     0,     0,     0,    90,     0,
       0,    33,    34,     0,     0,    41,     0,    43,     0,    60,
       0,    62,    49,    51,    55,     0,     0,    63,     6,     0,
     113,   111,   112,     0,     0,     0,   134,   133,   132,     0,
       0,   125,   126,   127,   128,   129,     0,     0,    94,    96,
      95,     0,    88,    74,     0,     0,   120,    83,     0,     0,
       0,     0,     0,     0,     0,   106,   102,     0,   130,   131,
       0,     0,     0,    89,    42,   121,     0,    44,    61,    50,
      52,    56,    57,     0,     0,   105,    97,     0,     0,     0,
     122,   107,     0,     0,     0,   123,   103,     0,     0,    98,
      99
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    27,   153,    28,    31,    60,    62,    56,
      47,    83,    74,    87,    67,    79,    93,    95,    97,    99,
     101,   103,    89,    91,   105,    57,    68,    58,    69,    49,
     187,    59,    50,    51,    52,    53,   116,   203,    54,   227,
     121
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -89
static const yytype_int16 yypact[] =
{
     -89,    18,   -89,   163,    -5,   -13,   219,   253,   253,   335,
     194,    16,   274,   369,   418,   418,   253,   253,   253,   253,
     240,     0,     0,   253,   -17,    19,   -89,   -89,    24,   -89,
     -89,   -89,   479,   479,   -89,   -89,   -89,   -89,   -89,   -89,
     -89,   -89,   111,   -89,   335,   397,   479,   -89,   -89,   -89,
     -89,   -89,   -89,    33,    51,   390,   -89,   -89,    65,   -89,
     -89,    72,   -89,    73,   356,   111,   314,   -89,   -89,   -89,
     -89,   -89,   -89,    74,   -89,    30,   335,   -89,   -89,   -89,
      70,   422,   479,   -89,   -89,    82,    86,   -89,    88,   -89,
      89,   -89,    90,   -89,    91,   -89,    92,   -89,   101,   -89,
     105,   -89,   114,   -89,   -89,   -89,   116,   -89,   479,   479,
     -89,   -89,   -89,   118,   479,   479,   120,   -89,     5,   115,
     -89,    83,   -89,   133,   -12,   404,   -89,   -89,   439,   -89,
     -89,   -89,   335,   253,   -89,   -89,   120,   -89,     9,   479,
     -89,   -89,   422,   141,   119,   452,   253,   335,   335,   335,
     335,   335,   253,   163,   327,   327,    70,   -89,   -89,     4,
     479,   143,   -89,   479,   479,   479,   190,   191,   479,   479,
     479,   479,   479,   -89,   187,     6,   148,   152,   -89,   470,
     156,   -89,   -89,   158,   162,   -89,     8,   -89,   164,   -89,
     166,   -89,   170,   171,   -89,   169,   172,   -89,   -89,   173,
     -89,   -89,   -89,   161,   176,   199,   102,   530,   537,   479,
     479,    39,    39,   -89,   -89,   -89,   479,   479,   185,   -89,
     -89,   189,   -89,   -89,     0,   208,   234,   -89,   193,     0,
     211,   212,   479,   240,   217,   -89,   -89,   255,    55,    55,
     214,   215,    59,   -89,   -89,   260,   241,   -89,   -89,   -89,
     -89,   -89,   -89,   222,   479,   -89,   -89,   262,   276,   256,
     -89,   -89,   242,   479,   479,   -89,   -89,   243,   246,   -89,
     -89
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -89,   -89,   -89,   134,   -89,   -89,   289,   -89,   -89,   -89,
     290,   -89,   -89,   -89,   -89,   -89,   -89,   -89,   -89,   -89,
     -89,   -89,   -89,   -89,   -89,    -2,   237,    11,   -11,    -9,
      76,    -8,    87,    -4,     2,    -3,   -56,   -89,   -10,   -89,
     -88
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      73,    70,    84,    86,    48,    85,    71,    63,    75,   136,
      48,   100,    72,   102,   104,   160,   217,   225,     2,    61,
     154,   155,   111,   112,   140,   175,   176,    92,    94,    96,
      98,   107,   177,   108,   106,   120,   122,   200,    30,   201,
     226,   139,   117,   183,    29,   130,   175,   176,   202,   170,
     171,   172,   129,   177,    55,   134,    73,    70,   161,   218,
     135,    42,    71,   168,   169,   170,   171,   172,    72,   109,
     141,    86,   120,   110,   117,   206,   207,   208,   114,   115,
     211,   212,   213,   214,   215,   123,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   172,   257,   258,   120,   120,
     199,    88,    90,   124,   157,   158,   164,   165,   166,   167,
     168,   169,   170,   171,   172,   112,   131,   113,   120,   114,
     115,   238,   239,   132,   133,   156,   138,    32,    33,   184,
     181,   185,    86,   142,   188,   190,   173,   189,   143,   144,
     145,   146,   147,   148,   182,   192,   193,   194,   195,   196,
     204,    34,   149,   120,   120,   120,   150,   191,   120,   120,
     120,   120,   120,   197,     4,   151,    43,   152,   162,   112,
     174,    82,   159,   186,    46,   183,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,   205,   209,   216,   210,   120,
     120,   219,    32,    33,    64,   220,   240,   241,    24,   222,
      25,   223,    26,   224,   235,   228,   244,   229,   230,   231,
     232,   248,   251,   233,   252,   234,    34,    32,    33,   236,
      35,    36,    37,    38,    39,    40,   237,   242,    41,    65,
     245,    43,   243,   246,   262,    66,    45,   186,   249,    46,
     250,    34,   253,   267,   268,    35,    36,    37,    38,    39,
      40,    32,    33,    41,    42,   254,    43,   255,   256,   259,
      44,    45,   263,   260,    46,   261,    35,    36,    37,    38,
      39,    40,    32,    33,    41,    34,   264,   198,   265,    35,
      36,    37,    38,    39,    40,   266,   269,    41,    42,   270,
      43,    77,    78,   137,   247,    45,    34,    55,    46,     0,
      35,    36,    37,    38,    39,    40,     0,     0,    41,    42,
       0,    43,    32,    33,    64,    76,    45,     0,     0,    46,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
       0,     0,     0,    32,    33,     0,    34,     0,     0,     0,
      35,    36,    37,    38,    39,    40,     0,     0,    41,    65,
       0,    43,     0,     0,    32,    33,    45,    34,     0,    46,
       0,    35,    36,    37,    38,    39,    40,    32,    33,    41,
      42,     0,    43,     0,     0,     0,     0,    45,    34,     0,
      46,     0,    35,    36,    37,    38,    39,    40,    32,   125,
      41,    34,     0,    43,     0,    32,    33,     0,    45,     0,
       0,    46,    32,    33,    80,     0,    43,     0,     0,     0,
      81,    82,    34,    55,    46,     0,    32,    33,     0,    34,
      32,    33,   126,   127,   118,    42,    34,    43,     0,     0,
       0,   119,   128,     0,    43,    46,   178,    32,   179,    82,
      34,    43,    46,     0,    34,     0,    82,     0,     0,    46,
      32,    33,     0,    42,     0,    43,     0,    80,     0,    43,
      45,    34,     0,    46,    82,     0,     0,    46,    32,    33,
       0,   180,     0,     0,    34,     0,    43,    32,    33,     0,
       0,    82,     0,     0,    46,     0,     0,     0,     0,    43,
       0,     0,    34,     0,    82,     0,    55,    46,     0,     0,
       0,    34,   221,     0,     0,     0,     0,    43,     0,     0,
       0,     0,    82,     0,     0,    46,    43,     0,     0,     0,
       0,    82,     0,     0,    46,   165,   166,   167,   168,   169,
     170,   171,   172,   166,   167,   168,   169,   170,   171,   172
};

static const yytype_int16 yycheck[] =
{
      10,    10,    13,    13,     6,    13,    10,     9,    11,    65,
      12,    20,    10,    21,    22,    10,    10,     9,     0,     8,
     108,   109,    32,    33,    80,    37,    38,    16,    17,    18,
      19,    48,    44,    50,    23,    45,    46,    33,    51,    35,
      32,    11,    44,    34,    49,    55,    37,    38,    44,    10,
      11,    12,    55,    44,    54,    64,    66,    66,    53,    53,
      64,    45,    66,     8,     9,    10,    11,    12,    66,    50,
      81,    81,    82,    49,    76,   163,   164,   165,     8,     9,
     168,   169,   170,   171,   172,    52,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    37,    38,   108,   109,
     156,    14,    15,    52,   114,   115,     4,     5,     6,     7,
       8,     9,    10,    11,    12,   125,    51,     6,   128,     8,
       9,   209,   210,    51,    51,     7,    52,     8,     9,   139,
     132,   142,   142,    51,   144,   145,    53,   145,    52,    51,
      51,    51,    51,    51,   133,   147,   148,   149,   150,   151,
     160,    32,    51,   163,   164,   165,    51,   146,   168,   169,
     170,   171,   172,   152,     1,    51,    47,    51,    53,   179,
      37,    52,    52,    54,    55,    34,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    52,     6,    10,     7,   209,
     210,    53,     8,     9,    10,    53,   216,   217,    45,    53,
      47,    53,    49,    51,    53,    51,   224,    51,    48,    48,
      51,   229,   232,    51,   233,    52,    32,     8,     9,    53,
      36,    37,    38,    39,    40,    41,    37,    52,    44,    45,
      32,    47,    53,     9,   254,    51,    52,    54,    37,    55,
      38,    32,    35,   263,   264,    36,    37,    38,    39,    40,
      41,     8,     9,    44,    45,    10,    47,    53,    53,     9,
      51,    52,    10,    32,    55,    53,    36,    37,    38,    39,
      40,    41,     8,     9,    44,    32,    10,   153,    32,    36,
      37,    38,    39,    40,    41,    53,    53,    44,    45,    53,
      47,    12,    12,    66,   228,    52,    32,    54,    55,    -1,
      36,    37,    38,    39,    40,    41,    -1,    -1,    44,    45,
      -1,    47,     8,     9,    10,    51,    52,    -1,    -1,    55,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      -1,    -1,    -1,     8,     9,    -1,    32,    -1,    -1,    -1,
      36,    37,    38,    39,    40,    41,    -1,    -1,    44,    45,
      -1,    47,    -1,    -1,     8,     9,    52,    32,    -1,    55,
      -1,    36,    37,    38,    39,    40,    41,     8,     9,    44,
      45,    -1,    47,    -1,    -1,    -1,    -1,    52,    32,    -1,
      55,    -1,    36,    37,    38,    39,    40,    41,     8,     9,
      44,    32,    -1,    47,    -1,     8,     9,    -1,    52,    -1,
      -1,    55,     8,     9,    45,    -1,    47,    -1,    -1,    -1,
      51,    52,    32,    54,    55,    -1,     8,     9,    -1,    32,
       8,     9,    42,    43,    37,    45,    32,    47,    -1,    -1,
      -1,    44,    52,    -1,    47,    55,    42,     8,     9,    52,
      32,    47,    55,    -1,    32,    -1,    52,    -1,    -1,    55,
       8,     9,    -1,    45,    -1,    47,    -1,    45,    -1,    47,
      52,    32,    -1,    55,    52,    -1,    -1,    55,     8,     9,
      -1,    42,    -1,    -1,    32,    -1,    47,     8,     9,    -1,
      -1,    52,    -1,    -1,    55,    -1,    -1,    -1,    -1,    47,
      -1,    -1,    32,    -1,    52,    -1,    54,    55,    -1,    -1,
      -1,    32,    42,    -1,    -1,    -1,    -1,    47,    -1,    -1,
      -1,    -1,    52,    -1,    -1,    55,    47,    -1,    -1,    -1,
      -1,    52,    -1,    -1,    55,     5,     6,     7,     8,     9,
      10,    11,    12,     6,     7,     8,     9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    57,     0,    58,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    45,    47,    49,    59,    61,    49,
      51,    62,     8,     9,    32,    36,    37,    38,    39,    40,
      41,    44,    45,    47,    51,    52,    55,    66,    81,    85,
      88,    89,    90,    91,    94,    54,    65,    81,    83,    87,
      63,    83,    64,    81,    10,    45,    51,    70,    82,    84,
      85,    89,    90,    94,    68,    91,    51,    62,    66,    71,
      45,    51,    52,    67,    84,    87,    94,    69,    88,    78,
      88,    79,    83,    72,    83,    73,    83,    74,    83,    75,
      85,    76,    87,    77,    87,    80,    83,    48,    50,    50,
      49,    94,    94,     6,     8,     9,    92,    81,    37,    44,
      94,    96,    94,    52,    52,     9,    42,    43,    52,    91,
      94,    51,    51,    51,    85,    89,    92,    82,    52,    11,
      92,    84,    51,    52,    51,    51,    51,    51,    51,    51,
      51,    51,    51,    60,    96,    96,     7,    94,    94,    52,
      10,    53,    53,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    53,    37,    37,    38,    44,    42,     9,
      42,    81,    83,    34,    94,    84,    54,    86,    94,    87,
      94,    83,    81,    81,    81,    81,    81,    83,    59,    92,
      33,    35,    44,    93,    94,    52,    96,    96,    96,     6,
       7,    96,    96,    96,    96,    96,    10,    10,    53,    53,
      53,    42,    53,    53,    51,     9,    32,    95,    51,    51,
      48,    48,    51,    51,    52,    53,    53,    37,    96,    96,
      94,    94,    52,    53,    87,    32,     9,    86,    87,    37,
      38,    94,    85,    35,    10,    53,    53,    37,    38,     9,
      32,    53,    94,    10,    10,    32,    53,    94,    94,    53,
      53
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

  case 18:
#line 105 "a.y"
    { outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr2)); }
    break;

  case 19:
#line 106 "a.y"
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

  case 28:
#line 115 "a.y"
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
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
	}
    break;

  case 43:
#line 195 "a.y"
    {
		settext((yyvsp[(1) - (3)].addr).sym);
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 44:
#line 201 "a.y"
    {
		settext((yyvsp[(1) - (5)].addr).sym);
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
	}
    break;

  case 45:
#line 210 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(2) - (2)].addr);
	}
    break;

  case 46:
#line 215 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 49:
#line 226 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 50:
#line 231 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).from.index != D_NONE)
			yyerror("dp shift with lhs index");
		(yyval.addr2).from.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 51:
#line 241 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 52:
#line 246 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		if((yyval.addr2).to.index != D_NONE)
			yyerror("dp move with lhs index");
		(yyval.addr2).to.index = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 53:
#line 256 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (2)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 54:
#line 261 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 55:
#line 266 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 56:
#line 273 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to.offset = (yyvsp[(5) - (5)].lval);
	}
    break;

  case 57:
#line 281 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(3) - (5)].addr);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
		if((yyvsp[(1) - (5)].addr).type != D_CONST)
			yyerror("illegal constant");
		(yyval.addr2).to.offset = (yyvsp[(1) - (5)].addr).offset;
	}
    break;

  case 58:
#line 290 "a.y"
    {
		(yyval.addr2).from = nullgen;
		(yyval.addr2).to = nullgen;
	}
    break;

  case 59:
#line 295 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (1)].addr);
		(yyval.addr2).to = nullgen;
	}
    break;

  case 60:
#line 302 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (3)].addr);
		(yyval.addr2).to = (yyvsp[(3) - (3)].addr);
	}
    break;

  case 61:
#line 307 "a.y"
    {
		(yyval.addr2).from = (yyvsp[(1) - (5)].addr);
		(yyval.addr2).from.scale = (yyvsp[(3) - (5)].lval);
		(yyval.addr2).to = (yyvsp[(5) - (5)].addr);
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

  case 74:
#line 357 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 75:
#line 363 "a.y"
    {
		(yyvsp[(1) - (2)].sym) = labellookup((yyvsp[(1) - (2)].sym));
		(yyval.addr) = nullgen;
		if(pass == 2 && (yyvsp[(1) - (2)].sym)->type != LLAB)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->labelname);
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 76:
#line 374 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 77:
#line 379 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 78:
#line 384 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 79:
#line 389 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 80:
#line 394 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SP;
	}
    break;

  case 81:
#line 399 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 82:
#line 404 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 83:
#line 410 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 84:
#line 418 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 85:
#line 424 "a.y"
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
#line 435 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 87:
#line 441 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 88:
#line 447 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(3) - (4)].dval);
	}
    break;

  case 89:
#line 453 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(4) - (5)].dval);
	}
    break;

  case 90:
#line 459 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 93:
#line 471 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_NONE;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 94:
#line 477 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 95:
#line 483 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_SP;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 96:
#line 489 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 97:
#line 495 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_NONE;
		(yyval.addr).offset = (yyvsp[(1) - (6)].lval);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 98:
#line 504 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(3) - (9)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (9)].lval);
		(yyval.addr).index = (yyvsp[(6) - (9)].lval);
		(yyval.addr).scale = (yyvsp[(8) - (9)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 99:
#line 513 "a.y"
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
#line 522 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(2) - (3)].lval);
	}
    break;

  case 101:
#line 527 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_SP;
	}
    break;

  case 102:
#line 532 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+D_NONE;
		(yyval.addr).index = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 103:
#line 540 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_INDIR+(yyvsp[(2) - (8)].lval);
		(yyval.addr).index = (yyvsp[(5) - (8)].lval);
		(yyval.addr).scale = (yyvsp[(7) - (8)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 104:
#line 550 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (1)].addr);
	}
    break;

  case 105:
#line 554 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (6)].addr);
		(yyval.addr).index = (yyvsp[(3) - (6)].lval);
		(yyval.addr).scale = (yyvsp[(5) - (6)].lval);
		checkscale((yyval.addr).scale);
	}
    break;

  case 106:
#line 563 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 107:
#line 570 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 1);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 108:
#line 578 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 109:
#line 582 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 110:
#line 586 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 112:
#line 593 "a.y"
    {
		(yyval.lval) = D_AUTO;
	}
    break;

  case 115:
#line 601 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 116:
#line 605 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 117:
#line 609 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 118:
#line 613 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 119:
#line 617 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 120:
#line 623 "a.y"
    {
		(yyval.lval) = ((yyvsp[(1) - (1)].lval) & 0xffffffffLL) +
			((vlong)ArgsSizeUnknown << 32);
	}
    break;

  case 121:
#line 628 "a.y"
    {
		(yyval.lval) = (-(yyvsp[(2) - (2)].lval) & 0xffffffffLL) +
			((vlong)ArgsSizeUnknown << 32);
	}
    break;

  case 122:
#line 633 "a.y"
    {
		(yyval.lval) = ((yyvsp[(1) - (3)].lval) & 0xffffffffLL) +
			(((yyvsp[(3) - (3)].lval) & 0xffffLL) << 32);
	}
    break;

  case 123:
#line 638 "a.y"
    {
		(yyval.lval) = (-(yyvsp[(2) - (4)].lval) & 0xffffffffLL) +
			(((yyvsp[(4) - (4)].lval) & 0xffffLL) << 32);
	}
    break;

  case 125:
#line 646 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:
#line 650 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 127:
#line 654 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:
#line 658 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:
#line 662 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 130:
#line 666 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 131:
#line 670 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 132:
#line 674 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 133:
#line 678 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 134:
#line 682 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1267 of yacc.c.  */
#line 2572 "y.tab.c"
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



