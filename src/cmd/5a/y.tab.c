
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
     LCONST = 282,
     LSP = 283,
     LSB = 284,
     LFP = 285,
     LPC = 286,
     LTYPEX = 287,
     LR = 288,
     LREG = 289,
     LF = 290,
     LFREG = 291,
     LC = 292,
     LCREG = 293,
     LPSR = 294,
     LFCR = 295,
     LCOND = 296,
     LS = 297,
     LAT = 298,
     LFCONST = 299,
     LSCONST = 300,
     LNAME = 301,
     LLAB = 302,
     LVAR = 303
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
#define LCONST 282
#define LSP 283
#define LSB 284
#define LFP 285
#define LPC 286
#define LTYPEX 287
#define LR 288
#define LREG 289
#define LF 290
#define LFREG 291
#define LC 292
#define LCREG 293
#define LPSR 294
#define LFCR 295
#define LCOND 296
#define LS 297
#define LAT 298
#define LFCONST 299
#define LSCONST 300
#define LNAME 301
#define LLAB 302
#define LVAR 303




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
#line 223 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 235 "y.tab.c"

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
#define YYLAST   643

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  69
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  35
/* YYNRULES -- Number of rules.  */
#define YYNRULES  129
/* YYNRULES -- Number of states.  */
#define YYNSTATES  327

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   303

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    67,    12,     5,     2,
      65,    66,    10,     8,    62,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    59,    61,
       6,    60,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    63,     2,    64,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    68,     2,     2,     2,
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
      55,    56,    57,    58
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
     199,   212,   220,   230,   233,   234,   237,   240,   241,   244,
     249,   252,   255,   258,   261,   266,   269,   271,   274,   278,
     280,   284,   288,   290,   292,   294,   299,   301,   303,   305,
     307,   309,   311,   313,   317,   319,   324,   326,   331,   333,
     335,   337,   339,   342,   344,   350,   355,   360,   365,   370,
     372,   374,   376,   378,   383,   385,   387,   389,   394,   396,
     398,   400,   405,   410,   416,   424,   425,   428,   431,   433,
     435,   437,   439,   441,   444,   447,   450,   454,   455,   458,
     460,   464,   468,   472,   476,   480,   485,   490,   494,   498
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      70,     0,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,
      57,    59,    73,    72,    -1,    -1,    56,    59,    74,    72,
      -1,    56,    60,   103,    61,    -1,    58,    60,   103,    61,
      -1,    61,    -1,    75,    61,    -1,     1,    61,    -1,    13,
      76,    87,    62,    94,    62,    89,    -1,    13,    76,    87,
      62,    94,    62,    -1,    13,    76,    87,    62,    89,    -1,
      14,    76,    87,    62,    89,    -1,    15,    76,    82,    62,
      82,    -1,    16,    76,    77,    78,    -1,    16,    76,    77,
      83,    -1,    36,    77,    84,    -1,    17,    77,    78,    -1,
      18,    76,    77,    82,    -1,    19,    76,    87,    62,    94,
      77,    -1,    20,    76,    85,    62,    63,    81,    64,    -1,
      20,    76,    63,    81,    64,    62,    85,    -1,    21,    76,
      89,    62,    84,    62,    89,    -1,    21,    76,    89,    62,
      84,    77,    -1,    21,    76,    77,    84,    62,    89,    -1,
      22,    76,    77,    -1,    23,    98,    62,    88,    -1,    23,
      98,    62,   101,    62,    88,    -1,    24,    98,    11,   101,
      62,    79,    -1,    25,    76,    89,    77,    -1,    29,    77,
      79,    -1,    30,    76,    97,    62,    97,    -1,    32,    76,
      96,    62,    97,    -1,    32,    76,    96,    62,    46,    62,
      97,    -1,    33,    76,    97,    62,    97,    77,    -1,    31,
      76,   101,    62,   103,    62,    94,    62,    95,    62,    95,
     102,    -1,    34,    76,    89,    62,    89,    62,    90,    -1,
      35,    76,    89,    62,    89,    62,    89,    62,    94,    -1,
      26,    77,    -1,    -1,    76,    51,    -1,    76,    52,    -1,
      -1,    62,    77,    -1,   101,    65,    41,    66,    -1,    56,
      99,    -1,    57,    99,    -1,    67,   101,    -1,    67,    86,
      -1,    67,    10,    67,    86,    -1,    67,    55,    -1,    80,
      -1,    67,    54,    -1,    67,     9,    54,    -1,    94,    -1,
      94,     9,    94,    -1,    94,    77,    81,    -1,    89,    -1,
      79,    -1,    91,    -1,    91,    65,    94,    66,    -1,    49,
      -1,    50,    -1,   101,    -1,    86,    -1,    97,    -1,    84,
      -1,    98,    -1,    65,    94,    66,    -1,    84,    -1,   101,
      65,    93,    66,    -1,    98,    -1,    98,    65,    93,    66,
      -1,    85,    -1,    89,    -1,    88,    -1,    91,    -1,    67,
     101,    -1,    94,    -1,    65,    94,    62,    94,    66,    -1,
      94,     6,     6,    92,    -1,    94,     7,     7,    92,    -1,
      94,     9,     7,    92,    -1,    94,    53,     7,    92,    -1,
      94,    -1,   101,    -1,    44,    -1,    41,    -1,    43,    65,
     103,    66,    -1,    93,    -1,    38,    -1,    48,    -1,    47,
      65,   103,    66,    -1,    97,    -1,    80,    -1,    46,    -1,
      45,    65,   101,    66,    -1,   101,    65,   100,    66,    -1,
      56,    99,    65,   100,    66,    -1,    56,     6,     7,    99,
      65,    39,    66,    -1,    -1,     8,   101,    -1,     9,   101,
      -1,    39,    -1,    38,    -1,    40,    -1,    37,    -1,    58,
      -1,     9,   101,    -1,     8,   101,    -1,    68,   101,    -1,
      65,   103,    66,    -1,    -1,    62,   103,    -1,   101,    -1,
     103,     8,   103,    -1,   103,     9,   103,    -1,   103,    10,
     103,    -1,   103,    11,   103,    -1,   103,    12,   103,    -1,
     103,     6,     6,   103,    -1,   103,     7,     7,   103,    -1,
     103,     5,   103,    -1,   103,     4,   103,    -1,   103,     3,
     103,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    67,    67,    69,    68,    76,    75,    83,    82,    88,
      93,    99,   100,   101,   107,   111,   115,   122,   129,   136,
     140,   147,   154,   161,   168,   175,   184,   196,   200,   204,
     211,   218,   222,   229,   236,   243,   250,   254,   258,   262,
     269,   291,   298,   307,   313,   316,   320,   325,   326,   329,
     335,   344,   352,   358,   363,   368,   374,   377,   383,   391,
     395,   404,   410,   411,   412,   413,   418,   424,   430,   436,
     437,   440,   441,   449,   458,   459,   468,   469,   475,   478,
     479,   480,   482,   490,   498,   507,   513,   519,   525,   533,
     539,   547,   548,   552,   560,   561,   567,   568,   576,   577,
     580,   586,   594,   602,   610,   620,   623,   627,   633,   634,
     635,   638,   639,   643,   647,   651,   655,   661,   664,   670,
     671,   675,   679,   683,   687,   691,   695,   699,   703,   707
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
  "LTYPEJ", "LTYPEK", "LTYPEL", "LTYPEM", "LTYPEN", "LTYPEBX", "LCONST",
  "LSP", "LSB", "LFP", "LPC", "LTYPEX", "LR", "LREG", "LF", "LFREG", "LC",
  "LCREG", "LPSR", "LFCR", "LCOND", "LS", "LAT", "LFCONST", "LSCONST",
  "LNAME", "LLAB", "LVAR", "':'", "'='", "';'", "','", "'['", "']'", "'('",
  "')'", "'$'", "'~'", "$accept", "prog", "$@1", "line", "$@2", "$@3",
  "inst", "cond", "comma", "rel", "ximm", "fcon", "reglist", "gen",
  "nireg", "ireg", "ioreg", "oreg", "imsr", "imm", "reg", "regreg",
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
     295,   296,   297,   298,   299,   300,   301,   302,   303,    58,
      61,    59,    44,    91,    93,    40,    41,    36,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    69,    70,    71,    70,    73,    72,    74,    72,    72,
      72,    72,    72,    72,    75,    75,    75,    75,    75,    75,
      75,    75,    75,    75,    75,    75,    75,    75,    75,    75,
      75,    75,    75,    75,    75,    75,    75,    75,    75,    75,
      75,    75,    75,    75,    76,    76,    76,    77,    77,    78,
      78,    78,    79,    79,    79,    79,    79,    80,    80,    81,
      81,    81,    82,    82,    82,    82,    82,    82,    82,    82,
      82,    83,    83,    84,    85,    85,    86,    86,    86,    87,
      87,    87,    88,    89,    90,    91,    91,    91,    91,    92,
      92,    93,    93,    93,    94,    94,    95,    95,    96,    96,
      97,    97,    98,    98,    98,    99,    99,    99,   100,   100,
     100,   101,   101,   101,   101,   101,   101,   102,   102,   103,
     103,   103,   103,   103,   103,   103,   103,   103,   103,   103
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     0,     4,     4,
       4,     1,     2,     2,     7,     6,     5,     5,     5,     4,
       4,     3,     3,     4,     6,     7,     7,     7,     6,     6,
       3,     4,     6,     6,     4,     3,     5,     5,     7,     6,
      12,     7,     9,     2,     0,     2,     2,     0,     2,     4,
       2,     2,     2,     2,     4,     2,     1,     2,     3,     1,
       3,     3,     1,     1,     1,     4,     1,     1,     1,     1,
       1,     1,     1,     3,     1,     4,     1,     4,     1,     1,
       1,     1,     2,     1,     5,     4,     4,     4,     4,     1,
       1,     1,     1,     4,     1,     1,     1,     4,     1,     1,
       1,     4,     4,     5,     7,     0,     2,     2,     1,     1,
       1,     1,     1,     2,     2,     2,     3,     0,     2,     1,
       3,     3,     3,     3,     3,     4,     4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    44,    44,    44,    44,    47,
      44,    44,    44,    44,    44,     0,     0,    44,    47,    47,
      44,    44,    44,    44,    44,    44,    47,     0,     0,     0,
      11,     4,     0,    13,     0,     0,     0,    47,    47,     0,
      47,     0,     0,    47,    47,     0,     0,   111,   105,   112,
       0,     0,     0,     0,     0,     0,    43,     0,     0,     0,
       0,     0,     0,     0,     0,     7,     0,     5,     0,    12,
      95,    92,     0,    91,    45,    46,     0,     0,    80,    79,
      81,    94,    83,     0,     0,   100,    66,    67,     0,     0,
      63,    56,     0,    74,    78,    69,    62,    64,    70,    76,
      68,     0,    48,   105,   105,    22,     0,     0,     0,     0,
       0,     0,     0,     0,    83,    30,   114,   113,     0,     0,
       0,     0,   119,     0,   115,     0,     0,     0,    47,    35,
       0,     0,     0,    99,     0,    98,     0,     0,     0,     0,
      21,     0,     0,     0,     0,     0,    82,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    57,    55,    53,
      52,     0,     0,     0,     0,   105,    19,    20,    71,    72,
       0,    50,    51,     0,    23,     0,     0,    47,     0,     0,
       0,     0,   105,   106,   107,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   116,    31,     0,   109,
     108,   110,     0,     0,    34,     0,     0,     0,     0,     0,
       0,     0,     8,     9,     6,    10,     0,    16,    83,     0,
       0,     0,     0,    17,     0,    73,    58,     0,    18,     0,
       0,     0,    50,     0,     0,    47,     0,     0,     0,     0,
       0,    47,     0,     0,   129,   128,   127,     0,     0,   120,
     121,   122,   123,   124,     0,   102,     0,    36,     0,   100,
      37,    47,     0,     0,    93,    15,    85,    89,    90,    86,
      87,    88,   101,    54,     0,    65,    77,    75,    49,    24,
       0,    60,    61,     0,    29,    47,    28,     0,   103,   125,
     126,    32,    33,     0,     0,    39,     0,     0,    14,    26,
      25,    27,     0,     0,    38,     0,    41,     0,   104,     0,
       0,     0,     0,    96,     0,     0,    42,     0,     0,     0,
       0,   117,    84,    97,     0,    40,   118
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    31,   143,   141,    32,    34,   102,   105,
      90,    91,   176,    92,   167,    93,    94,    95,    77,    78,
      79,   306,    80,   266,    81,   114,   314,   134,    98,    99,
     121,   202,   122,   325,   123
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -147
static const yytype_int16 yypact[] =
{
    -147,    19,  -147,   305,   -37,  -147,  -147,  -147,  -147,   -40,
    -147,  -147,  -147,  -147,  -147,   448,   448,  -147,   -40,   -40,
    -147,  -147,  -147,  -147,  -147,  -147,   -40,    -8,   -22,   -21,
    -147,  -147,   -20,  -147,   102,   102,   334,   108,   -40,   416,
     108,   102,   397,    48,   108,   482,   482,  -147,    95,  -147,
     482,   482,   -19,    18,    34,   223,  -147,     5,   174,   424,
     132,   174,   223,   223,    20,  -147,   482,  -147,   482,  -147,
    -147,  -147,    30,  -147,  -147,  -147,   482,    35,  -147,  -147,
    -147,  -147,    49,    76,    84,  -147,  -147,  -147,   205,   339,
    -147,  -147,    93,  -147,  -147,  -147,  -147,   110,  -147,   124,
     125,   442,  -147,    98,    98,  -147,   128,   373,   134,    36,
     144,   147,    20,   153,  -147,  -147,  -147,  -147,    60,   482,
     482,   175,  -147,   114,  -147,   249,    21,   482,   -40,  -147,
     182,   183,    12,  -147,   185,  -147,   188,   189,   198,    36,
    -147,   305,   562,   305,   572,   482,  -147,    36,   256,   258,
     261,   269,    36,   482,   212,   460,   216,  -147,  -147,  -147,
     125,   373,    36,   138,   586,    95,  -147,  -147,  -147,  -147,
     217,  -147,  -147,   247,  -147,    36,   226,     7,   229,   138,
     227,    20,    98,  -147,  -147,    21,   482,   482,   482,   287,
     301,   482,   482,   482,   482,   482,  -147,  -147,   248,  -147,
    -147,  -147,   243,   250,  -147,    66,   482,   257,   106,    66,
      36,    36,  -147,  -147,  -147,  -147,   225,  -147,   251,   205,
     205,   205,   205,  -147,   266,  -147,  -147,   478,  -147,   267,
     278,   279,   175,   215,   280,   -40,   253,    36,    36,    36,
      36,   297,   295,   298,   601,   345,   609,   482,   482,   190,
     190,  -147,  -147,  -147,   300,  -147,     5,  -147,   552,   303,
    -147,   -40,   306,   307,  -147,    36,  -147,  -147,  -147,  -147,
    -147,  -147,  -147,  -147,   125,  -147,  -147,  -147,  -147,  -147,
     486,  -147,  -147,   309,  -147,   130,  -147,   331,  -147,   121,
     121,  -147,  -147,    36,    66,  -147,   322,    36,  -147,  -147,
    -147,  -147,   308,   326,  -147,    36,  -147,   327,  -147,   119,
     329,    36,   333,  -147,   338,    36,  -147,   482,   119,   330,
     292,   341,  -147,  -147,   482,  -147,   631
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -147,  -147,  -147,  -114,  -147,  -147,  -147,   579,    44,   311,
     -49,   348,     0,   -97,  -147,   -44,   -39,   -80,    -9,  -119,
     -13,  -147,   -25,     2,  -146,   -34,    91,  -147,   -14,    -3,
     -89,   228,   -11,  -147,   -30
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -60
static const yytype_int16 yytable[] =
{
      82,    82,    82,   110,    53,    53,   197,    82,   129,   159,
     174,    97,    52,    54,   171,   172,   237,   230,   231,     2,
     140,   207,    38,    96,    33,   100,    83,   212,   106,   214,
     113,   111,   108,   231,   116,   117,   142,    67,   144,    68,
     124,    69,   128,   125,   130,   127,   135,   136,   131,   137,
     138,    65,    66,    39,   154,   148,   149,   168,   150,   199,
     200,   201,    56,    57,   228,   146,   157,   182,   180,    38,
      64,   -59,    89,    82,    70,   177,   232,    71,   160,    72,
      73,   101,    97,   126,   107,   139,    70,   112,   115,    71,
     170,    72,    73,   242,    96,   145,   100,   147,   169,    74,
      75,   118,   151,   119,   120,   154,   119,   120,   183,   184,
      38,    84,    85,   218,   198,   216,   203,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,    82,   229,   191,
     192,   193,   194,   195,   217,   291,    97,   241,   152,   223,
      70,   235,   224,    71,   117,    72,    73,   273,    96,   153,
     100,    84,   259,    74,    75,   161,   244,   245,   246,    74,
      75,   249,   250,   251,   252,   253,   312,   313,    70,    76,
      38,    71,   204,    72,    73,   162,   258,    84,    85,    71,
     196,    72,    73,    74,    75,   267,   267,   267,   267,   163,
     164,   257,    38,   173,   260,   261,   175,   262,   263,   132,
     193,   194,   195,   281,   177,   177,   178,   292,   268,   268,
     268,   268,   179,    45,    46,   181,   274,   289,   290,    84,
      85,   238,   269,   270,   271,    74,    75,   284,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   282,   283,
     185,   299,    47,    70,   205,   206,    71,   208,    72,    73,
     209,   210,   298,   199,   200,   201,   234,    45,    46,   303,
     211,    70,   219,    49,    71,   220,    72,    73,   221,   111,
      50,   310,   301,    51,    74,    75,   222,   316,   225,   279,
     304,   319,   233,   227,   307,   286,    47,   320,   234,   240,
     236,   264,   239,   247,   326,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   295,     4,    49,   248,   255,
     254,   226,   256,   265,    50,   280,    76,    51,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,   272,   275,    19,    20,    21,    22,    23,    24,
      25,    26,    45,    46,   276,   277,   278,    45,   155,   156,
     188,   189,   190,   191,   192,   193,   194,   195,   323,   285,
     287,    27,    28,    29,   288,   294,    30,    76,   296,   297,
     302,    47,    70,   300,   308,    71,    47,    72,    73,    84,
      85,    45,    46,    86,    87,    74,    75,   305,   309,   311,
      48,   315,    49,   157,   158,    48,   322,    49,   317,    88,
     318,    89,    51,   324,    88,    45,    46,    51,   133,   321,
      47,    70,   166,   243,    71,     0,    72,    73,    84,    85,
       0,     0,    86,    87,    45,    46,     0,     0,     0,    48,
       0,    49,    45,    46,    47,     0,     0,     0,    88,     0,
      89,    51,     0,     0,     0,     0,     0,     0,    74,    75,
      45,    46,     0,    47,     0,    49,    45,    46,     0,     0,
     109,    47,    88,     0,     0,    51,     0,     0,    45,    46,
       0,     0,   103,   104,    49,    74,    75,     0,     0,    47,
       0,    50,    49,     0,    51,    47,    45,    46,     0,    50,
      45,    46,    51,     0,    45,    46,     0,    47,   165,   104,
      49,     0,     0,     0,    48,     0,    49,    88,     0,     0,
      51,     0,     0,    50,   226,    47,    51,     0,    49,    47,
       0,     0,     0,    47,     0,    50,     0,     0,    51,     0,
       0,     0,     0,     0,    48,     0,    49,     0,     0,     0,
      49,     0,     0,    88,    49,     0,    51,    50,     0,     0,
      51,    88,     0,     0,    51,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,    35,    36,    37,     0,    40,
      41,    42,    43,    44,     0,     0,    55,     0,     0,    58,
      59,    60,    61,    62,    63,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   293,   189,   190,   191,   192,   193,
     194,   195,     0,   213,   199,   200,   201,    71,     0,    72,
      73,     0,     0,   215,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195
};

static const yytype_int16 yycheck[] =
{
      34,    35,    36,    42,    15,    16,   125,    41,    57,    89,
     107,    36,    15,    16,   103,   104,     9,   163,   164,     0,
      64,     9,    62,    36,    61,    36,    35,   141,    39,   143,
      43,    42,    41,   179,    45,    46,    66,    59,    68,    60,
      51,    61,    55,    62,    58,    11,    60,    61,    59,    62,
      63,    59,    60,     9,    88,     6,     7,   101,     9,    38,
      39,    40,    18,    19,   161,    76,    54,     7,   112,    62,
      26,    64,    67,   107,    38,   109,   165,    41,    89,    43,
      44,    37,   107,    65,    40,    65,    38,    43,    44,    41,
     101,    43,    44,   182,   107,    65,   107,    62,   101,    51,
      52,     6,    53,     8,     9,   139,     8,     9,   119,   120,
      62,    45,    46,   147,   125,   145,   127,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,   161,   162,     8,
       9,    10,    11,    12,   147,   254,   161,   181,    62,   152,
      38,   175,   153,    41,   155,    43,    44,   227,   161,    65,
     161,    45,    46,    51,    52,    62,   186,   187,   188,    51,
      52,   191,   192,   193,   194,   195,    47,    48,    38,    67,
      62,    41,   128,    43,    44,    65,   206,    45,    46,    41,
      66,    43,    44,    51,    52,   219,   220,   221,   222,    65,
      65,   205,    62,    65,   208,   209,    62,   210,   211,    67,
      10,    11,    12,   237,   238,   239,    62,   256,   219,   220,
     221,   222,    65,     8,     9,    62,   227,   247,   248,    45,
      46,   177,   220,   221,   222,    51,    52,   240,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,   238,   239,
      65,   280,    37,    38,    62,    62,    41,    62,    43,    44,
      62,    62,   265,    38,    39,    40,    41,     8,     9,   293,
      62,    38,     6,    58,    41,     7,    43,    44,     7,   280,
      65,   305,   285,    68,    51,    52,     7,   311,    66,   235,
     294,   315,    65,    67,   297,   241,    37,   317,    41,    62,
      64,    66,    63,     6,   324,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,   261,     1,    58,     7,    66,
      62,    54,    62,    62,    65,    62,    67,    68,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    66,    66,    29,    30,    31,    32,    33,    34,
      35,    36,     8,     9,    66,    66,    66,     8,     9,    10,
       5,     6,     7,     8,     9,    10,    11,    12,    66,    62,
      65,    56,    57,    58,    66,    62,    61,    67,    62,    62,
      39,    37,    38,    64,    66,    41,    37,    43,    44,    45,
      46,     8,     9,    49,    50,    51,    52,    65,    62,    62,
      56,    62,    58,    54,    55,    56,    66,    58,    65,    65,
      62,    67,    68,    62,    65,     8,     9,    68,    60,   318,
      37,    38,   101,   185,    41,    -1,    43,    44,    45,    46,
      -1,    -1,    49,    50,     8,     9,    -1,    -1,    -1,    56,
      -1,    58,     8,     9,    37,    -1,    -1,    -1,    65,    -1,
      67,    68,    -1,    -1,    -1,    -1,    -1,    -1,    51,    52,
       8,     9,    -1,    37,    -1,    58,     8,     9,    -1,    -1,
      63,    37,    65,    -1,    -1,    68,    -1,    -1,     8,     9,
      -1,    -1,    56,    57,    58,    51,    52,    -1,    -1,    37,
      -1,    65,    58,    -1,    68,    37,     8,     9,    -1,    65,
       8,     9,    68,    -1,     8,     9,    -1,    37,    56,    57,
      58,    -1,    -1,    -1,    56,    -1,    58,    65,    -1,    -1,
      68,    -1,    -1,    65,    54,    37,    68,    -1,    58,    37,
      -1,    -1,    -1,    37,    -1,    65,    -1,    -1,    68,    -1,
      -1,    -1,    -1,    -1,    56,    -1,    58,    -1,    -1,    -1,
      58,    -1,    -1,    65,    58,    -1,    68,    65,    -1,    -1,
      68,    65,    -1,    -1,    68,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,     6,     7,     8,    -1,    10,
      11,    12,    13,    14,    -1,    -1,    17,    -1,    -1,    20,
      21,    22,    23,    24,    25,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    62,     6,     7,     8,     9,    10,
      11,    12,    -1,    61,    38,    39,    40,    41,    -1,    43,
      44,    -1,    -1,    61,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    70,     0,    71,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    29,
      30,    31,    32,    33,    34,    35,    36,    56,    57,    58,
      61,    72,    75,    61,    76,    76,    76,    76,    62,    77,
      76,    76,    76,    76,    76,     8,     9,    37,    56,    58,
      65,    68,    98,   101,    98,    76,    77,    77,    76,    76,
      76,    76,    76,    76,    77,    59,    60,    59,    60,    61,
      38,    41,    43,    44,    51,    52,    67,    87,    88,    89,
      91,    93,    94,    87,    45,    46,    49,    50,    65,    67,
      79,    80,    82,    84,    85,    86,    89,    91,    97,    98,
     101,    77,    77,    56,    57,    78,   101,    77,    87,    63,
      85,   101,    77,    89,    94,    77,   101,   101,     6,     8,
       9,    99,   101,   103,   101,    62,    65,    11,    89,    79,
      97,   101,    67,    80,    96,    97,    97,    89,    89,    65,
      84,    74,   103,    73,   103,    65,   101,    62,     6,     7,
       9,    53,    62,    65,    94,     9,    10,    54,    55,    86,
     101,    62,    65,    65,    65,    56,    78,    83,    84,    98,
     101,    99,    99,    65,    82,    62,    81,    94,    62,    65,
      84,    62,     7,   101,   101,    65,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    66,    88,   101,    38,
      39,    40,   100,   101,    77,    62,    62,     9,    62,    62,
      62,    62,    72,    61,    72,    61,   103,    89,    94,     6,
       7,     7,     7,    89,   101,    66,    54,    67,    82,    94,
      93,    93,    99,    65,    41,    94,    64,     9,    77,    63,
      62,    84,    99,   100,   103,   103,   103,     6,     7,   103,
     103,   103,   103,   103,    62,    66,    62,    97,   103,    46,
      97,    97,    89,    89,    66,    62,    92,    94,   101,    92,
      92,    92,    66,    86,   101,    66,    66,    66,    66,    77,
      62,    94,    81,    81,    89,    62,    77,    65,    66,   103,
     103,    88,    79,    62,    62,    77,    62,    62,    89,    85,
      64,    89,    39,    94,    97,    65,    90,    89,    66,    62,
      94,    62,    47,    48,    95,    62,    94,    65,    62,    94,
     103,    95,    66,    66,    62,   102,   103
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
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].gen), NREG, &(yyvsp[(4) - (4)].gen));
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
		outcode((yyvsp[(1) - (2)].lval), Always, &nullgen, NREG, &nullgen);
	}
    break;

  case 44:

/* Line 1455 of yacc.c  */
#line 313 "a.y"
    {
		(yyval.lval) = Always;
	}
    break;

  case 45:

/* Line 1455 of yacc.c  */
#line 317 "a.y"
    {
		(yyval.lval) = ((yyvsp[(1) - (2)].lval) & ~C_SCOND) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 46:

/* Line 1455 of yacc.c  */
#line 321 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (2)].lval) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 49:

/* Line 1455 of yacc.c  */
#line 330 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 336 "a.y"
    {
		(yyval.gen) = nullgen;
		if(pass == 2)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->name);
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 345 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_BRANCH;
		(yyval.gen).sym = (yyvsp[(1) - (2)].sym);
		(yyval.gen).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 353 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST;
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 359 "a.y"
    {
		(yyval.gen) = (yyvsp[(2) - (2)].gen);
		(yyval.gen).type = D_CONST;
	}
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 364 "a.y"
    {
		(yyval.gen) = (yyvsp[(4) - (4)].gen);
		(yyval.gen).type = D_OCONST;
	}
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 369 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SCONST;
		memcpy((yyval.gen).sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.gen).sval));
	}
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 378 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 384 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FCONST;
		(yyval.gen).dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 392 "a.y"
    {
		(yyval.lval) = 1 << (yyvsp[(1) - (1)].lval);
	}
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 396 "a.y"
    {
		int i;
		(yyval.lval)=0;
		for(i=(yyvsp[(1) - (3)].lval); i<=(yyvsp[(3) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
		for(i=(yyvsp[(3) - (3)].lval); i<=(yyvsp[(1) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
	}
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 405 "a.y"
    {
		(yyval.lval) = (1<<(yyvsp[(1) - (3)].lval)) | (yyvsp[(3) - (3)].lval);
	}
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 414 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (4)].gen);
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 419 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_PSR;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 425 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FPCR;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 431 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 442 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (1)].gen);
		if((yyvsp[(1) - (1)].gen).name != D_EXTERN && (yyvsp[(1) - (1)].gen).name != D_STATIC) {
		}
	}
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 450 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).reg = (yyvsp[(2) - (3)].lval);
		(yyval.gen).offset = 0;
	}
    break;

  case 75:

/* Line 1455 of yacc.c  */
#line 460 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 470 "a.y"
    {
		(yyval.gen) = (yyvsp[(1) - (4)].gen);
		(yyval.gen).type = D_OREG;
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 82:

/* Line 1455 of yacc.c  */
#line 483 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_CONST;
		(yyval.gen).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 491 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_REG;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 499 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_REGREG;
		(yyval.gen).reg = (yyvsp[(2) - (5)].lval);
		(yyval.gen).offset = (yyvsp[(4) - (5)].lval);
	}
    break;

  case 85:

/* Line 1455 of yacc.c  */
#line 508 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (0 << 5);
	}
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 514 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (1 << 5);
	}
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 520 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (2 << 5);
	}
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 526 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_SHIFT;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (3 << 5);
	}
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 534 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 16)
			print("register value out of range\n");
		(yyval.lval) = (((yyvsp[(1) - (1)].lval)&15) << 8) | (1 << 4);
	}
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 540 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 32)
			print("shift value out of range\n");
		(yyval.lval) = ((yyvsp[(1) - (1)].lval)&31) << 7;
	}
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 549 "a.y"
    {
		(yyval.lval) = REGPC;
	}
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 553 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 562 "a.y"
    {
		(yyval.lval) = REGSP;
	}
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 569 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 581 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FREG;
		(yyval.gen).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 587 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_FREG;
		(yyval.gen).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 595 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).name = (yyvsp[(3) - (4)].lval);
		(yyval.gen).sym = S;
		(yyval.gen).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 603 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).name = (yyvsp[(4) - (5)].lval);
		(yyval.gen).sym = (yyvsp[(1) - (5)].sym);
		(yyval.gen).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 611 "a.y"
    {
		(yyval.gen) = nullgen;
		(yyval.gen).type = D_OREG;
		(yyval.gen).name = D_STATIC;
		(yyval.gen).sym = (yyvsp[(1) - (7)].sym);
		(yyval.gen).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 105:

/* Line 1455 of yacc.c  */
#line 620 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 106:

/* Line 1455 of yacc.c  */
#line 624 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 107:

/* Line 1455 of yacc.c  */
#line 628 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 640 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 644 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 648 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 652 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 656 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 661 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 665 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 672 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 676 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 680 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 684 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 688 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 692 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 696 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 700 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 704 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 708 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;



/* Line 1455 of yacc.c  */
#line 2740 "y.tab.c"
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



