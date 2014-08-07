/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison implementation for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
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
#define YYBISON_VERSION "2.5"

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

/* Line 268 of yacc.c  */
#line 30 "a.y"

#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and a.h re-#defines getc */
#include <libc.h>
#include "a.h"
#include "../../pkg/runtime/funcdata.h"


/* Line 268 of yacc.c  */
#line 80 "y.tab.c"

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
     LMOVW = 258,
     LMOVB = 259,
     LABS = 260,
     LLOGW = 261,
     LSHW = 262,
     LADDW = 263,
     LCMP = 264,
     LCROP = 265,
     LBRA = 266,
     LFMOV = 267,
     LFCONV = 268,
     LFCMP = 269,
     LFADD = 270,
     LFMA = 271,
     LTRAP = 272,
     LXORW = 273,
     LNOP = 274,
     LEND = 275,
     LRETT = 276,
     LWORD = 277,
     LTEXT = 278,
     LDATA = 279,
     LRETRN = 280,
     LCONST = 281,
     LSP = 282,
     LSB = 283,
     LFP = 284,
     LPC = 285,
     LCREG = 286,
     LFLUSH = 287,
     LREG = 288,
     LFREG = 289,
     LR = 290,
     LCR = 291,
     LF = 292,
     LFPSCR = 293,
     LLR = 294,
     LCTR = 295,
     LSPR = 296,
     LSPREG = 297,
     LSEG = 298,
     LMSR = 299,
     LPCDAT = 300,
     LFUNCDAT = 301,
     LSCHED = 302,
     LXLD = 303,
     LXST = 304,
     LXOP = 305,
     LXMV = 306,
     LRLWM = 307,
     LMOVMW = 308,
     LMOVEM = 309,
     LMOVFL = 310,
     LMTFSB = 311,
     LMA = 312,
     LFCONST = 313,
     LSCONST = 314,
     LNAME = 315,
     LLAB = 316,
     LVAR = 317
   };
#endif
/* Tokens.  */
#define LMOVW 258
#define LMOVB 259
#define LABS 260
#define LLOGW 261
#define LSHW 262
#define LADDW 263
#define LCMP 264
#define LCROP 265
#define LBRA 266
#define LFMOV 267
#define LFCONV 268
#define LFCMP 269
#define LFADD 270
#define LFMA 271
#define LTRAP 272
#define LXORW 273
#define LNOP 274
#define LEND 275
#define LRETT 276
#define LWORD 277
#define LTEXT 278
#define LDATA 279
#define LRETRN 280
#define LCONST 281
#define LSP 282
#define LSB 283
#define LFP 284
#define LPC 285
#define LCREG 286
#define LFLUSH 287
#define LREG 288
#define LFREG 289
#define LR 290
#define LCR 291
#define LF 292
#define LFPSCR 293
#define LLR 294
#define LCTR 295
#define LSPR 296
#define LSPREG 297
#define LSEG 298
#define LMSR 299
#define LPCDAT 300
#define LFUNCDAT 301
#define LSCHED 302
#define LXLD 303
#define LXST 304
#define LXOP 305
#define LXMV 306
#define LRLWM 307
#define LMOVMW 308
#define LMOVEM 309
#define LMOVFL 310
#define LMTFSB 311
#define LMA 312
#define LFCONST 313
#define LSCONST 314
#define LNAME 315
#define LLAB 316
#define LVAR 317




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 293 of yacc.c  */
#line 38 "a.y"

	Sym	*sym;
	vlong	lval;
	double	dval;
	char	sval[8];
	Addr	addr;



/* Line 293 of yacc.c  */
#line 250 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 262 "y.tab.c"

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
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   836

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  81
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  32
/* YYNRULES -- Number of rules.  */
#define YYNRULES  186
/* YYNRULES -- Number of states.  */
#define YYNSTATES  459

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   317

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    79,    12,     5,     2,
      77,    78,    10,     8,    76,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    73,    75,
       6,    74,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    80,     2,     2,     2,
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
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,     8,    13,    14,    19,    24,
      29,    32,    34,    37,    40,    45,    50,    55,    60,    65,
      70,    75,    80,    85,    90,    95,   100,   105,   110,   115,
     120,   125,   130,   135,   140,   147,   152,   157,   162,   169,
     174,   179,   186,   193,   200,   205,   210,   217,   222,   229,
     234,   241,   246,   251,   254,   261,   266,   271,   276,   283,
     288,   293,   298,   303,   308,   313,   318,   323,   326,   329,
     334,   338,   342,   348,   353,   358,   365,   370,   375,   382,
     389,   396,   405,   410,   415,   419,   422,   427,   432,   439,
     448,   453,   460,   465,   470,   477,   484,   493,   502,   511,
     520,   525,   530,   535,   542,   547,   554,   559,   564,   567,
     570,   574,   578,   582,   586,   589,   593,   597,   602,   607,
     610,   615,   622,   631,   638,   645,   652,   655,   660,   663,
     666,   668,   670,   672,   674,   676,   678,   680,   682,   687,
     689,   691,   696,   698,   703,   705,   710,   712,   716,   719,
     722,   725,   729,   732,   734,   739,   743,   749,   751,   756,
     761,   767,   775,   776,   778,   779,   782,   785,   787,   789,
     791,   793,   795,   798,   801,   804,   808,   810,   814,   818,
     822,   826,   830,   835,   840,   844,   848
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      82,     0,    -1,    -1,    82,    83,    -1,    -1,    71,    73,
      84,    83,    -1,    -1,    70,    73,    85,    83,    -1,    70,
      74,   112,    75,    -1,    72,    74,   112,    75,    -1,    57,
      75,    -1,    75,    -1,    86,    75,    -1,     1,    75,    -1,
      13,    88,    76,    88,    -1,    13,   106,    76,    88,    -1,
      13,   105,    76,    88,    -1,    14,    88,    76,    88,    -1,
      14,   106,    76,    88,    -1,    14,   105,    76,    88,    -1,
      22,   106,    76,    97,    -1,    22,   105,    76,    97,    -1,
      22,   102,    76,    97,    -1,    22,    97,    76,    97,    -1,
      22,    97,    76,   106,    -1,    22,    97,    76,   105,    -1,
      13,    88,    76,   106,    -1,    13,    88,    76,   105,    -1,
      14,    88,    76,   106,    -1,    14,    88,    76,   105,    -1,
      13,    97,    76,   106,    -1,    13,    97,    76,   105,    -1,
      13,    95,    76,    97,    -1,    13,    97,    76,    95,    -1,
      13,    97,    76,   103,    76,    95,    -1,    13,    95,    76,
      98,    -1,    13,   103,    76,    96,    -1,    66,   103,    76,
     111,    -1,    13,    88,    76,   103,    76,    91,    -1,    13,
      88,    76,    98,    -1,    13,    88,    76,    91,    -1,    18,
      88,    76,   104,    76,    88,    -1,    18,   103,    76,   104,
      76,    88,    -1,    18,    88,    76,   103,    76,    88,    -1,
      18,    88,    76,    88,    -1,    18,   103,    76,    88,    -1,
      16,    88,    76,   104,    76,    88,    -1,    16,    88,    76,
      88,    -1,    17,    88,    76,   104,    76,    88,    -1,    17,
      88,    76,    88,    -1,    17,   103,    76,   104,    76,    88,
      -1,    17,   103,    76,    88,    -1,    15,    88,    76,    88,
      -1,    15,    88,    -1,    67,    88,    76,   104,    76,    88,
      -1,    13,   103,    76,    88,    -1,    13,   101,    76,    88,
      -1,    20,    99,    76,    99,    -1,    20,    99,    76,   111,
      76,    99,    -1,    13,    98,    76,    98,    -1,    13,    94,
      76,    98,    -1,    13,    91,    76,    88,    -1,    13,    94,
      76,    88,    -1,    13,    89,    76,    88,    -1,    13,    88,
      76,    89,    -1,    13,    98,    76,    94,    -1,    13,    88,
      76,    94,    -1,    21,    87,    -1,    21,   106,    -1,    21,
      77,    89,    78,    -1,    21,    76,    87,    -1,    21,    76,
     106,    -1,    21,    76,    77,    89,    78,    -1,    21,    98,
      76,    87,    -1,    21,    98,    76,   106,    -1,    21,    98,
      76,    77,    89,    78,    -1,    21,   111,    76,    87,    -1,
      21,   111,    76,   106,    -1,    21,   111,    76,    77,    89,
      78,    -1,    21,   111,    76,   111,    76,    87,    -1,    21,
     111,    76,   111,    76,   106,    -1,    21,   111,    76,   111,
      76,    77,    89,    78,    -1,    27,    88,    76,   104,    -1,
      27,   103,    76,   104,    -1,    27,    88,   108,    -1,    27,
     108,    -1,    23,    97,    76,    97,    -1,    25,    97,    76,
      97,    -1,    25,    97,    76,    97,    76,    97,    -1,    26,
      97,    76,    97,    76,    97,    76,    97,    -1,    24,    97,
      76,    97,    -1,    24,    97,    76,    97,    76,    98,    -1,
      19,    88,    76,    88,    -1,    19,    88,    76,   103,    -1,
      19,    88,    76,    88,    76,    98,    -1,    19,    88,    76,
     103,    76,    98,    -1,    62,   103,    76,    88,    76,   103,
      76,    88,    -1,    62,   103,    76,    88,    76,   100,    76,
      88,    -1,    62,    88,    76,    88,    76,   103,    76,    88,
      -1,    62,    88,    76,    88,    76,   100,    76,    88,    -1,
      63,   106,    76,    88,    -1,    63,    88,    76,   106,    -1,
      58,   105,    76,    88,    -1,    58,   105,    76,   103,    76,
      88,    -1,    59,    88,    76,   105,    -1,    59,    88,    76,
     103,    76,   105,    -1,    61,   105,    76,    88,    -1,    61,
      88,    76,   105,    -1,    60,   105,    -1,    29,   108,    -1,
      29,    88,   108,    -1,    29,    97,   108,    -1,    29,    76,
      88,    -1,    29,    76,    97,    -1,    29,   103,    -1,    32,
     103,   108,    -1,    32,   101,   108,    -1,    55,   103,    76,
     103,    -1,    56,   103,    76,   106,    -1,    30,   108,    -1,
      33,   107,    76,   103,    -1,    33,   107,    76,   111,    76,
     103,    -1,    33,   107,    76,   111,    76,   103,     9,   111,
      -1,    34,   107,    11,   111,    76,   103,    -1,    34,   107,
      11,   111,    76,   101,    -1,    34,   107,    11,   111,    76,
     102,    -1,    35,   108,    -1,   111,    77,    40,    78,    -1,
      70,   109,    -1,    71,   109,    -1,   104,    -1,    90,    -1,
      92,    -1,    49,    -1,    46,    -1,    50,    -1,    54,    -1,
      52,    -1,    51,    77,   111,    78,    -1,    93,    -1,    48,
      -1,    48,    77,   111,    78,    -1,    44,    -1,    47,    77,
     111,    78,    -1,    41,    -1,    46,    77,   111,    78,    -1,
     111,    -1,   111,    76,   111,    -1,    79,   106,    -1,    79,
      69,    -1,    79,    68,    -1,    79,     9,    68,    -1,    79,
     111,    -1,    43,    -1,    45,    77,   111,    78,    -1,    77,
     104,    78,    -1,    77,   104,     8,   104,    78,    -1,   107,
      -1,   111,    77,   104,    78,    -1,   111,    77,   110,    78,
      -1,    70,   109,    77,   110,    78,    -1,    70,     6,     7,
     109,    77,    38,    78,    -1,    -1,    76,    -1,    -1,     8,
     111,    -1,     9,   111,    -1,    38,    -1,    37,    -1,    39,
      -1,    36,    -1,    72,    -1,     9,   111,    -1,     8,   111,
      -1,    80,   111,    -1,    77,   112,    78,    -1,   111,    -1,
     112,     8,   112,    -1,   112,     9,   112,    -1,   112,    10,
     112,    -1,   112,    11,   112,    -1,   112,    12,   112,    -1,
     112,     6,     6,   112,    -1,   112,     7,     7,   112,    -1,
     112,     5,   112,    -1,   112,     4,   112,    -1,   112,     3,
     112,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    66,    66,    67,    71,    70,    78,    77,    83,    88,
      94,    98,    99,   100,   106,   110,   114,   118,   122,   126,
     133,   137,   141,   145,   149,   153,   160,   164,   168,   172,
     179,   183,   190,   194,   198,   202,   206,   210,   217,   221,
     225,   235,   239,   243,   247,   251,   255,   259,   263,   267,
     271,   275,   279,   283,   290,   297,   301,   308,   312,   320,
     324,   328,   332,   336,   340,   344,   348,   357,   361,   365,
     369,   373,   377,   381,   385,   389,   393,   397,   401,   405,
     413,   421,   432,   436,   440,   444,   451,   455,   459,   463,
     467,   471,   478,   482,   486,   490,   497,   501,   505,   509,
     516,   520,   528,   532,   536,   540,   544,   548,   552,   559,
     563,   567,   571,   575,   579,   586,   590,   597,   606,   617,
     624,   628,   634,   643,   647,   651,   658,   664,   670,   678,
     686,   694,   695,   698,   706,   714,   722,   729,   735,   741,
     744,   752,   760,   766,   774,   780,   788,   796,   817,   822,
     830,   836,   843,   851,   852,   860,   867,   877,   878,   887,
     895,   903,   912,   913,   916,   919,   923,   929,   930,   931,
     934,   935,   939,   943,   947,   951,   957,   958,   962,   966,
     970,   974,   978,   982,   986,   990,   994
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'|'", "'^'", "'&'", "'<'", "'>'", "'+'",
  "'-'", "'*'", "'/'", "'%'", "LMOVW", "LMOVB", "LABS", "LLOGW", "LSHW",
  "LADDW", "LCMP", "LCROP", "LBRA", "LFMOV", "LFCONV", "LFCMP", "LFADD",
  "LFMA", "LTRAP", "LXORW", "LNOP", "LEND", "LRETT", "LWORD", "LTEXT",
  "LDATA", "LRETRN", "LCONST", "LSP", "LSB", "LFP", "LPC", "LCREG",
  "LFLUSH", "LREG", "LFREG", "LR", "LCR", "LF", "LFPSCR", "LLR", "LCTR",
  "LSPR", "LSPREG", "LSEG", "LMSR", "LPCDAT", "LFUNCDAT", "LSCHED", "LXLD",
  "LXST", "LXOP", "LXMV", "LRLWM", "LMOVMW", "LMOVEM", "LMOVFL", "LMTFSB",
  "LMA", "LFCONST", "LSCONST", "LNAME", "LLAB", "LVAR", "':'", "'='",
  "';'", "','", "'('", "')'", "'$'", "'~'", "$accept", "prog", "line",
  "$@1", "$@2", "inst", "rel", "rreg", "xlreg", "lr", "lcr", "ctr", "msr",
  "psr", "fpscr", "fpscrf", "freg", "creg", "cbit", "mask", "ximm", "fimm",
  "imm", "sreg", "regaddr", "addr", "name", "comma", "offset", "pointer",
  "con", "expr", 0
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
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,    58,    61,    59,    44,    40,    41,    36,
     126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    81,    82,    82,    84,    83,    85,    83,    83,    83,
      83,    83,    83,    83,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    87,    87,    87,
      88,    89,    89,    90,    91,    92,    93,    94,    94,    94,
      95,    96,    97,    97,    98,    98,    99,   100,   101,   101,
     102,   102,   103,   104,   104,   105,   105,   106,   106,   107,
     107,   107,   108,   108,   109,   109,   109,   110,   110,   110,
     111,   111,   111,   111,   111,   111,   112,   112,   112,   112,
     112,   112,   112,   112,   112,   112,   112
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     0,     4,     0,     4,     4,     4,
       2,     1,     2,     2,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     6,     4,     4,     4,     6,     4,
       4,     6,     6,     6,     4,     4,     6,     4,     6,     4,
       6,     4,     4,     2,     6,     4,     4,     4,     6,     4,
       4,     4,     4,     4,     4,     4,     4,     2,     2,     4,
       3,     3,     5,     4,     4,     6,     4,     4,     6,     6,
       6,     8,     4,     4,     3,     2,     4,     4,     6,     8,
       4,     6,     4,     4,     6,     6,     8,     8,     8,     8,
       4,     4,     4,     6,     4,     6,     4,     4,     2,     2,
       3,     3,     3,     3,     2,     3,     3,     4,     4,     2,
       4,     6,     8,     6,     6,     6,     2,     4,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     4,     1,
       1,     4,     1,     4,     1,     4,     1,     3,     2,     2,
       2,     3,     2,     1,     4,     3,     5,     1,     4,     4,
       5,     7,     0,     1,     0,     2,     2,     1,     1,     1,
       1,     1,     2,     2,     2,     3,     1,     3,     3,     3,
       3,     3,     4,     4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     0,     1,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   162,   162,
     162,     0,     0,     0,   162,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    11,
       3,     0,    13,     0,     0,   170,   144,   153,   142,     0,
     134,     0,   140,   133,   135,     0,   137,   136,   164,   171,
       0,     0,     0,     0,     0,   131,     0,   132,   139,     0,
       0,     0,     0,     0,     0,   130,     0,     0,   157,     0,
       0,     0,     0,    53,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   146,     0,   164,   164,     0,     0,    67,
       0,    68,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   163,   162,     0,    85,   163,   162,   162,   114,
     109,   119,   162,   162,     0,     0,     0,   126,     0,     0,
      10,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     6,     0,     4,     0,    12,   173,   172,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   176,
       0,   149,   148,   152,   174,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   152,     0,     0,     0,     0,     0,     0,   128,
     129,     0,    70,    71,     0,     0,     0,     0,     0,     0,
     150,     0,     0,     0,     0,     0,     0,     0,     0,   163,
      84,     0,   112,   113,   110,   111,   116,   115,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   164,   165,   166,     0,     0,   155,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   175,    14,    64,
      40,    66,    39,     0,    27,    26,    63,    61,    62,    60,
      32,    35,    33,     0,    31,    30,    65,    59,    56,     0,
      55,    36,    16,    15,   168,   167,   169,     0,     0,    17,
      29,    28,    19,    18,    52,    47,   130,    49,   130,    51,
     130,    44,     0,   130,    45,   130,    92,    93,    57,   146,
       0,    69,     0,    73,    74,     0,    76,    77,     0,     0,
     151,    23,    25,    24,    22,    21,    20,    86,    90,    87,
       0,    82,    83,   120,     0,     0,   117,   118,   102,     0,
       0,   104,   107,   106,     0,     0,   101,   100,    37,     0,
       7,     8,     5,     9,   154,   145,   143,   138,     0,     0,
       0,   186,   185,   184,     0,     0,   177,   178,   179,   180,
     181,     0,     0,     0,   158,   159,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    72,     0,     0,     0,   127,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   160,   156,   182,   183,   134,    38,    34,     0,    46,
      48,    50,    43,    41,    42,    94,    95,    58,    75,    78,
       0,    79,    80,    91,    88,     0,   121,     0,   124,   125,
     123,   103,   105,     0,     0,     0,     0,     0,    54,     0,
     141,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     161,    81,    89,   122,    99,    98,   147,    97,    96
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,    40,   235,   233,    41,    99,    63,    64,    65,
      66,    67,    68,    69,    70,   281,    71,    72,    92,   433,
      73,   105,    74,    75,    76,   162,    78,   115,   157,   288,
     159,   160
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -178
static const yytype_int16 yypact[] =
{
    -178,   471,  -178,   -66,   567,   640,    32,    32,   -26,   -26,
      32,   756,   626,    24,    55,    55,    55,    55,   -14,    73,
     -60,   -54,   743,   743,   -60,   -44,   -44,   -32,   -23,    32,
     -23,    -1,   -26,   644,   -44,    32,    35,    17,   -10,  -178,
    -178,    48,  -178,   756,   756,  -178,  -178,  -178,  -178,     4,
      63,    88,  -178,  -178,  -178,    94,  -178,  -178,   130,  -178,
     710,   508,   756,   101,   114,  -178,   117,  -178,  -178,   123,
     128,   140,   155,   166,   170,  -178,   172,   177,  -178,   174,
     181,   190,   192,   193,   202,   756,   203,   206,   208,   220,
     221,   756,   224,  -178,    63,   130,   175,   700,    42,  -178,
     229,  -178,   143,     6,   232,   235,   238,   240,   245,   246,
     255,   257,  -178,   259,   262,  -178,   285,   -60,   -60,  -178,
    -178,  -178,   -60,   -60,   265,   268,   306,  -178,   270,   271,
    -178,    32,   272,   301,  -178,   302,   315,   316,   317,   319,
     320,   321,   324,  -178,   756,  -178,   756,  -178,  -178,  -178,
     756,   756,   756,   756,   394,   756,   756,   328,    15,  -178,
     347,  -178,  -178,   174,  -178,   614,    32,    32,    86,    26,
     665,   258,    32,    -9,    32,    32,    18,   640,    32,    32,
      32,    32,  -178,    32,    32,   -26,    32,   -26,   756,   328,
    -178,    42,  -178,  -178,   330,   332,   714,   725,   157,   340,
    -178,   696,    55,    55,    55,    55,    55,    55,    55,    32,
    -178,    32,  -178,  -178,  -178,  -178,  -178,  -178,   390,   106,
     756,   -44,   743,   -26,    49,   -23,    32,    32,    32,   743,
      32,   756,    32,   534,   357,   534,   377,   335,   337,   338,
     339,   175,  -178,  -178,   106,    32,  -178,   756,   756,   756,
     406,   411,   756,   756,   756,   756,   756,  -178,  -178,  -178,
    -178,  -178,  -178,   343,  -178,  -178,  -178,  -178,  -178,  -178,
    -178,  -178,  -178,   351,  -178,  -178,  -178,  -178,  -178,   352,
    -178,  -178,  -178,  -178,  -178,  -178,  -178,   350,   353,  -178,
    -178,  -178,  -178,  -178,  -178,  -178,   361,  -178,   362,  -178,
     363,  -178,   366,   369,  -178,   370,   371,   372,  -178,   373,
     375,  -178,    42,  -178,  -178,    42,  -178,  -178,   184,   376,
    -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178,   374,   379,
     380,  -178,  -178,  -178,   381,   382,  -178,  -178,  -178,   383,
     388,  -178,  -178,  -178,   389,   392,  -178,  -178,  -178,   397,
    -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178,   398,   396,
     399,   620,   513,   147,   756,   756,   216,   216,  -178,  -178,
    -178,   405,   418,   756,  -178,  -178,    32,    32,    32,    32,
      32,    32,    59,    59,   756,  -178,   403,   404,   739,  -178,
      59,    55,    55,   -44,   420,    32,   -23,   390,   390,    32,
     438,  -178,  -178,   283,   283,  -178,  -178,  -178,   424,  -178,
    -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178,
      42,  -178,  -178,  -178,  -178,   431,   499,   334,  -178,  -178,
    -178,  -178,  -178,   436,   439,   460,   463,   464,  -178,   467,
    -178,   484,    55,   756,   721,    32,    32,   756,    32,    32,
    -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178,  -178
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -178,  -178,     8,  -178,  -178,  -178,   -90,    -5,   -76,  -178,
    -157,  -178,  -178,  -153,  -160,  -178,    69,    40,  -177,   167,
     -15,   176,   116,   104,    82,    33,   241,   127,   -75,   327,
      36,    71
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      80,    83,    84,    86,    88,    90,   122,   192,   260,    42,
     272,   308,   261,   113,   117,   199,   112,    47,   276,    49,
     189,   190,   195,   245,   133,    61,   135,   137,   139,    47,
     142,    49,    43,    44,    47,    85,    49,    77,    82,   279,
      79,    79,    47,   130,    49,   101,   107,    93,   102,    79,
      43,    44,   100,    85,   131,   284,   285,   286,   125,   125,
      45,    47,   112,    49,   146,    85,   140,    46,    48,    79,
      48,    51,    94,    51,   200,    47,   131,    49,    45,   148,
     149,   150,   104,   108,   109,   110,   111,    81,   118,   259,
     145,    53,    54,   246,    58,   106,    59,   163,   164,    48,
      46,    60,    51,   103,    62,    94,   313,   316,   143,   144,
     132,   212,   134,   136,    59,   310,    47,    48,    49,    91,
      51,   182,    62,   147,    87,    89,   131,    46,    85,    47,
     193,    49,    94,   194,   114,   119,   154,   123,   155,   156,
     151,   128,   129,   284,   285,   286,   120,   121,   138,   116,
     141,   127,    85,   250,   251,   252,   253,   254,   255,   256,
     258,   266,   267,   268,   158,   152,   358,   278,   280,   282,
     283,   153,   289,   292,   293,   294,   295,   165,   297,   299,
     301,   304,   306,   155,   156,   213,   237,   238,   239,   240,
     166,   242,   243,   167,   284,   285,   286,   319,   265,   168,
      47,    79,    49,   275,   169,   262,    79,   417,   269,   271,
     291,   277,   407,    79,   406,   234,   170,   236,   338,   197,
     198,   343,   344,   345,   309,   347,   254,   255,   256,   314,
     317,   171,   194,   318,   323,   158,   386,    79,   270,   387,
     210,   350,   172,   352,   214,   215,   173,   264,   174,   216,
     217,   176,   274,   175,   334,   337,   335,   177,    79,   290,
     388,   198,   346,   124,   126,    79,   178,   348,   179,   180,
     321,   324,   325,   326,   327,   328,   329,   330,   181,   183,
     287,   263,   184,   322,   185,   296,   273,   298,   300,   303,
     305,   252,   253,   254,   255,   256,   186,   187,   421,    46,
     188,   302,   287,   307,    94,   196,   341,   342,   201,    55,
      56,   202,    57,   331,   203,   332,   204,   220,   361,   362,
     363,   205,   206,   366,   367,   368,   369,   370,    47,    48,
      49,   207,    51,   208,   333,   209,   349,   336,   211,   339,
     340,   218,    43,   444,   441,   219,   221,   222,   223,   360,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
      45,   409,   410,   411,   412,   413,   414,   224,   225,   428,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     431,   226,   227,   228,   438,   229,   230,   231,    43,    44,
     232,   241,   200,   161,    58,   244,    59,   198,   320,   408,
     311,    91,   364,   354,    62,   355,   356,   357,   365,   371,
      93,   422,   415,   416,   194,   257,    45,   372,   374,   373,
     423,   375,   351,   435,   435,   403,   404,   376,   377,   378,
     454,   455,   379,   457,   458,   380,   381,   382,   383,   384,
     390,   405,   353,   385,   389,   391,   392,   393,   394,   395,
     424,   425,    59,   163,   396,   397,    52,    91,   398,    85,
      62,     2,     3,   399,   401,   400,   439,   402,   432,   453,
     149,   418,   419,   456,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,   427,
      19,    20,   440,    21,    22,    23,    24,   442,   443,   426,
     430,   452,   445,   434,   437,   446,    43,    44,   249,   250,
     251,   252,   253,   254,   255,   256,    25,    26,    27,    28,
      29,    30,    31,    32,    33,     3,   447,    34,    35,   448,
     449,    36,    37,    38,    45,   450,    39,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,   451,    19,    20,   436,    21,    22,    23,    24,
     429,   359,     0,     0,     0,    43,    44,   161,    58,     0,
      59,     0,     0,     0,     0,    91,     0,     0,    62,    25,
      26,    27,    28,    29,    30,    31,    32,    33,     0,     0,
      34,    35,     0,    45,    36,    37,    38,     0,    46,    39,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
       0,    57,    43,    44,   248,   249,   250,   251,   252,   253,
     254,   255,   256,     0,    43,    44,     0,    58,     0,    59,
       0,     0,     0,     0,    60,     0,    61,    62,    43,    44,
      45,     0,    43,    44,     0,    46,     0,    47,     0,    49,
      50,     0,    45,    53,    54,    55,    56,    46,    57,     0,
       0,     0,    94,    43,    44,     0,    45,     0,     0,     0,
      45,     0,     0,    47,    58,    49,    59,    47,     0,    49,
       0,    60,     0,    85,    62,     0,    95,    96,    59,     0,
       0,    45,    97,    98,    43,    44,    62,     0,    43,    44,
      58,     0,    59,    52,    58,     0,    59,    60,    43,    44,
      62,    91,    43,    44,    62,     0,     0,     0,     0,    43,
      44,     0,    45,    43,    44,    58,    45,    59,     0,     0,
      48,     0,    60,    51,    85,    62,    45,    43,    44,     0,
      45,    43,    44,    47,     0,    49,     0,    45,     0,     0,
       0,    45,     0,     0,    43,    44,    58,     0,    59,     0,
      95,    96,    59,    60,     0,    45,    62,   191,     0,    45,
      62,     0,    59,     0,    95,    96,    59,    91,     0,   320,
      62,   312,    45,    59,    62,    95,    96,    59,    91,     0,
       0,    62,   315,     0,     0,    62,     0,     0,     0,    95,
      96,    59,     0,    58,     0,    59,   420,     0,     0,    62,
      91,     0,     0,    62,     0,     0,     0,     0,    59,     0,
       0,     0,     0,    91,     0,     0,    62
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-178))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       5,     6,     7,     8,     9,    10,    21,    97,   165,    75,
     170,   188,   165,    18,    19,     9,    76,    43,   171,    45,
      95,    96,    98,     8,    29,    79,    31,    32,    33,    43,
      35,    45,     8,     9,    43,    79,    45,     4,     5,    48,
       4,     5,    43,    75,    45,    12,    13,    11,    12,    13,
       8,     9,    12,    79,    77,    37,    38,    39,    22,    23,
      36,    43,    76,    45,    74,    79,    33,    41,    44,    33,
      44,    47,    46,    47,    68,    43,    77,    45,    36,    43,
      44,    77,    13,    14,    15,    16,    17,     5,    19,   165,
      73,    49,    50,    78,    70,    13,    72,    61,    62,    44,
      41,    77,    47,    79,    80,    46,   196,   197,    73,    74,
      28,   116,    30,    31,    72,   191,    43,    44,    45,    77,
      47,    85,    80,    75,     8,     9,    77,    41,    79,    43,
      97,    45,    46,    97,    18,    19,     6,    21,     8,     9,
      77,    25,    26,    37,    38,    39,    19,    20,    32,    76,
      34,    24,    79,     6,     7,     8,     9,    10,    11,    12,
     165,   166,   167,   168,    60,    77,   241,   172,   173,   174,
     175,    77,   177,   178,   179,   180,   181,    76,   183,   184,
     185,   186,   187,     8,     9,   116,   150,   151,   152,   153,
      76,   155,   156,    76,    37,    38,    39,    40,   165,    76,
      43,   165,    45,   170,    76,   165,   170,   384,   168,   169,
     177,   171,   372,   177,   371,   144,    76,   146,   223,    76,
      77,   226,   227,   228,   188,   230,    10,    11,    12,   196,
     197,    76,   196,   197,   201,   131,   312,   201,   169,   315,
     113,   233,    76,   235,   117,   118,    76,   165,    76,   122,
     123,    77,   170,    76,   218,   222,   220,    76,   222,   177,
      76,    77,   229,    22,    23,   229,    76,   231,    76,    76,
     201,   202,   203,   204,   205,   206,   207,   208,    76,    76,
     176,   165,    76,   201,    76,   181,   170,   183,   184,   185,
     186,     8,     9,    10,    11,    12,    76,    76,   388,    41,
      76,   185,   198,   187,    46,    76,   224,   225,    76,    51,
      52,    76,    54,   209,    76,   211,    76,    11,   247,   248,
     249,    76,    76,   252,   253,   254,   255,   256,    43,    44,
      45,    76,    47,    76,   218,    76,   232,   221,    76,   223,
     224,    76,     8,     9,   420,    77,    76,    76,    76,   245,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      36,   376,   377,   378,   379,   380,   381,    76,    76,   394,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
     395,    76,    76,    76,   399,    76,    76,    76,     8,     9,
      76,     7,    68,    69,    70,    77,    72,    77,    68,   373,
      78,    77,     6,    78,    80,    78,    78,    78,     7,    76,
     384,   388,   382,   383,   388,    78,    36,    76,    78,    77,
     390,    78,    75,   397,   398,   364,   365,    76,    76,    76,
     445,   446,    76,   448,   449,    76,    76,    76,    76,    76,
      76,    46,    75,    78,    78,    76,    76,    76,    76,    76,
     391,   392,    72,   427,    76,    76,    48,    77,    76,    79,
      80,     0,     1,    76,    78,    77,    38,    78,   396,   443,
     444,    78,    78,   447,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    79,
      29,    30,    78,    32,    33,    34,    35,    76,     9,   393,
     394,   442,    76,   397,   398,    76,     8,     9,     5,     6,
       7,     8,     9,    10,    11,    12,    55,    56,    57,    58,
      59,    60,    61,    62,    63,     1,    76,    66,    67,    76,
      76,    70,    71,    72,    36,    78,    75,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    78,    29,    30,   398,    32,    33,    34,    35,
     394,   244,    -1,    -1,    -1,     8,     9,    69,    70,    -1,
      72,    -1,    -1,    -1,    -1,    77,    -1,    -1,    80,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    -1,    -1,
      66,    67,    -1,    36,    70,    71,    72,    -1,    41,    75,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      -1,    54,     8,     9,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    -1,     8,     9,    -1,    70,    -1,    72,
      -1,    -1,    -1,    -1,    77,    -1,    79,    80,     8,     9,
      36,    -1,     8,     9,    -1,    41,    -1,    43,    -1,    45,
      46,    -1,    36,    49,    50,    51,    52,    41,    54,    -1,
      -1,    -1,    46,     8,     9,    -1,    36,    -1,    -1,    -1,
      36,    -1,    -1,    43,    70,    45,    72,    43,    -1,    45,
      -1,    77,    -1,    79,    80,    -1,    70,    71,    72,    -1,
      -1,    36,    76,    77,     8,     9,    80,    -1,     8,     9,
      70,    -1,    72,    48,    70,    -1,    72,    77,     8,     9,
      80,    77,     8,     9,    80,    -1,    -1,    -1,    -1,     8,
       9,    -1,    36,     8,     9,    70,    36,    72,    -1,    -1,
      44,    -1,    77,    47,    79,    80,    36,     8,     9,    -1,
      36,     8,     9,    43,    -1,    45,    -1,    36,    -1,    -1,
      -1,    36,    -1,    -1,     8,     9,    70,    -1,    72,    -1,
      70,    71,    72,    77,    -1,    36,    80,    77,    -1,    36,
      80,    -1,    72,    -1,    70,    71,    72,    77,    -1,    68,
      80,    77,    36,    72,    80,    70,    71,    72,    77,    -1,
      -1,    80,    77,    -1,    -1,    80,    -1,    -1,    -1,    70,
      71,    72,    -1,    70,    -1,    72,    77,    -1,    -1,    80,
      77,    -1,    -1,    80,    -1,    -1,    -1,    -1,    72,    -1,
      -1,    -1,    -1,    77,    -1,    -1,    80
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    82,     0,     1,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    29,
      30,    32,    33,    34,    35,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    66,    67,    70,    71,    72,    75,
      83,    86,    75,     8,     9,    36,    41,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    54,    70,    72,
      77,    79,    80,    88,    89,    90,    91,    92,    93,    94,
      95,    97,    98,   101,   103,   104,   105,   106,   107,   111,
      88,   105,   106,    88,    88,    79,    88,   103,    88,   103,
      88,    77,    99,   111,    46,    70,    71,    76,    77,    87,
      98,   106,   111,    79,    97,   102,   105,   106,    97,    97,
      97,    97,    76,    88,   103,   108,    76,    88,    97,   103,
     108,   108,   101,   103,   107,   111,   107,   108,   103,   103,
      75,    77,   105,    88,   105,    88,   105,    88,   103,    88,
     106,   103,    88,    73,    74,    73,    74,    75,   111,   111,
      77,    77,    77,    77,     6,     8,     9,   109,   104,   111,
     112,    69,   106,   111,   111,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    76,    77,    76,    76,    76,
      76,    76,   111,    76,    76,    76,    76,    76,    76,   109,
     109,    77,    87,   106,   111,    89,    76,    76,    77,     9,
      68,    76,    76,    76,    76,    76,    76,    76,    76,    76,
     108,    76,    88,    97,   108,   108,   108,   108,    76,    77,
      11,    76,    76,    76,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    85,   112,    84,   112,   111,   111,   111,
     111,     7,   111,   111,    77,     8,    78,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    78,    88,    89,
      91,    94,    98,   103,   105,   106,    88,    88,    88,    98,
      97,    98,    95,   103,   105,   106,    94,    98,    88,    48,
      88,    96,    88,    88,    37,    38,    39,   104,   110,    88,
     105,   106,    88,    88,    88,    88,   104,    88,   104,    88,
     104,    88,   103,   104,    88,   104,    88,   103,    99,   111,
      89,    78,    77,    87,   106,    77,    87,   106,   111,    40,
      68,    97,   105,   106,    97,    97,    97,    97,    97,    97,
      97,   104,   104,   103,   111,   111,   103,   106,    88,   103,
     103,   105,   105,    88,    88,    88,   106,    88,   111,   104,
      83,    75,    83,    75,    78,    78,    78,    78,   109,   110,
     104,   112,   112,   112,     6,     7,   112,   112,   112,   112,
     112,    76,    76,    77,    78,    78,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    78,    89,    89,    76,    78,
      76,    76,    76,    76,    76,    76,    76,    76,    76,    76,
      77,    78,    78,   112,   112,    46,    91,    95,   111,    88,
      88,    88,    88,    88,    88,    98,    98,    99,    78,    78,
      77,    87,   106,    98,    97,    97,   103,    79,   101,   102,
     103,    88,   105,   100,   103,   111,   100,   103,    88,    38,
      78,    89,    76,     9,     9,    76,    76,    76,    76,    76,
      78,    78,    97,   111,    88,    88,   111,    88,    88
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

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
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
  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = 0;
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
                yysize1 = yysize + yytnamerr (0, yytname[yyx]);
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
        case 4:

/* Line 1806 of yacc.c  */
#line 71 "a.y"
    {
		if((yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->name);
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 6:

/* Line 1806 of yacc.c  */
#line 78 "a.y"
    {
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 8:

/* Line 1806 of yacc.c  */
#line 84 "a.y"
    {
		(yyvsp[(1) - (4)].sym)->type = LVAR;
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 89 "a.y"
    {
		if((yyvsp[(1) - (4)].sym)->value != (yyvsp[(3) - (4)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (4)].sym)->name);
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 95 "a.y"
    {
		nosched = (yyvsp[(1) - (2)].lval);
	}
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 107 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 15:

/* Line 1806 of yacc.c  */
#line 111 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 16:

/* Line 1806 of yacc.c  */
#line 115 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 119 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 123 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 127 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 134 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 138 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 142 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 146 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 150 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 154 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 161 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 165 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 169 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 173 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 180 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 184 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 191 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 195 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 199 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 203 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 36:

/* Line 1806 of yacc.c  */
#line 207 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 211 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].lval), &nullgen);
	}
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 218 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 39:

/* Line 1806 of yacc.c  */
#line 222 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 226 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 236 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 240 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 244 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 248 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 252 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 256 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 260 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 264 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 268 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 272 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 276 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 280 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 284 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), NREG, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 54:

/* Line 1806 of yacc.c  */
#line 291 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 298 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 302 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 309 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].addr).reg, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 58:

/* Line 1806 of yacc.c  */
#line 313 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 321 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 325 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 329 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 333 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 337 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 341 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 345 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 349 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 358 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 362 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &(yyvsp[(2) - (2)].addr));
	}
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 366 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &nullgen, NREG, &(yyvsp[(3) - (4)].addr));
	}
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 370 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 71:

/* Line 1806 of yacc.c  */
#line 374 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 72:

/* Line 1806 of yacc.c  */
#line 378 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), &nullgen, NREG, &(yyvsp[(4) - (5)].addr));
	}
    break;

  case 73:

/* Line 1806 of yacc.c  */
#line 382 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 74:

/* Line 1806 of yacc.c  */
#line 386 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 75:

/* Line 1806 of yacc.c  */
#line 390 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(5) - (6)].addr));
	}
    break;

  case 76:

/* Line 1806 of yacc.c  */
#line 394 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &nullgen, (yyvsp[(2) - (4)].lval), &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 77:

/* Line 1806 of yacc.c  */
#line 398 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &nullgen, (yyvsp[(2) - (4)].lval), &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 78:

/* Line 1806 of yacc.c  */
#line 402 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &nullgen, (yyvsp[(2) - (6)].lval), &(yyvsp[(5) - (6)].addr));
	}
    break;

  case 79:

/* Line 1806 of yacc.c  */
#line 406 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(2) - (6)].lval);
		outcode((yyvsp[(1) - (6)].lval), &g, (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 80:

/* Line 1806 of yacc.c  */
#line 414 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(2) - (6)].lval);
		outcode((yyvsp[(1) - (6)].lval), &g, (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 81:

/* Line 1806 of yacc.c  */
#line 422 "a.y"
    {
		Addr g;
		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(2) - (8)].lval);
		outcode((yyvsp[(1) - (8)].lval), &g, (yyvsp[(4) - (8)].lval), &(yyvsp[(7) - (8)].addr));
	}
    break;

  case 82:

/* Line 1806 of yacc.c  */
#line 433 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].lval), &nullgen);
	}
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 437 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), (yyvsp[(4) - (4)].lval), &nullgen);
	}
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 441 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 445 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 452 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 456 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 460 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].addr).reg, &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 464 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 90:

/* Line 1806 of yacc.c  */
#line 468 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 472 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(4) - (6)].addr));
	}
    break;

  case 92:

/* Line 1806 of yacc.c  */
#line 479 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 93:

/* Line 1806 of yacc.c  */
#line 483 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 487 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(4) - (6)].addr));
	}
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 491 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(4) - (6)].addr));
	}
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 498 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 502 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 506 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 510 "a.y"
    {
		outgcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].addr).reg, &(yyvsp[(6) - (8)].addr), &(yyvsp[(8) - (8)].addr));
	}
    break;

  case 100:

/* Line 1806 of yacc.c  */
#line 517 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 521 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 102:

/* Line 1806 of yacc.c  */
#line 529 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 103:

/* Line 1806 of yacc.c  */
#line 533 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 104:

/* Line 1806 of yacc.c  */
#line 537 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 105:

/* Line 1806 of yacc.c  */
#line 541 "a.y"
    {
		outgcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), NREG, &(yyvsp[(4) - (6)].addr), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 106:

/* Line 1806 of yacc.c  */
#line 545 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 107:

/* Line 1806 of yacc.c  */
#line 549 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 108:

/* Line 1806 of yacc.c  */
#line 553 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), NREG, &nullgen);
	}
    break;

  case 109:

/* Line 1806 of yacc.c  */
#line 560 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 110:

/* Line 1806 of yacc.c  */
#line 564 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 111:

/* Line 1806 of yacc.c  */
#line 568 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 112:

/* Line 1806 of yacc.c  */
#line 572 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 113:

/* Line 1806 of yacc.c  */
#line 576 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 114:

/* Line 1806 of yacc.c  */
#line 580 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &(yyvsp[(2) - (2)].addr), NREG, &nullgen);
	}
    break;

  case 115:

/* Line 1806 of yacc.c  */
#line 587 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 116:

/* Line 1806 of yacc.c  */
#line 591 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), &(yyvsp[(2) - (3)].addr), NREG, &nullgen);
	}
    break;

  case 117:

/* Line 1806 of yacc.c  */
#line 598 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != D_CONST || (yyvsp[(4) - (4)].addr).type != D_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 118:

/* Line 1806 of yacc.c  */
#line 607 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != D_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(4) - (4)].addr).type != D_EXTERN && (yyvsp[(4) - (4)].addr).type != D_STATIC && (yyvsp[(4) - (4)].addr).type != D_OREG)
			yyerror("value for FUNCDATA must be symbol reference");
 		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 119:

/* Line 1806 of yacc.c  */
#line 618 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 120:

/* Line 1806 of yacc.c  */
#line 625 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 121:

/* Line 1806 of yacc.c  */
#line 629 "a.y"
    {
		(yyvsp[(6) - (6)].addr).offset &= 0xffffffffull;
		(yyvsp[(6) - (6)].addr).offset |= (vlong)ArgsSizeUnknown << 32;
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 635 "a.y"
    {
		(yyvsp[(6) - (8)].addr).offset &= 0xffffffffull;
		(yyvsp[(6) - (8)].addr).offset |= ((yyvsp[(8) - (8)].lval) & 0xffffffffull) << 32;
		outcode((yyvsp[(1) - (8)].lval), &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].lval), &(yyvsp[(6) - (8)].addr));
	}
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 644 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 124:

/* Line 1806 of yacc.c  */
#line 648 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 125:

/* Line 1806 of yacc.c  */
#line 652 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 126:

/* Line 1806 of yacc.c  */
#line 659 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 127:

/* Line 1806 of yacc.c  */
#line 665 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 128:

/* Line 1806 of yacc.c  */
#line 671 "a.y"
    {
		(yyval.addr) = nullgen;
		if(pass == 2)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->name);
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 129:

/* Line 1806 of yacc.c  */
#line 679 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 130:

/* Line 1806 of yacc.c  */
#line 687 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 133:

/* Line 1806 of yacc.c  */
#line 699 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SPR;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 134:

/* Line 1806 of yacc.c  */
#line 707 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CREG;
		(yyval.addr).reg = NREG;	/* whole register */
	}
    break;

  case 135:

/* Line 1806 of yacc.c  */
#line 715 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SPR;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 723 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_MSR;
	}
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 730 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SPR;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 138:

/* Line 1806 of yacc.c  */
#line 736 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = (yyvsp[(1) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 140:

/* Line 1806 of yacc.c  */
#line 745 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FPSCR;
		(yyval.addr).reg = NREG;
	}
    break;

  case 141:

/* Line 1806 of yacc.c  */
#line 753 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FPSCR;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 142:

/* Line 1806 of yacc.c  */
#line 761 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FREG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 143:

/* Line 1806 of yacc.c  */
#line 767 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 144:

/* Line 1806 of yacc.c  */
#line 775 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CREG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 145:

/* Line 1806 of yacc.c  */
#line 781 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 789 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 147:

/* Line 1806 of yacc.c  */
#line 797 "a.y"
    {
		int mb, me;
		uint32 v;

		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		mb = (yyvsp[(1) - (3)].lval);
		me = (yyvsp[(3) - (3)].lval);
		if(mb < 0 || mb > 31 || me < 0 || me > 31){
			yyerror("illegal mask start/end value(s)");
			mb = me = 0;
		}
		if(mb <= me)
			v = ((uint32)~0L>>mb) & (~0L<<(31-me));
		else
			v = ~(((uint32)~0L>>(me+1)) & (~0L<<(31-(mb-1))));
		(yyval.addr).offset = v;
	}
    break;

  case 148:

/* Line 1806 of yacc.c  */
#line 818 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = D_CONST;
	}
    break;

  case 149:

/* Line 1806 of yacc.c  */
#line 823 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 150:

/* Line 1806 of yacc.c  */
#line 831 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 151:

/* Line 1806 of yacc.c  */
#line 837 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 152:

/* Line 1806 of yacc.c  */
#line 844 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 154:

/* Line 1806 of yacc.c  */
#line 853 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 155:

/* Line 1806 of yacc.c  */
#line 861 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 156:

/* Line 1806 of yacc.c  */
#line 868 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(2) - (5)].lval);
		(yyval.addr).scale = (yyvsp[(4) - (5)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 158:

/* Line 1806 of yacc.c  */
#line 879 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 159:

/* Line 1806 of yacc.c  */
#line 888 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = (yyvsp[(3) - (4)].lval);
		(yyval.addr).sym = nil;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 160:

/* Line 1806 of yacc.c  */
#line 896 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 161:

/* Line 1806 of yacc.c  */
#line 904 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = D_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 164:

/* Line 1806 of yacc.c  */
#line 916 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 165:

/* Line 1806 of yacc.c  */
#line 920 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 166:

/* Line 1806 of yacc.c  */
#line 924 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 171:

/* Line 1806 of yacc.c  */
#line 936 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 172:

/* Line 1806 of yacc.c  */
#line 940 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 173:

/* Line 1806 of yacc.c  */
#line 944 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 174:

/* Line 1806 of yacc.c  */
#line 948 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 175:

/* Line 1806 of yacc.c  */
#line 952 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 177:

/* Line 1806 of yacc.c  */
#line 959 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 178:

/* Line 1806 of yacc.c  */
#line 963 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 179:

/* Line 1806 of yacc.c  */
#line 967 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 180:

/* Line 1806 of yacc.c  */
#line 971 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 181:

/* Line 1806 of yacc.c  */
#line 975 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 182:

/* Line 1806 of yacc.c  */
#line 979 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 183:

/* Line 1806 of yacc.c  */
#line 983 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 184:

/* Line 1806 of yacc.c  */
#line 987 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 185:

/* Line 1806 of yacc.c  */
#line 991 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 186:

/* Line 1806 of yacc.c  */
#line 995 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;



/* Line 1806 of yacc.c  */
#line 3566 "y.tab.c"
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
        char const *yymsgp = YY_("syntax error");
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



