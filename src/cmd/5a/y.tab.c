/* A Bison parser, made by GNU Bison 2.7.12-4996.  */

/* Bison implementation for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2013 Free Software Foundation, Inc.
   
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
#define YYBISON_VERSION "2.7.12-4996"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
/* Line 371 of yacc.c  */
#line 31 "a.y"

#include <u.h>
#include <stdio.h>	/* if we don't, bison will, and a.h re-#defines getc */
#include <libc.h>
#include "a.h"
#include "../../runtime/funcdata.h"

/* Line 371 of yacc.c  */
#line 76 "y.tab.c"

# ifndef YY_NULL
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULL nullptr
#  else
#   define YY_NULL 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
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
     LTYPEG = 272,
     LTYPEH = 273,
     LTYPEI = 274,
     LTYPEJ = 275,
     LTYPEK = 276,
     LTYPEL = 277,
     LTYPEM = 278,
     LTYPEN = 279,
     LTYPEBX = 280,
     LTYPEPLD = 281,
     LCONST = 282,
     LSP = 283,
     LSB = 284,
     LFP = 285,
     LPC = 286,
     LTYPEX = 287,
     LTYPEPC = 288,
     LTYPEF = 289,
     LR = 290,
     LREG = 291,
     LF = 292,
     LFREG = 293,
     LC = 294,
     LCREG = 295,
     LPSR = 296,
     LFCR = 297,
     LCOND = 298,
     LS = 299,
     LAT = 300,
     LFCONST = 301,
     LSCONST = 302,
     LNAME = 303,
     LLAB = 304,
     LVAR = 305
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
#define LTYPEG 272
#define LTYPEH 273
#define LTYPEI 274
#define LTYPEJ 275
#define LTYPEK 276
#define LTYPEL 277
#define LTYPEM 278
#define LTYPEN 279
#define LTYPEBX 280
#define LTYPEPLD 281
#define LCONST 282
#define LSP 283
#define LSB 284
#define LFP 285
#define LPC 286
#define LTYPEX 287
#define LTYPEPC 288
#define LTYPEF 289
#define LR 290
#define LREG 291
#define LF 292
#define LFREG 293
#define LC 294
#define LCREG 295
#define LPSR 296
#define LFCR 297
#define LCOND 298
#define LS 299
#define LAT 300
#define LFCONST 301
#define LSCONST 302
#define LNAME 303
#define LLAB 304
#define LVAR 305



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{
/* Line 387 of yacc.c  */
#line 39 "a.y"

	Sym	*sym;
	int32	lval;
	double	dval;
	char	sval[8];
	Addr	addr;


/* Line 387 of yacc.c  */
#line 228 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;

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

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

/* Line 390 of yacc.c  */
#line 256 "y.tab.c"

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
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef __attribute__
/* This feature is available in gcc versions 2.5 and later.  */
# if (! defined __GNUC__ || __GNUC__ < 2 \
      || (__GNUC__ == 2 && __GNUC_MINOR__ < 5))
#  define __attribute__(Spec) /* empty */
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif


/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(N) (N)
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
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
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
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (YYID (0))
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   609

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  71
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  35
/* YYNRULES -- Number of rules.  */
#define YYNRULES  133
/* YYNRULES -- Number of states.  */
#define YYNSTATES  339

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   305

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    69,    12,     5,     2,
      67,    68,    10,     8,    64,     9,     2,    11,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    61,    63,
       6,    62,     7,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    65,     2,    66,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     3,     2,    70,     2,     2,     2,
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
      55,    56,    57,    58,    59,    60
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     5,     9,    10,    15,    16,    21,
      26,    31,    33,    36,    39,    47,    54,    60,    66,    72,
      77,    82,    86,    90,    95,   102,   110,   118,   126,   133,
     140,   144,   149,   156,   165,   172,   177,   181,   187,   193,
     201,   208,   221,   229,   239,   242,   247,   252,   255,   256,
     259,   262,   263,   266,   271,   274,   277,   280,   283,   288,
     291,   293,   296,   300,   302,   306,   310,   312,   314,   316,
     321,   323,   325,   327,   329,   331,   333,   335,   339,   341,
     346,   348,   353,   355,   357,   359,   361,   364,   366,   372,
     377,   382,   387,   392,   394,   396,   398,   400,   405,   407,
     409,   411,   416,   418,   420,   422,   427,   432,   438,   446,
     447,   450,   453,   455,   457,   459,   461,   463,   466,   469,
     472,   476,   477,   480,   482,   486,   490,   494,   498,   502,
     507,   512,   516,   520
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      72,     0,    -1,    -1,    -1,    72,    73,    74,    -1,    -1,
      59,    61,    75,    74,    -1,    -1,    58,    61,    76,    74,
      -1,    58,    62,   105,    63,    -1,    60,    62,   105,    63,
      -1,    63,    -1,    77,    63,    -1,     1,    63,    -1,    13,
      78,    89,    64,    96,    64,    91,    -1,    13,    78,    89,
      64,    96,    64,    -1,    13,    78,    89,    64,    91,    -1,
      14,    78,    89,    64,    91,    -1,    15,    78,    84,    64,
      84,    -1,    16,    78,    79,    80,    -1,    16,    78,    79,
      85,    -1,    35,    79,    86,    -1,    17,    79,    80,    -1,
      18,    78,    79,    84,    -1,    19,    78,    89,    64,    96,
      79,    -1,    20,    78,    87,    64,    65,    83,    66,    -1,
      20,    78,    65,    83,    66,    64,    87,    -1,    21,    78,
      91,    64,    86,    64,    91,    -1,    21,    78,    91,    64,
      86,    79,    -1,    21,    78,    79,    86,    64,    91,    -1,
      22,    78,    79,    -1,    23,   100,    64,    90,    -1,    23,
     100,    64,   103,    64,    90,    -1,    23,   100,    64,   103,
      64,    90,     9,   103,    -1,    24,   100,    11,   103,    64,
      81,    -1,    25,    78,    91,    79,    -1,    28,    79,    81,
      -1,    29,    78,    99,    64,    99,    -1,    31,    78,    98,
      64,    99,    -1,    31,    78,    98,    64,    48,    64,    99,
      -1,    32,    78,    99,    64,    99,    79,    -1,    30,    78,
     103,    64,   105,    64,    96,    64,    97,    64,    97,   104,
      -1,    33,    78,    91,    64,    91,    64,    92,    -1,    34,
      78,    91,    64,    91,    64,    91,    64,    96,    -1,    36,
      88,    -1,    43,    84,    64,    84,    -1,    44,    84,    64,
      84,    -1,    26,    79,    -1,    -1,    78,    53,    -1,    78,
      54,    -1,    -1,    64,    79,    -1,   103,    67,    41,    68,
      -1,    58,   101,    -1,    59,   101,    -1,    69,   103,    -1,
      69,    88,    -1,    69,    10,    69,    88,    -1,    69,    57,
      -1,    82,    -1,    69,    56,    -1,    69,     9,    56,    -1,
      96,    -1,    96,     9,    96,    -1,    96,    79,    83,    -1,
      91,    -1,    81,    -1,    93,    -1,    93,    67,    96,    68,
      -1,    51,    -1,    52,    -1,   103,    -1,    88,    -1,    99,
      -1,    86,    -1,   100,    -1,    67,    96,    68,    -1,    86,
      -1,   103,    67,    95,    68,    -1,   100,    -1,   100,    67,
      95,    68,    -1,    87,    -1,    91,    -1,    90,    -1,    93,
      -1,    69,   103,    -1,    96,    -1,    67,    96,    64,    96,
      68,    -1,    96,     6,     6,    94,    -1,    96,     7,     7,
      94,    -1,    96,     9,     7,    94,    -1,    96,    55,     7,
      94,    -1,    96,    -1,   103,    -1,    46,    -1,    41,    -1,
      45,    67,   105,    68,    -1,    95,    -1,    38,    -1,    50,
      -1,    49,    67,   105,    68,    -1,    99,    -1,    82,    -1,
      48,    -1,    47,    67,   103,    68,    -1,   103,    67,   102,
      68,    -1,    58,   101,    67,   102,    68,    -1,    58,     6,
       7,   101,    67,    39,    68,    -1,    -1,     8,   103,    -1,
       9,   103,    -1,    39,    -1,    38,    -1,    40,    -1,    37,
      -1,    60,    -1,     9,   103,    -1,     8,   103,    -1,    70,
     103,    -1,    67,   105,    68,    -1,    -1,    64,   105,    -1,
     103,    -1,   105,     8,   105,    -1,   105,     9,   105,    -1,
     105,    10,   105,    -1,   105,    11,   105,    -1,   105,    12,
     105,    -1,   105,     6,     6,   105,    -1,   105,     7,     7,
     105,    -1,   105,     5,   105,    -1,   105,     4,   105,    -1,
     105,     3,   105,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    68,    68,    70,    69,    77,    76,    84,    83,    89,
      94,   100,   101,   102,   108,   112,   116,   123,   130,   137,
     141,   148,   155,   162,   169,   176,   185,   197,   201,   205,
     212,   219,   225,   231,   240,   247,   254,   261,   265,   269,
     273,   280,   302,   310,   319,   326,   335,   346,   352,   355,
     359,   364,   365,   368,   374,   382,   389,   395,   400,   405,
     411,   414,   420,   428,   432,   441,   447,   448,   449,   450,
     455,   461,   467,   473,   474,   477,   478,   486,   495,   496,
     505,   506,   512,   515,   516,   517,   519,   527,   535,   544,
     550,   556,   562,   570,   576,   584,   585,   589,   597,   598,
     604,   605,   613,   614,   617,   623,   631,   639,   647,   657,
     660,   664,   670,   671,   672,   675,   676,   680,   684,   688,
     692,   698,   701,   707,   708,   712,   716,   720,   724,   728,
     732,   736,   740,   744
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'|'", "'^'", "'&'", "'<'", "'>'", "'+'",
  "'-'", "'*'", "'/'", "'%'", "LTYPE1", "LTYPE2", "LTYPE3", "LTYPE4",
  "LTYPE5", "LTYPE6", "LTYPE7", "LTYPE8", "LTYPE9", "LTYPEA", "LTYPEB",
  "LTYPEC", "LTYPED", "LTYPEE", "LTYPEG", "LTYPEH", "LTYPEI", "LTYPEJ",
  "LTYPEK", "LTYPEL", "LTYPEM", "LTYPEN", "LTYPEBX", "LTYPEPLD", "LCONST",
  "LSP", "LSB", "LFP", "LPC", "LTYPEX", "LTYPEPC", "LTYPEF", "LR", "LREG",
  "LF", "LFREG", "LC", "LCREG", "LPSR", "LFCR", "LCOND", "LS", "LAT",
  "LFCONST", "LSCONST", "LNAME", "LLAB", "LVAR", "':'", "'='", "';'",
  "','", "'['", "']'", "'('", "')'", "'$'", "'~'", "$accept", "prog",
  "$@1", "line", "$@2", "$@3", "inst", "cond", "comma", "rel", "ximm",
  "fcon", "reglist", "gen", "nireg", "ireg", "ioreg", "oreg", "imsr",
  "imm", "reg", "regreg", "shift", "rcon", "sreg", "spreg", "creg",
  "frcon", "freg", "name", "offset", "pointer", "con", "oexpr", "expr", YY_NULL
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
     305,    58,    61,    59,    44,    91,    93,    40,    41,    36,
     126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    71,    72,    73,    72,    75,    74,    76,    74,    74,
      74,    74,    74,    74,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    77,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    77,    78,    78,
      78,    79,    79,    80,    80,    80,    81,    81,    81,    81,
      81,    82,    82,    83,    83,    83,    84,    84,    84,    84,
      84,    84,    84,    84,    84,    85,    85,    86,    87,    87,
      88,    88,    88,    89,    89,    89,    90,    91,    92,    93,
      93,    93,    93,    94,    94,    95,    95,    95,    96,    96,
      97,    97,    98,    98,    99,    99,   100,   100,   100,   101,
     101,   101,   102,   102,   102,   103,   103,   103,   103,   103,
     103,   104,   104,   105,   105,   105,   105,   105,   105,   105,
     105,   105,   105,   105
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     0,     3,     0,     4,     0,     4,     4,
       4,     1,     2,     2,     7,     6,     5,     5,     5,     4,
       4,     3,     3,     4,     6,     7,     7,     7,     6,     6,
       3,     4,     6,     8,     6,     4,     3,     5,     5,     7,
       6,    12,     7,     9,     2,     4,     4,     2,     0,     2,
       2,     0,     2,     4,     2,     2,     2,     2,     4,     2,
       1,     2,     3,     1,     3,     3,     1,     1,     1,     4,
       1,     1,     1,     1,     1,     1,     1,     3,     1,     4,
       1,     4,     1,     1,     1,     1,     2,     1,     5,     4,
       4,     4,     4,     1,     1,     1,     1,     4,     1,     1,
       1,     4,     1,     1,     1,     4,     4,     5,     7,     0,
       2,     2,     1,     1,     1,     1,     1,     2,     2,     2,
       3,     0,     2,     1,     3,     3,     3,     3,     3,     4,
       4,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     3,     1,     0,     0,    48,    48,    48,    48,    51,
      48,    48,    48,    48,    48,     0,     0,    48,    51,    51,
      48,    48,    48,    48,    48,    48,    51,     0,     0,     0,
       0,     0,     0,    11,     4,     0,    13,     0,     0,     0,
      51,    51,     0,    51,     0,     0,    51,    51,     0,     0,
     115,   109,   116,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    78,
      82,    44,    80,     0,    99,    96,     0,    95,     0,   104,
      70,    71,     0,    67,    60,     0,    73,    66,    68,    98,
      87,    74,    72,     0,     7,     0,     5,     0,    12,    49,
      50,     0,     0,    84,    83,    85,     0,     0,     0,    52,
     109,   109,    22,     0,     0,     0,     0,     0,     0,     0,
       0,    87,    30,   118,   117,     0,     0,     0,     0,   123,
       0,   119,     0,     0,     0,    51,    36,     0,     0,     0,
     103,     0,   102,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    61,    59,    57,    56,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      86,     0,     0,     0,   109,    19,    20,    75,    76,     0,
      54,    55,     0,    23,     0,     0,    51,     0,     0,     0,
       0,   109,   110,   111,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   120,    31,     0,   113,   112,
     114,     0,     0,    35,     0,     0,     0,     0,     0,     0,
       0,    77,     0,     0,     0,     0,    62,     0,    45,     0,
       0,     0,     0,     0,    46,     8,     9,     6,    10,    16,
      87,    17,    18,    54,     0,     0,    51,     0,     0,     0,
       0,     0,    51,     0,     0,   133,   132,   131,     0,     0,
     124,   125,   126,   127,   128,     0,   106,     0,    37,     0,
     104,    38,    51,     0,     0,    81,    79,    97,   105,    58,
      69,    89,    93,    94,    90,    91,    92,    15,    53,    24,
       0,    64,    65,     0,    29,    51,    28,     0,   107,   129,
     130,    32,    34,     0,     0,    40,     0,     0,    14,    26,
      25,    27,     0,     0,     0,    39,     0,    42,     0,   108,
      33,     0,     0,     0,     0,   100,     0,     0,    43,     0,
       0,     0,     0,   121,    88,   101,     0,    41,   122
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     3,    34,   168,   166,    35,    37,   109,   112,
      83,    84,   185,    85,   176,    69,    70,    86,   102,   103,
      87,   317,    88,   281,    89,   121,   326,   141,    91,    72,
     128,   211,   129,   337,   130
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -128
static const yytype_int16 yypact[] =
{
    -128,     4,  -128,   315,   -35,  -128,  -128,  -128,  -128,   -10,
    -128,  -128,  -128,  -128,  -128,    44,    44,  -128,   -10,   -10,
    -128,  -128,  -128,  -128,  -128,  -128,   -10,   416,   371,   371,
     -49,     9,    32,  -128,  -128,    38,  -128,   487,   487,   344,
      69,   -10,   391,    69,   487,   209,   489,    69,   317,   317,
    -128,    49,  -128,   317,   317,    42,    48,   106,    67,  -128,
      61,   191,    25,    93,   191,    67,    67,    68,   170,  -128,
    -128,  -128,    72,    84,  -128,  -128,    86,  -128,   109,  -128,
    -128,  -128,   233,  -128,  -128,    80,  -128,  -128,   115,  -128,
     426,  -128,    84,   120,  -128,   317,  -128,   317,  -128,  -128,
    -128,   317,   137,  -128,  -128,  -128,   148,   155,   397,  -128,
      74,    74,  -128,   164,   371,   204,   240,   207,   206,    68,
     223,  -128,  -128,  -128,  -128,   270,   317,   317,   227,  -128,
     183,  -128,    90,   160,   317,   -10,  -128,   234,   237,    16,
    -128,   254,  -128,   255,   256,   257,   240,  -128,   212,   168,
     548,   317,   317,   428,   258,  -128,  -128,  -128,    84,   371,
     240,   318,   316,   335,   348,   371,   315,   502,   315,   512,
    -128,   240,   240,   371,    49,  -128,  -128,  -128,  -128,   289,
    -128,  -128,   330,  -128,   240,   291,    11,   307,   168,   312,
      68,    74,  -128,  -128,   160,   317,   317,   317,   377,   379,
     317,   317,   317,   317,   317,  -128,  -128,   324,  -128,  -128,
    -128,   325,   337,  -128,    77,   317,   338,   126,    77,   240,
     240,  -128,   339,   342,   249,   347,  -128,   416,  -128,   352,
     170,   170,   170,   170,  -128,  -128,  -128,  -128,  -128,  -128,
     362,  -128,  -128,   227,    -2,   359,   -10,   366,   240,   240,
     240,   240,   375,   336,   384,   562,   590,   597,   317,   317,
     213,   213,  -128,  -128,  -128,   385,  -128,    61,  -128,   357,
     395,  -128,   -10,   396,   398,  -128,  -128,  -128,  -128,  -128,
    -128,  -128,  -128,  -128,  -128,  -128,  -128,   240,  -128,  -128,
     434,  -128,  -128,   400,  -128,   432,  -128,   424,  -128,   436,
     436,   459,  -128,   240,    77,  -128,   402,   240,  -128,  -128,
    -128,  -128,   404,   317,   411,  -128,   240,  -128,   415,  -128,
    -128,   216,   418,   240,   413,  -128,   421,   240,  -128,   317,
     216,   419,   302,   425,  -128,  -128,   317,  -128,   573
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -128,  -128,  -128,   -77,  -128,  -128,  -128,   538,    50,   382,
     -57,   429,    33,    -7,  -128,   -48,   -43,   -21,    36,  -127,
     -23,  -128,    29,    17,  -101,   -28,   161,  -128,   -37,    -8,
     -65,   299,     2,  -128,   -32
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -64
static const yytype_int16 yytable[] =
{
      90,    90,   117,   136,     2,   206,    71,    55,    57,    90,
      90,    90,    94,    95,   104,   104,    90,    56,    56,   147,
     248,   104,    93,   120,   137,   216,   142,   143,    36,    73,
      92,    92,   107,    48,    49,   135,   208,   209,   210,   245,
     148,    92,   144,   145,   113,   180,   181,   118,   222,   223,
     123,   124,    48,    49,    41,   125,   131,   126,   127,    42,
     177,   157,    50,   167,   138,   169,   105,   105,    59,    60,
      96,   189,   155,   105,   106,    41,    67,   -63,    99,   100,
     115,    50,   126,   127,   158,    52,    90,   223,   186,   235,
     108,   237,    53,   114,    97,    54,   119,   122,    48,    49,
     178,    98,    51,   170,    52,    74,   132,   183,    75,   243,
     179,    53,    76,    77,    54,   133,    92,   134,   148,   224,
      99,   100,    99,   100,    78,    79,   253,    50,   192,   193,
      82,    90,   229,    41,   207,   146,   212,    90,   301,   149,
      78,    79,   252,   240,   159,    90,    99,   100,   239,   241,
      52,   150,   228,   151,   225,   124,   246,    53,   234,   101,
      54,    92,   139,   255,   256,   257,   242,    92,   260,   261,
     262,   263,   264,    78,   270,    92,   152,   268,    48,    49,
     271,   272,   160,   269,   165,   213,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   273,   274,   208,   209,
     210,   171,   282,   282,   282,   282,   279,    50,    74,    75,
     302,    75,   172,    76,    77,    76,    77,    48,    49,   173,
     291,   186,   186,   202,   203,   204,   299,   300,   294,    73,
      52,   182,   283,   283,   283,   283,   249,    53,    78,    79,
      54,    48,   153,   154,    99,   100,    50,   309,   284,   285,
     286,   205,   195,   196,   197,   198,   199,   200,   201,   202,
     203,   204,    99,   100,   308,   324,   325,   315,   184,    52,
      50,   187,   311,   188,   116,   314,    68,   191,    74,    54,
     221,    75,   292,   293,   318,    76,    77,   190,   322,   155,
     156,    51,   118,    52,   194,   328,   289,   332,   214,   331,
      68,   215,   296,    54,   338,   195,   196,   197,   198,   199,
     200,   201,   202,   203,   204,   320,     4,   277,   217,   218,
     219,   220,   305,   231,   230,    48,    49,   227,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,   232,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    48,    49,    50,   233,   244,   247,    28,    29,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     335,   245,   250,    30,    31,    32,   251,    52,    33,    48,
      49,    50,    74,   258,    53,    75,   259,    54,   265,    76,
      77,    78,    79,   266,   226,    80,    81,    99,   100,    48,
      49,   267,    51,   297,    52,    48,    49,   275,    50,    74,
     276,    68,    75,    82,    54,   278,    76,    77,    78,    79,
     280,   303,    80,    81,    48,    49,   287,   288,    50,    51,
     290,    52,   161,   162,    50,   163,    48,    49,    68,   295,
      82,    54,    48,    49,   200,   201,   202,   203,   204,   110,
     111,    52,   298,    50,   101,   174,   111,    52,    53,   304,
     306,    54,   307,   312,    68,    50,   310,    54,   313,   316,
      74,    50,   319,    75,    51,   321,    52,    76,    77,   323,
     329,   164,   327,    68,   226,   330,    54,   334,    52,   336,
     175,   333,   140,   254,    52,    53,    41,     0,    54,     0,
       0,    68,     0,     0,    54,   195,   196,   197,   198,   199,
     200,   201,   202,   203,   204,   195,   196,   197,   198,   199,
     200,   201,   202,   203,   204,    74,     0,    74,    75,     0,
      75,     0,    76,    77,    76,    77,     0,     0,     0,     0,
      99,   100,    99,   100,    38,    39,    40,     0,    43,    44,
      45,    46,    47,    41,     0,    58,   101,     0,    61,    62,
      63,    64,    65,    66,     0,   236,   196,   197,   198,   199,
     200,   201,   202,   203,   204,   238,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   208,   209,   210,    75,
       0,     0,     0,    76,    77,   197,   198,   199,   200,   201,
     202,   203,   204,   198,   199,   200,   201,   202,   203,   204
};

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-128)))

#define yytable_value_is_error(Yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
      28,    29,    45,    60,     0,   132,    27,    15,    16,    37,
      38,    39,    61,    62,    37,    38,    44,    15,    16,    67,
       9,    44,    29,    46,    61,     9,    63,    64,    63,    27,
      28,    29,    39,     8,     9,    58,    38,    39,    40,    41,
      68,    39,    65,    66,    42,   110,   111,    45,   149,   150,
      48,    49,     8,     9,    64,     6,    54,     8,     9,     9,
     108,    82,    37,    95,    62,    97,    37,    38,    18,    19,
      61,   119,    56,    44,    38,    64,    26,    66,    53,    54,
      44,    37,     8,     9,    82,    60,   114,   188,   116,   166,
      40,   168,    67,    43,    62,    70,    46,    47,     8,     9,
     108,    63,    58,   101,    60,    38,    64,   114,    41,   174,
     108,    67,    45,    46,    70,    67,   114,    11,   146,   151,
      53,    54,    53,    54,    47,    48,   191,    37,   126,   127,
      69,   159,   160,    64,   132,    67,   134,   165,   265,    67,
      47,    48,   190,   171,    64,   173,    53,    54,   171,   172,
      60,    67,   159,    67,   152,   153,   184,    67,   165,    69,
      70,   159,    69,   195,   196,   197,   173,   165,   200,   201,
     202,   203,   204,    47,    48,   173,    67,   214,     8,     9,
     217,   218,    67,   215,    64,   135,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,   219,   220,    38,    39,
      40,    64,   230,   231,   232,   233,   227,    37,    38,    41,
     267,    41,    64,    45,    46,    45,    46,     8,     9,    64,
     248,   249,   250,    10,    11,    12,   258,   259,   251,   227,
      60,    67,   230,   231,   232,   233,   186,    67,    47,    48,
      70,     8,     9,    10,    53,    54,    37,   290,   231,   232,
     233,    68,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    53,    54,   287,    49,    50,   304,    64,    60,
      37,    64,   295,    67,    65,   303,    67,     7,    38,    70,
      68,    41,   249,   250,   307,    45,    46,    64,   316,    56,
      57,    58,   290,    60,    67,   323,   246,   329,    64,   327,
      67,    64,   252,    70,   336,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,   313,     1,    68,    64,    64,
      64,    64,   272,     7,     6,     8,     9,    69,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,     7,    28,    29,    30,    31,    32,    33,    34,
      35,    36,     8,     9,    37,     7,    67,    66,    43,    44,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      68,    41,    65,    58,    59,    60,    64,    60,    63,     8,
       9,    37,    38,     6,    67,    41,     7,    70,    64,    45,
      46,    47,    48,    68,    56,    51,    52,    53,    54,     8,
       9,    64,    58,    67,    60,     8,     9,    68,    37,    38,
      68,    67,    41,    69,    70,    68,    45,    46,    47,    48,
      68,    64,    51,    52,     8,     9,    64,    68,    37,    58,
      64,    60,     6,     7,    37,     9,     8,     9,    67,    64,
      69,    70,     8,     9,     8,     9,    10,    11,    12,    58,
      59,    60,    68,    37,    69,    58,    59,    60,    67,    64,
      64,    70,    64,    39,    67,    37,    66,    70,     9,    67,
      38,    37,    68,    41,    58,    64,    60,    45,    46,    64,
      67,    55,    64,    67,    56,    64,    70,    68,    60,    64,
     108,   330,    63,   194,    60,    67,    64,    -1,    70,    -1,
      -1,    67,    -1,    -1,    70,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    38,    -1,    38,    41,    -1,
      41,    -1,    45,    46,    45,    46,    -1,    -1,    -1,    -1,
      53,    54,    53,    54,     6,     7,     8,    -1,    10,    11,
      12,    13,    14,    64,    -1,    17,    69,    -1,    20,    21,
      22,    23,    24,    25,    -1,    63,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    63,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    38,    39,    40,    41,
      -1,    -1,    -1,    45,    46,     5,     6,     7,     8,     9,
      10,    11,    12,     6,     7,     8,     9,    10,    11,    12
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    72,     0,    73,     1,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    43,    44,
      58,    59,    60,    63,    74,    77,    63,    78,    78,    78,
      78,    64,    79,    78,    78,    78,    78,    78,     8,     9,
      37,    58,    60,    67,    70,   100,   103,   100,    78,    79,
      79,    78,    78,    78,    78,    78,    78,    79,    67,    86,
      87,    88,   100,   103,    38,    41,    45,    46,    47,    48,
      51,    52,    69,    81,    82,    84,    88,    91,    93,    95,
      96,    99,   103,    84,    61,    62,    61,    62,    63,    53,
      54,    69,    89,    90,    91,    93,    89,    84,    79,    79,
      58,    59,    80,   103,    79,    89,    65,    87,   103,    79,
      91,    96,    79,   103,   103,     6,     8,     9,   101,   103,
     105,   103,    64,    67,    11,    91,    81,    99,   103,    69,
      82,    98,    99,    99,    91,    91,    67,    86,    96,    67,
      67,    67,    67,     9,    10,    56,    57,    88,   103,    64,
      67,     6,     7,     9,    55,    64,    76,   105,    75,   105,
     103,    64,    64,    64,    58,    80,    85,    86,   100,   103,
     101,   101,    67,    84,    64,    83,    96,    64,    67,    86,
      64,     7,   103,   103,    67,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    68,    90,   103,    38,    39,
      40,   102,   103,    79,    64,    64,     9,    64,    64,    64,
      64,    68,    95,    95,   105,   103,    56,    69,    84,    96,
       6,     7,     7,     7,    84,    74,    63,    74,    63,    91,
      96,    91,    84,   101,    67,    41,    96,    66,     9,    79,
      65,    64,    86,   101,   102,   105,   105,   105,     6,     7,
     105,   105,   105,   105,   105,    64,    68,    64,    99,   105,
      48,    99,    99,    91,    91,    68,    68,    68,    68,    88,
      68,    94,    96,   103,    94,    94,    94,    64,    68,    79,
      64,    96,    83,    83,    91,    64,    79,    67,    68,   105,
     105,    90,    81,    64,    64,    79,    64,    64,    91,    87,
      66,    91,    39,     9,    96,    99,    67,    92,    91,    68,
     103,    64,    96,    64,    49,    50,    97,    64,    96,    67,
      64,    96,   105,    97,    68,    68,    64,   104,   105
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

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))

/* Error token number */
#define YYTERROR	1
#define YYERRCODE	256


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
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  YYUSE (yytype);
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
  YYSIZE_T yysize0 = yytnamerr (YY_NULL, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULL;
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
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULL, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
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

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

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

  YYUSE (yytype);
}




/* The lookahead symbol.  */
int yychar;


#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval YY_INITIAL_VALUE(yyval_default);

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

       Refer to the stacks through separate pointers, to allow yyoverflow
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
  int yytoken = 0;
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

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
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
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

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
/* Line 1787 of yacc.c  */
#line 70 "a.y"
    {
		stmtline = lineno;
	}
    break;

  case 5:
/* Line 1787 of yacc.c  */
#line 77 "a.y"
    {
		if((yyvsp[(1) - (2)].sym)->value != pc)
			yyerror("redeclaration of %s", (yyvsp[(1) - (2)].sym)->name);
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 7:
/* Line 1787 of yacc.c  */
#line 84 "a.y"
    {
		(yyvsp[(1) - (2)].sym)->type = LLAB;
		(yyvsp[(1) - (2)].sym)->value = pc;
	}
    break;

  case 9:
/* Line 1787 of yacc.c  */
#line 90 "a.y"
    {
		(yyvsp[(1) - (4)].sym)->type = LVAR;
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 10:
/* Line 1787 of yacc.c  */
#line 95 "a.y"
    {
		if((yyvsp[(1) - (4)].sym)->value != (yyvsp[(3) - (4)].lval))
			yyerror("redeclaration of %s", (yyvsp[(1) - (4)].sym)->name);
		(yyvsp[(1) - (4)].sym)->value = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 14:
/* Line 1787 of yacc.c  */
#line 109 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), (yyvsp[(5) - (7)].lval), &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 15:
/* Line 1787 of yacc.c  */
#line 113 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].addr), (yyvsp[(5) - (6)].lval), &nullgen);
	}
    break;

  case 16:
/* Line 1787 of yacc.c  */
#line 117 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 17:
/* Line 1787 of yacc.c  */
#line 124 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 18:
/* Line 1787 of yacc.c  */
#line 131 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 19:
/* Line 1787 of yacc.c  */
#line 138 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 20:
/* Line 1787 of yacc.c  */
#line 142 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 21:
/* Line 1787 of yacc.c  */
#line 149 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 22:
/* Line 1787 of yacc.c  */
#line 156 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 23:
/* Line 1787 of yacc.c  */
#line 163 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &nullgen, NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 24:
/* Line 1787 of yacc.c  */
#line 170 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].addr), (yyvsp[(5) - (6)].lval), &nullgen);
	}
    break;

  case 25:
/* Line 1787 of yacc.c  */
#line 177 "a.y"
    {
		Addr g;

		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(6) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), NREG, &g);
	}
    break;

  case 26:
/* Line 1787 of yacc.c  */
#line 186 "a.y"
    {
		Addr g;

		g = nullgen;
		g.type = D_CONST;
		g.offset = (yyvsp[(4) - (7)].lval);
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &g, NREG, &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 27:
/* Line 1787 of yacc.c  */
#line 198 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(5) - (7)].addr), (yyvsp[(3) - (7)].addr).reg, &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 28:
/* Line 1787 of yacc.c  */
#line 202 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(5) - (6)].addr), (yyvsp[(3) - (6)].addr).reg, &(yyvsp[(3) - (6)].addr));
	}
    break;

  case 29:
/* Line 1787 of yacc.c  */
#line 206 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(4) - (6)].addr), (yyvsp[(6) - (6)].addr).reg, &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 30:
/* Line 1787 of yacc.c  */
#line 213 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), (yyvsp[(2) - (3)].lval), &nullgen, NREG, &nullgen);
	}
    break;

  case 31:
/* Line 1787 of yacc.c  */
#line 220 "a.y"
    {
		(yyvsp[(4) - (4)].addr).type = D_CONST2;
		(yyvsp[(4) - (4)].addr).offset2 = ArgsSizeUnknown;
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), 0, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 32:
/* Line 1787 of yacc.c  */
#line 226 "a.y"
    {
		(yyvsp[(6) - (6)].addr).type = D_CONST2;
		(yyvsp[(6) - (6)].addr).offset2 = ArgsSizeUnknown;
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 33:
/* Line 1787 of yacc.c  */
#line 232 "a.y"
    {
		(yyvsp[(6) - (8)].addr).type = D_CONST2;
		(yyvsp[(6) - (8)].addr).offset2 = (yyvsp[(8) - (8)].lval);
		outcode((yyvsp[(1) - (8)].lval), Always, &(yyvsp[(2) - (8)].addr), (yyvsp[(4) - (8)].lval), &(yyvsp[(6) - (8)].addr));
	}
    break;

  case 34:
/* Line 1787 of yacc.c  */
#line 241 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), Always, &(yyvsp[(2) - (6)].addr), (yyvsp[(4) - (6)].lval), &(yyvsp[(6) - (6)].addr));
	}
    break;

  case 35:
/* Line 1787 of yacc.c  */
#line 248 "a.y"
    {
		outcode((yyvsp[(1) - (4)].lval), (yyvsp[(2) - (4)].lval), &(yyvsp[(3) - (4)].addr), NREG, &nullgen);
	}
    break;

  case 36:
/* Line 1787 of yacc.c  */
#line 255 "a.y"
    {
		outcode((yyvsp[(1) - (3)].lval), Always, &nullgen, NREG, &(yyvsp[(3) - (3)].addr));
	}
    break;

  case 37:
/* Line 1787 of yacc.c  */
#line 262 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 38:
/* Line 1787 of yacc.c  */
#line 266 "a.y"
    {
		outcode((yyvsp[(1) - (5)].lval), (yyvsp[(2) - (5)].lval), &(yyvsp[(3) - (5)].addr), NREG, &(yyvsp[(5) - (5)].addr));
	}
    break;

  case 39:
/* Line 1787 of yacc.c  */
#line 270 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), (yyvsp[(5) - (7)].lval), &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 40:
/* Line 1787 of yacc.c  */
#line 274 "a.y"
    {
		outcode((yyvsp[(1) - (6)].lval), (yyvsp[(2) - (6)].lval), &(yyvsp[(3) - (6)].addr), (yyvsp[(5) - (6)].addr).reg, &nullgen);
	}
    break;

  case 41:
/* Line 1787 of yacc.c  */
#line 281 "a.y"
    {
		Addr g;

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
		outcode(AMRC, Always, &nullgen, NREG, &g);
	}
    break;

  case 42:
/* Line 1787 of yacc.c  */
#line 303 "a.y"
    {
		outcode((yyvsp[(1) - (7)].lval), (yyvsp[(2) - (7)].lval), &(yyvsp[(3) - (7)].addr), (yyvsp[(5) - (7)].addr).reg, &(yyvsp[(7) - (7)].addr));
	}
    break;

  case 43:
/* Line 1787 of yacc.c  */
#line 311 "a.y"
    {
		(yyvsp[(7) - (9)].addr).type = D_REGREG2;
		(yyvsp[(7) - (9)].addr).offset = (yyvsp[(9) - (9)].lval);
		outcode((yyvsp[(1) - (9)].lval), (yyvsp[(2) - (9)].lval), &(yyvsp[(3) - (9)].addr), (yyvsp[(5) - (9)].addr).reg, &(yyvsp[(7) - (9)].addr));
	}
    break;

  case 44:
/* Line 1787 of yacc.c  */
#line 320 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &(yyvsp[(2) - (2)].addr), NREG, &nullgen);
	}
    break;

  case 45:
/* Line 1787 of yacc.c  */
#line 327 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != D_CONST || (yyvsp[(4) - (4)].addr).type != D_CONST)
			yyerror("arguments to PCDATA must be integer constants");
		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 46:
/* Line 1787 of yacc.c  */
#line 336 "a.y"
    {
		if((yyvsp[(2) - (4)].addr).type != D_CONST)
			yyerror("index for FUNCDATA must be integer constant");
		if((yyvsp[(4) - (4)].addr).type != D_EXTERN && (yyvsp[(4) - (4)].addr).type != D_STATIC && (yyvsp[(4) - (4)].addr).type != D_OREG)
			yyerror("value for FUNCDATA must be symbol reference");
 		outcode((yyvsp[(1) - (4)].lval), Always, &(yyvsp[(2) - (4)].addr), NREG, &(yyvsp[(4) - (4)].addr));
	}
    break;

  case 47:
/* Line 1787 of yacc.c  */
#line 347 "a.y"
    {
		outcode((yyvsp[(1) - (2)].lval), Always, &nullgen, NREG, &nullgen);
	}
    break;

  case 48:
/* Line 1787 of yacc.c  */
#line 352 "a.y"
    {
		(yyval.lval) = Always;
	}
    break;

  case 49:
/* Line 1787 of yacc.c  */
#line 356 "a.y"
    {
		(yyval.lval) = ((yyvsp[(1) - (2)].lval) & ~C_SCOND) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 50:
/* Line 1787 of yacc.c  */
#line 360 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (2)].lval) | (yyvsp[(2) - (2)].lval);
	}
    break;

  case 53:
/* Line 1787 of yacc.c  */
#line 369 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) + pc;
	}
    break;

  case 54:
/* Line 1787 of yacc.c  */
#line 375 "a.y"
    {
		(yyval.addr) = nullgen;
		if(pass == 2)
			yyerror("undefined label: %s", (yyvsp[(1) - (2)].sym)->name);
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 55:
/* Line 1787 of yacc.c  */
#line 383 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_BRANCH;
		(yyval.addr).offset = (yyvsp[(1) - (2)].sym)->value + (yyvsp[(2) - (2)].lval);
	}
    break;

  case 56:
/* Line 1787 of yacc.c  */
#line 390 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 57:
/* Line 1787 of yacc.c  */
#line 396 "a.y"
    {
		(yyval.addr) = (yyvsp[(2) - (2)].addr);
		(yyval.addr).type = D_CONST;
	}
    break;

  case 58:
/* Line 1787 of yacc.c  */
#line 401 "a.y"
    {
		(yyval.addr) = (yyvsp[(4) - (4)].addr);
		(yyval.addr).type = D_OCONST;
	}
    break;

  case 59:
/* Line 1787 of yacc.c  */
#line 406 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SCONST;
		memcpy((yyval.addr).u.sval, (yyvsp[(2) - (2)].sval), sizeof((yyval.addr).u.sval));
	}
    break;

  case 61:
/* Line 1787 of yacc.c  */
#line 415 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = (yyvsp[(2) - (2)].dval);
	}
    break;

  case 62:
/* Line 1787 of yacc.c  */
#line 421 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FCONST;
		(yyval.addr).u.dval = -(yyvsp[(3) - (3)].dval);
	}
    break;

  case 63:
/* Line 1787 of yacc.c  */
#line 429 "a.y"
    {
		(yyval.lval) = 1 << (yyvsp[(1) - (1)].lval);
	}
    break;

  case 64:
/* Line 1787 of yacc.c  */
#line 433 "a.y"
    {
		int i;
		(yyval.lval)=0;
		for(i=(yyvsp[(1) - (3)].lval); i<=(yyvsp[(3) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
		for(i=(yyvsp[(3) - (3)].lval); i<=(yyvsp[(1) - (3)].lval); i++)
			(yyval.lval) |= 1<<i;
	}
    break;

  case 65:
/* Line 1787 of yacc.c  */
#line 442 "a.y"
    {
		(yyval.lval) = (1<<(yyvsp[(1) - (3)].lval)) | (yyvsp[(3) - (3)].lval);
	}
    break;

  case 69:
/* Line 1787 of yacc.c  */
#line 451 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (4)].addr);
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 70:
/* Line 1787 of yacc.c  */
#line 456 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_PSR;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 71:
/* Line 1787 of yacc.c  */
#line 462 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FPCR;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 72:
/* Line 1787 of yacc.c  */
#line 468 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).offset = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 76:
/* Line 1787 of yacc.c  */
#line 479 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (1)].addr);
		if((yyvsp[(1) - (1)].addr).name != D_EXTERN && (yyvsp[(1) - (1)].addr).name != D_STATIC) {
		}
	}
    break;

  case 77:
/* Line 1787 of yacc.c  */
#line 487 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(2) - (3)].lval);
		(yyval.addr).offset = 0;
	}
    break;

  case 79:
/* Line 1787 of yacc.c  */
#line 497 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 81:
/* Line 1787 of yacc.c  */
#line 507 "a.y"
    {
		(yyval.addr) = (yyvsp[(1) - (4)].addr);
		(yyval.addr).type = D_OREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 86:
/* Line 1787 of yacc.c  */
#line 520 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_CONST;
		(yyval.addr).offset = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 87:
/* Line 1787 of yacc.c  */
#line 528 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_REG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 88:
/* Line 1787 of yacc.c  */
#line 536 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_REGREG;
		(yyval.addr).reg = (yyvsp[(2) - (5)].lval);
		(yyval.addr).offset = (yyvsp[(4) - (5)].lval);
	}
    break;

  case 89:
/* Line 1787 of yacc.c  */
#line 545 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (0 << 5);
	}
    break;

  case 90:
/* Line 1787 of yacc.c  */
#line 551 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (1 << 5);
	}
    break;

  case 91:
/* Line 1787 of yacc.c  */
#line 557 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (2 << 5);
	}
    break;

  case 92:
/* Line 1787 of yacc.c  */
#line 563 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_SHIFT;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval) | (yyvsp[(4) - (4)].lval) | (3 << 5);
	}
    break;

  case 93:
/* Line 1787 of yacc.c  */
#line 571 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 16)
			print("register value out of range\n");
		(yyval.lval) = (((yyvsp[(1) - (1)].lval)&15) << 8) | (1 << 4);
	}
    break;

  case 94:
/* Line 1787 of yacc.c  */
#line 577 "a.y"
    {
		if((yyval.lval) < 0 || (yyval.lval) >= 32)
			print("shift value out of range\n");
		(yyval.lval) = ((yyvsp[(1) - (1)].lval)&31) << 7;
	}
    break;

  case 96:
/* Line 1787 of yacc.c  */
#line 586 "a.y"
    {
		(yyval.lval) = REGPC;
	}
    break;

  case 97:
/* Line 1787 of yacc.c  */
#line 590 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 99:
/* Line 1787 of yacc.c  */
#line 599 "a.y"
    {
		(yyval.lval) = REGSP;
	}
    break;

  case 101:
/* Line 1787 of yacc.c  */
#line 606 "a.y"
    {
		if((yyvsp[(3) - (4)].lval) < 0 || (yyvsp[(3) - (4)].lval) >= NREG)
			print("register value out of range\n");
		(yyval.lval) = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 104:
/* Line 1787 of yacc.c  */
#line 618 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FREG;
		(yyval.addr).reg = (yyvsp[(1) - (1)].lval);
	}
    break;

  case 105:
/* Line 1787 of yacc.c  */
#line 624 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_FREG;
		(yyval.addr).reg = (yyvsp[(3) - (4)].lval);
	}
    break;

  case 106:
/* Line 1787 of yacc.c  */
#line 632 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = (yyvsp[(3) - (4)].lval);
		(yyval.addr).sym = nil;
		(yyval.addr).offset = (yyvsp[(1) - (4)].lval);
	}
    break;

  case 107:
/* Line 1787 of yacc.c  */
#line 640 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = (yyvsp[(4) - (5)].lval);
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (5)].sym)->name, 0);
		(yyval.addr).offset = (yyvsp[(2) - (5)].lval);
	}
    break;

  case 108:
/* Line 1787 of yacc.c  */
#line 648 "a.y"
    {
		(yyval.addr) = nullgen;
		(yyval.addr).type = D_OREG;
		(yyval.addr).name = D_STATIC;
		(yyval.addr).sym = linklookup(ctxt, (yyvsp[(1) - (7)].sym)->name, 1);
		(yyval.addr).offset = (yyvsp[(4) - (7)].lval);
	}
    break;

  case 109:
/* Line 1787 of yacc.c  */
#line 657 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 110:
/* Line 1787 of yacc.c  */
#line 661 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 111:
/* Line 1787 of yacc.c  */
#line 665 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 116:
/* Line 1787 of yacc.c  */
#line 677 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (1)].sym)->value;
	}
    break;

  case 117:
/* Line 1787 of yacc.c  */
#line 681 "a.y"
    {
		(yyval.lval) = -(yyvsp[(2) - (2)].lval);
	}
    break;

  case 118:
/* Line 1787 of yacc.c  */
#line 685 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 119:
/* Line 1787 of yacc.c  */
#line 689 "a.y"
    {
		(yyval.lval) = ~(yyvsp[(2) - (2)].lval);
	}
    break;

  case 120:
/* Line 1787 of yacc.c  */
#line 693 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (3)].lval);
	}
    break;

  case 121:
/* Line 1787 of yacc.c  */
#line 698 "a.y"
    {
		(yyval.lval) = 0;
	}
    break;

  case 122:
/* Line 1787 of yacc.c  */
#line 702 "a.y"
    {
		(yyval.lval) = (yyvsp[(2) - (2)].lval);
	}
    break;

  case 124:
/* Line 1787 of yacc.c  */
#line 709 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) + (yyvsp[(3) - (3)].lval);
	}
    break;

  case 125:
/* Line 1787 of yacc.c  */
#line 713 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) - (yyvsp[(3) - (3)].lval);
	}
    break;

  case 126:
/* Line 1787 of yacc.c  */
#line 717 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) * (yyvsp[(3) - (3)].lval);
	}
    break;

  case 127:
/* Line 1787 of yacc.c  */
#line 721 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) / (yyvsp[(3) - (3)].lval);
	}
    break;

  case 128:
/* Line 1787 of yacc.c  */
#line 725 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) % (yyvsp[(3) - (3)].lval);
	}
    break;

  case 129:
/* Line 1787 of yacc.c  */
#line 729 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) << (yyvsp[(4) - (4)].lval);
	}
    break;

  case 130:
/* Line 1787 of yacc.c  */
#line 733 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (4)].lval) >> (yyvsp[(4) - (4)].lval);
	}
    break;

  case 131:
/* Line 1787 of yacc.c  */
#line 737 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) & (yyvsp[(3) - (3)].lval);
	}
    break;

  case 132:
/* Line 1787 of yacc.c  */
#line 741 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) ^ (yyvsp[(3) - (3)].lval);
	}
    break;

  case 133:
/* Line 1787 of yacc.c  */
#line 745 "a.y"
    {
		(yyval.lval) = (yyvsp[(1) - (3)].lval) | (yyvsp[(3) - (3)].lval);
	}
    break;


/* Line 1787 of yacc.c  */
#line 2707 "y.tab.c"
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

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


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

#if !defined yyoverflow || YYERROR_VERBOSE
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


