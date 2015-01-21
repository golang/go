//line a.y:32
package main

import __yyfmt__ "fmt"

//line a.y:32
import (
	"cmd/internal/asm"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

//line a.y:41
type yySymType struct {
	yys   int
	sym   *asm.Sym
	lval  int64
	dval  float64
	sval  string
	addr  obj.Addr
	addr2 Addr2
}

const LTYPE0 = 57346
const LTYPE1 = 57347
const LTYPE2 = 57348
const LTYPE3 = 57349
const LTYPE4 = 57350
const LTYPEC = 57351
const LTYPED = 57352
const LTYPEN = 57353
const LTYPER = 57354
const LTYPET = 57355
const LTYPEG = 57356
const LTYPEPC = 57357
const LTYPES = 57358
const LTYPEM = 57359
const LTYPEI = 57360
const LTYPEXC = 57361
const LTYPEX = 57362
const LTYPERT = 57363
const LTYPEF = 57364
const LCONST = 57365
const LFP = 57366
const LPC = 57367
const LSB = 57368
const LBREG = 57369
const LLREG = 57370
const LSREG = 57371
const LFREG = 57372
const LMREG = 57373
const LXREG = 57374
const LFCONST = 57375
const LSCONST = 57376
const LSP = 57377
const LNAME = 57378
const LLAB = 57379
const LVAR = 57380

var yyToknames = []string{
	"'|'",
	"'^'",
	"'&'",
	"'<'",
	"'>'",
	"'+'",
	"'-'",
	"'*'",
	"'/'",
	"'%'",
	"LTYPE0",
	"LTYPE1",
	"LTYPE2",
	"LTYPE3",
	"LTYPE4",
	"LTYPEC",
	"LTYPED",
	"LTYPEN",
	"LTYPER",
	"LTYPET",
	"LTYPEG",
	"LTYPEPC",
	"LTYPES",
	"LTYPEM",
	"LTYPEI",
	"LTYPEXC",
	"LTYPEX",
	"LTYPERT",
	"LTYPEF",
	"LCONST",
	"LFP",
	"LPC",
	"LSB",
	"LBREG",
	"LLREG",
	"LSREG",
	"LFREG",
	"LMREG",
	"LXREG",
	"LFCONST",
	"LSCONST",
	"LSP",
	"LNAME",
	"LLAB",
	"LVAR",
}
var yyStatenames = []string{}

const yyEofCode = 1
const yyErrCode = 2
const yyMaxDepth = 200

//line yacctab:1
var yyExca = []int{
	-1, 1,
	1, -1,
	-2, 2,
}

const yyNprod = 134
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 565

var yyAct = []int{

	49, 61, 186, 123, 38, 3, 81, 80, 51, 62,
	47, 188, 269, 268, 267, 71, 70, 118, 86, 48,
	263, 69, 84, 256, 75, 101, 103, 99, 85, 209,
	112, 254, 59, 112, 170, 242, 240, 82, 238, 222,
	220, 211, 55, 54, 210, 64, 171, 111, 241, 235,
	113, 112, 93, 95, 97, 120, 121, 122, 212, 107,
	109, 55, 133, 128, 174, 145, 52, 138, 119, 71,
	115, 129, 208, 232, 112, 136, 139, 169, 55, 54,
	86, 53, 231, 230, 84, 52, 73, 142, 143, 56,
	85, 146, 224, 60, 144, 131, 130, 223, 57, 82,
	53, 154, 52, 153, 37, 132, 152, 66, 56, 151,
	150, 149, 37, 148, 147, 72, 155, 53, 141, 137,
	135, 68, 73, 134, 62, 56, 176, 177, 127, 34,
	114, 32, 31, 112, 120, 28, 229, 29, 71, 30,
	228, 185, 187, 57, 183, 252, 253, 40, 42, 45,
	41, 43, 46, 195, 194, 44, 248, 112, 112, 112,
	112, 112, 166, 168, 112, 112, 112, 247, 182, 167,
	237, 213, 173, 257, 198, 199, 200, 201, 202, 219,
	120, 205, 206, 207, 184, 114, 196, 197, 165, 164,
	163, 161, 162, 156, 157, 158, 159, 160, 184, 264,
	227, 166, 168, 258, 112, 112, 140, 218, 167, 216,
	236, 55, 54, 55, 54, 239, 246, 261, 217, 260,
	35, 233, 234, 226, 255, 243, 214, 244, 181, 33,
	124, 249, 125, 126, 251, 52, 250, 52, 172, 90,
	116, 189, 190, 191, 192, 193, 259, 117, 89, 245,
	53, 7, 53, 125, 126, 73, 262, 73, 56, 62,
	56, 265, 266, 9, 10, 11, 12, 13, 17, 15,
	18, 14, 16, 25, 26, 19, 20, 21, 22, 23,
	24, 27, 55, 54, 83, 156, 157, 158, 159, 160,
	39, 158, 159, 160, 204, 4, 175, 8, 203, 5,
	6, 110, 2, 55, 54, 1, 52, 77, 108, 106,
	40, 42, 45, 41, 43, 46, 105, 104, 44, 87,
	102, 53, 55, 54, 100, 79, 50, 52, 98, 56,
	96, 40, 42, 45, 41, 43, 46, 94, 92, 44,
	57, 88, 53, 55, 54, 83, 52, 50, 78, 62,
	56, 76, 74, 65, 63, 58, 221, 67, 215, 225,
	0, 53, 0, 0, 55, 54, 73, 52, 0, 56,
	0, 40, 42, 45, 41, 43, 46, 0, 0, 44,
	87, 0, 53, 0, 0, 55, 54, 50, 52, 0,
	56, 0, 40, 42, 45, 41, 43, 46, 0, 0,
	44, 57, 0, 53, 0, 0, 0, 91, 50, 52,
	0, 56, 0, 40, 42, 45, 41, 43, 46, 55,
	54, 44, 57, 0, 53, 0, 0, 0, 36, 50,
	0, 0, 56, 0, 0, 0, 55, 54, 0, 0,
	55, 54, 0, 52, 0, 0, 0, 40, 42, 45,
	41, 43, 46, 55, 54, 44, 57, 0, 53, 0,
	52, 0, 0, 50, 52, 0, 56, 0, 40, 42,
	45, 41, 43, 46, 0, 53, 44, 52, 0, 53,
	73, 0, 188, 56, 50, 55, 54, 56, 0, 0,
	72, 0, 53, 0, 55, 179, 0, 73, 55, 54,
	56, 163, 161, 162, 156, 157, 158, 159, 160, 52,
	161, 162, 156, 157, 158, 159, 160, 0, 52, 180,
	0, 0, 52, 0, 53, 0, 0, 0, 178, 73,
	0, 0, 56, 53, 0, 57, 0, 53, 73, 0,
	0, 56, 50, 0, 0, 56, 165, 164, 163, 161,
	162, 156, 157, 158, 159, 160, 164, 163, 161, 162,
	156, 157, 158, 159, 160,
}
var yyPact = []int{

	-1000, -1000, 249, -1000, 86, -1000, 89, 82, 80, 77,
	376, 294, 294, 410, 69, 97, 489, 273, 355, 294,
	294, 294, 110, -46, -46, 489, 294, 294, -1000, 33,
	-1000, -1000, 33, -1000, -1000, -1000, 410, -1000, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, 17,
	202, 15, -1000, -1000, 33, 33, 33, 223, -1000, 76,
	-1000, -1000, 52, -1000, 71, -1000, 68, -1000, 444, -1000,
	67, 14, 244, 33, -1000, 194, -1000, 66, -1000, 334,
	-1000, -1000, -1000, 431, -1000, -1000, 12, 223, -1000, -1000,
	-1000, 410, -1000, 62, -1000, 61, -1000, 59, -1000, 58,
	-1000, 57, -1000, -1000, -1000, 54, -1000, 51, -1000, 49,
	249, 542, -1000, 542, -1000, 124, 23, -8, 184, 134,
	-1000, -1000, -1000, 11, 288, 33, 33, -1000, -1000, -1000,
	-1000, -1000, 485, 476, 410, 294, -1000, 444, 149, -1000,
	33, 427, -1000, -1000, -1000, 163, 11, 410, 410, 410,
	410, 410, 204, 294, 294, -1000, 33, 33, 33, 33,
	33, 291, 286, 33, 33, 33, 18, -10, -13, 5,
	33, -1000, -1000, 215, 173, 244, -1000, -1000, -14, 313,
	-1000, -1000, -1000, -1000, -15, 45, -1000, 40, 190, 91,
	87, -1000, 31, 30, -1000, 21, -1000, -1000, 280, 280,
	-1000, -1000, -1000, 33, 33, 503, 495, 551, -4, 33,
	-1000, -1000, 132, -16, 33, -18, -1000, -1000, -1000, -5,
	-1000, -19, -1000, -46, -44, -1000, 239, 183, 129, 117,
	33, 110, -46, 276, 276, 107, -23, 213, -1000, -31,
	-1000, 137, -1000, -1000, -1000, 170, 236, -1000, -1000, -1000,
	-1000, -1000, 208, 206, -1000, 33, -1000, -34, -1000, 166,
	33, 33, -40, -1000, -1000, -41, -42, -1000, -1000, -1000,
}
var yyPgo = []int{

	0, 0, 359, 17, 358, 3, 290, 1, 2, 4,
	8, 6, 93, 32, 7, 10, 19, 229, 357, 220,
	355, 354, 353, 352, 351, 348, 341, 338, 337, 330,
	328, 324, 320, 317, 309, 308, 305, 302, 5, 301,
	300,
}
var yyR1 = []int{

	0, 36, 37, 36, 39, 38, 38, 38, 38, 40,
	40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
	40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
	17, 17, 21, 22, 20, 20, 19, 19, 18, 18,
	18, 23, 24, 24, 25, 25, 26, 26, 27, 27,
	28, 28, 29, 29, 29, 30, 31, 32, 32, 33,
	33, 34, 35, 12, 12, 14, 14, 14, 14, 14,
	14, 13, 13, 11, 11, 9, 9, 9, 9, 9,
	9, 9, 8, 7, 7, 7, 7, 7, 7, 7,
	6, 6, 15, 15, 15, 15, 15, 15, 15, 15,
	15, 15, 15, 16, 16, 10, 10, 5, 5, 5,
	4, 4, 4, 1, 1, 1, 1, 1, 1, 2,
	2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3,
}
var yyR2 = []int{

	0, 0, 0, 3, 0, 4, 1, 2, 2, 3,
	3, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	0, 1, 3, 3, 2, 1, 2, 1, 2, 1,
	3, 5, 3, 5, 2, 1, 1, 1, 3, 5,
	3, 5, 2, 1, 3, 5, 5, 0, 1, 3,
	5, 3, 3, 1, 1, 1, 1, 2, 2, 1,
	1, 1, 1, 4, 2, 1, 1, 1, 1, 1,
	1, 1, 2, 2, 2, 2, 2, 4, 5, 3,
	1, 1, 1, 4, 4, 4, 6, 9, 9, 3,
	3, 5, 8, 1, 6, 5, 7, 0, 2, 2,
	1, 1, 1, 1, 1, 2, 2, 2, 3, 1,
	2, 3, 4, 1, 3, 3, 3, 3, 3, 4,
	4, 3, 3, 3,
}
var yyChk = []int{

	-1000, -36, -37, -38, 46, 50, -40, 2, 48, 14,
	15, 16, 17, 18, 22, 20, 23, 19, 21, 26,
	27, 28, 29, 30, 31, 24, 25, 32, 49, 51,
	50, 50, 51, -17, 52, -19, 52, -12, -9, -6,
	37, 40, 38, 41, 45, 39, 42, -15, -16, -1,
	53, -10, 33, 48, 10, 9, 56, 46, -20, -13,
	-12, -7, 55, -21, -13, -22, -12, -18, 52, -11,
	-7, -1, 46, 53, -23, -10, -24, -6, -25, 52,
	-14, -11, -16, 11, -9, -15, -1, 46, -26, -17,
	-19, 52, -27, -13, -28, -13, -29, -13, -30, -9,
	-31, -7, -32, -7, -33, -6, -34, -13, -35, -13,
	-39, -3, -1, -3, -12, 53, 38, 45, -3, 53,
	-1, -1, -1, -5, 7, 9, 10, 52, -1, -10,
	44, 43, 53, 10, 52, 52, -11, 52, 53, -5,
	12, 52, -14, -9, -15, 53, -5, 52, 52, 52,
	52, 52, 52, 52, 52, -38, 9, 10, 11, 12,
	13, 7, 8, 6, 5, 4, 38, 45, 39, 54,
	11, 54, 54, 38, 53, 8, -1, -1, 43, 10,
	43, -12, -13, -11, 35, -1, -8, -1, 55, -12,
	-12, -12, -12, -12, -7, -1, -13, -13, -3, -3,
	-3, -3, -3, 7, 8, -3, -3, -3, 54, 11,
	54, 54, 53, -1, 11, -4, 36, 45, 34, -5,
	54, 43, 54, 52, 52, -2, 33, 10, 49, 49,
	52, 52, 52, -3, -3, 53, -1, 38, 54, -1,
	54, 53, 54, -7, -8, 10, 33, 38, 39, -1,
	-9, -7, 38, 39, 54, 11, 54, 36, 33, 10,
	11, 11, -1, 54, 33, -1, -1, 54, 54, 54,
}
var yyDef = []int{

	1, -2, 0, 3, 0, 6, 0, 0, 0, 30,
	0, 0, 0, 0, 0, 0, 0, 0, 30, 0,
	0, 0, 0, 0, 57, 0, 0, 0, 4, 0,
	7, 8, 0, 11, 31, 12, 0, 37, 63, 64,
	75, 76, 77, 78, 79, 80, 81, 90, 91, 92,
	0, 103, 113, 114, 0, 0, 0, 107, 13, 35,
	71, 72, 0, 14, 0, 15, 0, 16, 0, 39,
	0, 0, 107, 0, 17, 0, 18, 0, 19, 0,
	45, 65, 66, 0, 69, 70, 92, 107, 20, 46,
	47, 31, 21, 0, 22, 0, 23, 53, 24, 0,
	25, 0, 26, 58, 27, 0, 28, 0, 29, 0,
	0, 9, 123, 10, 36, 0, 0, 0, 0, 0,
	115, 116, 117, 0, 0, 0, 0, 34, 83, 84,
	85, 86, 0, 0, 0, 0, 38, 0, 0, 74,
	0, 0, 44, 67, 68, 0, 74, 0, 0, 52,
	0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 99,
	0, 100, 118, 0, 0, 107, 108, 109, 0, 0,
	89, 32, 33, 40, 0, 0, 42, 0, 0, 48,
	50, 54, 0, 0, 59, 0, 61, 62, 124, 125,
	126, 127, 128, 0, 0, 131, 132, 133, 93, 0,
	94, 95, 0, 0, 0, 0, 110, 111, 112, 0,
	87, 0, 73, 0, 0, 82, 119, 0, 0, 0,
	0, 0, 0, 129, 130, 0, 0, 0, 101, 0,
	105, 0, 88, 41, 43, 0, 120, 49, 51, 55,
	56, 60, 0, 0, 96, 0, 104, 0, 121, 0,
	0, 0, 0, 106, 122, 0, 0, 102, 97, 98,
}
var yyTok1 = []int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 55, 13, 6, 3,
	53, 54, 11, 9, 52, 10, 3, 12, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 49, 50,
	7, 51, 8, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 5, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 4, 3, 56,
}
var yyTok2 = []int{

	2, 3, 14, 15, 16, 17, 18, 19, 20, 21,
	22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
	42, 43, 44, 45, 46, 47, 48,
}
var yyTok3 = []int{
	0,
}

//line yaccpar:1

/*	parser for yacc output	*/

var yyDebug = 0

type yyLexer interface {
	Lex(lval *yySymType) int
	Error(s string)
}

const yyFlag = -1000

func yyTokname(c int) string {
	// 4 is TOKSTART above
	if c >= 4 && c-4 < len(yyToknames) {
		if yyToknames[c-4] != "" {
			return yyToknames[c-4]
		}
	}
	return __yyfmt__.Sprintf("tok-%v", c)
}

func yyStatname(s int) string {
	if s >= 0 && s < len(yyStatenames) {
		if yyStatenames[s] != "" {
			return yyStatenames[s]
		}
	}
	return __yyfmt__.Sprintf("state-%v", s)
}

func yylex1(lex yyLexer, lval *yySymType) int {
	c := 0
	char := lex.Lex(lval)
	if char <= 0 {
		c = yyTok1[0]
		goto out
	}
	if char < len(yyTok1) {
		c = yyTok1[char]
		goto out
	}
	if char >= yyPrivate {
		if char < yyPrivate+len(yyTok2) {
			c = yyTok2[char-yyPrivate]
			goto out
		}
	}
	for i := 0; i < len(yyTok3); i += 2 {
		c = yyTok3[i+0]
		if c == char {
			c = yyTok3[i+1]
			goto out
		}
	}

out:
	if c == 0 {
		c = yyTok2[1] /* unknown char */
	}
	if yyDebug >= 3 {
		__yyfmt__.Printf("lex %s(%d)\n", yyTokname(c), uint(char))
	}
	return c
}

func yyParse(yylex yyLexer) int {
	var yyn int
	var yylval yySymType
	var yyVAL yySymType
	yyS := make([]yySymType, yyMaxDepth)

	Nerrs := 0   /* number of errors */
	Errflag := 0 /* error recovery flag */
	yystate := 0
	yychar := -1
	yyp := -1
	goto yystack

ret0:
	return 0

ret1:
	return 1

yystack:
	/* put a state and value onto the stack */
	if yyDebug >= 4 {
		__yyfmt__.Printf("char %v in %v\n", yyTokname(yychar), yyStatname(yystate))
	}

	yyp++
	if yyp >= len(yyS) {
		nyys := make([]yySymType, len(yyS)*2)
		copy(nyys, yyS)
		yyS = nyys
	}
	yyS[yyp] = yyVAL
	yyS[yyp].yys = yystate

yynewstate:
	yyn = yyPact[yystate]
	if yyn <= yyFlag {
		goto yydefault /* simple state */
	}
	if yychar < 0 {
		yychar = yylex1(yylex, &yylval)
	}
	yyn += yychar
	if yyn < 0 || yyn >= yyLast {
		goto yydefault
	}
	yyn = yyAct[yyn]
	if yyChk[yyn] == yychar { /* valid shift */
		yychar = -1
		yyVAL = yylval
		yystate = yyn
		if Errflag > 0 {
			Errflag--
		}
		goto yystack
	}

yydefault:
	/* default state action */
	yyn = yyDef[yystate]
	if yyn == -2 {
		if yychar < 0 {
			yychar = yylex1(yylex, &yylval)
		}

		/* look through exception table */
		xi := 0
		for {
			if yyExca[xi+0] == -1 && yyExca[xi+1] == yystate {
				break
			}
			xi += 2
		}
		for xi += 2; ; xi += 2 {
			yyn = yyExca[xi+0]
			if yyn < 0 || yyn == yychar {
				break
			}
		}
		yyn = yyExca[xi+1]
		if yyn < 0 {
			goto ret0
		}
	}
	if yyn == 0 {
		/* error ... attempt to resume parsing */
		switch Errflag {
		case 0: /* brand new error */
			yylex.Error("syntax error")
			Nerrs++
			if yyDebug >= 1 {
				__yyfmt__.Printf("%s", yyStatname(yystate))
				__yyfmt__.Printf(" saw %s\n", yyTokname(yychar))
			}
			fallthrough

		case 1, 2: /* incompletely recovered error ... try again */
			Errflag = 3

			/* find a state where "error" is a legal shift action */
			for yyp >= 0 {
				yyn = yyPact[yyS[yyp].yys] + yyErrCode
				if yyn >= 0 && yyn < yyLast {
					yystate = yyAct[yyn] /* simulate a shift of "error" */
					if yyChk[yystate] == yyErrCode {
						goto yystack
					}
				}

				/* the current p has no shift on "error", pop stack */
				if yyDebug >= 2 {
					__yyfmt__.Printf("error recovery pops state %d\n", yyS[yyp].yys)
				}
				yyp--
			}
			/* there is no state on the stack with an error shift ... abort */
			goto ret1

		case 3: /* no shift yet; clobber input char */
			if yyDebug >= 2 {
				__yyfmt__.Printf("error recovery discards %s\n", yyTokname(yychar))
			}
			if yychar == yyEofCode {
				goto ret1
			}
			yychar = -1
			goto yynewstate /* try again in the same state */
		}
	}

	/* reduction by production yyn */
	if yyDebug >= 2 {
		__yyfmt__.Printf("reduce %v in:\n\t%v\n", yyn, yyStatname(yystate))
	}

	yynt := yyn
	yypt := yyp
	_ = yypt // guard against "declared and not used"

	yyp -= yyR2[yyn]
	// yyp is now the index of $0. Perform the default action. Iff the
	// reduced production is ε, $1 is possibly out of range.
	if yyp+1 >= len(yyS) {
		nyys := make([]yySymType, len(yyS)*2)
		copy(nyys, yyS)
		yyS = nyys
	}
	yyVAL = yyS[yyp+1]

	/* consult goto table to find next state */
	yyn = yyR1[yyn]
	yyg := yyPgo[yyn]
	yyj := yyg + yyS[yyp].yys + 1

	if yyj >= yyLast {
		yystate = yyAct[yyg]
	} else {
		yystate = yyAct[yyj]
		if yyChk[yystate] != -yyn {
			yystate = yyAct[yyg]
		}
	}
	// dummy call; replaced with literal code
	switch yynt {

	case 2:
		//line a.y:72
		{
			stmtline = asm.Lineno
		}
	case 4:
		//line a.y:79
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			if yyS[yypt-1].sym.Type == LLAB && yyS[yypt-1].sym.Value != int64(asm.PC) {
				yyerror("redeclaration of %s (%s)", yyS[yypt-1].sym.Labelname, yyS[yypt-1].sym.Name)
			}
			yyS[yypt-1].sym.Type = LLAB
			yyS[yypt-1].sym.Value = int64(asm.PC)
		}
	case 9:
		//line a.y:94
		{
			yyS[yypt-2].sym.Type = LVAR
			yyS[yypt-2].sym.Value = yyS[yypt-0].lval
		}
	case 10:
		//line a.y:99
		{
			if yyS[yypt-2].sym.Value != yyS[yypt-0].lval {
				yyerror("redeclaration of %s", yyS[yypt-2].sym.Name)
			}
			yyS[yypt-2].sym.Value = yyS[yypt-0].lval
		}
	case 11:
		//line a.y:105
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 12:
		//line a.y:106
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 13:
		//line a.y:107
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 14:
		//line a.y:108
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 15:
		//line a.y:109
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 16:
		//line a.y:110
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 17:
		//line a.y:111
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 18:
		//line a.y:112
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 19:
		//line a.y:113
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 20:
		//line a.y:114
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 21:
		//line a.y:115
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 22:
		//line a.y:116
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 23:
		//line a.y:117
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 24:
		//line a.y:118
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 25:
		//line a.y:119
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 26:
		//line a.y:120
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 27:
		//line a.y:121
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 28:
		//line a.y:122
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 29:
		//line a.y:123
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 30:
		//line a.y:126
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 31:
		//line a.y:131
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 32:
		//line a.y:138
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 33:
		//line a.y:145
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 34:
		//line a.y:152
		{
			yyVAL.addr2.from = yyS[yypt-1].addr
			yyVAL.addr2.to = nullgen
		}
	case 35:
		//line a.y:157
		{
			yyVAL.addr2.from = yyS[yypt-0].addr
			yyVAL.addr2.to = nullgen
		}
	case 36:
		//line a.y:164
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 37:
		//line a.y:169
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 38:
		//line a.y:176
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 39:
		//line a.y:181
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 40:
		//line a.y:186
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 41:
		//line a.y:193
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.from.Scale = int8(yyS[yypt-2].lval)
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 42:
		//line a.y:201
		{
			asm.Settext(yyS[yypt-2].addr.Sym)
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 43:
		//line a.y:207
		{
			asm.Settext(yyS[yypt-4].addr.Sym)
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.from.Scale = int8(yyS[yypt-2].lval)
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 44:
		//line a.y:216
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 45:
		//line a.y:221
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 46:
		yyVAL.addr2 = yyS[yypt-0].addr2
	case 47:
		yyVAL.addr2 = yyS[yypt-0].addr2
	case 48:
		//line a.y:232
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 49:
		//line a.y:237
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.to = yyS[yypt-2].addr
			if yyVAL.addr2.from.Index != x86.D_NONE {
				yyerror("dp shift with lhs index")
			}
			yyVAL.addr2.from.Index = uint8(yyS[yypt-0].lval)
		}
	case 50:
		//line a.y:248
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 51:
		//line a.y:253
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.to = yyS[yypt-2].addr
			if yyVAL.addr2.to.Index != x86.D_NONE {
				yyerror("dp move with lhs index")
			}
			yyVAL.addr2.to.Index = uint8(yyS[yypt-0].lval)
		}
	case 52:
		//line a.y:264
		{
			yyVAL.addr2.from = yyS[yypt-1].addr
			yyVAL.addr2.to = nullgen
		}
	case 53:
		//line a.y:269
		{
			yyVAL.addr2.from = yyS[yypt-0].addr
			yyVAL.addr2.to = nullgen
		}
	case 54:
		//line a.y:274
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 55:
		//line a.y:281
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.to = yyS[yypt-2].addr
			yyVAL.addr2.to.Offset = yyS[yypt-0].lval
		}
	case 56:
		//line a.y:289
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
			if yyS[yypt-4].addr.Type != x86.D_CONST {
				yyerror("illegal constant")
			}
			yyVAL.addr2.to.Offset = yyS[yypt-4].addr.Offset
		}
	case 57:
		//line a.y:299
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 58:
		//line a.y:304
		{
			yyVAL.addr2.from = yyS[yypt-0].addr
			yyVAL.addr2.to = nullgen
		}
	case 59:
		//line a.y:311
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 60:
		//line a.y:316
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.from.Scale = int8(yyS[yypt-2].lval)
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 61:
		//line a.y:324
		{
			if yyS[yypt-2].addr.Type != x86.D_CONST || yyS[yypt-0].addr.Type != x86.D_CONST {
				yyerror("arguments to asm.PCDATA must be integer constants")
			}
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 62:
		//line a.y:334
		{
			if yyS[yypt-2].addr.Type != x86.D_CONST {
				yyerror("index for FUNCDATA must be integer constant")
			}
			if yyS[yypt-0].addr.Type != x86.D_EXTERN && yyS[yypt-0].addr.Type != x86.D_STATIC {
				yyerror("value for FUNCDATA must be symbol reference")
			}
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 63:
		yyVAL.addr = yyS[yypt-0].addr
	case 64:
		yyVAL.addr = yyS[yypt-0].addr
	case 65:
		yyVAL.addr = yyS[yypt-0].addr
	case 66:
		yyVAL.addr = yyS[yypt-0].addr
	case 67:
		//line a.y:353
		{
			yyVAL.addr = yyS[yypt-0].addr
		}
	case 68:
		//line a.y:357
		{
			yyVAL.addr = yyS[yypt-0].addr
		}
	case 69:
		yyVAL.addr = yyS[yypt-0].addr
	case 70:
		yyVAL.addr = yyS[yypt-0].addr
	case 71:
		yyVAL.addr = yyS[yypt-0].addr
	case 72:
		yyVAL.addr = yyS[yypt-0].addr
	case 73:
		//line a.y:369
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_BRANCH
			yyVAL.addr.Offset = yyS[yypt-3].lval + int64(asm.PC)
		}
	case 74:
		//line a.y:375
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyS[yypt-1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyS[yypt-1].sym.Labelname)
			}
			yyVAL.addr.Type = x86.D_BRANCH
			yyVAL.addr.Offset = yyS[yypt-1].sym.Value + yyS[yypt-0].lval
		}
	case 75:
		//line a.y:387
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 76:
		//line a.y:392
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 77:
		//line a.y:397
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 78:
		//line a.y:402
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 79:
		//line a.y:407
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_SP
		}
	case 80:
		//line a.y:412
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 81:
		//line a.y:417
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 82:
		//line a.y:423
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_CONST
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 83:
		//line a.y:431
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_CONST
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 84:
		//line a.y:437
		{
			yyVAL.addr = yyS[yypt-0].addr
			yyVAL.addr.Index = uint8(yyS[yypt-0].addr.Type)
			yyVAL.addr.Type = x86.D_ADDR
			/*
				if($2.Type == x86.D_AUTO || $2.Type == x86.D_PARAM)
					yyerror("constant cannot be automatic: %s",
						$2.sym.Name);
			*/
		}
	case 85:
		//line a.y:447
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_SCONST
			yyVAL.addr.U.Sval = (yyS[yypt-0].sval + "\x00\x00\x00\x00\x00\x00\x00\x00")[:8]
		}
	case 86:
		//line a.y:453
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_FCONST
			yyVAL.addr.U.Dval = yyS[yypt-0].dval
		}
	case 87:
		//line a.y:459
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_FCONST
			yyVAL.addr.U.Dval = yyS[yypt-1].dval
		}
	case 88:
		//line a.y:465
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_FCONST
			yyVAL.addr.U.Dval = -yyS[yypt-1].dval
		}
	case 89:
		//line a.y:471
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_FCONST
			yyVAL.addr.U.Dval = -yyS[yypt-0].dval
		}
	case 90:
		yyVAL.addr = yyS[yypt-0].addr
	case 91:
		yyVAL.addr = yyS[yypt-0].addr
	case 92:
		//line a.y:483
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_INDIR + x86.D_NONE
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 93:
		//line a.y:489
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + yyS[yypt-1].lval)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 94:
		//line a.y:495
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + x86.D_SP)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 95:
		//line a.y:501
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + yyS[yypt-1].lval)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 96:
		//line a.y:507
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + x86.D_NONE)
			yyVAL.addr.Offset = yyS[yypt-5].lval
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 97:
		//line a.y:516
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + yyS[yypt-6].lval)
			yyVAL.addr.Offset = yyS[yypt-8].lval
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 98:
		//line a.y:525
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + yyS[yypt-6].lval)
			yyVAL.addr.Offset = yyS[yypt-8].lval
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 99:
		//line a.y:534
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + yyS[yypt-1].lval)
		}
	case 100:
		//line a.y:539
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + x86.D_SP)
		}
	case 101:
		//line a.y:544
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + x86.D_NONE)
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 102:
		//line a.y:552
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(x86.D_INDIR + yyS[yypt-6].lval)
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 103:
		//line a.y:562
		{
			yyVAL.addr = yyS[yypt-0].addr
		}
	case 104:
		//line a.y:566
		{
			yyVAL.addr = yyS[yypt-5].addr
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 105:
		//line a.y:575
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-1].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-4].sym.Name, 0)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 106:
		//line a.y:582
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = x86.D_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-6].sym.Name, 1)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 107:
		//line a.y:590
		{
			yyVAL.lval = 0
		}
	case 108:
		//line a.y:594
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 109:
		//line a.y:598
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 110:
		yyVAL.lval = yyS[yypt-0].lval
	case 111:
		//line a.y:605
		{
			yyVAL.lval = x86.D_AUTO
		}
	case 112:
		yyVAL.lval = yyS[yypt-0].lval
	case 113:
		yyVAL.lval = yyS[yypt-0].lval
	case 114:
		//line a.y:613
		{
			yyVAL.lval = yyS[yypt-0].sym.Value
		}
	case 115:
		//line a.y:617
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 116:
		//line a.y:621
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 117:
		//line a.y:625
		{
			yyVAL.lval = ^yyS[yypt-0].lval
		}
	case 118:
		//line a.y:629
		{
			yyVAL.lval = yyS[yypt-1].lval
		}
	case 119:
		//line a.y:635
		{
			yyVAL.lval = int64(uint64(yyS[yypt-0].lval&0xffffffff) + (obj.ArgsSizeUnknown << 32))
		}
	case 120:
		//line a.y:639
		{
			yyVAL.lval = int64(uint64(-yyS[yypt-0].lval&0xffffffff) + (obj.ArgsSizeUnknown << 32))
		}
	case 121:
		//line a.y:643
		{
			yyVAL.lval = (yyS[yypt-2].lval & 0xffffffff) + ((yyS[yypt-0].lval & 0xffff) << 32)
		}
	case 122:
		//line a.y:647
		{
			yyVAL.lval = (-yyS[yypt-2].lval & 0xffffffff) + ((yyS[yypt-0].lval & 0xffff) << 32)
		}
	case 123:
		yyVAL.lval = yyS[yypt-0].lval
	case 124:
		//line a.y:654
		{
			yyVAL.lval = yyS[yypt-2].lval + yyS[yypt-0].lval
		}
	case 125:
		//line a.y:658
		{
			yyVAL.lval = yyS[yypt-2].lval - yyS[yypt-0].lval
		}
	case 126:
		//line a.y:662
		{
			yyVAL.lval = yyS[yypt-2].lval * yyS[yypt-0].lval
		}
	case 127:
		//line a.y:666
		{
			yyVAL.lval = yyS[yypt-2].lval / yyS[yypt-0].lval
		}
	case 128:
		//line a.y:670
		{
			yyVAL.lval = yyS[yypt-2].lval % yyS[yypt-0].lval
		}
	case 129:
		//line a.y:674
		{
			yyVAL.lval = yyS[yypt-3].lval << uint(yyS[yypt-0].lval)
		}
	case 130:
		//line a.y:678
		{
			yyVAL.lval = yyS[yypt-3].lval >> uint(yyS[yypt-0].lval)
		}
	case 131:
		//line a.y:682
		{
			yyVAL.lval = yyS[yypt-2].lval & yyS[yypt-0].lval
		}
	case 132:
		//line a.y:686
		{
			yyVAL.lval = yyS[yypt-2].lval ^ yyS[yypt-0].lval
		}
	case 133:
		//line a.y:690
		{
			yyVAL.lval = yyS[yypt-2].lval | yyS[yypt-0].lval
		}
	}
	goto yystack /* stack new state and value */
}
