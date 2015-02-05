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

const yyNprod = 133
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 593

var yyAct = []int{

	52, 227, 41, 3, 80, 208, 269, 64, 123, 50,
	51, 79, 54, 170, 268, 74, 267, 118, 85, 72,
	83, 263, 73, 255, 253, 98, 241, 84, 81, 239,
	237, 100, 102, 112, 221, 219, 112, 210, 209, 171,
	240, 107, 234, 62, 211, 174, 143, 138, 65, 207,
	111, 119, 115, 113, 112, 231, 67, 169, 120, 121,
	122, 249, 230, 92, 94, 96, 128, 226, 225, 224,
	104, 106, 74, 58, 57, 154, 136, 112, 129, 85,
	153, 83, 151, 150, 139, 141, 149, 148, 84, 81,
	140, 147, 142, 146, 145, 144, 63, 55, 58, 57,
	137, 43, 45, 48, 44, 46, 49, 40, 135, 47,
	69, 134, 56, 127, 155, 40, 34, 37, 53, 31,
	59, 32, 55, 35, 33, 223, 176, 177, 222, 217,
	60, 215, 220, 112, 120, 243, 114, 56, 74, 242,
	216, 236, 183, 76, 173, 59, 58, 57, 256, 166,
	168, 251, 252, 192, 194, 196, 167, 112, 112, 112,
	112, 112, 195, 184, 112, 112, 112, 264, 58, 57,
	55, 212, 257, 248, 197, 198, 199, 200, 201, 182,
	120, 204, 205, 206, 218, 56, 42, 114, 152, 38,
	65, 76, 55, 59, 190, 191, 184, 261, 260, 166,
	168, 229, 258, 112, 112, 75, 167, 56, 89, 235,
	36, 71, 65, 76, 238, 59, 108, 109, 254, 213,
	232, 233, 125, 126, 228, 244, 247, 203, 245, 88,
	124, 181, 125, 126, 246, 158, 159, 160, 175, 250,
	202, 25, 185, 186, 187, 188, 189, 16, 15, 6,
	110, 259, 7, 2, 1, 262, 156, 157, 158, 159,
	160, 265, 266, 105, 9, 10, 11, 12, 13, 17,
	28, 18, 14, 29, 30, 26, 19, 20, 21, 22,
	23, 24, 27, 58, 57, 82, 165, 164, 163, 161,
	162, 156, 157, 158, 159, 160, 4, 103, 8, 101,
	5, 99, 97, 58, 57, 95, 93, 55, 91, 87,
	77, 43, 45, 48, 44, 46, 49, 68, 66, 47,
	86, 61, 56, 70, 214, 0, 78, 55, 53, 0,
	59, 43, 45, 48, 44, 46, 49, 172, 0, 47,
	60, 0, 56, 58, 57, 82, 0, 65, 53, 0,
	59, 43, 45, 48, 44, 46, 49, 0, 0, 47,
	0, 0, 0, 58, 57, 0, 0, 55, 0, 0,
	0, 43, 45, 48, 44, 46, 49, 0, 0, 47,
	86, 0, 56, 58, 57, 0, 0, 55, 53, 0,
	59, 43, 45, 48, 44, 46, 49, 0, 0, 47,
	60, 0, 56, 58, 57, 0, 90, 55, 53, 0,
	59, 43, 45, 48, 44, 46, 49, 58, 133, 47,
	60, 0, 56, 0, 0, 0, 39, 55, 53, 0,
	59, 43, 45, 48, 44, 46, 49, 58, 57, 47,
	60, 55, 56, 0, 58, 57, 0, 0, 53, 0,
	59, 131, 130, 0, 60, 0, 56, 58, 57, 0,
	0, 55, 132, 0, 59, 0, 116, 0, 55, 58,
	57, 0, 0, 117, 0, 0, 56, 0, 0, 0,
	0, 55, 76, 56, 59, 58, 179, 0, 193, 76,
	0, 59, 0, 55, 75, 0, 56, 58, 57, 0,
	0, 0, 76, 180, 59, 0, 0, 0, 56, 55,
	0, 58, 57, 0, 76, 0, 59, 0, 0, 178,
	0, 55, 0, 0, 56, 0, 0, 0, 0, 0,
	76, 0, 59, 0, 60, 55, 56, 0, 0, 0,
	0, 0, 53, 0, 59, 0, 0, 0, 0, 0,
	56, 0, 0, 0, 0, 0, 76, 0, 59, 165,
	164, 163, 161, 162, 156, 157, 158, 159, 160, 164,
	163, 161, 162, 156, 157, 158, 159, 160, 163, 161,
	162, 156, 157, 158, 159, 160, 161, 162, 156, 157,
	158, 159, 160,
}
var yyPact = []int{

	-1000, -1000, 250, -1000, 70, -1000, 74, 66, 72, 65,
	374, 294, 294, 394, 159, -1000, -1000, 274, 354, 294,
	294, 294, 314, -5, -5, -1000, 294, 294, 84, 488,
	488, -1000, 502, -1000, -1000, 502, -1000, -1000, -1000, 394,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, -1000, -2, 428, -3, -1000, -1000, 502, 502, 502,
	223, -1000, 61, -1000, -1000, 408, -1000, 59, -1000, 56,
	-1000, 448, -1000, 48, -7, 213, 502, -1000, 334, -1000,
	-1000, -1000, 64, -1000, -1000, -8, 223, -1000, -1000, -1000,
	394, -1000, 42, -1000, 41, -1000, 39, -1000, 35, -1000,
	34, -1000, -1000, -1000, 31, -1000, 30, 176, 28, 23,
	250, 555, -1000, 555, -1000, 111, 2, -16, 282, 106,
	-1000, -1000, -1000, -9, 230, 502, 502, -1000, -1000, -1000,
	-1000, -1000, 476, 460, 394, 294, -1000, 448, 128, -1000,
	-1000, -1000, -1000, 161, -9, 394, 394, 394, 394, 394,
	294, 294, 502, 435, 137, -1000, 502, 502, 502, 502,
	502, 233, 219, 502, 502, 502, -6, -17, -18, -10,
	502, -1000, -1000, 208, 95, 213, -1000, -1000, -20, 89,
	-1000, -1000, -1000, -1000, -21, 79, 76, -1000, 17, 16,
	-1000, -1000, 15, 191, 10, -1000, 3, 224, 224, -1000,
	-1000, -1000, 502, 502, 579, 572, 564, -12, 502, -1000,
	-1000, 103, -25, 502, -26, -1000, -1000, -1000, -14, -1000,
	-29, -1000, 101, 96, 502, 314, -5, -1000, 216, 140,
	8, -5, 247, 247, 113, -31, 207, -1000, -32, -1000,
	112, -1000, -1000, -1000, -1000, -1000, -1000, 139, 192, 191,
	-1000, 187, 186, -1000, 502, -1000, -34, -1000, 134, -1000,
	502, 502, -39, -1000, -1000, -41, -49, -1000, -1000, -1000,
}
var yyPgo = []int{

	0, 0, 17, 324, 8, 186, 7, 1, 2, 12,
	4, 96, 43, 11, 9, 10, 210, 323, 189, 321,
	318, 317, 310, 309, 308, 306, 305, 302, 301, 299,
	297, 263, 254, 253, 3, 250, 249, 248, 247, 241,
}
var yyR1 = []int{

	0, 32, 33, 32, 35, 34, 34, 34, 34, 36,
	36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
	36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
	16, 16, 20, 21, 19, 19, 18, 18, 17, 17,
	17, 37, 38, 38, 39, 39, 22, 22, 23, 23,
	24, 24, 25, 25, 26, 26, 26, 27, 28, 29,
	29, 30, 31, 11, 11, 13, 13, 13, 13, 13,
	13, 12, 12, 10, 10, 8, 8, 8, 8, 8,
	8, 8, 6, 6, 6, 6, 6, 6, 6, 5,
	5, 14, 14, 14, 14, 14, 14, 14, 14, 14,
	14, 14, 15, 15, 9, 9, 4, 4, 4, 3,
	3, 3, 1, 1, 1, 1, 1, 1, 7, 7,
	7, 7, 2, 2, 2, 2, 2, 2, 2, 2,
	2, 2, 2,
}
var yyR2 = []int{

	0, 0, 0, 3, 0, 4, 1, 2, 2, 3,
	3, 2, 2, 2, 2, 2, 2, 1, 1, 2,
	2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
	0, 1, 3, 3, 2, 1, 2, 1, 2, 1,
	3, 6, 5, 7, 4, 6, 2, 1, 1, 1,
	3, 5, 3, 5, 2, 1, 3, 5, 5, 0,
	1, 3, 3, 1, 1, 1, 1, 2, 2, 1,
	1, 1, 1, 4, 2, 1, 1, 1, 1, 1,
	1, 1, 2, 2, 2, 2, 4, 5, 3, 1,
	1, 1, 4, 4, 4, 6, 9, 9, 3, 3,
	5, 8, 1, 6, 5, 7, 0, 2, 2, 1,
	1, 1, 1, 1, 2, 2, 2, 3, 1, 2,
	3, 4, 1, 3, 3, 3, 3, 3, 4, 4,
	3, 3, 3,
}
var yyChk = []int{

	-1000, -32, -33, -34, 46, 50, -36, 2, 48, 14,
	15, 16, 17, 18, 22, -37, -38, 19, 21, 26,
	27, 28, 29, 30, 31, -39, 25, 32, 20, 23,
	24, 49, 51, 50, 50, 51, -16, 52, -18, 52,
	-11, -8, -5, 37, 40, 38, 41, 45, 39, 42,
	-14, -15, -1, 54, -9, 33, 48, 10, 9, 56,
	46, -19, -12, -11, -6, 53, -20, -12, -21, -11,
	-17, 52, -10, -6, -1, 46, 54, -22, 52, -13,
	-10, -15, 11, -8, -14, -1, 46, -23, -16, -18,
	52, -24, -12, -25, -12, -26, -12, -27, -8, -28,
	-6, -29, -6, -30, -12, -31, -12, -9, -5, -5,
	-35, -2, -1, -2, -11, 54, 38, 45, -2, 54,
	-1, -1, -1, -4, 7, 9, 10, 52, -1, -9,
	44, 43, 54, 10, 52, 52, -10, 52, 54, -4,
	-13, -8, -14, 54, -4, 52, 52, 52, 52, 52,
	52, 52, 12, 52, 52, -34, 9, 10, 11, 12,
	13, 7, 8, 6, 5, 4, 38, 45, 39, 55,
	11, 55, 55, 38, 54, 8, -1, -1, 43, 10,
	43, -11, -12, -10, 35, -11, -11, -11, -11, -11,
	-12, -12, -1, 53, -1, -6, -1, -2, -2, -2,
	-2, -2, 7, 8, -2, -2, -2, 55, 11, 55,
	55, 54, -1, 11, -3, 36, 45, 34, -4, 55,
	43, 55, 49, 49, 52, 52, 52, -7, 33, 10,
	52, 52, -2, -2, 54, -1, 38, 55, -1, 55,
	54, 55, 38, 39, -1, -8, -6, 10, 33, 53,
	-6, 38, 39, 55, 11, 55, 36, 33, 10, -7,
	11, 11, -1, 55, 33, -1, -1, 55, 55, 55,
}
var yyDef = []int{

	1, -2, 0, 3, 0, 6, 0, 0, 0, 30,
	0, 0, 0, 0, 0, 17, 18, 0, 30, 0,
	0, 0, 0, 0, 59, 27, 0, 0, 0, 0,
	0, 4, 0, 7, 8, 0, 11, 31, 12, 0,
	37, 63, 64, 75, 76, 77, 78, 79, 80, 81,
	89, 90, 91, 0, 102, 112, 113, 0, 0, 0,
	106, 13, 35, 71, 72, 0, 14, 0, 15, 0,
	16, 0, 39, 0, 0, 106, 0, 19, 0, 47,
	65, 66, 0, 69, 70, 91, 106, 20, 48, 49,
	31, 21, 0, 22, 0, 23, 55, 24, 0, 25,
	0, 26, 60, 28, 0, 29, 0, 0, 0, 0,
	0, 9, 122, 10, 36, 0, 0, 0, 0, 0,
	114, 115, 116, 0, 0, 0, 0, 34, 82, 83,
	84, 85, 0, 0, 0, 0, 38, 0, 0, 74,
	46, 67, 68, 0, 74, 0, 0, 54, 0, 0,
	0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 98,
	0, 99, 117, 0, 0, 106, 107, 108, 0, 0,
	88, 32, 33, 40, 0, 50, 52, 56, 0, 0,
	61, 62, 0, 0, 0, 44, 0, 123, 124, 125,
	126, 127, 0, 0, 130, 131, 132, 92, 0, 93,
	94, 0, 0, 0, 0, 109, 110, 111, 0, 86,
	0, 73, 0, 0, 0, 0, 0, 42, 118, 0,
	0, 0, 128, 129, 0, 0, 0, 100, 0, 104,
	0, 87, 51, 53, 57, 58, 41, 0, 119, 0,
	45, 0, 0, 95, 0, 103, 0, 120, 0, 43,
	0, 0, 0, 105, 121, 0, 0, 101, 96, 97,
}
var yyTok1 = []int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 53, 13, 6, 3,
	54, 55, 11, 9, 52, 10, 3, 12, 3, 3,
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
			var a Addr2
			a.from = yyS[yypt-4].addr
			a.to = yyS[yypt-0].addr
			outcode(obj.ADATA, &a)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyS[yypt-2].lval
			}
		}
	case 42:
		//line a.y:206
		{
			asm.Settext(yyS[yypt-3].addr.Sym)
			outcode(obj.ATEXT, &Addr2{yyS[yypt-3].addr, yyS[yypt-0].addr})
		}
	case 43:
		//line a.y:211
		{
			asm.Settext(yyS[yypt-5].addr.Sym)
			outcode(obj.ATEXT, &Addr2{yyS[yypt-5].addr, yyS[yypt-0].addr})
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyS[yypt-3].lval
			}
		}
	case 44:
		//line a.y:222
		{
			asm.Settext(yyS[yypt-2].addr.Sym)
			outcode(obj.AGLOBL, &Addr2{yyS[yypt-2].addr, yyS[yypt-0].addr})
		}
	case 45:
		//line a.y:227
		{
			asm.Settext(yyS[yypt-4].addr.Sym)
			outcode(obj.AGLOBL, &Addr2{yyS[yypt-4].addr, yyS[yypt-0].addr})
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyS[yypt-2].lval
			}
		}
	case 46:
		//line a.y:238
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 47:
		//line a.y:243
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 48:
		yyVAL.addr2 = yyS[yypt-0].addr2
	case 49:
		yyVAL.addr2 = yyS[yypt-0].addr2
	case 50:
		//line a.y:254
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 51:
		//line a.y:259
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.to = yyS[yypt-2].addr
			if yyVAL.addr2.from.Index != obj.TYPE_NONE {
				yyerror("dp shift with lhs index")
			}
			yyVAL.addr2.from.Index = int16(yyS[yypt-0].lval)
		}
	case 52:
		//line a.y:270
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 53:
		//line a.y:275
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.to = yyS[yypt-2].addr
			if yyVAL.addr2.to.Index != obj.TYPE_NONE {
				yyerror("dp move with lhs index")
			}
			yyVAL.addr2.to.Index = int16(yyS[yypt-0].lval)
		}
	case 54:
		//line a.y:286
		{
			yyVAL.addr2.from = yyS[yypt-1].addr
			yyVAL.addr2.to = nullgen
		}
	case 55:
		//line a.y:291
		{
			yyVAL.addr2.from = yyS[yypt-0].addr
			yyVAL.addr2.to = nullgen
		}
	case 56:
		//line a.y:296
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 57:
		//line a.y:303
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.to = yyS[yypt-2].addr
			yyVAL.addr2.to.Offset = yyS[yypt-0].lval
		}
	case 58:
		//line a.y:311
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
			if yyS[yypt-4].addr.Type != obj.TYPE_CONST {
				yyerror("illegal constant")
			}
			yyVAL.addr2.to.Offset = yyS[yypt-4].addr.Offset
		}
	case 59:
		//line a.y:321
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 60:
		//line a.y:326
		{
			yyVAL.addr2.from = yyS[yypt-0].addr
			yyVAL.addr2.to = nullgen
		}
	case 61:
		//line a.y:333
		{
			if yyS[yypt-2].addr.Type != obj.TYPE_CONST || yyS[yypt-0].addr.Type != obj.TYPE_CONST {
				yyerror("arguments to asm.PCDATA must be integer constants")
			}
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 62:
		//line a.y:343
		{
			if yyS[yypt-2].addr.Type != obj.TYPE_CONST {
				yyerror("index for FUNCDATA must be integer constant")
			}
			if yyS[yypt-0].addr.Type != obj.TYPE_MEM || (yyS[yypt-0].addr.Name != obj.NAME_EXTERN && yyS[yypt-0].addr.Name != obj.NAME_STATIC) {
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
		//line a.y:362
		{
			yyVAL.addr = yyS[yypt-0].addr
		}
	case 68:
		//line a.y:366
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
		//line a.y:378
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyS[yypt-3].lval + int64(asm.PC)
		}
	case 74:
		//line a.y:384
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyS[yypt-1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyS[yypt-1].sym.Labelname)
			}
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyS[yypt-1].sym.Value + yyS[yypt-0].lval
		}
	case 75:
		//line a.y:396
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 76:
		//line a.y:402
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 77:
		//line a.y:408
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 78:
		//line a.y:414
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 79:
		//line a.y:420
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = x86.REG_SP
		}
	case 80:
		//line a.y:426
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 81:
		//line a.y:432
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 82:
		//line a.y:440
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 83:
		//line a.y:446
		{
			yyVAL.addr = yyS[yypt-0].addr
			yyVAL.addr.Type = obj.TYPE_ADDR
			/*
				if($2.Type == x86.D_AUTO || $2.Type == x86.D_PARAM)
					yyerror("constant cannot be automatic: %s",
						$2.sym.Name);
			*/
		}
	case 84:
		//line a.y:455
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SCONST
			yyVAL.addr.U.Sval = (yyS[yypt-0].sval + "\x00\x00\x00\x00\x00\x00\x00\x00")[:8]
		}
	case 85:
		//line a.y:461
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = yyS[yypt-0].dval
		}
	case 86:
		//line a.y:467
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = yyS[yypt-1].dval
		}
	case 87:
		//line a.y:473
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = -yyS[yypt-1].dval
		}
	case 88:
		//line a.y:479
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = -yyS[yypt-0].dval
		}
	case 89:
		yyVAL.addr = yyS[yypt-0].addr
	case 90:
		yyVAL.addr = yyS[yypt-0].addr
	case 91:
		//line a.y:491
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 92:
		//line a.y:497
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyS[yypt-1].lval)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 93:
		//line a.y:504
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = x86.REG_SP
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 94:
		//line a.y:511
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyS[yypt-1].lval)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 95:
		//line a.y:518
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Offset = yyS[yypt-5].lval
			yyVAL.addr.Index = int16(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 96:
		//line a.y:527
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyS[yypt-6].lval)
			yyVAL.addr.Offset = yyS[yypt-8].lval
			yyVAL.addr.Index = int16(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 97:
		//line a.y:537
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyS[yypt-6].lval)
			yyVAL.addr.Offset = yyS[yypt-8].lval
			yyVAL.addr.Index = int16(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 98:
		//line a.y:547
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyS[yypt-1].lval)
		}
	case 99:
		//line a.y:553
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = x86.REG_SP
		}
	case 100:
		//line a.y:559
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Index = int16(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 101:
		//line a.y:567
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyS[yypt-6].lval)
			yyVAL.addr.Index = int16(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 102:
		//line a.y:578
		{
			yyVAL.addr = yyS[yypt-0].addr
		}
	case 103:
		//line a.y:582
		{
			yyVAL.addr = yyS[yypt-5].addr
			yyVAL.addr.Index = int16(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 104:
		//line a.y:591
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyS[yypt-1].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-4].sym.Name, 0)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 105:
		//line a.y:599
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = obj.NAME_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-6].sym.Name, 1)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 106:
		//line a.y:608
		{
			yyVAL.lval = 0
		}
	case 107:
		//line a.y:612
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 108:
		//line a.y:616
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 109:
		yyVAL.lval = yyS[yypt-0].lval
	case 110:
		//line a.y:623
		{
			yyVAL.lval = obj.NAME_AUTO
		}
	case 111:
		yyVAL.lval = yyS[yypt-0].lval
	case 112:
		yyVAL.lval = yyS[yypt-0].lval
	case 113:
		//line a.y:631
		{
			yyVAL.lval = yyS[yypt-0].sym.Value
		}
	case 114:
		//line a.y:635
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 115:
		//line a.y:639
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 116:
		//line a.y:643
		{
			yyVAL.lval = ^yyS[yypt-0].lval
		}
	case 117:
		//line a.y:647
		{
			yyVAL.lval = yyS[yypt-1].lval
		}
	case 118:
		//line a.y:653
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = yyS[yypt-0].lval
			yyVAL.addr.U.Argsize = obj.ArgsSizeUnknown
		}
	case 119:
		//line a.y:660
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -yyS[yypt-0].lval
			yyVAL.addr.U.Argsize = obj.ArgsSizeUnknown
		}
	case 120:
		//line a.y:667
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = yyS[yypt-2].lval
			yyVAL.addr.U.Argsize = int32(yyS[yypt-0].lval)
		}
	case 121:
		//line a.y:674
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -yyS[yypt-2].lval
			yyVAL.addr.U.Argsize = int32(yyS[yypt-0].lval)
		}
	case 122:
		yyVAL.lval = yyS[yypt-0].lval
	case 123:
		//line a.y:684
		{
			yyVAL.lval = yyS[yypt-2].lval + yyS[yypt-0].lval
		}
	case 124:
		//line a.y:688
		{
			yyVAL.lval = yyS[yypt-2].lval - yyS[yypt-0].lval
		}
	case 125:
		//line a.y:692
		{
			yyVAL.lval = yyS[yypt-2].lval * yyS[yypt-0].lval
		}
	case 126:
		//line a.y:696
		{
			yyVAL.lval = yyS[yypt-2].lval / yyS[yypt-0].lval
		}
	case 127:
		//line a.y:700
		{
			yyVAL.lval = yyS[yypt-2].lval % yyS[yypt-0].lval
		}
	case 128:
		//line a.y:704
		{
			yyVAL.lval = yyS[yypt-3].lval << uint(yyS[yypt-0].lval)
		}
	case 129:
		//line a.y:708
		{
			yyVAL.lval = yyS[yypt-3].lval >> uint(yyS[yypt-0].lval)
		}
	case 130:
		//line a.y:712
		{
			yyVAL.lval = yyS[yypt-2].lval & yyS[yypt-0].lval
		}
	case 131:
		//line a.y:716
		{
			yyVAL.lval = yyS[yypt-2].lval ^ yyS[yypt-0].lval
		}
	case 132:
		//line a.y:720
		{
			yyVAL.lval = yyS[yypt-2].lval | yyS[yypt-0].lval
		}
	}
	goto yystack /* stack new state and value */
}
