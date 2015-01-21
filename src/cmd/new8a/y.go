//line a.y:32
package main

import __yyfmt__ "fmt"

//line a.y:32
import (
	"cmd/internal/asm"
	"cmd/internal/obj"
	. "cmd/internal/obj/i386"
)

//line a.y:41
type yySymType struct {
	yys  int
	sym  *asm.Sym
	lval int64
	con2 struct {
		v1 int32
		v2 int32
	}
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
const LTYPES = 57356
const LTYPEM = 57357
const LTYPEI = 57358
const LTYPEG = 57359
const LTYPEXC = 57360
const LTYPEX = 57361
const LTYPEPC = 57362
const LTYPEF = 57363
const LCONST = 57364
const LFP = 57365
const LPC = 57366
const LSB = 57367
const LBREG = 57368
const LLREG = 57369
const LSREG = 57370
const LFREG = 57371
const LXREG = 57372
const LFCONST = 57373
const LSCONST = 57374
const LSP = 57375
const LNAME = 57376
const LLAB = 57377
const LVAR = 57378

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
	"LTYPES",
	"LTYPEM",
	"LTYPEI",
	"LTYPEG",
	"LTYPEXC",
	"LTYPEX",
	"LTYPEPC",
	"LTYPEF",
	"LCONST",
	"LFP",
	"LPC",
	"LSB",
	"LBREG",
	"LLREG",
	"LSREG",
	"LFREG",
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

const yyNprod = 132
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 586

var yyAct = []int{

	47, 37, 59, 185, 45, 120, 80, 3, 49, 78,
	60, 187, 268, 57, 267, 69, 208, 68, 85, 82,
	84, 67, 83, 169, 73, 100, 62, 102, 46, 109,
	266, 115, 109, 92, 94, 96, 262, 255, 253, 104,
	106, 241, 239, 237, 221, 219, 81, 210, 209, 109,
	170, 240, 234, 117, 118, 119, 231, 207, 211, 173,
	108, 125, 144, 110, 168, 135, 116, 69, 112, 126,
	230, 229, 109, 133, 53, 52, 136, 223, 85, 82,
	84, 142, 83, 222, 143, 153, 152, 139, 141, 151,
	58, 150, 145, 149, 148, 53, 52, 50, 147, 146,
	138, 36, 113, 134, 64, 132, 81, 131, 114, 36,
	124, 51, 33, 31, 30, 154, 71, 29, 50, 54,
	27, 228, 28, 175, 176, 227, 111, 220, 165, 167,
	109, 117, 51, 55, 166, 69, 247, 71, 184, 186,
	54, 182, 142, 246, 183, 143, 181, 165, 167, 236,
	192, 172, 191, 166, 251, 252, 109, 109, 109, 109,
	109, 256, 183, 109, 109, 109, 195, 196, 263, 257,
	212, 226, 217, 245, 215, 53, 130, 137, 34, 117,
	218, 111, 216, 38, 260, 259, 32, 197, 198, 199,
	200, 201, 254, 225, 204, 205, 206, 89, 50, 121,
	75, 122, 123, 109, 109, 88, 98, 128, 127, 235,
	55, 213, 51, 258, 238, 122, 123, 129, 203, 244,
	54, 174, 180, 202, 6, 242, 107, 243, 157, 158,
	159, 249, 248, 250, 232, 233, 2, 188, 189, 190,
	1, 193, 194, 105, 39, 41, 44, 40, 42, 103,
	7, 43, 101, 99, 97, 261, 95, 93, 91, 87,
	264, 265, 9, 10, 11, 12, 13, 17, 15, 18,
	14, 16, 19, 20, 21, 22, 23, 24, 25, 26,
	53, 52, 79, 163, 162, 160, 161, 155, 156, 157,
	158, 159, 4, 76, 8, 74, 5, 72, 63, 61,
	53, 52, 140, 50, 56, 65, 224, 39, 41, 44,
	40, 42, 214, 0, 43, 86, 0, 51, 0, 0,
	0, 77, 48, 50, 60, 54, 0, 39, 41, 44,
	40, 42, 53, 52, 43, 86, 0, 51, 0, 0,
	0, 0, 48, 0, 60, 54, 155, 156, 157, 158,
	159, 0, 53, 52, 0, 50, 0, 0, 0, 39,
	41, 44, 40, 42, 0, 0, 43, 55, 0, 51,
	0, 0, 53, 52, 48, 50, 60, 54, 0, 39,
	41, 44, 40, 42, 0, 0, 43, 55, 0, 51,
	0, 0, 0, 90, 48, 50, 0, 54, 0, 39,
	41, 44, 40, 42, 53, 52, 43, 55, 0, 51,
	0, 0, 0, 35, 48, 0, 0, 54, 0, 0,
	53, 52, 0, 0, 53, 52, 0, 50, 0, 0,
	0, 39, 41, 44, 40, 42, 0, 0, 43, 55,
	0, 51, 0, 50, 53, 52, 48, 50, 0, 54,
	0, 39, 41, 44, 40, 42, 0, 51, 43, 53,
	52, 51, 71, 0, 60, 54, 48, 50, 0, 54,
	164, 163, 162, 160, 161, 155, 156, 157, 158, 159,
	0, 51, 50, 0, 53, 52, 71, 0, 187, 54,
	53, 52, 0, 0, 70, 0, 51, 0, 0, 0,
	66, 71, 0, 60, 54, 53, 178, 50, 0, 0,
	0, 53, 52, 50, 0, 53, 52, 0, 171, 70,
	0, 51, 179, 0, 0, 0, 71, 51, 50, 54,
	0, 0, 71, 0, 50, 54, 0, 177, 50, 0,
	0, 0, 51, 0, 0, 0, 55, 71, 51, 0,
	54, 0, 51, 48, 0, 0, 54, 71, 0, 0,
	54, 164, 163, 162, 160, 161, 155, 156, 157, 158,
	159, 162, 160, 161, 155, 156, 157, 158, 159, 160,
	161, 155, 156, 157, 158, 159,
}
var yyPact = []int{

	-1000, -1000, 248, -1000, 73, -1000, 69, 66, 64, 62,
	363, 323, 323, 395, 450, 89, 502, 271, 343, 323,
	323, 323, 502, 208, -43, 323, 323, -1000, 506, -1000,
	-1000, 506, -1000, -1000, -1000, 395, -1000, -1000, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, 17, 65, 15,
	-1000, -1000, 506, 506, 506, 192, -1000, 60, -1000, -1000,
	166, -1000, 57, -1000, 55, -1000, 475, -1000, 53, 14,
	206, 506, -1000, 165, -1000, 50, -1000, 291, -1000, 395,
	-1000, -1000, -1000, -1000, -1000, 11, 192, -1000, -1000, -1000,
	395, -1000, 49, -1000, 48, -1000, 44, -1000, 43, -1000,
	41, -1000, 39, -1000, 36, -1000, 35, 248, 557, -1000,
	557, -1000, 91, 12, -2, 466, 114, -1000, -1000, -1000,
	8, 213, 506, 506, -1000, -1000, -1000, -1000, -1000, 496,
	481, 395, 323, -1000, 475, 128, -1000, 506, 435, -1000,
	415, -1000, -1000, -1000, 110, 8, 395, 395, 395, 411,
	395, 395, 323, 323, -1000, 506, 506, 506, 506, 506,
	216, 210, 506, 506, 506, 5, -4, -5, 7, 506,
	-1000, -1000, 200, 139, 206, -1000, -1000, -7, 86, -1000,
	-1000, -1000, -1000, -8, 33, -1000, 27, 161, 78, 74,
	-1000, -1000, 21, 20, 6, -1000, -1000, 217, 217, -1000,
	-1000, -1000, 506, 506, 572, 565, 278, 1, 506, -1000,
	-1000, 112, -9, 506, -10, -1000, -1000, -1000, 0, -1000,
	-11, -1000, -43, -42, -1000, 209, 141, 106, 98, -43,
	506, 208, 337, 337, 117, -14, 181, -1000, -15, -1000,
	126, -1000, -1000, -1000, 137, 203, -1000, -1000, -1000, -1000,
	-1000, 174, 173, -1000, 506, -1000, -16, -1000, 136, 506,
	506, -22, -1000, -1000, -38, -40, -1000, -1000, -1000,
}
var yyPgo = []int{

	0, 0, 31, 312, 5, 306, 183, 2, 3, 1,
	8, 6, 90, 13, 9, 4, 28, 186, 305, 178,
	304, 299, 298, 297, 295, 293, 259, 258, 257, 256,
	254, 253, 252, 249, 243, 240, 236, 7, 226, 224,
}
var yyR1 = []int{

	0, 35, 36, 35, 38, 37, 37, 37, 37, 39,
	39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
	39, 39, 39, 39, 39, 39, 39, 39, 39, 17,
	17, 21, 22, 20, 20, 19, 19, 18, 18, 18,
	23, 24, 24, 25, 25, 25, 26, 26, 27, 27,
	28, 28, 29, 29, 29, 30, 30, 31, 32, 33,
	34, 12, 12, 14, 14, 14, 14, 14, 14, 14,
	13, 13, 11, 11, 9, 9, 9, 9, 9, 9,
	7, 7, 7, 7, 7, 7, 7, 8, 5, 5,
	5, 5, 6, 6, 15, 15, 15, 15, 15, 15,
	15, 15, 15, 15, 15, 16, 16, 10, 10, 4,
	4, 4, 3, 3, 3, 1, 1, 1, 1, 1,
	1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	2, 2,
}
var yyR2 = []int{

	0, 0, 0, 3, 0, 4, 1, 2, 2, 3,
	3, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
	1, 3, 3, 2, 1, 2, 1, 2, 1, 3,
	5, 3, 5, 2, 1, 2, 1, 1, 3, 5,
	3, 5, 2, 1, 3, 3, 5, 5, 5, 3,
	3, 1, 1, 1, 1, 2, 2, 1, 1, 1,
	1, 1, 4, 2, 1, 1, 1, 1, 1, 1,
	2, 2, 2, 2, 4, 5, 3, 2, 1, 2,
	3, 4, 1, 1, 1, 4, 4, 6, 9, 9,
	3, 3, 4, 5, 8, 1, 6, 5, 7, 0,
	2, 2, 1, 1, 1, 1, 1, 2, 2, 2,
	3, 1, 3, 3, 3, 3, 3, 4, 4, 3,
	3, 3,
}
var yyChk = []int{

	-1000, -35, -36, -37, 44, 48, -39, 2, 46, 14,
	15, 16, 17, 18, 22, 20, 23, 19, 21, 24,
	25, 26, 27, 28, 29, 30, 31, 47, 49, 48,
	48, 49, -17, 50, -19, 50, -12, -9, -6, 36,
	39, 37, 40, 43, 38, -15, -16, -1, 51, -10,
	32, 46, 10, 9, 54, 44, -20, -13, -12, -7,
	53, -21, -13, -22, -12, -18, 50, -11, -7, -1,
	44, 51, -23, -10, -24, -6, -25, 50, -14, 11,
	-11, -16, -9, -15, -7, -1, 44, -26, -17, -19,
	50, -27, -13, -28, -13, -29, -13, -30, -6, -31,
	-9, -32, -7, -33, -13, -34, -13, -38, -2, -1,
	-2, -12, 51, 37, 43, -2, 51, -1, -1, -1,
	-4, 7, 9, 10, 50, -1, -10, 42, 41, 51,
	10, 50, 50, -11, 50, 51, -4, 12, 50, -14,
	11, -10, -9, -15, 51, -4, 50, 50, 50, 50,
	50, 50, 50, 50, -37, 9, 10, 11, 12, 13,
	7, 8, 6, 5, 4, 37, 43, 38, 52, 11,
	52, 52, 37, 51, 8, -1, -1, 41, 10, 41,
	-12, -13, -11, 34, -1, -8, -1, 53, -12, -12,
	-12, -7, -1, -12, -12, -13, -13, -2, -2, -2,
	-2, -2, 7, 8, -2, -2, -2, 52, 11, 52,
	52, 51, -1, 11, -3, 35, 43, 33, -4, 52,
	41, 52, 50, 50, -5, 32, 10, 47, 47, 50,
	50, 50, -2, -2, 51, -1, 37, 52, -1, 52,
	51, 52, -7, -8, 10, 32, 37, 38, -7, -1,
	-9, 37, 38, 52, 11, 52, 35, 32, 10, 11,
	11, -1, 52, 32, -1, -1, 52, 52, 52,
}
var yyDef = []int{

	1, -2, 0, 3, 0, 6, 0, 0, 0, 29,
	0, 0, 0, 0, 0, 0, 0, 0, 29, 0,
	0, 0, 0, 0, 0, 0, 0, 4, 0, 7,
	8, 0, 11, 30, 12, 0, 36, 61, 62, 74,
	75, 76, 77, 78, 79, 92, 93, 94, 0, 105,
	115, 116, 0, 0, 0, 109, 13, 34, 70, 71,
	0, 14, 0, 15, 0, 16, 0, 38, 0, 0,
	109, 0, 17, 0, 18, 0, 19, 0, 44, 0,
	63, 64, 67, 68, 69, 94, 109, 20, 46, 47,
	30, 21, 0, 22, 0, 23, 53, 24, 0, 25,
	0, 26, 0, 27, 0, 28, 0, 0, 9, 121,
	10, 35, 0, 0, 0, 0, 0, 117, 118, 119,
	0, 0, 0, 0, 33, 80, 81, 82, 83, 0,
	0, 0, 0, 37, 0, 0, 73, 0, 0, 43,
	0, 45, 65, 66, 0, 73, 0, 0, 52, 0,
	0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 100, 0,
	101, 120, 0, 0, 109, 110, 111, 0, 0, 86,
	31, 32, 39, 0, 0, 41, 0, 0, 48, 50,
	54, 55, 0, 0, 0, 59, 60, 122, 123, 124,
	125, 126, 0, 0, 129, 130, 131, 95, 0, 96,
	102, 0, 0, 0, 0, 112, 113, 114, 0, 84,
	0, 72, 0, 0, 87, 88, 0, 0, 0, 0,
	0, 0, 127, 128, 0, 0, 0, 103, 0, 107,
	0, 85, 40, 42, 0, 89, 49, 51, 56, 57,
	58, 0, 0, 97, 0, 106, 0, 90, 0, 0,
	0, 0, 108, 91, 0, 0, 104, 98, 99,
}
var yyTok1 = []int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 53, 13, 6, 3,
	51, 52, 11, 9, 50, 10, 3, 12, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 47, 48,
	7, 49, 8, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 5, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 4, 3, 54,
}
var yyTok2 = []int{

	2, 3, 14, 15, 16, 17, 18, 19, 20, 21,
	22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
	42, 43, 44, 45, 46,
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
		//line a.y:75
		{
			stmtline = asm.Lineno
		}
	case 4:
		//line a.y:82
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			if yyS[yypt-1].sym.Type == LLAB && yyS[yypt-1].sym.Value != int64(asm.PC) {
				yyerror("redeclaration of %s", yyS[yypt-1].sym.Labelname)
			}
			yyS[yypt-1].sym.Type = LLAB
			yyS[yypt-1].sym.Value = int64(asm.PC)
		}
	case 9:
		//line a.y:97
		{
			yyS[yypt-2].sym.Type = LVAR
			yyS[yypt-2].sym.Value = yyS[yypt-0].lval
		}
	case 10:
		//line a.y:102
		{
			if yyS[yypt-2].sym.Value != int64(yyS[yypt-0].lval) {
				yyerror("redeclaration of %s", yyS[yypt-2].sym.Name)
			}
			yyS[yypt-2].sym.Value = yyS[yypt-0].lval
		}
	case 11:
		//line a.y:108
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 12:
		//line a.y:109
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 13:
		//line a.y:110
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 14:
		//line a.y:111
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 15:
		//line a.y:112
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 16:
		//line a.y:113
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 17:
		//line a.y:114
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 18:
		//line a.y:115
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 19:
		//line a.y:116
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 20:
		//line a.y:117
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 21:
		//line a.y:118
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 22:
		//line a.y:119
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 23:
		//line a.y:120
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 24:
		//line a.y:121
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 25:
		//line a.y:122
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 26:
		//line a.y:123
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 27:
		//line a.y:124
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 28:
		//line a.y:125
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr2)
		}
	case 29:
		//line a.y:128
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 30:
		//line a.y:133
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 31:
		//line a.y:140
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 32:
		//line a.y:147
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 33:
		//line a.y:154
		{
			yyVAL.addr2.from = yyS[yypt-1].addr
			yyVAL.addr2.to = nullgen
		}
	case 34:
		//line a.y:159
		{
			yyVAL.addr2.from = yyS[yypt-0].addr
			yyVAL.addr2.to = nullgen
		}
	case 35:
		//line a.y:166
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 36:
		//line a.y:171
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 37:
		//line a.y:178
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 38:
		//line a.y:183
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 39:
		//line a.y:188
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 40:
		//line a.y:195
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.from.Scale = int8(yyS[yypt-2].lval)
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 41:
		//line a.y:203
		{
			asm.Settext(yyS[yypt-2].addr.Sym)
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 42:
		//line a.y:209
		{
			asm.Settext(yyS[yypt-4].addr.Sym)
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.from.Scale = int8(yyS[yypt-2].lval)
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 43:
		//line a.y:218
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 44:
		//line a.y:223
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 45:
		//line a.y:228
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyS[yypt-0].addr
			yyVAL.addr2.to.Index = uint8(yyS[yypt-0].addr.Type)
			yyVAL.addr2.to.Type = D_INDIR + D_ADDR
		}
	case 46:
		yyVAL.addr2 = yyS[yypt-0].addr2
	case 47:
		yyVAL.addr2 = yyS[yypt-0].addr2
	case 48:
		//line a.y:241
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 49:
		//line a.y:246
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.to = yyS[yypt-2].addr
			if yyVAL.addr2.from.Index != D_NONE {
				yyerror("dp shift with lhs index")
			}
			yyVAL.addr2.from.Index = uint8(yyS[yypt-0].lval)
		}
	case 50:
		//line a.y:257
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 51:
		//line a.y:262
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.to = yyS[yypt-2].addr
			if yyVAL.addr2.to.Index != D_NONE {
				yyerror("dp move with lhs index")
			}
			yyVAL.addr2.to.Index = uint8(yyS[yypt-0].lval)
		}
	case 52:
		//line a.y:273
		{
			yyVAL.addr2.from = yyS[yypt-1].addr
			yyVAL.addr2.to = nullgen
		}
	case 53:
		//line a.y:278
		{
			yyVAL.addr2.from = yyS[yypt-0].addr
			yyVAL.addr2.to = nullgen
		}
	case 54:
		//line a.y:283
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 55:
		//line a.y:290
		{
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 56:
		//line a.y:295
		{
			yyVAL.addr2.from = yyS[yypt-4].addr
			yyVAL.addr2.from.Scale = int8(yyS[yypt-2].lval)
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
			if yyS[yypt-4].addr.Type != D_CONST {
				yyerror("illegal constant")
			}
			yyVAL.addr2.to.Offset = yyS[yypt-4].addr.Offset
		}
	case 59:
		//line a.y:322
		{
			if yyS[yypt-2].addr.Type != D_CONST || yyS[yypt-0].addr.Type != D_CONST {
				yyerror("arguments to PCDATA must be integer constants")
			}
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 60:
		//line a.y:332
		{
			if yyS[yypt-2].addr.Type != D_CONST {
				yyerror("index for FUNCDATA must be integer constant")
			}
			if yyS[yypt-0].addr.Type != D_EXTERN && yyS[yypt-0].addr.Type != D_STATIC {
				yyerror("value for FUNCDATA must be symbol reference")
			}
			yyVAL.addr2.from = yyS[yypt-2].addr
			yyVAL.addr2.to = yyS[yypt-0].addr
		}
	case 61:
		yyVAL.addr = yyS[yypt-0].addr
	case 62:
		yyVAL.addr = yyS[yypt-0].addr
	case 63:
		yyVAL.addr = yyS[yypt-0].addr
	case 64:
		yyVAL.addr = yyS[yypt-0].addr
	case 65:
		//line a.y:351
		{
			yyVAL.addr = yyS[yypt-0].addr
		}
	case 66:
		//line a.y:355
		{
			yyVAL.addr = yyS[yypt-0].addr
		}
	case 67:
		yyVAL.addr = yyS[yypt-0].addr
	case 68:
		yyVAL.addr = yyS[yypt-0].addr
	case 69:
		yyVAL.addr = yyS[yypt-0].addr
	case 70:
		yyVAL.addr = yyS[yypt-0].addr
	case 71:
		yyVAL.addr = yyS[yypt-0].addr
	case 72:
		//line a.y:368
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_BRANCH
			yyVAL.addr.Offset = yyS[yypt-3].lval + int64(asm.PC)
		}
	case 73:
		//line a.y:374
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyS[yypt-1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyS[yypt-1].sym.Labelname)
			}
			yyVAL.addr.Type = D_BRANCH
			yyVAL.addr.Offset = yyS[yypt-1].sym.Value + yyS[yypt-0].lval
		}
	case 74:
		//line a.y:386
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 75:
		//line a.y:391
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 76:
		//line a.y:396
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 77:
		//line a.y:401
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 78:
		//line a.y:406
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SP
		}
	case 79:
		//line a.y:411
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-0].lval)
		}
	case 80:
		//line a.y:418
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_CONST
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 81:
		//line a.y:424
		{
			yyVAL.addr = yyS[yypt-0].addr
			yyVAL.addr.Index = uint8(yyS[yypt-0].addr.Type)
			yyVAL.addr.Type = D_ADDR
			/*
				if($2.Type == D_AUTO || $2.Type == D_PARAM)
					yyerror("constant cannot be automatic: %s",
						$2.Sym.name);
			*/
		}
	case 82:
		//line a.y:434
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SCONST
			yyVAL.addr.U.Sval = yyS[yypt-0].sval
		}
	case 83:
		//line a.y:440
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FCONST
			yyVAL.addr.U.Dval = yyS[yypt-0].dval
		}
	case 84:
		//line a.y:446
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FCONST
			yyVAL.addr.U.Dval = yyS[yypt-1].dval
		}
	case 85:
		//line a.y:452
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FCONST
			yyVAL.addr.U.Dval = -yyS[yypt-1].dval
		}
	case 86:
		//line a.y:458
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FCONST
			yyVAL.addr.U.Dval = -yyS[yypt-0].dval
		}
	case 87:
		//line a.y:466
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_CONST2
			yyVAL.addr.Offset = int64(yyS[yypt-0].con2.v1)
			yyVAL.addr.Offset2 = int32(yyS[yypt-0].con2.v2)
		}
	case 88:
		//line a.y:475
		{
			yyVAL.con2.v1 = int32(yyS[yypt-0].lval)
			yyVAL.con2.v2 = -obj.ArgsSizeUnknown
		}
	case 89:
		//line a.y:480
		{
			yyVAL.con2.v1 = int32(-yyS[yypt-0].lval)
			yyVAL.con2.v2 = -obj.ArgsSizeUnknown
		}
	case 90:
		//line a.y:485
		{
			yyVAL.con2.v1 = int32(yyS[yypt-2].lval)
			yyVAL.con2.v2 = int32(yyS[yypt-0].lval)
		}
	case 91:
		//line a.y:490
		{
			yyVAL.con2.v1 = int32(-yyS[yypt-2].lval)
			yyVAL.con2.v2 = int32(yyS[yypt-0].lval)
		}
	case 92:
		yyVAL.addr = yyS[yypt-0].addr
	case 93:
		yyVAL.addr = yyS[yypt-0].addr
	case 94:
		//line a.y:501
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_INDIR + D_NONE
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 95:
		//line a.y:507
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(D_INDIR + yyS[yypt-1].lval)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 96:
		//line a.y:513
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_INDIR + D_SP
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 97:
		//line a.y:519
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_INDIR + D_NONE
			yyVAL.addr.Offset = yyS[yypt-5].lval
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 98:
		//line a.y:528
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(D_INDIR + yyS[yypt-6].lval)
			yyVAL.addr.Offset = yyS[yypt-8].lval
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 99:
		//line a.y:537
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(D_INDIR + yyS[yypt-6].lval)
			yyVAL.addr.Offset = yyS[yypt-8].lval
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 100:
		//line a.y:546
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(D_INDIR + yyS[yypt-1].lval)
		}
	case 101:
		//line a.y:551
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_INDIR + D_SP
		}
	case 102:
		//line a.y:556
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(D_INDIR + yyS[yypt-1].lval)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 103:
		//line a.y:562
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_INDIR + D_NONE
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 104:
		//line a.y:570
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(D_INDIR + yyS[yypt-6].lval)
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 105:
		//line a.y:580
		{
			yyVAL.addr = yyS[yypt-0].addr
		}
	case 106:
		//line a.y:584
		{
			yyVAL.addr = yyS[yypt-5].addr
			yyVAL.addr.Index = uint8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 107:
		//line a.y:593
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-1].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-4].sym.Name, 0)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 108:
		//line a.y:600
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-6].sym.Name, 1)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 109:
		//line a.y:608
		{
			yyVAL.lval = 0
		}
	case 110:
		//line a.y:612
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 111:
		//line a.y:616
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 112:
		yyVAL.lval = yyS[yypt-0].lval
	case 113:
		//line a.y:623
		{
			yyVAL.lval = D_AUTO
		}
	case 114:
		yyVAL.lval = yyS[yypt-0].lval
	case 115:
		yyVAL.lval = yyS[yypt-0].lval
	case 116:
		//line a.y:631
		{
			yyVAL.lval = yyS[yypt-0].sym.Value
		}
	case 117:
		//line a.y:635
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 118:
		//line a.y:639
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 119:
		//line a.y:643
		{
			yyVAL.lval = ^yyS[yypt-0].lval
		}
	case 120:
		//line a.y:647
		{
			yyVAL.lval = yyS[yypt-1].lval
		}
	case 121:
		yyVAL.lval = yyS[yypt-0].lval
	case 122:
		//line a.y:654
		{
			yyVAL.lval = yyS[yypt-2].lval + yyS[yypt-0].lval
		}
	case 123:
		//line a.y:658
		{
			yyVAL.lval = yyS[yypt-2].lval - yyS[yypt-0].lval
		}
	case 124:
		//line a.y:662
		{
			yyVAL.lval = yyS[yypt-2].lval * yyS[yypt-0].lval
		}
	case 125:
		//line a.y:666
		{
			yyVAL.lval = yyS[yypt-2].lval / yyS[yypt-0].lval
		}
	case 126:
		//line a.y:670
		{
			yyVAL.lval = yyS[yypt-2].lval % yyS[yypt-0].lval
		}
	case 127:
		//line a.y:674
		{
			yyVAL.lval = yyS[yypt-3].lval << uint(yyS[yypt-0].lval)
		}
	case 128:
		//line a.y:678
		{
			yyVAL.lval = yyS[yypt-3].lval >> uint(yyS[yypt-0].lval)
		}
	case 129:
		//line a.y:682
		{
			yyVAL.lval = yyS[yypt-2].lval & yyS[yypt-0].lval
		}
	case 130:
		//line a.y:686
		{
			yyVAL.lval = yyS[yypt-2].lval ^ yyS[yypt-0].lval
		}
	case 131:
		//line a.y:690
		{
			yyVAL.lval = yyS[yypt-2].lval | yyS[yypt-0].lval
		}
	}
	goto yystack /* stack new state and value */
}
