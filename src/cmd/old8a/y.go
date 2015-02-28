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

const yyNprod = 131
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 556

var yyAct = []int{

	50, 226, 120, 40, 48, 3, 268, 207, 62, 79,
	77, 169, 49, 267, 266, 72, 60, 262, 84, 254,
	52, 81, 82, 71, 70, 252, 83, 97, 115, 65,
	80, 240, 109, 99, 238, 109, 91, 93, 95, 236,
	220, 218, 101, 103, 209, 208, 170, 239, 104, 206,
	233, 210, 109, 168, 173, 142, 117, 118, 119, 135,
	108, 116, 112, 110, 125, 63, 248, 56, 55, 78,
	72, 230, 229, 225, 224, 109, 136, 84, 223, 133,
	81, 82, 140, 141, 126, 83, 153, 137, 143, 80,
	53, 152, 150, 149, 42, 44, 47, 43, 45, 139,
	61, 46, 85, 148, 54, 147, 146, 145, 76, 63,
	51, 39, 57, 154, 67, 144, 134, 132, 131, 39,
	124, 36, 34, 175, 176, 30, 222, 31, 33, 32,
	109, 117, 221, 58, 242, 72, 241, 183, 235, 111,
	165, 167, 140, 141, 182, 216, 166, 214, 172, 181,
	250, 251, 191, 193, 195, 215, 109, 109, 109, 109,
	109, 255, 194, 109, 109, 109, 189, 190, 165, 167,
	211, 183, 56, 130, 166, 56, 55, 217, 263, 117,
	256, 247, 37, 151, 196, 197, 198, 199, 200, 228,
	111, 203, 204, 205, 260, 53, 41, 35, 53, 56,
	55, 88, 109, 109, 128, 127, 259, 58, 234, 54,
	73, 227, 54, 237, 253, 129, 87, 57, 74, 212,
	57, 257, 53, 246, 243, 105, 106, 113, 244, 202,
	231, 232, 180, 114, 245, 121, 54, 122, 123, 249,
	122, 123, 74, 174, 57, 184, 185, 186, 187, 188,
	258, 7, 201, 22, 261, 42, 44, 47, 43, 45,
	264, 265, 46, 9, 10, 11, 12, 13, 17, 27,
	18, 14, 28, 19, 20, 21, 29, 23, 24, 25,
	26, 56, 55, 138, 163, 162, 160, 161, 155, 156,
	157, 158, 159, 4, 16, 8, 15, 5, 56, 55,
	6, 107, 56, 55, 53, 157, 158, 159, 42, 44,
	47, 43, 45, 2, 1, 46, 85, 102, 54, 100,
	98, 53, 96, 63, 51, 53, 57, 56, 55, 42,
	44, 47, 43, 45, 94, 54, 46, 58, 92, 54,
	63, 74, 90, 57, 63, 51, 86, 57, 56, 55,
	53, 75, 66, 64, 42, 44, 47, 43, 45, 59,
	68, 46, 58, 213, 54, 0, 0, 0, 89, 0,
	51, 53, 57, 56, 55, 42, 44, 47, 43, 45,
	0, 0, 46, 58, 0, 54, 0, 0, 0, 38,
	0, 51, 0, 57, 56, 55, 53, 0, 0, 0,
	42, 44, 47, 43, 45, 0, 0, 46, 58, 0,
	54, 155, 156, 157, 158, 159, 51, 53, 57, 0,
	0, 42, 44, 47, 43, 45, 0, 0, 46, 56,
	55, 54, 0, 0, 0, 0, 0, 51, 0, 57,
	164, 163, 162, 160, 161, 155, 156, 157, 158, 159,
	56, 55, 53, 0, 56, 55, 0, 56, 55, 0,
	0, 0, 0, 0, 73, 0, 54, 0, 0, 0,
	69, 63, 74, 53, 57, 56, 55, 53, 56, 178,
	53, 0, 219, 0, 0, 56, 55, 54, 0, 171,
	0, 54, 58, 74, 54, 57, 192, 74, 53, 57,
	51, 53, 57, 0, 0, 0, 0, 179, 53, 0,
	177, 0, 54, 0, 0, 54, 0, 0, 74, 0,
	57, 74, 54, 57, 0, 0, 0, 0, 74, 0,
	57, 164, 163, 162, 160, 161, 155, 156, 157, 158,
	159, 162, 160, 161, 155, 156, 157, 158, 159, 160,
	161, 155, 156, 157, 158, 159,
}
var yyPact = []int{

	-1000, -1000, 249, -1000, 78, -1000, 81, 80, 73, 71,
	339, 293, 293, 364, 420, -1000, -1000, 58, 318, 293,
	293, 293, -1000, 219, 14, 293, 293, 89, 448, 448,
	-1000, 476, -1000, -1000, 476, -1000, -1000, -1000, 364, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	10, 190, 9, -1000, -1000, 476, 476, 476, 228, -1000,
	70, -1000, -1000, 163, -1000, 68, -1000, 67, -1000, 166,
	-1000, 66, 7, 231, 476, -1000, 272, -1000, 364, -1000,
	-1000, -1000, -1000, -1000, 3, 228, -1000, -1000, -1000, 364,
	-1000, 65, -1000, 57, -1000, 56, -1000, 55, -1000, 53,
	-1000, 43, -1000, 42, 171, 41, 36, 249, 527, -1000,
	527, -1000, 131, 0, -7, 436, 111, -1000, -1000, -1000,
	2, 235, 476, 476, -1000, -1000, -1000, -1000, -1000, 469,
	466, 364, 293, -1000, 166, 137, -1000, -1000, 385, -1000,
	-1000, -1000, 103, 2, 364, 364, 364, 364, 364, 293,
	293, 476, 445, 289, -1000, 476, 476, 476, 476, 476,
	245, 221, 476, 476, 476, -4, -8, -9, -1, 476,
	-1000, -1000, 208, 112, 231, -1000, -1000, -12, 441, -1000,
	-1000, -1000, -1000, -13, 85, 79, -1000, 28, 24, -1000,
	-1000, 23, 179, 22, -1000, 21, 294, 294, -1000, -1000,
	-1000, 476, 476, 542, 535, 279, -2, 476, -1000, -1000,
	101, -14, 476, -19, -1000, -1000, -1000, -5, -1000, -22,
	-1000, 99, 96, 476, 219, 14, -1000, 213, 149, 15,
	14, 402, 402, 113, -28, 203, -1000, -34, -1000, 126,
	-1000, -1000, -1000, -1000, -1000, -1000, 148, 211, 179, -1000,
	195, 183, -1000, 476, -1000, -36, -1000, 146, -1000, 476,
	476, -39, -1000, -1000, -40, -47, -1000, -1000, -1000,
}
var yyPgo = []int{

	0, 0, 28, 363, 2, 196, 8, 3, 20, 9,
	100, 16, 10, 4, 12, 1, 197, 360, 182, 359,
	353, 352, 351, 346, 342, 338, 334, 322, 320, 319,
	317, 314, 313, 5, 301, 300, 296, 294, 253,
}
var yyR1 = []int{

	0, 31, 32, 31, 34, 33, 33, 33, 33, 35,
	35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
	35, 35, 35, 35, 35, 35, 35, 35, 35, 16,
	16, 20, 21, 19, 19, 18, 18, 17, 17, 17,
	36, 37, 37, 38, 38, 22, 22, 22, 23, 23,
	24, 24, 25, 25, 26, 26, 26, 27, 28, 29,
	30, 10, 10, 12, 12, 12, 12, 12, 12, 12,
	11, 11, 9, 9, 7, 7, 7, 7, 7, 7,
	6, 6, 6, 6, 6, 6, 6, 15, 15, 15,
	15, 5, 5, 13, 13, 13, 13, 13, 13, 13,
	13, 13, 13, 13, 14, 14, 8, 8, 4, 4,
	4, 3, 3, 3, 1, 1, 1, 1, 1, 1,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	2,
}
var yyR2 = []int{

	0, 0, 0, 3, 0, 4, 1, 2, 2, 3,
	3, 2, 2, 2, 2, 2, 2, 1, 1, 2,
	2, 2, 2, 2, 1, 2, 2, 2, 2, 0,
	1, 3, 3, 2, 1, 2, 1, 2, 1, 3,
	6, 5, 7, 4, 6, 2, 1, 2, 1, 1,
	3, 5, 3, 5, 2, 1, 3, 5, 5, 3,
	3, 1, 1, 1, 1, 2, 2, 1, 1, 1,
	1, 1, 4, 2, 1, 1, 1, 1, 1, 1,
	2, 2, 2, 2, 4, 5, 3, 1, 2, 3,
	4, 1, 1, 1, 4, 4, 6, 9, 9, 3,
	3, 4, 5, 8, 1, 6, 5, 7, 0, 2,
	2, 1, 1, 1, 1, 1, 2, 2, 2, 3,
	1, 3, 3, 3, 3, 3, 4, 4, 3, 3,
	3,
}
var yyChk = []int{

	-1000, -31, -32, -33, 44, 48, -35, 2, 46, 14,
	15, 16, 17, 18, 22, -36, -37, 19, 21, 24,
	25, 26, -38, 28, 29, 30, 31, 20, 23, 27,
	47, 49, 48, 48, 49, -16, 50, -18, 50, -10,
	-7, -5, 36, 39, 37, 40, 43, 38, -13, -14,
	-1, 52, -8, 32, 46, 10, 9, 54, 44, -19,
	-11, -10, -6, 51, -20, -11, -21, -10, -17, 50,
	-9, -6, -1, 44, 52, -22, 50, -12, 11, -9,
	-14, -7, -13, -6, -1, 44, -23, -16, -18, 50,
	-24, -11, -25, -11, -26, -11, -27, -7, -28, -6,
	-29, -11, -30, -11, -8, -5, -5, -34, -2, -1,
	-2, -10, 52, 37, 43, -2, 52, -1, -1, -1,
	-4, 7, 9, 10, 50, -1, -8, 42, 41, 52,
	10, 50, 50, -9, 50, 52, -4, -12, 11, -8,
	-7, -13, 52, -4, 50, 50, 50, 50, 50, 50,
	50, 12, 50, 50, -33, 9, 10, 11, 12, 13,
	7, 8, 6, 5, 4, 37, 43, 38, 53, 11,
	53, 53, 37, 52, 8, -1, -1, 41, 10, 41,
	-10, -11, -9, 34, -10, -10, -10, -10, -10, -11,
	-11, -1, 51, -1, -6, -1, -2, -2, -2, -2,
	-2, 7, 8, -2, -2, -2, 53, 11, 53, 53,
	52, -1, 11, -3, 35, 43, 33, -4, 53, 41,
	53, 47, 47, 50, 50, 50, -15, 32, 10, 50,
	50, -2, -2, 52, -1, 37, 53, -1, 53, 52,
	53, 37, 38, -1, -7, -6, 10, 32, 51, -6,
	37, 38, 53, 11, 53, 35, 32, 10, -15, 11,
	11, -1, 53, 32, -1, -1, 53, 53, 53,
}
var yyDef = []int{

	1, -2, 0, 3, 0, 6, 0, 0, 0, 29,
	0, 0, 0, 0, 0, 17, 18, 0, 29, 0,
	0, 0, 24, 0, 0, 0, 0, 0, 0, 0,
	4, 0, 7, 8, 0, 11, 30, 12, 0, 36,
	61, 62, 74, 75, 76, 77, 78, 79, 91, 92,
	93, 0, 104, 114, 115, 0, 0, 0, 108, 13,
	34, 70, 71, 0, 14, 0, 15, 0, 16, 0,
	38, 0, 0, 108, 0, 19, 0, 46, 0, 63,
	64, 67, 68, 69, 93, 108, 20, 48, 49, 30,
	21, 0, 22, 0, 23, 55, 25, 0, 26, 0,
	27, 0, 28, 0, 0, 0, 0, 0, 9, 120,
	10, 35, 0, 0, 0, 0, 0, 116, 117, 118,
	0, 0, 0, 0, 33, 80, 81, 82, 83, 0,
	0, 0, 0, 37, 0, 0, 73, 45, 0, 47,
	65, 66, 0, 73, 0, 0, 54, 0, 0, 0,
	0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 99, 0,
	100, 119, 0, 0, 108, 109, 110, 0, 0, 86,
	31, 32, 39, 0, 50, 52, 56, 0, 0, 59,
	60, 0, 0, 0, 43, 0, 121, 122, 123, 124,
	125, 0, 0, 128, 129, 130, 94, 0, 95, 101,
	0, 0, 0, 0, 111, 112, 113, 0, 84, 0,
	72, 0, 0, 0, 0, 0, 41, 87, 0, 0,
	0, 126, 127, 0, 0, 0, 102, 0, 106, 0,
	85, 51, 53, 57, 58, 40, 0, 88, 0, 44,
	0, 0, 96, 0, 105, 0, 89, 0, 42, 0,
	0, 0, 107, 90, 0, 0, 103, 97, 98,
}
var yyTok1 = []int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 51, 13, 6, 3,
	52, 53, 11, 9, 50, 10, 3, 12, 3, 3,
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

type yyParser interface {
	Parse(yyLexer) int
	Lookahead() int
}

type yyParserImpl struct {
	lookahead func() int
}

func (p *yyParserImpl) Lookahead() int {
	return p.lookahead()
}

func yyNewParser() yyParser {
	p := &yyParserImpl{
		lookahead: func() int { return -1 },
	}
	return p
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

func yylex1(lex yyLexer, lval *yySymType) (char, token int) {
	token = 0
	char = lex.Lex(lval)
	if char <= 0 {
		token = yyTok1[0]
		goto out
	}
	if char < len(yyTok1) {
		token = yyTok1[char]
		goto out
	}
	if char >= yyPrivate {
		if char < yyPrivate+len(yyTok2) {
			token = yyTok2[char-yyPrivate]
			goto out
		}
	}
	for i := 0; i < len(yyTok3); i += 2 {
		token = yyTok3[i+0]
		if token == char {
			token = yyTok3[i+1]
			goto out
		}
	}

out:
	if token == 0 {
		token = yyTok2[1] /* unknown char */
	}
	if yyDebug >= 3 {
		__yyfmt__.Printf("lex %s(%d)\n", yyTokname(token), uint(char))
	}
	return char, token
}

func yyParse(yylex yyLexer) int {
	return yyNewParser().Parse(yylex)
}

func (yyrcvr *yyParserImpl) Parse(yylex yyLexer) int {
	var yyn int
	var yylval yySymType
	var yyVAL yySymType
	var yyDollar []yySymType
	yyS := make([]yySymType, yyMaxDepth)

	Nerrs := 0   /* number of errors */
	Errflag := 0 /* error recovery flag */
	yystate := 0
	yychar := -1
	yytoken := -1 // yychar translated into internal numbering
	yyrcvr.lookahead = func() int { return yychar }
	defer func() {
		// Make sure we report no lookahead when not parsing.
		yychar = -1
		yytoken = -1
	}()
	yyp := -1
	goto yystack

ret0:
	return 0

ret1:
	return 1

yystack:
	/* put a state and value onto the stack */
	if yyDebug >= 4 {
		__yyfmt__.Printf("char %v in %v\n", yyTokname(yytoken), yyStatname(yystate))
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
		yychar, yytoken = yylex1(yylex, &yylval)
	}
	yyn += yytoken
	if yyn < 0 || yyn >= yyLast {
		goto yydefault
	}
	yyn = yyAct[yyn]
	if yyChk[yyn] == yytoken { /* valid shift */
		yychar = -1
		yytoken = -1
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
			yychar, yytoken = yylex1(yylex, &yylval)
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
			if yyn < 0 || yyn == yytoken {
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
				__yyfmt__.Printf(" saw %s\n", yyTokname(yytoken))
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
				__yyfmt__.Printf("error recovery discards %s\n", yyTokname(yytoken))
			}
			if yytoken == yyEofCode {
				goto ret1
			}
			yychar = -1
			yytoken = -1
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
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:74
		{
			stmtline = asm.Lineno
		}
	case 4:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:81
		{
			yyDollar[1].sym = asm.LabelLookup(yyDollar[1].sym)
			if yyDollar[1].sym.Type == LLAB && yyDollar[1].sym.Value != int64(asm.PC) {
				yyerror("redeclaration of %s", yyDollar[1].sym.Labelname)
			}
			yyDollar[1].sym.Type = LLAB
			yyDollar[1].sym.Value = int64(asm.PC)
		}
	case 9:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:96
		{
			yyDollar[1].sym.Type = LVAR
			yyDollar[1].sym.Value = yyDollar[3].lval
		}
	case 10:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:101
		{
			if yyDollar[1].sym.Value != int64(yyDollar[3].lval) {
				yyerror("redeclaration of %s", yyDollar[1].sym.Name)
			}
			yyDollar[1].sym.Value = yyDollar[3].lval
		}
	case 11:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:107
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 12:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:108
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 13:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:109
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 14:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:110
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 15:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:111
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 16:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:112
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 19:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:115
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 20:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:116
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 21:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:117
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 22:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:118
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 23:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:119
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 25:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:121
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 26:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:122
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 27:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:123
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 28:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:124
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 29:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:127
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 30:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:132
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 31:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:139
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 32:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:146
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 33:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:153
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 34:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:158
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 35:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:165
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[2].addr
		}
	case 36:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:170
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[1].addr
		}
	case 37:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:177
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[2].addr
		}
	case 38:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:182
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[1].addr
		}
	case 39:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:187
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 40:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:194
		{
			outcode(obj.ADATA, &Addr2{yyDollar[2].addr, yyDollar[6].addr})
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 41:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:204
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.ATEXT, &Addr2{yyDollar[2].addr, yyDollar[5].addr})
		}
	case 42:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:209
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.ATEXT, &Addr2{yyDollar[2].addr, yyDollar[7].addr})
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 43:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:220
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.AGLOBL, &Addr2{yyDollar[2].addr, yyDollar[4].addr})
		}
	case 44:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:225
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.AGLOBL, &Addr2{yyDollar[2].addr, yyDollar[6].addr})
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 45:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:237
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[2].addr
		}
	case 46:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:242
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[1].addr
		}
	case 47:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:247
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[2].addr
			yyVAL.addr2.to.Type = obj.TYPE_INDIR
		}
	case 48:
		yyVAL.addr2 = yyS[yypt-0].addr2
	case 49:
		yyVAL.addr2 = yyS[yypt-0].addr2
	case 50:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:259
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 51:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:264
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
			if yyVAL.addr2.from.Index != obj.TYPE_NONE {
				yyerror("dp shift with lhs index")
			}
			yyVAL.addr2.from.Index = int16(yyDollar[5].lval)
		}
	case 52:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:275
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 53:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:280
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
			if yyVAL.addr2.to.Index != obj.TYPE_NONE {
				yyerror("dp move with lhs index")
			}
			yyVAL.addr2.to.Index = int16(yyDollar[5].lval)
		}
	case 54:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:291
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 55:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:296
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 56:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:301
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 57:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:308
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
			yyVAL.addr2.to.Offset = yyDollar[5].lval
		}
	case 58:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:316
		{
			yyVAL.addr2.from = yyDollar[3].addr
			yyVAL.addr2.to = yyDollar[5].addr
			if yyDollar[1].addr.Type != obj.TYPE_CONST {
				yyerror("illegal constant")
			}
			yyVAL.addr2.to.Offset = yyDollar[1].addr.Offset
		}
	case 59:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:327
		{
			if yyDollar[1].addr.Type != obj.TYPE_CONST || yyDollar[3].addr.Type != obj.TYPE_CONST {
				yyerror("arguments to PCDATA must be integer constants")
			}
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 60:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:337
		{
			if yyDollar[1].addr.Type != obj.TYPE_CONST {
				yyerror("index for FUNCDATA must be integer constant")
			}
			if yyDollar[3].addr.Type != obj.TYPE_MEM || (yyDollar[3].addr.Name != obj.NAME_EXTERN && yyDollar[3].addr.Name != obj.NAME_STATIC) {
				yyerror("value for FUNCDATA must be symbol reference")
			}
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
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
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:356
		{
			yyVAL.addr = yyDollar[2].addr
		}
	case 66:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:360
		{
			yyVAL.addr = yyDollar[2].addr
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
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:373
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyDollar[1].lval + int64(asm.PC)
		}
	case 73:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:379
		{
			yyDollar[1].sym = asm.LabelLookup(yyDollar[1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyDollar[1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyDollar[1].sym.Labelname)
			}
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyDollar[1].sym.Value + yyDollar[2].lval
		}
	case 74:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:391
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 75:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:397
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 76:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:403
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 77:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:409
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 78:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:415
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = REG_SP
		}
	case 79:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:421
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 80:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:429
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
			yyVAL.addr.Offset = yyDollar[2].lval
		}
	case 81:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:435
		{
			yyVAL.addr = yyDollar[2].addr
			yyVAL.addr.Type = obj.TYPE_ADDR
			/*
				if($2.Type == D_AUTO || $2.Type == D_PARAM)
					yyerror("constant cannot be automatic: %s",
						$2.Sym.name);
			*/
		}
	case 82:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:444
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SCONST
			yyVAL.addr.U.Sval = yyDollar[2].sval
		}
	case 83:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:450
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = yyDollar[2].dval
		}
	case 84:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:456
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = yyDollar[3].dval
		}
	case 85:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:462
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = -yyDollar[4].dval
		}
	case 86:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:468
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = -yyDollar[3].dval
		}
	case 87:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:476
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.U.Argsize = obj.ArgsSizeUnknown
		}
	case 88:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:483
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -yyDollar[2].lval
			yyVAL.addr.U.Argsize = obj.ArgsSizeUnknown
		}
	case 89:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:490
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.U.Argsize = int32(yyDollar[3].lval)
		}
	case 90:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:497
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -yyDollar[2].lval
			yyVAL.addr.U.Argsize = int32(yyDollar[4].lval)
		}
	case 91:
		yyVAL.addr = yyS[yypt-0].addr
	case 92:
		yyVAL.addr = yyS[yypt-0].addr
	case 93:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:511
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 94:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:517
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 95:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:524
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = REG_SP
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 96:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:531
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.Index = int16(yyDollar[3].lval)
			yyVAL.addr.Scale = int16(yyDollar[5].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 97:
		yyDollar = yyS[yypt-9 : yypt+1]
		//line a.y:540
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.Index = int16(yyDollar[6].lval)
			yyVAL.addr.Scale = int16(yyDollar[8].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 98:
		yyDollar = yyS[yypt-9 : yypt+1]
		//line a.y:550
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.Index = int16(yyDollar[6].lval)
			yyVAL.addr.Scale = int16(yyDollar[8].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 99:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:560
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[2].lval)
		}
	case 100:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:566
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = REG_SP
		}
	case 101:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:572
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 102:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:579
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Index = int16(yyDollar[2].lval)
			yyVAL.addr.Scale = int16(yyDollar[4].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 103:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line a.y:587
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[2].lval)
			yyVAL.addr.Index = int16(yyDollar[5].lval)
			yyVAL.addr.Scale = int16(yyDollar[7].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 104:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:598
		{
			yyVAL.addr = yyDollar[1].addr
		}
	case 105:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:602
		{
			yyVAL.addr = yyDollar[1].addr
			yyVAL.addr.Index = int16(yyDollar[3].lval)
			yyVAL.addr.Scale = int16(yyDollar[5].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 106:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:611
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyDollar[4].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 0)
			yyVAL.addr.Offset = yyDollar[2].lval
		}
	case 107:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:619
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = obj.NAME_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 1)
			yyVAL.addr.Offset = yyDollar[4].lval
		}
	case 108:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:628
		{
			yyVAL.lval = 0
		}
	case 109:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:632
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 110:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:636
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 111:
		yyVAL.lval = yyS[yypt-0].lval
	case 112:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:643
		{
			yyVAL.lval = obj.NAME_AUTO
		}
	case 113:
		yyVAL.lval = yyS[yypt-0].lval
	case 114:
		yyVAL.lval = yyS[yypt-0].lval
	case 115:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:651
		{
			yyVAL.lval = yyDollar[1].sym.Value
		}
	case 116:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:655
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 117:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:659
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 118:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:663
		{
			yyVAL.lval = ^yyDollar[2].lval
		}
	case 119:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:667
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 120:
		yyVAL.lval = yyS[yypt-0].lval
	case 121:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:674
		{
			yyVAL.lval = yyDollar[1].lval + yyDollar[3].lval
		}
	case 122:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:678
		{
			yyVAL.lval = yyDollar[1].lval - yyDollar[3].lval
		}
	case 123:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:682
		{
			yyVAL.lval = yyDollar[1].lval * yyDollar[3].lval
		}
	case 124:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:686
		{
			yyVAL.lval = yyDollar[1].lval / yyDollar[3].lval
		}
	case 125:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:690
		{
			yyVAL.lval = yyDollar[1].lval % yyDollar[3].lval
		}
	case 126:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:694
		{
			yyVAL.lval = yyDollar[1].lval << uint(yyDollar[4].lval)
		}
	case 127:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:698
		{
			yyVAL.lval = yyDollar[1].lval >> uint(yyDollar[4].lval)
		}
	case 128:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:702
		{
			yyVAL.lval = yyDollar[1].lval & yyDollar[3].lval
		}
	case 129:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:706
		{
			yyVAL.lval = yyDollar[1].lval ^ yyDollar[3].lval
		}
	case 130:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:710
		{
			yyVAL.lval = yyDollar[1].lval | yyDollar[3].lval
		}
	}
	goto yystack /* stack new state and value */
}
