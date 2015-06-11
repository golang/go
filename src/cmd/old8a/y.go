//line a.y:32
package main

import __yyfmt__ "fmt"

//line a.y:32
import (
	"cmd/internal/asm"
	"cmd/internal/obj"
	. "cmd/internal/obj/x86"
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

var yyToknames = [...]string{
	"$end",
	"error",
	"$unk",
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
	"':'",
	"';'",
	"'='",
	"','",
	"'$'",
	"'('",
	"')'",
	"'~'",
}
var yyStatenames = [...]string{}

const yyEofCode = 1
const yyErrCode = 2
const yyMaxDepth = 200

//line yacctab:1
var yyExca = [...]int{
	-1, 1,
	1, -1,
	-2, 2,
}

const yyNprod = 131
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 594

var yyAct = [...]int{

	50, 226, 40, 3, 79, 77, 120, 49, 62, 207,
	268, 267, 48, 169, 266, 72, 60, 262, 84, 70,
	81, 254, 252, 71, 240, 80, 83, 115, 238, 65,
	82, 236, 109, 99, 220, 109, 91, 93, 95, 52,
	218, 209, 101, 103, 208, 170, 239, 233, 210, 173,
	63, 206, 109, 56, 55, 168, 117, 118, 119, 108,
	142, 135, 110, 116, 125, 112, 248, 104, 230, 229,
	72, 225, 224, 223, 133, 109, 53, 84, 153, 81,
	136, 140, 137, 152, 80, 83, 61, 150, 73, 82,
	54, 141, 143, 149, 69, 63, 74, 39, 57, 148,
	67, 147, 146, 126, 145, 39, 144, 134, 132, 131,
	97, 154, 124, 36, 30, 34, 31, 33, 139, 32,
	222, 221, 58, 175, 176, 111, 216, 242, 214, 241,
	109, 117, 56, 55, 183, 72, 215, 165, 167, 182,
	235, 140, 172, 166, 250, 251, 255, 183, 263, 181,
	187, 141, 191, 193, 195, 53, 109, 109, 109, 109,
	109, 256, 194, 109, 109, 109, 189, 190, 247, 54,
	211, 228, 56, 130, 63, 74, 111, 57, 37, 117,
	35, 217, 41, 196, 197, 198, 199, 200, 165, 167,
	203, 204, 205, 227, 166, 53, 151, 88, 121, 87,
	122, 123, 109, 109, 128, 127, 260, 58, 234, 54,
	259, 105, 106, 237, 253, 129, 212, 57, 180, 42,
	44, 47, 43, 45, 243, 257, 46, 244, 246, 231,
	232, 184, 185, 186, 245, 188, 157, 158, 159, 249,
	164, 163, 162, 160, 161, 155, 156, 157, 158, 159,
	258, 7, 122, 123, 261, 155, 156, 157, 158, 159,
	264, 265, 202, 9, 10, 11, 12, 13, 17, 27,
	18, 14, 28, 19, 20, 21, 29, 23, 24, 25,
	26, 56, 55, 78, 174, 201, 22, 16, 15, 171,
	6, 107, 2, 4, 1, 8, 102, 5, 100, 98,
	96, 94, 92, 90, 53, 56, 55, 138, 42, 44,
	47, 43, 45, 86, 75, 46, 85, 66, 54, 64,
	59, 68, 76, 63, 51, 213, 57, 0, 53, 56,
	55, 0, 42, 44, 47, 43, 45, 0, 0, 46,
	85, 0, 54, 0, 0, 0, 0, 63, 51, 0,
	57, 0, 53, 56, 55, 0, 42, 44, 47, 43,
	45, 0, 0, 46, 58, 0, 54, 0, 0, 0,
	0, 63, 51, 0, 57, 0, 53, 56, 55, 0,
	42, 44, 47, 43, 45, 0, 0, 46, 58, 0,
	54, 0, 0, 0, 89, 0, 51, 0, 57, 0,
	53, 56, 55, 0, 42, 44, 47, 43, 45, 0,
	0, 46, 58, 0, 54, 0, 0, 0, 38, 0,
	51, 0, 57, 0, 53, 56, 55, 0, 42, 44,
	47, 43, 45, 0, 0, 46, 58, 0, 54, 0,
	0, 56, 55, 0, 51, 0, 57, 0, 53, 0,
	56, 55, 42, 44, 47, 43, 45, 56, 55, 46,
	0, 0, 54, 0, 53, 0, 56, 55, 51, 113,
	57, 0, 0, 53, 0, 114, 0, 0, 54, 0,
	53, 0, 219, 0, 74, 0, 57, 54, 0, 53,
	56, 55, 0, 74, 54, 57, 0, 56, 178, 192,
	74, 73, 57, 54, 0, 0, 0, 56, 55, 74,
	0, 57, 0, 53, 56, 55, 0, 0, 0, 0,
	53, 0, 179, 0, 0, 0, 0, 54, 0, 177,
	53, 0, 0, 74, 54, 57, 0, 53, 0, 0,
	74, 0, 57, 0, 54, 0, 0, 0, 0, 58,
	74, 54, 57, 0, 0, 0, 0, 51, 0, 57,
	164, 163, 162, 160, 161, 155, 156, 157, 158, 159,
	163, 162, 160, 161, 155, 156, 157, 158, 159, 162,
	160, 161, 155, 156, 157, 158, 159, 160, 161, 155,
	156, 157, 158, 159,
}
var yyPact = [...]int{

	-1000, -1000, 249, -1000, 67, -1000, 71, 69, 66, 63,
	368, 320, 320, 392, 44, -1000, -1000, 272, 344, 320,
	320, 320, -1000, 392, -1, 320, 320, 78, 505, 505,
	-1000, 498, -1000, -1000, 498, -1000, -1000, -1000, 392, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	13, 432, 11, -1000, -1000, 498, 498, 498, 191, -1000,
	62, -1000, -1000, 163, -1000, 59, -1000, 58, -1000, 457,
	-1000, 57, 9, 243, 498, -1000, 296, -1000, 392, -1000,
	-1000, -1000, -1000, -1000, 8, 191, -1000, -1000, -1000, 392,
	-1000, 56, -1000, 54, -1000, 52, -1000, 51, -1000, 49,
	-1000, 43, -1000, 37, 184, 33, 28, 249, 556, -1000,
	556, -1000, 151, 2, -8, 236, 105, -1000, -1000, -1000,
	-3, 276, 498, 498, -1000, -1000, -1000, -1000, -1000, 488,
	481, 392, 320, -1000, 457, 113, -1000, -1000, 416, -1000,
	-1000, -1000, 100, -3, 392, 392, 392, 183, 392, 320,
	320, 498, 448, 123, -1000, 498, 498, 498, 498, 498,
	278, 254, 498, 498, 498, -2, -9, -12, -4, 498,
	-1000, -1000, 205, 93, 243, -1000, -1000, -13, 441, -1000,
	-1000, -1000, -1000, -19, 74, 73, -1000, 23, 22, -1000,
	-1000, 21, 161, 19, -1000, 18, 225, 225, -1000, -1000,
	-1000, 498, 498, 580, 573, 565, -5, 498, -1000, -1000,
	103, -22, 498, -25, -1000, -1000, -1000, -6, -1000, -29,
	-1000, 92, 89, 498, 183, -1, -1000, 218, 136, 15,
	-1, 246, 246, 107, -31, 203, -1000, -32, -1000, 111,
	-1000, -1000, -1000, -1000, -1000, -1000, 129, 215, 161, -1000,
	199, 195, -1000, 498, -1000, -36, -1000, 116, -1000, 498,
	498, -39, -1000, -1000, -42, -43, -1000, -1000, -1000,
}
var yyPgo = [...]int{

	0, 0, 27, 325, 6, 182, 8, 2, 39, 4,
	86, 16, 5, 12, 7, 1, 180, 321, 178, 320,
	319, 317, 314, 313, 303, 302, 301, 300, 299, 298,
	296, 294, 292, 3, 291, 290, 288, 287, 286,
}
var yyR1 = [...]int{

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
var yyR2 = [...]int{

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
var yyChk = [...]int{

	-1000, -31, -32, -33, 44, 48, -35, 2, 46, 14,
	15, 16, 17, 18, 22, -36, -37, 19, 21, 24,
	25, 26, -38, 28, 29, 30, 31, 20, 23, 27,
	47, 49, 48, 48, 49, -16, 50, -18, 50, -10,
	-7, -5, 36, 39, 37, 40, 43, 38, -13, -14,
	-1, 52, -8, 32, 46, 10, 9, 54, 44, -19,
	-11, -10, -6, 51, -20, -11, -21, -10, -17, 50,
	-9, -6, -1, 44, 52, -22, 50, -12, 11, -9,
	-14, -7, -13, -6, -1, 44, -23, -16, -18, 50,
	-24, -11, -25, -11, -26, -11, -27, -10, -28, -6,
	-29, -11, -30, -11, -8, -5, -5, -34, -2, -1,
	-2, -10, 52, 37, 43, -2, 52, -1, -1, -1,
	-4, 7, 9, 10, 50, -1, -8, 42, 41, 52,
	10, 50, 50, -9, 50, 52, -4, -12, 11, -8,
	-7, -13, 52, -4, 50, 50, 50, 50, 50, 50,
	50, 12, 50, 50, -33, 9, 10, 11, 12, 13,
	7, 8, 6, 5, 4, 37, 43, 38, 53, 11,
	53, 53, 37, 52, 8, -1, -1, 41, 10, 41,
	-10, -11, -9, 34, -10, -10, -10, -7, -10, -11,
	-11, -1, 51, -1, -6, -1, -2, -2, -2, -2,
	-2, 7, 8, -2, -2, -2, 53, 11, 53, 53,
	52, -1, 11, -3, 35, 43, 33, -4, 53, 41,
	53, 47, 47, 50, 50, 50, -15, 32, 10, 50,
	50, -2, -2, 52, -1, 37, 53, -1, 53, 52,
	53, 37, 38, -1, -7, -6, 10, 32, 51, -6,
	37, 38, 53, 11, 53, 35, 32, 10, -15, 11,
	11, -1, 53, 32, -1, -1, 53, 53, 53,
}
var yyDef = [...]int{

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
var yyTok1 = [...]int{

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
var yyTok2 = [...]int{

	2, 3, 14, 15, 16, 17, 18, 19, 20, 21,
	22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
	42, 43, 44, 45, 46,
}
var yyTok3 = [...]int{
	0,
}

var yyErrorMessages = [...]struct {
	state int
	token int
	msg   string
}{}

//line yaccpar:1

/*	parser for yacc output	*/

var (
	yyDebug        = 0
	yyErrorVerbose = false
)

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
	if c >= 1 && c-1 < len(yyToknames) {
		if yyToknames[c-1] != "" {
			return yyToknames[c-1]
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

func yyErrorMessage(state, lookAhead int) string {
	const TOKSTART = 4

	if !yyErrorVerbose {
		return "syntax error"
	}

	for _, e := range yyErrorMessages {
		if e.state == state && e.token == lookAhead {
			return "syntax error: " + e.msg
		}
	}

	res := "syntax error: unexpected " + yyTokname(lookAhead)

	// To match Bison, suggest at most four expected tokens.
	expected := make([]int, 0, 4)

	// Look for shiftable tokens.
	base := yyPact[state]
	for tok := TOKSTART; tok-1 < len(yyToknames); tok++ {
		if n := base + tok; n >= 0 && n < yyLast && yyChk[yyAct[n]] == tok {
			if len(expected) == cap(expected) {
				return res
			}
			expected = append(expected, tok)
		}
	}

	if yyDef[state] == -2 {
		i := 0
		for yyExca[i] != -1 || yyExca[i+1] != state {
			i += 2
		}

		// Look for tokens that we accept or reduce.
		for i += 2; yyExca[i] >= 0; i += 2 {
			tok := yyExca[i]
			if tok < TOKSTART || yyExca[i+1] == 0 {
				continue
			}
			if len(expected) == cap(expected) {
				return res
			}
			expected = append(expected, tok)
		}

		// If the default action is to accept or reduce, give up.
		if yyExca[i+1] != 0 {
			return res
		}
	}

	for i, tok := range expected {
		if i == 0 {
			res += ", expecting "
		} else {
			res += " or "
		}
		res += yyTokname(tok)
	}
	return res
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
		yystate = -1
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
			yylex.Error(yyErrorMessage(yystate, yytoken))
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
				lastpc.From3 = new(obj.Addr)
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 41:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:205
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.ATEXT, &Addr2{yyDollar[2].addr, yyDollar[5].addr})
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
			}
		}
	case 42:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:213
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.ATEXT, &Addr2{yyDollar[2].addr, yyDollar[7].addr})
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 43:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:225
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.AGLOBL, &Addr2{yyDollar[2].addr, yyDollar[4].addr})
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
			}
		}
	case 44:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:233
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.AGLOBL, &Addr2{yyDollar[2].addr, yyDollar[6].addr})
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 45:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:246
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[2].addr
		}
	case 46:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:251
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[1].addr
		}
	case 47:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:256
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[2].addr
			yyVAL.addr2.to.Type = obj.TYPE_INDIR
		}
	case 50:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:268
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 51:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:273
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
		//line a.y:284
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 53:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:289
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
		//line a.y:300
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 55:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:305
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 56:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:310
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 57:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:317
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
			yyVAL.addr2.to.Offset = yyDollar[5].lval
		}
	case 58:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:325
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
		//line a.y:336
		{
			if yyDollar[1].addr.Type != obj.TYPE_CONST || yyDollar[3].addr.Type != obj.TYPE_CONST {
				yyerror("arguments to PCDATA must be integer constants")
			}
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 60:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:346
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
	case 65:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:365
		{
			yyVAL.addr = yyDollar[2].addr
		}
	case 66:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:369
		{
			yyVAL.addr = yyDollar[2].addr
		}
	case 72:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:382
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyDollar[1].lval + int64(asm.PC)
		}
	case 73:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:388
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
		//line a.y:400
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 75:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:406
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 76:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:412
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 77:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:418
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 78:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:424
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = REG_SP
		}
	case 79:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:430
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 80:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:438
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
			yyVAL.addr.Offset = yyDollar[2].lval
		}
	case 81:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:444
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
		//line a.y:453
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SCONST
			yyVAL.addr.Val = yyDollar[2].sval
		}
	case 83:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:459
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = yyDollar[2].dval
		}
	case 84:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:465
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = yyDollar[3].dval
		}
	case 85:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:471
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = -yyDollar[4].dval
		}
	case 86:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:477
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = -yyDollar[3].dval
		}
	case 87:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:485
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.Val = int32(obj.ArgsSizeUnknown)
		}
	case 88:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:492
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -yyDollar[2].lval
			yyVAL.addr.Val = int32(obj.ArgsSizeUnknown)
		}
	case 89:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:499
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.Val = int32(yyDollar[3].lval)
		}
	case 90:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:506
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -yyDollar[2].lval
			yyVAL.addr.Val = int32(yyDollar[4].lval)
		}
	case 93:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:520
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 94:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:526
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 95:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:533
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = REG_SP
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 96:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:540
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
		//line a.y:549
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
		//line a.y:559
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
		//line a.y:569
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[2].lval)
		}
	case 100:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:575
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = REG_SP
		}
	case 101:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:581
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 102:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:588
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Index = int16(yyDollar[2].lval)
			yyVAL.addr.Scale = int16(yyDollar[4].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 103:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line a.y:596
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
		//line a.y:607
		{
			yyVAL.addr = yyDollar[1].addr
		}
	case 105:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:611
		{
			yyVAL.addr = yyDollar[1].addr
			yyVAL.addr.Index = int16(yyDollar[3].lval)
			yyVAL.addr.Scale = int16(yyDollar[5].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 106:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:620
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyDollar[4].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 0)
			yyVAL.addr.Offset = yyDollar[2].lval
		}
	case 107:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:628
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = obj.NAME_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 1)
			yyVAL.addr.Offset = yyDollar[4].lval
		}
	case 108:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:637
		{
			yyVAL.lval = 0
		}
	case 109:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:641
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 110:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:645
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 112:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:652
		{
			yyVAL.lval = obj.NAME_AUTO
		}
	case 115:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:660
		{
			yyVAL.lval = yyDollar[1].sym.Value
		}
	case 116:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:664
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 117:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:668
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 118:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:672
		{
			yyVAL.lval = ^yyDollar[2].lval
		}
	case 119:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:676
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 121:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:683
		{
			yyVAL.lval = yyDollar[1].lval + yyDollar[3].lval
		}
	case 122:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:687
		{
			yyVAL.lval = yyDollar[1].lval - yyDollar[3].lval
		}
	case 123:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:691
		{
			yyVAL.lval = yyDollar[1].lval * yyDollar[3].lval
		}
	case 124:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:695
		{
			yyVAL.lval = yyDollar[1].lval / yyDollar[3].lval
		}
	case 125:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:699
		{
			yyVAL.lval = yyDollar[1].lval % yyDollar[3].lval
		}
	case 126:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:703
		{
			yyVAL.lval = yyDollar[1].lval << uint(yyDollar[4].lval)
		}
	case 127:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:707
		{
			yyVAL.lval = yyDollar[1].lval >> uint(yyDollar[4].lval)
		}
	case 128:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:711
		{
			yyVAL.lval = yyDollar[1].lval & yyDollar[3].lval
		}
	case 129:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:715
		{
			yyVAL.lval = yyDollar[1].lval ^ yyDollar[3].lval
		}
	case 130:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:719
		{
			yyVAL.lval = yyDollar[1].lval | yyDollar[3].lval
		}
	}
	goto yystack /* stack new state and value */
}
