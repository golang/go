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

const yyNprod = 133
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 593

var yyAct = [...]int{

	52, 227, 41, 3, 80, 208, 269, 64, 123, 50,
	51, 79, 54, 170, 268, 74, 267, 118, 85, 72,
	83, 263, 73, 255, 253, 241, 239, 84, 81, 237,
	221, 100, 102, 112, 219, 210, 112, 209, 171, 240,
	234, 107, 211, 62, 174, 143, 138, 119, 65, 207,
	111, 115, 249, 113, 112, 231, 67, 169, 120, 121,
	122, 230, 226, 92, 94, 96, 128, 225, 224, 154,
	104, 106, 74, 58, 57, 153, 136, 112, 129, 85,
	151, 83, 150, 149, 139, 141, 148, 147, 84, 81,
	140, 146, 142, 145, 137, 144, 63, 55, 58, 57,
	135, 43, 45, 48, 44, 46, 49, 40, 134, 47,
	69, 127, 56, 37, 155, 40, 35, 34, 53, 98,
	59, 31, 55, 32, 33, 223, 176, 177, 222, 217,
	60, 215, 220, 112, 120, 243, 114, 56, 74, 242,
	216, 236, 183, 76, 173, 59, 58, 57, 256, 166,
	168, 188, 184, 192, 194, 196, 167, 112, 112, 112,
	112, 112, 195, 229, 112, 112, 112, 258, 58, 57,
	55, 212, 251, 252, 197, 198, 199, 200, 201, 182,
	120, 204, 205, 206, 218, 56, 228, 114, 264, 257,
	65, 76, 55, 59, 190, 191, 184, 248, 38, 166,
	168, 152, 42, 112, 112, 75, 167, 56, 36, 235,
	261, 71, 65, 76, 238, 59, 124, 89, 125, 126,
	232, 233, 158, 159, 160, 244, 260, 88, 245, 254,
	213, 181, 108, 109, 246, 125, 126, 247, 203, 250,
	175, 202, 185, 186, 187, 25, 189, 16, 15, 6,
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
var yyPact = [...]int{

	-1000, -1000, 250, -1000, 72, -1000, 74, 67, 65, 61,
	374, 294, 294, 394, 159, -1000, -1000, 274, 354, 294,
	294, 294, 394, -5, -5, -1000, 294, 294, 84, 488,
	488, -1000, 502, -1000, -1000, 502, -1000, -1000, -1000, 394,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, -1000, -3, 428, -7, -1000, -1000, 502, 502, 502,
	209, -1000, 59, -1000, -1000, 408, -1000, 56, -1000, 48,
	-1000, 448, -1000, 42, -8, 226, 502, -1000, 334, -1000,
	-1000, -1000, 64, -1000, -1000, -9, 209, -1000, -1000, -1000,
	394, -1000, 41, -1000, 39, -1000, 35, -1000, 34, -1000,
	31, -1000, -1000, -1000, 30, -1000, 28, 189, 23, 17,
	250, 555, -1000, 555, -1000, 111, 2, -17, 282, 106,
	-1000, -1000, -1000, -10, 232, 502, 502, -1000, -1000, -1000,
	-1000, -1000, 476, 460, 394, 294, -1000, 448, 117, -1000,
	-1000, -1000, -1000, 161, -10, 394, 394, 394, 314, 394,
	294, 294, 502, 435, 137, -1000, 502, 502, 502, 502,
	502, 234, 230, 502, 502, 502, -6, -18, -20, -12,
	502, -1000, -1000, 219, 95, 226, -1000, -1000, -21, 89,
	-1000, -1000, -1000, -1000, -25, 79, 76, -1000, 16, 15,
	-1000, -1000, 10, 153, 9, -1000, 3, 211, 211, -1000,
	-1000, -1000, 502, 502, 579, 572, 564, -14, 502, -1000,
	-1000, 103, -26, 502, -29, -1000, -1000, -1000, -15, -1000,
	-30, -1000, 101, 96, 502, 314, -5, -1000, 227, 164,
	-1, -5, 247, 247, 134, -31, 218, -1000, -32, -1000,
	112, -1000, -1000, -1000, -1000, -1000, -1000, 156, 157, 153,
	-1000, 215, 199, -1000, 502, -1000, -34, -1000, 155, -1000,
	502, 502, -39, -1000, -1000, -41, -49, -1000, -1000, -1000,
}
var yyPgo = [...]int{

	0, 0, 17, 324, 8, 202, 7, 1, 2, 12,
	4, 96, 43, 11, 9, 10, 208, 323, 198, 321,
	318, 317, 310, 309, 308, 306, 305, 302, 301, 299,
	297, 263, 254, 253, 3, 250, 249, 248, 247, 245,
}
var yyR1 = [...]int{

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
var yyR2 = [...]int{

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
var yyChk = [...]int{

	-1000, -32, -33, -34, 46, 50, -36, 2, 48, 14,
	15, 16, 17, 18, 22, -37, -38, 19, 21, 26,
	27, 28, 29, 30, 31, -39, 25, 32, 20, 23,
	24, 49, 51, 50, 50, 51, -16, 52, -18, 52,
	-11, -8, -5, 37, 40, 38, 41, 45, 39, 42,
	-14, -15, -1, 54, -9, 33, 48, 10, 9, 56,
	46, -19, -12, -11, -6, 53, -20, -12, -21, -11,
	-17, 52, -10, -6, -1, 46, 54, -22, 52, -13,
	-10, -15, 11, -8, -14, -1, 46, -23, -16, -18,
	52, -24, -12, -25, -12, -26, -12, -27, -11, -28,
	-6, -29, -6, -30, -12, -31, -12, -9, -5, -5,
	-35, -2, -1, -2, -11, 54, 38, 45, -2, 54,
	-1, -1, -1, -4, 7, 9, 10, 52, -1, -9,
	44, 43, 54, 10, 52, 52, -10, 52, 54, -4,
	-13, -8, -14, 54, -4, 52, 52, 52, 52, 52,
	52, 52, 12, 52, 52, -34, 9, 10, 11, 12,
	13, 7, 8, 6, 5, 4, 38, 45, 39, 55,
	11, 55, 55, 38, 54, 8, -1, -1, 43, 10,
	43, -11, -12, -10, 35, -11, -11, -11, -8, -11,
	-12, -12, -1, 53, -1, -6, -1, -2, -2, -2,
	-2, -2, 7, 8, -2, -2, -2, 55, 11, 55,
	55, 54, -1, 11, -3, 36, 45, 34, -4, 55,
	43, 55, 49, 49, 52, 52, 52, -7, 33, 10,
	52, 52, -2, -2, 54, -1, 38, 55, -1, 55,
	54, 55, 38, 39, -1, -8, -6, 10, 33, 53,
	-6, 38, 39, 55, 11, 55, 36, 33, 10, -7,
	11, 11, -1, 55, 33, -1, -1, 55, 55, 55,
}
var yyDef = [...]int{

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
var yyTok1 = [...]int{

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
var yyTok2 = [...]int{

	2, 3, 14, 15, 16, 17, 18, 19, 20, 21,
	22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
	42, 43, 44, 45, 46, 47, 48,
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
		//line a.y:72
		{
			stmtline = asm.Lineno
		}
	case 4:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:79
		{
			yyDollar[1].sym = asm.LabelLookup(yyDollar[1].sym)
			if yyDollar[1].sym.Type == LLAB && yyDollar[1].sym.Value != int64(asm.PC) {
				yyerror("redeclaration of %s (%s)", yyDollar[1].sym.Labelname, yyDollar[1].sym.Name)
			}
			yyDollar[1].sym.Type = LLAB
			yyDollar[1].sym.Value = int64(asm.PC)
		}
	case 9:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:94
		{
			yyDollar[1].sym.Type = LVAR
			yyDollar[1].sym.Value = yyDollar[3].lval
		}
	case 10:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:99
		{
			if yyDollar[1].sym.Value != yyDollar[3].lval {
				yyerror("redeclaration of %s", yyDollar[1].sym.Name)
			}
			yyDollar[1].sym.Value = yyDollar[3].lval
		}
	case 11:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:105
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 12:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:106
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 13:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:107
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 14:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:108
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 15:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:109
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 16:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:110
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 19:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:113
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 20:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:114
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 21:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:115
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 22:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:116
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 23:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:117
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 24:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:118
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 25:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:119
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 26:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:120
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 28:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:122
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 29:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:123
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr2)
		}
	case 30:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:126
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 31:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:131
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 32:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:138
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 33:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:145
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 34:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:152
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 35:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:157
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 36:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:164
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[2].addr
		}
	case 37:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:169
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[1].addr
		}
	case 38:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:176
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[2].addr
		}
	case 39:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:181
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[1].addr
		}
	case 40:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:186
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 41:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:193
		{
			var a Addr2
			a.from = yyDollar[2].addr
			a.to = yyDollar[6].addr
			outcode(obj.ADATA, &a)
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 42:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:207
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.ATEXT, &Addr2{from: yyDollar[2].addr, to: yyDollar[5].addr})
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
			}
		}
	case 43:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:215
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.ATEXT, &Addr2{from: yyDollar[2].addr, to: yyDollar[7].addr})
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 44:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:227
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.AGLOBL, &Addr2{from: yyDollar[2].addr, to: yyDollar[4].addr})
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
			}
		}
	case 45:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:235
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(obj.AGLOBL, &Addr2{from: yyDollar[2].addr, to: yyDollar[6].addr})
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 46:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:247
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[2].addr
		}
	case 47:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:252
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = yyDollar[1].addr
		}
	case 50:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:263
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 51:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:268
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
		//line a.y:279
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 53:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:284
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
		//line a.y:295
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 55:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:300
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 56:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:305
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 57:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:312
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.from3 = yyDollar[3].addr
			yyVAL.addr2.to.Type = obj.TYPE_MEM // to give library something to do
			yyVAL.addr2.to.Offset = yyDollar[5].lval
		}
	case 58:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:321
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.from3 = yyDollar[3].addr
			yyVAL.addr2.to = yyDollar[5].addr
		}
	case 59:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:328
		{
			yyVAL.addr2.from = nullgen
			yyVAL.addr2.to = nullgen
		}
	case 60:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:333
		{
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = nullgen
		}
	case 61:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:340
		{
			if yyDollar[1].addr.Type != obj.TYPE_CONST || yyDollar[3].addr.Type != obj.TYPE_CONST {
				yyerror("arguments to asm.PCDATA must be integer constants")
			}
			yyVAL.addr2.from = yyDollar[1].addr
			yyVAL.addr2.to = yyDollar[3].addr
		}
	case 62:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:350
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
	case 67:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:369
		{
			yyVAL.addr = yyDollar[2].addr
		}
	case 68:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:373
		{
			yyVAL.addr = yyDollar[2].addr
		}
	case 73:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:385
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyDollar[1].lval + int64(asm.PC)
		}
	case 74:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:391
		{
			yyDollar[1].sym = asm.LabelLookup(yyDollar[1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyDollar[1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyDollar[1].sym.Labelname)
			}
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyDollar[1].sym.Value + yyDollar[2].lval
		}
	case 75:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:403
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 76:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:409
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 77:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:415
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 78:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:421
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 79:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:427
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = x86.REG_SP
		}
	case 80:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:433
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 81:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:439
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 82:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:447
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
			yyVAL.addr.Offset = yyDollar[2].lval
		}
	case 83:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:453
		{
			yyVAL.addr = yyDollar[2].addr
			yyVAL.addr.Type = obj.TYPE_ADDR
			/*
				if($2.Type == x86.D_AUTO || $2.Type == x86.D_PARAM)
					yyerror("constant cannot be automatic: %s",
						$2.sym.Name);
			*/
		}
	case 84:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:462
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SCONST
			yyVAL.addr.Val = (yyDollar[2].sval + "\x00\x00\x00\x00\x00\x00\x00\x00")[:8]
		}
	case 85:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:468
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = yyDollar[2].dval
		}
	case 86:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:474
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = yyDollar[3].dval
		}
	case 87:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:480
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = -yyDollar[4].dval
		}
	case 88:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:486
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = -yyDollar[3].dval
		}
	case 91:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:498
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 92:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:504
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 93:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:511
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = x86.REG_SP
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 94:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:518
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 95:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:525
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.Index = int16(yyDollar[3].lval)
			yyVAL.addr.Scale = int16(yyDollar[5].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 96:
		yyDollar = yyS[yypt-9 : yypt+1]
		//line a.y:534
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.Index = int16(yyDollar[6].lval)
			yyVAL.addr.Scale = int16(yyDollar[8].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 97:
		yyDollar = yyS[yypt-9 : yypt+1]
		//line a.y:544
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
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:554
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[2].lval)
		}
	case 99:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:560
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = x86.REG_SP
		}
	case 100:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:566
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Index = int16(yyDollar[2].lval)
			yyVAL.addr.Scale = int16(yyDollar[4].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 101:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line a.y:574
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[2].lval)
			yyVAL.addr.Index = int16(yyDollar[5].lval)
			yyVAL.addr.Scale = int16(yyDollar[7].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 102:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:585
		{
			yyVAL.addr = yyDollar[1].addr
		}
	case 103:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:589
		{
			yyVAL.addr = yyDollar[1].addr
			yyVAL.addr.Index = int16(yyDollar[3].lval)
			yyVAL.addr.Scale = int16(yyDollar[5].lval)
			checkscale(yyVAL.addr.Scale)
		}
	case 104:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:598
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyDollar[4].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 0)
			yyVAL.addr.Offset = yyDollar[2].lval
		}
	case 105:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:606
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = obj.NAME_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 1)
			yyVAL.addr.Offset = yyDollar[4].lval
		}
	case 106:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:615
		{
			yyVAL.lval = 0
		}
	case 107:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:619
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 108:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:623
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 110:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:630
		{
			yyVAL.lval = obj.NAME_AUTO
		}
	case 113:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:638
		{
			yyVAL.lval = yyDollar[1].sym.Value
		}
	case 114:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:642
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 115:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:646
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 116:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:650
		{
			yyVAL.lval = ^yyDollar[2].lval
		}
	case 117:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:654
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 118:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:660
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.Val = int32(obj.ArgsSizeUnknown)
		}
	case 119:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:667
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -yyDollar[2].lval
			yyVAL.addr.Val = int32(obj.ArgsSizeUnknown)
		}
	case 120:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:674
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = yyDollar[1].lval
			yyVAL.addr.Val = int32(yyDollar[3].lval)
		}
	case 121:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:681
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -yyDollar[2].lval
			yyVAL.addr.Val = int32(yyDollar[4].lval)
		}
	case 123:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:691
		{
			yyVAL.lval = yyDollar[1].lval + yyDollar[3].lval
		}
	case 124:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:695
		{
			yyVAL.lval = yyDollar[1].lval - yyDollar[3].lval
		}
	case 125:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:699
		{
			yyVAL.lval = yyDollar[1].lval * yyDollar[3].lval
		}
	case 126:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:703
		{
			yyVAL.lval = yyDollar[1].lval / yyDollar[3].lval
		}
	case 127:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:707
		{
			yyVAL.lval = yyDollar[1].lval % yyDollar[3].lval
		}
	case 128:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:711
		{
			yyVAL.lval = yyDollar[1].lval << uint(yyDollar[4].lval)
		}
	case 129:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:715
		{
			yyVAL.lval = yyDollar[1].lval >> uint(yyDollar[4].lval)
		}
	case 130:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:719
		{
			yyVAL.lval = yyDollar[1].lval & yyDollar[3].lval
		}
	case 131:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:723
		{
			yyVAL.lval = yyDollar[1].lval ^ yyDollar[3].lval
		}
	case 132:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:727
		{
			yyVAL.lval = yyDollar[1].lval | yyDollar[3].lval
		}
	}
	goto yystack /* stack new state and value */
}
