//line a.y:32
package main

import __yyfmt__ "fmt"

//line a.y:32
import (
	"cmd/internal/asm"
	"cmd/internal/obj"
	. "cmd/internal/obj/arm"
)

//line a.y:41
type yySymType struct {
	yys  int
	sym  *asm.Sym
	lval int32
	dval float64
	sval string
	addr obj.Addr
}

const LTYPE1 = 57346
const LTYPE2 = 57347
const LTYPE3 = 57348
const LTYPE4 = 57349
const LTYPE5 = 57350
const LTYPE6 = 57351
const LTYPE7 = 57352
const LTYPE8 = 57353
const LTYPE9 = 57354
const LTYPEA = 57355
const LTYPEB = 57356
const LTYPEC = 57357
const LTYPED = 57358
const LTYPEE = 57359
const LTYPEG = 57360
const LTYPEH = 57361
const LTYPEI = 57362
const LTYPEJ = 57363
const LTYPEK = 57364
const LTYPEL = 57365
const LTYPEM = 57366
const LTYPEN = 57367
const LTYPEBX = 57368
const LTYPEPLD = 57369
const LCONST = 57370
const LSP = 57371
const LSB = 57372
const LFP = 57373
const LPC = 57374
const LTYPEX = 57375
const LTYPEPC = 57376
const LTYPEF = 57377
const LR = 57378
const LREG = 57379
const LF = 57380
const LFREG = 57381
const LC = 57382
const LCREG = 57383
const LPSR = 57384
const LFCR = 57385
const LCOND = 57386
const LS = 57387
const LAT = 57388
const LGLOBL = 57389
const LFCONST = 57390
const LSCONST = 57391
const LNAME = 57392
const LLAB = 57393
const LVAR = 57394

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
	"LTYPE1",
	"LTYPE2",
	"LTYPE3",
	"LTYPE4",
	"LTYPE5",
	"LTYPE6",
	"LTYPE7",
	"LTYPE8",
	"LTYPE9",
	"LTYPEA",
	"LTYPEB",
	"LTYPEC",
	"LTYPED",
	"LTYPEE",
	"LTYPEG",
	"LTYPEH",
	"LTYPEI",
	"LTYPEJ",
	"LTYPEK",
	"LTYPEL",
	"LTYPEM",
	"LTYPEN",
	"LTYPEBX",
	"LTYPEPLD",
	"LCONST",
	"LSP",
	"LSB",
	"LFP",
	"LPC",
	"LTYPEX",
	"LTYPEPC",
	"LTYPEF",
	"LR",
	"LREG",
	"LF",
	"LFREG",
	"LC",
	"LCREG",
	"LPSR",
	"LFCR",
	"LCOND",
	"LS",
	"LAT",
	"LGLOBL",
	"LFCONST",
	"LSCONST",
	"LNAME",
	"LLAB",
	"LVAR",
	"':'",
	"'='",
	"';'",
	"','",
	"'['",
	"']'",
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
	-1, 196,
	68, 63,
	-2, 53,
}

const yyNprod = 134
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 708

var yyAct = [...]int{

	125, 328, 259, 73, 202, 79, 85, 106, 91, 195,
	3, 129, 84, 115, 75, 72, 338, 324, 278, 136,
	78, 77, 178, 177, 176, 174, 175, 169, 170, 171,
	172, 173, 301, 86, 86, 289, 52, 61, 62, 90,
	89, 86, 86, 86, 71, 103, 104, 284, 277, 86,
	58, 57, 276, 120, 275, 96, 99, 101, 263, 112,
	145, 105, 105, 224, 110, 334, 321, 302, 206, 105,
	140, 123, 141, 143, 146, 113, 249, 152, 151, 55,
	114, 58, 57, 197, 139, 92, 190, 165, 94, 341,
	148, 149, 95, 93, 44, 46, 164, 154, 150, 231,
	160, 128, 87, 56, 108, 64, 300, 311, 254, 167,
	55, 60, 45, 59, 152, 97, 253, 58, 57, 86,
	88, 255, 196, 340, 111, 184, 188, 189, 118, 191,
	333, 124, 126, 331, 56, 327, 325, 309, 39, 199,
	192, 108, 60, 308, 59, 211, 55, 58, 57, 92,
	45, 305, 94, 295, 86, 226, 95, 93, 292, 222,
	223, 288, 103, 104, 103, 104, 267, 86, 266, 262,
	56, 103, 104, 258, 38, 225, 55, 45, 60, 108,
	59, 245, 221, 45, 86, 233, 220, 144, 234, 235,
	236, 237, 238, 239, 252, 219, 242, 243, 244, 250,
	56, 246, 218, 247, 37, 248, 223, 200, 60, 216,
	59, 264, 100, 257, 215, 198, 194, 193, 183, 265,
	214, 182, 268, 269, 180, 271, 166, 153, 279, 279,
	279, 279, 137, 53, 53, 53, 127, 35, 36, 272,
	231, 273, 274, 317, 74, 83, 83, 281, 282, 283,
	217, 330, 329, 251, 196, 83, 293, 196, 323, 116,
	286, 287, 122, 291, 58, 57, 294, 90, 89, 314,
	133, 134, 135, 304, 303, 90, 270, 256, 261, 297,
	299, 147, 178, 177, 176, 174, 175, 169, 170, 171,
	172, 173, 138, 55, 92, 315, 312, 94, 162, 298,
	159, 95, 93, 316, 155, 156, 260, 157, 319, 310,
	58, 57, 318, 90, 89, 240, 313, 56, 241, 103,
	104, 181, 326, 80, 186, 60, 230, 59, 332, 229,
	322, 83, 335, 290, 92, 336, 228, 94, 296, 55,
	201, 95, 93, 207, 208, 209, 204, 203, 205, 285,
	212, 213, 306, 158, 337, 103, 104, 94, 131, 132,
	342, 95, 93, 56, 107, 107, 83, 227, 121, 102,
	8, 76, 107, 59, 7, 98, 133, 232, 130, 83,
	131, 132, 9, 10, 11, 12, 14, 15, 16, 17,
	18, 19, 20, 22, 23, 34, 83, 24, 25, 28,
	26, 27, 29, 30, 13, 31, 171, 172, 173, 58,
	57, 109, 32, 33, 2, 58, 57, 1, 119, 92,
	185, 142, 94, 320, 339, 21, 95, 93, 4, 0,
	5, 0, 0, 6, 103, 104, 0, 0, 55, 0,
	280, 280, 280, 280, 55, 92, 45, 0, 94, 0,
	58, 57, 95, 93, 90, 89, 0, 0, 81, 82,
	103, 104, 56, 0, 0, 0, 54, 0, 56, 0,
	60, 0, 59, 0, 0, 87, 76, 0, 59, 55,
	92, 0, 0, 94, 0, 0, 0, 95, 93, 90,
	89, 0, 0, 81, 82, 58, 163, 0, 58, 57,
	0, 54, 0, 56, 0, 122, 204, 203, 205, 251,
	87, 76, 0, 59, 178, 177, 176, 174, 175, 169,
	170, 171, 172, 173, 55, 58, 57, 55, 0, 0,
	0, 58, 57, 0, 58, 57, 0, 58, 57, 169,
	170, 171, 172, 173, 162, 161, 54, 0, 56, 187,
	0, 56, 0, 0, 55, 0, 76, 0, 59, 76,
	55, 59, 0, 55, 0, 0, 55, 0, 0, 0,
	0, 0, 204, 203, 205, 94, 117, 0, 56, 95,
	93, 210, 54, 0, 56, 54, 60, 56, 59, 0,
	56, 0, 76, 0, 59, 60, 0, 59, 76, 0,
	59, 178, 177, 176, 174, 175, 169, 170, 171, 172,
	173, 178, 177, 176, 174, 175, 169, 170, 171, 172,
	173, 178, 177, 176, 174, 175, 169, 170, 171, 172,
	173, 92, 0, 0, 94, 0, 0, 0, 95, 93,
	40, 0, 0, 0, 0, 0, 103, 104, 0, 0,
	0, 41, 42, 43, 0, 0, 47, 48, 49, 50,
	51, 0, 0, 307, 63, 0, 65, 66, 67, 68,
	69, 70, 179, 177, 176, 174, 175, 169, 170, 171,
	172, 173, 168, 178, 177, 176, 174, 175, 169, 170,
	171, 172, 173, 176, 174, 175, 169, 170, 171, 172,
	173, 174, 175, 169, 170, 171, 172, 173,
}
var yyPact = [...]int{

	-1000, -1000, 368, -1000, 174, 140, -1000, 109, 73, -1000,
	-1000, -1000, -1000, 84, 84, -1000, -1000, -1000, -1000, -1000,
	525, 525, 525, -1000, 84, -1000, -1000, -1000, -1000, -1000,
	-1000, 522, 441, 441, 84, -1000, 400, 400, -1000, -1000,
	110, 110, 406, 117, 5, 84, 516, 117, 110, 301,
	380, 117, 170, 31, 371, -1000, -1000, 400, 400, 400,
	400, 166, 280, 592, 33, 265, -9, 265, 108, 592,
	592, -1000, 28, -1000, 8, -1000, 255, 161, -1000, -1000,
	27, -1000, -1000, 8, -1000, -1000, 297, 486, -1000, -1000,
	26, -1000, -1000, -1000, -1000, 17, 160, -1000, 368, 617,
	-1000, 607, 158, -1000, -1000, -1000, -1000, -1000, 400, 155,
	152, 489, -1000, 295, -1000, -1000, 16, 349, 441, 151,
	150, 295, 13, 149, 5, -1000, -1000, 138, 307, -2,
	335, 400, 400, -1000, -1000, -1000, 510, 72, 400, 84,
	-1000, 148, 143, -1000, -1000, 240, 136, 129, 120, 116,
	315, 533, -8, 441, 295, 360, 328, 321, 318, 8,
	-1000, -1000, -1000, 41, 400, 400, 441, -1000, -1000, 400,
	400, 400, 400, 400, 308, 310, 400, 400, 400, -1000,
	295, -1000, 295, 441, -1000, -1000, 6, 371, -1000, -1000,
	211, -1000, -1000, 295, 49, 40, 111, 315, 5, 107,
	268, 103, -13, -1000, -1000, -1000, 307, 349, -1000, -1000,
	-1000, -1000, 102, 100, -1000, 219, 227, 182, 219, 400,
	295, 295, -17, -19, -1000, -1000, -23, 255, 255, 255,
	255, -1000, -24, 278, -1000, 395, 395, -1000, -1000, -1000,
	400, 400, 694, 687, 668, 95, -1000, -1000, -1000, 467,
	-2, -36, 84, 295, 92, 295, 295, 87, 295, -1000,
	289, 242, 37, -1000, -39, -3, 35, 33, -1000, -1000,
	85, 84, 597, 77, 71, -1000, -1000, -1000, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, 530, 530, 295, -1000,
	-1000, 39, 528, -1000, -1000, 46, -1000, -1000, 231, 285,
	268, -1000, 203, -1000, -1000, 219, -1000, 295, -4, 295,
	-1000, -1000, -1000, -1000, -1000, 220, -1000, -54, -1000, 70,
	-1000, 295, 69, -1000, -1000, 201, 67, 295, 64, -1000,
	-5, 295, -1000, 201, 400, -55, 57, 18, -1000, -1000,
	400, -1000, 679,
}
var yyPgo = [...]int{

	0, 212, 19, 424, 4, 11, 8, 0, 1, 18,
	640, 9, 21, 13, 20, 423, 6, 323, 120, 421,
	2, 7, 5, 15, 12, 14, 420, 3, 369, 417,
	414, 10, 375, 374, 80,
}
var yyR1 = [...]int{

	0, 29, 30, 29, 32, 31, 31, 31, 31, 31,
	31, 33, 33, 33, 33, 33, 33, 33, 33, 33,
	33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
	33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
	33, 33, 33, 33, 33, 33, 20, 20, 20, 20,
	10, 10, 10, 34, 34, 13, 13, 22, 22, 22,
	22, 18, 18, 11, 11, 11, 12, 12, 12, 12,
	12, 12, 12, 12, 12, 26, 26, 25, 27, 27,
	24, 24, 24, 28, 28, 28, 21, 14, 15, 17,
	17, 17, 17, 9, 9, 6, 6, 6, 7, 7,
	8, 8, 19, 19, 16, 16, 23, 23, 23, 5,
	5, 5, 4, 4, 4, 1, 1, 1, 1, 1,
	1, 3, 3, 2, 2, 2, 2, 2, 2, 2,
	2, 2, 2, 2,
}
var yyR2 = [...]int{

	0, 0, 0, 3, 0, 4, 4, 4, 1, 2,
	2, 7, 6, 5, 5, 5, 4, 4, 3, 3,
	4, 6, 7, 7, 7, 6, 6, 3, 5, 7,
	4, 6, 6, 4, 3, 5, 5, 7, 6, 12,
	7, 9, 2, 4, 4, 2, 1, 2, 3, 4,
	0, 2, 2, 0, 2, 4, 2, 2, 2, 2,
	1, 2, 3, 1, 3, 3, 1, 1, 1, 4,
	1, 1, 1, 1, 1, 1, 1, 3, 1, 4,
	1, 4, 1, 1, 1, 1, 2, 1, 5, 4,
	4, 4, 4, 1, 1, 1, 1, 4, 1, 1,
	1, 4, 1, 1, 1, 4, 4, 5, 7, 0,
	2, 2, 1, 1, 1, 1, 1, 2, 2, 2,
	3, 0, 2, 1, 3, 3, 3, 3, 3, 4,
	4, 3, 3, 3,
}
var yyChk = [...]int{

	-1000, -29, -30, -31, 60, 62, 65, -33, 2, 14,
	15, 16, 17, 36, 18, 19, 20, 21, 22, 23,
	24, 57, 25, 26, 29, 30, 32, 33, 31, 34,
	35, 37, 44, 45, 27, 63, 64, 64, 65, 65,
	-10, -10, -10, -10, -34, 66, -34, -10, -10, -10,
	-10, -10, -23, -1, 60, 38, 62, 10, 9, 72,
	70, -23, -23, -10, -34, -10, -10, -10, -10, -10,
	-10, -24, -23, -27, -1, -25, 70, -12, -14, -22,
	-17, 52, 53, -1, -24, -16, -7, 69, -18, 49,
	48, -6, 39, 47, 42, 46, -12, -34, -32, -2,
	-1, -2, -28, 54, 55, -14, -21, -17, 69, -28,
	-12, -34, -25, 70, -34, -13, -1, 60, -34, -28,
	-27, 67, -1, -14, -34, -7, -34, 66, 70, -5,
	7, 9, 10, -1, -1, -1, -2, 66, 12, -14,
	-22, -16, -19, -16, -18, 69, -16, -1, -14, -14,
	70, 70, -7, 66, 70, 7, 8, 10, 56, -1,
	-24, 59, 58, 10, 70, 70, 66, -31, 65, 9,
	10, 11, 12, 13, 7, 8, 6, 5, 4, 65,
	66, -1, 66, 66, -13, -26, -1, 60, -25, -23,
	70, -5, -12, 66, 66, -11, -7, 70, 66, -25,
	69, -1, -4, 40, 39, 41, 70, 8, -1, -1,
	71, -21, -1, -1, -34, 66, 66, 10, 66, 66,
	66, 66, -6, -6, 71, -12, -7, 7, 8, 8,
	8, 58, -1, -2, -12, -2, -2, -2, -2, -2,
	7, 8, -2, -2, -2, -7, -14, -14, -12, 70,
	-5, 42, -7, 67, 68, 10, -34, -25, 66, -20,
	38, 10, 66, 71, -4, -5, 66, 66, -16, -16,
	49, -16, -2, -14, -14, 71, 71, 71, -9, -7,
	-1, -9, -9, -9, 71, 71, -2, -2, 66, 71,
	-34, -11, 66, -7, -11, 66, -34, -14, 10, 38,
	69, 71, 70, -21, -22, 66, -34, 66, 66, 66,
	-14, 68, -27, -14, 38, 10, -20, 40, -16, -7,
	-15, 70, -14, 38, 71, 66, -7, 66, -8, 51,
	50, 66, -7, 66, 70, -7, -8, -2, 71, -3,
	66, 71, -2,
}
var yyDef = [...]int{

	1, -2, 0, 3, 0, 0, 8, 0, 0, 50,
	50, 50, 50, 53, 53, 50, 50, 50, 50, 50,
	0, 0, 0, 50, 53, 50, 50, 50, 50, 50,
	50, 0, 0, 0, 53, 4, 0, 0, 9, 10,
	0, 0, 0, 53, 0, 53, 0, 53, 0, 0,
	53, 53, 0, 0, 109, 115, 116, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 42, 80, 82, 0, 78, 0, 0, 66, 67,
	68, 70, 71, 72, 73, 74, 87, 0, 60, 104,
	0, 98, 99, 95, 96, 0, 0, 45, 0, 0,
	123, 0, 0, 51, 52, 83, 84, 85, 0, 0,
	0, 0, 18, 0, 54, 19, 0, 109, 0, 0,
	0, 0, 0, 0, 0, 87, 27, 0, 0, 0,
	0, 0, 0, 117, 118, 119, 0, 0, 0, 53,
	34, 0, 0, 102, 103, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 57,
	58, 59, 61, 0, 0, 0, 0, 5, 6, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
	0, 86, 0, 0, 16, 17, 0, 109, 75, 76,
	0, 56, 20, 0, 0, 0, -2, 0, 0, 0,
	0, 0, 0, 112, 113, 114, 0, 109, 110, 111,
	120, 30, 0, 0, 33, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 77, 43, 0, 0, 0, 0,
	0, 62, 0, 0, 44, 124, 125, 126, 127, 128,
	0, 0, 131, 132, 133, 87, 13, 14, 15, 0,
	56, 0, 53, 0, 0, 0, 0, 53, 0, 28,
	46, 0, 0, 106, 0, 0, 0, 0, 35, 36,
	104, 53, 0, 0, 0, 81, 79, 69, 89, 93,
	94, 90, 91, 92, 105, 97, 129, 130, 12, 55,
	21, 0, 0, 64, 65, 53, 25, 26, 0, 47,
	0, 107, 0, 31, 32, 0, 38, 0, 0, 0,
	11, 22, 23, 24, 48, 0, 29, 0, 37, 0,
	40, 0, 0, 49, 108, 0, 0, 0, 0, 100,
	0, 0, 41, 0, 0, 0, 121, 0, 88, 39,
	0, 101, 122,
}
var yyTok1 = [...]int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 69, 13, 6, 3,
	70, 71, 11, 9, 66, 10, 3, 12, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 63, 65,
	7, 64, 8, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 67, 3, 68, 5, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 4, 3, 72,
}
var yyTok2 = [...]int{

	2, 3, 14, 15, 16, 17, 18, 19, 20, 21,
	22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
	42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
	52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
	62,
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
		//line a.y:73
		{
			stmtline = asm.Lineno
		}
	case 4:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:80
		{
			yyDollar[1].sym = asm.LabelLookup(yyDollar[1].sym)
			if yyDollar[1].sym.Type == LLAB && yyDollar[1].sym.Value != int64(asm.PC) {
				yyerror("redeclaration of %s", yyDollar[1].sym.Labelname)
			}
			yyDollar[1].sym.Type = LLAB
			yyDollar[1].sym.Value = int64(asm.PC)
		}
	case 6:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:90
		{
			yyDollar[1].sym.Type = LVAR
			yyDollar[1].sym.Value = int64(yyDollar[3].lval)
		}
	case 7:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:95
		{
			if yyDollar[1].sym.Value != int64(yyDollar[3].lval) {
				yyerror("redeclaration of %s", yyDollar[1].sym.Name)
			}
			yyDollar[1].sym.Value = int64(yyDollar[3].lval)
		}
	case 11:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:110
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, yyDollar[5].lval, &yyDollar[7].addr)
		}
	case 12:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:114
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, yyDollar[5].lval, &nullgen)
		}
	case 13:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:118
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, 0, &yyDollar[5].addr)
		}
	case 14:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:125
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, 0, &yyDollar[5].addr)
		}
	case 15:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:132
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, 0, &yyDollar[5].addr)
		}
	case 16:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:139
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &nullgen, 0, &yyDollar[4].addr)
		}
	case 17:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:143
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &nullgen, 0, &yyDollar[4].addr)
		}
	case 18:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:150
		{
			outcode(yyDollar[1].lval, Always, &nullgen, 0, &yyDollar[3].addr)
		}
	case 19:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:157
		{
			outcode(yyDollar[1].lval, Always, &nullgen, 0, &yyDollar[3].addr)
		}
	case 20:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:164
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &nullgen, 0, &yyDollar[4].addr)
		}
	case 21:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:171
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, yyDollar[5].lval, &nullgen)
		}
	case 22:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:178
		{
			var g obj.Addr

			g = nullgen
			g.Type = obj.TYPE_REGLIST
			g.Offset = int64(yyDollar[6].lval)
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, 0, &g)
		}
	case 23:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:187
		{
			var g obj.Addr

			g = nullgen
			g.Type = obj.TYPE_REGLIST
			g.Offset = int64(yyDollar[4].lval)
			outcode(yyDollar[1].lval, yyDollar[2].lval, &g, 0, &yyDollar[7].addr)
		}
	case 24:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:199
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[5].addr, int32(yyDollar[3].addr.Reg), &yyDollar[7].addr)
		}
	case 25:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:203
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[5].addr, int32(yyDollar[3].addr.Reg), &yyDollar[3].addr)
		}
	case 26:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:207
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[4].addr, int32(yyDollar[6].addr.Reg), &yyDollar[6].addr)
		}
	case 27:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:214
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &nullgen, 0, &nullgen)
		}
	case 28:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:221
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(yyDollar[1].lval, Always, &yyDollar[2].addr, 0, &yyDollar[5].addr)
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
			}
		}
	case 29:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:229
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(yyDollar[1].lval, Always, &yyDollar[2].addr, 0, &yyDollar[7].addr)
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = int64(yyDollar[4].lval)
			}
		}
	case 30:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:242
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(yyDollar[1].lval, Always, &yyDollar[2].addr, 0, &yyDollar[4].addr)
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
			}
		}
	case 31:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:250
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(yyDollar[1].lval, Always, &yyDollar[2].addr, 0, &yyDollar[6].addr)
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = int64(yyDollar[4].lval)
			}
		}
	case 32:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:264
		{
			outcode(yyDollar[1].lval, Always, &yyDollar[2].addr, 0, &yyDollar[6].addr)
			if asm.Pass > 1 {
				lastpc.From3 = new(obj.Addr)
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = int64(yyDollar[4].lval)
			}
		}
	case 33:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:276
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, 0, &nullgen)
		}
	case 34:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:283
		{
			outcode(yyDollar[1].lval, Always, &nullgen, 0, &yyDollar[3].addr)
		}
	case 35:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:290
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, 0, &yyDollar[5].addr)
		}
	case 36:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:294
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, 0, &yyDollar[5].addr)
		}
	case 37:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:298
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, yyDollar[5].lval, &yyDollar[7].addr)
		}
	case 38:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:302
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, int32(yyDollar[5].addr.Reg), &nullgen)
		}
	case 39:
		yyDollar = yyS[yypt-12 : yypt+1]
		//line a.y:309
		{
			var g obj.Addr

			g = nullgen
			g.Type = obj.TYPE_CONST
			g.Offset = int64(
				(0xe << 24) | /* opcode */
					(yyDollar[1].lval << 20) | /* MCR/MRC */
					((yyDollar[2].lval ^ C_SCOND_XOR) << 28) | /* scond */
					((yyDollar[3].lval & 15) << 8) | /* coprocessor number */
					((yyDollar[5].lval & 7) << 21) | /* coprocessor operation */
					((yyDollar[7].lval & 15) << 12) | /* arm register */
					((yyDollar[9].lval & 15) << 16) | /* Crn */
					((yyDollar[11].lval & 15) << 0) | /* Crm */
					((yyDollar[12].lval & 7) << 5) | /* coprocessor information */
					(1 << 4)) /* must be set */
			outcode(AMRC, Always, &nullgen, 0, &g)
		}
	case 40:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:321
		{
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, int32(yyDollar[5].addr.Reg), &yyDollar[7].addr)
		}
	case 41:
		yyDollar = yyS[yypt-9 : yypt+1]
		//line a.y:329
		{
			yyDollar[7].addr.Type = obj.TYPE_REGREG2
			yyDollar[7].addr.Offset = int64(yyDollar[9].lval)
			outcode(yyDollar[1].lval, yyDollar[2].lval, &yyDollar[3].addr, int32(yyDollar[5].addr.Reg), &yyDollar[7].addr)
		}
	case 42:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:338
		{
			outcode(yyDollar[1].lval, Always, &yyDollar[2].addr, 0, &nullgen)
		}
	case 43:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:345
		{
			if yyDollar[2].addr.Type != obj.TYPE_CONST || yyDollar[4].addr.Type != obj.TYPE_CONST {
				yyerror("arguments to PCDATA must be integer constants")
			}
			outcode(yyDollar[1].lval, Always, &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 44:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:355
		{
			if yyDollar[2].addr.Type != obj.TYPE_CONST {
				yyerror("index for FUNCDATA must be integer constant")
			}
			if yyDollar[4].addr.Type != obj.NAME_EXTERN && yyDollar[4].addr.Type != obj.NAME_STATIC && yyDollar[4].addr.Type != obj.TYPE_MEM {
				yyerror("value for FUNCDATA must be symbol reference")
			}
			outcode(yyDollar[1].lval, Always, &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 45:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:368
		{
			outcode(yyDollar[1].lval, Always, &nullgen, 0, &nullgen)
		}
	case 46:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:374
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = int64(yyDollar[1].lval)
			yyVAL.addr.Val = int32(obj.ArgsSizeUnknown)
		}
	case 47:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:381
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -int64(yyDollar[2].lval)
			yyVAL.addr.Val = int32(obj.ArgsSizeUnknown)
		}
	case 48:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:388
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = int64(yyDollar[1].lval)
			yyVAL.addr.Val = int32(yyDollar[3].lval)
		}
	case 49:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:395
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -int64(yyDollar[2].lval)
			yyVAL.addr.Val = int32(yyDollar[4].lval)
		}
	case 50:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:403
		{
			yyVAL.lval = Always
		}
	case 51:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:407
		{
			yyVAL.lval = (yyDollar[1].lval & ^C_SCOND) | yyDollar[2].lval
		}
	case 52:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:411
		{
			yyVAL.lval = yyDollar[1].lval | yyDollar[2].lval
		}
	case 55:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:420
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = int64(yyDollar[1].lval) + int64(asm.PC)
		}
	case 56:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:426
		{
			yyDollar[1].sym = asm.LabelLookup(yyDollar[1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyDollar[1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyDollar[1].sym.Labelname)
			}
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyDollar[1].sym.Value + int64(yyDollar[2].lval)
		}
	case 57:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:437
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
			yyVAL.addr.Offset = int64(yyDollar[2].lval)
		}
	case 58:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:443
		{
			yyVAL.addr = yyDollar[2].addr
			yyVAL.addr.Type = obj.TYPE_ADDR
		}
	case 59:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:448
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SCONST
			yyVAL.addr.Val = yyDollar[2].sval
		}
	case 61:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:457
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = yyDollar[2].dval
		}
	case 62:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:463
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.Val = -yyDollar[3].dval
		}
	case 63:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:471
		{
			yyVAL.lval = 1 << uint(yyDollar[1].lval&15)
		}
	case 64:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:475
		{
			yyVAL.lval = 0
			for i := yyDollar[1].lval; i <= yyDollar[3].lval; i++ {
				yyVAL.lval |= 1 << uint(i&15)
			}
			for i := yyDollar[3].lval; i <= yyDollar[1].lval; i++ {
				yyVAL.lval |= 1 << uint(i&15)
			}
		}
	case 65:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:485
		{
			yyVAL.lval = (1 << uint(yyDollar[1].lval&15)) | yyDollar[3].lval
		}
	case 69:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:494
		{
			yyVAL.addr = yyDollar[1].addr
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
		}
	case 70:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:499
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 71:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:505
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 72:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:511
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Offset = int64(yyDollar[1].lval)
		}
	case 76:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:522
		{
			yyVAL.addr = yyDollar[1].addr
			if yyDollar[1].addr.Name != obj.NAME_EXTERN && yyDollar[1].addr.Name != obj.NAME_STATIC {
			}
		}
	case 77:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:530
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[2].lval)
			yyVAL.addr.Offset = 0
		}
	case 79:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:540
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = int64(yyDollar[1].lval)
		}
	case 81:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:550
		{
			yyVAL.addr = yyDollar[1].addr
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
		}
	case 86:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:563
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
			yyVAL.addr.Offset = int64(yyDollar[2].lval)
		}
	case 87:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:571
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 88:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:579
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REGREG
			yyVAL.addr.Reg = int16(yyDollar[2].lval)
			yyVAL.addr.Offset = int64(yyDollar[4].lval)
		}
	case 89:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:588
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SHIFT
			yyVAL.addr.Offset = int64(yyDollar[1].lval&15) | int64(yyDollar[4].lval) | (0 << 5)
		}
	case 90:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:594
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SHIFT
			yyVAL.addr.Offset = int64(yyDollar[1].lval&15) | int64(yyDollar[4].lval) | (1 << 5)
		}
	case 91:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:600
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SHIFT
			yyVAL.addr.Offset = int64(yyDollar[1].lval&15) | int64(yyDollar[4].lval) | (2 << 5)
		}
	case 92:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:606
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SHIFT
			yyVAL.addr.Offset = int64(yyDollar[1].lval&15) | int64(yyDollar[4].lval) | (3 << 5)
		}
	case 93:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:614
		{
			if yyVAL.lval < REG_R0 || yyVAL.lval > REG_R15 {
				print("register value out of range\n")
			}
			yyVAL.lval = ((yyDollar[1].lval & 15) << 8) | (1 << 4)
		}
	case 94:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:621
		{
			if yyVAL.lval < 0 || yyVAL.lval >= 32 {
				print("shift value out of range\n")
			}
			yyVAL.lval = (yyDollar[1].lval & 31) << 7
		}
	case 96:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:631
		{
			yyVAL.lval = REGPC
		}
	case 97:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:635
		{
			if yyDollar[3].lval < 0 || yyDollar[3].lval >= NREG {
				print("register value out of range\n")
			}
			yyVAL.lval = REG_R0 + yyDollar[3].lval
		}
	case 99:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:645
		{
			yyVAL.lval = REGSP
		}
	case 101:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:652
		{
			if yyDollar[3].lval < 0 || yyDollar[3].lval >= NREG {
				print("register value out of range\n")
			}
			yyVAL.lval = yyDollar[3].lval // TODO(rsc): REG_C0+$3
		}
	case 104:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:665
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 105:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:671
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(REG_F0 + yyDollar[3].lval)
		}
	case 106:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:679
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyDollar[3].lval)
			yyVAL.addr.Sym = nil
			yyVAL.addr.Offset = int64(yyDollar[1].lval)
		}
	case 107:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:687
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyDollar[4].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 0)
			yyVAL.addr.Offset = int64(yyDollar[2].lval)
		}
	case 108:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:695
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = obj.NAME_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 1)
			yyVAL.addr.Offset = int64(yyDollar[4].lval)
		}
	case 109:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:704
		{
			yyVAL.lval = 0
		}
	case 110:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:708
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 111:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:712
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 116:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:724
		{
			yyVAL.lval = int32(yyDollar[1].sym.Value)
		}
	case 117:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:728
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 118:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:732
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 119:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:736
		{
			yyVAL.lval = ^yyDollar[2].lval
		}
	case 120:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:740
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 121:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:745
		{
			yyVAL.lval = 0
		}
	case 122:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:749
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 124:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:756
		{
			yyVAL.lval = yyDollar[1].lval + yyDollar[3].lval
		}
	case 125:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:760
		{
			yyVAL.lval = yyDollar[1].lval - yyDollar[3].lval
		}
	case 126:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:764
		{
			yyVAL.lval = yyDollar[1].lval * yyDollar[3].lval
		}
	case 127:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:768
		{
			yyVAL.lval = yyDollar[1].lval / yyDollar[3].lval
		}
	case 128:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:772
		{
			yyVAL.lval = yyDollar[1].lval % yyDollar[3].lval
		}
	case 129:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:776
		{
			yyVAL.lval = yyDollar[1].lval << uint(yyDollar[4].lval)
		}
	case 130:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:780
		{
			yyVAL.lval = yyDollar[1].lval >> uint(yyDollar[4].lval)
		}
	case 131:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:784
		{
			yyVAL.lval = yyDollar[1].lval & yyDollar[3].lval
		}
	case 132:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:788
		{
			yyVAL.lval = yyDollar[1].lval ^ yyDollar[3].lval
		}
	case 133:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:792
		{
			yyVAL.lval = yyDollar[1].lval | yyDollar[3].lval
		}
	}
	goto yystack /* stack new state and value */
}
