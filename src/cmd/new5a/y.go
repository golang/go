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
const LFCONST = 57389
const LSCONST = 57390
const LNAME = 57391
const LLAB = 57392
const LVAR = 57393

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
	"LFCONST",
	"LSCONST",
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
	-1, 194,
	67, 59,
	-2, 48,
}

const yyNprod = 130
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 694

var yyAct = []int{

	123, 317, 71, 83, 98, 104, 200, 77, 82, 193,
	89, 127, 271, 73, 113, 85, 3, 106, 227, 134,
	327, 88, 87, 313, 76, 52, 52, 101, 102, 293,
	283, 278, 84, 84, 270, 72, 81, 81, 70, 69,
	84, 84, 84, 142, 269, 268, 81, 257, 84, 220,
	114, 118, 323, 120, 310, 97, 99, 110, 75, 51,
	60, 131, 132, 133, 103, 103, 294, 138, 140, 143,
	137, 144, 103, 90, 121, 149, 92, 204, 111, 246,
	93, 91, 112, 148, 195, 188, 136, 163, 101, 102,
	156, 94, 145, 146, 157, 162, 43, 45, 151, 147,
	108, 126, 302, 251, 106, 252, 62, 101, 102, 57,
	56, 179, 149, 165, 184, 250, 95, 84, 44, 329,
	194, 81, 322, 186, 182, 109, 320, 189, 316, 116,
	199, 198, 122, 124, 206, 207, 197, 314, 54, 90,
	209, 300, 92, 299, 44, 296, 93, 91, 187, 289,
	286, 84, 222, 282, 260, 81, 256, 255, 218, 219,
	44, 55, 217, 216, 215, 84, 131, 229, 59, 81,
	214, 58, 212, 90, 211, 190, 92, 196, 192, 242,
	93, 91, 84, 230, 191, 181, 81, 232, 233, 234,
	235, 236, 249, 180, 239, 240, 241, 247, 178, 44,
	164, 150, 125, 243, 86, 244, 219, 38, 37, 221,
	254, 258, 34, 35, 36, 261, 262, 259, 264, 210,
	57, 56, 228, 231, 272, 272, 272, 272, 273, 273,
	273, 273, 72, 248, 213, 265, 277, 274, 275, 276,
	245, 266, 267, 306, 57, 161, 158, 319, 318, 54,
	135, 194, 305, 287, 194, 88, 87, 280, 281, 238,
	285, 78, 292, 288, 226, 101, 102, 90, 295, 141,
	92, 225, 55, 54, 93, 91, 224, 253, 100, 59,
	291, 160, 58, 57, 56, 152, 153, 205, 154, 303,
	237, 120, 160, 159, 53, 223, 55, 88, 263, 308,
	307, 105, 105, 74, 129, 130, 58, 301, 7, 105,
	312, 315, 54, 96, 304, 90, 2, 321, 92, 107,
	1, 324, 93, 91, 325, 311, 117, 183, 101, 102,
	101, 102, 284, 139, 155, 55, 309, 290, 8, 328,
	119, 44, 74, 326, 0, 58, 0, 297, 0, 331,
	9, 10, 11, 12, 14, 15, 16, 17, 18, 19,
	20, 21, 22, 33, 0, 23, 24, 27, 25, 26,
	28, 29, 13, 30, 57, 56, 202, 201, 203, 248,
	31, 32, 176, 175, 174, 172, 173, 167, 168, 169,
	170, 171, 88, 87, 0, 4, 0, 5, 101, 102,
	6, 57, 56, 54, 90, 92, 0, 92, 0, 93,
	91, 93, 91, 88, 87, 0, 0, 79, 80, 101,
	102, 202, 201, 203, 53, 0, 55, 169, 170, 171,
	54, 90, 0, 74, 92, 85, 58, 0, 93, 91,
	88, 87, 0, 0, 79, 80, 128, 330, 129, 130,
	0, 53, 0, 55, 0, 57, 56, 0, 0, 0,
	74, 0, 85, 58, 176, 175, 174, 172, 173, 167,
	168, 169, 170, 171, 176, 175, 174, 172, 173, 167,
	168, 169, 170, 171, 54, 0, 57, 56, 0, 0,
	57, 56, 0, 0, 57, 56, 202, 201, 203, 92,
	0, 0, 0, 93, 91, 53, 0, 55, 57, 56,
	0, 0, 57, 56, 74, 54, 0, 58, 0, 54,
	0, 57, 56, 54, 0, 57, 56, 0, 0, 279,
	0, 0, 0, 0, 228, 0, 0, 54, 55, 208,
	0, 54, 55, 0, 185, 59, 55, 0, 58, 59,
	54, 106, 58, 74, 54, 0, 58, 0, 115, 0,
	55, 0, 53, 0, 55, 0, 0, 59, 0, 0,
	58, 59, 0, 55, 58, 0, 0, 55, 0, 0,
	59, 0, 0, 58, 74, 0, 0, 58, 176, 175,
	174, 172, 173, 167, 168, 169, 170, 171, 176, 175,
	174, 172, 173, 167, 168, 169, 170, 171, 176, 175,
	174, 172, 173, 167, 168, 169, 170, 171, 90, 0,
	0, 92, 0, 0, 0, 93, 91, 39, 167, 168,
	169, 170, 171, 101, 102, 0, 0, 0, 40, 41,
	42, 0, 0, 46, 47, 48, 49, 50, 0, 298,
	61, 0, 63, 64, 65, 66, 67, 68, 177, 175,
	174, 172, 173, 167, 168, 169, 170, 171, 166, 176,
	175, 174, 172, 173, 167, 168, 169, 170, 171, 174,
	172, 173, 167, 168, 169, 170, 171, 172, 173, 167,
	168, 169, 170, 171,
}
var yyPact = []int{

	-1000, -1000, 336, -1000, 150, 151, -1000, 144, 143, -1000,
	-1000, -1000, -1000, 79, 79, -1000, -1000, -1000, -1000, -1000,
	503, 503, -1000, 79, -1000, -1000, -1000, -1000, -1000, -1000,
	446, 392, 392, 79, -1000, 512, 512, -1000, -1000, 34,
	34, 365, 53, 10, 79, 499, 53, 34, 274, 276,
	53, 137, 33, 439, -1000, -1000, 512, 512, 512, 512,
	238, 579, -55, 344, -27, 344, 211, 579, 579, -1000,
	31, -1000, 15, -1000, 100, 136, -1000, -1000, 30, -1000,
	-1000, 15, -1000, -1000, 278, 235, -1000, -1000, 27, -1000,
	-1000, -1000, -1000, 19, 135, -1000, 336, 604, -1000, 594,
	133, -1000, -1000, -1000, -1000, -1000, 512, 128, 120, 485,
	-1000, 228, -1000, -1000, 17, 295, 392, 119, 113, 228,
	16, 112, 10, -1000, -1000, 481, 382, 9, 279, 512,
	512, -1000, -1000, -1000, 470, 512, 79, -1000, 109, 107,
	-1000, -1000, 224, 105, 99, 98, 97, 363, 457, -20,
	392, 228, 288, 268, 263, 256, 15, -1000, -52, -1000,
	-1000, 477, 512, 512, 392, -1000, -1000, 512, 512, 512,
	512, 512, 283, 251, 512, 512, 512, -1000, 228, -1000,
	228, 392, -1000, -1000, 11, 439, -1000, -1000, 191, -1000,
	-1000, 228, 49, 36, 95, 363, 10, 92, -1000, 91,
	-22, -1000, -1000, -1000, 382, 295, -1000, -1000, -1000, 89,
	-1000, 207, 249, 165, 207, 512, 228, 228, -24, -25,
	-1000, -1000, -35, 100, 100, 100, 100, 446, -1000, -38,
	460, -1000, 416, 416, -1000, -1000, -1000, 512, 512, 680,
	673, 654, 88, -1000, -1000, -1000, 337, 9, -39, 79,
	228, 85, 228, 228, 84, 228, -53, -1000, -40, -2,
	-55, -1000, -1000, 80, 79, 584, 78, 76, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	619, 619, 228, -1000, -1000, 35, 516, -1000, -1000, 134,
	-1000, -1000, 242, -1000, 203, -1000, 207, -1000, 228, -14,
	228, -1000, -1000, -1000, -1000, 512, -46, -1000, 72, -1000,
	228, 63, -1000, -1000, 197, 61, 228, 57, -1000, -16,
	228, -1000, 197, 512, -49, 54, 378, -1000, -1000, 512,
	-1000, 665,
}
var yyPgo = []int{

	0, 4, 19, 339, 6, 11, 10, 0, 1, 12,
	627, 9, 58, 14, 24, 336, 3, 261, 204, 333,
	5, 7, 38, 8, 13, 327, 2, 278, 320, 316,
	16, 313, 308, 82,
}
var yyR1 = []int{

	0, 28, 29, 28, 31, 30, 30, 30, 30, 30,
	30, 32, 32, 32, 32, 32, 32, 32, 32, 32,
	32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
	32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
	32, 32, 32, 32, 32, 10, 10, 10, 33, 33,
	13, 13, 21, 21, 21, 21, 21, 18, 18, 11,
	11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
	12, 25, 25, 24, 26, 26, 23, 23, 23, 27,
	27, 27, 20, 14, 15, 17, 17, 17, 17, 9,
	9, 6, 6, 6, 7, 7, 8, 8, 19, 19,
	16, 16, 22, 22, 22, 5, 5, 5, 4, 4,
	4, 1, 1, 1, 1, 1, 1, 3, 3, 2,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
}
var yyR2 = []int{

	0, 0, 0, 3, 0, 4, 4, 4, 1, 2,
	2, 7, 6, 5, 5, 5, 4, 4, 3, 3,
	4, 6, 7, 7, 7, 6, 6, 3, 4, 6,
	8, 6, 4, 3, 5, 5, 7, 6, 12, 7,
	9, 2, 4, 4, 2, 0, 2, 2, 0, 2,
	4, 2, 2, 2, 4, 2, 1, 2, 3, 1,
	3, 3, 1, 1, 1, 4, 1, 1, 1, 1,
	1, 1, 1, 3, 1, 4, 1, 4, 1, 1,
	1, 1, 2, 1, 5, 4, 4, 4, 4, 1,
	1, 1, 1, 4, 1, 1, 1, 4, 1, 1,
	1, 4, 4, 5, 7, 0, 2, 2, 1, 1,
	1, 1, 1, 2, 2, 2, 3, 0, 2, 1,
	3, 3, 3, 3, 3, 4, 4, 3, 3, 3,
}
var yyChk = []int{

	-1000, -28, -29, -30, 59, 61, 64, -32, 2, 14,
	15, 16, 17, 36, 18, 19, 20, 21, 22, 23,
	24, 25, 26, 29, 30, 32, 33, 31, 34, 35,
	37, 44, 45, 27, 62, 63, 63, 64, 64, -10,
	-10, -10, -10, -33, 65, -33, -10, -10, -10, -10,
	-10, -22, -1, 59, 38, 61, 10, 9, 71, 68,
	-22, -10, -33, -10, -10, -10, -10, -10, -10, -23,
	-22, -26, -1, -24, 68, -12, -14, -21, -17, 52,
	53, -1, -23, -16, -7, 70, -18, 49, 48, -6,
	39, 47, 42, 46, -12, -33, -31, -2, -1, -2,
	-27, 54, 55, -14, -20, -17, 70, -27, -12, -33,
	-24, 68, -33, -13, -1, 59, -33, -27, -26, 66,
	-1, -14, -33, -7, -33, 65, 68, -5, 7, 9,
	10, -1, -1, -1, -2, 12, -14, -21, -16, -19,
	-16, -18, 70, -16, -1, -14, -14, 68, 68, -7,
	65, 68, 7, 8, 10, 56, -1, -23, 11, 58,
	57, 10, 68, 68, 65, -30, 64, 9, 10, 11,
	12, 13, 7, 8, 6, 5, 4, 64, 65, -1,
	65, 65, -13, -25, -1, 59, -24, -22, 68, -5,
	-12, 65, 65, -11, -7, 68, 65, -24, -20, -1,
	-4, 40, 39, 41, 68, 8, -1, -1, 69, -1,
	-33, 65, 65, 10, 65, 65, 65, 65, -6, -6,
	69, -12, -7, 7, 8, 8, 8, 70, 57, -1,
	-2, -12, -2, -2, -2, -2, -2, 7, 8, -2,
	-2, -2, -7, -14, -14, -12, 68, -5, 42, -7,
	66, 67, 10, -33, -24, 65, 65, 69, -4, -5,
	65, -16, -16, 49, -16, -2, -14, -14, 69, 69,
	69, -9, -7, -1, -9, -9, -9, -23, 69, 69,
	-2, -2, 65, 69, -33, -11, 65, -7, -11, 65,
	-33, -14, -20, 69, 68, -21, 65, -33, 65, 65,
	65, -14, 67, -26, -14, 10, 40, -16, -7, -15,
	68, -14, -1, 69, 65, -7, 65, -8, 51, 50,
	65, -7, 65, 68, -7, -8, -2, 69, -3, 65,
	69, -2,
}
var yyDef = []int{

	1, -2, 0, 3, 0, 0, 8, 0, 0, 45,
	45, 45, 45, 48, 48, 45, 45, 45, 45, 45,
	0, 0, 45, 48, 45, 45, 45, 45, 45, 45,
	0, 0, 0, 48, 4, 0, 0, 9, 10, 0,
	0, 0, 48, 0, 48, 0, 48, 0, 0, 48,
	48, 0, 0, 105, 111, 112, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 41,
	76, 78, 0, 74, 0, 0, 62, 63, 64, 66,
	67, 68, 69, 70, 83, 0, 56, 100, 0, 94,
	95, 91, 92, 0, 0, 44, 0, 0, 119, 0,
	0, 46, 47, 79, 80, 81, 0, 0, 0, 0,
	18, 0, 49, 19, 0, 105, 0, 0, 0, 0,
	0, 0, 0, 83, 27, 0, 0, 0, 0, 0,
	0, 113, 114, 115, 0, 0, 48, 33, 0, 0,
	98, 99, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 52, 53, 0, 55,
	57, 0, 0, 0, 0, 5, 6, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 7, 0, 82,
	0, 0, 16, 17, 0, 105, 71, 72, 0, 51,
	20, 0, 0, 0, -2, 0, 0, 0, 28, 0,
	0, 108, 109, 110, 0, 105, 106, 107, 116, 0,
	32, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	73, 42, 0, 0, 0, 0, 0, 0, 58, 0,
	0, 43, 120, 121, 122, 123, 124, 0, 0, 127,
	128, 129, 83, 13, 14, 15, 0, 51, 0, 48,
	0, 0, 0, 0, 48, 0, 0, 102, 0, 0,
	0, 34, 35, 100, 48, 0, 0, 0, 77, 75,
	65, 85, 89, 90, 86, 87, 88, 54, 101, 93,
	125, 126, 12, 50, 21, 0, 0, 60, 61, 48,
	25, 26, 29, 103, 0, 31, 0, 37, 0, 0,
	0, 11, 22, 23, 24, 0, 0, 36, 0, 39,
	0, 0, 30, 104, 0, 0, 0, 0, 96, 0,
	0, 40, 0, 0, 0, 117, 0, 84, 38, 0,
	97, 118,
}
var yyTok1 = []int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 70, 13, 6, 3,
	68, 69, 11, 9, 65, 10, 3, 12, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 62, 64,
	7, 63, 8, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 66, 3, 67, 5, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 4, 3, 71,
}
var yyTok2 = []int{

	2, 3, 14, 15, 16, 17, 18, 19, 20, 21,
	22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
	42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
	52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
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
		//line a.y:73
		{
			stmtline = asm.Lineno
		}
	case 4:
		//line a.y:80
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			if yyS[yypt-1].sym.Type == LLAB && yyS[yypt-1].sym.Value != int64(asm.PC) {
				yyerror("redeclaration of %s", yyS[yypt-1].sym.Labelname)
			}
			yyS[yypt-1].sym.Type = LLAB
			yyS[yypt-1].sym.Value = int64(asm.PC)
		}
	case 6:
		//line a.y:90
		{
			yyS[yypt-3].sym.Type = LVAR
			yyS[yypt-3].sym.Value = int64(yyS[yypt-1].lval)
		}
	case 7:
		//line a.y:95
		{
			if yyS[yypt-3].sym.Value != int64(yyS[yypt-1].lval) {
				yyerror("redeclaration of %s", yyS[yypt-3].sym.Name)
			}
			yyS[yypt-3].sym.Value = int64(yyS[yypt-1].lval)
		}
	case 11:
		//line a.y:110
		{
			outcode(yyS[yypt-6].lval, yyS[yypt-5].lval, &yyS[yypt-4].addr, yyS[yypt-2].lval, &yyS[yypt-0].addr)
		}
	case 12:
		//line a.y:114
		{
			outcode(yyS[yypt-5].lval, yyS[yypt-4].lval, &yyS[yypt-3].addr, yyS[yypt-1].lval, &nullgen)
		}
	case 13:
		//line a.y:118
		{
			outcode(yyS[yypt-4].lval, yyS[yypt-3].lval, &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 14:
		//line a.y:125
		{
			outcode(yyS[yypt-4].lval, yyS[yypt-3].lval, &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 15:
		//line a.y:132
		{
			outcode(yyS[yypt-4].lval, yyS[yypt-3].lval, &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 16:
		//line a.y:139
		{
			outcode(yyS[yypt-3].lval, yyS[yypt-2].lval, &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 17:
		//line a.y:143
		{
			outcode(yyS[yypt-3].lval, yyS[yypt-2].lval, &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 18:
		//line a.y:150
		{
			outcode(yyS[yypt-2].lval, Always, &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 19:
		//line a.y:157
		{
			outcode(yyS[yypt-2].lval, Always, &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 20:
		//line a.y:164
		{
			outcode(yyS[yypt-3].lval, yyS[yypt-2].lval, &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 21:
		//line a.y:171
		{
			outcode(yyS[yypt-5].lval, yyS[yypt-4].lval, &yyS[yypt-3].addr, yyS[yypt-1].lval, &nullgen)
		}
	case 22:
		//line a.y:178
		{
			var g obj.Addr

			g = nullgen
			g.Type = D_CONST
			g.Offset = int64(yyS[yypt-1].lval)
			outcode(yyS[yypt-6].lval, yyS[yypt-5].lval, &yyS[yypt-4].addr, NREG, &g)
		}
	case 23:
		//line a.y:187
		{
			var g obj.Addr

			g = nullgen
			g.Type = D_CONST
			g.Offset = int64(yyS[yypt-3].lval)
			outcode(yyS[yypt-6].lval, yyS[yypt-5].lval, &g, NREG, &yyS[yypt-0].addr)
		}
	case 24:
		//line a.y:199
		{
			outcode(yyS[yypt-6].lval, yyS[yypt-5].lval, &yyS[yypt-2].addr, int32(yyS[yypt-4].addr.Reg), &yyS[yypt-0].addr)
		}
	case 25:
		//line a.y:203
		{
			outcode(yyS[yypt-5].lval, yyS[yypt-4].lval, &yyS[yypt-1].addr, int32(yyS[yypt-3].addr.Reg), &yyS[yypt-3].addr)
		}
	case 26:
		//line a.y:207
		{
			outcode(yyS[yypt-5].lval, yyS[yypt-4].lval, &yyS[yypt-2].addr, int32(yyS[yypt-0].addr.Reg), &yyS[yypt-0].addr)
		}
	case 27:
		//line a.y:214
		{
			outcode(yyS[yypt-2].lval, yyS[yypt-1].lval, &nullgen, NREG, &nullgen)
		}
	case 28:
		//line a.y:221
		{
			asm.Settext(yyS[yypt-2].addr.Sym)
			yyS[yypt-0].addr.Type = D_CONST2
			yyS[yypt-0].addr.Offset2 = -obj.ArgsSizeUnknown
			outcode(yyS[yypt-3].lval, Always, &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 29:
		//line a.y:228
		{
			asm.Settext(yyS[yypt-4].addr.Sym)
			yyS[yypt-0].addr.Type = D_CONST2
			yyS[yypt-0].addr.Offset2 = -obj.ArgsSizeUnknown
			outcode(yyS[yypt-5].lval, Always, &yyS[yypt-4].addr, yyS[yypt-2].lval, &yyS[yypt-0].addr)
		}
	case 30:
		//line a.y:235
		{
			asm.Settext(yyS[yypt-6].addr.Sym)
			yyS[yypt-2].addr.Type = D_CONST2
			yyS[yypt-2].addr.Offset2 = yyS[yypt-0].lval
			outcode(yyS[yypt-7].lval, Always, &yyS[yypt-6].addr, yyS[yypt-4].lval, &yyS[yypt-2].addr)
		}
	case 31:
		//line a.y:245
		{
			outcode(yyS[yypt-5].lval, Always, &yyS[yypt-4].addr, yyS[yypt-2].lval, &yyS[yypt-0].addr)
		}
	case 32:
		//line a.y:252
		{
			outcode(yyS[yypt-3].lval, yyS[yypt-2].lval, &yyS[yypt-1].addr, NREG, &nullgen)
		}
	case 33:
		//line a.y:259
		{
			outcode(yyS[yypt-2].lval, Always, &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 34:
		//line a.y:266
		{
			outcode(yyS[yypt-4].lval, yyS[yypt-3].lval, &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 35:
		//line a.y:270
		{
			outcode(yyS[yypt-4].lval, yyS[yypt-3].lval, &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 36:
		//line a.y:274
		{
			outcode(yyS[yypt-6].lval, yyS[yypt-5].lval, &yyS[yypt-4].addr, yyS[yypt-2].lval, &yyS[yypt-0].addr)
		}
	case 37:
		//line a.y:278
		{
			outcode(yyS[yypt-5].lval, yyS[yypt-4].lval, &yyS[yypt-3].addr, int32(yyS[yypt-1].addr.Reg), &nullgen)
		}
	case 38:
		//line a.y:285
		{
			var g obj.Addr

			g = nullgen
			g.Type = D_CONST
			g.Offset = int64(
				(0xe << 24) | /* opcode */
					(yyS[yypt-11].lval << 20) | /* MCR/MRC */
					(yyS[yypt-10].lval << 28) | /* scond */
					((yyS[yypt-9].lval & 15) << 8) | /* coprocessor number */
					((yyS[yypt-7].lval & 7) << 21) | /* coprocessor operation */
					((yyS[yypt-5].lval & 15) << 12) | /* arm register */
					((yyS[yypt-3].lval & 15) << 16) | /* Crn */
					((yyS[yypt-1].lval & 15) << 0) | /* Crm */
					((yyS[yypt-0].lval & 7) << 5) | /* coprocessor information */
					(1 << 4)) /* must be set */
			outcode(AMRC, Always, &nullgen, NREG, &g)
		}
	case 39:
		//line a.y:297
		{
			outcode(yyS[yypt-6].lval, yyS[yypt-5].lval, &yyS[yypt-4].addr, int32(yyS[yypt-2].addr.Reg), &yyS[yypt-0].addr)
		}
	case 40:
		//line a.y:305
		{
			yyS[yypt-2].addr.Type = D_REGREG2
			yyS[yypt-2].addr.Offset = int64(yyS[yypt-0].lval)
			outcode(yyS[yypt-8].lval, yyS[yypt-7].lval, &yyS[yypt-6].addr, int32(yyS[yypt-4].addr.Reg), &yyS[yypt-2].addr)
		}
	case 41:
		//line a.y:314
		{
			outcode(yyS[yypt-1].lval, Always, &yyS[yypt-0].addr, NREG, &nullgen)
		}
	case 42:
		//line a.y:321
		{
			if yyS[yypt-2].addr.Type != D_CONST || yyS[yypt-0].addr.Type != D_CONST {
				yyerror("arguments to PCDATA must be integer constants")
			}
			outcode(yyS[yypt-3].lval, Always, &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 43:
		//line a.y:331
		{
			if yyS[yypt-2].addr.Type != D_CONST {
				yyerror("index for FUNCDATA must be integer constant")
			}
			if yyS[yypt-0].addr.Type != D_EXTERN && yyS[yypt-0].addr.Type != D_STATIC && yyS[yypt-0].addr.Type != D_OREG {
				yyerror("value for FUNCDATA must be symbol reference")
			}
			outcode(yyS[yypt-3].lval, Always, &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 44:
		//line a.y:344
		{
			outcode(yyS[yypt-1].lval, Always, &nullgen, NREG, &nullgen)
		}
	case 45:
		//line a.y:349
		{
			yyVAL.lval = Always
		}
	case 46:
		//line a.y:353
		{
			yyVAL.lval = (yyS[yypt-1].lval & ^C_SCOND) | yyS[yypt-0].lval
		}
	case 47:
		//line a.y:357
		{
			yyVAL.lval = yyS[yypt-1].lval | yyS[yypt-0].lval
		}
	case 50:
		//line a.y:366
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_BRANCH
			yyVAL.addr.Offset = int64(yyS[yypt-3].lval) + int64(asm.PC)
		}
	case 51:
		//line a.y:372
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyS[yypt-1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyS[yypt-1].sym.Labelname)
			}
			yyVAL.addr.Type = D_BRANCH
			yyVAL.addr.Offset = yyS[yypt-1].sym.Value + int64(yyS[yypt-0].lval)
		}
	case 52:
		//line a.y:383
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_CONST
			yyVAL.addr.Offset = int64(yyS[yypt-0].lval)
		}
	case 53:
		//line a.y:389
		{
			yyVAL.addr = yyS[yypt-0].addr
			yyVAL.addr.Type = D_CONST
		}
	case 54:
		//line a.y:394
		{
			yyVAL.addr = yyS[yypt-0].addr
			yyVAL.addr.Type = D_OCONST
		}
	case 55:
		//line a.y:399
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SCONST
			yyVAL.addr.U.Sval = yyS[yypt-0].sval
		}
	case 56:
		yyVAL.addr = yyS[yypt-0].addr
	case 57:
		//line a.y:408
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FCONST
			yyVAL.addr.U.Dval = yyS[yypt-0].dval
		}
	case 58:
		//line a.y:414
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FCONST
			yyVAL.addr.U.Dval = -yyS[yypt-0].dval
		}
	case 59:
		//line a.y:422
		{
			yyVAL.lval = 1 << uint(yyS[yypt-0].lval)
		}
	case 60:
		//line a.y:426
		{
			yyVAL.lval = 0
			for i := yyS[yypt-2].lval; i <= yyS[yypt-0].lval; i++ {
				yyVAL.lval |= 1 << uint(i)
			}
			for i := yyS[yypt-0].lval; i <= yyS[yypt-2].lval; i++ {
				yyVAL.lval |= 1 << uint(i)
			}
		}
	case 61:
		//line a.y:436
		{
			yyVAL.lval = (1 << uint(yyS[yypt-2].lval)) | yyS[yypt-0].lval
		}
	case 62:
		yyVAL.addr = yyS[yypt-0].addr
	case 63:
		yyVAL.addr = yyS[yypt-0].addr
	case 64:
		yyVAL.addr = yyS[yypt-0].addr
	case 65:
		//line a.y:445
		{
			yyVAL.addr = yyS[yypt-3].addr
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
		}
	case 66:
		//line a.y:450
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_PSR
			yyVAL.addr.Reg = int8(yyS[yypt-0].lval)
		}
	case 67:
		//line a.y:456
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FPCR
			yyVAL.addr.Reg = int8(yyS[yypt-0].lval)
		}
	case 68:
		//line a.y:462
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Offset = int64(yyS[yypt-0].lval)
		}
	case 69:
		yyVAL.addr = yyS[yypt-0].addr
	case 70:
		yyVAL.addr = yyS[yypt-0].addr
	case 71:
		yyVAL.addr = yyS[yypt-0].addr
	case 72:
		//line a.y:473
		{
			yyVAL.addr = yyS[yypt-0].addr
			if yyS[yypt-0].addr.Name != D_EXTERN && yyS[yypt-0].addr.Name != D_STATIC {
			}
		}
	case 73:
		//line a.y:481
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
			yyVAL.addr.Offset = 0
		}
	case 74:
		yyVAL.addr = yyS[yypt-0].addr
	case 75:
		//line a.y:491
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
			yyVAL.addr.Offset = int64(yyS[yypt-3].lval)
		}
	case 76:
		yyVAL.addr = yyS[yypt-0].addr
	case 77:
		//line a.y:501
		{
			yyVAL.addr = yyS[yypt-3].addr
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
		}
	case 78:
		yyVAL.addr = yyS[yypt-0].addr
	case 79:
		yyVAL.addr = yyS[yypt-0].addr
	case 80:
		yyVAL.addr = yyS[yypt-0].addr
	case 81:
		yyVAL.addr = yyS[yypt-0].addr
	case 82:
		//line a.y:514
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_CONST
			yyVAL.addr.Offset = int64(yyS[yypt-0].lval)
		}
	case 83:
		//line a.y:522
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_REG
			yyVAL.addr.Reg = int8(yyS[yypt-0].lval)
		}
	case 84:
		//line a.y:530
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_REGREG
			yyVAL.addr.Reg = int8(yyS[yypt-3].lval)
			yyVAL.addr.Offset = int64(yyS[yypt-1].lval)
		}
	case 85:
		//line a.y:539
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SHIFT
			yyVAL.addr.Offset = int64(yyS[yypt-3].lval) | int64(yyS[yypt-0].lval) | (0 << 5)
		}
	case 86:
		//line a.y:545
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SHIFT
			yyVAL.addr.Offset = int64(yyS[yypt-3].lval) | int64(yyS[yypt-0].lval) | (1 << 5)
		}
	case 87:
		//line a.y:551
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SHIFT
			yyVAL.addr.Offset = int64(yyS[yypt-3].lval) | int64(yyS[yypt-0].lval) | (2 << 5)
		}
	case 88:
		//line a.y:557
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SHIFT
			yyVAL.addr.Offset = int64(yyS[yypt-3].lval) | int64(yyS[yypt-0].lval) | (3 << 5)
		}
	case 89:
		//line a.y:565
		{
			if yyVAL.lval < 0 || yyVAL.lval >= 16 {
				print("register value out of range\n")
			}
			yyVAL.lval = ((yyS[yypt-0].lval & 15) << 8) | (1 << 4)
		}
	case 90:
		//line a.y:572
		{
			if yyVAL.lval < 0 || yyVAL.lval >= 32 {
				print("shift value out of range\n")
			}
			yyVAL.lval = (yyS[yypt-0].lval & 31) << 7
		}
	case 91:
		yyVAL.lval = yyS[yypt-0].lval
	case 92:
		//line a.y:582
		{
			yyVAL.lval = REGPC
		}
	case 93:
		//line a.y:586
		{
			if yyS[yypt-1].lval < 0 || yyS[yypt-1].lval >= NREG {
				print("register value out of range\n")
			}
			yyVAL.lval = yyS[yypt-1].lval
		}
	case 94:
		yyVAL.lval = yyS[yypt-0].lval
	case 95:
		//line a.y:596
		{
			yyVAL.lval = REGSP
		}
	case 96:
		yyVAL.lval = yyS[yypt-0].lval
	case 97:
		//line a.y:603
		{
			if yyS[yypt-1].lval < 0 || yyS[yypt-1].lval >= NREG {
				print("register value out of range\n")
			}
			yyVAL.lval = yyS[yypt-1].lval
		}
	case 98:
		yyVAL.addr = yyS[yypt-0].addr
	case 99:
		yyVAL.addr = yyS[yypt-0].addr
	case 100:
		//line a.y:616
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FREG
			yyVAL.addr.Reg = int8(yyS[yypt-0].lval)
		}
	case 101:
		//line a.y:622
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FREG
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
		}
	case 102:
		//line a.y:630
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Name = int8(yyS[yypt-1].lval)
			yyVAL.addr.Sym = nil
			yyVAL.addr.Offset = int64(yyS[yypt-3].lval)
		}
	case 103:
		//line a.y:638
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Name = int8(yyS[yypt-1].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-4].sym.Name, 0)
			yyVAL.addr.Offset = int64(yyS[yypt-3].lval)
		}
	case 104:
		//line a.y:646
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Name = D_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-6].sym.Name, 1)
			yyVAL.addr.Offset = int64(yyS[yypt-3].lval)
		}
	case 105:
		//line a.y:655
		{
			yyVAL.lval = 0
		}
	case 106:
		//line a.y:659
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 107:
		//line a.y:663
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 108:
		yyVAL.lval = yyS[yypt-0].lval
	case 109:
		yyVAL.lval = yyS[yypt-0].lval
	case 110:
		yyVAL.lval = yyS[yypt-0].lval
	case 111:
		yyVAL.lval = yyS[yypt-0].lval
	case 112:
		//line a.y:675
		{
			yyVAL.lval = int32(yyS[yypt-0].sym.Value)
		}
	case 113:
		//line a.y:679
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 114:
		//line a.y:683
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 115:
		//line a.y:687
		{
			yyVAL.lval = ^yyS[yypt-0].lval
		}
	case 116:
		//line a.y:691
		{
			yyVAL.lval = yyS[yypt-1].lval
		}
	case 117:
		//line a.y:696
		{
			yyVAL.lval = 0
		}
	case 118:
		//line a.y:700
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 119:
		yyVAL.lval = yyS[yypt-0].lval
	case 120:
		//line a.y:707
		{
			yyVAL.lval = yyS[yypt-2].lval + yyS[yypt-0].lval
		}
	case 121:
		//line a.y:711
		{
			yyVAL.lval = yyS[yypt-2].lval - yyS[yypt-0].lval
		}
	case 122:
		//line a.y:715
		{
			yyVAL.lval = yyS[yypt-2].lval * yyS[yypt-0].lval
		}
	case 123:
		//line a.y:719
		{
			yyVAL.lval = yyS[yypt-2].lval / yyS[yypt-0].lval
		}
	case 124:
		//line a.y:723
		{
			yyVAL.lval = yyS[yypt-2].lval % yyS[yypt-0].lval
		}
	case 125:
		//line a.y:727
		{
			yyVAL.lval = yyS[yypt-3].lval << uint(yyS[yypt-0].lval)
		}
	case 126:
		//line a.y:731
		{
			yyVAL.lval = yyS[yypt-3].lval >> uint(yyS[yypt-0].lval)
		}
	case 127:
		//line a.y:735
		{
			yyVAL.lval = yyS[yypt-2].lval & yyS[yypt-0].lval
		}
	case 128:
		//line a.y:739
		{
			yyVAL.lval = yyS[yypt-2].lval ^ yyS[yypt-0].lval
		}
	case 129:
		//line a.y:743
		{
			yyVAL.lval = yyS[yypt-2].lval | yyS[yypt-0].lval
		}
	}
	goto yystack /* stack new state and value */
}
