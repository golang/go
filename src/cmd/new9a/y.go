//line a.y:31
package main

import __yyfmt__ "fmt"

//line a.y:31
import (
	"cmd/internal/asm"
	"cmd/internal/obj"
	. "cmd/internal/obj/ppc64"
)

//line a.y:40
type yySymType struct {
	yys  int
	sym  *asm.Sym
	lval int64
	dval float64
	sval string
	addr obj.Addr
}

const LMOVW = 57346
const LMOVB = 57347
const LABS = 57348
const LLOGW = 57349
const LSHW = 57350
const LADDW = 57351
const LCMP = 57352
const LCROP = 57353
const LBRA = 57354
const LFMOV = 57355
const LFCONV = 57356
const LFCMP = 57357
const LFADD = 57358
const LFMA = 57359
const LTRAP = 57360
const LXORW = 57361
const LNOP = 57362
const LEND = 57363
const LRETT = 57364
const LWORD = 57365
const LTEXT = 57366
const LDATA = 57367
const LRETRN = 57368
const LCONST = 57369
const LSP = 57370
const LSB = 57371
const LFP = 57372
const LPC = 57373
const LCREG = 57374
const LFLUSH = 57375
const LREG = 57376
const LFREG = 57377
const LR = 57378
const LCR = 57379
const LF = 57380
const LFPSCR = 57381
const LLR = 57382
const LCTR = 57383
const LSPR = 57384
const LSPREG = 57385
const LSEG = 57386
const LMSR = 57387
const LPCDAT = 57388
const LFUNCDAT = 57389
const LSCHED = 57390
const LXLD = 57391
const LXST = 57392
const LXOP = 57393
const LXMV = 57394
const LRLWM = 57395
const LMOVMW = 57396
const LMOVEM = 57397
const LMOVFL = 57398
const LMTFSB = 57399
const LMA = 57400
const LFCONST = 57401
const LSCONST = 57402
const LNAME = 57403
const LLAB = 57404
const LVAR = 57405

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
	"LMOVW",
	"LMOVB",
	"LABS",
	"LLOGW",
	"LSHW",
	"LADDW",
	"LCMP",
	"LCROP",
	"LBRA",
	"LFMOV",
	"LFCONV",
	"LFCMP",
	"LFADD",
	"LFMA",
	"LTRAP",
	"LXORW",
	"LNOP",
	"LEND",
	"LRETT",
	"LWORD",
	"LTEXT",
	"LDATA",
	"LRETRN",
	"LCONST",
	"LSP",
	"LSB",
	"LFP",
	"LPC",
	"LCREG",
	"LFLUSH",
	"LREG",
	"LFREG",
	"LR",
	"LCR",
	"LF",
	"LFPSCR",
	"LLR",
	"LCTR",
	"LSPR",
	"LSPREG",
	"LSEG",
	"LMSR",
	"LPCDAT",
	"LFUNCDAT",
	"LSCHED",
	"LXLD",
	"LXST",
	"LXOP",
	"LXMV",
	"LRLWM",
	"LMOVMW",
	"LMOVEM",
	"LMOVFL",
	"LMTFSB",
	"LMA",
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
	-2, 0,
}

const yyNprod = 183
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 885

var yyAct = []int{

	46, 52, 88, 421, 100, 431, 103, 169, 2, 92,
	64, 83, 56, 276, 93, 95, 96, 98, 99, 50,
	54, 111, 49, 271, 55, 444, 119, 121, 123, 443,
	126, 128, 432, 131, 89, 136, 72, 411, 73, 78,
	77, 125, 125, 92, 115, 116, 117, 118, 410, 72,
	61, 73, 72, 62, 73, 132, 400, 51, 161, 399,
	78, 442, 381, 380, 202, 440, 377, 75, 366, 53,
	91, 94, 92, 97, 365, 364, 78, 77, 112, 363,
	81, 82, 133, 275, 120, 92, 125, 146, 75, 107,
	134, 135, 137, 138, 72, 361, 73, 59, 59, 59,
	145, 147, 360, 76, 75, 314, 101, 108, 102, 65,
	401, 79, 359, 196, 110, 59, 282, 203, 195, 202,
	183, 164, 74, 159, 76, 141, 141, 114, 45, 102,
	92, 48, 79, 229, 222, 201, 202, 166, 109, 168,
	76, 167, 85, 87, 106, 105, 162, 439, 79, 244,
	252, 253, 165, 231, 261, 263, 223, 267, 268, 269,
	124, 250, 127, 129, 438, 173, 174, 175, 437, 251,
	436, 256, 435, 249, 392, 258, 265, 286, 289, 290,
	186, 391, 390, 389, 388, 387, 386, 385, 301, 303,
	305, 307, 309, 310, 199, 384, 383, 382, 376, 312,
	375, 374, 291, 292, 293, 294, 247, 316, 319, 257,
	373, 315, 330, 332, 333, 334, 372, 336, 248, 340,
	371, 370, 259, 369, 358, 264, 266, 357, 228, 227,
	326, 327, 328, 329, 226, 114, 59, 219, 218, 59,
	217, 216, 215, 214, 213, 300, 212, 211, 210, 209,
	278, 163, 208, 207, 279, 280, 281, 206, 204, 284,
	285, 47, 84, 86, 59, 200, 194, 193, 192, 331,
	59, 104, 191, 298, 337, 339, 78, 77, 190, 122,
	246, 189, 313, 255, 342, 188, 344, 113, 199, 322,
	187, 368, 347, 348, 349, 350, 351, 185, 182, 354,
	355, 356, 181, 59, 75, 57, 367, 180, 288, 179,
	178, 72, 177, 73, 296, 59, 345, 176, 346, 158,
	130, 157, 156, 155, 139, 154, 153, 143, 152, 78,
	77, 378, 151, 150, 379, 149, 148, 44, 74, 43,
	76, 42, 40, 41, 297, 60, 63, 396, 79, 338,
	72, 341, 73, 61, 184, 262, 62, 75, 197, 65,
	433, 78, 77, 230, 110, 61, 160, 72, 62, 73,
	402, 403, 404, 405, 406, 407, 408, 441, 397, 353,
	65, 409, 395, 61, 283, 110, 62, 412, 352, 75,
	425, 74, 424, 76, 429, 430, 171, 172, 60, 205,
	245, 79, 7, 254, 144, 415, 416, 1, 78, 77,
	69, 393, 394, 183, 72, 61, 73, 260, 62, 220,
	221, 297, 71, 224, 225, 76, 70, 434, 287, 0,
	102, 160, 0, 79, 295, 58, 75, 446, 447, 0,
	449, 450, 78, 77, 0, 420, 423, 398, 0, 427,
	428, 0, 317, 320, 417, 418, 419, 65, 445, 72,
	0, 73, 110, 0, 101, 270, 0, 335, 199, 0,
	75, 0, 76, 140, 142, 422, 422, 102, 61, 343,
	79, 62, 0, 242, 241, 240, 238, 239, 233, 234,
	235, 236, 237, 299, 302, 304, 306, 308, 0, 311,
	0, 273, 272, 274, 74, 0, 76, 72, 270, 73,
	324, 60, 325, 90, 79, 273, 272, 274, 170, 165,
	171, 172, 426, 65, 0, 0, 448, 0, 110, 451,
	173, 8, 0, 68, 67, 0, 80, 233, 234, 235,
	236, 237, 0, 9, 10, 16, 14, 15, 13, 25,
	18, 19, 11, 21, 24, 22, 23, 20, 277, 32,
	36, 0, 33, 37, 38, 39, 0, 78, 77, 0,
	0, 78, 77, 273, 272, 274, 323, 0, 0, 72,
	0, 73, 362, 0, 0, 34, 35, 5, 28, 29,
	31, 30, 26, 27, 0, 75, 12, 17, 0, 75,
	3, 0, 4, 0, 65, 6, 72, 61, 73, 66,
	62, 63, 81, 82, 68, 67, 0, 80, 78, 77,
	241, 240, 238, 239, 233, 234, 235, 236, 237, 78,
	77, 76, 413, 74, 0, 76, 102, 0, 92, 79,
	60, 0, 64, 79, 78, 77, 75, 0, 78, 77,
	0, 65, 0, 72, 0, 73, 66, 75, 0, 81,
	82, 68, 67, 0, 80, 0, 0, 78, 77, 63,
	0, 0, 75, 235, 236, 237, 75, 0, 0, 72,
	74, 73, 76, 78, 77, 0, 0, 60, 0, 92,
	79, 74, 0, 76, 0, 75, 78, 77, 60, 0,
	92, 79, 72, 0, 73, 0, 74, 0, 76, 164,
	74, 75, 76, 102, 78, 77, 79, 102, 78, 77,
	79, 0, 0, 0, 75, 0, 0, 0, 0, 78,
	77, 76, 0, 0, 0, 0, 102, 0, 0, 79,
	0, 0, 75, 0, 0, 109, 75, 76, 0, 0,
	0, 0, 414, 0, 0, 79, 0, 75, 74, 0,
	76, 0, 0, 0, 0, 102, 0, 0, 79, 238,
	239, 233, 234, 235, 236, 237, 109, 0, 76, 0,
	109, 0, 76, 321, 0, 0, 79, 318, 0, 0,
	79, 109, 0, 76, 0, 0, 0, 0, 198, 0,
	0, 79, 242, 241, 240, 238, 239, 233, 234, 235,
	236, 237, 242, 241, 240, 238, 239, 233, 234, 235,
	236, 237, 240, 238, 239, 233, 234, 235, 236, 237,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 243, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 232,
}
var yyPact = []int{

	-1000, 529, -1000, 268, 266, 263, -1000, 261, 52, 562,
	267, 433, -71, -8, 323, -8, 323, 323, 399, 67,
	50, 308, 308, 308, 308, 323, -8, 635, -36, 323,
	8, -36, 5, -70, -71, -71, 158, 687, 687, 158,
	-1000, 399, 399, -1000, -1000, -1000, 259, 258, 256, 255,
	251, 249, 248, 246, 245, 244, 242, -1000, -1000, 45,
	658, -1000, 68, -1000, 639, -1000, 59, -1000, 63, -1000,
	-1000, -1000, -1000, 61, 511, -1000, -1000, 399, 399, 399,
	-1000, -1000, -1000, 240, 235, 233, 232, 230, 225, 221,
	344, 220, 399, 213, 208, 204, 201, 195, 191, 190,
	189, -1000, 399, -1000, -1000, 30, 720, 188, 58, 511,
	59, 181, 180, -1000, -1000, 176, 175, 172, 171, 170,
	169, 167, 166, 165, 164, 323, 163, 161, 160, -1000,
	-1000, 158, 158, 370, -1000, 158, 158, 157, 152, -1000,
	151, 55, 351, -1000, 529, 808, -1000, 798, 609, 323,
	323, 620, 338, 306, 323, 481, 415, 323, 323, 463,
	4, 479, 399, -1000, -1000, 45, 399, 399, 399, 38,
	376, 399, 399, -1000, -1000, -1000, 267, 323, 323, 308,
	308, 308, 320, -1000, 275, 399, -1000, -8, 323, 323,
	323, 323, 323, 323, 399, 26, -1000, -1000, 30, 41,
	709, 705, 535, 38, 323, -1000, 323, 308, 308, 308,
	308, -8, 323, 323, 323, 687, -8, -37, 323, -36,
	-1000, -1000, -1000, -1000, -1000, -1000, -71, 687, 558, 477,
	399, -1000, -1000, 399, 399, 399, 399, 399, 381, 371,
	399, 399, 399, -1000, -1000, -1000, -1000, 150, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, 147, -1000, -1000,
	-1000, -1000, 34, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	23, 16, -1000, -1000, -1000, -1000, 323, -1000, 0, -4,
	-5, -11, 477, 387, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, 146,
	144, -1000, 143, -1000, 139, -1000, 133, -1000, 124, -1000,
	-1000, 123, -1000, 121, -1000, -13, -1000, -1000, 30, -1000,
	-1000, 30, -14, -17, -1000, -1000, -1000, 120, 119, 118,
	110, 109, 108, 107, -1000, -1000, -1000, 106, -1000, 105,
	-1000, -1000, -1000, -1000, -1000, 104, 97, 662, 662, -1000,
	-1000, -1000, 399, 399, 762, 816, 615, 300, 297, 399,
	-1000, -1000, -20, -1000, -1000, -1000, -1000, -23, 32, 323,
	323, 323, 323, 323, 323, 323, 399, -1000, -31, -42,
	674, -1000, 308, 308, 317, 317, 317, 558, 558, 323,
	-36, -71, -75, 528, 528, -1000, -1000, -1000, -47, -1000,
	-1000, 321, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, -1000, -1000, -1000, 30, -1000, 95, -1000, -1000, -1000,
	93, 91, 87, 70, -12, -1000, -1000, 367, -1000, -1000,
	-1000, 51, -1000, -50, -54, 308, 323, 323, 399, 323,
	323, 399, 352, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, -1000,
}
var yyPgo = []int{

	0, 87, 58, 23, 7, 305, 251, 0, 131, 435,
	69, 22, 12, 426, 422, 57, 1, 2, 6, 20,
	24, 4, 19, 417, 410, 3, 407, 8, 404, 402,
	287,
}
var yyR1 = []int{

	0, 26, 26, 28, 27, 27, 27, 27, 27, 27,
	27, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 18, 18, 7, 12, 12, 13,
	20, 14, 24, 19, 19, 19, 22, 23, 11, 11,
	10, 10, 21, 25, 16, 16, 17, 17, 15, 5,
	5, 8, 8, 6, 6, 9, 9, 9, 30, 30,
	4, 4, 4, 3, 3, 3, 1, 1, 1, 1,
	1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
	2, 2, 2,
}
var yyR2 = []int{

	0, 0, 2, 0, 4, 4, 4, 2, 1, 2,
	2, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	4, 6, 4, 4, 4, 6, 4, 4, 6, 6,
	6, 4, 4, 6, 4, 6, 4, 6, 4, 4,
	2, 6, 4, 4, 4, 6, 4, 4, 4, 4,
	4, 4, 4, 4, 2, 2, 4, 3, 3, 5,
	4, 4, 6, 4, 4, 6, 6, 6, 8, 4,
	4, 3, 2, 4, 4, 6, 8, 4, 6, 4,
	4, 6, 6, 8, 8, 8, 8, 4, 4, 4,
	6, 4, 6, 4, 4, 2, 2, 3, 3, 3,
	3, 2, 3, 3, 4, 4, 2, 4, 6, 8,
	6, 6, 6, 2, 4, 2, 1, 1, 1, 1,
	1, 1, 1, 1, 4, 1, 1, 4, 1, 4,
	1, 4, 1, 3, 2, 2, 2, 3, 2, 1,
	4, 3, 5, 1, 4, 4, 5, 7, 0, 1,
	0, 2, 2, 1, 1, 1, 1, 1, 2, 2,
	2, 3, 1, 3, 3, 3, 3, 3, 4, 4,
	3, 3, 3,
}
var yyChk = []int{

	-1000, -26, -27, 71, 73, 58, 76, -29, 2, 14,
	15, 23, 67, 19, 17, 18, 16, 68, 21, 22,
	28, 24, 26, 27, 25, 20, 63, 64, 59, 60,
	62, 61, 30, 33, 56, 57, 31, 34, 35, 36,
	74, 75, 75, 76, 76, 76, -7, -6, -8, -11,
	-22, -15, -16, -10, -19, -20, -12, -5, -9, -1,
	78, 45, 48, 49, 80, 42, 47, 53, 52, -24,
	-13, -14, 44, 46, 71, 37, 73, 10, 9, 81,
	55, 50, 51, -7, -6, -8, -6, -8, -17, -11,
	80, -15, 80, -7, -15, -7, -7, -15, -7, -7,
	-21, -1, 78, -18, -6, 78, 77, -10, -1, 71,
	47, -7, -15, -30, 77, -11, -11, -11, -11, -7,
	-15, -7, -6, -7, -8, 78, -7, -8, -7, -8,
	-30, -7, -11, 77, -15, -15, -16, -15, -15, -30,
	-9, -1, -9, -30, -28, -2, -1, -2, 77, 77,
	77, 77, 77, 77, 77, 77, 77, 77, 77, 78,
	-5, -2, 78, -6, 70, -1, 78, 78, 78, -4,
	7, 9, 10, -1, -1, -1, 77, 77, 77, 77,
	77, 77, 77, 69, 10, 77, -1, 77, 77, 77,
	77, 77, 77, 77, 77, -12, -18, -6, 78, -1,
	77, 77, 78, -4, 77, -30, 77, 77, 77, 77,
	77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
	-30, -30, -7, -11, -30, -30, 77, 77, 77, 78,
	12, -27, 76, 9, 10, 11, 12, 13, 7, 8,
	6, 5, 4, 76, -7, -6, -8, -15, -10, -20,
	-12, -19, -7, -7, -6, -8, -22, -15, -11, -10,
	-23, -7, 49, -7, -10, -19, -10, -7, -7, -7,
	-5, -3, 39, 38, 40, 79, 9, 79, -1, -1,
	-1, -1, 78, 8, -1, -1, -7, -6, -8, -7,
	-7, -11, -11, -11, -11, -6, -8, 69, -1, -5,
	-15, -7, -5, -7, -5, -7, -5, -7, -5, -7,
	-7, -5, -21, -1, 79, -12, -18, -6, 78, -18,
	-6, 78, -1, 41, -5, -5, -11, -11, -11, -11,
	-7, -15, -7, -7, -7, -6, -7, -15, -8, -15,
	-7, -8, -15, -6, -15, -1, -1, -2, -2, -2,
	-2, -2, 7, 8, -2, -2, -2, 77, 77, 78,
	79, 79, -5, 79, 79, 79, 79, -3, -4, 77,
	77, 77, 77, 77, 77, 77, 77, 79, -12, -12,
	77, 79, 77, 77, 77, 77, 77, 77, 77, 77,
	77, 77, 77, -2, -2, -20, 47, -22, -1, 79,
	79, 78, -7, -7, -7, -7, -7, -7, -7, -21,
	79, 79, -18, -6, 78, -11, -11, -10, -10, -10,
	-15, -25, -1, -15, -25, -7, -8, -15, -15, -16,
	-17, 80, 79, 39, -12, 77, 77, 77, 77, 77,
	77, 10, 10, 79, 79, -11, -7, -7, -1, -7,
	-7, -1,
}
var yyDef = []int{

	1, -2, 2, 0, 0, 0, 8, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	158, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 158, 0, 0, 0, 158, 0, 0, 158,
	3, 0, 0, 7, 9, 10, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 126, 153, 0,
	0, 138, 0, 136, 0, 140, 130, 133, 0, 135,
	127, 128, 149, 0, 160, 166, 167, 0, 0, 0,
	132, 129, 131, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 50, 0,
	0, 142, 0, 64, 65, 0, 0, 0, 0, 160,
	0, 158, 0, 82, 159, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 105,
	106, 158, 158, 159, 111, 158, 158, 0, 0, 116,
	0, 0, 0, 123, 0, 0, 172, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 144, 145, 148, 0, 0, 0, 0,
	0, 0, 0, 168, 169, 170, 0, 0, 0, 0,
	0, 0, 0, 146, 0, 0, 148, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 67, 68, 0, 0,
	0, 0, 0, 125, 159, 81, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	107, 108, 109, 110, 112, 113, 0, 0, 0, 0,
	0, 4, 5, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 6, 11, 23, 24, 0, 36, 37,
	61, 63, 12, 13, 27, 28, 30, 0, 29, 32,
	33, 52, 0, 53, 56, 62, 57, 59, 58, 60,
	0, 0, 163, 164, 165, 151, 0, 171, 0, 0,
	0, 0, 0, 160, 161, 162, 14, 25, 26, 15,
	16, 17, 18, 19, 20, 21, 22, 147, 34, 126,
	0, 41, 126, 42, 126, 44, 126, 46, 126, 48,
	49, 0, 54, 142, 66, 0, 70, 71, 0, 73,
	74, 0, 0, 0, 79, 80, 83, 84, 0, 87,
	89, 90, 0, 0, 97, 98, 99, 0, 101, 0,
	103, 104, 114, 115, 117, 0, 0, 173, 174, 175,
	176, 177, 0, 0, 180, 181, 182, 0, 0, 0,
	154, 155, 0, 139, 141, 134, 150, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 69, 0, 0,
	0, 124, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 178, 179, 35, 130, 31, 0, 152,
	156, 0, 38, 40, 39, 43, 45, 47, 51, 55,
	72, 75, 76, 77, 0, 85, 0, 88, 91, 92,
	0, 0, 0, 0, 0, 100, 102, 118, 120, 121,
	122, 0, 137, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 157, 78, 86, 93, 94, 143, 95,
	96, 119,
}
var yyTok1 = []int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 80, 13, 6, 3,
	78, 79, 11, 9, 77, 10, 3, 12, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 74, 76,
	7, 75, 8, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 5, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 4, 3, 81,
}
var yyTok2 = []int{

	2, 3, 14, 15, 16, 17, 18, 19, 20, 21,
	22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
	42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
	52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
	62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
	72, 73,
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

	case 3:
		//line a.y:75
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			if yyS[yypt-1].sym.Type == LLAB && yyS[yypt-1].sym.Value != int64(asm.PC) {
				yyerror("redeclaration of %s", yyS[yypt-1].sym.Labelname)
			}
			yyS[yypt-1].sym.Type = LLAB
			yyS[yypt-1].sym.Value = int64(asm.PC)
		}
	case 5:
		//line a.y:85
		{
			yyS[yypt-3].sym.Type = LVAR
			yyS[yypt-3].sym.Value = yyS[yypt-1].lval
		}
	case 6:
		//line a.y:90
		{
			if yyS[yypt-3].sym.Value != yyS[yypt-1].lval {
				yyerror("redeclaration of %s", yyS[yypt-3].sym.Name)
			}
			yyS[yypt-3].sym.Value = yyS[yypt-1].lval
		}
	case 7:
		//line a.y:97
		{
			nosched = int(yyS[yypt-1].lval)
		}
	case 11:
		//line a.y:109
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 12:
		//line a.y:113
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 13:
		//line a.y:117
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 14:
		//line a.y:121
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 15:
		//line a.y:125
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 16:
		//line a.y:129
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 17:
		//line a.y:136
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 18:
		//line a.y:140
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 19:
		//line a.y:144
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 20:
		//line a.y:148
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 21:
		//line a.y:152
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 22:
		//line a.y:156
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 23:
		//line a.y:163
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 24:
		//line a.y:167
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 25:
		//line a.y:171
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 26:
		//line a.y:175
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 27:
		//line a.y:182
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 28:
		//line a.y:186
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 29:
		//line a.y:193
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 30:
		//line a.y:197
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 31:
		//line a.y:201
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, NREG, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 32:
		//line a.y:205
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 33:
		//line a.y:209
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 34:
		//line a.y:213
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, int(yyS[yypt-0].lval), &nullgen)
		}
	case 35:
		//line a.y:220
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, NREG, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 36:
		//line a.y:224
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 37:
		//line a.y:228
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 38:
		//line a.y:238
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 39:
		//line a.y:242
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 40:
		//line a.y:246
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, NREG, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 41:
		//line a.y:250
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 42:
		//line a.y:254
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 43:
		//line a.y:258
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 44:
		//line a.y:262
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 45:
		//line a.y:266
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 46:
		//line a.y:270
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 47:
		//line a.y:274
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 48:
		//line a.y:278
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 49:
		//line a.y:282
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 50:
		//line a.y:286
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr, NREG, &yyS[yypt-0].addr)
		}
	case 51:
		//line a.y:293
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 52:
		//line a.y:300
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 53:
		//line a.y:304
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 54:
		//line a.y:311
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, int(yyS[yypt-0].addr.Reg), &yyS[yypt-0].addr)
		}
	case 55:
		//line a.y:315
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 56:
		//line a.y:323
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 57:
		//line a.y:327
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 58:
		//line a.y:331
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 59:
		//line a.y:335
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 60:
		//line a.y:339
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 61:
		//line a.y:343
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 62:
		//line a.y:347
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 63:
		//line a.y:351
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 64:
		//line a.y:360
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 65:
		//line a.y:364
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 66:
		//line a.y:368
		{
			outcode(int(yyS[yypt-3].lval), &nullgen, NREG, &yyS[yypt-1].addr)
		}
	case 67:
		//line a.y:372
		{
			outcode(int(yyS[yypt-2].lval), &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 68:
		//line a.y:376
		{
			outcode(int(yyS[yypt-2].lval), &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 69:
		//line a.y:380
		{
			outcode(int(yyS[yypt-4].lval), &nullgen, NREG, &yyS[yypt-1].addr)
		}
	case 70:
		//line a.y:384
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 71:
		//line a.y:388
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 72:
		//line a.y:392
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, NREG, &yyS[yypt-1].addr)
		}
	case 73:
		//line a.y:396
		{
			outcode(int(yyS[yypt-3].lval), &nullgen, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 74:
		//line a.y:400
		{
			outcode(int(yyS[yypt-3].lval), &nullgen, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 75:
		//line a.y:404
		{
			outcode(int(yyS[yypt-5].lval), &nullgen, int(yyS[yypt-4].lval), &yyS[yypt-1].addr)
		}
	case 76:
		//line a.y:408
		{
			var g obj.Addr
			g = nullgen
			g.Type = D_CONST
			g.Offset = yyS[yypt-4].lval
			outcode(int(yyS[yypt-5].lval), &g, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 77:
		//line a.y:416
		{
			var g obj.Addr
			g = nullgen
			g.Type = D_CONST
			g.Offset = yyS[yypt-4].lval
			outcode(int(yyS[yypt-5].lval), &g, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 78:
		//line a.y:424
		{
			var g obj.Addr
			g = nullgen
			g.Type = D_CONST
			g.Offset = yyS[yypt-6].lval
			outcode(int(yyS[yypt-7].lval), &g, int(yyS[yypt-4].lval), &yyS[yypt-1].addr)
		}
	case 79:
		//line a.y:435
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, int(yyS[yypt-0].lval), &nullgen)
		}
	case 80:
		//line a.y:439
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, int(yyS[yypt-0].lval), &nullgen)
		}
	case 81:
		//line a.y:443
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, NREG, &nullgen)
		}
	case 82:
		//line a.y:447
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, NREG, &nullgen)
		}
	case 83:
		//line a.y:454
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 84:
		//line a.y:458
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 85:
		//line a.y:462
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].addr.Reg), &yyS[yypt-0].addr)
		}
	case 86:
		//line a.y:466
		{
			outgcode(int(yyS[yypt-7].lval), &yyS[yypt-6].addr, int(yyS[yypt-4].addr.Reg), &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 87:
		//line a.y:470
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 88:
		//line a.y:474
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-0].addr.Reg), &yyS[yypt-2].addr)
		}
	case 89:
		//line a.y:481
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 90:
		//line a.y:485
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 91:
		//line a.y:489
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-0].addr.Reg), &yyS[yypt-2].addr)
		}
	case 92:
		//line a.y:493
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-0].addr.Reg), &yyS[yypt-2].addr)
		}
	case 93:
		//line a.y:500
		{
			outgcode(int(yyS[yypt-7].lval), &yyS[yypt-6].addr, int(yyS[yypt-4].addr.Reg), &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 94:
		//line a.y:504
		{
			outgcode(int(yyS[yypt-7].lval), &yyS[yypt-6].addr, int(yyS[yypt-4].addr.Reg), &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 95:
		//line a.y:508
		{
			outgcode(int(yyS[yypt-7].lval), &yyS[yypt-6].addr, int(yyS[yypt-4].addr.Reg), &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 96:
		//line a.y:512
		{
			outgcode(int(yyS[yypt-7].lval), &yyS[yypt-6].addr, int(yyS[yypt-4].addr.Reg), &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 97:
		//line a.y:519
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 98:
		//line a.y:523
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 99:
		//line a.y:531
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 100:
		//line a.y:535
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, NREG, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 101:
		//line a.y:539
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 102:
		//line a.y:543
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, NREG, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 103:
		//line a.y:547
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 104:
		//line a.y:551
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 105:
		//line a.y:555
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr, NREG, &nullgen)
		}
	case 106:
		//line a.y:562
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, NREG, &nullgen)
		}
	case 107:
		//line a.y:566
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, NREG, &nullgen)
		}
	case 108:
		//line a.y:570
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, NREG, &nullgen)
		}
	case 109:
		//line a.y:574
		{
			outcode(int(yyS[yypt-2].lval), &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 110:
		//line a.y:578
		{
			outcode(int(yyS[yypt-2].lval), &nullgen, NREG, &yyS[yypt-0].addr)
		}
	case 111:
		//line a.y:582
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr, NREG, &nullgen)
		}
	case 112:
		//line a.y:589
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, NREG, &nullgen)
		}
	case 113:
		//line a.y:593
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, NREG, &nullgen)
		}
	case 114:
		//line a.y:600
		{
			if yyS[yypt-2].addr.Type != D_CONST || yyS[yypt-0].addr.Type != D_CONST {
				yyerror("arguments to PCDATA must be integer constants")
			}
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 115:
		//line a.y:610
		{
			if yyS[yypt-2].addr.Type != D_CONST {
				yyerror("index for FUNCDATA must be integer constant")
			}
			if yyS[yypt-0].addr.Type != D_EXTERN && yyS[yypt-0].addr.Type != D_STATIC && yyS[yypt-0].addr.Type != D_OREG {
				yyerror("value for FUNCDATA must be symbol reference")
			}
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 116:
		//line a.y:623
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, NREG, &nullgen)
		}
	case 117:
		//line a.y:630
		{
			asm.Settext(yyS[yypt-2].addr.Sym)
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, NREG, &yyS[yypt-0].addr)
		}
	case 118:
		//line a.y:635
		{
			asm.Settext(yyS[yypt-4].addr.Sym)
			yyS[yypt-0].addr.Offset &= 0xffffffff
			yyS[yypt-0].addr.Offset |= -obj.ArgsSizeUnknown << 32
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 119:
		//line a.y:642
		{
			asm.Settext(yyS[yypt-6].addr.Sym)
			yyS[yypt-2].addr.Offset &= 0xffffffff
			yyS[yypt-2].addr.Offset |= (yyS[yypt-0].lval & 0xffffffff) << 32
			outcode(int(yyS[yypt-7].lval), &yyS[yypt-6].addr, int(yyS[yypt-4].lval), &yyS[yypt-2].addr)
		}
	case 120:
		//line a.y:652
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 121:
		//line a.y:656
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 122:
		//line a.y:660
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 123:
		//line a.y:667
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, NREG, &nullgen)
		}
	case 124:
		//line a.y:673
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_BRANCH
			yyVAL.addr.Offset = yyS[yypt-3].lval + int64(asm.PC)
		}
	case 125:
		//line a.y:679
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyS[yypt-1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyS[yypt-1].sym.Labelname)
			}
			yyVAL.addr.Type = D_BRANCH
			yyVAL.addr.Offset = yyS[yypt-1].sym.Value + yyS[yypt-0].lval
		}
	case 126:
		//line a.y:691
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_REG
			yyVAL.addr.Reg = int8(yyS[yypt-0].lval)
		}
	case 127:
		yyVAL.addr = yyS[yypt-0].addr
	case 128:
		yyVAL.addr = yyS[yypt-0].addr
	case 129:
		//line a.y:703
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SPR
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 130:
		//line a.y:711
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_CREG
			yyVAL.addr.Reg = NREG /* whole register */
		}
	case 131:
		//line a.y:718
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SPR
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 132:
		//line a.y:726
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_MSR
		}
	case 133:
		//line a.y:733
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SPR
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 134:
		//line a.y:739
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = int16(yyS[yypt-3].lval)
			yyVAL.addr.Offset = yyS[yypt-1].lval
		}
	case 135:
		yyVAL.addr = yyS[yypt-0].addr
	case 136:
		//line a.y:748
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FPSCR
			yyVAL.addr.Reg = NREG
		}
	case 137:
		//line a.y:756
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FPSCR
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
		}
	case 138:
		//line a.y:764
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FREG
			yyVAL.addr.Reg = int8(yyS[yypt-0].lval)
		}
	case 139:
		//line a.y:770
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FREG
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
		}
	case 140:
		//line a.y:778
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_CREG
			yyVAL.addr.Reg = int8(yyS[yypt-0].lval)
		}
	case 141:
		//line a.y:784
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_CREG
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
		}
	case 142:
		//line a.y:792
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_REG
			yyVAL.addr.Reg = int8(yyS[yypt-0].lval)
		}
	case 143:
		//line a.y:800
		{
			var mb, me int
			var v uint32

			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_CONST
			mb = int(yyS[yypt-2].lval)
			me = int(yyS[yypt-0].lval)
			if mb < 0 || mb > 31 || me < 0 || me > 31 {
				yyerror("illegal mask start/end value(s)")
				mb = 0
				me = 0
			}
			if mb <= me {
				v = (^uint32(0) >> uint(mb)) & (^uint32(0) << uint(31-me))
			} else {
				v = (^uint32(0) >> uint(me+1)) & (^uint32(0) << uint(31-(mb-1)))
			}
			yyVAL.addr.Offset = int64(v)
		}
	case 144:
		//line a.y:823
		{
			yyVAL.addr = yyS[yypt-0].addr
			yyVAL.addr.Type = D_CONST
		}
	case 145:
		//line a.y:828
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_SCONST
			yyVAL.addr.U.Sval = yyS[yypt-0].sval
		}
	case 146:
		//line a.y:836
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FCONST
			yyVAL.addr.U.Dval = yyS[yypt-0].dval
		}
	case 147:
		//line a.y:842
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_FCONST
			yyVAL.addr.U.Dval = -yyS[yypt-0].dval
		}
	case 148:
		//line a.y:849
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_CONST
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 149:
		yyVAL.lval = yyS[yypt-0].lval
	case 150:
		//line a.y:858
		{
			if yyVAL.lval < 0 || yyVAL.lval >= NREG {
				print("register value out of range\n")
			}
			yyVAL.lval = yyS[yypt-1].lval
		}
	case 151:
		//line a.y:867
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
			yyVAL.addr.Offset = 0
		}
	case 152:
		//line a.y:874
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Reg = int8(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			yyVAL.addr.Offset = 0
		}
	case 153:
		yyVAL.addr = yyS[yypt-0].addr
	case 154:
		//line a.y:885
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Reg = int8(yyS[yypt-1].lval)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 155:
		//line a.y:894
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Name = int8(yyS[yypt-1].lval)
			yyVAL.addr.Sym = nil
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 156:
		//line a.y:902
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Name = int8(yyS[yypt-1].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-4].sym.Name, 0)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 157:
		//line a.y:910
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = D_OREG
			yyVAL.addr.Name = D_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-6].sym.Name, 0)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 160:
		//line a.y:922
		{
			yyVAL.lval = 0
		}
	case 161:
		//line a.y:926
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 162:
		//line a.y:930
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 163:
		yyVAL.lval = yyS[yypt-0].lval
	case 164:
		yyVAL.lval = yyS[yypt-0].lval
	case 165:
		yyVAL.lval = yyS[yypt-0].lval
	case 166:
		yyVAL.lval = yyS[yypt-0].lval
	case 167:
		//line a.y:942
		{
			yyVAL.lval = yyS[yypt-0].sym.Value
		}
	case 168:
		//line a.y:946
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 169:
		//line a.y:950
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 170:
		//line a.y:954
		{
			yyVAL.lval = ^yyS[yypt-0].lval
		}
	case 171:
		//line a.y:958
		{
			yyVAL.lval = yyS[yypt-1].lval
		}
	case 172:
		yyVAL.lval = yyS[yypt-0].lval
	case 173:
		//line a.y:965
		{
			yyVAL.lval = yyS[yypt-2].lval + yyS[yypt-0].lval
		}
	case 174:
		//line a.y:969
		{
			yyVAL.lval = yyS[yypt-2].lval - yyS[yypt-0].lval
		}
	case 175:
		//line a.y:973
		{
			yyVAL.lval = yyS[yypt-2].lval * yyS[yypt-0].lval
		}
	case 176:
		//line a.y:977
		{
			yyVAL.lval = yyS[yypt-2].lval / yyS[yypt-0].lval
		}
	case 177:
		//line a.y:981
		{
			yyVAL.lval = yyS[yypt-2].lval % yyS[yypt-0].lval
		}
	case 178:
		//line a.y:985
		{
			yyVAL.lval = yyS[yypt-3].lval << uint(yyS[yypt-0].lval)
		}
	case 179:
		//line a.y:989
		{
			yyVAL.lval = yyS[yypt-3].lval >> uint(yyS[yypt-0].lval)
		}
	case 180:
		//line a.y:993
		{
			yyVAL.lval = yyS[yypt-2].lval & yyS[yypt-0].lval
		}
	case 181:
		//line a.y:997
		{
			yyVAL.lval = yyS[yypt-2].lval ^ yyS[yypt-0].lval
		}
	case 182:
		//line a.y:1001
		{
			yyVAL.lval = yyS[yypt-2].lval | yyS[yypt-0].lval
		}
	}
	goto yystack /* stack new state and value */
}
