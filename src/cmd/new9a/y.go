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
const LGLOBL = 57368
const LRETRN = 57369
const LCONST = 57370
const LSP = 57371
const LSB = 57372
const LFP = 57373
const LPC = 57374
const LCREG = 57375
const LFLUSH = 57376
const LREG = 57377
const LFREG = 57378
const LR = 57379
const LCR = 57380
const LF = 57381
const LFPSCR = 57382
const LLR = 57383
const LCTR = 57384
const LSPR = 57385
const LSPREG = 57386
const LSEG = 57387
const LMSR = 57388
const LPCDAT = 57389
const LFUNCDAT = 57390
const LSCHED = 57391
const LXLD = 57392
const LXST = 57393
const LXOP = 57394
const LXMV = 57395
const LRLWM = 57396
const LMOVMW = 57397
const LMOVEM = 57398
const LMOVFL = 57399
const LMTFSB = 57400
const LMA = 57401
const LFCONST = 57402
const LSCONST = 57403
const LNAME = 57404
const LLAB = 57405
const LVAR = 57406

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
	"LGLOBL",
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

const yyNprod = 186
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 922

var yyAct = []int{

	47, 393, 53, 89, 426, 272, 439, 57, 51, 101,
	93, 84, 171, 55, 94, 96, 97, 99, 100, 56,
	50, 112, 73, 434, 74, 2, 120, 122, 124, 65,
	127, 129, 90, 132, 104, 52, 137, 126, 277, 93,
	453, 452, 116, 117, 118, 119, 416, 415, 92, 95,
	114, 98, 73, 133, 74, 115, 113, 73, 93, 74,
	405, 404, 121, 383, 382, 204, 447, 379, 135, 136,
	138, 139, 245, 244, 243, 241, 242, 236, 237, 238,
	239, 240, 368, 131, 367, 366, 126, 140, 148, 365,
	49, 145, 126, 93, 363, 163, 362, 315, 60, 60,
	60, 86, 88, 203, 204, 406, 283, 102, 109, 276,
	204, 73, 62, 74, 197, 63, 60, 161, 231, 125,
	165, 128, 130, 205, 168, 170, 142, 142, 142, 54,
	48, 85, 87, 169, 164, 224, 446, 445, 147, 149,
	105, 444, 198, 443, 134, 246, 46, 93, 123, 108,
	442, 247, 255, 256, 167, 225, 263, 264, 253, 268,
	269, 270, 259, 207, 254, 398, 397, 175, 176, 177,
	252, 266, 234, 396, 392, 261, 391, 390, 389, 287,
	290, 291, 188, 222, 223, 388, 250, 226, 227, 260,
	302, 304, 306, 308, 310, 311, 201, 387, 386, 385,
	384, 378, 292, 293, 294, 295, 313, 377, 316, 376,
	375, 374, 373, 372, 331, 333, 334, 335, 371, 337,
	361, 341, 360, 232, 230, 301, 229, 228, 199, 115,
	327, 328, 329, 330, 221, 220, 219, 317, 320, 60,
	218, 249, 60, 217, 258, 216, 215, 214, 213, 332,
	212, 211, 210, 279, 338, 340, 209, 280, 281, 282,
	208, 206, 285, 286, 343, 202, 196, 60, 347, 289,
	195, 248, 194, 60, 257, 297, 299, 193, 192, 191,
	251, 190, 79, 451, 262, 314, 189, 265, 267, 369,
	187, 201, 323, 184, 183, 58, 182, 370, 181, 288,
	180, 179, 178, 160, 159, 296, 60, 158, 157, 156,
	339, 76, 342, 155, 154, 153, 152, 151, 60, 346,
	150, 348, 349, 318, 321, 45, 44, 380, 41, 42,
	381, 43, 350, 351, 352, 353, 354, 186, 336, 357,
	358, 359, 298, 185, 166, 75, 62, 77, 64, 63,
	344, 66, 103, 402, 440, 80, 111, 162, 79, 78,
	66, 69, 68, 62, 81, 111, 63, 73, 62, 74,
	403, 63, 407, 408, 409, 410, 411, 412, 413, 66,
	401, 73, 460, 74, 111, 79, 78, 76, 414, 79,
	78, 395, 430, 66, 73, 429, 74, 185, 111, 448,
	433, 437, 438, 79, 78, 420, 421, 274, 273, 275,
	324, 233, 449, 73, 76, 74, 432, 417, 76, 394,
	356, 75, 162, 77, 284, 425, 428, 441, 61, 173,
	174, 80, 76, 435, 436, 355, 450, 66, 73, 7,
	74, 146, 111, 1, 455, 456, 298, 458, 459, 70,
	77, 399, 400, 72, 77, 103, 71, 271, 80, 103,
	0, 93, 80, 454, 0, 0, 110, 102, 77, 0,
	0, 201, 107, 106, 0, 0, 80, 0, 427, 427,
	274, 273, 275, 431, 59, 300, 303, 305, 307, 309,
	0, 312, 0, 274, 273, 275, 238, 239, 240, 73,
	271, 74, 325, 418, 326, 245, 244, 243, 241, 242,
	236, 237, 238, 239, 240, 0, 422, 423, 424, 0,
	0, 0, 141, 143, 144, 0, 0, 0, 167, 8,
	172, 0, 173, 174, 457, 236, 237, 238, 239, 240,
	175, 9, 10, 16, 14, 15, 13, 25, 18, 19,
	11, 21, 24, 22, 23, 20, 0, 32, 36, 0,
	33, 37, 39, 38, 40, 79, 78, 0, 0, 79,
	78, 0, 0, 364, 241, 242, 236, 237, 238, 239,
	240, 278, 0, 0, 34, 35, 5, 28, 29, 31,
	30, 26, 27, 0, 76, 12, 17, 0, 76, 3,
	0, 4, 0, 66, 6, 73, 62, 74, 67, 63,
	64, 82, 83, 69, 68, 0, 81, 79, 78, 244,
	243, 241, 242, 236, 237, 238, 239, 240, 79, 78,
	77, 0, 75, 0, 77, 103, 0, 345, 80, 61,
	0, 65, 80, 0, 79, 78, 76, 0, 0, 0,
	0, 66, 0, 73, 0, 74, 67, 76, 0, 82,
	83, 69, 68, 0, 81, 62, 79, 78, 63, 0,
	79, 78, 0, 76, 0, 0, 79, 78, 0, 0,
	75, 62, 77, 0, 63, 0, 0, 61, 0, 93,
	80, 75, 0, 77, 0, 76, 79, 78, 61, 76,
	91, 80, 73, 0, 74, 76, 0, 75, 0, 77,
	0, 64, 0, 0, 61, 79, 78, 80, 82, 83,
	0, 79, 78, 0, 0, 76, 0, 79, 78, 75,
	0, 77, 0, 75, 0, 77, 103, 0, 0, 80,
	61, 77, 93, 80, 76, 0, 103, 79, 78, 80,
	76, 73, 0, 74, 0, 0, 76, 0, 166, 75,
	0, 77, 0, 0, 79, 78, 103, 0, 0, 80,
	79, 78, 0, 0, 79, 78, 76, 0, 0, 0,
	77, 0, 0, 0, 110, 103, 77, 0, 80, 0,
	75, 419, 77, 76, 80, 0, 0, 103, 0, 76,
	80, 0, 0, 76, 0, 0, 0, 0, 0, 0,
	110, 0, 77, 0, 0, 0, 0, 322, 0, 0,
	80, 0, 0, 0, 0, 0, 0, 110, 0, 77,
	0, 0, 0, 110, 319, 77, 0, 80, 0, 77,
	200, 0, 0, 80, 103, 0, 0, 80, 245, 244,
	243, 241, 242, 236, 237, 238, 239, 240, 243, 241,
	242, 236, 237, 238, 239, 240, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 235,
}
var yyPact = []int{

	-1000, 527, -1000, 253, 255, 249, -1000, 248, 69, 560,
	349, 619, -71, 12, 393, 12, 393, 393, 765, 394,
	-23, 300, 300, 300, 300, 393, 12, 657, 13, 393,
	7, 13, 66, -52, -71, -71, 151, 718, 718, 718,
	151, -1000, 765, 765, -1000, -1000, -1000, 242, 239, 238,
	237, 236, 235, 231, 230, 229, 226, 225, -1000, -1000,
	38, 706, -1000, 55, -1000, 687, -1000, 45, -1000, 54,
	-1000, -1000, -1000, -1000, 46, 523, -1000, -1000, 765, 765,
	765, -1000, -1000, -1000, 224, 223, 222, 220, 218, 216,
	215, 327, 212, 765, 208, 203, 201, 200, 199, 194,
	192, 188, -1000, 765, -1000, -1000, 667, 761, 187, 25,
	523, 45, 183, 182, -1000, -1000, 178, 174, 173, 172,
	170, 169, 168, 167, 165, 162, 393, 158, 157, 156,
	-1000, -1000, 151, 151, 322, -1000, 151, 151, 149, 148,
	-1000, 146, 39, 145, 399, -1000, 527, 844, -1000, 68,
	608, 393, 393, 661, 317, 393, 393, 308, 336, 393,
	393, 454, 29, 501, 765, -1000, -1000, 38, 765, 765,
	765, 27, 416, 765, 765, -1000, -1000, -1000, 349, 393,
	393, 300, 300, 300, 635, -1000, 272, 765, -1000, 12,
	393, 393, 393, 393, 393, 393, 765, 17, -1000, -1000,
	667, 31, 755, 738, 368, 27, 393, -1000, 393, 300,
	300, 300, 300, 12, 393, 393, 393, 718, 12, -42,
	393, 13, -1000, -1000, -1000, -1000, -1000, -1000, -71, 718,
	556, 441, 380, 765, -1000, -1000, 765, 765, 765, 765,
	765, 428, 412, 765, 765, 765, -1000, -1000, -1000, -1000,
	144, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	142, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, 16, 14, -1000, -1000, -1000, -1000, 393, -1000, 9,
	5, 4, 2, 441, 420, -1000, -1000, -1000, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	140, 135, -1000, 134, -1000, 133, -1000, 132, -1000, 131,
	-1000, -1000, 129, -1000, 123, -1000, -13, -1000, -1000, 667,
	-1000, -1000, 667, -14, -17, -1000, -1000, -1000, 122, 121,
	120, 119, 107, 100, 99, -1000, -1000, -1000, 98, -1000,
	96, -1000, -1000, -1000, -1000, 381, 95, -1000, 88, 87,
	485, 485, -1000, -1000, -1000, 765, 765, 567, 852, 614,
	305, 298, -1000, -1000, -19, -1000, -1000, -1000, -1000, -20,
	26, 393, 393, 393, 393, 393, 393, 393, 765, -1000,
	-33, -34, 712, -1000, 300, 300, 350, 350, 350, 380,
	380, 393, 13, -1000, 406, 362, -58, -71, -75, 526,
	526, -1000, -1000, -1000, -1000, -1000, 314, -1000, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, 667,
	-1000, 72, -1000, -1000, -1000, 65, 63, 59, 58, -12,
	-1000, -1000, 361, 402, 381, -1000, -1000, -1000, -1000, 273,
	-39, -40, 300, 393, 393, 765, 393, 393, -1000, 344,
	-1000, 376, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000,
}
var yyPgo = []int{

	0, 88, 95, 5, 12, 295, 120, 0, 90, 484,
	129, 20, 7, 456, 453, 1, 35, 2, 3, 34,
	13, 19, 9, 8, 449, 4, 443, 25, 441, 439,
	50,
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
	29, 29, 29, 29, 19, 19, 7, 12, 12, 13,
	21, 14, 24, 20, 20, 20, 23, 11, 11, 10,
	10, 22, 25, 15, 15, 15, 15, 17, 17, 18,
	18, 16, 5, 5, 8, 8, 6, 6, 9, 9,
	9, 30, 30, 4, 4, 4, 3, 3, 3, 1,
	1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2,
}
var yyR2 = []int{

	0, 0, 2, 0, 4, 4, 4, 2, 1, 2,
	2, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	4, 6, 4, 4, 6, 4, 4, 6, 6, 6,
	4, 4, 6, 4, 6, 4, 6, 4, 4, 2,
	6, 4, 4, 4, 6, 4, 4, 4, 4, 4,
	4, 4, 4, 2, 2, 4, 3, 3, 5, 4,
	4, 6, 4, 4, 6, 6, 6, 8, 4, 4,
	3, 2, 4, 4, 6, 8, 4, 6, 4, 4,
	6, 6, 8, 8, 8, 8, 4, 4, 4, 6,
	4, 6, 4, 4, 2, 2, 3, 3, 3, 3,
	2, 3, 3, 4, 4, 2, 5, 7, 4, 6,
	6, 6, 6, 2, 4, 2, 1, 1, 1, 1,
	1, 1, 1, 1, 4, 1, 1, 1, 4, 1,
	4, 1, 3, 1, 2, 3, 4, 2, 2, 2,
	3, 2, 1, 4, 3, 5, 1, 4, 4, 5,
	7, 0, 1, 0, 2, 2, 1, 1, 1, 1,
	1, 2, 2, 2, 3, 1, 3, 3, 3, 3,
	3, 4, 4, 3, 3, 3,
}
var yyChk = []int{

	-1000, -26, -27, 72, 74, 59, 77, -29, 2, 14,
	15, 23, 68, 19, 17, 18, 16, 69, 21, 22,
	28, 24, 26, 27, 25, 20, 64, 65, 60, 61,
	63, 62, 30, 33, 57, 58, 31, 34, 36, 35,
	37, 75, 76, 76, 77, 77, 77, -7, -6, -8,
	-11, -23, -16, -17, -10, -20, -21, -12, -5, -9,
	-1, 79, 46, 49, 50, 81, 43, 48, 54, 53,
	-24, -13, -14, 45, 47, 72, 38, 74, 10, 9,
	82, 56, 51, 52, -7, -6, -8, -6, -8, -18,
	-11, 81, -16, 81, -7, -16, -7, -7, -16, -7,
	-7, -22, -1, 79, -19, -6, 79, 78, -10, -1,
	72, 48, -7, -16, -30, 78, -11, -11, -11, -11,
	-7, -16, -7, -6, -7, -8, 79, -7, -8, -7,
	-8, -30, -7, -11, 78, -16, -16, -17, -16, -16,
	-30, -9, -1, -9, -9, -30, -28, -2, -1, -2,
	78, 78, 78, 78, 78, 78, 78, 78, 78, 78,
	78, 79, -5, -2, 79, -6, 71, -1, 79, 79,
	79, -4, 7, 9, 10, -1, -1, -1, 78, 78,
	78, 78, 78, 78, 78, 70, 10, 78, -1, 78,
	78, 78, 78, 78, 78, 78, 78, -12, -19, -6,
	79, -1, 78, 78, 79, -4, 78, -30, 78, 78,
	78, 78, 78, 78, 78, 78, 78, 78, 78, 78,
	78, 78, -30, -30, -7, -11, -30, -30, 78, 78,
	78, 79, 78, 12, -27, 77, 9, 10, 11, 12,
	13, 7, 8, 6, 5, 4, 77, -7, -6, -8,
	-16, -10, -21, -12, -20, -7, -7, -6, -8, -23,
	-16, -11, -10, -7, -7, -10, -20, -10, -7, -7,
	-7, -5, -3, 40, 39, 41, 80, 9, 80, -1,
	-1, -1, -1, 79, 8, -1, -1, -7, -6, -8,
	-7, -7, -11, -11, -11, -11, -6, -8, 70, -1,
	-5, -16, -7, -5, -7, -5, -7, -5, -7, -5,
	-7, -7, -5, -22, -1, 80, -12, -19, -6, 79,
	-19, -6, 79, -1, 42, -5, -5, -11, -11, -11,
	-11, -7, -16, -7, -7, -7, -6, -7, -16, -8,
	-16, -7, -8, -16, -6, 81, -1, -16, -1, -1,
	-2, -2, -2, -2, -2, 7, 8, -2, -2, -2,
	78, 78, 80, 80, -5, 80, 80, 80, 80, -3,
	-4, 78, 78, 78, 78, 78, 78, 78, 78, 80,
	-12, -12, 78, 80, 78, 78, 78, 78, 78, 78,
	78, 78, 78, -15, 38, 10, 78, 78, 78, -2,
	-2, -21, 48, -23, 80, 80, 79, -7, -7, -7,
	-7, -7, -7, -7, -22, 80, 80, -19, -6, 79,
	-11, -11, -10, -10, -10, -16, -25, -1, -16, -25,
	-7, -8, 10, 38, 81, -16, -16, -17, -18, 81,
	40, -12, 78, 78, 78, 78, 78, 78, 38, 10,
	-15, 10, 80, 80, -11, -7, -7, -1, -7, -7,
	38,
}
var yyDef = []int{

	1, -2, 2, 0, 0, 0, 8, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	161, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 161, 0, 0, 0, 161, 0, 0, 0,
	161, 3, 0, 0, 7, 9, 10, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 126, 156,
	0, 0, 137, 0, 136, 0, 139, 130, 133, 0,
	135, 127, 128, 152, 0, 163, 169, 170, 0, 0,
	0, 132, 129, 131, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 49,
	0, 0, 141, 0, 63, 64, 0, 0, 0, 0,
	163, 0, 161, 0, 81, 162, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	104, 105, 161, 161, 162, 110, 161, 161, 0, 0,
	115, 0, 0, 0, 0, 123, 0, 0, 175, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 147, 148, 151, 0, 0,
	0, 0, 0, 0, 0, 171, 172, 173, 0, 0,
	0, 0, 0, 0, 0, 149, 0, 0, 151, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 66, 67,
	0, 0, 0, 0, 0, 125, 162, 80, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 106, 107, 108, 109, 111, 112, 0, 0,
	0, 0, 0, 0, 4, 5, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 6, 11, 23, 24,
	0, 35, 36, 60, 62, 12, 13, 27, 28, 30,
	0, 29, 32, 51, 52, 55, 61, 56, 58, 57,
	59, 0, 0, 166, 167, 168, 154, 0, 174, 0,
	0, 0, 0, 0, 163, 164, 165, 14, 25, 26,
	15, 16, 17, 18, 19, 20, 21, 22, 150, 33,
	126, 0, 40, 126, 41, 126, 43, 126, 45, 126,
	47, 48, 0, 53, 141, 65, 0, 69, 70, 0,
	72, 73, 0, 0, 0, 78, 79, 82, 83, 0,
	86, 88, 89, 0, 0, 96, 97, 98, 0, 100,
	0, 102, 103, 113, 114, 0, 0, 118, 0, 0,
	176, 177, 178, 179, 180, 0, 0, 183, 184, 185,
	0, 0, 157, 158, 0, 138, 140, 134, 153, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 68,
	0, 0, 0, 124, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 116, 143, 0, 0, 0, 0, 181,
	182, 34, 130, 31, 155, 159, 0, 37, 39, 38,
	42, 44, 46, 50, 54, 71, 74, 75, 76, 0,
	84, 0, 87, 90, 91, 0, 0, 0, 0, 0,
	99, 101, 0, 144, 0, 119, 120, 121, 122, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 145, 0,
	117, 0, 160, 77, 85, 92, 93, 142, 94, 95,
	146,
}
var yyTok1 = []int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 81, 13, 6, 3,
	79, 80, 11, 9, 78, 10, 3, 12, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 75, 77,
	7, 76, 8, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 5, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 4, 3, 82,
}
var yyTok2 = []int{

	2, 3, 14, 15, 16, 17, 18, 19, 20, 21,
	22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
	42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
	52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
	62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
	72, 73, 74,
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
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 12:
		//line a.y:113
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 13:
		//line a.y:117
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 14:
		//line a.y:121
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 15:
		//line a.y:125
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 16:
		//line a.y:129
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 17:
		//line a.y:136
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 18:
		//line a.y:140
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 19:
		//line a.y:144
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 20:
		//line a.y:148
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 21:
		//line a.y:152
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 22:
		//line a.y:156
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 23:
		//line a.y:163
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 24:
		//line a.y:167
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 25:
		//line a.y:171
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 26:
		//line a.y:175
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 27:
		//line a.y:182
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 28:
		//line a.y:186
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 29:
		//line a.y:193
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 30:
		//line a.y:197
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 31:
		//line a.y:201
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 32:
		//line a.y:205
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 33:
		//line a.y:209
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, int(yyS[yypt-0].lval), &nullgen)
		}
	case 34:
		//line a.y:216
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 35:
		//line a.y:220
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 36:
		//line a.y:224
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 37:
		//line a.y:234
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 38:
		//line a.y:238
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 39:
		//line a.y:242
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 40:
		//line a.y:246
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 41:
		//line a.y:250
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 42:
		//line a.y:254
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 43:
		//line a.y:258
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 44:
		//line a.y:262
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 45:
		//line a.y:266
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 46:
		//line a.y:270
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 47:
		//line a.y:274
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 48:
		//line a.y:278
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 49:
		//line a.y:282
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr, 0, &yyS[yypt-0].addr)
		}
	case 50:
		//line a.y:289
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 51:
		//line a.y:296
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 52:
		//line a.y:300
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 53:
		//line a.y:307
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, int(yyS[yypt-0].addr.Reg), &yyS[yypt-0].addr)
		}
	case 54:
		//line a.y:311
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 55:
		//line a.y:319
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 56:
		//line a.y:323
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 57:
		//line a.y:327
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 58:
		//line a.y:331
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 59:
		//line a.y:335
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 60:
		//line a.y:339
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 61:
		//line a.y:343
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 62:
		//line a.y:347
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 63:
		//line a.y:356
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, 0, &yyS[yypt-0].addr)
		}
	case 64:
		//line a.y:360
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, 0, &yyS[yypt-0].addr)
		}
	case 65:
		//line a.y:364
		{
			outcode(int(yyS[yypt-3].lval), &nullgen, 0, &yyS[yypt-1].addr)
		}
	case 66:
		//line a.y:368
		{
			outcode(int(yyS[yypt-2].lval), &nullgen, 0, &yyS[yypt-0].addr)
		}
	case 67:
		//line a.y:372
		{
			outcode(int(yyS[yypt-2].lval), &nullgen, 0, &yyS[yypt-0].addr)
		}
	case 68:
		//line a.y:376
		{
			outcode(int(yyS[yypt-4].lval), &nullgen, 0, &yyS[yypt-1].addr)
		}
	case 69:
		//line a.y:380
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 70:
		//line a.y:384
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 71:
		//line a.y:388
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-1].addr)
		}
	case 72:
		//line a.y:392
		{
			outcode(int(yyS[yypt-3].lval), &nullgen, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 73:
		//line a.y:396
		{
			outcode(int(yyS[yypt-3].lval), &nullgen, int(yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 74:
		//line a.y:400
		{
			outcode(int(yyS[yypt-5].lval), &nullgen, int(yyS[yypt-4].lval), &yyS[yypt-1].addr)
		}
	case 75:
		//line a.y:404
		{
			var g obj.Addr
			g = nullgen
			g.Type = obj.TYPE_CONST
			g.Offset = yyS[yypt-4].lval
			outcode(int(yyS[yypt-5].lval), &g, int(REG_R0+yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 76:
		//line a.y:412
		{
			var g obj.Addr
			g = nullgen
			g.Type = obj.TYPE_CONST
			g.Offset = yyS[yypt-4].lval
			outcode(int(yyS[yypt-5].lval), &g, int(REG_R0+yyS[yypt-2].lval), &yyS[yypt-0].addr)
		}
	case 77:
		//line a.y:420
		{
			var g obj.Addr
			g = nullgen
			g.Type = obj.TYPE_CONST
			g.Offset = yyS[yypt-6].lval
			outcode(int(yyS[yypt-7].lval), &g, int(REG_R0+yyS[yypt-4].lval), &yyS[yypt-1].addr)
		}
	case 78:
		//line a.y:431
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, int(yyS[yypt-0].lval), &nullgen)
		}
	case 79:
		//line a.y:435
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, int(yyS[yypt-0].lval), &nullgen)
		}
	case 80:
		//line a.y:439
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, 0, &nullgen)
		}
	case 81:
		//line a.y:443
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, 0, &nullgen)
		}
	case 82:
		//line a.y:450
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 83:
		//line a.y:454
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 84:
		//line a.y:458
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-2].addr.Reg), &yyS[yypt-0].addr)
		}
	case 85:
		//line a.y:462
		{
			outgcode(int(yyS[yypt-7].lval), &yyS[yypt-6].addr, int(yyS[yypt-4].addr.Reg), &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 86:
		//line a.y:466
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 87:
		//line a.y:470
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-0].addr.Reg), &yyS[yypt-2].addr)
		}
	case 88:
		//line a.y:477
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 89:
		//line a.y:481
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 90:
		//line a.y:485
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-0].addr.Reg), &yyS[yypt-2].addr)
		}
	case 91:
		//line a.y:489
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, int(yyS[yypt-0].addr.Reg), &yyS[yypt-2].addr)
		}
	case 92:
		//line a.y:496
		{
			outgcode(int(yyS[yypt-7].lval), &yyS[yypt-6].addr, int(yyS[yypt-4].addr.Reg), &yyS[yypt-2].addr, &yyS[yypt-0].addr)
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
		//line a.y:515
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 97:
		//line a.y:519
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 98:
		//line a.y:527
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 99:
		//line a.y:531
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 100:
		//line a.y:535
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 101:
		//line a.y:539
		{
			outgcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-2].addr, &yyS[yypt-0].addr)
		}
	case 102:
		//line a.y:543
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 103:
		//line a.y:547
		{
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 104:
		//line a.y:551
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr, 0, &nullgen)
		}
	case 105:
		//line a.y:558
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, 0, &nullgen)
		}
	case 106:
		//line a.y:562
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, 0, &nullgen)
		}
	case 107:
		//line a.y:566
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, 0, &nullgen)
		}
	case 108:
		//line a.y:570
		{
			outcode(int(yyS[yypt-2].lval), &nullgen, 0, &yyS[yypt-0].addr)
		}
	case 109:
		//line a.y:574
		{
			outcode(int(yyS[yypt-2].lval), &nullgen, 0, &yyS[yypt-0].addr)
		}
	case 110:
		//line a.y:578
		{
			outcode(int(yyS[yypt-1].lval), &yyS[yypt-0].addr, 0, &nullgen)
		}
	case 111:
		//line a.y:585
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, 0, &nullgen)
		}
	case 112:
		//line a.y:589
		{
			outcode(int(yyS[yypt-2].lval), &yyS[yypt-1].addr, 0, &nullgen)
		}
	case 113:
		//line a.y:596
		{
			if yyS[yypt-2].addr.Type != obj.TYPE_CONST || yyS[yypt-0].addr.Type != obj.TYPE_CONST {
				yyerror("arguments to PCDATA must be integer constants")
			}
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 114:
		//line a.y:606
		{
			if yyS[yypt-2].addr.Type != obj.TYPE_CONST {
				yyerror("index for FUNCDATA must be integer constant")
			}
			if yyS[yypt-0].addr.Type != obj.TYPE_MEM || (yyS[yypt-0].addr.Name != obj.NAME_EXTERN && yyS[yypt-0].addr.Name != obj.NAME_STATIC) {
				yyerror("value for FUNCDATA must be symbol reference")
			}
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 115:
		//line a.y:619
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, 0, &nullgen)
		}
	case 116:
		//line a.y:626
		{
			asm.Settext(yyS[yypt-3].addr.Sym)
			outcode(int(yyS[yypt-4].lval), &yyS[yypt-3].addr, 0, &yyS[yypt-0].addr)
		}
	case 117:
		//line a.y:631
		{
			asm.Settext(yyS[yypt-5].addr.Sym)
			outcode(int(yyS[yypt-6].lval), &yyS[yypt-5].addr, int(yyS[yypt-3].lval), &yyS[yypt-0].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyS[yypt-3].lval
			}
		}
	case 118:
		//line a.y:643
		{
			asm.Settext(yyS[yypt-2].addr.Sym)
			outcode(int(yyS[yypt-3].lval), &yyS[yypt-2].addr, 0, &yyS[yypt-0].addr)
		}
	case 119:
		//line a.y:648
		{
			asm.Settext(yyS[yypt-4].addr.Sym)
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-0].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyS[yypt-2].lval
			}
		}
	case 120:
		//line a.y:661
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-0].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyS[yypt-2].lval
			}
		}
	case 121:
		//line a.y:669
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-0].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyS[yypt-2].lval
			}
		}
	case 122:
		//line a.y:677
		{
			outcode(int(yyS[yypt-5].lval), &yyS[yypt-4].addr, 0, &yyS[yypt-0].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyS[yypt-2].lval
			}
		}
	case 123:
		//line a.y:688
		{
			outcode(int(yyS[yypt-1].lval), &nullgen, 0, &nullgen)
		}
	case 124:
		//line a.y:694
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyS[yypt-3].lval + int64(asm.PC)
		}
	case 125:
		//line a.y:700
		{
			yyS[yypt-1].sym = asm.LabelLookup(yyS[yypt-1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyS[yypt-1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyS[yypt-1].sym.Labelname)
			}
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyS[yypt-1].sym.Value + yyS[yypt-0].lval
		}
	case 126:
		//line a.y:712
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 127:
		yyVAL.addr = yyS[yypt-0].addr
	case 128:
		yyVAL.addr = yyS[yypt-0].addr
	case 129:
		//line a.y:724
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 130:
		//line a.y:732
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval) /* whole register */
		}
	case 131:
		//line a.y:739
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 132:
		//line a.y:747
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 133:
		//line a.y:755
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 134:
		//line a.y:761
		{
			if yyS[yypt-1].lval < 0 || yyS[yypt-1].lval >= 1024 {
				yyerror("SPR/DCR out of range")
			}
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-3].lval + yyS[yypt-1].lval)
		}
	case 135:
		yyVAL.addr = yyS[yypt-0].addr
	case 136:
		//line a.y:773
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 137:
		//line a.y:781
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 138:
		//line a.y:787
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(REG_F0 + yyS[yypt-1].lval)
		}
	case 139:
		//line a.y:795
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 140:
		//line a.y:801
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(REG_C0 + yyS[yypt-1].lval)
		}
	case 141:
		//line a.y:809
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyS[yypt-0].lval)
		}
	case 142:
		//line a.y:817
		{
			var mb, me int
			var v uint32

			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
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
	case 143:
		//line a.y:840
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = int64(yyS[yypt-0].lval)
			yyVAL.addr.U.Argsize = obj.ArgsSizeUnknown
		}
	case 144:
		//line a.y:847
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -int64(yyS[yypt-0].lval)
			yyVAL.addr.U.Argsize = obj.ArgsSizeUnknown
		}
	case 145:
		//line a.y:854
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = int64(yyS[yypt-2].lval)
			yyVAL.addr.U.Argsize = int32(yyS[yypt-0].lval)
		}
	case 146:
		//line a.y:861
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -int64(yyS[yypt-2].lval)
			yyVAL.addr.U.Argsize = int32(yyS[yypt-0].lval)
		}
	case 147:
		//line a.y:870
		{
			yyVAL.addr = yyS[yypt-0].addr
			yyVAL.addr.Type = obj.TYPE_ADDR
		}
	case 148:
		//line a.y:875
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SCONST
			yyVAL.addr.U.Sval = yyS[yypt-0].sval
		}
	case 149:
		//line a.y:883
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = yyS[yypt-0].dval
		}
	case 150:
		//line a.y:889
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = -yyS[yypt-0].dval
		}
	case 151:
		//line a.y:896
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
			yyVAL.addr.Offset = yyS[yypt-0].lval
		}
	case 152:
		yyVAL.lval = yyS[yypt-0].lval
	case 153:
		//line a.y:905
		{
			if yyVAL.lval < 0 || yyVAL.lval >= NREG {
				print("register value out of range\n")
			}
			yyVAL.lval = REG_R0 + yyS[yypt-1].lval
		}
	case 154:
		//line a.y:914
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyS[yypt-1].lval)
			yyVAL.addr.Offset = 0
		}
	case 155:
		//line a.y:921
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyS[yypt-3].lval)
			yyVAL.addr.Scale = int8(yyS[yypt-1].lval)
			yyVAL.addr.Offset = 0
		}
	case 156:
		yyVAL.addr = yyS[yypt-0].addr
	case 157:
		//line a.y:932
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyS[yypt-1].lval)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 158:
		//line a.y:941
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyS[yypt-1].lval)
			yyVAL.addr.Sym = nil
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 159:
		//line a.y:949
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyS[yypt-1].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-4].sym.Name, 0)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 160:
		//line a.y:957
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = obj.NAME_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyS[yypt-6].sym.Name, 0)
			yyVAL.addr.Offset = yyS[yypt-3].lval
		}
	case 163:
		//line a.y:969
		{
			yyVAL.lval = 0
		}
	case 164:
		//line a.y:973
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 165:
		//line a.y:977
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 166:
		yyVAL.lval = yyS[yypt-0].lval
	case 167:
		yyVAL.lval = yyS[yypt-0].lval
	case 168:
		yyVAL.lval = yyS[yypt-0].lval
	case 169:
		yyVAL.lval = yyS[yypt-0].lval
	case 170:
		//line a.y:989
		{
			yyVAL.lval = yyS[yypt-0].sym.Value
		}
	case 171:
		//line a.y:993
		{
			yyVAL.lval = -yyS[yypt-0].lval
		}
	case 172:
		//line a.y:997
		{
			yyVAL.lval = yyS[yypt-0].lval
		}
	case 173:
		//line a.y:1001
		{
			yyVAL.lval = ^yyS[yypt-0].lval
		}
	case 174:
		//line a.y:1005
		{
			yyVAL.lval = yyS[yypt-1].lval
		}
	case 175:
		yyVAL.lval = yyS[yypt-0].lval
	case 176:
		//line a.y:1012
		{
			yyVAL.lval = yyS[yypt-2].lval + yyS[yypt-0].lval
		}
	case 177:
		//line a.y:1016
		{
			yyVAL.lval = yyS[yypt-2].lval - yyS[yypt-0].lval
		}
	case 178:
		//line a.y:1020
		{
			yyVAL.lval = yyS[yypt-2].lval * yyS[yypt-0].lval
		}
	case 179:
		//line a.y:1024
		{
			yyVAL.lval = yyS[yypt-2].lval / yyS[yypt-0].lval
		}
	case 180:
		//line a.y:1028
		{
			yyVAL.lval = yyS[yypt-2].lval % yyS[yypt-0].lval
		}
	case 181:
		//line a.y:1032
		{
			yyVAL.lval = yyS[yypt-3].lval << uint(yyS[yypt-0].lval)
		}
	case 182:
		//line a.y:1036
		{
			yyVAL.lval = yyS[yypt-3].lval >> uint(yyS[yypt-0].lval)
		}
	case 183:
		//line a.y:1040
		{
			yyVAL.lval = yyS[yypt-2].lval & yyS[yypt-0].lval
		}
	case 184:
		//line a.y:1044
		{
			yyVAL.lval = yyS[yypt-2].lval ^ yyS[yypt-0].lval
		}
	case 185:
		//line a.y:1048
		{
			yyVAL.lval = yyS[yypt-2].lval | yyS[yypt-0].lval
		}
	}
	goto yystack /* stack new state and value */
}
