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
	-2, 2,
}

const yyNprod = 187
const yyPrivate = 57344

var yyTokenNames []string
var yyStates []string

const yyLast = 900

var yyAct = []int{

	48, 394, 54, 90, 427, 273, 440, 58, 52, 102,
	80, 79, 85, 172, 94, 95, 97, 98, 100, 101,
	51, 57, 113, 3, 80, 79, 56, 121, 123, 125,
	435, 128, 130, 91, 133, 53, 278, 138, 74, 77,
	75, 66, 164, 117, 118, 119, 120, 454, 453, 93,
	96, 65, 99, 77, 134, 417, 127, 114, 94, 74,
	416, 75, 74, 122, 75, 406, 83, 84, 105, 136,
	137, 139, 140, 76, 94, 78, 80, 79, 405, 384,
	62, 127, 94, 81, 383, 205, 148, 150, 149, 78,
	50, 380, 116, 369, 104, 94, 127, 81, 368, 61,
	61, 61, 87, 89, 367, 77, 366, 277, 103, 110,
	364, 363, 316, 63, 407, 198, 64, 61, 284, 55,
	126, 205, 129, 131, 162, 206, 232, 143, 143, 143,
	169, 74, 63, 75, 171, 64, 225, 204, 205, 76,
	109, 78, 170, 165, 448, 47, 62, 447, 92, 81,
	446, 445, 248, 256, 257, 168, 226, 264, 265, 254,
	269, 270, 271, 260, 135, 444, 443, 94, 176, 177,
	178, 235, 399, 253, 398, 397, 262, 199, 255, 393,
	288, 291, 292, 189, 392, 267, 391, 251, 390, 389,
	261, 303, 305, 307, 309, 311, 312, 202, 388, 387,
	166, 386, 385, 293, 294, 295, 296, 314, 379, 317,
	115, 49, 86, 88, 378, 332, 334, 335, 336, 377,
	338, 106, 342, 376, 375, 374, 302, 373, 372, 124,
	362, 328, 329, 330, 331, 361, 233, 231, 230, 229,
	61, 116, 250, 61, 132, 259, 222, 221, 141, 220,
	333, 219, 146, 218, 280, 339, 341, 217, 281, 282,
	283, 216, 215, 286, 287, 344, 214, 213, 61, 348,
	290, 252, 318, 321, 61, 263, 298, 300, 266, 268,
	351, 352, 353, 354, 355, 212, 315, 358, 359, 360,
	370, 211, 202, 324, 59, 210, 80, 79, 209, 371,
	207, 203, 197, 196, 195, 194, 193, 61, 192, 200,
	191, 340, 190, 343, 188, 185, 184, 80, 79, 61,
	347, 183, 349, 350, 208, 77, 182, 181, 381, 180,
	67, 382, 74, 63, 75, 68, 64, 65, 83, 84,
	70, 69, 179, 82, 223, 224, 77, 161, 227, 228,
	160, 159, 249, 158, 157, 258, 156, 163, 155, 76,
	154, 78, 153, 152, 151, 46, 62, 45, 66, 81,
	44, 404, 187, 408, 409, 410, 411, 412, 413, 414,
	289, 299, 78, 402, 42, 43, 297, 104, 63, 415,
	81, 64, 67, 431, 65, 63, 430, 112, 64, 400,
	401, 403, 438, 439, 319, 322, 421, 422, 246, 245,
	244, 242, 243, 237, 238, 239, 240, 241, 67, 337,
	441, 461, 163, 112, 449, 434, 426, 429, 442, 234,
	450, 345, 186, 433, 436, 437, 357, 451, 74, 63,
	75, 74, 64, 75, 285, 456, 457, 356, 459, 460,
	67, 8, 418, 60, 67, 112, 74, 272, 75, 112,
	70, 69, 396, 82, 455, 275, 274, 276, 103, 174,
	175, 74, 202, 75, 275, 274, 276, 80, 452, 428,
	428, 247, 147, 2, 432, 301, 304, 306, 308, 310,
	395, 313, 142, 144, 145, 275, 274, 276, 325, 9,
	272, 74, 326, 75, 327, 1, 77, 423, 424, 425,
	71, 10, 11, 17, 15, 16, 14, 26, 19, 20,
	12, 22, 25, 23, 24, 21, 73, 33, 37, 168,
	34, 38, 40, 39, 41, 458, 72, 0, 186, 167,
	76, 176, 78, 80, 79, 0, 173, 104, 174, 175,
	81, 239, 240, 241, 35, 36, 6, 29, 30, 32,
	31, 27, 28, 80, 79, 13, 18, 0, 0, 4,
	0, 5, 77, 365, 7, 0, 0, 67, 0, 74,
	0, 75, 68, 0, 419, 83, 84, 70, 69, 0,
	82, 0, 77, 0, 80, 79, 0, 67, 0, 0,
	80, 79, 112, 0, 0, 0, 76, 0, 78, 80,
	79, 0, 0, 62, 0, 94, 81, 237, 238, 239,
	240, 241, 0, 77, 0, 0, 111, 0, 78, 77,
	0, 63, 108, 107, 64, 0, 81, 0, 77, 80,
	79, 0, 0, 0, 0, 74, 0, 75, 245, 244,
	242, 243, 237, 238, 239, 240, 241, 76, 0, 78,
	0, 0, 167, 76, 62, 78, 0, 81, 77, 0,
	104, 0, 76, 81, 78, 74, 0, 75, 0, 62,
	0, 0, 81, 246, 245, 244, 242, 243, 237, 238,
	239, 240, 241, 80, 79, 80, 79, 80, 79, 0,
	0, 0, 76, 0, 78, 0, 80, 79, 0, 104,
	80, 79, 81, 0, 0, 0, 0, 0, 0, 80,
	79, 0, 77, 0, 77, 0, 77, 0, 0, 74,
	0, 75, 80, 79, 0, 77, 0, 0, 0, 77,
	0, 0, 0, 80, 79, 0, 0, 0, 77, 0,
	0, 0, 80, 79, 0, 0, 299, 0, 78, 279,
	78, 77, 78, 104, 0, 104, 81, 104, 81, 94,
	81, 78, 77, 111, 0, 78, 104, 0, 346, 81,
	420, 77, 76, 81, 78, 0, 0, 0, 0, 104,
	0, 0, 81, 0, 0, 111, 0, 78, 0, 0,
	0, 0, 323, 0, 0, 81, 111, 0, 78, 0,
	0, 0, 0, 320, 0, 111, 81, 78, 0, 0,
	0, 0, 201, 0, 0, 81, 246, 245, 244, 242,
	243, 237, 238, 239, 240, 241, 244, 242, 243, 237,
	238, 239, 240, 241, 242, 243, 237, 238, 239, 240,
	241, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 236,
}
var yyPact = []int{

	-1000, -1000, 497, -1000, 309, 294, 290, -1000, 288, 68,
	287, 600, 67, -67, -7, 396, -7, 396, 396, 308,
	554, 14, 342, 342, 342, 342, 396, -7, 630, 2,
	396, 17, 2, 86, -40, -67, -67, 163, 710, 710,
	710, 163, -1000, 308, 308, -1000, -1000, -1000, 286, 285,
	284, 282, 280, 278, 276, 275, 273, 272, 269, -1000,
	-1000, 45, 684, -1000, 64, -1000, 591, -1000, 51, -1000,
	63, -1000, -1000, -1000, -1000, 55, 539, -1000, -1000, 308,
	308, 308, -1000, -1000, -1000, 264, 251, 249, 248, 243,
	238, 237, 362, 236, 308, 234, 232, 230, 228, 227,
	226, 225, 224, -1000, 308, -1000, -1000, 15, 743, 223,
	59, 539, 51, 222, 220, -1000, -1000, 217, 213, 207,
	189, 188, 184, 183, 179, 175, 173, 396, 171, 169,
	168, -1000, -1000, 163, 163, 393, -1000, 163, 163, 161,
	160, -1000, 159, 47, 158, 417, -1000, 497, 822, -1000,
	404, 534, 396, 396, 1, 349, 396, 396, 407, 411,
	396, 396, 426, 27, 679, 308, -1000, -1000, 45, 308,
	308, 308, 39, 436, 308, 308, -1000, -1000, -1000, 600,
	396, 396, 342, 342, 342, 585, -1000, 311, 308, -1000,
	-7, 396, 396, 396, 396, 396, 396, 308, 32, -1000,
	-1000, 15, 42, 734, 723, 456, 39, 396, -1000, 396,
	342, 342, 342, 342, -7, 396, 396, 396, 710, -7,
	-23, 396, 2, -1000, -1000, -1000, -1000, -1000, -1000, -67,
	710, 697, 435, 688, 308, -1000, -1000, 308, 308, 308,
	308, 308, 440, 428, 308, 308, 308, -1000, -1000, -1000,
	-1000, 157, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, 152, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, -1000, 31, 30, -1000, -1000, -1000, -1000, 396, -1000,
	26, 24, 18, 13, 435, 460, -1000, -1000, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, 150, 149, -1000, 147, -1000, 146, -1000, 145, -1000,
	141, -1000, -1000, 136, -1000, 130, -1000, 11, -1000, -1000,
	15, -1000, -1000, 15, 6, -1, -1000, -1000, -1000, 124,
	123, 121, 120, 111, 110, 108, -1000, -1000, -1000, 106,
	-1000, 101, -1000, -1000, -1000, -1000, 452, 97, -1000, 96,
	94, 540, 540, -1000, -1000, -1000, 308, 308, 837, 830,
	643, 353, 344, -1000, -1000, -2, -1000, -1000, -1000, -1000,
	-15, 35, 396, 396, 396, 396, 396, 396, 396, 308,
	-1000, -20, -25, 701, -1000, 342, 342, 375, 375, 375,
	688, 688, 396, 2, -1000, 423, 387, -51, -67, -75,
	608, 608, -1000, -1000, -1000, -1000, -1000, 380, -1000, -1000,
	-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	15, -1000, 88, -1000, -1000, -1000, 87, 73, 72, 69,
	66, -1000, -1000, 386, 420, 452, -1000, -1000, -1000, -1000,
	468, -32, -33, 342, 396, 396, 308, 396, 396, -1000,
	383, -1000, 686, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
	-1000, -1000,
}
var yyPgo = []int{

	0, 88, 42, 5, 13, 294, 200, 0, 90, 453,
	119, 20, 7, 536, 526, 1, 35, 2, 3, 68,
	26, 21, 9, 8, 510, 4, 505, 483, 23, 482,
	451, 210,
}
var yyR1 = []int{

	0, 26, 27, 26, 29, 28, 28, 28, 28, 28,
	28, 28, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
	30, 30, 30, 30, 30, 19, 19, 7, 12, 12,
	13, 21, 14, 24, 20, 20, 20, 23, 11, 11,
	10, 10, 22, 25, 15, 15, 15, 15, 17, 17,
	18, 18, 16, 5, 5, 8, 8, 6, 6, 9,
	9, 9, 31, 31, 4, 4, 4, 3, 3, 3,
	1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2, 2,
}
var yyR2 = []int{

	0, 0, 0, 3, 0, 4, 4, 4, 2, 1,
	2, 2, 4, 4, 4, 4, 4, 4, 4, 4,
	4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	4, 4, 6, 4, 4, 6, 4, 4, 6, 6,
	6, 4, 4, 6, 4, 6, 4, 6, 4, 4,
	2, 6, 4, 4, 4, 6, 4, 4, 4, 4,
	4, 4, 4, 4, 2, 2, 4, 3, 3, 5,
	4, 4, 6, 4, 4, 6, 6, 6, 8, 4,
	4, 3, 2, 4, 4, 6, 8, 4, 6, 4,
	4, 6, 6, 8, 8, 8, 8, 4, 4, 4,
	6, 4, 6, 4, 4, 2, 2, 3, 3, 3,
	3, 2, 3, 3, 4, 4, 2, 5, 7, 4,
	6, 6, 6, 6, 2, 4, 2, 1, 1, 1,
	1, 1, 1, 1, 1, 4, 1, 1, 1, 4,
	1, 4, 1, 3, 1, 2, 3, 4, 2, 2,
	2, 3, 2, 1, 4, 3, 5, 1, 4, 4,
	5, 7, 0, 1, 0, 2, 2, 1, 1, 1,
	1, 1, 2, 2, 2, 3, 1, 3, 3, 3,
	3, 3, 4, 4, 3, 3, 3,
}
var yyChk = []int{

	-1000, -26, -27, -28, 72, 74, 59, 77, -30, 2,
	14, 15, 23, 68, 19, 17, 18, 16, 69, 21,
	22, 28, 24, 26, 27, 25, 20, 64, 65, 60,
	61, 63, 62, 30, 33, 57, 58, 31, 34, 36,
	35, 37, 75, 76, 76, 77, 77, 77, -7, -6,
	-8, -11, -23, -16, -17, -10, -20, -21, -12, -5,
	-9, -1, 79, 46, 49, 50, 81, 43, 48, 54,
	53, -24, -13, -14, 45, 47, 72, 38, 74, 10,
	9, 82, 56, 51, 52, -7, -6, -8, -6, -8,
	-18, -11, 81, -16, 81, -7, -16, -7, -7, -16,
	-7, -7, -22, -1, 79, -19, -6, 79, 78, -10,
	-1, 72, 48, -7, -16, -31, 78, -11, -11, -11,
	-11, -7, -16, -7, -6, -7, -8, 79, -7, -8,
	-7, -8, -31, -7, -11, 78, -16, -16, -17, -16,
	-16, -31, -9, -1, -9, -9, -31, -29, -2, -1,
	-2, 78, 78, 78, 78, 78, 78, 78, 78, 78,
	78, 78, 79, -5, -2, 79, -6, 71, -1, 79,
	79, 79, -4, 7, 9, 10, -1, -1, -1, 78,
	78, 78, 78, 78, 78, 78, 70, 10, 78, -1,
	78, 78, 78, 78, 78, 78, 78, 78, -12, -19,
	-6, 79, -1, 78, 78, 79, -4, 78, -31, 78,
	78, 78, 78, 78, 78, 78, 78, 78, 78, 78,
	78, 78, 78, -31, -31, -7, -11, -31, -31, 78,
	78, 78, 79, 78, 12, -28, 77, 9, 10, 11,
	12, 13, 7, 8, 6, 5, 4, 77, -7, -6,
	-8, -16, -10, -21, -12, -20, -7, -7, -6, -8,
	-23, -16, -11, -10, -7, -7, -10, -20, -10, -7,
	-7, -7, -5, -3, 40, 39, 41, 80, 9, 80,
	-1, -1, -1, -1, 79, 8, -1, -1, -7, -6,
	-8, -7, -7, -11, -11, -11, -11, -6, -8, 70,
	-1, -5, -16, -7, -5, -7, -5, -7, -5, -7,
	-5, -7, -7, -5, -22, -1, 80, -12, -19, -6,
	79, -19, -6, 79, -1, 42, -5, -5, -11, -11,
	-11, -11, -7, -16, -7, -7, -7, -6, -7, -16,
	-8, -16, -7, -8, -16, -6, 81, -1, -16, -1,
	-1, -2, -2, -2, -2, -2, 7, 8, -2, -2,
	-2, 78, 78, 80, 80, -5, 80, 80, 80, 80,
	-3, -4, 78, 78, 78, 78, 78, 78, 78, 78,
	80, -12, -12, 78, 80, 78, 78, 78, 78, 78,
	78, 78, 78, 78, -15, 38, 10, 78, 78, 78,
	-2, -2, -21, 48, -23, 80, 80, 79, -7, -7,
	-7, -7, -7, -7, -7, -22, 80, 80, -19, -6,
	79, -11, -11, -10, -10, -10, -16, -25, -1, -16,
	-25, -7, -8, 10, 38, 81, -16, -16, -17, -18,
	81, 40, -12, 78, 78, 78, 78, 78, 78, 38,
	10, -15, 10, 80, 80, -11, -7, -7, -1, -7,
	-7, 38,
}
var yyDef = []int{

	1, -2, 0, 3, 0, 0, 0, 9, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 162, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 162, 0, 0, 0, 162, 0, 0,
	0, 162, 4, 0, 0, 8, 10, 11, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 127,
	157, 0, 0, 138, 0, 137, 0, 140, 131, 134,
	0, 136, 128, 129, 153, 0, 164, 170, 171, 0,
	0, 0, 133, 130, 132, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	50, 0, 0, 142, 0, 64, 65, 0, 0, 0,
	0, 164, 0, 162, 0, 82, 163, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 105, 106, 162, 162, 163, 111, 162, 162, 0,
	0, 116, 0, 0, 0, 0, 124, 0, 0, 176,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 148, 149, 152, 0,
	0, 0, 0, 0, 0, 0, 172, 173, 174, 0,
	0, 0, 0, 0, 0, 0, 150, 0, 0, 152,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 67,
	68, 0, 0, 0, 0, 0, 126, 163, 81, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 107, 108, 109, 110, 112, 113, 0,
	0, 0, 0, 0, 0, 5, 6, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 7, 12, 24,
	25, 0, 36, 37, 61, 63, 13, 14, 28, 29,
	31, 0, 30, 33, 52, 53, 56, 62, 57, 59,
	58, 60, 0, 0, 167, 168, 169, 155, 0, 175,
	0, 0, 0, 0, 0, 164, 165, 166, 15, 26,
	27, 16, 17, 18, 19, 20, 21, 22, 23, 151,
	34, 127, 0, 41, 127, 42, 127, 44, 127, 46,
	127, 48, 49, 0, 54, 142, 66, 0, 70, 71,
	0, 73, 74, 0, 0, 0, 79, 80, 83, 84,
	0, 87, 89, 90, 0, 0, 97, 98, 99, 0,
	101, 0, 103, 104, 114, 115, 0, 0, 119, 0,
	0, 177, 178, 179, 180, 181, 0, 0, 184, 185,
	186, 0, 0, 158, 159, 0, 139, 141, 135, 154,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	69, 0, 0, 0, 125, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 117, 144, 0, 0, 0, 0,
	182, 183, 35, 131, 32, 156, 160, 0, 38, 40,
	39, 43, 45, 47, 51, 55, 72, 75, 76, 77,
	0, 85, 0, 88, 91, 92, 0, 0, 0, 0,
	0, 100, 102, 0, 145, 0, 120, 121, 122, 123,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 146,
	0, 118, 0, 161, 78, 86, 93, 94, 143, 95,
	96, 147,
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
				yyerror("redeclaration of %s", yyDollar[1].sym.Labelname)
			}
			yyDollar[1].sym.Type = LLAB
			yyDollar[1].sym.Value = int64(asm.PC)
		}
	case 6:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:89
		{
			yyDollar[1].sym.Type = LVAR
			yyDollar[1].sym.Value = yyDollar[3].lval
		}
	case 7:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:94
		{
			if yyDollar[1].sym.Value != yyDollar[3].lval {
				yyerror("redeclaration of %s", yyDollar[1].sym.Name)
			}
			yyDollar[1].sym.Value = yyDollar[3].lval
		}
	case 8:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:101
		{
			nosched = int(yyDollar[1].lval)
		}
	case 12:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:113
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 13:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:117
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 14:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:121
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 15:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:125
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 16:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:129
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 17:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:133
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 18:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:140
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 19:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:144
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 20:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:148
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 21:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:152
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 22:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:156
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 23:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:160
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 24:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:167
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 25:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:171
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 26:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:175
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 27:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:179
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 28:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:186
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 29:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:190
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 30:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:197
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 31:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:201
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 32:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:205
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr, &yyDollar[6].addr)
		}
	case 33:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:209
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 34:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:213
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &nullgen)
		}
	case 35:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:220
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr, &yyDollar[6].addr)
		}
	case 36:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:224
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 37:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:228
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 38:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:238
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &yyDollar[6].addr)
		}
	case 39:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:242
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &yyDollar[6].addr)
		}
	case 40:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:246
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr, &yyDollar[6].addr)
		}
	case 41:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:250
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 42:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:254
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 43:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:258
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &yyDollar[6].addr)
		}
	case 44:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:262
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 45:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:266
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &yyDollar[6].addr)
		}
	case 46:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:270
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 47:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:274
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &yyDollar[6].addr)
		}
	case 48:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:278
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 49:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:282
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 50:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:286
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[2].addr)
		}
	case 51:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:293
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &yyDollar[6].addr)
		}
	case 52:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:300
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 53:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:304
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 54:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:311
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].addr.Reg), &yyDollar[4].addr)
		}
	case 55:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:315
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &yyDollar[6].addr)
		}
	case 56:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:323
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 57:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:327
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 58:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:331
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 59:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:335
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 60:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:339
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 61:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:343
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 62:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:347
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 63:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:351
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 64:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:360
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &yyDollar[2].addr)
		}
	case 65:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:364
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &yyDollar[2].addr)
		}
	case 66:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:368
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &yyDollar[3].addr)
		}
	case 67:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:372
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &yyDollar[3].addr)
		}
	case 68:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:376
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &yyDollar[3].addr)
		}
	case 69:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:380
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &yyDollar[4].addr)
		}
	case 70:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:384
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 71:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:388
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 72:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:392
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[5].addr)
		}
	case 73:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:396
		{
			outcode(int(yyDollar[1].lval), &nullgen, int(yyDollar[2].lval), &yyDollar[4].addr)
		}
	case 74:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:400
		{
			outcode(int(yyDollar[1].lval), &nullgen, int(yyDollar[2].lval), &yyDollar[4].addr)
		}
	case 75:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:404
		{
			outcode(int(yyDollar[1].lval), &nullgen, int(yyDollar[2].lval), &yyDollar[5].addr)
		}
	case 76:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:408
		{
			var g obj.Addr
			g = nullgen
			g.Type = obj.TYPE_CONST
			g.Offset = yyDollar[2].lval
			outcode(int(yyDollar[1].lval), &g, int(REG_R0+yyDollar[4].lval), &yyDollar[6].addr)
		}
	case 77:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:416
		{
			var g obj.Addr
			g = nullgen
			g.Type = obj.TYPE_CONST
			g.Offset = yyDollar[2].lval
			outcode(int(yyDollar[1].lval), &g, int(REG_R0+yyDollar[4].lval), &yyDollar[6].addr)
		}
	case 78:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line a.y:424
		{
			var g obj.Addr
			g = nullgen
			g.Type = obj.TYPE_CONST
			g.Offset = yyDollar[2].lval
			outcode(int(yyDollar[1].lval), &g, int(REG_R0+yyDollar[4].lval), &yyDollar[7].addr)
		}
	case 79:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:435
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &nullgen)
		}
	case 80:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:439
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &nullgen)
		}
	case 81:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:443
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &nullgen)
		}
	case 82:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:447
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &nullgen)
		}
	case 83:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:454
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 84:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:458
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 85:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:462
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].addr.Reg), &yyDollar[6].addr)
		}
	case 86:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line a.y:466
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].addr.Reg), &yyDollar[6].addr, &yyDollar[8].addr)
		}
	case 87:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:470
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 88:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:474
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[6].addr.Reg), &yyDollar[4].addr)
		}
	case 89:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:481
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 90:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:485
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 91:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:489
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[6].addr.Reg), &yyDollar[4].addr)
		}
	case 92:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:493
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[6].addr.Reg), &yyDollar[4].addr)
		}
	case 93:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line a.y:500
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].addr.Reg), &yyDollar[6].addr, &yyDollar[8].addr)
		}
	case 94:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line a.y:504
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].addr.Reg), &yyDollar[6].addr, &yyDollar[8].addr)
		}
	case 95:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line a.y:508
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].addr.Reg), &yyDollar[6].addr, &yyDollar[8].addr)
		}
	case 96:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line a.y:512
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].addr.Reg), &yyDollar[6].addr, &yyDollar[8].addr)
		}
	case 97:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:519
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 98:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:523
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 99:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:531
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 100:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:535
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr, &yyDollar[6].addr)
		}
	case 101:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:539
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 102:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:543
		{
			outgcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr, &yyDollar[6].addr)
		}
	case 103:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:547
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 104:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:551
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 105:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:555
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &nullgen)
		}
	case 106:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:562
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &nullgen)
		}
	case 107:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:566
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &nullgen)
		}
	case 108:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:570
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &nullgen)
		}
	case 109:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:574
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &yyDollar[3].addr)
		}
	case 110:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:578
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &yyDollar[3].addr)
		}
	case 111:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:582
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &nullgen)
		}
	case 112:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:589
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &nullgen)
		}
	case 113:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:593
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &nullgen)
		}
	case 114:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:600
		{
			if yyDollar[2].addr.Type != obj.TYPE_CONST || yyDollar[4].addr.Type != obj.TYPE_CONST {
				yyerror("arguments to PCDATA must be integer constants")
			}
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 115:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:610
		{
			if yyDollar[2].addr.Type != obj.TYPE_CONST {
				yyerror("index for FUNCDATA must be integer constant")
			}
			if yyDollar[4].addr.Type != obj.TYPE_MEM || (yyDollar[4].addr.Name != obj.NAME_EXTERN && yyDollar[4].addr.Name != obj.NAME_STATIC) {
				yyerror("value for FUNCDATA must be symbol reference")
			}
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 116:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:623
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &nullgen)
		}
	case 117:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:630
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[5].addr)
		}
	case 118:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:635
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, int(yyDollar[4].lval), &yyDollar[7].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 119:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:647
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[4].addr)
		}
	case 120:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:652
		{
			asm.Settext(yyDollar[2].addr.Sym)
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[6].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 121:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:665
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[6].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 122:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:673
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[6].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 123:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line a.y:681
		{
			outcode(int(yyDollar[1].lval), &yyDollar[2].addr, 0, &yyDollar[6].addr)
			if asm.Pass > 1 {
				lastpc.From3.Type = obj.TYPE_CONST
				lastpc.From3.Offset = yyDollar[4].lval
			}
		}
	case 124:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:692
		{
			outcode(int(yyDollar[1].lval), &nullgen, 0, &nullgen)
		}
	case 125:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:698
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyDollar[1].lval + int64(asm.PC)
		}
	case 126:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:704
		{
			yyDollar[1].sym = asm.LabelLookup(yyDollar[1].sym)
			yyVAL.addr = nullgen
			if asm.Pass == 2 && yyDollar[1].sym.Type != LLAB {
				yyerror("undefined label: %s", yyDollar[1].sym.Labelname)
			}
			yyVAL.addr.Type = obj.TYPE_BRANCH
			yyVAL.addr.Offset = yyDollar[1].sym.Value + yyDollar[2].lval
		}
	case 127:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:716
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 128:
		yyVAL.addr = yyS[yypt-0].addr
	case 129:
		yyVAL.addr = yyS[yypt-0].addr
	case 130:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:728
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 131:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:736
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval) /* whole register */
		}
	case 132:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:743
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 133:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:751
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 134:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:759
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 135:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:765
		{
			if yyDollar[3].lval < 0 || yyDollar[3].lval >= 1024 {
				yyerror("SPR/DCR out of range")
			}
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval + yyDollar[3].lval)
		}
	case 136:
		yyVAL.addr = yyS[yypt-0].addr
	case 137:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:777
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 138:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:785
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 139:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:791
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(REG_F0 + yyDollar[3].lval)
		}
	case 140:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:799
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 141:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:805
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(REG_C0 + yyDollar[3].lval)
		}
	case 142:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:813
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_REG
			yyVAL.addr.Reg = int16(yyDollar[1].lval)
		}
	case 143:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:821
		{
			var mb, me int
			var v uint32

			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
			mb = int(yyDollar[1].lval)
			me = int(yyDollar[3].lval)
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
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:844
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = int64(yyDollar[1].lval)
			yyVAL.addr.U.Argsize = obj.ArgsSizeUnknown
		}
	case 145:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:851
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -int64(yyDollar[2].lval)
			yyVAL.addr.U.Argsize = obj.ArgsSizeUnknown
		}
	case 146:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:858
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = int64(yyDollar[1].lval)
			yyVAL.addr.U.Argsize = int32(yyDollar[3].lval)
		}
	case 147:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:865
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_TEXTSIZE
			yyVAL.addr.Offset = -int64(yyDollar[2].lval)
			yyVAL.addr.U.Argsize = int32(yyDollar[4].lval)
		}
	case 148:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:874
		{
			yyVAL.addr = yyDollar[2].addr
			yyVAL.addr.Type = obj.TYPE_ADDR
		}
	case 149:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:879
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_SCONST
			yyVAL.addr.U.Sval = yyDollar[2].sval
		}
	case 150:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:887
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = yyDollar[2].dval
		}
	case 151:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:893
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_FCONST
			yyVAL.addr.U.Dval = -yyDollar[3].dval
		}
	case 152:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:900
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_CONST
			yyVAL.addr.Offset = yyDollar[2].lval
		}
	case 153:
		yyVAL.lval = yyS[yypt-0].lval
	case 154:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:909
		{
			if yyVAL.lval < 0 || yyVAL.lval >= NREG {
				print("register value out of range\n")
			}
			yyVAL.lval = REG_R0 + yyDollar[3].lval
		}
	case 155:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:918
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[2].lval)
			yyVAL.addr.Offset = 0
		}
	case 156:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:925
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[2].lval)
			yyVAL.addr.Scale = int8(yyDollar[4].lval)
			yyVAL.addr.Offset = 0
		}
	case 157:
		yyVAL.addr = yyS[yypt-0].addr
	case 158:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:936
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Reg = int16(yyDollar[3].lval)
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 159:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:945
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyDollar[3].lval)
			yyVAL.addr.Sym = nil
			yyVAL.addr.Offset = yyDollar[1].lval
		}
	case 160:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line a.y:953
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = int8(yyDollar[4].lval)
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 0)
			yyVAL.addr.Offset = yyDollar[2].lval
		}
	case 161:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line a.y:961
		{
			yyVAL.addr = nullgen
			yyVAL.addr.Type = obj.TYPE_MEM
			yyVAL.addr.Name = obj.NAME_STATIC
			yyVAL.addr.Sym = obj.Linklookup(asm.Ctxt, yyDollar[1].sym.Name, 1)
			yyVAL.addr.Offset = yyDollar[4].lval
		}
	case 164:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line a.y:973
		{
			yyVAL.lval = 0
		}
	case 165:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:977
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 166:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:981
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 167:
		yyVAL.lval = yyS[yypt-0].lval
	case 168:
		yyVAL.lval = yyS[yypt-0].lval
	case 169:
		yyVAL.lval = yyS[yypt-0].lval
	case 170:
		yyVAL.lval = yyS[yypt-0].lval
	case 171:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line a.y:993
		{
			yyVAL.lval = yyDollar[1].sym.Value
		}
	case 172:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:997
		{
			yyVAL.lval = -yyDollar[2].lval
		}
	case 173:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:1001
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 174:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line a.y:1005
		{
			yyVAL.lval = ^yyDollar[2].lval
		}
	case 175:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:1009
		{
			yyVAL.lval = yyDollar[2].lval
		}
	case 176:
		yyVAL.lval = yyS[yypt-0].lval
	case 177:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:1016
		{
			yyVAL.lval = yyDollar[1].lval + yyDollar[3].lval
		}
	case 178:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:1020
		{
			yyVAL.lval = yyDollar[1].lval - yyDollar[3].lval
		}
	case 179:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:1024
		{
			yyVAL.lval = yyDollar[1].lval * yyDollar[3].lval
		}
	case 180:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:1028
		{
			yyVAL.lval = yyDollar[1].lval / yyDollar[3].lval
		}
	case 181:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:1032
		{
			yyVAL.lval = yyDollar[1].lval % yyDollar[3].lval
		}
	case 182:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:1036
		{
			yyVAL.lval = yyDollar[1].lval << uint(yyDollar[4].lval)
		}
	case 183:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line a.y:1040
		{
			yyVAL.lval = yyDollar[1].lval >> uint(yyDollar[4].lval)
		}
	case 184:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:1044
		{
			yyVAL.lval = yyDollar[1].lval & yyDollar[3].lval
		}
	case 185:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:1048
		{
			yyVAL.lval = yyDollar[1].lval ^ yyDollar[3].lval
		}
	case 186:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line a.y:1052
		{
			yyVAL.lval = yyDollar[1].lval | yyDollar[3].lval
		}
	}
	goto yystack /* stack new state and value */
}
