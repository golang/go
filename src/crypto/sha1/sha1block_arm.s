// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// ARM version of md5block.go

#include "textflag.h"

// SHA1 block routine. See sha1block.go for Go equivalent.
//
// There are 80 rounds of 4 types:
//   - rounds 0-15 are type 1 and load data (ROUND1 macro).
//   - rounds 16-19 are type 1 and do not load data (ROUND1x macro).
//   - rounds 20-39 are type 2 and do not load data (ROUND2 macro).
//   - rounds 40-59 are type 3 and do not load data (ROUND3 macro).
//   - rounds 60-79 are type 4 and do not load data (ROUND4 macro).
//
// Each round loads or shuffles the data, then computes a per-round
// function of b, c, d, and then mixes the result into and rotates the
// five registers a, b, c, d, e holding the intermediate results.
//
// The register rotation is implemented by rotating the arguments to
// the round macros instead of by explicit move instructions.

// Register definitions
data = 0	// Pointer to incoming data
const = 1	// Current constant for SHA round
a = 2		// SHA1 accumulator
b = 3		// SHA1 accumulator
c = 4		// SHA1 accumulator
d = 5		// SHA1 accumulator
e = 6		// SHA1 accumulator
t0 = 7		// Temporary
t1 = 8		// Temporary
// r9, r10 are forbidden
// r11 is OK provided you check the assembler that no synthetic instructions use it
t2 = 11		// Temporary
ctr = 12	// loop counter
w = 14		// point to w buffer

// func block(dig *digest, p []byte)
// 0(FP) is *digest
// 4(FP) is p.array (struct Slice)
// 8(FP) is p.len
//12(FP) is p.cap
//
// Stack frame
p_end = -4		// -4(SP) pointer to the end of data
p_data = p_end - 4	// -8(SP) current data pointer
w_buf = p_data - 4*80	// -328(SP) 80 words temporary buffer w uint32[80]
saved = w_buf - 4*5	// -348(SP) saved sha1 registers a,b,c,d,e - these must be last
// Total size +4 for saved LR is 352

	// w[i] = p[j]<<24 | p[j+1]<<16 | p[j+2]<<8 | p[j+3]
	// e += w[i]
#define LOAD(e) \
	MOVBU	2(R(data)), R(t0) ; \
	MOVBU	3(R(data)), R(t1) ; \
	MOVBU	1(R(data)), R(t2) ; \
	ORR	R(t0)<<8, R(t1), R(t0)	    ; \
	MOVBU.P	4(R(data)), R(t1) ; \
	ORR	R(t2)<<16, R(t0), R(t0)	    ; \
	ORR	R(t1)<<24, R(t0), R(t0)	    ; \
	MOVW.P	R(t0), 4(R(w))		    ; \
	ADD	R(t0), R(e), R(e)
	
	// tmp := w[(i-3)&0xf] ^ w[(i-8)&0xf] ^ w[(i-14)&0xf] ^ w[(i)&0xf]
	// w[i&0xf] = tmp<<1 | tmp>>(32-1)
	// e += w[i&0xf] 
#define SHUFFLE(e) \
	MOVW	(-16*4)(R(w)), R(t0) ; \
	MOVW	(-14*4)(R(w)), R(t1) ; \
	MOVW	(-8*4)(R(w)), R(t2)  ; \
	EOR	R(t0), R(t1), R(t0)  ; \
	MOVW	(-3*4)(R(w)), R(t1)  ; \
	EOR	R(t2), R(t0), R(t0)  ; \
	EOR	R(t0), R(t1), R(t0)  ; \
	MOVW	R(t0)@>(32-1), R(t0)  ; \
	MOVW.P	R(t0), 4(R(w))	  ; \
	ADD	R(t0), R(e), R(e)

	// t1 = (b & c) | ((~b) & d)
#define FUNC1(a, b, c, d, e) \
	MVN	R(b), R(t1)	   ; \
	AND	R(b), R(c), R(t0)  ; \
	AND	R(d), R(t1), R(t1) ; \
	ORR	R(t0), R(t1), R(t1)

	// t1 = b ^ c ^ d
#define FUNC2(a, b, c, d, e) \
	EOR	R(b), R(c), R(t1) ; \
	EOR	R(d), R(t1), R(t1)

	// t1 = (b & c) | (b & d) | (c & d) =
	// t1 = (b & c) | ((b | c) & d)
#define FUNC3(a, b, c, d, e) \
	ORR	R(b), R(c), R(t0)  ; \
	AND	R(b), R(c), R(t1)  ; \
	AND	R(d), R(t0), R(t0) ; \
	ORR	R(t0), R(t1), R(t1)

#define FUNC4 FUNC2

	// a5 := a<<5 | a>>(32-5)
	// b = b<<30 | b>>(32-30)
	// e = a5 + t1 + e + const
#define MIX(a, b, c, d, e) \
	ADD	R(t1), R(e), R(e)	 ; \
	MOVW	R(b)@>(32-30), R(b)	 ; \
	ADD	R(a)@>(32-5), R(e), R(e) ; \
	ADD	R(const), R(e), R(e)

#define ROUND1(a, b, c, d, e) \
	LOAD(e)		; \
	FUNC1(a, b, c, d, e)	; \
	MIX(a, b, c, d, e)

#define ROUND1x(a, b, c, d, e) \
	SHUFFLE(e)	; \
	FUNC1(a, b, c, d, e)	; \
	MIX(a, b, c, d, e)

#define ROUND2(a, b, c, d, e) \
	SHUFFLE(e)	; \
	FUNC2(a, b, c, d, e)	; \
	MIX(a, b, c, d, e)

#define ROUND3(a, b, c, d, e) \
	SHUFFLE(e)	; \
	FUNC3(a, b, c, d, e)	; \
	MIX(a, b, c, d, e)

#define ROUND4(a, b, c, d, e) \
	SHUFFLE(e)	; \
	FUNC4(a, b, c, d, e)	; \
	MIX(a, b, c, d, e)


// func block(dig *digest, p []byte)
TEXT	·block(SB), 0, $352-16
	MOVW	p+4(FP), R(data)	// pointer to the data
	MOVW	p_len+8(FP), R(t0)	// number of bytes
	ADD	R(data), R(t0)
	MOVW	R(t0), p_end(SP)	// pointer to end of data

	// Load up initial SHA1 accumulator
	MOVW	dig+0(FP), R(t0)
	MOVM.IA (R(t0)), [R(a),R(b),R(c),R(d),R(e)]

loop:
	// Save registers at SP+4 onwards
	MOVM.IB [R(a),R(b),R(c),R(d),R(e)], (R13)

	MOVW	$w_buf(SP), R(w)
	MOVW	$0x5A827999, R(const)
	MOVW	$3, R(ctr)
loop1:	ROUND1(a, b, c, d, e)
	ROUND1(e, a, b, c, d)
	ROUND1(d, e, a, b, c)
	ROUND1(c, d, e, a, b)
	ROUND1(b, c, d, e, a)
	SUB.S	$1, R(ctr)
	BNE	loop1

	ROUND1(a, b, c, d, e)
	ROUND1x(e, a, b, c, d)
	ROUND1x(d, e, a, b, c)
	ROUND1x(c, d, e, a, b)
	ROUND1x(b, c, d, e, a)
	
	MOVW	$0x6ED9EBA1, R(const)
	MOVW	$4, R(ctr)
loop2:	ROUND2(a, b, c, d, e)
	ROUND2(e, a, b, c, d)
	ROUND2(d, e, a, b, c)
	ROUND2(c, d, e, a, b)
	ROUND2(b, c, d, e, a)
	SUB.S	$1, R(ctr)
	BNE	loop2
	
	MOVW	$0x8F1BBCDC, R(const)
	MOVW	$4, R(ctr)
loop3:	ROUND3(a, b, c, d, e)
	ROUND3(e, a, b, c, d)
	ROUND3(d, e, a, b, c)
	ROUND3(c, d, e, a, b)
	ROUND3(b, c, d, e, a)
	SUB.S	$1, R(ctr)
	BNE	loop3
	
	MOVW	$0xCA62C1D6, R(const)
	MOVW	$4, R(ctr)
loop4:	ROUND4(a, b, c, d, e)
	ROUND4(e, a, b, c, d)
	ROUND4(d, e, a, b, c)
	ROUND4(c, d, e, a, b)
	ROUND4(b, c, d, e, a)
	SUB.S	$1, R(ctr)
	BNE	loop4

	// Accumulate - restoring registers from SP+4
	MOVM.IB (R13), [R(t0),R(t1),R(t2),R(ctr),R(w)]
	ADD	R(t0), R(a)
	ADD	R(t1), R(b)
	ADD	R(t2), R(c)
	ADD	R(ctr), R(d)
	ADD	R(w), R(e)

	MOVW	p_end(SP), R(t0)
	CMP	R(t0), R(data)
	BLO	loop

	// Save final SHA1 accumulator
	MOVW	dig+0(FP), R(t0)
	MOVM.IA [R(a),R(b),R(c),R(d),R(e)], (R(t0))

	RET
