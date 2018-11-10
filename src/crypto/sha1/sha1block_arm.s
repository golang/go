// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// ARM version of md5block.go

#include "textflag.h"

// SHA-1 block routine. See sha1block.go for Go equivalent.
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
#define Rdata	R0	// Pointer to incoming data
#define Rconst	R1	// Current constant for SHA round
#define Ra	R2		// SHA-1 accumulator
#define Rb	R3		// SHA-1 accumulator
#define Rc	R4		// SHA-1 accumulator
#define Rd	R5		// SHA-1 accumulator
#define Re	R6		// SHA-1 accumulator
#define Rt0	R7		// Temporary
#define Rt1	R8		// Temporary
// r9, r10 are forbidden
// r11 is OK provided you check the assembler that no synthetic instructions use it
#define Rt2	R11		// Temporary
#define Rctr	R12	// loop counter
#define Rw	R14		// point to w buffer

// func block(dig *digest, p []byte)
// 0(FP) is *digest
// 4(FP) is p.array (struct Slice)
// 8(FP) is p.len
//12(FP) is p.cap
//
// Stack frame
#define p_end	end-4(SP)		// pointer to the end of data
#define p_data	data-8(SP)	// current data pointer (unused?)
#define w_buf	buf-(8+4*80)(SP)	//80 words temporary buffer w uint32[80]
#define saved	abcde-(8+4*80+4*5)(SP)	// saved sha1 registers a,b,c,d,e - these must be last (unused?)
// Total size +4 for saved LR is 352

	// w[i] = p[j]<<24 | p[j+1]<<16 | p[j+2]<<8 | p[j+3]
	// e += w[i]
#define LOAD(Re) \
	MOVBU	2(Rdata), Rt0 ; \
	MOVBU	3(Rdata), Rt1 ; \
	MOVBU	1(Rdata), Rt2 ; \
	ORR	Rt0<<8, Rt1, Rt0	    ; \
	MOVBU.P	4(Rdata), Rt1 ; \
	ORR	Rt2<<16, Rt0, Rt0	    ; \
	ORR	Rt1<<24, Rt0, Rt0	    ; \
	MOVW.P	Rt0, 4(Rw)		    ; \
	ADD	Rt0, Re, Re
	
	// tmp := w[(i-3)&0xf] ^ w[(i-8)&0xf] ^ w[(i-14)&0xf] ^ w[(i)&0xf]
	// w[i&0xf] = tmp<<1 | tmp>>(32-1)
	// e += w[i&0xf] 
#define SHUFFLE(Re) \
	MOVW	(-16*4)(Rw), Rt0 ; \
	MOVW	(-14*4)(Rw), Rt1 ; \
	MOVW	(-8*4)(Rw), Rt2  ; \
	EOR	Rt0, Rt1, Rt0  ; \
	MOVW	(-3*4)(Rw), Rt1  ; \
	EOR	Rt2, Rt0, Rt0  ; \
	EOR	Rt0, Rt1, Rt0  ; \
	MOVW	Rt0@>(32-1), Rt0  ; \
	MOVW.P	Rt0, 4(Rw)	  ; \
	ADD	Rt0, Re, Re

	// t1 = (b & c) | ((~b) & d)
#define FUNC1(Ra, Rb, Rc, Rd, Re) \
	MVN	Rb, Rt1	   ; \
	AND	Rb, Rc, Rt0  ; \
	AND	Rd, Rt1, Rt1 ; \
	ORR	Rt0, Rt1, Rt1

	// t1 = b ^ c ^ d
#define FUNC2(Ra, Rb, Rc, Rd, Re) \
	EOR	Rb, Rc, Rt1 ; \
	EOR	Rd, Rt1, Rt1

	// t1 = (b & c) | (b & d) | (c & d) =
	// t1 = (b & c) | ((b | c) & d)
#define FUNC3(Ra, Rb, Rc, Rd, Re) \
	ORR	Rb, Rc, Rt0  ; \
	AND	Rb, Rc, Rt1  ; \
	AND	Rd, Rt0, Rt0 ; \
	ORR	Rt0, Rt1, Rt1

#define FUNC4 FUNC2

	// a5 := a<<5 | a>>(32-5)
	// b = b<<30 | b>>(32-30)
	// e = a5 + t1 + e + const
#define MIX(Ra, Rb, Rc, Rd, Re) \
	ADD	Rt1, Re, Re	 ; \
	MOVW	Rb@>(32-30), Rb	 ; \
	ADD	Ra@>(32-5), Re, Re ; \
	ADD	Rconst, Re, Re

#define ROUND1(Ra, Rb, Rc, Rd, Re) \
	LOAD(Re)		; \
	FUNC1(Ra, Rb, Rc, Rd, Re)	; \
	MIX(Ra, Rb, Rc, Rd, Re)

#define ROUND1x(Ra, Rb, Rc, Rd, Re) \
	SHUFFLE(Re)	; \
	FUNC1(Ra, Rb, Rc, Rd, Re)	; \
	MIX(Ra, Rb, Rc, Rd, Re)

#define ROUND2(Ra, Rb, Rc, Rd, Re) \
	SHUFFLE(Re)	; \
	FUNC2(Ra, Rb, Rc, Rd, Re)	; \
	MIX(Ra, Rb, Rc, Rd, Re)

#define ROUND3(Ra, Rb, Rc, Rd, Re) \
	SHUFFLE(Re)	; \
	FUNC3(Ra, Rb, Rc, Rd, Re)	; \
	MIX(Ra, Rb, Rc, Rd, Re)

#define ROUND4(Ra, Rb, Rc, Rd, Re) \
	SHUFFLE(Re)	; \
	FUNC4(Ra, Rb, Rc, Rd, Re)	; \
	MIX(Ra, Rb, Rc, Rd, Re)


// func block(dig *digest, p []byte)
TEXT	·block(SB), 0, $352-16
	MOVW	p+4(FP), Rdata	// pointer to the data
	MOVW	p_len+8(FP), Rt0	// number of bytes
	ADD	Rdata, Rt0
	MOVW	Rt0, p_end	// pointer to end of data

	// Load up initial SHA-1 accumulator
	MOVW	dig+0(FP), Rt0
	MOVM.IA (Rt0), [Ra,Rb,Rc,Rd,Re]

loop:
	// Save registers at SP+4 onwards
	MOVM.IB [Ra,Rb,Rc,Rd,Re], (R13)

	MOVW	$w_buf, Rw
	MOVW	$0x5A827999, Rconst
	MOVW	$3, Rctr
loop1:	ROUND1(Ra, Rb, Rc, Rd, Re)
	ROUND1(Re, Ra, Rb, Rc, Rd)
	ROUND1(Rd, Re, Ra, Rb, Rc)
	ROUND1(Rc, Rd, Re, Ra, Rb)
	ROUND1(Rb, Rc, Rd, Re, Ra)
	SUB.S	$1, Rctr
	BNE	loop1

	ROUND1(Ra, Rb, Rc, Rd, Re)
	ROUND1x(Re, Ra, Rb, Rc, Rd)
	ROUND1x(Rd, Re, Ra, Rb, Rc)
	ROUND1x(Rc, Rd, Re, Ra, Rb)
	ROUND1x(Rb, Rc, Rd, Re, Ra)
	
	MOVW	$0x6ED9EBA1, Rconst
	MOVW	$4, Rctr
loop2:	ROUND2(Ra, Rb, Rc, Rd, Re)
	ROUND2(Re, Ra, Rb, Rc, Rd)
	ROUND2(Rd, Re, Ra, Rb, Rc)
	ROUND2(Rc, Rd, Re, Ra, Rb)
	ROUND2(Rb, Rc, Rd, Re, Ra)
	SUB.S	$1, Rctr
	BNE	loop2
	
	MOVW	$0x8F1BBCDC, Rconst
	MOVW	$4, Rctr
loop3:	ROUND3(Ra, Rb, Rc, Rd, Re)
	ROUND3(Re, Ra, Rb, Rc, Rd)
	ROUND3(Rd, Re, Ra, Rb, Rc)
	ROUND3(Rc, Rd, Re, Ra, Rb)
	ROUND3(Rb, Rc, Rd, Re, Ra)
	SUB.S	$1, Rctr
	BNE	loop3
	
	MOVW	$0xCA62C1D6, Rconst
	MOVW	$4, Rctr
loop4:	ROUND4(Ra, Rb, Rc, Rd, Re)
	ROUND4(Re, Ra, Rb, Rc, Rd)
	ROUND4(Rd, Re, Ra, Rb, Rc)
	ROUND4(Rc, Rd, Re, Ra, Rb)
	ROUND4(Rb, Rc, Rd, Re, Ra)
	SUB.S	$1, Rctr
	BNE	loop4

	// Accumulate - restoring registers from SP+4
	MOVM.IB (R13), [Rt0,Rt1,Rt2,Rctr,Rw]
	ADD	Rt0, Ra
	ADD	Rt1, Rb
	ADD	Rt2, Rc
	ADD	Rctr, Rd
	ADD	Rw, Re

	MOVW	p_end, Rt0
	CMP	Rt0, Rdata
	BLO	loop

	// Save final SHA-1 accumulator
	MOVW	dig+0(FP), Rt0
	MOVM.IA [Ra,Rb,Rc,Rd,Re], (Rt0)

	RET
