// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64le || ppc64

#include "go_asm.h"
#include "textflag.h"

TEXT ·Count<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
#ifdef GOEXPERIMENT_regabiargs
// R3 = byte array pointer 
// R4 = length
        MOVBZ R6,R5               // R5 = byte
#else

	MOVD  b_base+0(FP), R3    // R3 = byte array pointer
	MOVD  b_len+8(FP), R4     // R4 = length
	MOVBZ c+24(FP), R5        // R5 = byte
	MOVD  $ret+32(FP), R14    // R14 = &ret
#endif
	BR    countbytebody<>(SB)

TEXT ·CountString<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-32
#ifdef GOEXPERIMENT_regabiargs
// R3 = byte array pointer
// R4 = length
        MOVBZ R5,R5               // R5 = byte
#else
	MOVD  s_base+0(FP), R3    // R3 = string
	MOVD  s_len+8(FP), R4     // R4 = length
	MOVBZ c+16(FP), R5        // R5 = byte
	MOVD  $ret+24(FP), R14    // R14 = &ret
#endif
	BR    countbytebody<>(SB)

// R3: addr of string
// R4: len of string
// R5: byte to count
// R14: addr for return value when not regabi
// endianness shouldn't matter since we are just counting and order
// is irrelevant
TEXT countbytebody<>(SB), NOSPLIT|NOFRAME, $0-0
	DCBT (R3)    // Prepare cache line.
	MOVD R0, R18 // byte count
	MOVD R3, R19 // Save base address for calculating the index later.
	MOVD R4, R16

	MOVD   R5, R6
	RLDIMI $8, R6, $48, R6
	RLDIMI $16, R6, $32, R6
	RLDIMI $32, R6, $0, R6  // fill reg with the byte to count

	VSPLTISW $3, V4     // used for shift
	MTVRD    R6, V1     // move compare byte
	VSPLTB   $7, V1, V1 // replicate byte across V1

	CMPU   R4, $32          // Check if it's a small string (<32 bytes)
	BLT    tail             // Jump to the small string case
	XXLXOR VS37, VS37, VS37 // clear V5 (aka VS37) to use as accumulator

cmploop:
	LXVW4X (R3), VS32 // load bytes from string

	// when the bytes match, the corresponding byte contains all 1s
	VCMPEQUB V1, V0, V2     // compare bytes
	VPOPCNTD V2, V3         // each double word contains its count
	VADDUDM  V3, V5, V5     // accumulate bit count in each double word
	ADD      $16, R3, R3    // increment pointer
	SUB      $16, R16, R16  // remaining bytes
	CMP      R16, $16       // at least 16 remaining?
	BGE      cmploop
	VSRD     V5, V4, V5     // shift by 3 to convert bits to bytes
	VSLDOI   $8, V5, V5, V6 // get the double word values from vector
	MFVSRD   V5, R9
	MFVSRD   V6, R10
	ADD      R9, R10, R9
	ADD      R9, R18, R18

tail:
	CMP R16, $8 // 8 bytes left?
	BLT small

	MOVD    (R3), R12     // load 8 bytes
	CMPB    R12, R6, R17  // compare bytes
	POPCNTD R17, R15      // bit count
	SRD     $3, R15, R15  // byte count
	ADD     R15, R18, R18 // add to byte count

next1:
	ADD $8, R3, R3
	SUB $8, R16, R16 // remaining bytes
	BR  tail

small:
	CMP   $0, R16   // any remaining
	BEQ   done
	MOVBZ (R3), R12 // check each remaining byte
	CMP   R12, R5
	BNE   next2
	ADD   $1, R18

next2:
	SUB $1, R16
	ADD $1, R3  // inc address
	BR  small

done:
#ifdef GOEXPERIMENT_regabiargs
        MOVD R18, R3    // return count
#else
	MOVD R18, (R14) // return count
#endif

	RET
