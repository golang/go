// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm,!gccgo,!appengine,!nacl

#include "textflag.h"

// This code was translated into a form compatible with 5a from the public
// domain source by Andrew Moon: github.com/floodyberry/poly1305-opt/blob/master/app/extensions/poly1305.

DATA ·poly1305_init_constants_armv6<>+0x00(SB)/4, $0x3ffffff
DATA ·poly1305_init_constants_armv6<>+0x04(SB)/4, $0x3ffff03
DATA ·poly1305_init_constants_armv6<>+0x08(SB)/4, $0x3ffc0ff
DATA ·poly1305_init_constants_armv6<>+0x0c(SB)/4, $0x3f03fff
DATA ·poly1305_init_constants_armv6<>+0x10(SB)/4, $0x00fffff
GLOBL ·poly1305_init_constants_armv6<>(SB), 8, $20

// Warning: the linker may use R11 to synthesize certain instructions. Please
// take care and verify that no synthetic instructions use it.

TEXT poly1305_init_ext_armv6<>(SB), NOSPLIT, $0
	// Needs 16 bytes of stack and 64 bytes of space pointed to by R0.  (It
	// might look like it's only 60 bytes of space but the final four bytes
	// will be written by another function.) We need to skip over four
	// bytes of stack because that's saving the value of 'g'.
	ADD       $4, R13, R8
	MOVM.IB   [R4-R7], (R8)
	MOVM.IA.W (R1), [R2-R5]
	MOVW      $·poly1305_init_constants_armv6<>(SB), R7
	MOVW      R2, R8
	MOVW      R2>>26, R9
	MOVW      R3>>20, g
	MOVW      R4>>14, R11
	MOVW      R5>>8, R12
	ORR       R3<<6, R9, R9
	ORR       R4<<12, g, g
	ORR       R5<<18, R11, R11
	MOVM.IA   (R7), [R2-R6]
	AND       R8, R2, R2
	AND       R9, R3, R3
	AND       g, R4, R4
	AND       R11, R5, R5
	AND       R12, R6, R6
	MOVM.IA.W [R2-R6], (R0)
	EOR       R2, R2, R2
	EOR       R3, R3, R3
	EOR       R4, R4, R4
	EOR       R5, R5, R5
	EOR       R6, R6, R6
	MOVM.IA.W [R2-R6], (R0)
	MOVM.IA.W (R1), [R2-R5]
	MOVM.IA   [R2-R6], (R0)
	ADD       $20, R13, R0
	MOVM.DA   (R0), [R4-R7]
	RET

#define MOVW_UNALIGNED(Rsrc, Rdst, Rtmp, offset) \
	MOVBU (offset+0)(Rsrc), Rtmp; \
	MOVBU Rtmp, (offset+0)(Rdst); \
	MOVBU (offset+1)(Rsrc), Rtmp; \
	MOVBU Rtmp, (offset+1)(Rdst); \
	MOVBU (offset+2)(Rsrc), Rtmp; \
	MOVBU Rtmp, (offset+2)(Rdst); \
	MOVBU (offset+3)(Rsrc), Rtmp; \
	MOVBU Rtmp, (offset+3)(Rdst)

TEXT poly1305_blocks_armv6<>(SB), NOSPLIT, $0
	// Needs 24 bytes of stack for saved registers and then 88 bytes of
	// scratch space after that. We assume that 24 bytes at (R13) have
	// already been used: four bytes for the link register saved in the
	// prelude of poly1305_auth_armv6, four bytes for saving the value of g
	// in that function and 16 bytes of scratch space used around
	// poly1305_finish_ext_armv6_skip1.
	ADD     $24, R13, R12
	MOVM.IB [R4-R8, R14], (R12)
	MOVW    R0, 88(R13)
	MOVW    R1, 92(R13)
	MOVW    R2, 96(R13)
	MOVW    R1, R14
	MOVW    R2, R12
	MOVW    56(R0), R8
	WORD    $0xe1180008                // TST R8, R8 not working see issue 5921
	EOR     R6, R6, R6
	MOVW.EQ $(1<<24), R6
	MOVW    R6, 84(R13)
	ADD     $116, R13, g
	MOVM.IA (R0), [R0-R9]
	MOVM.IA [R0-R4], (g)
	CMP     $16, R12
	BLO     poly1305_blocks_armv6_done

poly1305_blocks_armv6_mainloop:
	WORD    $0xe31e0003                            // TST R14, #3 not working see issue 5921
	BEQ     poly1305_blocks_armv6_mainloop_aligned
	ADD     $100, R13, g
	MOVW_UNALIGNED(R14, g, R0, 0)
	MOVW_UNALIGNED(R14, g, R0, 4)
	MOVW_UNALIGNED(R14, g, R0, 8)
	MOVW_UNALIGNED(R14, g, R0, 12)
	MOVM.IA (g), [R0-R3]
	ADD     $16, R14
	B       poly1305_blocks_armv6_mainloop_loaded

poly1305_blocks_armv6_mainloop_aligned:
	MOVM.IA.W (R14), [R0-R3]

poly1305_blocks_armv6_mainloop_loaded:
	MOVW    R0>>26, g
	MOVW    R1>>20, R11
	MOVW    R2>>14, R12
	MOVW    R14, 92(R13)
	MOVW    R3>>8, R4
	ORR     R1<<6, g, g
	ORR     R2<<12, R11, R11
	ORR     R3<<18, R12, R12
	BIC     $0xfc000000, R0, R0
	BIC     $0xfc000000, g, g
	MOVW    84(R13), R3
	BIC     $0xfc000000, R11, R11
	BIC     $0xfc000000, R12, R12
	ADD     R0, R5, R5
	ADD     g, R6, R6
	ORR     R3, R4, R4
	ADD     R11, R7, R7
	ADD     $116, R13, R14
	ADD     R12, R8, R8
	ADD     R4, R9, R9
	MOVM.IA (R14), [R0-R4]
	MULLU   R4, R5, (R11, g)
	MULLU   R3, R5, (R14, R12)
	MULALU  R3, R6, (R11, g)
	MULALU  R2, R6, (R14, R12)
	MULALU  R2, R7, (R11, g)
	MULALU  R1, R7, (R14, R12)
	ADD     R4<<2, R4, R4
	ADD     R3<<2, R3, R3
	MULALU  R1, R8, (R11, g)
	MULALU  R0, R8, (R14, R12)
	MULALU  R0, R9, (R11, g)
	MULALU  R4, R9, (R14, R12)
	MOVW    g, 76(R13)
	MOVW    R11, 80(R13)
	MOVW    R12, 68(R13)
	MOVW    R14, 72(R13)
	MULLU   R2, R5, (R11, g)
	MULLU   R1, R5, (R14, R12)
	MULALU  R1, R6, (R11, g)
	MULALU  R0, R6, (R14, R12)
	MULALU  R0, R7, (R11, g)
	MULALU  R4, R7, (R14, R12)
	ADD     R2<<2, R2, R2
	ADD     R1<<2, R1, R1
	MULALU  R4, R8, (R11, g)
	MULALU  R3, R8, (R14, R12)
	MULALU  R3, R9, (R11, g)
	MULALU  R2, R9, (R14, R12)
	MOVW    g, 60(R13)
	MOVW    R11, 64(R13)
	MOVW    R12, 52(R13)
	MOVW    R14, 56(R13)
	MULLU   R0, R5, (R11, g)
	MULALU  R4, R6, (R11, g)
	MULALU  R3, R7, (R11, g)
	MULALU  R2, R8, (R11, g)
	MULALU  R1, R9, (R11, g)
	ADD     $52, R13, R0
	MOVM.IA (R0), [R0-R7]
	MOVW    g>>26, R12
	MOVW    R4>>26, R14
	ORR     R11<<6, R12, R12
	ORR     R5<<6, R14, R14
	BIC     $0xfc000000, g, g
	BIC     $0xfc000000, R4, R4
	ADD.S   R12, R0, R0
	ADC     $0, R1, R1
	ADD.S   R14, R6, R6
	ADC     $0, R7, R7
	MOVW    R0>>26, R12
	MOVW    R6>>26, R14
	ORR     R1<<6, R12, R12
	ORR     R7<<6, R14, R14
	BIC     $0xfc000000, R0, R0
	BIC     $0xfc000000, R6, R6
	ADD     R14<<2, R14, R14
	ADD.S   R12, R2, R2
	ADC     $0, R3, R3
	ADD     R14, g, g
	MOVW    R2>>26, R12
	MOVW    g>>26, R14
	ORR     R3<<6, R12, R12
	BIC     $0xfc000000, g, R5
	BIC     $0xfc000000, R2, R7
	ADD     R12, R4, R4
	ADD     R14, R0, R0
	MOVW    R4>>26, R12
	BIC     $0xfc000000, R4, R8
	ADD     R12, R6, R9
	MOVW    96(R13), R12
	MOVW    92(R13), R14
	MOVW    R0, R6
	CMP     $32, R12
	SUB     $16, R12, R12
	MOVW    R12, 96(R13)
	BHS     poly1305_blocks_armv6_mainloop

poly1305_blocks_armv6_done:
	MOVW    88(R13), R12
	MOVW    R5, 20(R12)
	MOVW    R6, 24(R12)
	MOVW    R7, 28(R12)
	MOVW    R8, 32(R12)
	MOVW    R9, 36(R12)
	ADD     $48, R13, R0
	MOVM.DA (R0), [R4-R8, R14]
	RET

#define MOVHUP_UNALIGNED(Rsrc, Rdst, Rtmp) \
	MOVBU.P 1(Rsrc), Rtmp; \
	MOVBU.P Rtmp, 1(Rdst); \
	MOVBU.P 1(Rsrc), Rtmp; \
	MOVBU.P Rtmp, 1(Rdst)

#define MOVWP_UNALIGNED(Rsrc, Rdst, Rtmp) \
	MOVHUP_UNALIGNED(Rsrc, Rdst, Rtmp); \
	MOVHUP_UNALIGNED(Rsrc, Rdst, Rtmp)

// func poly1305_auth_armv6(out *[16]byte, m *byte, mlen uint32, key *[32]key)
TEXT ·poly1305_auth_armv6(SB), $196-16
	// The value 196, just above, is the sum of 64 (the size of the context
	// structure) and 132 (the amount of stack needed).
	//
	// At this point, the stack pointer (R13) has been moved down. It
	// points to the saved link register and there's 196 bytes of free
	// space above it.
	//
	// The stack for this function looks like:
	//
	// +---------------------
	// |
	// | 64 bytes of context structure
	// |
	// +---------------------
	// |
	// | 112 bytes for poly1305_blocks_armv6
	// |
	// +---------------------
	// | 16 bytes of final block, constructed at
	// | poly1305_finish_ext_armv6_skip8
	// +---------------------
	// | four bytes of saved 'g'
	// +---------------------
	// | lr, saved by prelude    <- R13 points here
	// +---------------------
	MOVW g, 4(R13)

	MOVW out+0(FP), R4
	MOVW m+4(FP), R5
	MOVW mlen+8(FP), R6
	MOVW key+12(FP), R7

	ADD  $136, R13, R0 // 136 = 4 + 4 + 16 + 112
	MOVW R7, R1

	// poly1305_init_ext_armv6 will write to the stack from R13+4, but
	// that's ok because none of the other values have been written yet.
	BL    poly1305_init_ext_armv6<>(SB)
	BIC.S $15, R6, R2
	BEQ   poly1305_auth_armv6_noblocks
	ADD   $136, R13, R0
	MOVW  R5, R1
	ADD   R2, R5, R5
	SUB   R2, R6, R6
	BL    poly1305_blocks_armv6<>(SB)

poly1305_auth_armv6_noblocks:
	ADD  $136, R13, R0
	MOVW R5, R1
	MOVW R6, R2
	MOVW R4, R3

	MOVW  R0, R5
	MOVW  R1, R6
	MOVW  R2, R7
	MOVW  R3, R8
	AND.S R2, R2, R2
	BEQ   poly1305_finish_ext_armv6_noremaining
	EOR   R0, R0
	ADD   $8, R13, R9                           // 8 = offset to 16 byte scratch space
	MOVW  R0, (R9)
	MOVW  R0, 4(R9)
	MOVW  R0, 8(R9)
	MOVW  R0, 12(R9)
	WORD  $0xe3110003                           // TST R1, #3 not working see issue 5921
	BEQ   poly1305_finish_ext_armv6_aligned
	WORD  $0xe3120008                           // TST R2, #8 not working see issue 5921
	BEQ   poly1305_finish_ext_armv6_skip8
	MOVWP_UNALIGNED(R1, R9, g)
	MOVWP_UNALIGNED(R1, R9, g)

poly1305_finish_ext_armv6_skip8:
	WORD $0xe3120004                     // TST $4, R2 not working see issue 5921
	BEQ  poly1305_finish_ext_armv6_skip4
	MOVWP_UNALIGNED(R1, R9, g)

poly1305_finish_ext_armv6_skip4:
	WORD $0xe3120002                     // TST $2, R2 not working see issue 5921
	BEQ  poly1305_finish_ext_armv6_skip2
	MOVHUP_UNALIGNED(R1, R9, g)
	B    poly1305_finish_ext_armv6_skip2

poly1305_finish_ext_armv6_aligned:
	WORD      $0xe3120008                             // TST R2, #8 not working see issue 5921
	BEQ       poly1305_finish_ext_armv6_skip8_aligned
	MOVM.IA.W (R1), [g-R11]
	MOVM.IA.W [g-R11], (R9)

poly1305_finish_ext_armv6_skip8_aligned:
	WORD   $0xe3120004                             // TST $4, R2 not working see issue 5921
	BEQ    poly1305_finish_ext_armv6_skip4_aligned
	MOVW.P 4(R1), g
	MOVW.P g, 4(R9)

poly1305_finish_ext_armv6_skip4_aligned:
	WORD    $0xe3120002                     // TST $2, R2 not working see issue 5921
	BEQ     poly1305_finish_ext_armv6_skip2
	MOVHU.P 2(R1), g
	MOVH.P  g, 2(R9)

poly1305_finish_ext_armv6_skip2:
	WORD    $0xe3120001                     // TST $1, R2 not working see issue 5921
	BEQ     poly1305_finish_ext_armv6_skip1
	MOVBU.P 1(R1), g
	MOVBU.P g, 1(R9)

poly1305_finish_ext_armv6_skip1:
	MOVW  $1, R11
	MOVBU R11, 0(R9)
	MOVW  R11, 56(R5)
	MOVW  R5, R0
	ADD   $8, R13, R1
	MOVW  $16, R2
	BL    poly1305_blocks_armv6<>(SB)

poly1305_finish_ext_armv6_noremaining:
	MOVW      20(R5), R0
	MOVW      24(R5), R1
	MOVW      28(R5), R2
	MOVW      32(R5), R3
	MOVW      36(R5), R4
	MOVW      R4>>26, R12
	BIC       $0xfc000000, R4, R4
	ADD       R12<<2, R12, R12
	ADD       R12, R0, R0
	MOVW      R0>>26, R12
	BIC       $0xfc000000, R0, R0
	ADD       R12, R1, R1
	MOVW      R1>>26, R12
	BIC       $0xfc000000, R1, R1
	ADD       R12, R2, R2
	MOVW      R2>>26, R12
	BIC       $0xfc000000, R2, R2
	ADD       R12, R3, R3
	MOVW      R3>>26, R12
	BIC       $0xfc000000, R3, R3
	ADD       R12, R4, R4
	ADD       $5, R0, R6
	MOVW      R6>>26, R12
	BIC       $0xfc000000, R6, R6
	ADD       R12, R1, R7
	MOVW      R7>>26, R12
	BIC       $0xfc000000, R7, R7
	ADD       R12, R2, g
	MOVW      g>>26, R12
	BIC       $0xfc000000, g, g
	ADD       R12, R3, R11
	MOVW      $-(1<<26), R12
	ADD       R11>>26, R12, R12
	BIC       $0xfc000000, R11, R11
	ADD       R12, R4, R9
	MOVW      R9>>31, R12
	SUB       $1, R12
	AND       R12, R6, R6
	AND       R12, R7, R7
	AND       R12, g, g
	AND       R12, R11, R11
	AND       R12, R9, R9
	MVN       R12, R12
	AND       R12, R0, R0
	AND       R12, R1, R1
	AND       R12, R2, R2
	AND       R12, R3, R3
	AND       R12, R4, R4
	ORR       R6, R0, R0
	ORR       R7, R1, R1
	ORR       g, R2, R2
	ORR       R11, R3, R3
	ORR       R9, R4, R4
	ORR       R1<<26, R0, R0
	MOVW      R1>>6, R1
	ORR       R2<<20, R1, R1
	MOVW      R2>>12, R2
	ORR       R3<<14, R2, R2
	MOVW      R3>>18, R3
	ORR       R4<<8, R3, R3
	MOVW      40(R5), R6
	MOVW      44(R5), R7
	MOVW      48(R5), g
	MOVW      52(R5), R11
	ADD.S     R6, R0, R0
	ADC.S     R7, R1, R1
	ADC.S     g, R2, R2
	ADC.S     R11, R3, R3
	MOVM.IA   [R0-R3], (R8)
	MOVW      R5, R12
	EOR       R0, R0, R0
	EOR       R1, R1, R1
	EOR       R2, R2, R2
	EOR       R3, R3, R3
	EOR       R4, R4, R4
	EOR       R5, R5, R5
	EOR       R6, R6, R6
	EOR       R7, R7, R7
	MOVM.IA.W [R0-R7], (R12)
	MOVM.IA   [R0-R7], (R12)
	MOVW      4(R13), g
	RET
