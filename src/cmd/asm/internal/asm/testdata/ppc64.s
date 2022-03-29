// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This contains the majority of valid opcode combinations
// available in cmd/internal/obj/ppc64/asm9.go with
// their valid instruction encodings.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$0
	// move constants
	MOVD $1, R3                     // 38600001
	MOVD $-1, R4                    // 3880ffff
	MOVD $65535, R5                 // 6005ffff
	MOVD $65536, R6                 // 64060001
	MOVD $-32767, R5                // 38a08001
	MOVD $-32768, R6                // 38c08000
	MOVD $1234567, R5               // 6405001260a5d687
	MOVW $1, R3                     // 38600001
	MOVW $-1, R4                    // 3880ffff
	MOVW $65535, R5                 // 6005ffff
	MOVW $65536, R6                 // 64060001
	MOVW $-32767, R5                // 38a08001
	MOVW $-32768, R6                // 38c08000
	MOVW $1234567, R5               // 6405001260a5d687
	MOVD 8(R3), R4			// e8830008
	MOVD (R3)(R4), R5               // 7ca4182a
	MOVW 4(R3), R4                  // e8830006
	MOVW (R3)(R4), R5               // 7ca41aaa
	MOVWZ 4(R3), R4                 // 80830004
	MOVWZ (R3)(R4), R5              // 7ca4182e
	MOVH 4(R3), R4                  // a8830004
	MOVH (R3)(R4), R5               // 7ca41aae
	MOVHZ 2(R3), R4                 // a0830002
	MOVHZ (R3)(R4), R5              // 7ca41a2e
	MOVB 1(R3), R4                  // 888300017c840774
	MOVB (R3)(R4), R5               // 7ca418ae7ca50774
	MOVBZ 1(R3), R4                 // 88830001
	MOVBZ (R3)(R4), R5              // 7ca418ae
	MOVDBR (R3)(R4), R5             // 7ca41c28
	MOVWBR (R3)(R4), R5             // 7ca41c2c
	MOVHBR (R3)(R4), R5             // 7ca41e2c
	MOVD $foo+4009806848(FP), R5    // 3ca1ef0138a5cc20
	MOVD $foo(SB), R5               // 3ca0000038a50000

	MOVDU 8(R3), R4                 // e8830009
	MOVDU (R3)(R4), R5              // 7ca4186a
	MOVWU (R3)(R4), R5              // 7ca41aea
	MOVWZU 4(R3), R4                // 84830004
	MOVWZU (R3)(R4), R5             // 7ca4186e
	MOVHU 2(R3), R4                 // ac830002
	MOVHU (R3)(R4), R5              // 7ca41aee
	MOVHZU 2(R3), R4                // a4830002
	MOVHZU (R3)(R4), R5             // 7ca41a6e
	MOVBU 1(R3), R4                 // 8c8300017c840774
	MOVBU (R3)(R4), R5              // 7ca418ee7ca50774
	MOVBZU 1(R3), R4                // 8c830001
	MOVBZU (R3)(R4), R5             // 7ca418ee

	MOVD R4, 8(R3)                  // f8830008
	MOVD R5, (R3)(R4)               // 7ca4192a
	MOVW R4, 4(R3)                  // 90830004
	MOVW R5, (R3)(R4)               // 7ca4192e
	MOVH R4, 2(R3)                  // b0830002
	MOVH R5, (R3)(R4)               // 7ca41b2e
	MOVB R4, 1(R3)                  // 98830001
	MOVB R5, (R3)(R4)               // 7ca419ae
	MOVDBR R5, (R3)(R4)             // 7ca41d28
	MOVWBR R5, (R3)(R4)             // 7ca41d2c
	MOVHBR R5, (R3)(R4)             // 7ca41f2c

	MOVDU R4, 8(R3)                 // f8830009
	MOVDU R5, (R3)(R4)              // 7ca4196a
	MOVWU R4, 4(R3)                 // 94830004
	MOVWU R5, (R3)(R4)              // 7ca4196e
	MOVHU R4, 2(R3)                 // b4830002
	MOVHU R5, (R3)(R4)              // 7ca41b6e
	MOVBU R4, 1(R3)                 // 9c830001
	MOVBU R5, (R3)(R4)              // 7ca419ee

	MOVB $0, R4			// 38800000
	MOVBZ $0, R4			// 38800000
	MOVH $0, R4			// 38800000
	MOVHZ $0, R4			// 38800000
	MOVW $0, R4			// 38800000
	MOVWZ $0, R4			// 38800000
	MOVD $0, R4			// 38800000
	MOVD $0, R0			// 38000000

	ADD $1, R3                      // 38630001
	ADD $1, R3, R4                  // 38830001
	ADD $-1, R4                     // 3884ffff
	ADD $-1, R4, R5                 // 38a4ffff
	ADD $65535, R5                  // 601fffff7cbf2a14
	ADD $65535, R5, R6              // 601fffff7cdf2a14
	ADD $65536, R6                  // 3cc60001
	ADD $65536, R6, R7              // 3ce60001
	ADD $-32767, R5                 // 38a58001
	ADD $-32767, R5, R4             // 38858001
	ADD $-32768, R6                 // 38c68000
	ADD $-32768, R6, R5             // 38a68000
	ADD $1234567, R5                // 641f001263ffd6877cbf2a14
	ADD $1234567, R5, R6            // 641f001263ffd6877cdf2a14
	ADDEX R3, R5, $3, R6            // 7cc32f54
	ADDIS $8, R3                    // 3c630008
	ADDIS $1000, R3, R4             // 3c8303e8

	ANDCC $1, R3                    // 70630001
	ANDCC $1, R3, R4                // 70640001
	ANDCC $-1, R4                   // 3be0ffff7fe42039
	ANDCC $-1, R4, R5               // 3be0ffff7fe52039
	ANDCC $65535, R5                // 70a5ffff
	ANDCC $65535, R5, R6            // 70a6ffff
	ANDCC $65536, R6                // 74c60001
	ANDCC $65536, R6, R7            // 74c70001
	ANDCC $-32767, R5               // 3be080017fe52839
	ANDCC $-32767, R5, R4           // 3be080017fe42839
	ANDCC $-32768, R6               // 3be080007fe63039
	ANDCC $-32768, R5, R6           // 3be080007fe62839
	ANDCC $1234567, R5              // 641f001263ffd6877fe52839
	ANDCC $1234567, R5, R6          // 641f001263ffd6877fe62839
	ANDISCC $1, R3                  // 74630001
	ANDISCC $1000, R3, R4           // 746403e8

	OR $1, R3                       // 60630001
	OR $1, R3, R4                   // 60640001
	OR $-1, R4                      // 3be0ffff7fe42378
	OR $-1, R4, R5                  // 3be0ffff7fe52378
	OR $65535, R5                   // 60a5ffff
	OR $65535, R5, R6               // 60a6ffff
	OR $65536, R6                   // 64c60001
	OR $65536, R6, R7               // 64c70001
	OR $-32767, R5                  // 3be080017fe52b78
	OR $-32767, R5, R6              // 3be080017fe62b78
	OR $-32768, R6                  // 3be080007fe63378
	OR $-32768, R6, R7              // 3be080007fe73378
	OR $1234567, R5                 // 641f001263ffd6877fe52b78
	OR $1234567, R5, R3             // 641f001263ffd6877fe32b78
	ORIS $255, R3, R4

	XOR $1, R3                      // 68630001
	XOR $1, R3, R4                  // 68640001
	XOR $-1, R4                     // 3be0ffff7fe42278
	XOR $-1, R4, R5                 // 3be0ffff7fe52278
	XOR $65535, R5                  // 68a5ffff
	XOR $65535, R5, R6              // 68a6ffff
	XOR $65536, R6                  // 6cc60001
	XOR $65536, R6, R7              // 6cc70001
	XOR $-32767, R5                 // 3be080017fe52a78
	XOR $-32767, R5, R6             // 3be080017fe62a78
	XOR $-32768, R6                 // 3be080007fe63278
	XOR $-32768, R6, R7             // 3be080007fe73278
	XOR $1234567, R5                // 641f001263ffd6877fe52a78
	XOR $1234567, R5, R3            // 641f001263ffd6877fe32a78
	XORIS $15, R3, R4

	// TODO: the order of CR operands don't match
	CMP R3, R4                      // 7c232000
	CMPU R3, R4                     // 7c232040
	CMPW R3, R4                     // 7c032000
	CMPWU R3, R4                    // 7c032040
	CMPB R3,R4,R4                   // 7c6423f8
	CMPEQB R3,R4,CR6                // 7f0321c0

	// TODO: constants for ADDC?
	ADD R3, R4                      // 7c841a14
	ADD R3, R4, R5                  // 7ca41a14
	ADDC R3, R4                     // 7c841814
	ADDC R3, R4, R5                 // 7ca41814
	ADDE R3, R4                     // 7c841914
	ADDECC R3, R4                   // 7c841915
	ADDEV R3, R4                    // 7c841d14
	ADDEVCC R3, R4                  // 7c841d15
	ADDV R3, R4                     // 7c841e14
	ADDVCC R3, R4                   // 7c841e15
	ADDCCC R3, R4, R5               // 7ca41815
	ADDME R3, R4                    // 7c8301d4
	ADDMECC R3, R4                  // 7c8301d5
	ADDMEV R3, R4                   // 7c8305d4
	ADDMEVCC R3, R4                 // 7c8305d5
	ADDCV R3, R4                    // 7c841c14
	ADDCVCC R3, R4                  // 7c841c15
	ADDZE R3, R4                    // 7c830194
	ADDZECC R3, R4                  // 7c830195
	ADDZEV R3, R4                   // 7c830594
	ADDZEVCC R3, R4                 // 7c830595
	SUBME R3, R4                    // 7c8301d0
	SUBMECC R3, R4                  // 7c8301d1
	SUBMEV R3, R4                   // 7c8305d0
	SUBZE R3, R4                    // 7c830190
	SUBZECC R3, R4                  // 7c830191
	SUBZEV R3, R4                   // 7c830590
	SUBZEVCC R3, R4                 // 7c830591

	AND R3, R4                      // 7c841838
	AND R3, R4, R5                  // 7c851838
	ANDN R3, R4, R5                 // 7c851878
	ANDCC R3, R4, R5                // 7c851839
	OR R3, R4                       // 7c841b78
	OR R3, R4, R5                   // 7c851b78
	ORN R3, R4, R5                  // 7c851b38
	ORCC R3, R4, R5                 // 7c851b79
	XOR R3, R4                      // 7c841a78
	XOR R3, R4, R5                  // 7c851a78
	XORCC R3, R4, R5                // 7c851a79
	NAND R3, R4, R5                 // 7c851bb8
	NANDCC R3, R4, R5               // 7c851bb9
	EQV R3, R4, R5                  // 7c851a38
	EQVCC R3, R4, R5                // 7c851a39
	NOR R3, R4, R5                  // 7c8518f8
	NORCC R3, R4, R5                // 7c8518f9

	SUB R3, R4                      // 7c832050
	SUB R3, R4, R5                  // 7ca32050
	SUBC R3, R4                     // 7c832010
	SUBC R3, R4, R5                 // 7ca32010

	MULLW R3, R4                    // 7c8419d6
	MULLW R3, R4, R5                // 7ca419d6
	MULLW $10, R3                   // 1c63000a
	MULLW $10000000, R3             // 641f009863ff96807c7f19d6

	MULLWCC R3, R4, R5              // 7ca419d7
	MULHW R3, R4, R5                // 7ca41896

	MULHWU R3, R4, R5               // 7ca41816
	MULLD R3, R4                    // 7c8419d2
	MULLD R4, R4, R5                // 7ca421d2
	MULLD $20, R4                   // 1c840014
	MULLD $200000000, R4            // 641f0beb63ffc2007c9f21d2

	MULLDCC R3, R4, R5              // 7ca419d3
	MULHD R3, R4, R5                // 7ca41892
	MULHDCC R3, R4, R5              // 7ca41893

	MULLWV R3, R4                   // 7c841dd6
	MULLWV R3, R4, R5               // 7ca41dd6
	MULLWVCC R3, R4, R5             // 7ca41dd7
	MULHWUCC R3, R4, R5             // 7ca41817
	MULLDV R3, R4, R5               // 7ca41dd2
	MULLDVCC R3, R4, R5             // 7ca41dd3

	DIVD R3,R4                      // 7c841bd2
	DIVD R3, R4, R5                 // 7ca41bd2
	DIVDCC R3,R4, R5                // 7ca41bd3
	DIVDU R3, R4, R5                // 7ca41b92
	DIVDV R3, R4, R5                // 7ca41fd2
	DIVDUCC R3, R4, R5              // 7ca41b93
	DIVDVCC R3, R4, R5              // 7ca41fd3
	DIVDUV R3, R4, R5               // 7ca41f92
	DIVDUVCC R3, R4, R5             // 7ca41f93
	DIVDE R3, R4, R5                // 7ca41b52
	DIVDECC R3, R4, R5              // 7ca41b53
	DIVDEU R3, R4, R5               // 7ca41b12
	DIVDEUCC R3, R4, R5             // 7ca41b13

	REM R3, R4, R5                  // 7fe41bd67fff19d67cbf2050
	REMU R3, R4, R5                 // 7fe41b967fff19d67bff00287cbf2050
	REMD R3, R4, R5                 // 7fe41bd27fff19d27cbf2050
	REMDU R3, R4, R5                // 7fe41b927fff19d27cbf2050

	MADDHD R3,R4,R5,R6              // 10c32170
	MADDHDU R3,R4,R5,R6             // 10c32171

	MODUD R3, R4, R5                // 7ca41a12
	MODUW R3, R4, R5                // 7ca41a16
	MODSD R3, R4, R5                // 7ca41e12
	MODSW R3, R4, R5                // 7ca41e16

	SLW $8, R3, R4                  // 5464402e
	SLW R3, R4, R5                  // 7c851830
	SLWCC R3, R4                    // 7c841831
	SLD $16, R3, R4                 // 786483e4
	SLD R3, R4, R5                  // 7c851836
	SLDCC R3, R4                    // 7c841837

	SRW $8, R3, R4                  // 5464c23e
	SRW R3, R4, R5                  // 7c851c30
	SRWCC R3, R4                    // 7c841c31
	SRAW $8, R3, R4                 // 7c644670
	SRAW R3, R4, R5                 // 7c851e30
	SRAWCC R3, R4                   // 7c841e31
	SRD $16, R3, R4                 // 78648402
	SRD R3, R4, R5                  // 7c851c36
	SRDCC R3, R4                    // 7c841c37
	SRAD $16, R3, R4                // 7c648674
	SRAD R3, R4, R5                 // 7c851e34
	SRDCC R3, R4                    // 7c841c37
	ROTLW $16, R3, R4               // 5464803e
	ROTLW R3, R4, R5                // 5c85183e
	EXTSWSLI $3, R4, R5             // 7c851ef4
	RLWMI $7, R3, $65535, R6        // 50663c3e
	RLWMI $7, R3, $16, $31, R6      // 50663c3e
	RLWMICC $7, R3, $65535, R6      // 50663c3f
	RLWMICC $7, R3, $16, $31, R6    // 50663c3f
	RLWNM $3, R4, $7, R6            // 54861f7e
	RLWNM $3, R4, $29, $31, R6      // 54861f7e
	RLWNM R3, R4, $7, R6            // 5c861f7e
	RLWNM R3, R4, $29, $31, R6      // 5c861f7e
	RLWNMCC $3, R4, $7, R6          // 54861f7f
	RLWNMCC $3, R4, $29, $31, R6    // 54861f7f
	RLWNMCC R3, R4, $7, R6          // 5c861f7f
	RLWNMCC R3, R4, $29, $31, R6    // 5c861f7f
	RLDMI $0, R4, $7, R6            // 7886076c
	RLDMICC $0, R4, $7, R6          // 7886076d
	RLDIMI $0, R4, $7, R6           // 788601cc
	RLDIMICC $0, R4, $7, R6         // 788601cd
	RLDC $0, R4, $15, R6            // 78860728
	RLDCCC $0, R4, $15, R6          // 78860729
	RLDCL $0, R4, $7, R6            // 78860770
	RLDCLCC $0, R4, $15, R6         // 78860721
	RLDCR $0, R4, $-16, R6          // 788606f2
	RLDCRCC $0, R4, $-16, R6        // 788606f3
	RLDICL $0, R4, $15, R6          // 788603c0
	RLDICLCC $0, R4, $15, R6        // 788603c1
	RLDICR $0, R4, $15, R6          // 788603c4
	RLDICRCC $0, R4, $15, R6        // 788603c5
	RLDIC $0, R4, $15, R6           // 788603c8
	RLDICCC $0, R4, $15, R6         // 788603c9
	CLRLSLWI $16, R5, $8, R4        // 54a4422e
	CLRLSLDI $24, R4, $2, R3        // 78831588
	RLDCR	$1, R1, $-16, R1        // 78210ee4
	RLDCRCC	$1, R1, $-16, R1        // 78210ee5

	BEQ 0(PC)                       // 41820000
	BEQ CR1,0(PC)                   // 41860000
	BGE 0(PC)                       // 40800000
	BGE CR2,0(PC)                   // 40880000
	BGT 4(PC)                       // 41810010
	BGT CR3,4(PC)                   // 418d0010
	BLE 0(PC)                       // 40810000
	BLE CR4,0(PC)                   // 40910000
	BLT 0(PC)                       // 41800000
	BLT CR5,0(PC)                   // 41940000
	BNE 0(PC)                       // 40820000
	BLT CR6,0(PC)                   // 41980000
	JMP 8(PC)                       // 48000010

	NOP
	NOP R2
	NOP F2
	NOP $4

	CRAND CR0GT, CR0EQ, CR0SO       // 4c620a02
	CRANDN CR0GT, CR0EQ, CR0SO      // 4c620902
	CREQV CR0GT, CR0EQ, CR0SO       // 4c620a42
	CRNAND CR0GT, CR0EQ, CR0SO      // 4c6209c2
	CRNOR CR0GT, CR0EQ, CR0SO       // 4c620842
	CROR CR0GT, CR0EQ, CR0SO        // 4c620b82
	CRORN CR0GT, CR0EQ, CR0SO       // 4c620b42
	CRXOR CR0GT, CR0EQ, CR0SO       // 4c620982

	ISEL $1, R3, R4, R5             // 7ca3205e
	ISEL $0, R3, R4, R5             // 7ca3201e
	ISEL $2, R3, R4, R5             // 7ca3209e
	ISEL $3, R3, R4, R5             // 7ca320de
	ISEL $4, R3, R4, R5             // 7ca3211e
	POPCNTB R3, R4                  // 7c6400f4
	POPCNTW R3, R4                  // 7c6402f4
	POPCNTD R3, R4                  // 7c6403f4

	PASTECC R3, R4                  // 7c23270d
	COPY R3, R4                     // 7c23260c

	// load-and-reserve
	LBAR (R4)(R3*1),$1,R5           // 7ca32069
	LBAR (R4),$0,R5                 // 7ca02068
	LBAR (R3),R5                    // 7ca01868
	LHAR (R4)(R3*1),$1,R5           // 7ca320e9
	LHAR (R4),$0,R5                 // 7ca020e8
	LHAR (R3),R5                    // 7ca018e8
	LWAR (R4)(R3*1),$1,R5           // 7ca32029
	LWAR (R4),$0,R5                 // 7ca02028
	LWAR (R3),R5                    // 7ca01828
	LDAR (R4)(R3*1),$1,R5           // 7ca320a9
	LDAR (R4),$0,R5                 // 7ca020a8
	LDAR (R3),R5                    // 7ca018a8

	STBCCC R3, (R4)(R5)             // 7c65256d
	STWCCC R3, (R4)(R5)             // 7c65212d
	STDCCC R3, (R4)(R5)             // 7c6521ad
	STHCCC R3, (R4)(R5)
	STSW R3, (R4)(R5)

	SYNC                            // 7c0004ac
	ISYNC                           // 4c00012c
	LWSYNC                          // 7c2004ac

	DARN $1, R5                     // 7ca105e6

	DCBF (R3)(R4)                   // 7c0418ac
	DCBI (R3)(R4)                   // 7c041bac
	DCBST (R3)(R4)                  // 7c04186c
	DCBZ (R3)(R4)                   // 7c041fec
	DCBT (R3)(R4)                   // 7c041a2c
	ICBI (R3)(R4)                   // 7c041fac

	// float constants
	FMOVD $(0.0), F1                // f0210cd0
	FMOVD $(-0.0), F1               // f0210cd0fc200850

	FMOVD 8(R3), F1                 // c8230008
	FMOVD (R3)(R4), F1              // 7c241cae
	FMOVDU 8(R3), F1                // cc230008
	FMOVDU (R3)(R4), F1             // 7c241cee
	FMOVS 4(R3), F1                 // c0230004
	FMOVS (R3)(R4), F1              // 7c241c2e
	FMOVSU 4(R3), F1                // c4230004
	FMOVSU (R3)(R4), F1             // 7c241c6e

	FMOVD F1, 8(R3)                 // d8230008
	FMOVD F1, (R3)(R4)              // 7c241dae
	FMOVDU F1, 8(R3)                // dc230008
	FMOVDU F1, (R3)(R4)             // 7c241dee
	FMOVS F1, 4(R3)                 // d0230004
	FMOVS F1, (R3)(R4)              // 7c241d2e
	FMOVSU F1, 4(R3)                // d4230004
	FMOVSU F1, (R3)(R4)             // 7c241d6e
	FADD F1, F2                     // fc42082a
	FADD F1, F2, F3                 // fc62082a
	FADDCC F1, F2, F3               // fc62082b
	FADDS F1, F2                    // ec42082a
	FADDS F1, F2, F3                // ec62082a
	FADDSCC F1, F2, F3              // ec62082b
	FSUB F1, F2                     // fc420828
	FSUB F1, F2, F3                 // fc620828
	FSUBCC F1, F2, F3               // fc620829
	FSUBS F1, F2                    // ec420828
	FSUBS F1, F2, F3                // ec620828
	FSUBCC F1, F2, F3               // fc620829
	FMUL F1, F2                     // fc420072
	FMUL F1, F2, F3                 // fc620072
	FMULCC F1, F2, F3               // fc620073
	FMULS F1, F2                    // ec420072
	FMULS F1, F2, F3                // ec620072
	FMULSCC F1, F2, F3              // ec620073
	FDIV F1, F2                     // fc420824
	FDIV F1, F2, F3                 // fc620824
	FDIVCC F1, F2, F3               // fc620825
	FDIVS F1, F2                    // ec420824
	FDIVS F1, F2, F3                // ec620824
	FDIVSCC F1, F2, F3              // ec620825
	FMADD F1, F2, F3, F4            // fc8110fa
	FMADDCC F1, F2, F3, F4          // fc8110fb
	FMADDS F1, F2, F3, F4           // ec8110fa
	FMADDSCC F1, F2, F3, F4         // ec8110fb
	FMSUB F1, F2, F3, F4            // fc8110f8
	FMSUBCC F1, F2, F3, F4          // fc8110f9
	FMSUBS F1, F2, F3, F4           // ec8110f8
	FMSUBSCC F1, F2, F3, F4         // ec8110f9
	FNMADD F1, F2, F3, F4           // fc8110fe
	FNMADDCC F1, F2, F3, F4         // fc8110ff
	FNMADDS F1, F2, F3, F4          // ec8110fe
	FNMADDSCC F1, F2, F3, F4        // ec8110ff
	FNMSUB F1, F2, F3, F4           // fc8110fc
	FNMSUBCC F1, F2, F3, F4         // fc8110fd
	FNMSUBS F1, F2, F3, F4          // ec8110fc
	FNMSUBSCC F1, F2, F3, F4        // ec8110fd
	FSEL F1, F2, F3, F4             // fc8110ee
	FSELCC F1, F2, F3, F4           // fc8110ef
	FABS F1, F2                     // fc400a10
	FABSCC F1, F2                   // fc400a11
	FNEG F1, F2                     // fc400850
	FABSCC F1, F2                   // fc400a11
	FRSP F1, F2                     // fc400818
	FRSPCC F1, F2                   // fc400819
	FCTIW F1, F2                    // fc40081c
	FCTIWCC F1, F2                  // fc40081d
	FCTIWZ F1, F2                   // fc40081e
	FCTIWZCC F1, F2                 // fc40081f
	FCTID F1, F2                    // fc400e5c
	FCTIDCC F1, F2                  // fc400e5d
	FCTIDZ F1, F2                   // fc400e5e
	FCTIDZCC F1, F2                 // fc400e5f
	FCFID F1, F2                    // fc400e9c
	FCFIDCC F1, F2                  // fc400e9d
	FCFIDU F1, F2                   // fc400f9c
	FCFIDUCC F1, F2                 // fc400f9d
	FCFIDS F1, F2                   // ec400e9c
	FCFIDSCC F1, F2                 // ec400e9d
	FRES F1, F2                     // ec400830
	FRESCC F1, F2                   // ec400831
	FRIM F1, F2                     // fc400bd0
	FRIMCC F1, F2                   // fc400bd1
	FRIP F1, F2                     // fc400b90
	FRIPCC F1, F2                   // fc400b91
	FRIZ F1, F2                     // fc400b50
	FRIZCC F1, F2                   // fc400b51
	FRIN F1, F2                     // fc400b10
	FRINCC F1, F2                   // fc400b11
	FRSQRTE F1, F2                  // fc400834
	FRSQRTECC F1, F2                // fc400835
	FSQRT F1, F2                    // fc40082c
	FSQRTCC F1, F2                  // fc40082d
	FSQRTS F1, F2                   // ec40082c
	FSQRTSCC F1, F2                 // ec40082d
	FCPSGN F1, F2                   // fc420810
	FCPSGNCC F1, F2                 // fc420811
	FCMPO F1, F2                    // fc011040
	FCMPU F1, F2                    // fc011000
	LVX (R3)(R4), V1                // 7c2418ce
	LVXL (R3)(R4), V1               // 7c241ace
	LVSL (R3)(R4), V1               // 7c24180c
	LVSR (R3)(R4), V1               // 7c24184c
	LVEBX (R3)(R4), V1              // 7c24180e
	LVEHX (R3)(R4), V1              // 7c24184e
	LVEWX (R3)(R4), V1              // 7c24188e
	STVX V1, (R3)(R4)               // 7c2419ce
	STVXL V1, (R3)(R4)              // 7c241bce
	STVEBX V1, (R3)(R4)             // 7c24190e
	STVEHX V1, (R3)(R4)             // 7c24194e
	STVEWX V1, (R3)(R4)             // 7c24198e

	VAND V1, V2, V3                 // 10611404
	VANDC V1, V2, V3                // 10611444
	VNAND V1, V2, V3                // 10611584
	VOR V1, V2, V3                  // 10611484
	VORC V1, V2, V3                 // 10611544
	VXOR V1, V2, V3                 // 106114c4
	VNOR V1, V2, V3                 // 10611504
	VEQV V1, V2, V3                 // 10611684
	VADDUBM V1, V2, V3              // 10611000
	VADDUHM V1, V2, V3              // 10611040
	VADDUWM V1, V2, V3              // 10611080
	VADDUDM V1, V2, V3              // 106110c0
	VADDUQM V1, V2, V3              // 10611100
	VADDCUQ V1, V2, V3              // 10611140
	VADDCUW V1, V2, V3              // 10611180
	VADDUBS V1, V2, V3              // 10611200
	VADDUHS V1, V2, V3              // 10611240
	VADDUWS V1, V2, V3              // 10611280
	VSUBUBM V1, V2, V3              // 10611400
	VSUBUHM V1, V2, V3              // 10611440
	VSUBUWM V1, V2, V3              // 10611480
	VSUBUDM V1, V2, V3              // 106114c0
	VSUBUQM V1, V2, V3              // 10611500
	VSUBCUQ V1, V2, V3              // 10611540
	VSUBCUW V1, V2, V3              // 10611580
	VSUBUBS V1, V2, V3              // 10611600
	VSUBUHS V1, V2, V3              // 10611640
	VSUBUWS V1, V2, V3              // 10611680
	VSUBSBS V1, V2, V3              // 10611700
	VSUBSHS V1, V2, V3              // 10611740
	VSUBSWS V1, V2, V3              // 10611780
	VSUBEUQM V1, V2, V3, V4         // 108110fe
	VSUBECUQ V1, V2, V3, V4         // 108110ff
	VMULESB V1, V2, V3              // 10611308
	VMULOSB V1, V2, V3              // 10611108
	VMULEUB V1, V2, V3              // 10611208
	VMULOUB V1, V2, V3              // 10611008
	VMULESH V1, V2, V3              // 10611348
	VMULOSH V1, V2, V3              // 10611148
	VMULEUH V1, V2, V3              // 10611248
	VMULOUH V1, V2, V3              // 10611048
	VMULESH V1, V2, V3              // 10611348
	VMULOSW V1, V2, V3              // 10611188
	VMULEUW V1, V2, V3              // 10611288
	VMULOUW V1, V2, V3              // 10611088
	VMULUWM V1, V2, V3              // 10611089
	VPMSUMB V1, V2, V3              // 10611408
	VPMSUMH V1, V2, V3              // 10611448
	VPMSUMW V1, V2, V3              // 10611488
	VPMSUMD V1, V2, V3              // 106114c8
	VMSUMUDM V1, V2, V3, V4         // 108110e3
	VRLB V1, V2, V3                 // 10611004
	VRLH V1, V2, V3                 // 10611044
	VRLW V1, V2, V3                 // 10611084
	VRLD V1, V2, V3                 // 106110c4
	VSLB V1, V2, V3                 // 10611104
	VSLH V1, V2, V3                 // 10611144
	VSLW V1, V2, V3                 // 10611184
	VSL V1, V2, V3                  // 106111c4
	VSLO V1, V2, V3                 // 1061140c
	VSRB V1, V2, V3                 // 10611204
	VSRH V1, V2, V3                 // 10611244
	VSRW V1, V2, V3                 // 10611284
	VSR V1, V2, V3                  // 106112c4
	VSRO V1, V2, V3                 // 1061144c
	VSLD V1, V2, V3                 // 106115c4
	VSRAB V1, V2, V3                // 10611304
	VSRAH V1, V2, V3                // 10611344
	VSRAW V1, V2, V3                // 10611384
	VSRAD V1, V2, V3                // 106113c4
	VSLDOI $3, V1, V2, V3           // 106110ec
	VCLZB V1, V2                    // 10400f02
	VCLZH V1, V2                    // 10400f42
	VCLZW V1, V2                    // 10400f82
	VCLZD V1, V2                    // 10400fc2
	VPOPCNTB V1, V2                 // 10400f03
	VPOPCNTH V1, V2                 // 10400f43
	VPOPCNTW V1, V2                 // 10400f83
	VPOPCNTD V1, V2                 // 10400fc3
	VCMPEQUB V1, V2, V3             // 10611006
	VCMPEQUBCC V1, V2, V3           // 10611406
	VCMPEQUH V1, V2, V3             // 10611046
	VCMPEQUHCC V1, V2, V3           // 10611446
	VCMPEQUW V1, V2, V3             // 10611086
	VCMPEQUWCC V1, V2, V3           // 10611486
	VCMPEQUD V1, V2, V3             // 106110c7
	VCMPEQUDCC V1, V2, V3           // 106114c7
	VCMPGTUB V1, V2, V3             // 10611206
	VCMPGTUBCC V1, V2, V3           // 10611606
	VCMPGTUH V1, V2, V3             // 10611246
	VCMPGTUHCC V1, V2, V3           // 10611646
	VCMPGTUW V1, V2, V3             // 10611286
	VCMPGTUWCC V1, V2, V3           // 10611686
	VCMPGTUD V1, V2, V3             // 106112c7
	VCMPGTUDCC V1, V2, V3           // 106116c7
	VCMPGTSB V1, V2, V3             // 10611306
	VCMPGTSBCC V1, V2, V3           // 10611706
	VCMPGTSH V1, V2, V3             // 10611346
	VCMPGTSHCC V1, V2, V3           // 10611746
	VCMPGTSW V1, V2, V3             // 10611386
	VCMPGTSWCC V1, V2, V3           // 10611786
	VCMPGTSD V1, V2, V3             // 106113c7
	VCMPGTSDCC V1, V2, V3           // 106117c7
	VCMPNEZB V1, V2, V3             // 10611107
	VCMPNEZBCC V1, V2, V3           // 10611507
	VCMPNEB V1, V2, V3              // 10611007
	VCMPNEBCC V1, V2, V3            // 10611407
	VCMPNEH V1, V2, V3              // 10611047
	VCMPNEHCC V1, V2, V3            // 10611447
	VCMPNEW V1, V2, V3              // 10611087
	VCMPNEWCC V1, V2, V3            // 10611487
	VPERM V1, V2, V3, V4            // 108110eb
	VPERMR V1, V2, V3, V4           // 108110fb
	VPERMXOR V1, V2, V3, V4         // 108110ed
	VBPERMQ V1, V2, V3              // 1061154c
	VBPERMD V1, V2, V3              // 106115cc
	VSEL V1, V2, V3, V4             // 108110ea
	VSPLTB $1, V1, V2               // 10410a0c
	VSPLTH $1, V1, V2               // 10410a4c
	VSPLTW $1, V1, V2               // 10410a8c
	VSPLTISB $1, V1                 // 1021030c
	VSPLTISW $1, V1                 // 1021038c
	VSPLTISH $1, V1                 // 1021034c
	VCIPHER V1, V2, V3              // 10611508
	VCIPHERLAST V1, V2, V3          // 10611509
	VNCIPHER V1, V2, V3             // 10611548
	VNCIPHERLAST V1, V2, V3         // 10611549
	VSBOX V1, V2                    // 104105c8
	VSHASIGMAW $1, V1, $15, V2      // 10418e82
	VSHASIGMAD $2, V1, $15, V2      // 104196c2

	LXVD2X (R3)(R4), VS1            // 7c241e98
	LXVDSX (R3)(R4), VS1            // 7c241a98
	LXVH8X (R3)(R4), VS1            // 7c241e58
	LXVB16X (R3)(R4), VS1           // 7c241ed8
	LXVW4X (R3)(R4), VS1            // 7c241e18
	LXV 16(R3), VS1                 // f4230011
	LXV 16(R3), VS33                // f4230019
	LXV 16(R3), V1                  // f4230019
	LXVL R3, R4, VS1                // 7c23221a
	LXVLL R3, R4, VS1               // 7c23225a
	LXVX R3, R4, VS1                // 7c232218
	LXSDX (R3)(R4), VS1             // 7c241c98
	STXVD2X VS1, (R3)(R4)           // 7c241f98
	STXV VS1,16(R3)                 // f4230015
	STXVL VS1, R3, R4               // 7c23231a
	STXVLL VS1, R3, R4              // 7c23235a
	STXVX VS1, R3, R4               // 7c232318
	STXVB16X VS1, (R4)(R5)          // 7c2527d8
	STXVH8X VS1, (R4)(R5)           // 7c252758

	STXSDX VS1, (R3)(R4)            // 7c241d98
	LXSIWAX (R3)(R4), VS1           // 7c241898
	STXSIWX VS1, (R3)(R4)           // 7c241918
	MFVSRD VS1, R3                  // 7c230066
	MTFPRD R3, F0                   // 7c030166
	MFVRD V0, R3                    // 7c030067
	MFVSRLD VS63,R4                 // 7fe40267
	MFVSRLD V31,R4                  // 7fe40267
	MFVSRWZ VS33,R4                 // 7c2400e7
	MFVSRWZ V1,R4                   // 7c2400e7
	MTVSRD R3, VS1                  // 7c230166
	MTVSRDD R3, R4, VS1             // 7c232366
	MTVSRDD R3, R4, VS33            // 7c232367
	MTVSRDD R3, R4, V1              // 7c232367
	MTVRD R3, V13                   // 7da30167
	MTVSRWA R4, VS31                // 7fe401a6
	MTVSRWS R4, VS32                // 7c040327
	MTVSRWZ R4, VS63                // 7fe401e7
	XXBRD VS0, VS1                  // f037076c
	XXBRW VS1, VS2                  // f04f0f6c
	XXBRH VS2, VS3                  // f067176c
	XXLAND VS1, VS2, VS3            // f0611410
	XXLAND V1, V2, V3               // f0611417
	XXLAND VS33, VS34, VS35         // f0611417
	XXLANDC VS1, VS2, VS3           // f0611450
	XXLEQV VS0, VS1, VS2            // f0400dd0
	XXLNAND VS0, VS1, VS2           // f0400d90
	XXLNOR VS0, VS1, VS32           // f0000d11
	XXLOR VS1, VS2, VS3             // f0611490
	XXLORC VS1, VS2, VS3            // f0611550
	XXLORQ VS1, VS2, VS3            // f0611490
	XXLXOR VS1, VS2, VS3            // f06114d0
	XXSEL VS1, VS2, VS3, VS4        // f08110f0
	XXSEL VS33, VS34, VS35, VS36    // f08110ff
	XXSEL V1, V2, V3, V4            // f08110ff
	XXMRGHW VS1, VS2, VS3           // f0611090
	XXMRGLW VS1, VS2, VS3           // f0611190
	XXSPLTW VS1, $1, VS2            // f0410a90
	XXSPLTW VS33, $1, VS34          // f0410a93
	XXSPLTW V1, $1, V2              // f0410a93
	XXPERM VS1, VS2, VS3            // f06110d0
	XXSLDWI VS1, VS2, $1, VS3       // f0611110
	XXSLDWI V1, V2, $1, V3          // f0611117
	XXSLDWI VS33, VS34, $1, VS35    // f0611117
	XSCVDPSP VS1, VS2               // f0400c24
	XVCVDPSP VS1, VS2               // f0400e24
	XSCVSXDDP VS1, VS2              // f0400de0
	XVCVDPSXDS VS1, VS2             // f0400f60
	XVCVSXDDP VS1, VS2              // f0400fe0
	XSCVDPSPN   VS1,VS32            // f0000c2d
	XSCVDPSP    VS1,VS32            // f0000c25
	XSCVDPSXDS  VS1,VS32            // f0000d61
	XSCVDPSXWS  VS1,VS32            // f0000961
	XSCVDPUXDS  VS1,VS32            // f0000d21
	XSCVDPUXWS  VS1,VS32            // f0000921
	XSCVSPDPN   VS1,VS32            // f0000d2d
	XSCVSPDP    VS1,VS32            // f0000d25
	XSCVSXDDP   VS1,VS32            // f0000de1
	XSCVSXDSP   VS1,VS32            // f0000ce1
	XSCVUXDDP   VS1,VS32            // f0000da1
	XSCVUXDSP   VS1,VS32            // f0000ca1
	XVCVDPSP    VS1,VS32            // f0000e25
	XVCVDPSXDS  VS1,VS32            // f0000f61
	XVCVDPSXWS  VS1,VS32            // f0000b61
	XVCVDPUXDS  VS1,VS32            // f0000f21
	XVCVDPUXWS  VS1,VS32            // f0000b21
	XVCVSPDP    VS1,VS32            // f0000f25
	XVCVSPSXDS  VS1,VS32            // f0000e61
	XVCVSPSXWS  VS1,VS32            // f0000a61
	XVCVSPUXDS  VS1,VS32            // f0000e21
	XVCVSPUXWS  VS1,VS32            // f0000a21
	XVCVSXDDP   VS1,VS32            // f0000fe1
	XVCVSXDSP   VS1,VS32            // f0000ee1
	XVCVSXWDP   VS1,VS32            // f0000be1
	XVCVSXWSP   VS1,VS32            // f0000ae1
	XVCVUXDDP   VS1,VS32            // f0000fa1
	XVCVUXDSP   VS1,VS32            // f0000ea1
	XVCVUXWDP   VS1,VS32            // f0000ba1
	XVCVUXWSP   VS1,VS32            // f0000aa1

	MOVD R3, LR                     // 7c6803a6
	MOVD R3, CTR                    // 7c6903a6
	MOVD R3, XER                    // 7c6103a6
	MOVD LR, R3                     // 7c6802a6
	MOVD CTR, R3                    // 7c6902a6
	MOVD XER, R3                    // 7c6102a6
	MOVFL CR3, CR1                  // 4c8c0000

	MOVW CR0, R1                    // 7c380026
	MOVW CR7, R1                    // 7c301026
	MOVW CR, R1                     // 7c200026

	MOVW R1, CR                     // 7c2ff120
	MOVFL R1, CR                    // 7c2ff120
	MOVW R1, CR2                    // 7c320120
	MOVFL R1, CR2                   // 7c320120
	MOVFL R1, $255                  // 7c2ff120
	MOVFL R1, $1                    // 7c301120
	MOVFL R1, $128                  // 7c380120
	MOVFL R1, $3                    // 7c203120

	// Verify supported bdnz/bdz encodings.
	BC 16,0,0(PC)                   // BC $16,R0,0(PC) // 42000000
	BDNZ 0(PC)                      // 42000000
	BDZ 0(PC)                       // 42400000
	BC 18,0,0(PC)                   // BC $18,R0,0(PC) // 42400000

	RET
