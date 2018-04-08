// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial coefficients and other constants
DATA ·cbrtrodataL9<> + 0(SB)/8, $-.00016272731015974436E+00
DATA ·cbrtrodataL9<> + 8(SB)/8, $0.66639548758285293179E+00
DATA ·cbrtrodataL9<> + 16(SB)/8, $0.55519402697349815993E+00
DATA ·cbrtrodataL9<> + 24(SB)/8, $0.49338566048766782004E+00
DATA ·cbrtrodataL9<> + 32(SB)/8, $0.45208160036325611486E+00
DATA ·cbrtrodataL9<> + 40(SB)/8, $0.43099892837778637816E+00
DATA ·cbrtrodataL9<> + 48(SB)/8, $1.000244140625
DATA ·cbrtrodataL9<> + 56(SB)/8, $0.33333333333333333333E+00
DATA ·cbrtrodataL9<> + 64(SB)/8, $79228162514264337593543950336.
GLOBL ·cbrtrodataL9<> + 0(SB), RODATA, $72

// Index tables
DATA ·cbrttab32069<> + 0(SB)/8, $0x404030303020202
DATA ·cbrttab32069<> + 8(SB)/8, $0x101010101000000
DATA ·cbrttab32069<> + 16(SB)/8, $0x808070706060605
DATA ·cbrttab32069<> + 24(SB)/8, $0x505040404040303
DATA ·cbrttab32069<> + 32(SB)/8, $0xe0d0c0c0b0b0b0a
DATA ·cbrttab32069<> + 40(SB)/8, $0xa09090908080808
DATA ·cbrttab32069<> + 48(SB)/8, $0x11111010100f0f0f
DATA ·cbrttab32069<> + 56(SB)/8, $0xe0e0e0e0e0d0d0d
DATA ·cbrttab32069<> + 64(SB)/8, $0x1515141413131312
DATA ·cbrttab32069<> + 72(SB)/8, $0x1212111111111010
GLOBL ·cbrttab32069<> + 0(SB), RODATA, $80

DATA ·cbrttab22068<> + 0(SB)/8, $0x151015001420141
DATA ·cbrttab22068<> + 8(SB)/8, $0x140013201310130
DATA ·cbrttab22068<> + 16(SB)/8, $0x122012101200112
DATA ·cbrttab22068<> + 24(SB)/8, $0x111011001020101
DATA ·cbrttab22068<> + 32(SB)/8, $0x10000f200f100f0
DATA ·cbrttab22068<> + 40(SB)/8, $0xe200e100e000d2
DATA ·cbrttab22068<> + 48(SB)/8, $0xd100d000c200c1
DATA ·cbrttab22068<> + 56(SB)/8, $0xc000b200b100b0
DATA ·cbrttab22068<> + 64(SB)/8, $0xa200a100a00092
DATA ·cbrttab22068<> + 72(SB)/8, $0x91009000820081
DATA ·cbrttab22068<> + 80(SB)/8, $0x80007200710070
DATA ·cbrttab22068<> + 88(SB)/8, $0x62006100600052
DATA ·cbrttab22068<> + 96(SB)/8, $0x51005000420041
DATA ·cbrttab22068<> + 104(SB)/8, $0x40003200310030
DATA ·cbrttab22068<> + 112(SB)/8, $0x22002100200012
DATA ·cbrttab22068<> + 120(SB)/8, $0x11001000020001
GLOBL ·cbrttab22068<> + 0(SB), RODATA, $128

DATA ·cbrttab12067<> + 0(SB)/8, $0x53e1529051324fe1
DATA ·cbrttab12067<> + 8(SB)/8, $0x4e904d324be14a90
DATA ·cbrttab12067<> + 16(SB)/8, $0x493247e146904532
DATA ·cbrttab12067<> + 24(SB)/8, $0x43e1429041323fe1
DATA ·cbrttab12067<> + 32(SB)/8, $0x3e903d323be13a90
DATA ·cbrttab12067<> + 40(SB)/8, $0x393237e136903532
DATA ·cbrttab12067<> + 48(SB)/8, $0x33e1329031322fe1
DATA ·cbrttab12067<> + 56(SB)/8, $0x2e902d322be12a90
DATA ·cbrttab12067<> + 64(SB)/8, $0xd3e1d290d132cfe1
DATA ·cbrttab12067<> + 72(SB)/8, $0xce90cd32cbe1ca90
DATA ·cbrttab12067<> + 80(SB)/8, $0xc932c7e1c690c532
DATA ·cbrttab12067<> + 88(SB)/8, $0xc3e1c290c132bfe1
DATA ·cbrttab12067<> + 96(SB)/8, $0xbe90bd32bbe1ba90
DATA ·cbrttab12067<> + 104(SB)/8, $0xb932b7e1b690b532
DATA ·cbrttab12067<> + 112(SB)/8, $0xb3e1b290b132afe1
DATA ·cbrttab12067<> + 120(SB)/8, $0xae90ad32abe1aa90
GLOBL ·cbrttab12067<> + 0(SB), RODATA, $128

// Cbrt returns the cube root of the argument.
//
// Special cases are:
//      Cbrt(±0) = ±0
//      Cbrt(±Inf) = ±Inf
//      Cbrt(NaN) = NaN
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT	·cbrtAsm(SB), NOSPLIT, $0-16
	FMOVD	x+0(FP), F0
	MOVD	$·cbrtrodataL9<>+0(SB), R9
	WORD	$0xB3CD0020	//lgdr %r2, %f0
	WORD	$0xC039000F	//iilf	%r3,1048575
	BYTE	$0xFF
	BYTE	$0xFF
	SRAD	$32, R2
	WORD	$0xB9170012	//llgtr	%r1,%r2
	MOVW	R1, R6
	MOVW	R3, R7
	CMPBLE	R6, R7, L2
	WORD	$0xC0397FEF	//iilf	%r3,2146435071
	BYTE	$0xFF
	BYTE	$0xFF
	MOVW	R3, R7
	CMPBLE	R6, R7, L8
L1:
	FMOVD	F0, ret+8(FP)
	RET
L3:
L2:
	LTDBR	F0, F0
	BEQ	L1
	FMOVD	F0, F2
	WORD	$0xED209040	//mdb	%f2,.L10-.L9(%r9)
	BYTE	$0x00
	BYTE	$0x1C
	MOVH	$0x200, R4
	WORD	$0xB3CD0022	//lgdr %r2, %f2
	SRAD	$32, R2
L4:
	WORD	$0xEC3239BE	//risbg	%r3,%r2,57,128+62,64-25
	BYTE	$0x27
	BYTE	$0x55
	MOVD	$·cbrttab12067<>+0(SB), R1
	WORD	$0x48131000	//lh	%r1,0(%r3,%r1)
	WORD	$0xEC3239BE	//risbg	%r3,%r2,57,128+62,64-19
	BYTE	$0x2D
	BYTE	$0x55
	MOVD	$·cbrttab22068<>+0(SB), R5
	WORD	$0xEC223CBF	//risbgn	%r2,%r2,64-4,128+63,64+44+4
	BYTE	$0x70
	BYTE	$0x59
	WORD	$0x4A135000	//ah	%r1,0(%r3,%r5)
	BYTE	$0x18	//lr	%r3,%r1
	BYTE	$0x31
	MOVD	$·cbrttab32069<>+0(SB), R1
	FMOVD	56(R9), F1
	FMOVD	48(R9), F5
	WORD	$0xEC23393B	//rosbg	%r2,%r3,57,59,4
	BYTE	$0x04
	BYTE	$0x56
	WORD	$0xE3121000	//llc	%r1,0(%r2,%r1)
	BYTE	$0x00
	BYTE	$0x94
	ADDW	R3, R1
	ADDW	R4, R1
	SLW	$16, R1, R1
	SLD	$32, R1, R1
	WORD	$0xB3C10021	//ldgr	%f2,%r1
	WFMDB	V2, V2, V4
	WFMDB	V4, V0, V6
	WFMSDB	V4, V6, V2, V4
	FMOVD	40(R9), F6
	FMSUB	F1, F4, F2
	FMOVD	32(R9), F4
	WFMDB	V2, V2, V3
	FMOVD	24(R9), F1
	FMUL	F3, F0
	FMOVD	16(R9), F3
	WFMADB	V2, V0, V5, V2
	FMOVD	8(R9), F5
	FMADD	F6, F2, F4
	WFMADB	V2, V1, V3, V1
	WFMDB	V2, V2, V6
	FMOVD	0(R9), F3
	WFMADB	V4, V6, V1, V4
	WFMADB	V2, V5, V3, V2
	FMADD	F4, F6, F2
	FMADD	F2, F0, F0
	FMOVD	F0, ret+8(FP)
	RET
L8:
	MOVH	$0x0, R4
	BR	L4
