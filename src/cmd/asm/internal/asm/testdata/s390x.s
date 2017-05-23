// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT main·foo(SB),7,$16-0 // TEXT main.foo(SB), 7, $16-0
	MOVD	R1, R2                // b9040021
	MOVW	R3, R4                // b9140043
	MOVH	R5, R6                // b9070065
	MOVB	R7, R8                // b9060087
	MOVWZ	R1, R2                // b9160021
	MOVHZ	R2, R3                // b9850032
	MOVBZ	R4, R5                // b9840054
	MOVDBR	R1, R2                // b90f0021
	MOVWBR	R3, R4                // b91f0043

	MOVD	(R15), R1             // e310f0000004
	MOVW	(R15), R2             // e320f0000014
	MOVH	(R15), R3             // e330f0000015
	MOVB	(R15), R4             // e340f0000077
	MOVWZ	(R15), R5             // e350f0000016
	MOVHZ	(R15), R6             // e360f0000091
	MOVBZ	(R15), R7             // e370f0000090
	MOVDBR	(R15), R8             // e380f000000f
	MOVWBR	(R15), R9             // e390f000001e

	MOVD	R1, n-8(SP)           // e310f0100024
	MOVW	R2, n-8(SP)           // e320f0100050
	MOVH	R3, n-8(SP)           // e330f0100070
	MOVB	R4, n-8(SP)           // e340f0100072
	MOVWZ	R5, n-8(SP)           // e350f0100050
	MOVHZ	R6, n-8(SP)           // e360f0100070
	MOVBZ	R7, n-8(SP)           // e370f0100072
	MOVDBR	R8, n-8(SP)           // e380f010002f
	MOVWBR	R9, n-8(SP)           // e390f010003e

	MOVD	$-8589934592, R1      // c01efffffffe
	MOVW	$-131072, R2          // c021fffe0000
	MOVH	$-512, R3             // a739fe00
	MOVB	$-1, R4               // a749ffff

	MOVD	$-2147483648, n-8(SP) // c0b180000000e3b0f0100024
	MOVW	$-131072, n-8(SP)     // c0b1fffe0000e3b0f0100050
	MOVH	$-512, n-8(SP)        // e544f010fe00
	MOVB	$-1, n-8(SP)          // 92fff010

	ADD	R1, R2                // b9e81022
	ADD	R1, R2, R3            // b9e81032
	ADD	$8192, R1             // c21800002000
	ADD	$8192, R1, R2         // ec21200000d9
	ADDC	R1, R2                // b9ea1022
	ADDC	$1, R1, R2            // b9040021c22a00000001
	ADDC	R1, R2, R3            // b9ea1032
	SUB	R3, R4                // b9090043
	SUB	R3, R4, R5            // b9e93054
	SUB	$8192, R3             // c238ffffe000
	SUB	$8192, R3, R4         // ec43e00000d9
	SUBC	R1, R2                // b90b0021
	SUBC	$1, R1, R2            // b9040021c22affffffff
	SUBC	R2, R3, R4            // b9eb2043
	MULLW	R6, R7                // b91c0076
	MULLW	R6, R7, R8            // b9040087b91c0086
	MULLW	$8192, R6             // a76d2000
	MULLW	$8192, R6, R7         // b9040076a77d2000
	MULLW	$-65537, R8           // c280fffeffff
	MULLW   $-65537, R8, R9       // b9040098c290fffeffff
	MULLD	$-2147483648, R1      // c21080000000
	MULLD   $-2147483648, R1, R2  // b9040021c22080000000
	MULHD	R9, R8                // b90400b8b98600a9ebb9003f000ab98000b8b90900abebb8003f000ab98000b9b9e9b08a
	MULHD	R7, R2, R1            // b90400b2b98600a7ebb7003f000ab98000b2b90900abebb2003f000ab98000b7b9e9b01a
	MULHDU	R3, R4                // b90400b4b98600a3b904004a
	MULHDU	R5, R6, R7            // b90400b6b98600a5b904007a
	DIVD	R1, R2                // b90400b2b90d00a1b904002b
	DIVD	R1, R2, R3            // b90400b2b90d00a1b904003b
	DIVW	R4, R5                // b90400b5b91d00a4b904005b
	DIVW	R4, R5, R6            // b90400b5b91d00a4b904006b
	DIVDU	R7, R8                // b90400a0b90400b8b98700a7b904008b
	DIVDU	R7, R8, R9            // b90400a0b90400b8b98700a7b904009b
	DIVWU	R1, R2                // b90400a0b90400b2b99700a1b904002b
	DIVWU	R1, R2, R3            // b90400a0b90400b2b99700a1b904003b

	XC	$8, (R15), n-8(SP)       // XC  (R15), $8, n-8(SP)       // d707f010f000
	NC	$8, (R15), n-8(SP)       // NC  (R15), $8, n-8(SP)       // d407f010f000
	OC	$8, (R15), n-8(SP)       // OC  (R15), $8, n-8(SP)       // d607f010f000
	MVC	$8, (R15), n-8(SP)       // MVC (R15), $8, n-8(SP)       // d207f010f000
	CLC	$8, (R15), n-8(SP)       // CLC (R15), $8, n-8(SP)       // d507f000f010
	XC	$256, -8(R15), -8(R15)   // XC  -8(R15), $256, -8(R15)   // b90400afc2a8fffffff8d7ffa000a000
	MVC	$256, 8192(R1), 8192(R2) // MVC 8192(R1), $256, 8192(R2) // b90400a2c2a800002000b90400b1c2b800002000d2ffa000b000

	CMP	R1, R2                 // b9200012
	CMP	R3, $-2147483648       // c23c80000000
	CMPU	R4, R5                 // b9210045
	CMPU	R6, $4294967295        // c26effffffff
	CMPW	R7, R8                 // 1978
	CMPW	R9, $-2147483648       // c29d80000000
	CMPWU	R1, R2                 // 1512
	CMPWU	R3, $4294967295        // c23fffffffff

	BNE	0(PC)                  // a7740000
	BEQ	0(PC)                  // a7840000
	BLT	0(PC)                  // a7440000
	BLE	0(PC)                  // a7c40000
	BGT	0(PC)                  // a7240000
	BGE	0(PC)                  // a7a40000

	CMPBNE	R1, R2, 0(PC)          // ec1200007064
	CMPBEQ	R3, R4, 0(PC)          // ec3400008064
	CMPBLT	R5, R6, 0(PC)          // ec5600004064
	CMPBLE	R7, R8, 0(PC)          // ec780000c064
	CMPBGT	R9, R1, 0(PC)          // ec9100002064
	CMPBGE	R2, R3, 0(PC)          // ec230000a064

	CMPBNE	R1, $-127, 0(PC)       // ec170000817c
	CMPBEQ	R3, $0, 0(PC)          // ec380000007c
	CMPBLT	R5, $128, 0(PC)        // ec540000807c
	CMPBLE	R7, $127, 0(PC)        // ec7c00007f7c
	CMPBGT	R9, $0, 0(PC)          // ec920000007c
	CMPBGE	R2, $128, 0(PC)        // ec2a0000807c

	CMPUBNE	R1, R2, 0(PC)          // ec1200007065
	CMPUBEQ	R3, R4, 0(PC)          // ec3400008065
	CMPUBLT	R5, R6, 0(PC)          // ec5600004065
	CMPUBLE	R7, R8, 0(PC)          // ec780000c065
	CMPUBGT	R9, R1, 0(PC)          // ec9100002065
	CMPUBGE	R2, R3, 0(PC)          // ec230000a065

	CMPUBNE	R1, $256, 0(PC)        // ec170000007d
	CMPUBEQ	R3, $0, 0(PC)          // ec380000007d
	CMPUBLT	R5, $256, 0(PC)        // ec540000007d
	CMPUBLE	R7, $0, 0(PC)          // ec7c0000007d
	CMPUBGT	R9, $256, 0(PC)        // ec920000007d
	CMPUBGE	R2, $0, 0(PC)          // ec2a0000007d

	CEFBRA	R0, F15                // b39400f0
	CDFBRA	R1, F14                // b39500e1
	CEGBRA	R2, F13                // b3a400d2
	CDGBRA	R3, F12                // b3a500c3

	CELFBR	R0, F15                // b39000f0
	CDLFBR	R1, F14                // b39100e1
	CELGBR	R2, F13                // b3a000d2
	CDLGBR	R3, F12                // b3a100c3

	CFEBRA	F15, R1                // b398501f
	CFDBRA	F14, R2                // b399502e
	CGEBRA	F13, R3                // b3a8503d
	CGDBRA	F12, R4                // b3a9504c

	CLFEBR	F15, R1                // b39c501f
	CLFDBR	F14, R2                // b39d502e
	CLGEBR	F13, R3                // b3ac503d
	CLGDBR	F12, R4                // b3ad504c

	FMOVS	$0, F11                // b37400b0
	FMOVD	$0, F12                // b37500c0
	FMOVS	(R1)(R2*1), F0         // ed0210000064
	FMOVS	n-8(SP), F15           // edf0f0100064
	FMOVD	-9999999(R8)(R9*1), F8 // c0a1ff67698141aa9000ed8a80000065
	FMOVD	F4, F5                 // 2854
	FADDS	F0, F15                // b30a00f0
	FADD	F1, F14                // b31a00e1
	FSUBS	F2, F13                // b30b00d2
	FSUB	F3, F12                // b31b00c3
	FMULS	F4, F11                // b31700b4
	FMUL	F5, F10                // b31c00a5
	FDIVS	F6, F9                 // b30d0096
	FDIV	F7, F8                 // b31d0087
	FABS	F1, F2                 // b3100021
	FSQRTS	F3, F4                 // b3140043
	FSQRT	F5, F15                // b31500f5

	VL	(R15), V1              // e710f0000006
	VST	V1, (R15)              // e710f000000e
	VL	(R15), V31             // e7f0f0000806
	VST	V31, (R15)             // e7f0f000080e
	VESLB	$5, V14                // e7ee00050030
	VESRAG	$0, V15, V16           // e70f0000383a
	VLM	(R15), V8, V23         // e787f0000436
	VSTM	V8, V23, (R15)         // e787f000043e
	VONE	V1                     // e710ffff0044
	VZERO	V16                    // e70000000844
	VGBM	$52428, V31            // e7f0cccc0844
	VREPIB	$255, V4               // e74000ff0045
	VREPG	$1, V4, V16            // e7040001384d
	VREPB	$4, V31, V1            // e71f0004044d
	VFTCIDB	$4095, V1, V2          // e721fff0304a
	WFTCIDB	$3276, V15, V16        // e70fccc8384a
	VPOPCT	V8, V19                // e73800000850
	VFEEZBS	V1, V2, V31            // e7f120300880
	WFCHDBS	V22, V23, V4           // e746701836eb
	VMNH	V1, V2, V30            // e7e1200018fe
	VO	V2, V1, V0             // e7021000006a
	VERLLVF	V2, V30, V27           // e7be20002c73
	VSCBIB	V0, V23, V24           // e78700000cf5
	VNOT	V16, V1                // e7101000046b
	VCLZF	V16, V17               // e71000002c53
	VLVGP	R3, R4, V8             // e78340000062

	// Some vector instructions have their inputs reordered.
	// Typically the reordering puts the length/index input into From3.
	VGEG	$1, 8(R15)(V30*1), V31  // VGEG    8(R15)(V30*1), $1, V31  // e7fef0081c12
	VSCEG	$1, V31, 16(R15)(V30*1) // VSCEG   V31, $1, 16(R15)(V30*1) // e7fef0101c1a
	VGEF	$0, 2048(R15)(V1*1), V2 // VGEF    2048(R15)(V1*1), $0, V2 // e721f8000013
	VSCEF	$0, V2, 4095(R15)(V1*1) // VSCEF   V2, $0, 4095(R15)(V1*1) // e721ffff001b
	VLL	R0, (R15), V1           // VLL     (R15), R0, V1           // e710f0000037
	VSTL	R0, V16, (R15)          // VSTL    V16, R0, (R15)          // e700f000083f
	VGMH	$8, $16, V12            // VGMH    $16, $8, V12            // e7c008101046
	VLEIF	$2, $-43, V16           // VLEIF   $-43, $2, V16           // e700ffd52843
	VSLDB	$3, V1, V16, V18        // VSLDB   V1, V16, $3, V18        // e72100030a77
	VERIMB	$2, V31, V1, V2         // VERIMB  V31, V1, $2, V2         // e72f10020472
	VSEL	V1, V2, V3, V4          // VSEL    V2, V3, V1, V4          // e7412000308d
	VGFMAH	V21, V31, V24, V0       // VGFMAH  V31, V24, V21, V0       // e705f10087bc
	WFMSDB	V2, V25, V24, V31       // WFMSDB  V25, V24, V2, V31       // e7f298038b8e
	VPERM	V31, V0, V2, V3         // VPERM   V0, V2, V31, V3         // e73f0000248c
	VPDI	$1, V2, V31, V1         // VPDI    V2, V31, $1, V1         // e712f0001284

	RET

TEXT main·init(SB),7,$0 // TEXT main.init(SB), 7, $0
	RET

TEXT main·main(SB),7,$0 // TEXT main.main(SB), 7, $0
	BL      main·foo(SB)    // CALL main.foo(SB)
	RET
