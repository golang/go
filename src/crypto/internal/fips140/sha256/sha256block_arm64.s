// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

#define HASHUPDATE \
	SHA256H	V9.S4, V3, V2 \
	SHA256H2	V9.S4, V8, V3 \
	VMOV	V2.B16, V8.B16

// func blockSHA2(dig *Digest, p []byte)
TEXT ·blockSHA2(SB),NOSPLIT,$0
	MOVD	dig+0(FP), R0                              // Hash value first address
	MOVD	p_base+8(FP), R1                           // message first address
	MOVD	p_len+16(FP), R3                           // message length
	MOVD	$·_K+0(SB), R2                             // k constants first address
	VLD1	(R0), [V0.S4, V1.S4]                       // load h(a,b,c,d,e,f,g,h)
	VLD1.P	64(R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P	64(R2), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P	64(R2), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1	(R2), [V28.S4, V29.S4, V30.S4, V31.S4]     //load 64*4bytes K constant(K0-K63)

blockloop:

	VLD1.P	16(R1), [V4.B16]                            // load 16bytes message
	VLD1.P	16(R1), [V5.B16]                            // load 16bytes message
	VLD1.P	16(R1), [V6.B16]                            // load 16bytes message
	VLD1.P	16(R1), [V7.B16]                            // load 16bytes message
	VMOV	V0.B16, V2.B16                              // backup: VO h(dcba)
	VMOV	V1.B16, V3.B16                              // backup: V1 h(hgfe)
	VMOV	V2.B16, V8.B16
	VREV32	V4.B16, V4.B16                              // prepare for using message in Byte format
	VREV32	V5.B16, V5.B16
	VREV32	V6.B16, V6.B16
	VREV32	V7.B16, V7.B16

	VADD	V16.S4, V4.S4, V9.S4                        // V18(W0+K0...W3+K3)
	SHA256SU0	V5.S4, V4.S4                        // V4: (su0(W1)+W0,...,su0(W4)+W3)
	HASHUPDATE                                          // H4

	VADD	V17.S4, V5.S4, V9.S4                        // V18(W4+K4...W7+K7)
	SHA256SU0	V6.S4, V5.S4                        // V5: (su0(W5)+W4,...,su0(W8)+W7)
	SHA256SU1	V7.S4, V6.S4, V4.S4                 // V4: W16-W19
	HASHUPDATE                                          // H8

	VADD	V18.S4, V6.S4, V9.S4                        // V18(W8+K8...W11+K11)
	SHA256SU0	V7.S4, V6.S4                        // V6: (su0(W9)+W8,...,su0(W12)+W11)
	SHA256SU1	V4.S4, V7.S4, V5.S4                 // V5: W20-W23
	HASHUPDATE                                          // H12

	VADD	V19.S4, V7.S4, V9.S4                        // V18(W12+K12...W15+K15)
	SHA256SU0	V4.S4, V7.S4                        // V7: (su0(W13)+W12,...,su0(W16)+W15)
	SHA256SU1	V5.S4, V4.S4, V6.S4                 // V6: W24-W27
	HASHUPDATE                                          // H16

	VADD	V20.S4, V4.S4, V9.S4                        // V18(W16+K16...W19+K19)
	SHA256SU0	V5.S4, V4.S4                        // V4: (su0(W17)+W16,...,su0(W20)+W19)
	SHA256SU1	V6.S4, V5.S4, V7.S4                 // V7: W28-W31
	HASHUPDATE                                          // H20

	VADD	V21.S4, V5.S4, V9.S4                        // V18(W20+K20...W23+K23)
	SHA256SU0	V6.S4, V5.S4                        // V5: (su0(W21)+W20,...,su0(W24)+W23)
	SHA256SU1	V7.S4, V6.S4, V4.S4                 // V4: W32-W35
	HASHUPDATE                                          // H24

	VADD	V22.S4, V6.S4, V9.S4                        // V18(W24+K24...W27+K27)
	SHA256SU0	V7.S4, V6.S4                        // V6: (su0(W25)+W24,...,su0(W28)+W27)
	SHA256SU1	V4.S4, V7.S4, V5.S4                 // V5: W36-W39
	HASHUPDATE                                          // H28

	VADD	V23.S4, V7.S4, V9.S4                        // V18(W28+K28...W31+K31)
	SHA256SU0	V4.S4, V7.S4                        // V7: (su0(W29)+W28,...,su0(W32)+W31)
	SHA256SU1	V5.S4, V4.S4, V6.S4                 // V6: W40-W43
	HASHUPDATE                                          // H32

	VADD	V24.S4, V4.S4, V9.S4                        // V18(W32+K32...W35+K35)
	SHA256SU0	V5.S4, V4.S4                        // V4: (su0(W33)+W32,...,su0(W36)+W35)
	SHA256SU1	V6.S4, V5.S4, V7.S4                 // V7: W44-W47
	HASHUPDATE                                          // H36

	VADD	V25.S4, V5.S4, V9.S4                        // V18(W36+K36...W39+K39)
	SHA256SU0	V6.S4, V5.S4                        // V5: (su0(W37)+W36,...,su0(W40)+W39)
	SHA256SU1	V7.S4, V6.S4, V4.S4                 // V4: W48-W51
	HASHUPDATE                                          // H40

	VADD	V26.S4, V6.S4, V9.S4                        // V18(W40+K40...W43+K43)
	SHA256SU0	V7.S4, V6.S4                        // V6: (su0(W41)+W40,...,su0(W44)+W43)
	SHA256SU1	V4.S4, V7.S4, V5.S4                 // V5: W52-W55
	HASHUPDATE                                          // H44

	VADD	V27.S4, V7.S4, V9.S4                        // V18(W44+K44...W47+K47)
	SHA256SU0	V4.S4, V7.S4                        // V7: (su0(W45)+W44,...,su0(W48)+W47)
	SHA256SU1	V5.S4, V4.S4, V6.S4                 // V6: W56-W59
	HASHUPDATE                                          // H48

	VADD	V28.S4, V4.S4, V9.S4                        // V18(W48+K48,...,W51+K51)
	HASHUPDATE                                          // H52
	SHA256SU1	V6.S4, V5.S4, V7.S4                 // V7: W60-W63

	VADD	V29.S4, V5.S4, V9.S4                        // V18(W52+K52,...,W55+K55)
	HASHUPDATE                                          // H56

	VADD	V30.S4, V6.S4, V9.S4                        // V18(W59+K59,...,W59+K59)
	HASHUPDATE                                          // H60

	VADD	V31.S4, V7.S4, V9.S4                        // V18(W60+K60,...,W63+K63)
	HASHUPDATE                                          // H64

	SUB	$64, R3, R3                                 // message length - 64bytes, then compare with 64bytes
	VADD	V2.S4, V0.S4, V0.S4
	VADD	V3.S4, V1.S4, V1.S4
	CBNZ	R3, blockloop

sha256ret:

	VST1	[V0.S4, V1.S4], (R0)                       // store hash value H
	RET

