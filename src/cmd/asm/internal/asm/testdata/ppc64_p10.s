// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This contains the valid opcode combinations available
// in cmd/internal/obj/ppc64/asm9.go which exist for
// POWER10/ISA 3.1.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB), DUPOK|NOSPLIT, $0
	BRD R1, R2                              // 7c220176
	BRH R1, R2                              // 7c2201b6
	BRW R1, R2                              // 7c220136
	CFUGED R1, R2, R3                       // 7c2311b8
	CNTLZDM R2, R3, R1                      // 7c411876
	CNTTZDM R2, R3, R1                      // 7c411c76
	DCFFIXQQ V1, F2                         // fc400fc4
	DCTFIXQQ F2, V3                         // fc6117c4
	LXVKQ $0, VS33                          // f03f02d1
	LXVP 12352(R5), VS6                     // 18c53040
	LXVPX (R1)(R2), VS4                     // 7c820a9a
	LXVRBX (R1)(R2), VS4                    // 7c82081a
	LXVRDX (R1)(R2), VS4                    // 7c8208da
	LXVRHX (R1)(R2), VS4                    // 7c82085a
	LXVRWX (R1)(R2), VS4                    // 7c82089a
	MTVSRBM R1, V1                          // 10300e42
	MTVSRBMI $5, V1                         // 10220015
	MTVSRDM R1, V1                          // 10330e42
	MTVSRHM R1, V1                          // 10310e42
	MTVSRQM R1, V1                          // 10340e42
	MTVSRWM R1, V1                          // 10320e42
	PADDI R3, $1234567890, $1, R4           // 06104996388302d2
	PADDI R0, $1234567890, $0, R4           // 06004996388002d2
	PADDI R0, $1234567890, $1, R4           // 06104996388002d2
	PDEPD R1, R2, R3                        // 7c231138
	PEXTD R1, R2, R3                        // 7c231178
	PLBZ 1234(R1), $0, R3                   // 06000000886104d260000000
	// Note, PLD crosses a 64B boundary, and a nop is inserted between PLBZ and PLD
	PLD 1234(R1), $0, R3                    // 04000000e46104d2
	PLFD 1234(R1), $0, F3                   // 06000000c86104d2
	PLFS 1234567890(R4), $0, F3             // 06004996c06402d2
	PLFS 1234567890(R0), $1, F3             // 06104996c06002d2
	PLHA 1234(R1), $0, R3                   // 06000000a86104d2
	PLHZ 1234(R1), $0, R3                   // 06000000a06104d2
	PLQ 1234(R1), $0, R4                    // 04000000e08104d2
	PLWA 1234(R1), $0, R3                   // 04000000a46104d2
	PLWZ 1234567890(R4), $0, R3             // 06004996806402d2
	PLWZ 1234567890(R0), $1, R3             // 06104996806002d2
	PLXSD 1234(R1), $0, V1                  // 04000000a82104d2
	PLXSSP 5(R1), $0, V2                    // 04000000ac410005
	PLXSSP 5(R0), $1, V2                    // 04100000ac400005
	PLXV 12346891(R6), $1, VS44             // 041000bccd86660b
	PLXVP 12345678(R4), $1, VS4             // 041000bce884614e
	PMXVBF16GER2 VS1, VS2, $1, $2, $3, A1   // 0790c012ec811198
	PMXVBF16GER2NN VS1, VS2, $1, $2, $3, A1 // 0790c012ec811790
	PMXVBF16GER2NP VS1, VS2, $1, $2, $3, A1 // 0790c012ec811390
	PMXVBF16GER2PN VS1, VS2, $1, $2, $3, A1 // 0790c012ec811590
	PMXVBF16GER2PP VS1, VS2, $1, $2, $3, A1 // 0790c012ec811190
	PMXVF16GER2 VS1, VS2, $1, $2, $3, A1    // 0790c012ec811098
	PMXVF16GER2NN VS1, VS2, $1, $2, $3, A1  // 0790c012ec811690
	PMXVF16GER2NP VS1, VS2, $1, $2, $3, A1  // 0790c012ec811290
	PMXVF16GER2PN VS1, VS2, $1, $2, $3, A1  // 0790c012ec811490
	PMXVF16GER2PP VS1, VS2, $1, $2, $3, A1  // 0790c012ec811090
	PMXVF32GER VS1, VS2, $1, $2, A1         // 07900012ec8110d8
	PMXVF32GERNN VS1, VS2, $1, $2, A1       // 07900012ec8116d0
	PMXVF32GERNP VS1, VS2, $1, $2, A1       // 07900012ec8112d0
	PMXVF32GERPN VS1, VS2, $1, $2, A1       // 07900012ec8114d0
	PMXVF32GERPP VS1, VS2, $1, $2, A1       // 07900012ec8110d0
	PMXVF64GER VS4, VS2, $1, $2, A1         // 07900018ec8411d8
	PMXVF64GERNN VS4, VS2, $1, $2, A1       // 07900018ec8417d0
	PMXVF64GERNP VS4, VS2, $1, $2, A1       // 07900018ec8413d0
	PMXVF64GERPN VS4, VS2, $1, $2, A1       // 07900018ec8415d0
	PMXVF64GERPP VS4, VS2, $1, $2, A1       // 07900018ec8411d0
	PMXVI16GER2 VS1, VS2, $1, $2, $3, A1    // 0790c012ec811258
	PMXVI16GER2PP VS1, VS2, $1, $2, $3, A1  // 0790c012ec811358
	PMXVI16GER2S VS1, VS2, $1, $2, $3, A1   // 0790c012ec811158
	PMXVI16GER2SPP VS1, VS2, $1, $2, $3, A1 // 0790c012ec811150
	PMXVI4GER8 VS1, VS2, $1, $2, $3, A1     // 07900312ec811118
	PMXVI4GER8PP VS1, VS2, $1, $2, $3, A1   // 07900312ec811110
	PMXVI8GER4 VS1, VS2, $1, $2, $3, A1     // 07903012ec811018
	PMXVI8GER4PP VS1, VS2, $1, $2, $3, A1   // 07903012ec811010
	PMXVI8GER4SPP VS1, VS2, $1, $2, $3, A1  // 07903012ec811318
	PNOP                                    // 0700000000000000
	PSTB R1, $1, 12345678(R2)               // 061000bc9822614e
	PSTD R1, $1, 12345678(R2)               // 041000bcf422614e
	PSTFD F1, $1, 12345678(R2)              // 061000bcd822614e
	PSTFS F1, $1, 123456789(R7)             // 0610075bd027cd15
	PSTH R1, $1, 12345678(R2)               // 061000bcb022614e
	PSTQ R2, $1, 12345678(R2)               // 041000bcf042614e
	PSTW R1, $1, 12345678(R2)               // 061000bc9022614e
	PSTW R24, $0, 45(R13)                   // 06000000930d002d
	PSTXSD V1, $1, 12345678(R2)             // 041000bcb822614e
	PSTXSSP V1, $1, 1234567890(R0)          // 04104996bc2002d2
	PSTXSSP V1, $1, 1234567890(R1)          // 04104996bc2102d2
	PSTXSSP V1, $0, 1234567890(R3)          // 04004996bc2302d2
	PSTXV VS6, $1, 1234567890(R5)           // 04104996d8c502d2
	PSTXVP VS2, $1, 12345678(R2)            // 041000bcf842614e
	PSTXVP VS62, $0, 5555555(R3)            // 04000054fbe3c563
	SETBC CR2EQ, R2                         // 7c4a0300
	SETBCR CR2LT, R2                        // 7c480340
	SETNBC CR2GT, R2                        // 7c490380
	SETNBCR CR6SO, R2                       // 7c5b03c0
	STXVP VS6, 12352(R5)                    // 18c53041
	STXVPX VS22, (R1)(R2)                   // 7ec20b9a
	STXVRBX VS2, (R1)(R2)                   // 7c42091a
	STXVRDX VS2, (R1)(R2)                   // 7c4209da
	STXVRHX VS2, (R1)(R2)                   // 7c42095a
	STXVRWX VS2, (R1)(R2)                   // 7c42099a
	VCFUGED V1, V2, V3                      // 1061154d
	VCLRLB V1, R2, V3                       // 1061118d
	VCLRRB V1, R2, V3                       // 106111cd
	VCLZDM V1, V2, V3                       // 10611784
	VCMPEQUQ V1, V2, V3                     // 106111c7
	VCMPEQUQCC V1, V2, V3                   // 106115c7
	VCMPGTSQ V1, V2, V3                     // 10611387
	VCMPGTSQCC V1, V2, V3                   // 10611787
	VCMPGTUQ V1, V2, V3                     // 10611287
	VCMPGTUQCC V1, V2, V3                   // 10611687
	VCMPSQ V1, V2, CR2                      // 11011141
	VCMPUQ V1, V2, CR3                      // 11811101
	VCNTMBB V1, $1, R3                      // 10790e42
	VCNTMBD V1, $1, R3                      // 107f0e42
	VCNTMBH V1, $1, R3                      // 107b0e42
	VCNTMBW V1, $1, R3                      // 107d0e42
	VCTZDM V1, V2, V3                       // 106117c4
	VDIVESD V1, V2, V3                      // 106113cb
	VDIVESQ V1, V2, V3                      // 1061130b
	VDIVESW V1, V2, V3                      // 1061138b
	VDIVEUD V1, V2, V3                      // 106112cb
	VDIVEUQ V1, V2, V3                      // 1061120b
	VDIVEUW V1, V2, V3                      // 1061128b
	VDIVSD V1, V2, V3                       // 106111cb
	VDIVSQ V1, V2, V3                       // 1061110b
	VDIVSW V1, V2, V3                       // 1061118b
	VDIVUD V1, V2, V3                       // 106110cb
	VDIVUQ V1, V2, V3                       // 1061100b
	VDIVUW V1, V2, V3                       // 1061108b
	VEXPANDBM V1, V2                        // 10400e42
	VEXPANDDM V1, V2                        // 10430e42
	VEXPANDHM V1, V2                        // 10410e42
	VEXPANDQM V1, V2                        // 10440e42
	VEXPANDWM V1, V2                        // 10420e42
	VEXTDDVLX V1, V2, R3, V4                // 108110de
	VEXTDDVRX V1, V2, R3, V4                // 108110df
	VEXTDUBVLX V1, V2, R3, V4               // 108110d8
	VEXTDUBVRX V1, V2, R3, V4               // 108110d9
	VEXTDUHVLX V1, V2, R3, V4               // 108110da
	VEXTDUHVRX V1, V2, R3, V4               // 108110db
	VEXTDUWVLX V1, V2, R3, V4               // 108110dc
	VEXTDUWVRX V1, V2, R5, V3               // 1061115d
	VEXTRACTBM V1, R2                       // 10480e42
	VEXTRACTDM V1, R2                       // 104b0e42
	VEXTRACTHM V1, R2                       // 10490e42
	VEXTRACTQM V1, R2                       // 104c0e42
	VEXTRACTWM V1, R6                       // 10ca0e42
	VEXTSD2Q V1, V2                         // 105b0e02
	VGNB V1, $1, R31                        // 13e10ccc
	VINSBLX R1, R2, V3                      // 1061120f
	VINSBRX R1, R2, V3                      // 1061130f
	VINSBVLX R1, V1, V2                     // 1041080f
	VINSBVRX R1, V1, V2                     // 1041090f
	VINSD R1, $2, V2                        // 104209cf
	VINSDLX R1, R2, V3                      // 106112cf
	VINSDRX R1, R2, V3                      // 106113cf
	VINSHLX R1, R2, V3                      // 1061124f
	VINSHRX R1, R2, V3                      // 1061134f
	VINSHVLX R1, V2, V3                     // 1061104f
	VINSHVRX R1, V2, V3                     // 1061114f
	VINSW R1, $4, V3                        // 106408cf
	VINSWLX R1, R2, V3                      // 1061128f
	VINSWRX R1, R2, V3                      // 1061138f
	VINSWVLX R1, V2, V3                     // 1061108f
	VINSWVRX R1, V2, V3                     // 1061118f
	VMODSD V1, V2, V3                       // 106117cb
	VMODSQ V1, V2, V3                       // 1061170b
	VMODSW V1, V2, V3                       // 1061178b
	VMODUD V1, V2, V3                       // 106116cb
	VMODUQ V1, V2, V3                       // 1061160b
	VMODUW V1, V2, V3                       // 1061168b
	VMSUMCUD V1, V2, V3, V4                 // 108110d7
	VMULESD V1, V2, V3                      // 106113c8
	VMULEUD V1, V2, V3                      // 106112c8
	VMULHSD V1, V2, V3                      // 106113c9
	VMULHSW V1, V2, V3                      // 10611389
	VMULHUD V1, V2, V3                      // 106112c9
	VMULHUW V1, V2, V3                      // 10611289
	VMULLD V1, V2, V3                       // 106111c9
	VMULOSD V1, V2, V3                      // 106111c8
	VMULOUD V1, V2, V3                      // 106110c8
	VPDEPD V1, V2, V3                       // 106115cd
	VPEXTD V1, V2, V3                       // 1061158d
	VRLQ V1, V2, V3                         // 10611005
	VRLQMI V1, V2, V3                       // 10611045
	VRLQNM V1, V2, V3                       // 10611145
	VSLDBI V1, V2, $3, V3                   // 106110d6
	VSLQ V1, V2, V3                         // 10611105
	VSRAQ V1, V2, V3                        // 10611305
	VSRDBI V1, V2, $3, V4                   // 108112d6
	VSRQ V1, V2, V3                         // 10611205
	VSTRIBL V1, V2                          // 1040080d
	VSTRIBLCC V1, V2                        // 10400c0d
	VSTRIBR V1, V2                          // 1041080d
	VSTRIBRCC V1, V2                        // 10410c0d
	VSTRIHL V1, V2                          // 1042080d
	VSTRIHLCC V1, V2                        // 10420c0d
	VSTRIHR V1, V2                          // 1043080d
	VSTRIHRCC V1, V2                        // 10430c0d
	XSCMPEQQP V1, V2, V3                    // fc611088
	XSCMPGEQP V1, V2, V3                    // fc611188
	XSCMPGTQP V1, V2, V3                    // fc6111c8
	XSCVQPSQZ V1, V2                        // fc480e88
	XSCVQPUQZ V1, V2                        // fc400e88
	XSCVSQQP V1, V2                         // fc4b0e88
	XSCVUQQP V2, V3                         // fc631688
	XSMAXCQP V1, V2, V3                     // fc611548
	XSMINCQP V1, V2, V4                     // fc8115c8
	XVBF16GER2 VS1, VS2, A1                 // ec811198
	XVBF16GER2NN VS1, VS2, A1               // ec811790
	XVBF16GER2NP VS1, VS2, A1               // ec811390
	XVBF16GER2PN VS1, VS2, A1               // ec811590
	XVBF16GER2PP VS1, VS2, A1               // ec811190
	XVCVBF16SPN VS2, VS3                    // f070176c
	XVCVSPBF16 VS1, VS4                     // f0910f6c
	XVF16GER2 VS1, VS2, A1                  // ec811098
	XVF16GER2NN VS1, VS2, A1                // ec811690
	XVF16GER2NP VS1, VS2, A1                // ec811290
	XVF16GER2PN VS1, VS2, A1                // ec811490
	XVF16GER2PP VS1, VS2, A1                // ec811090
	XVF32GER VS1, VS2, A1                   // ec8110d8
	XVF32GERNN VS1, VS2, A1                 // ec8116d0
	XVF32GERNP VS1, VS2, A1                 // ec8112d0
	XVF32GERPN VS1, VS2, A1                 // ec8114d0
	XVF32GERPP VS1, VS2, A1                 // ec8110d0
	XVF64GER VS2, VS1, A1                   // ec8209d8
	XVF64GERNN VS2, VS1, A1                 // ec820fd0
	XVF64GERNP VS2, VS1, A1                 // ec820bd0
	XVF64GERPN VS2, VS1, A1                 // ec820dd0
	XVF64GERPP VS2, VS1, A1                 // ec8209d0
	XVI16GER2 VS1, VS2, A1                  // ec811258
	XVI16GER2PP VS1, VS2, A1                // ec811358
	XVI16GER2S VS1, VS2, A1                 // ec811158
	XVI16GER2SPP VS1, VS2, A1               // ec811150
	XVI4GER8 VS1, VS2, A1                   // ec811118
	XVI4GER8PP VS1, VS2, A1                 // ec811110
	XVI8GER4 VS1, VS2, A1                   // ec811018
	XVI8GER4PP VS1, VS2, A1                 // ec811010
	XVI8GER4SPP VS4, VS6, A1                // ec843318
	XVTLSBB VS1, CR2                        // f1020f6c
	XXBLENDVB VS1, VS3, VS7, VS11           // 05000000856119c0
	XXBLENDVD VS1, VS3, VS7, VS11           // 05000000856119f0
	XXBLENDVH VS1, VS3, VS7, VS11           // 05000000856119d0
	XXBLENDVW VS1, VS3, VS7, VS11           // 05000000856119e0
	XXEVAL VS1, VS2, VS3, $2, VS4           // 05000002888110d0
	XXGENPCVBM V2, $2, VS3                  // f0621728
	XXGENPCVDM V2, $2, VS3                  // f062176a
	XXGENPCVHM V2, $2, VS3                  // f062172a
	XXGENPCVWM V2, $2, VS3                  // f0621768
	XXMFACC A1                              // 7c800162
	XXMTACC A1                              // 7c810162
	XXPERMX VS1, VS34, VS2, $2, VS3         // 0500000288611082
	XXSETACCZ A1                            // 7c830162
	XXSPLTI32DX $1, $1234, VS3              // 05000000806204d2
	XXSPLTIDP $12345678, VS4                // 050000bc8084614e
	XXSPLTIW $123456, VS3                   // 050000018066e240

	// ISA 3.1B
	HASHST R2, -8(R1)                       // 7fe115a5
	HASHSTP R2, -8(R1)                      // 7fe11525
	HASHCHK -8(R1), R2                      // 7fe115e5
	HASHCHKP -8(R1), R2                     // 7fe11565

	RET
