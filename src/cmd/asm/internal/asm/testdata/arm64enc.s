// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO means they cannot be handled now.
// Comment cases means they are handled incorrectly.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$-8

   AND $(1<<63), R1                           // AND $-9223372036854775808, R1     // 21004192
   ADCW ZR, R8, R10                           // 0a011f1a
   ADC R0, R2, R12                            // 4c00009a
   ADCSW R9, R21, R6                          // a602093a
   ADCS R23, R22, R22                         // d60217ba
   //TODO ADDW R5.UXTH, R8, R9                // 0921250b
   //TODO ADD R8.SXTB<<7, R23, R14            // ee9e288b
   ADDW $3076, R17, R3                        // 23123011
   ADDW $(3076<<12), R17, R3                  // ADDW $12599296, R17, R3           // 23127011
   ADD $2280, R25, R11                        // 2ba32391
   ADD $(2280<<12), R25, R11                  // ADD $9338880, R25, R11            // 2ba36391
   ADDW R13->5, R11, R7                       // 67158d0b
   ADD R25<<54, R17, R16                      // 30da198b
   //TODO ADDSW R12.SXTX<<1, R29, R7          // a7e72c2b
   //TODO ADDS R24.UXTX<<4, R25, R21          // 357338ab
   ADDSW $(3525<<12), R3, R11                 // ADDSW $14438400, R3, R11          // 6b147731
   ADDS $(3525<<12), R3, R11                  // ADDS $14438400, R3, R11           // 6b1477b1
   ADDSW R7->22, R14, R13                     // cd59872b
   ADDS R14>>7, ZR, R4                        // e41f4eab
   AND $-9223372036854775808, R1, R1          // 21004192
   ANDW $4026540031, R29, R2                  // a2430412
   AND $34903429696192636, R12, R19           // 93910e92
   ANDW R9@>7, R19, R26                       // 7a1ec90a
   AND R9@>7, R19, R26                        // 7a1ec98a
   //TODO TST $2863311530, R24                // 1ff32972
   ANDSW $2863311530, R24, R23                // 17f30172
   ANDS $-140737488289793, R2, R5             // 458051f2
   ANDSW R26->24, R21, R15                    // af629a6a
   ANDS R30@>44, R3, R26                      // 7ab0deea
   ASRW R12, R27, R25                         // 792bcc1a
   ASR R14, R27, R7                           // 672bce9a
   ASR $11, R27, R25                          // 79ff4b93
   ASRW $11, R27, R25                         // 797f0b13
   BLT -1(PC)                                 // ebffff54
   JMP -1(PC)                                 // ffffff17
   BFIW $16, R20, $6, R0                      // 80161033
   BFI $27, R21, $21, R25                     // b95265b3
   BFXILW $3, R27, $23, R14                   // 6e670333
   BFXIL $26, R8, $16, R20                    // 14a55ab3
   BICW R7@>15, R5, R16                       // b03ce70a
   BIC R12@>13, R12, R18                      // 9235ec8a
   BICSW R25->20, R3, R20                     // 7450b96a
   BICS R19->12, R1, R23                      // 3730b3ea
   BICS R19, R1, R23                          // 370033ea
   BICS R19>>0, R1, R23                       // 370073ea
   CALL -1(PC)                                // ffffff97
   CALL (R15)                                 // e0013fd6
   JMP  (R29)                                 // a0031fd6
   // BRK $35943                              // e08c31d4
   CBNZW R2, -1(PC)                           // e2ffff35
   CBNZ R7, -1(PC)                            // e7ffffb5
   CBZW R15, -1(PC)                           // efffff34
   CBZ R1, -1(PC)                             // e1ffffb4
   CCMN MI, ZR, R1, $4                        // e44341ba
   CCMNW AL, R26, $20, $11                    // 4beb543a
   CCMN PL, R24, $6, $1                       // 015b46ba
   CCMNW EQ, R20, R6, $6                      // 8602463a
   CCMN LE, R30, R12, $6                      // c6d34cba
   CCMPW VS, R29, $15, $7                     // a76b4f7a
   CCMP LE, R7, $19, $3                       // e3d853fa
   CCMPW HS, R18, R6, $0                      // 4022467a
   CCMP LT, R30, R6, $7                       // c7b346fa
   CCMN  MI, ZR, R1, $4                       // e44341ba
   CSINCW HS, ZR, R27, R14                    // ee279b1a
   CSINC VC, R2, R1, R1                       // 4174819a
   CSINVW EQ, R2, R21, R17                    // 5100955a
   CSINV LO, R2, R19, R23                     // 573093da
   CINCW LO, R27, R14                         // 6e279b1a
   CINCW HS, R27, ZR                          // 7f379b1a
   CINVW EQ, R2, R17                          // 5110825a
   CINV VS, R12, R7                           // 87718cda
   CINV VS, R30, R30                          // de739eda
   // CLREX $4                                // 5f3403d5
   // CLREX $0                                // 5f3003d5
   CLSW R15, R6                               // e615c05a
   CLS R15, ZR                                // ff15c0da
   CLZW R1, R14                               // 2e10c05a
   CLZ R21, R9                                // a912c0da
   //TODO CMNW R21.UXTB<<4, R15               // ff11352b
   //TODO CMN R0.UXTW<<4, R16                 // 1f5220ab
   CMNW R13>>8, R9                            // 3f214d2b
   CMN R6->17, R3                             // 7f4486ab
   CMNW $(2<<12), R5                          // CMNW $8192, R5                // bf084031
   CMN $(8<<12), R12                          // CMN $32768, R12               // 9f2140b1
   CMN R6->0, R3                              // 7f0086ab
   CMN R6, R3                                 // 7f0006ab
   CMNW R30, R5                               // bf001e2b
   CMNW $2, R5                                // bf080031
   CMN ZR, R3                                 // 7f001fab
   CMN R0, R3                                 // 7f0000ab
   //TODO CMPW R6.UXTB, R23                   // ff02266b
   //TODO CMP R25.SXTH<<5, R26                // 5fb739eb
   CMP $3817, R29                             // bfa73bf1
   CMP R7>>23, R3                             // 7f5c47eb
   CNEGW PL, R9, R14                          // 2e45895a
   CSNEGW HS, R5, R9, R14                     // ae24895a
   CSNEG PL, R14, R21, R3                     // c35595da
   CNEG  LO, R7, R15                          // ef2487da
   CRC32B R17, R8, R16                        // 1041d11a
   CRC32H R3, R21, R27                        // bb46c31a
   CRC32W R22, R30, R9                        // c94bd61a
   CRC32X R20, R4, R15                        // 8f4cd49a
   CRC32CB R18, R27, R22                      // 7653d21a
   CRC32CH R21, R0, R20                       // 1454d51a
   CRC32CW R9, R3, R21                        // 7558c91a
   CRC32CX R11, R0, R24                       // 185ccb9a
   CSELW LO, R4, R20, R12                     // 8c30941a
   CSEL GE, R0, R12, R14                      // 0ea08c9a
   CSETW GE, R3                               // e3b79f1a
   CSET LT, R30                               // fea79f9a
   CSETMW VC, R5                              // e5639f5a
   CSETM VS, R4                               // e4739fda
   CSINCW LE, R5, R24, R26                    // bad4981a
   CSINC VS, R26, R16, R17                    // 5167909a
   CSINVW AL, R23, R21, R5                    // e5e2955a
   CSINV LO, R2, R11, R14                     // 4e308bda
   CSNEGW HS, R16, R29, R10                   // 0a269d5a
   CSNEG NE, R21, R18, R11                    // ab1692da
   //TODO DC
   // DCPS1 $11378                            // 418ea5d4
   // DCPS2 $10699                            // 6239a5d4
   // DCPS3 $24415                            // e3ebabd4
   DMB $1                                     // bf3103d5
   DMB $0                                     // bf3003d5
   DRPS                                       // e003bfd6
   DSB  $1                                    // 9f3103d5
   EONW R21<<29, R6, R9                       // c974354a
   EON R14>>46, R4, R9                        // 89b86eca
   EOR $-2287828610704211969, R27, R22        // 76e343d2
   EORW R12->27, R10, R19                     // 536d8c4a
   EOR R2<<59, R30, R17                       // d1ef02ca
   ERET                                       // e0039fd6
   EXTRW $7, R8, R10, R25                     // 591d8813
   EXTR $35, R22, R12, R8                     // 888dd693
   SEVL                                       // bf2003d5
   HINT $6                                    // df2003d5
   // HLT $65509                              // a0fc5fd4
   // HVC $61428                              // 82fe1dd4
   ISB $1                                     // df3103d5
   ISB $15                                    // df3f03d5
   LDARW (R12), R29                           // 9dfddf88
   LDARW (R30), R22                           // d6ffdf88
   LDARW (RSP), R22                           // f6ffdf88
   LDAR (R27), R22                            // 76ffdfc8
   //TODO LDARB (R25), R2                     // 22ffdf08
   //TODO LDARH (R5), R7                      // a7fcdf48
   //TODO LDAXPW (R10), R16, R20              // 54c17f88
   //TODO LDAXP (R25), R11, R30               // 3eaf7fc8
   LDAXRW (R15), R2                           // e2fd5f88
   LDAXR (R15), R21                           // f5fd5fc8
   LDAXRB (R19), R16                          // 70fe5f08
   LDAXRH (R5), R8                            // a8fc5f48
   //TODO LDNP 0xcc(RSP), ZR, R12             // ecff5928
   //TODO LDNP 0x40(R28), R9, R5              // 852744a8
   //TODO LDPSW -0xd0(R2), R0, R12            // 4c00e668
   //TODO LDPSW 0x5c(R4), R8, R5              // 85a0cb69
   //TODO LDPSW 0x6c(R12), R2, R27            // 9b894d69
   MOVWU.P -84(R15), R9                       // e9c55ab8
   MOVD.P -46(R10), R8                        // 48255df8
   MOVD.P (R10), R8                           // 480540f8
   MOVWU.W -141(R3), R16                      // 703c57b8
   MOVD.W -134(R0), R29                       // 1dac57f8
   MOVWU 4156(R1), R25                        // 393c50b9
   MOVD 14616(R10), R9                        // 498d5cf9
   MOVBU.P 42(R2), R12                        // 4ca44238
   MOVBU.W -27(R2), R14                       // 4e5c5e38
   MOVBU 2916(R24), R3                        // 03936d39
   //TODO MOVBU R14<<1(R18), R23              // 577a6e38
   MOVHU.P 107(R13), R13                      // adb54678
   MOVHU.W 192(R2), R2                        // 420c4c78
   MOVHU 6844(R4), R18                        // 92787579
   //TODO MOVBW.P 77(R18), R11                // 4bd6c438
   MOVB.P 36(RSP), R27                        // fb478238
   //TODO MOVBW.W -57(R18), R13               // 4d7edc38
   MOVB.W -178(R16), R24                      // 18ee9438
   //TODO MOVBW 430(R8), R22                  // 16b9c639
   MOVB 997(R9), R23                          // 37958f39
   //TODO MOVBW R2<<1(R21), R15               // af7ae238
   //TODO MOVBW R26(R0), R21                  // 1568fa38
   //TODO MOVB R5(R15), R16                   // f069a538
   //TODO MOVHW.P 218(R22), R25               // d9a6cd78
   MOVH.P 179(R23), R5                        // e5368b78
   //TODO MOVHW.W 136(R2), R27                // 5b8cc878
   MOVH.W -63(R25), R22                       // 361f9c78
   //TODO MOVHW 5708(R25), R21                // 359bec79
   MOVH 54(R2), R13                           // 4d6c8079
   MOVW.P -58(R16), R2                        // 02669cb8
   MOVW.W -216(R18), R8                       // 488e92b8
   MOVW 4764(R23), R10                        // ea9e92b9
   //TODO LDTR -0x1e(R3), R4                  // 64285eb8
   //TODO LDTR -0xe5(R3), R10                 // 6ab851f8
   //TODO LDTRB 0xf0(R13), R10                // aa094f38
   //TODO LDTRH 0xe8(R13), R23                // b7894e78
   //TODO LDTRSB -0x24(R20), R5               // 85cadd38
   //TODO LDTRSB -0x75(R9), R13               // 2db99838
   //TODO LDTRSH 0xef(R3), LR                 // 7ef8ce78
   //TODO LDTRSH 0x96(R19), R24               // 786a8978
   //TODO LDTRSW 0x1e(LR), R5                 // c5eb81b8
   //TODO LDUR 0xbf(R13), R1                  // a1f14bb8
   //TODO LDUR -0x3c(R22), R3                 // c3425cf8
   //TODO LDURB -0xff(R17), R14               // 2e125038
   //TODO LDURH 0x80(R1), R6                  // 26004878
   //TODO LDURSB 0xde(LR), R3                 // c3e3cd38
   //TODO LDURSB 0x96(R9), R7                 // 27618938
   //TODO LDURSH -0x49(R11), R28              // 7c71db78
   //TODO LDURSH -0x1f(R0), R29               // 1d109e78
   //TODO LDURSW 0x48(R6), R20                // d48084b8
   LDXPW (R24), R11, R23                      // 172f7f88
   LDXP (R0), R13, R16                        // 10347fc8
   LDXRW (RSP), R30                           // fe7f5f88
   LDXR (R27), R12                            // 6c7f5fc8
   LDXRB (R0), R4                             // 047c5f08
   LDXRH (R12), R26                           // 9a7d5f48
   LSLW R11, R10, R15                         // 4f21cb1a
   LSL R27, R24, R21                          // 1523db9a
   LSLW $5, R7, R22                           // f6681b53
   LSL $57, R17, R2                           // 221a47d3
   LSRW R9, R3, R12                           // 6c24c91a
   LSR R10, R5, R2                            // a224ca9a
   LSRW $1, R3, R16                           // 707c0153
   LSR $12, R1, R20                           // 34fc4cd3
   MADDW R13, R23, R3, R10                    // 6a5c0d1b
   MADD R5, R23, R10, R4                      // 445d059b
   MNEGW R0, R9, R21                          // 35fd001b
   MNEG R14, R27, R23                         // 77ff0e9b
   MOVD  R2, R7                               // e70302aa
   MOVW $-24, R20                             // f4028012
   MOVD $-51096, ZR                           // fff29892
   MOVW $2507014144, R20                      // d4adb252
   MOVD $1313925191285342208, R7              // 8747e2d2
   ORRW $16252928, ZR, R21                    // f5130d32
   MOVD $-4260607558625, R11                  // eb6b16b2
   MOVD R30, R7                               // e7031eaa
   // MOVKW $(3905<<0), R21                   // MOVKW $3905, R21              // 35e88172
   // MOVKW $(3905<<16), R21                  // MOVKW $255918080, R21         // 35e8a172
   // MOVK $(3905<<32), R21                   // MOVK $16771847290880, R21     // 35e8c1f2
   MOVD $0, R5                                // 050080d2
   // MRS $4567, R16                          // f03a32d5
   // MRS $32345, R6                          // 26cb3fd5
   // MSR R25, $3452                          // 99af11d5
   // MSR R25, $16896                         // 194018d5
   // MSR $6, DAIFClr                         // ff4603d5
   MSUBW R1, R1, R12, R5                      // 8585011b
   MSUB R19, R16, R26, R2                     // 42c3139b
   MULW R26, R5, R22                          // b67c1a1b
   MUL R4, R3, R0                             // 607c049b
   MVNW R3@>13, R8                            // e837e32a
   MVN R13>>31, R9                            // e97f6daa
   NEGSW R23<<1, R30                          // fe07176b
   NEGS R20>>35, R22                          // f68f54eb
   NGCW R13, R8                               // e8030d5a
   NGC R2, R7                                 // e70302da
   NGCSW R10, R5                              // e5030a7a
   NGCS R24, R16                              // f00318fa
   //TODO NOP                                 // 1f2003d5
   ORNW R4@>11, R16, R3                       // 032ee42a
   ORN R22@>19, R3, R3                        // 634cf6aa
   ORRW $4294443071, R15, R24                 // f8490d32
   ORR $-3458764513820540929, R12, R22        // 96f542b2
   ORRW R13<<4, R8, R26                       // 1a110d2a
   ORR R3<<22, R5, R6                         // a65803aa
   //TODO PRFM 0x6400(R7), PSTL2STRM          // f300b2f9
   //TODO PRFM -215799(PC), PLIL2KEEP         // 2aa196d8
   //TODO PRFUM 0x42(R14), #0X06              // c62184f8
   RBITW R9, R22                              // 3601c05a
   RBIT R11, R4                               // 6401c0da
   RET                                        // c0035fd6
   REVW R8, R10                               // 0a09c05a
   REV R1, R2                                 // 220cc0da
   REV16W R21, R18                            // b206c05a
   REV16 R25, R4                              // 2407c0da
   REV32 R27, R21                             // 750bc0da
   EXTRW $27, R4, R25, R19                    // 336f8413
   EXTR $17, R10, R29, R15                    // af47ca93
   ROR $14, R14, R15                          // cf39ce93
   RORW $28, R14, R15                         // cf718e13
   RORW R3, R12, R3                           // 832dc31a
   ROR R0, R23, R2                            // e22ec09a
   SBCW R4, R8, R24                           // 1801045a
   SBC R25, R10, R26                          // 5a0119da
   SBCSW R27, R18, R18                        // 52021b7a
   SBCS R5, R9, R5                            // 250105fa
   SBFIZW $9, R10, $18, R22                   // 56451713
   SBFIZ $6, R11, $15, R20                    // 74397a93
   SBFXW $8, R15, $10, R20                    // f4450813
   SBFX $2, R27, $54, R7                      // 67df4293
   SDIVW R22, R14, R9                         // c90dd61a
   SDIV R13, R21, R9                          // a90ecd9a
   SEV                                        // 9f2003d5
   SEVL                                       // bf2003d5
   SMADDL R3, R7, R11, R9                     // 691d239b
   SMSUBL R5, R19, R11, R29                   // 7dcd259b
   SMNEGL R26, R3, R15                        // 6ffc3a9b
   SMULH R17, R21, R21                        // b57e519b
   SMULL R0, R5, R0                           // a07c209b
   // SMC $37977                              // 238b12d4
   STLRW R16, (R22)                           // d0fe9f88
   STLR R3, (R24)                             // 03ff9fc8
   //TODO STLRB R11, (R22)                    // cbfe9f08
   //TODO STLRH R16, (R23)                    // f0fe9f48
   STLXR R7, (R27), R8                        // 67ff08c8
   STLXRW R13, (R15), R14                     // edfd0e88
   STLXRB R24, (R23), R8                      // f8fe0808
   STLXRH R19, (R27), R11                     // 73ff0b48
   //TODO STLXPW (R22), R11, R6, R21          // c6ae3588
   //TODO STLXP (R22), LR, R6, R2             // c6fa22c8
   //TODO STNPW 44(R1), R3, R10               // 2a8c0528
   //TODO STNP 0x108(R3), ZR, R7              // 67fc10a8
   LDP.P -384(R3), (R22, R26)                 // 7668e8a8
   LDP.W 280(R8), (R18, R11)                  // 12add1a9
   STP.P (R22, R27), 352(R0)                  // 166c96a8
   STP.W (R17, R11), 96(R8)                   // 112d86a9
   MOVW.P R20, -28(R1)                        // 34441eb8
   MOVD.P R17, 191(R16)                       // 11f60bf8
   MOVW.W R1, -171(R14)                       // c15d15b8
   MOVD.W R14, -220(R13)                      // ae4d12f8
   MOVW R3, 14828(R24)                        // 03ef39b9
   MOVD R0, 20736(R17)                        // 208228f9
   MOVB.P ZR, -117(R7)                        // ffb41838
   MOVB.W R27, -96(R13)                       // bb0d1a38
   MOVB R17, 2200(R13)                        // b1612239
   MOVH.P R7, -72(R4)                         // 87841b78
   MOVH.W R12, -125(R14)                      // cc3d1878
   MOVH R19, 3686(R26)                        // 53cf1c79
   MOVW R21, 34(R0)                           // 152002b8
   MOVD R25, -137(R17)                        // 397217f8
   MOVH R11, -80(R23)                         // eb021b78
   //TODO MOVB R18, R0(R4)                    // 92682038
   //TODO MOVB R1, R6(R4)                     // 81682638
   //TODO MOVH R3, R13<<1(R11)                // 63792d78
   //TODO STTR 55(R4), R29                    // 9d7803b8
   //TODO STTR 124(R5), R25                   // b9c807f8
   //TODO STTRB -28(R23), R16                 // f04a1e38
   //TODO STTRH 9(R10), R18                   // 52990078
   //TODO STXP (R20), R18, R5, ZR             // 854a3f88
   //TODO STXP (R22), R9, R17, R0             // d12620c8
   // STXRW R2, (R19), R18                    // 627e1288
   // STXR R15, (R21), R13                    // af7e0dc8
   // STXRB R7, (R9), R24                     // 277d1808
   // STXRH R12, (R3), R8                     // 6c7c0848
   //TODO SUBW R20.UXTW<<7, R23, R18          // f25e344b
   //TODO SUB R5.SXTW<<2, R1, R26             // 3ac825cb
   SUB $(1923<<12), R4, R27                   // SUB $7876608, R4, R27         // 9b0c5ed1
   SUBW $(1923<<12), R4, R27                  // SUBW $7876608, R4, R27        // 9b0c5e51
   SUBW R12<<29, R7, R8                       // e8740c4b
   SUB R12<<61, R7, R8                        // e8f40ccb
   //TODO SUBSW R2.SXTH<<3, R13, R6           // a6ad226b
   //TODO SUBS R21.UXTX<<5, R27, R4           // 647735eb
   SUBSW $(44<<12), R6, R9                    // SUBSW $180224, R6, R9         // c9b04071
   SUBS $(1804<<12), R13, R9                  // SUBS $7389184, R13, R9        // a9315cf1
   SUBSW R22->28, R6, R7                      // c770966b
   SUBSW R22>>28, R6, R7                      // c770566b
   SUBS R26<<15, R6, R16                      // d03c1aeb
   SVC $0                                     // 010000d4
   SVC $7165                                  // a17f03d4
   SXTBW R8, R25                              // 191d0013
   SXTB R13, R9                               // a91d4093
   SXTHW R8, R8                               // 083d0013
   SXTH R17, R25                              // 393e4093
   SXTW R0, R27                               // 1b7c4093
   SYSL $285440, R12                          // 0c5b2cd5
   //TODO TLBI
   //TODO TST $0x80000007, R9                 // 3f0d0172
   //TODO TST $0xfffffff0, LR                 // df6f7cf2
   //TODO TSTW R10@>21, R2                    // 1f2f11ea
   //TODO TST R17<<11, R24                    // 1f2f11ea
   UBFIZW $3, R19, $14, R14                   // 6e361d53
   UBFIZ $3, R22, $14, R4                     // c4367dd3
   UBFXW $3, R7, $20, R15                     // ef580353
   UBFX $33, R17, $25, R5                     // 25e661d3
   UDIVW R8, R21, R15                         // af0ac81a
   UDIV R2, R18, R21                          // 550ac29a
   UMADDL R0, R20, R17, R17                   // 3152a09b
   UMSUBL R22, R4, R3, R7                     // 6790b69b
   UMNEGL R3, R18, R1                         // 41fea39b
   UMULH R24, R20, R24                        // 987ed89b
   UMULL R18, R22, R19                        // d37eb29b
   UXTBW R2, R6                               // 461c0053
   UXTHW R7, R20                              // f43c0053
   WFE                                        // 5f2003d5
   WFI                                        // 7f2003d5
   YIELD                                      // 3f2003d5
   //TODO FADD V21.D2, V10.D2, V21.D2         // 55d5754e
   FADDS F12, F2, F10                         // 4a282c1e
   FADDD F24, F14, F12                        // cc29781e
   FCCMPS LE, F17, F12, $14                   // 8ed5311e
   FCCMPD HI, F11, F15, $15                   // ef856b1e
   FCCMPES HS, F28, F13, $13                  // bd253c1e
   FCCMPED LT, F20, F4, $9                    // 99b4741e
   // FCMPS F3, F17                           // 2022231e
   // FCMPS $(0.0), F8                        // 0821201e
   // FCMPD F11, F27                          // 60236b1e
   // FCMPD $(0.0), F25                       // 2823601e
   // FCMPES F16, F30                         // d023301e
   // FCMPES $(0.0), F29                      // b823201e
   // FCMPED F13, F10                         // 50216d1e
   // FCMPED $(0.0), F25                      // 3823601e
   // FCSELS EQ, F26, F27, F25                // 590f3b1e
   // FCSELD PL, F8, F22, F7                  // 075d761e
   //TODO FCVTASW F21, R15                    // af02241e
   //TODO FCVTAS F20, ZR                      // 9f02249e
   //TODO FCVTASW F6, R11                     // cb00641e
   //TODO FCVTAS F6, R1                       // c100649e
   //TODO FCVTAUW F19, R26                    // 7a02251e
   //TODO FCVTAU F6, R5                       // c500259e
   //TODO FCVTAUW F6, R23                     // d700651e
   //TODO FCVTAU F27, R5                      // 6503659e
   //TODO FCVTMSW F15, R6                     // e601301e
   //TODO FCVTMS F15, ZR                      // ff01309e
   //TODO FCVTMSW F1, R14                     // 2e00701e
   //TODO FCVTMS F21, R9                      // a902709e
   //TODO FCVTMUW F20, R28                    // 9c02311e
   //TODO FCVTMU F23, R14                     // ee02319e
   //TODO FCVTMUW F18, R28                    // 5c02711e
   //TODO FCVTMU F24, R6                      // 0603719e
   //TODO FCVTNSW F12, R13                    // 8d01201e
   //TODO FCVTNS F9, R26                      // 3a01209e
   //TODO FCVTNSW F14, R8                     // c801601e
   //TODO FCVTNS F28, R10                     // 8a03609e
   //TODO FCVTNUW F22, R30                    // de02211e
   //TODO FCVTNU F20, R4                      // 8402219e
   //TODO FCVTNUW F18, R27                    // 5b02611e
   //TODO FCVTNU F21, R0                      // a002619e
   //TODO FCVTPSW F20, R6                     // 8602281e
   //TODO FCVTPS F12, R20                     // 9401289e
   //TODO FCVTPSW F22, R6                     // c602681e
   //TODO FCVTPS F21, R28                     // bc02689e
   //TODO FCVTPUW F24, R26                    // 1a03291e
   //TODO FCVTPU F16, R13                     // 0d02299e
   //TODO FCVTPUW F21, R29                    // bd02691e
   //TODO FCVTPU F11, R7                      // 6701699e
   FCVTZSSW F7, R15                           // ef00381e
   FCVTZSS F16, ZR                            // 1f02389e
   FCVTZSDW F19, R3                           // 6302781e
   FCVTZSD F7, R7                             // e700789e
   FCVTZUSW F2, R9                            // 4900391e
   FCVTZUS F12, R29                           // 9d01399e
   FCVTZUDW F27, R22                          // 7603791e
   FCVTZUD F25, R22                           // 3603799e
   //TODO FCVTZS $63, R18, R28                // 5c06189e
   //TODO FCVTZS $41, R11, R17                // 715d589e
   //TODO FCVTZU $1, ZR, R5                   // e5ff199e
   //TODO FCVTZUW $5, ZR, R20                 // f4ef591e
   //TODO FCVTZU $31, R3, R7                  // 6784599e
   FDIVS F16, F10, F20                        // 5419301e
   FDIVD F11, F25, F30                        // 3e1b6b1e
   //TODO FMADD R2, R15, R8, R1               // 01090f1f
   //TODO FMADD R21, R15, R25, R9             // 29574f1f
   FMAXS F5, F28, F27                         // 9b4b251e
   FMAXD F12, F31, F31                        // ff4b6c1e
   FMAXNMS F11, F24, F12                      // 0c6b2b1e
   FMAXNMD F20, F6, F16                       // d068741e
   FMINS F26, F18, F30                        // 5e5a3a1e
   FMIND F29, F4, F21                         // 95587d1e
   FMINNMS F23, F20, F1                       // 817a371e
   FMINNMD F8, F3, F24                        // 7878681e
   // FMOVS $(-1.625), F13                    // 0d503f1e
   // FMOVD $12.5, F30                        // 1e30651e
   //TODO FMOV R7, V25.D[1]                   // f900af9e
   FMOVD F2, R15                              // 4f00669e
   FMOVD R3, F11                              // 6b00679e
   FMOVS F20, R29                             // 9d02261e
   FMOVS R8, F15                              // 0f01271e
   FMOVD F2, F9                               // 4940601e
   FMOVS F4, F27                              // 9b40201e
   //TODO FMOV $3.125, V8.2D                  // 28f5006f
   //TODO FMSUB R21, R13, R13, R19            // b3d50d1f
   //TODO FMSUB R7, R11, R15, ZR              // ff9d4b1f
   FMULS F0, F6, F24                          // d808201e
   FMULD F5, F29, F9                          // a90b651e
   //TODO FNMADD R22, R17, R6, R20            // d458311f
   //TODO FNMADD R0, R15, R26, R20            // 54036f1f
   //TODO FNMSUB R16, R14, R27, R14           // 6ec32e1f
   //TODO FNMSUB R25, R29, R8, R10            // 0ae57d1f
   FNMULS F24, F22, F18                       // d28a381e
   FNMULD F14, F30, F7                        // c78b6e1e
   FSQRTS F0, F9                              // 09c0211e
   FSQRTD F14, F27                            // dbc1611e
   FSUBS F25, F23, F0                         // e03a391e
   FSUBD F11, F13, F24                        // b8396b1e
   //TODO SCVTFSS F30, F20                    // d4db215e
   //TODO SCVTF V7.2S, V17.2S                 // f1d8210e
   SCVTFWS R3, F16                            // 7000221e
   SCVTFWD R20, F4                            // 8402621e
   SCVTFS R16, F12                            // 0c02229e
   SCVTFD R26, F14                            // 4e03629e
   UCVTFWS R6, F4                             // c400231e
   UCVTFWD R10, F23                           // 5701631e
   UCVTFS R24, F29                            // 1d03239e
   UCVTFD R20, F11                            // 8b02639e

   RET
