// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package arm64 implements an ARM64 assembler. Go assembly syntax is different from GNU ARM64
syntax, but we can still follow the general rules to map between them.

Instructions mnemonics mapping rules

1. Most instructions use width suffixes of instruction names to indicate operand width rather than
using different register names.

  Examples:
    ADC R24, R14, R12          <=>     adc x12, x24
    ADDW R26->24, R21, R15     <=>     add w15, w21, w26, asr #24
    FCMPS F2, F3               <=>     fcmp s3, s2
    FCMPD F2, F3               <=>     fcmp d3, d2
    FCVTDH F2, F3              <=>     fcvt h3, d2

2. Go uses .P and .W suffixes to indicate post-increment and pre-increment.

  Examples:
    MOVD.P -8(R10), R8         <=>      ldr x8, [x10],#-8
    MOVB.W 16(R16), R10        <=>      ldr x10, [x16,#16]!

3. Go uses a series of MOV instructions as load and store.

64-bit variant ldr, str, stur => MOVD;
32-bit variant str, stur, ldrsw => MOVW;
32-bit variant ldr => MOVWU;
ldrb => MOVBU; ldrh => MOVHU;
ldrsb, sturb, strb => MOVB;
ldrsh, sturh, strh =>  MOVH.

4. Go moves conditions into opcode suffix, like BLT.

5. Go adds a V prefix for most floating-point and SIMD instructions, except cryptographic extension
instructions and floating-point(scalar) instructions.

  Examples:
    VADD V5.H8, V18.H8, V9.H8         <=>      add v9.8h, v18.8h, v5.8h
    VLD1.P (R6)(R11), [V31.D1]        <=>      ld1 {v31.1d}, [x6], x11
    VFMLA V29.S2, V20.S2, V14.S2      <=>      fmla v14.2s, v20.2s, v29.2s
    AESD V22.B16, V19.B16             <=>      aesd v19.16b, v22.16b
    SCVTFWS R3, F16                   <=>      scvtf s17, w6

Special Cases.

(1) umov is written as VMOV.

(2) br is renamed JMP, blr is renamed CALL.

(3) No need to add "W" suffix: LDARB, LDARH, LDAXRB, LDAXRH, LDTRH, LDXRB, LDXRH.

(4) In Go assembly syntax, NOP is a zero-width pseudo-instruction serves generic purpose, nothing
related to real ARM64 instruction. NOOP serves for the hardware nop instruction. NOOP is an alias of
HINT $0.

  Examples:
    VMOV V13.B[1], R20      <=>      mov x20, v13.b[1]
    VMOV V13.H[1], R20      <=>      mov w20, v13.h[1]
    JMP (R3)                <=>      br x3
    CALL (R17)              <=>      blr x17
    LDAXRB (R19), R16       <=>      ldaxrb w16, [x19]
    NOOP                    <=>      nop


Register mapping rules

1. All basic register names are written as Rn.

2. Go uses ZR as the zero register and RSP as the stack pointer.

3. Bn, Hn, Dn, Sn and Qn instructions are written as Fn in floating-point instructions and as Vn
in SIMD instructions.


Argument mapping rules

1. The operands appear in left-to-right assignment order.

Go reverses the arguments of most instructions.

    Examples:
      ADD R11.SXTB<<1, RSP, R25      <=>      add x25, sp, w11, sxtb #1
      VADD V16, V19, V14             <=>      add d14, d19, d16

Special Cases.

(1) Argument order is the same as in the GNU ARM64 syntax: cbz, cbnz and some store instructions,
such as str, stur, strb, sturb, strh, sturh stlr, stlrb. stlrh, st1.

  Examples:
    MOVD R29, 384(R19)    <=>    str x29, [x19,#384]
    MOVB.P R30, 30(R4)    <=>    strb w30, [x4],#30
    STLRH R21, (R19)      <=>    stlrh w21, [x19]

(2) MADD, MADDW, MSUB, MSUBW, SMADDL, SMSUBL, UMADDL, UMSUBL <Rm>, <Ra>, <Rn>, <Rd>

  Examples:
    MADD R2, R30, R22, R6       <=>    madd x6, x22, x2, x30
    SMSUBL R10, R3, R17, R27    <=>    smsubl x27, w17, w10, x3

(3) FMADDD, FMADDS, FMSUBD, FMSUBS, FNMADDD, FNMADDS, FNMSUBD, FNMSUBS <Fm>, <Fa>, <Fn>, <Fd>

  Examples:
    FMADDD F30, F20, F3, F29    <=>    fmadd d29, d3, d30, d20
    FNMSUBS F7, F25, F7, F22    <=>    fnmsub s22, s7, s7, s25

(4) BFI, BFXIL, SBFIZ, SBFX, UBFIZ, UBFX $<lsb>, <Rn>, $<width>, <Rd>

  Examples:
    BFIW $16, R20, $6, R0      <=>    bfi w0, w20, #16, #6
    UBFIZ $34, R26, $5, R20    <=>    ubfiz x20, x26, #34, #5

(5) FCCMPD, FCCMPS, FCCMPED, FCCMPES <cond>, Fm. Fn, $<nzcv>

  Examples:
    FCCMPD AL, F8, F26, $0     <=>    fccmp d26, d8, #0x0, al
    FCCMPS VS, F29, F4, $4     <=>    fccmp s4, s29, #0x4, vs
    FCCMPED LE, F20, F5, $13   <=>    fccmpe d5, d20, #0xd, le
    FCCMPES NE, F26, F10, $0   <=>    fccmpe s10, s26, #0x0, ne

(6) CCMN, CCMNW, CCMP, CCMPW <cond>, <Rn>, $<imm>, $<nzcv>

  Examples:
    CCMP MI, R22, $12, $13     <=>    ccmp x22, #0xc, #0xd, mi
    CCMNW AL, R1, $11, $8      <=>    ccmn w1, #0xb, #0x8, al

(7) CCMN, CCMNW, CCMP, CCMPW <cond>, <Rn>, <Rm>, $<nzcv>

  Examples:
    CCMN VS, R13, R22, $10     <=>    ccmn x13, x22, #0xa, vs
    CCMPW HS, R19, R14, $11    <=>    ccmp w19, w14, #0xb, cs

(9) CSEL, CSELW, CSNEG, CSNEGW, CSINC, CSINCW <cond>, <Rn>, <Rm>, <Rd> ;
FCSELD, FCSELS <cond>, <Fn>, <Fm>, <Fd>

  Examples:
    CSEL GT, R0, R19, R1        <=>    csel x1, x0, x19, gt
    CSNEGW GT, R7, R17, R8      <=>    csneg w8, w7, w17, gt
    FCSELD EQ, F15, F18, F16    <=>    fcsel d16, d15, d18, eq

(10) TBNZ, TBZ $<imm>, <Rt>, <label>


(11) STLXR, STLXRW, STXR, STXRW, STLXRB, STLXRH, STXRB, STXRH  <Rf>, (<Rn|RSP>), <Rs>

  Examples:
    STLXR ZR, (R15), R16    <=>    stlxr w16, xzr, [x15]
    STXRB R9, (R21), R19    <=>    stxrb w19, w9, [x21]

(12) STLXP, STLXPW, STXP, STXPW (<Rf1>, <Rf2>), (<Rn|RSP>), <Rs>

  Examples:
    STLXP (R17, R19), (R4), R5      <=>    stlxp w5, x17, x19, [x4]
    STXPW (R30, R25), (R22), R13    <=>    stxp w13, w30, w25, [x22]

2. Expressions for special arguments.

#<immediate> is written as $<immediate>.

Optionally-shifted immediate.

  Examples:
    ADD $(3151<<12), R14, R20     <=>    add x20, x14, #0xc4f, lsl #12
    ADDW $1864, R25, R6           <=>    add w6, w25, #0x748

Optionally-shifted registers are written as <Rm>{<shift><amount>}.
The <shift> can be <<(lsl), >>(lsr), ->(asr), @>(ror).

  Examples:
    ADD R19>>30, R10, R24     <=>    add x24, x10, x19, lsr #30
    ADDW R26->24, R21, R15    <=>    add w15, w21, w26, asr #24

Extended registers are written as <Rm>{.<extend>{<<<amount>}}.
<extend> can be UXTB, UXTH, UXTW, UXTX, SXTB, SXTH, SXTW or SXTX.

  Examples:
    ADDS R19.UXTB<<4, R9, R26     <=>    adds x26, x9, w19, uxtb #4
    ADDSW R14.SXTX, R14, R6       <=>    adds w6, w14, w14, sxtx

Memory references: [<Xn|SP>{,#0}] is written as (Rn|RSP), a base register and an immediate
offset is written as imm(Rn|RSP), a base register and an offset register is written as (Rn|RSP)(Rm).

  Examples:
    LDAR (R22), R9                  <=>    ldar x9, [x22]
    LDP 28(R17), (R15, R23)         <=>    ldp x15, x23, [x17,#28]
    MOVWU (R4)(R12<<2), R8          <=>    ldr w8, [x4, x12, lsl #2]
    MOVD (R7)(R11.UXTW<<3), R25     <=>    ldr x25, [x7,w11,uxtw #3]
    MOVBU (R27)(R23), R14           <=>    ldrb w14, [x27,x23]

Register pairs are written as (Rt1, Rt2).

  Examples:
    LDP.P -240(R11), (R12, R26)    <=>    ldp x12, x26, [x11],#-240

Register with arrangement and register with arrangement and index.

  Examples:
    VADD V5.H8, V18.H8, V9.H8                     <=>    add v9.8h, v18.8h, v5.8h
    VLD1 (R2), [V21.B16]                          <=>    ld1 {v21.16b}, [x2]
    VST1.P V9.S[1], (R16)(R21)                    <=>    st1 {v9.s}[1], [x16], x28
    VST1.P [V13.H8, V14.H8, V15.H8], (R3)(R14)    <=>    st1 {v13.8h-v15.8h}, [x3], x14
    VST1.P [V14.D1, V15.D1], (R7)(R23)            <=>    st1 {v14.1d, v15.1d}, [x7], x23
*/
package arm64
