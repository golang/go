// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package loong64 implements an LoongArch64 assembler. Go assembly syntax is different from
GNU LoongArch64 syntax, but we can still follow the general rules to map between them.

# Instructions mnemonics mapping rules

1. Bit widths represented by various instruction suffixes and prefixes
V (vlong)     = 64 bit
WU (word)     = 32 bit unsigned
W (word)      = 32 bit
H (half word) = 16 bit
HU            = 16 bit unsigned
B (byte)      = 8 bit
BU            = 8 bit unsigned
F (float)     = 32 bit float
D (double)    = 64 bit float

V  (LSX)      = 128 bit
XV (LASX)     = 256 bit

Examples:

	MOVB  (R2), R3  // Load 8 bit memory data into R3 register
	MOVH  (R2), R3  // Load 16 bit memory data into R3 register
	MOVW  (R2), R3  // Load 32 bit memory data into R3 register
	MOVV  (R2), R3  // Load 64 bit memory data into R3 register
	VMOVQ  (R2), V1 // Load 128 bit memory data into V1 register
	XVMOVQ (R2), X1 // Load 256 bit memory data into X1 register

2. Align directive
Go asm supports the PCALIGN directive, which indicates that the next instruction should
be aligned to a specified boundary by padding with NOOP instruction. The alignment value
supported on loong64 must be a power of 2 and in the range of [8, 2048].

Examples:

	PCALIGN	$16
	MOVV	$2, R4	// This instruction is aligned with 16 bytes.
	PCALIGN	$1024
	MOVV	$3, R5	// This instruction is aligned with 1024 bytes.

# On loong64, auto-align loop heads to 16-byte boundaries

Examples:

	TEXT ·Add(SB),NOSPLIT|NOFRAME,$0

start:

	MOVV	$1, R4	// This instruction is aligned with 16 bytes.
	MOVV	$-1, R5
	BNE	R5, start
	RET

# Register mapping rules

1. All generial-prupose register names are written as Rn.

2. All floating-point register names are written as Fn.

3. All LSX register names are written as Vn.

4. All LASX register names are written as Xn.

# Argument mapping rules

1. The operands appear in left-to-right assignment order.

Go reverses the arguments of most instructions.

Examples:

	ADDV	R11, R12, R13 <=> add.d R13, R12, R11
	LLV	(R4), R7      <=> ll.d R7, R4
	OR	R5, R6        <=> or R6, R6, R5

Special Cases.
(1) Argument order is the same as in the GNU Loong64 syntax: jump instructions,

Examples:

	BEQ	R0, R4, lable1  <=>  beq R0, R4, lable1
	JMP	lable1          <=>  b lable1

(2) BSTRINSW, BSTRINSV, BSTRPICKW, BSTRPICKV $<msb>, <Rj>, $<lsb>, <Rd>

Examples:

	BSTRPICKW $15, R4, $6, R5  <=>  bstrpick.w r5, r4, 15, 6

2. Expressions for special arguments.

Memory references: a base register and an offset register is written as (Rbase)(Roff).

Examples:

	MOVB (R4)(R5), R6  <=>  ldx.b R6, R4, R5
	MOVV (R4)(R5), R6  <=>  ldx.d R6, R4, R5
	MOVD (R4)(R5), F6  <=>  fldx.d F6, R4, R5
	MOVB R6, (R4)(R5)  <=>  stx.b R6, R5, R5
	MOVV R6, (R4)(R5)  <=>  stx.d R6, R5, R5
	MOVV F6, (R4)(R5)  <=>  fstx.d F6, R5, R5

3. Alphabetical list of SIMD instructions

Note: In the following sections 3.1 to 3.6, "ui4" (4-bit unsigned int immediate),
"ui3", "ui2", and "ui1" represent the related "index".

3.1 Move general-purpose register to a vector element:

	Instruction format:
	        VMOVQ  Rj, <Vd>.<T>[index]

	Mapping between Go and platform assembly:
	       Go assembly       |      platform assembly     |          semantics
	-------------------------------------------------------------------------------------
	 VMOVQ  Rj, Vd.B[index]  |  vinsgr2vr.b  Vd, Rj, ui4  |  VR[vd].b[ui4] = GR[rj][7:0]
	 VMOVQ  Rj, Vd.H[index]  |  vinsgr2vr.h  Vd, Rj, ui3  |  VR[vd].h[ui3] = GR[rj][15:0]
	 VMOVQ  Rj, Vd.W[index]  |  vinsgr2vr.w  Vd, Rj, ui2  |  VR[vd].w[ui2] = GR[rj][31:0]
	 VMOVQ  Rj, Vd.V[index]  |  vinsgr2vr.d  Vd, Rj, ui1  |  VR[vd].d[ui1] = GR[rj][63:0]
	XVMOVQ  Rj, Xd.W[index]  | xvinsgr2vr.w  Xd, Rj, ui3  |  XR[xd].w[ui3] = GR[rj][31:0]
	XVMOVQ  Rj, Xd.V[index]  | xvinsgr2vr.d  Xd, Rj, ui2  |  XR[xd].d[ui2] = GR[rj][63:0]

3.2 Move vector element to general-purpose register

	Instruction format:
	        VMOVQ     <Vj>.<T>[index], Rd

	Mapping between Go and platform assembly:
	        Go assembly       |       platform assembly      |            semantics
	---------------------------------------------------------------------------------------------
	 VMOVQ  Vj.B[index],  Rd  |   vpickve2gr.b   rd, vj, ui4 | GR[rd] = SignExtend(VR[vj].b[ui4])
	 VMOVQ  Vj.H[index],  Rd  |   vpickve2gr.h   rd, vj, ui3 | GR[rd] = SignExtend(VR[vj].h[ui3])
	 VMOVQ  Vj.W[index],  Rd  |   vpickve2gr.w   rd, vj, ui2 | GR[rd] = SignExtend(VR[vj].w[ui2])
	 VMOVQ  Vj.V[index],  Rd  |   vpickve2gr.d   rd, vj, ui1 | GR[rd] = SignExtend(VR[vj].d[ui1])
	 VMOVQ  Vj.BU[index], Rd  |   vpickve2gr.bu  rd, vj, ui4 | GR[rd] = ZeroExtend(VR[vj].bu[ui4])
	 VMOVQ  Vj.HU[index], Rd  |   vpickve2gr.hu  rd, vj, ui3 | GR[rd] = ZeroExtend(VR[vj].hu[ui3])
	 VMOVQ  Vj.WU[index], Rd  |   vpickve2gr.wu  rd, vj, ui2 | GR[rd] = ZeroExtend(VR[vj].wu[ui2])
	 VMOVQ  Vj.VU[index], Rd  |   vpickve2gr.du  rd, vj, ui1 | GR[rd] = ZeroExtend(VR[vj].du[ui1])
	XVMOVQ  Xj.W[index],  Rd  |  xvpickve2gr.w   rd, xj, ui3 | GR[rd] = SignExtend(VR[xj].w[ui3])
	XVMOVQ  Xj.V[index],  Rd  |  xvpickve2gr.d   rd, xj, ui2 | GR[rd] = SignExtend(VR[xj].d[ui2])
	XVMOVQ  Xj.WU[index], Rd  |  xvpickve2gr.wu  rd, xj, ui3 | GR[rd] = ZeroExtend(VR[xj].wu[ui3])
	XVMOVQ  Xj.VU[index], Rd  |  xvpickve2gr.du  rd, xj, ui2 | GR[rd] = ZeroExtend(VR[xj].du[ui2])

3.3 Duplicate general-purpose register to vector.

	Instruction format:
	        VMOVQ    Rj, <Vd>.<T>

	Mapping between Go and platform assembly:
	   Go assembly      |    platform assembly    |                    semantics
	------------------------------------------------------------------------------------------------
	 VMOVQ  Rj, Vd.B16  |   vreplgr2vr.b  Vd, Rj  |  for i in range(16): VR[vd].b[i] = GR[rj][7:0]
	 VMOVQ  Rj, Vd.H8   |   vreplgr2vr.h  Vd, Rj  |  for i in range(8) : VR[vd].h[i] = GR[rj][16:0]
	 VMOVQ  Rj, Vd.W4   |   vreplgr2vr.w  Vd, Rj  |  for i in range(4) : VR[vd].w[i] = GR[rj][31:0]
	 VMOVQ  Rj, Vd.V2   |   vreplgr2vr.d  Vd, Rj  |  for i in range(2) : VR[vd].d[i] = GR[rj][63:0]
	XVMOVQ  Rj, Xd.B32  |  xvreplgr2vr.b  Xd, Rj  |  for i in range(32): XR[xd].b[i] = GR[rj][7:0]
	XVMOVQ  Rj, Xd.H16  |  xvreplgr2vr.h  Xd, Rj  |  for i in range(16): XR[xd].h[i] = GR[rj][16:0]
	XVMOVQ  Rj, Xd.W8   |  xvreplgr2vr.w  Xd, Rj  |  for i in range(8) : XR[xd].w[i] = GR[rj][31:0]
	XVMOVQ  Rj, Xd.V4   |  xvreplgr2vr.d  Xd, Rj  |  for i in range(4) : XR[xd].d[i] = GR[rj][63:0]

3.4 Replace vector elements

	Instruction format:
	        XVMOVQ    Xj, <Xd>.<T>

	Mapping between Go and platform assembly:
	   Go assembly      |   platform assembly   |                semantics
	------------------------------------------------------------------------------------------------
	XVMOVQ  Xj, Xd.B32  |  xvreplve0.b  Xd, Xj  | for i in range(32): XR[xd].b[i] = XR[xj].b[0]
	XVMOVQ  Xj, Xd.H16  |  xvreplve0.h  Xd, Xj  | for i in range(16): XR[xd].h[i] = XR[xj].h[0]
	XVMOVQ  Xj, Xd.W8   |  xvreplve0.w  Xd, Xj  | for i in range(8) : XR[xd].w[i] = XR[xj].w[0]
	XVMOVQ  Xj, Xd.V4   |  xvreplve0.d  Xd, Xj  | for i in range(4) : XR[xd].d[i] = XR[xj].d[0]
	XVMOVQ  Xj, Xd.Q2   |  xvreplve0.q  Xd, Xj  | for i in range(2) : XR[xd].q[i] = XR[xj].q[0]

3.5 Move vector element to scalar

	Instruction format:
	        XVMOVQ  Xj, <Xd>.<T>[index]
	        XVMOVQ  Xj.<T>[index], Xd

	Mapping between Go and platform assembly:
	       Go assembly        |     platform assembly     |               semantics
	------------------------------------------------------------------------------------------------
	 XVMOVQ  Xj, Xd.W[index]  |  xvinsve0.w   xd, xj, ui3 | XR[xd].w[ui3] = XR[xj].w[0]
	 XVMOVQ  Xj, Xd.V[index]  |  xvinsve0.d   xd, xj, ui2 | XR[xd].d[ui2] = XR[xj].d[0]
	 XVMOVQ  Xj.W[index], Xd  |  xvpickve.w   xd, xj, ui3 | XR[xd].w[0] = XR[xj].w[ui3], XR[xd][255:32] = 0
	 XVMOVQ  Xj.V[index], Xd  |  xvpickve.d   xd, xj, ui2 | XR[xd].d[0] = XR[xj].d[ui2], XR[xd][255:64] = 0

3.6 Move vector element to vector register.

	Instruction format:
	VMOVQ     <Vn>.<T>[index], Vn.<T>

	Mapping between Go and platform assembly:
	         Go assembly      |    platform assembly   |               semantics
	VMOVQ Vj.B[index], Vd.B16 | vreplvei.b vd, vj, ui4 | for i in range(16): VR[vd].b[i] = VR[vj].b[ui4]
	VMOVQ Vj.H[index], Vd.H8  | vreplvei.h vd, vj, ui3 | for i in range(8) : VR[vd].h[i] = VR[vj].h[ui3]
	VMOVQ Vj.W[index], Vd.W4  | vreplvei.w vd, vj, ui2 | for i in range(4) : VR[vd].w[i] = VR[vj].w[ui2]
	VMOVQ Vj.V[index], Vd.V2  | vreplvei.d vd, vj, ui1 | for i in range(2) : VR[vd].d[i] = VR[vj].d[ui1]

3.7 Load data from memory and broadcast to each element of a vector register.

	Instruction format:
	        VMOVQ    offset(Rj), <Vd>.<T>

	Mapping between Go and platform assembly:
	   Go assembly              |     platform assembly      |                                semantics
	-------------------------------------------------------------------------------------------------------------------------------------------------------
	 VMOVQ  offset(Rj), Vd.B16  |   vldrepl.b  Vd, Rj, si12  |  for i in range(16): VR[vd].b[i] = load 8 bit memory data from (GR[rj]+SignExtend(si12))
	 VMOVQ  offset(Rj), Vd.H8   |   vldrepl.h  Vd, Rj, si11  |  for i in range(8) : VR[vd].h[i] = load 16 bit memory data from (GR[rj]+SignExtend(si11<<1))
	 VMOVQ  offset(Rj), Vd.W4   |   vldrepl.w  Vd, Rj, si10  |  for i in range(4) : VR[vd].w[i] = load 32 bit memory data from (GR[rj]+SignExtend(si10<<2))
	 VMOVQ  offset(Rj), Vd.V2   |   vldrepl.d  Vd, Rj, si9   |  for i in range(2) : VR[vd].d[i] = load 64 bit memory data from (GR[rj]+SignExtend(si9<<3))
	XVMOVQ  offset(Rj), Xd.B32  |  xvldrepl.b  Xd, Rj, si12  |  for i in range(32): XR[xd].b[i] = load 8 bit memory data from (GR[rj]+SignExtend(si12))
	XVMOVQ  offset(Rj), Xd.H16  |  xvldrepl.h  Xd, Rj, si11  |  for i in range(16): XR[xd].h[i] = load 16 bit memory data from (GR[rj]+SignExtend(si11<<1))
	XVMOVQ  offset(Rj), Xd.W8   |  xvldrepl.w  Xd, Rj, si10  |  for i in range(8) : XR[xd].w[i] = load 32 bit memory data from (GR[rj]+SignExtend(si10<<2))
	XVMOVQ  offset(Rj), Xd.V4   |  xvldrepl.d  Xd, Rj, si9   |  for i in range(4) : XR[xd].d[i] = load 64 bit memory data from (GR[rj]+SignExtend(si9<<3))

	note: In Go assembly, for ease of understanding, offset representing the actual address offset.
	      However, during platform encoding, the offset is shifted to increase the encodable offset range, as follows:

	   Go assembly           |      platform assembly
         VMOVQ  1(R4), V5.B16    |      vldrepl.b  v5, r4, $1
         VMOVQ  2(R4), V5.H8     |      vldrepl.h  v5, r4, $1
         VMOVQ  8(R4), V5.W4     |      vldrepl.w  v5, r4, $2
         VMOVQ  8(R4), V5.V2     |      vldrepl.d  v5, r4, $1

# Special instruction encoding definition and description on LoongArch

 1. DBAR hint encoding for LA664(Loongson 3A6000) and later micro-architectures, paraphrased
    from the Linux kernel implementation: https://git.kernel.org/torvalds/c/e031a5f3f1ed

    - Bit4: ordering or completion (0: completion, 1: ordering)
    - Bit3: barrier for previous read (0: true, 1: false)
    - Bit2: barrier for previous write (0: true, 1: false)
    - Bit1: barrier for succeeding read (0: true, 1: false)
    - Bit0: barrier for succeeding write (0: true, 1: false)
    - Hint 0x700: barrier for "read after read" from the same address

    Traditionally, on microstructures that do not support dbar grading such as LA464
    (Loongson 3A5000, 3C5000) all variants are treated as “dbar 0” (full barrier).

2. Notes on using atomic operation instructions

  - AM*_DB.W[U]/V[U] instructions such as AMSWAPDBW not only complete the corresponding
    atomic operation sequence, but also implement the complete full data barrier function.

  - When using the AM*_.W[U]/D[U] instruction, registers rd and rj cannot be the same,
    otherwise an exception is triggered, and rd and rk cannot be the same, otherwise
    the execution result is uncertain.

3. Prefetch instructions
    Instruction format:
      PRELD	offset(Rbase), $hint
      PRELDX	offset(Rbase), $n, $hint

    Mapping between Go and platform assembly:
               Go assembly            |    platform assembly
      PRELD  offset(Rbase), $hint     | preld hint, Rbase, offset
      PRELDX offset(Rbase), $n, $hint | move rk, $x; preldx hint, Rbase, rk

      note: $x is the value after $n and offset are reassembled

    Definition of hint value:
      0: load to L1
      2: load to L3
      8: store to L1

      The meaning of the rest of values is not defined yet, and the processor executes it as NOP

    Definition of $n in the PRELDX instruction:
      bit[0]: address sequence, 0 indicating ascending and 1 indicating descending
      bits[11:1]:  block size, the value range is [16, 1024], and it must be an integer multiple of 16
      bits[20:12]: block num, the value range is [1, 256]
      bits[36:21]: stride, the value range is [0, 0xffff]

4. ShiftAdd instructions
    Mapping between Go and platform assembly:
                Go assembly            |    platform assembly
     ALSL.W/WU/V $Imm, Rj, Rk, Rd      |    alsl.w/wu/d rd, rj, rk, $imm

    Instruction encoding format is as follows:

	| 31 ~ 17 | 16 ~ 15 | 14 ~ 10 | 9 ~ 5 | 4 ~ 0 |
	|  opcode |   sa2   |   rk    |   rj  |   rd  |

    The alsl.w/wu/v series of instructions shift the data in rj left by sa+1, add the value
    in rk, and write the result to rd.

    To allow programmers to directly write the desired shift amount in assembly code, we actually write
    the value of sa2+1 in the assembly code and then include the value of sa2 in the instruction encoding.

    For example:

            Go assembly      | instruction Encoding
        ALSLV $4, r4, r5, R6 |      002d9486

5. Note of special memory access instructions
    Instruction format:
      MOVWP	offset(Rj), Rd
      MOVVP	offset(Rj), Rd
      MOVWP	Rd, offset(Rj)
      MOVVP	Rd, offset(Rj)

    Mapping between Go and platform assembly:
               Go assembly      |      platform assembly
      MOVWP  offset(Rj), Rd     |    ldptr.w  rd, rj, si14
      MOVVP  offset(Rj), Rd     |    ldptr.d  rd, rj, si14
      MOVWP  Rd, offset(Rj)     |    stptr.w  rd, rj, si14
      MOVVP  Rd, offset(Rj)     |    stptr.d  rd, rj, si14

      note: In Go assembly, for ease of understanding, offset is a 16-bit immediate number representing
            the actual address offset, but in platform assembly, it need a 14-bit immediate number.
	    si14 = offset>>2

    The addressing calculation for the above instruction involves logically left-shifting the 14-bit
    immediate number si14 by 2 bits, then sign-extending it, and finally adding it to the value in the
    general-purpose register rj to obtain the sum.

    For example:

            Go assembly      |      platform assembly
         MOVWP  8(R4), R5    |      ldptr.w r5, r4, $2

6. Note of special add instrction
    Mapping between Go and platform assembly:
              Go assembly        |      platform assembly
      ADDV16  si16<<16, Rj, Rd   |    addu16i.d  rd, rj, si16

      note: si16 is a 16-bit immediate number, and si16<<16 is the actual operand.

    The addu16i.d instruction logically left-shifts the 16-bit immediate number si16 by 16 bits, then
    sign-extends it. The resulting data is added to the [63:0] bits of data in the general-purpose register
    rj, and the sum is written into the general-purpose register rd.
    The addu16i.d instruction is used in conjunction with the ldptr.w/d and stptr.w/d instructions to
    accelerate access based on the GOT table in position-independent code.
*/

package loong64
