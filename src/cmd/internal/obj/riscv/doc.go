// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package riscv implements the riscv64 assembler.

# Register naming

The integer registers are named X0 through to X31, however X4 must be accessed
through its RISC-V ABI name, TP, and X27, which holds a pointer to the Go
routine structure, must be referred to as g. Additionally, when building in
shared mode, X3 is unavailable and must be accessed via its RISC-V ABI name,
GP.

The floating-point registers are named F0 through to F31.

The vector registers are named V0 through to V31.

Both integer and floating-point registers can be referred to by their RISC-V
ABI names, e.g., A0 or FT0, with the exception that X27 cannot be referred to
by its RISC-V ABI name, S11.  It must be referred to as g.

Some of the integer registers are used by the Go runtime and assembler - X26 is
the closure pointer, X27 points to the Go routine structure and X31 is a
temporary register used by the Go assembler. Use of X31 should be avoided in
hand written assembly code as its value could be altered by the instruction
sequences emitted by the assembler.

# Instruction naming

Many RISC-V instructions contain one or more suffixes in their names. In the
[RISC-V ISA Manual] these suffixes are separated from themselves and the
name of the instruction mnemonic with a dot ('.'). In the Go assembler, the
separators are omitted and the suffixes are written in upper case.

Example:

	FMVWX           <=>     fmv.w.x

# Rounding modes

The Go toolchain does not set the FCSR register and requires the desired
rounding mode to be explicitly encoded within floating-point instructions.
The syntax the Go assembler uses to specify the rounding modes differs
from the syntax in the RISC-V specifications. In the [RISC-V ISA Manual]
the rounding mode is given as an extra operand at the end of an
assembly language instruction. In the Go assembler, the rounding modes are
converted to uppercase and follow the instruction mnemonic from which they
are separated with a dot ('.').

Example:

	FCVTLUS.RNE F0, X5      <=>     fcvt.lu.s x5, f0, rne

RTZ is assumed if the rounding mode is omitted.

# RISC-V extensions

By default the Go compiler targets the [rva20u64] profile. This profile mandates
all the general RISC-V instructions, allowing Go to use integer, multiplication,
division, floating-point and atomic instructions without having to
perform compile time or runtime checks to verify that their use is appropriate
for the target hardware. All widely available riscv64 devices support at least
[rva20u64]. The Go toolchain can be instructed to target later RISC-V profiles,
including, [rva22u64] and [rva23u64], via the GORISCV64 environment variable.
Instructions that are provided by newer profiles cannot typically be used in
handwritten assembly code without compile time guards (or runtime checks)
that ensure they are hardware supported.

The file asm_riscv64.h defines macros for each RISC-V extension that is enabled
by setting the GORISCV64 environment variable to a value other than [rva20u64].
For example, if GORISCV64=rva22u64 the macros hasZba, hasZbb and hasZbs will be
defined. If GORISCV64=rva23u64 hasV will be defined in addition to hasZba,
hasZbb and hasZbs. These macros can be used to determine whether it's safe
to use an instruction in hand-written assembly.

It is not always necessary to include asm_riscv64.h and use #ifdefs in your
code to safely take advantage of instructions present in the [rva22u64]
profile. In some cases the assembler can generate [rva20u64] compatible code
even when an [rva22u64] instruction is used in an assembly source file. When
GORISCV64=rva20u64 the assembler will synthesize certain [rva22u64]
instructions, e.g., ANDN, using multiple [rva20u64] instructions. Instructions
such as ANDN can then be freely used in assembly code without checking to see
whether the instruction is supported by the target profile. When building a
source file containing the ANDN instruction with GORISCV64=rva22u64 the
assembler will emit the Zbb ANDN instruction directly. When building the same
source file with GORISCV64=rva20u64 the assembler will emit multiple [rva20u64]
instructions to synthesize ANDN.

The assembler will also use [rva22u64] instructions to implement the zero and
sign extension instructions, e.g., MOVB and MOVHU, when GORISCV64=rva22u64 or
greater.

The instructions not implemented in the default profile ([rva20u64]) that can
be safely used in assembly code without compile time checks are:

  - ANDN
  - MAX
  - MAXU
  - MIN
  - MINU
  - MOVB
  - MOVH
  - MOVHU
  - MOVWU
  - ORN
  - ROL
  - ROLW
  - ROR
  - RORI
  - RORIW
  - RORW
  - XNOR

# Operand ordering

The ordering used for instruction operands in the Go assembler differs from the
ordering defined in the [RISC-V ISA Manual].

1. R-Type instructions

R-Type instructions are written in the reverse order to that given in the
[RISC-V ISA Manual], with the register order being rs2, rs1, rd.

Examples:

	ADD X10, X11, X12       <=>     add x12, x11, x10
	FADDD F10, F11, F12     <=>     fadd.d f12, f11, f10

2. I-Type arithmetic instructions

I-Type arithmetic instructions (not loads, fences, ebreak, ecall) use the same
ordering as the R-Type instructions, typically, imm12, rs1, rd.

Examples:

	ADDI $1, X11, X12       <=>     add x12, x11, 1
	SLTI $1, X11, X12       <=>     slti x12, x11, 1

3. Loads and Stores

Load instructions are written with the source operand (whether it be a register
or a memory address), first followed by the destination operand.

Examples:

	MOV 16(X2), X10         <=>     ld x10, 16(x2)
	MOV X10, (X2)           <=>     sd x10, 0(x2)

4. Branch instructions

The branch instructions use the same operand ordering as is given in the
[RISC-V ISA Manual], e.g., rs1, rs2, label.

Example:

	BLT X12, X23, loop1     <=>     blt x12, x23, loop1

BLT X12, X23, label will jump to label if X12 < X23. Note this is not the
same ordering as is used for the SLT instructions.

5. FMA instructions

The Go assembler uses a different ordering for the RISC-V FMA operands to
the ordering given in the [RISC-V ISA Manual]. The operands are rotated one
place to the left, so that the destination operand comes last.

Example:

	FMADDS  F1, F2, F3, F4  <=>     fmadd.s f4, f1, f2, f3

6. AMO instructions

The ordering used for the AMO operations is rs2, rs1, rd, i.e., the operands
as specified in the [RISC-V ISA Manual] are rotated one place to the left.

Example:

	AMOSWAPW X5, (X6), X7   <=>     amoswap.w x7, x5, (x6)

7. Vector instructions

The VSETVLI instruction uses the same symbolic names as the [RISC-V ISA Manual]
to represent the components of vtype, with the exception
that they are written in upper case. The ordering of the operands in the Go
assembler differs from the [RISC-V ISA Manual] in that the operands are
rotated one place to the left so that the destination register, the register
that holds the new vl, is the last operand.

Example:

	VSETVLI X10, E8, M1, TU, MU, X12        <=>     vsetvli x12, x10, e8, m1, tu, mu

Vector load and store instructions follow the pattern set by scalar loads and
stores, i.e., the source is always the first operand and the destination the
last. However, the ordering of the operands of these instructions is
complicated by the optional mask register and, in some cases, the use of an
additional stride or index register. In the Go assembler the index and stride
registers appear as the second operand in indexed or strided loads and stores,
while the mask register, if present, is always the penultimate operand.

Examples:

	VLE8V (X10), V3                 <=>     vle8.v  v3, (x10)
	VSE8V V3, (X10)                 <=>     vse8.v  v3, (x10)
	VLE8V (X10), V0, V3             <=>     vle8.v  v3, (x10), v0.t
	VSE8V V3, V0, (X10)             <=>     vse8.v  v3, (x10), v0.t
	VLSE8V (X10), X11, V3           <=>     vlse8.v v3, (x10), x11
	VSSE8V V3, X11, (X10)           <=>     vsse8.v v3, (x10), x11
	VLSE8V (X10), X11, V0, V3       <=>     vlse8.v v3, (x10), x11, v0.t
	VSSE8V V3, X11, V0, (X10)       <=>     vsse8.v v3, (x10), x11, v0.t
	VLUXEI8V (X10), V2, V3          <=>     vluxei8.v v3, (x10), v2
	VSUXEI8V V3, V2, (X10)          <=>     vsuxei8.v v3, (x10), v2
	VLUXEI8V (X10), V2, V0, V3      <=>     vluxei8.v v3, (x10), v2, v0.t
	VSUXEI8V V3, V2, V0, (X10)      <=>     vsuxei8.v v3, (x10), v2, v0.t
	VL1RE8V (X10), V3               <=>     vl1re8.v v3, (x10)
	VS1RV V3, (X11)                 <=>     vs1r.v  v3, (x11)

The ordering of operands for two and three argument vector arithmetic instructions is
reversed in the Go assembler.

Examples:

	VMVVV V2, V3                    <=> vmv.v.v v3, v2
	VADDVV V1, V2, V3               <=> vadd.vv v3, v2, v1
	VADDVX X10, V2, V3              <=> vadd.vx v3, v2, x10
	VMADCVI $15, V2, V3             <=> vmadc.vi v3, v2, 15

The mask register, when specified, is always the penultimate operand in a vector
arithmetic instruction, appearing before the destination register.

Examples:

	VANDVV V1, V2, V0, V3           <=> vand.vv v3, v2, v1, v0.t

# Ternary instructions

The Go assembler allows the second operand to be omitted from most ternary
instructions if it matches the third (destination) operand.

Examples:

	ADD X10, X12, X12       <=>     ADD X10, X12
	ANDI $3, X12, X12       <=>     ANDI $3, X12

The use of this abbreviated syntax is encouraged.

# Ordering of atomic instructions

It is not possible to specify the ordering bits in the FENCE, LR, SC or AMO
instructions.  The FENCE instruction is always emitted as a full fence, the
acquire and release bits are always set for the AMO instructions, the acquire
bit is always set for the LR instructions while the release bit is set for
the SC instructions.

# Immediate operands

In many cases, where an R-Type instruction has a corresponding I-Type
instruction, the R-Type mnemonic can be used in place of the I-Type mnemonic.
The assembler assumes that the immediate form of the instruction was intended
when the first operand is given as an immediate value rather than a register.

Example:

	AND $3, X12, X13        <=>     ANDI $3, X12, X13

# Integer constant materialization

The MOV instruction can be used to set a register to the value of any 64 bit
constant literal. The way this is achieved by the assembler varies depending
on the value of the constant. Where possible the assembler will synthesize the
constant using one or more RISC-V arithmetic instructions. If it is unable
to easily materialize the constant it will load the 64 bit literal from memory.

A 32 bit constant literal can be specified as an argument to ADDI, ANDI, ORI and
XORI. If the specified literal does not fit into 12 bits the assembler will
generate extra instructions to synthesize it.

Integer constants provided as operands to all other instructions must fit into
the number of bits allowed by the instructions' encodings for immediate values.
Otherwise, an error will be generated.

# Floating point constant materialization

The MOVF and MOVD instructions can be used to set a register to the value
of any 32 bit or 64 bit floating point constant literal, respectively.  Unless
the constant literal is 0.0, MOVF and MOVD will be encoded as FLW and FLD
instructions that load the constant from a location within the program's
binary.

# Compressed instructions

The Go assembler converts 32 bit RISC-V instructions to compressed
instructions when generating machine code. This conversion happens
automatically without the need for any direct involvement from the programmer,
although judicious choice of registers can improve the compression rate for
certain instructions (see the [RISC-V ISA Manual] for more details). This
behaviour is enabled by default for all of the supported RISC-V profiles, i.e.,
it is not affected by the value of the GORISCV64 environment variable.

The use of compressed instructions can be disabled via a debug flag,
compressinstructions:

  - Use -gcflags=all=-d=compressinstructions=0 to disable compressed
    instructions in Go code.
  - Use -asmflags=all=-d=compressinstructions=0 to disable compressed
    instructions in assembly code.

To completely disable automatic instruction compression in a Go binary both
options must be specified.

The assembler also permits the use of compressed instructions in hand coded
assembly language, but this should generally be avoided. Note that the
compressinstructions flag only prevents the automatic compression of 32
bit instructions. It has no effect on compressed instructions that are
hand coded directly into an assembly file.

[RISC-V ISA Manual]: https://github.com/riscv/riscv-isa-manual
[rva20u64]: https://github.com/riscv/riscv-profiles/blob/main/src/profiles.adoc#51-rva20u64-profile
[rva22u64]: https://github.com/riscv/riscv-profiles/blob/main/src/profiles.adoc#rva22u64-profile
[rva23u64]: https://github.com/riscv/riscv-profiles/blob/main/src/rva23-profile.adoc#rva23u64-profile
*/
package riscv
