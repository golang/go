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
*/
package loong64
