// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package ppc64 implements a PPC64 assembler that assembles Go asm into
the corresponding PPC64 instructions as defined by the Power ISA 3.0B.

This document provides information on how to write code in Go assembler
for PPC64, focusing on the differences between Go and PPC64 assembly language.
It assumes some knowledge of PPC64 assembler. The original implementation of
PPC64 in Go defined many opcodes that are different from PPC64 opcodes, but
updates to the Go assembly language used mnemonics that are mostly similar if not
identical to the PPC64 mneumonics, such as VMX and VSX instructions. Not all detail
is included here; refer to the Power ISA document if interested in more detail.

Starting with Go 1.15 the Go objdump supports the -gnu option, which provides a
side by side view of the Go assembler and the PPC64 assembler output. This is
extremely helpful in determining what final PPC64 assembly is generated from the
corresponding Go assembly.

In the examples below, the Go assembly is on the left, PPC64 assembly on the right.

1. Operand ordering

In Go asm, the last operand (right) is the target operand, but with PPC64 asm,
the first operand (left) is the target. The order of the remaining operands is
not consistent: in general opcodes with 3 operands that perform math or logical
operations have their operands in reverse order. Opcodes for vector instructions
and those with more than 3 operands usually have operands in the same order except
for the target operand, which is first in PPC64 asm and last in Go asm.

Example:

	ADD R3, R4, R5		<=>	add r5, r4, r3

2. Constant operands

In Go asm, an operand that starts with '$' indicates a constant value. If the
instruction using the constant has an immediate version of the opcode, then an
immediate value is used with the opcode if possible.

Example:

	ADD $1, R3, R4		<=> 	addi r4, r3, 1

3. Opcodes setting condition codes

In PPC64 asm, some instructions other than compares have variations that can set
the condition code where meaningful. This is indicated by adding '.' to the end
of the PPC64 instruction. In Go asm, these instructions have 'CC' at the end of
the opcode. The possible settings of the condition code depend on the instruction.
CR0 is the default for fixed-point instructions; CR1 for floating point; CR6 for
vector instructions.

Example:

	ANDCC R3, R4, R5		<=>	and. r5, r3, r4 (set CR0)

4. Loads and stores from memory

In Go asm, opcodes starting with 'MOV' indicate a load or store. When the target
is a memory reference, then it is a store; when the target is a register and the
source is a memory reference, then it is a load.

MOV{B,H,W,D} variations identify the size as byte, halfword, word, doubleword.

Adding 'Z' to the opcode for a load indicates zero extend; if omitted it is sign extend.
Adding 'U' to a load or store indicates an update of the base register with the offset.
Adding 'BR' to an opcode indicates byte-reversed load or store, or the order opposite
of the expected endian order. If 'BR' is used then zero extend is assumed.

Memory references n(Ra) indicate the address in Ra + n. When used with an update form
of an opcode, the value in Ra is incremented by n.

Memory references (Ra+Rb) or (Ra)(Rb) indicate the address Ra + Rb, used by indexed
loads or stores. Both forms are accepted. When used with an update then the base register
is updated by the value in the index register.

Examples:

	MOVD (R3), R4		<=>	ld r4,0(r3)
	MOVW (R3), R4		<=>	lwa r4,0(r3)
	MOVWZU 4(R3), R4		<=>	lwzu r4,4(r3)
	MOVWZ (R3+R5), R4		<=>	lwzx r4,r3,r5
	MOVHZ  (R3), R4		<=>	lhz r4,0(r3)
	MOVHU 2(R3), R4		<=>	lhau r4,2(r3)
	MOVBZ (R3), R4		<=>	lbz r4,0(r3)

	MOVD R4,(R3)		<=>	std r4,0(r3)
	MOVW R4,(R3)		<=>	stw r4,0(r3)
	MOVW R4,(R3+R5)		<=>	stwx r4,r3,r5
	MOVWU R4,4(R3)		<=>	stwu r4,4(r3)
	MOVH R4,2(R3)		<=>	sth r4,2(r3)
	MOVBU R4,(R3)(R5)		<=>	stbux r4,r3,r5

4. Compares

When an instruction does a compare or other operation that might
result in a condition code, then the resulting condition is set
in a field of the condition register. The condition register consists
of 8 4-bit fields named CR0 - CR7. When a compare instruction
identifies a CR then the resulting condition is set in that field
to be read by a later branch or isel instruction. Within these fields,
bits are set to indicate less than, greater than, or equal conditions.

Once an instruction sets a condition, then a subsequent branch, isel or
other instruction can read the condition field and operate based on the
bit settings.

Examples:

	CMP R3, R4			<=>	cmp r3, r4	(CR0 assumed)
	CMP R3, R4, CR1		<=>	cmp cr1, r3, r4

Note that the condition register is the target operand of compare opcodes, so
the remaining operands are in the same order for Go asm and PPC64 asm.
When CR0 is used then it is implicit and does not need to be specified.

5. Branches

Many branches are represented as a form of the BC instruction. There are
other extended opcodes to make it easier to see what type of branch is being
used.

The following is a brief description of the BC instruction and its commonly
used operands.

BC op1, op2, op3

	  op1: type of branch
	      16 -> bctr (branch on ctr)
	      12 -> bcr  (branch if cr bit is set)
	      8  -> bcr+bctr (branch on ctr and cr values)
		4  -> bcr != 0 (branch if specified cr bit is not set)

		There are more combinations but these are the most common.

	  op2: condition register field and condition bit

		This contains an immediate value indicating which condition field
		to read and what bits to test. Each field is 4 bits long with CR0
	      at bit 0, CR1 at bit 4, etc. The value is computed as 4*CR+condition
	      with these condition values:

	      0 -> LT
	      1 -> GT
	      2 -> EQ
	      3 -> OVG

		Thus 0 means test CR0 for LT, 5 means CR1 for GT, 30 means CR7 for EQ.

	  op3: branch target

Examples:

	BC 12, 0, target		<=>	blt cr0, target
	BC 12, 2, target		<=>	beq cr0, target
	BC 12, 5, target		<=>	bgt cr1, target
	BC 12, 30, target		<=>	beq cr7, target
	BC 4, 6, target		<=>	bne cr1, target
	BC 4, 1, target		<=>	ble cr1, target

	The following extended opcodes are available for ease of use and readability:

	BNE CR2, target		<=>	bne cr2, target
	BEQ CR4, target		<=>	beq cr4, target
	BLT target			<=>	blt target (cr0 default)
	BGE CR7, target		<=>	bge cr7, target

Refer to the ISA for more information on additional values for the BC instruction,
how to handle OVG information, and much more.

5. Align directive

Starting with Go 1.12, Go asm supports the PCALIGN directive, which indicates
that the next instruction should be aligned to the specified value. Currently
8 and 16 are the only supported values, and a maximum of 2 NOPs will be added
to align the code. That means in the case where the code is aligned to 4 but
PCALIGN $16 is at that location, the code will only be aligned to 8 to avoid
adding 3 NOPs.

The purpose of this directive is to improve performance for cases like loops
where better alignment (8 or 16 instead of 4) might be helpful. This directive
exists in PPC64 assembler and is frequently used by PPC64 assembler writers.

PCALIGN $16
PCALIGN $8

By default, functions in Go are aligned to 16 bytes, as is the case in all
other compilers for PPC64. If there is a PCALIGN directive requesting alignment
greater than 16, then the alignment of the containing function must be
promoted to that same alignment or greater.

The behavior of PCALIGN is changed in Go 1.21 to be more straightforward to
ensure the alignment required for some instructions in power10. The acceptable
values are 8, 16, 32 and 64, and the use of those values will always provide the
specified alignment.

6. Shift instructions

The simple scalar shifts on PPC64 expect a shift count that fits in 5 bits for
32-bit values or 6 bit for 64-bit values. If the shift count is a constant value
greater than the max then the assembler sets it to the max for that size (31 for
32 bit values, 63 for 64 bit values). If the shift count is in a register, then
only the low 5 or 6 bits of the register will be used as the shift count. The
Go compiler will add appropriate code to compare the shift value to achieve the
correct result, and the assembler does not add extra checking.

Examples:

	SRAD $8,R3,R4		=>	sradi r4,r3,8
	SRD $8,R3,R4		=>	rldicl r4,r3,56,8
	SLD $8,R3,R4		=>	rldicr r4,r3,8,55
	SRAW $16,R4,R5		=>	srawi r5,r4,16
	SRW $40,R4,R5		=>	rlwinm r5,r4,0,0,31
	SLW $12,R4,R5		=>	rlwinm r5,r4,12,0,19

Some non-simple shifts have operands in the Go assembly which don't map directly
onto operands in the PPC64 assembly. When an operand in a shift instruction in the
Go assembly is a bit mask, that mask is represented as a start and end bit in the
PPC64 assembly instead of a mask. See the ISA for more detail on these types of shifts.
Here are a few examples:

	RLWMI $7,R3,$65535,R6 	=>	rlwimi r6,r3,7,16,31
	RLDMI $0,R4,$7,R6 		=>	rldimi r6,r4,0,61

More recently, Go opcodes were added which map directly onto the PPC64 opcodes. It is
recommended to use the newer opcodes to avoid confusion.

	RLDICL $0,R4,$15,R6		=>	rldicl r6,r4,0,15
	RLDICR $0,R4,$15,R6		=>	rldicr r6.r4,0,15

# Register naming

1. Special register usage in Go asm

The following registers should not be modified by user Go assembler code.

	R0: Go code expects this register to contain the value 0.
	R1: Stack pointer
	R2: TOC pointer when compiled with -shared or -dynlink (a.k.a position independent code)
	R13: TLS pointer
	R30: g (goroutine)

Register names:

	Rn is used for general purpose registers. (0-31)
	Fn is used for floating point registers. (0-31)
	Vn is used for vector registers. Slot 0 of Vn overlaps with Fn. (0-31)
	VSn is used for vector-scalar registers. V0-V31 overlap with VS32-VS63. (0-63)
	CTR represents the count register.
	LR represents the link register.
	CR represents the condition register
	CRn represents a condition register field. (0-7)
	CRnLT represents CR bit 0 of CR field n. (0-7)
	CRnGT represents CR bit 1 of CR field n. (0-7)
	CRnEQ represents CR bit 2 of CR field n. (0-7)
	CRnSO represents CR bit 3 of CR field n. (0-7)

# GOPPC64 >= power10 and its effects on Go asm

When GOPPC64=power10 is used to compile a Go program for ppc64le/linux, MOV*, FMOV*, and ADD
opcodes which would require 2 or more machine instructions to emulate a 32 bit constant, or
symbolic reference are implemented using prefixed instructions.

A user who wishes granular control over the generated machine code is advised to use Go asm
opcodes which explicitly translate to one PPC64 machine instruction. Most common opcodes
are supported.

Some examples of how pseudo-op assembly changes with GOPPC64:

	Go asm                       GOPPC64 <= power9          GOPPC64 >= power10
	MOVD mypackage·foo(SB), R3   addis r2, r3, ...          pld r3, ...
	                             ld    r3, r3, ...

	MOVD 131072(R3), R4          addis r31, r4, 2           pld r4, 131072(r3)
	                             ld    r4, 0(R3)

	ADD $131073, R3              lis  r31, 2                paddi r3, r3, 131073
	                             addi r31, 1
	                             add  r3,r31,r3

	MOVD $131073, R3             lis  r3, 2                 pli r3, 131073
	                             addi r3, 1

	MOVD $mypackage·foo(SB), R3  addis r2, r3, ...          pla r3, ...
	                             addi  r3, r3, ...
*/
package ppc64
