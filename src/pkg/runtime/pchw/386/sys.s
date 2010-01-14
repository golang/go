// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Manipulation of segment tables.
//
// Descriptor entry format for system call
// is the native machine format, ugly as it is:
//
//	2-byte limit
//	3-byte base
//	1-byte: 0x80=present, 0x60=dpl<<5, 0x1F=type
//	1-byte: 0x80=limit is *4k, 0x40=32-bit operand size,
//		0x0F=4 more bits of limit
//	1 byte: 8 more bits of base

// Called to set up memory hardware.
// Already running in 32-bit mode thanks to boot block,
// but we need to install our new GDT that we can modify.
TEXT msetup(SB), 7, $0
	MOVL	gdtptr(SB), GDTR
	MOVL	$(1*8+0), AX
	MOVW	AX, DS
	MOVW	AX, ES
	MOVW	AX, SS
	MOVW	$0, AX
	MOVW	AX, FS
	MOVW	AX, GS

	// long jmp to cs:mret
	BYTE	$0xEA
	LONG $mret(SB)
	WORD $(2*8+0)
	
TEXT mret(SB), 7, $0
	RET

// GDT memory
TEXT gdt(SB), 7, $0
	// null segment
	LONG	$0
	LONG	$0
	
	// 4GB data segment
	LONG	$0xFFFF
	LONG	$0x00CF9200

	// 4GB code segment
	LONG	$0xFFFF
	LONG	$0x00CF9A00

	// null segment (will be thread-local storage segment)
	LONG	$0
	LONG	$0

// GDT pseudo-descriptor
TEXT gdtptr(SB), 7, $0
	WORD	$(4*8)
	LONG	$gdt(SB)

// Called to establish the per-thread segment.
// Write to gdt[3] and reload the gdt register.
// setldt(int entry, int address, int limit)
TEXT setldt(SB),7,$32
	MOVL	address+4(FP), BX	// aka base
	MOVL	limit+8(FP), CX

	// set up segment descriptor
	LEAL	gdt+(3*8)(SB), AX	// gdt entry #3
	MOVL	$0, 0(AX)
	MOVL	$0, 4(AX)

	MOVW	BX, 2(AX)
	SHRL	$16, BX
	MOVB	BX, 4(AX)
	SHRL	$8, BX
	MOVB	BX, 7(AX)

	MOVW	CX, 0(AX)
	SHRL	$16, CX
	ANDL	$0x0F, CX
	ORL	$0x40, CX		// 32-bit operand size
	MOVB	CX, 6(AX)
	MOVB	$0xF2, 5(AX)	// r/w data descriptor, dpl=3, present

	MOVL	gdtptr(SB), GDTR

	// Compute segment selector - (entry*8+0)
	MOVL	$(3*8+0), AX
	MOVW	AX, GS
	RET

