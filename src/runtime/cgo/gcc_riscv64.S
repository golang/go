// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * void crosscall1(void (*fn)(void), void (*setg_gcc)(void *g), void *g)
 *
 * Calling into the gc tool chain, where all registers are caller save.
 * Called from standard RISCV ELF psABI, where x8-x9, x18-x27, f8-f9 and
 * f18-f27 are callee-save, so they must be saved explicitly, along with
 * x1 (LR), x3 (GP) and x4 (TP).
 */
.globl crosscall1
crosscall1:
	sd	x1, -216(sp)
	addi	sp, sp, -216
	sd	x3, 8(sp)
	sd	x4, 16(sp)
	sd	x8, 24(sp)
	sd	x9, 32(sp)
	sd	x18, 40(sp)
	sd	x19, 48(sp)
	sd	x20, 56(sp)
	sd	x21, 64(sp)
	sd	x22, 72(sp)
	sd	x23, 80(sp)
	sd	x24, 88(sp)
	sd	x25, 96(sp)
	sd	x26, 104(sp)
	sd	x27, 112(sp)
	fsd	f8, 120(sp)
	fsd	f9, 128(sp)
	fsd	f18, 136(sp)
	fsd	f19, 144(sp)
	fsd	f20, 152(sp)
	fsd	f21, 160(sp)
	fsd	f22, 168(sp)
	fsd	f23, 176(sp)
	fsd	f24, 184(sp)
	fsd	f25, 192(sp)
	fsd	f26, 200(sp)
	fsd	f27, 208(sp)

	// a0 = *fn, a1 = *setg_gcc, a2 = *g
	mv	s1, a0
	mv	s0, a1
	mv	a0, a2
	jalr	ra, s0	// call setg_gcc (clobbers x30 aka g)
	jalr	ra, s1	// call fn

	ld	x1, 0(sp)
	ld	x3, 8(sp)
	ld	x4, 16(sp)
	ld	x8, 24(sp)
	ld	x9, 32(sp)
	ld	x18, 40(sp)
	ld	x19, 48(sp)
	ld	x20, 56(sp)
	ld	x21, 64(sp)
	ld	x22, 72(sp)
	ld	x23, 80(sp)
	ld	x24, 88(sp)
	ld	x25, 96(sp)
	ld	x26, 104(sp)
	ld	x27, 112(sp)
	fld	f8, 120(sp)
	fld	f9, 128(sp)
	fld	f18, 136(sp)
	fld	f19, 144(sp)
	fld	f20, 152(sp)
	fld	f21, 160(sp)
	fld	f22, 168(sp)
	fld	f23, 176(sp)
	fld	f24, 184(sp)
	fld	f25, 192(sp)
	fld	f26, 200(sp)
	fld	f27, 208(sp)
	addi	sp, sp, 216

	jr	ra

#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
