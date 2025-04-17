// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

.file "gcc_loong64.S"

/*
 * void crosscall1(void (*fn)(void), void (*setg_gcc)(void *g), void *g)
 *
 * Calling into the gc tool chain, where all registers are caller save.
 * Called from standard lp64d ABI, where $r1, $r3, $r23-$r30, and $f24-$f31
 * are callee-save, so they must be saved explicitly, along with $r1 (LR).
 */
.globl crosscall1
crosscall1:
	addi.d	$r3, $r3, -160
	st.d	$r1, $r3, 0
	st.d	$r23, $r3, 8
	st.d	$r24, $r3, 16
	st.d	$r25, $r3, 24
	st.d	$r26, $r3, 32
	st.d	$r27, $r3, 40
	st.d	$r28, $r3, 48
	st.d	$r29, $r3, 56
	st.d	$r30, $r3, 64
	st.d	$r2, $r3, 72
	st.d	$r22, $r3, 80
	fst.d	$f24, $r3, 88
	fst.d	$f25, $r3, 96
	fst.d	$f26, $r3, 104
	fst.d	$f27, $r3, 112
	fst.d	$f28, $r3, 120
	fst.d	$f29, $r3, 128
	fst.d	$f30, $r3, 136
	fst.d	$f31, $r3, 144

	// r4 = *fn, r5 = *setg_gcc, r6 = *g
	move	$r23, $r4	// save R4
	move	$r4, $r6
	jirl	$r1, $r5, 0	// call setg_gcc (clobbers R4)
	jirl	$r1, $r23, 0	// call fn

	ld.d	$r23, $r3, 8
	ld.d	$r24, $r3, 16
	ld.d	$r25, $r3, 24
	ld.d	$r26, $r3, 32
	ld.d	$r27, $r3, 40
	ld.d	$r28, $r3, 48
	ld.d	$r29, $r3, 56
	ld.d	$r30, $r3, 64
	ld.d	$r2, $r3, 72
	ld.d	$r22, $r3, 80
	fld.d	$f24, $r3, 88
	fld.d	$f25, $r3, 96
	fld.d	$f26, $r3, 104
	fld.d	$f27, $r3, 112
	fld.d	$f28, $r3, 120
	fld.d	$f29, $r3, 128
	fld.d	$f30, $r3, 136
	fld.d	$f31, $r3, 144
	ld.d	$r1, $r3, 0
	addi.d	$r3, $r3, 160
	jirl	$r0, $r1, 0


#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
