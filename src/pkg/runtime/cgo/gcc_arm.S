// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Apple still insists on underscore prefixes for C function names.
 */
#if defined(__APPLE__)
#define EXT(s) _##s
#else
#define EXT(s) s
#endif

/*
 * void crosscall_arm2(void (*fn)(void), void *g, void *m)
 *
 * Calling into the 5c tool chain, where all registers are caller save.
 * Called from standard ARM EABI, where r4-r11 are callee-save, so they
 * must be saved explicitly.
 */
.globl EXT(crosscall_arm2)
EXT(crosscall_arm2):
	push {r4, r5, r6, r7, r8, r9, r10, r11, ip, lr}
	mov r10, r1 // g
	mov r9, r2 // m
	mov r3, r0 // save r0, cgo_tls_set_gm will clobber it
	bl EXT(x_cgo_save_gm) // save current g and m into TLS variable
	mov lr, pc
	mov pc, r3
	pop {r4, r5, r6, r7, r8, r9, r10, r11, ip, pc}

.globl EXT(__stack_chk_fail_local)
EXT(__stack_chk_fail_local):
1:
	b 1b

#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
