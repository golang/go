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

/*
 * void crosscall2(void (*fn)(void*, int32), void*, int32)
 *
 * Save registers and call fn with two arguments.
 */
.globl EXT(crosscall2)
EXT(crosscall2):
	/* 
	 * We still need to save all callee save register as before, and then
	 *  push 2 args for fn (R1 and R2).
	 * Also note that at procedure entry in 5c/5g world, 4(R13) will be the
	 *  first arg, so we must push another dummy reg (R0) for 0(R13).
	 *  Additionally, cgo_tls_set_gm will clobber R0, so we need to save R0
	 *  nevertheless.
	 */
	push {r0, r1, r2, r4, r5, r6, r7, r8, r9, r10, r11, ip, lr}
	bl EXT(x_cgo_load_gm) // set up g and m from TLS
	mov lr, pc
	ldr pc, [sp, #0]
	pop {r0, r1, r2, r4, r5, r6, r7, r8, r9, r10, r11, ip, pc}

.globl EXT(__stack_chk_fail_local)
EXT(__stack_chk_fail_local):
1:
	b 1b

