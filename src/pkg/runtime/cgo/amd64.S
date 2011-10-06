// Copyright 2009 The Go Authors.  All rights reserved.
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
 * void crosscall_amd64(void (*fn)(void))
 *
 * Calling into the 6c tool chain, where all registers are caller save.
 * Called from standard x86-64 ABI, where %rbx, %rbp, %r12-%r15
 * are callee-save so they must be saved explicitly.
 * The standard x86-64 ABI passes the three arguments m, g, fn
 * in %rdi, %rsi, %rdx.
 *
 * Also need to set %r15 to g and %r14 to m (see ../pkg/runtime/mkasmh.sh)
 * during the call.
 */
.globl EXT(crosscall_amd64)
EXT(crosscall_amd64):
	pushq %rbx
	pushq %rbp
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15

#if defined(_WIN64)
	call *%rcx	/* fn */
#else
	call *%rdi	/* fn */
#endif

	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbp
	popq %rbx
	ret

/*
 * void crosscall2(void (*fn)(void*, int32), void *arg, int32 argsize)
 *
 * Save registers and call fn with two arguments.  fn is a Go function
 * which takes parameters on the stack rather than in registers.
 */
.globl EXT(crosscall2)
EXT(crosscall2):
	subq  $0x58, %rsp	/* keeps stack pointer 32-byte aligned */
	movq  %rbx, 0x10(%rsp)
	movq  %rbp, 0x18(%rsp)
	movq  %r12, 0x20(%rsp)
	movq  %r13, 0x28(%rsp)
	movq  %r14, 0x30(%rsp)
	movq  %r15, 0x38(%rsp)

#if defined(_WIN64)
	// Win64 save RBX, RBP, RDI, RSI, RSP, R12, R13, R14, and R15
	movq	%rdi, 0x40(%rsp)
	movq	%rsi, 0x48(%rsp)

	movq  %rdx, 0(%rsp)	/* arg */
	movq  %r8, 8(%rsp)	/* argsize (includes padding) */
	
	call %rcx	/* fn */
#else
	movq  %rsi, 0(%rsp)	/* arg */
	movq  %rdx, 8(%rsp)	/* argsize (includes padding) */

	call *%rdi	/* fn */
#endif

	movq  0x10(%rsp), %rbx
	movq  0x18(%rsp), %rbp
	movq  0x20(%rsp), %r12
	movq  0x28(%rsp), %r13
	movq  0x30(%rsp), %r14
	movq  0x38(%rsp), %r15
#if defined(__WIN64)
	movq	0x40(%rsp), %rdi
	movq	0x48(%rsp), %rsi
#endif	
	addq  $0x58, %rsp
	ret
