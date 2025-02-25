// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

TEXT runtime·rt0_go(SB), NOSPLIT|NOFRAME|TOPFRAME, $0
	// save m->g0 = g0
	MOVD $runtime·g0(SB), runtime·m0+m_g0(SB)
	// save m0 to g0->m
	MOVD $runtime·m0(SB), runtime·g0+g_m(SB)
	// set g to g0
	MOVD $runtime·g0(SB), g
	CALLNORESUME runtime·check(SB)
#ifdef GOOS_js
	CALLNORESUME runtime·args(SB)
#endif
	CALLNORESUME runtime·osinit(SB)
	CALLNORESUME runtime·schedinit(SB)
	MOVD $runtime·mainPC(SB), 0(SP)
	CALLNORESUME runtime·newproc(SB)
	CALL runtime·mstart(SB) // WebAssembly stack will unwind when switching to another goroutine
	UNDEF

TEXT runtime·mstart(SB),NOSPLIT|TOPFRAME,$0
	CALL	runtime·mstart0(SB)
	RET // not reached

DATA  runtime·mainPC+0(SB)/8,$runtime·main(SB)
GLOBL runtime·mainPC(SB),RODATA,$8

// func checkASM() bool
TEXT ·checkASM(SB), NOSPLIT, $0-1
	MOVB $1, ret+0(FP)
	RET

TEXT runtime·gogo(SB), NOSPLIT, $0-8
	MOVD buf+0(FP), R0
	MOVD gobuf_g(R0), R1
	MOVD 0(R1), R2	// make sure g != nil
	MOVD R1, g
	MOVD gobuf_sp(R0), SP

	// Put target PC at -8(SP), wasm_pc_f_loop will pick it up
	Get SP
	I32Const $8
	I32Sub
	I64Load gobuf_pc(R0)
	I64Store $0

	MOVD gobuf_ctxt(R0), CTXT
	// clear to help garbage collector
	MOVD $0, gobuf_sp(R0)
	MOVD $0, gobuf_ctxt(R0)

	I32Const $1
	Return

// func mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return. It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), NOSPLIT, $0-8
	// CTXT = fn
	MOVD fn+0(FP), CTXT
	// R1 = g.m
	MOVD g_m(g), R1
	// R2 = g0
	MOVD m_g0(R1), R2

	// save state in g->sched
	MOVD 0(SP), g_sched+gobuf_pc(g)     // caller's PC
	MOVD $fn+0(FP), g_sched+gobuf_sp(g) // caller's SP

	// if g == g0 call badmcall
	Get g
	Get R2
	I64Eq
	If
		JMP runtime·badmcall(SB)
	End

	// switch to g0's stack
	I64Load (g_sched+gobuf_sp)(R2)
	I64Const $8
	I64Sub
	I32WrapI64
	Set SP

	// set arg to current g
	MOVD g, 0(SP)

	// switch to g0
	MOVD R2, g

	// call fn
	Get CTXT
	I32WrapI64
	I64Load $0
	CALL

	Get SP
	I32Const $8
	I32Add
	Set SP

	JMP runtime·badmcall2(SB)

// func systemstack(fn func())
TEXT runtime·systemstack(SB), NOSPLIT, $0-8
	// R0 = fn
	MOVD fn+0(FP), R0
	// R1 = g.m
	MOVD g_m(g), R1
	// R2 = g0
	MOVD m_g0(R1), R2

	// if g == g0
	Get g
	Get R2
	I64Eq
	If
		// no switch:
		MOVD R0, CTXT

		Get CTXT
		I32WrapI64
		I64Load $0
		JMP
	End

	// if g != m.curg
	Get g
	I64Load m_curg(R1)
	I64Ne
	If
		CALLNORESUME runtime·badsystemstack(SB)
		CALLNORESUME runtime·abort(SB)
	End

	// switch:

	// save state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	MOVD $runtime·systemstack_switch(SB), g_sched+gobuf_pc(g)

	MOVD SP, g_sched+gobuf_sp(g)

	// switch to g0
	MOVD R2, g

	// make it look like mstart called systemstack on g0, to stop traceback
	I64Load (g_sched+gobuf_sp)(R2)
	I64Const $8
	I64Sub
	Set R3

	MOVD $runtime·mstart(SB), 0(R3)
	MOVD R3, SP

	// call fn
	MOVD R0, CTXT

	Get CTXT
	I32WrapI64
	I64Load $0
	CALL

	// switch back to g
	MOVD g_m(g), R1
	MOVD m_curg(R1), R2
	MOVD R2, g
	MOVD g_sched+gobuf_sp(R2), SP
	MOVD $0, g_sched+gobuf_sp(R2)
	RET

TEXT runtime·systemstack_switch(SB), NOSPLIT, $0-0
	RET

TEXT runtime·abort(SB),NOSPLIT|NOFRAME,$0-0
	UNDEF

// AES hashing not implemented for wasm
TEXT runtime·memhash(SB),NOSPLIT|NOFRAME,$0-32
	JMP	runtime·memhashFallback(SB)
TEXT runtime·strhash(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·strhashFallback(SB)
TEXT runtime·memhash32(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·memhash32Fallback(SB)
TEXT runtime·memhash64(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·memhash64Fallback(SB)

TEXT runtime·asminit(SB), NOSPLIT, $0-0
	// No per-thread init.
	RET

TEXT ·publicationBarrier(SB), NOSPLIT, $0-0
	RET

TEXT runtime·procyield(SB), NOSPLIT, $0-0 // FIXME
	RET

TEXT runtime·breakpoint(SB), NOSPLIT, $0-0
	UNDEF

// func switchToCrashStack0(fn func())
TEXT runtime·switchToCrashStack0(SB), NOSPLIT, $0-8
	MOVD fn+0(FP), CTXT	// context register
	MOVD	g_m(g), R2	// curm

	// set g to gcrash
	MOVD	$runtime·gcrash(SB), g	// g = &gcrash
	MOVD	R2, g_m(g)	// g.m = curm
	MOVD	g, m_g0(R2)	// curm.g0 = g

	// switch to crashstack
	I64Load (g_stack+stack_hi)(g)
	I64Const $(-4*8)
	I64Add
	I32WrapI64
	Set SP

	// call target function
	Get CTXT
	I32WrapI64
	I64Load $0
	CALL

	// should never return
	CALL	runtime·abort(SB)
	UNDEF

// Called during function prolog when more stack is needed.
//
// The traceback routines see morestack on a g0 as being
// the top of a stack (for example, morestack calling newstack
// calling the scheduler calling newm calling gc), so we must
// record an argument size. For that purpose, it has no arguments.
TEXT runtime·morestack(SB), NOSPLIT, $0-0
	// R1 = g.m
	MOVD g_m(g), R1

	// R2 = g0
	MOVD m_g0(R1), R2

	// Set g->sched to context in f.
	NOP	SP	// tell vet SP changed - stop checking offsets
	MOVD 0(SP), g_sched+gobuf_pc(g)
	MOVD $8(SP), g_sched+gobuf_sp(g) // f's SP
	MOVD CTXT, g_sched+gobuf_ctxt(g)

	// Cannot grow scheduler stack (m->g0).
	Get g
	Get R2
	I64Eq
	If
		CALLNORESUME runtime·badmorestackg0(SB)
		CALLNORESUME runtime·abort(SB)
	End

	// Cannot grow signal stack (m->gsignal).
	Get g
	I64Load m_gsignal(R1)
	I64Eq
	If
		CALLNORESUME runtime·badmorestackgsignal(SB)
		CALLNORESUME runtime·abort(SB)
	End

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVD 8(SP), m_morebuf+gobuf_pc(R1)
	MOVD $16(SP), m_morebuf+gobuf_sp(R1) // f's caller's SP
	MOVD g, m_morebuf+gobuf_g(R1)

	// Call newstack on m->g0's stack.
	MOVD R2, g
	MOVD g_sched+gobuf_sp(R2), SP
	CALL runtime·newstack(SB)
	UNDEF // crash if newstack returns

// morestack but not preserving ctxt.
TEXT runtime·morestack_noctxt(SB),NOSPLIT,$0
	MOVD $0, CTXT
	JMP runtime·morestack(SB)

TEXT ·asmcgocall(SB), NOSPLIT, $0-0
	UNDEF

#define DISPATCH(NAME, MAXSIZE) \
	Get R0; \
	I64Const $MAXSIZE; \
	I64LeU; \
	If; \
		JMP NAME(SB); \
	End

TEXT ·reflectcall(SB), NOSPLIT, $0-48
	I64Load fn+8(FP)
	I64Eqz
	If
		CALLNORESUME runtime·sigpanic<ABIInternal>(SB)
	End

	MOVW frameSize+32(FP), R0

	DISPATCH(runtime·call16, 16)
	DISPATCH(runtime·call32, 32)
	DISPATCH(runtime·call64, 64)
	DISPATCH(runtime·call128, 128)
	DISPATCH(runtime·call256, 256)
	DISPATCH(runtime·call512, 512)
	DISPATCH(runtime·call1024, 1024)
	DISPATCH(runtime·call2048, 2048)
	DISPATCH(runtime·call4096, 4096)
	DISPATCH(runtime·call8192, 8192)
	DISPATCH(runtime·call16384, 16384)
	DISPATCH(runtime·call32768, 32768)
	DISPATCH(runtime·call65536, 65536)
	DISPATCH(runtime·call131072, 131072)
	DISPATCH(runtime·call262144, 262144)
	DISPATCH(runtime·call524288, 524288)
	DISPATCH(runtime·call1048576, 1048576)
	DISPATCH(runtime·call2097152, 2097152)
	DISPATCH(runtime·call4194304, 4194304)
	DISPATCH(runtime·call8388608, 8388608)
	DISPATCH(runtime·call16777216, 16777216)
	DISPATCH(runtime·call33554432, 33554432)
	DISPATCH(runtime·call67108864, 67108864)
	DISPATCH(runtime·call134217728, 134217728)
	DISPATCH(runtime·call268435456, 268435456)
	DISPATCH(runtime·call536870912, 536870912)
	DISPATCH(runtime·call1073741824, 1073741824)
	JMP runtime·badreflectcall(SB)

#define CALLFN(NAME, MAXSIZE) \
TEXT NAME(SB), WRAPPER, $MAXSIZE-48; \
	NO_LOCAL_POINTERS; \
	MOVW stackArgsSize+24(FP), R0; \
	\
	Get R0; \
	I64Eqz; \
	Not; \
	If; \
		Get SP; \
		I64Load stackArgs+16(FP); \
		I32WrapI64; \
		I64Load stackArgsSize+24(FP); \
		I32WrapI64; \
		MemoryCopy; \
	End; \
	\
	MOVD f+8(FP), CTXT; \
	Get CTXT; \
	I32WrapI64; \
	I64Load $0; \
	CALL; \
	\
	I64Load32U stackRetOffset+28(FP); \
	Set R0; \
	\
	MOVD stackArgsType+0(FP), RET0; \
	\
	I64Load stackArgs+16(FP); \
	Get R0; \
	I64Add; \
	Set RET1; \
	\
	Get SP; \
	I64ExtendI32U; \
	Get R0; \
	I64Add; \
	Set RET2; \
	\
	I64Load32U stackArgsSize+24(FP); \
	Get R0; \
	I64Sub; \
	Set RET3; \
	\
	CALL callRet<>(SB); \
	RET

// callRet copies return values back at the end of call*. This is a
// separate function so it can allocate stack space for the arguments
// to reflectcallmove. It does not follow the Go ABI; it expects its
// arguments in registers.
TEXT callRet<>(SB), NOSPLIT, $40-0
	NO_LOCAL_POINTERS
	MOVD RET0, 0(SP)
	MOVD RET1, 8(SP)
	MOVD RET2, 16(SP)
	MOVD RET3, 24(SP)
	MOVD $0,   32(SP)
	CALL runtime·reflectcallmove(SB)
	RET

CALLFN(·call16, 16)
CALLFN(·call32, 32)
CALLFN(·call64, 64)
CALLFN(·call128, 128)
CALLFN(·call256, 256)
CALLFN(·call512, 512)
CALLFN(·call1024, 1024)
CALLFN(·call2048, 2048)
CALLFN(·call4096, 4096)
CALLFN(·call8192, 8192)
CALLFN(·call16384, 16384)
CALLFN(·call32768, 32768)
CALLFN(·call65536, 65536)
CALLFN(·call131072, 131072)
CALLFN(·call262144, 262144)
CALLFN(·call524288, 524288)
CALLFN(·call1048576, 1048576)
CALLFN(·call2097152, 2097152)
CALLFN(·call4194304, 4194304)
CALLFN(·call8388608, 8388608)
CALLFN(·call16777216, 16777216)
CALLFN(·call33554432, 33554432)
CALLFN(·call67108864, 67108864)
CALLFN(·call134217728, 134217728)
CALLFN(·call268435456, 268435456)
CALLFN(·call536870912, 536870912)
CALLFN(·call1073741824, 1073741824)

TEXT runtime·goexit(SB), NOSPLIT|TOPFRAME, $0-0
	NOP // first PC of goexit is skipped
	CALL runtime·goexit1(SB) // does not return
	UNDEF

TEXT runtime·cgocallback(SB), NOSPLIT, $0-24
	UNDEF

// gcWriteBarrier informs the GC about heap pointer writes.
//
// gcWriteBarrier does NOT follow the Go ABI. It accepts the
// number of bytes of buffer needed as a wasm argument
// (put on the TOS by the caller, lives in local R0 in this body)
// and returns a pointer to the buffer space as a wasm result
// (left on the TOS in this body, appears on the wasm stack
// in the caller).
TEXT gcWriteBarrier<>(SB), NOSPLIT, $0
	Loop
		// R3 = g.m
		MOVD g_m(g), R3
		// R4 = p
		MOVD m_p(R3), R4
		// R5 = wbBuf.next
		MOVD p_wbBuf+wbBuf_next(R4), R5

		// Increment wbBuf.next
		Get R5
		Get R0
		I64Add
		Set R5

		// Is the buffer full?
		Get R5
		I64Load (p_wbBuf+wbBuf_end)(R4)
		I64LeU
		If
			// Commit to the larger buffer.
			MOVD R5, p_wbBuf+wbBuf_next(R4)

			// Make return value (the original next position)
			Get R5
			Get R0
			I64Sub

			Return
		End

		// Flush
		CALLNORESUME runtime·wbBufFlush(SB)

		// Retry
		Br $0
	End

TEXT runtime·gcWriteBarrier1<ABIInternal>(SB),NOSPLIT,$0
	I64Const $8
	Call	gcWriteBarrier<>(SB)
	Return
TEXT runtime·gcWriteBarrier2<ABIInternal>(SB),NOSPLIT,$0
	I64Const $16
	Call	gcWriteBarrier<>(SB)
	Return
TEXT runtime·gcWriteBarrier3<ABIInternal>(SB),NOSPLIT,$0
	I64Const $24
	Call	gcWriteBarrier<>(SB)
	Return
TEXT runtime·gcWriteBarrier4<ABIInternal>(SB),NOSPLIT,$0
	I64Const $32
	Call	gcWriteBarrier<>(SB)
	Return
TEXT runtime·gcWriteBarrier5<ABIInternal>(SB),NOSPLIT,$0
	I64Const $40
	Call	gcWriteBarrier<>(SB)
	Return
TEXT runtime·gcWriteBarrier6<ABIInternal>(SB),NOSPLIT,$0
	I64Const $48
	Call	gcWriteBarrier<>(SB)
	Return
TEXT runtime·gcWriteBarrier7<ABIInternal>(SB),NOSPLIT,$0
	I64Const $56
	Call	gcWriteBarrier<>(SB)
	Return
TEXT runtime·gcWriteBarrier8<ABIInternal>(SB),NOSPLIT,$0
	I64Const $64
	Call	gcWriteBarrier<>(SB)
	Return

TEXT wasm_pc_f_loop(SB),NOSPLIT,$0
// Call the function for the current PC_F. Repeat until PAUSE != 0 indicates pause or exit.
// The WebAssembly stack may unwind, e.g. when switching goroutines.
// The Go stack on the linear memory is then used to jump to the correct functions
// with this loop, without having to restore the full WebAssembly stack.
// It is expected to have a pending call before entering the loop, so check PAUSE first.
	Get PAUSE
	I32Eqz
	If
	loop:
		Loop
			// Get PC_B & PC_F from -8(SP)
			Get SP
			I32Const $8
			I32Sub
			I32Load16U $0 // PC_B

			Get SP
			I32Const $8
			I32Sub
			I32Load16U $2 // PC_F

			CallIndirect $0
			Drop

			Get PAUSE
			I32Eqz
			BrIf loop
		End
	End

	I32Const $0
	Set PAUSE

	Return

// wasm_pc_f_loop_export is like wasm_pc_f_loop, except that this takes an
// argument (on Wasm stack) that is a PC_F, and the loop stops when we get
// to that PC in a normal return (not unwinding).
// This is for handling an wasmexport function when it needs to switch the
// stack.
TEXT wasm_pc_f_loop_export(SB),NOSPLIT,$0
	Get PAUSE
	I32Eqz
outer:
	If
		// R1 is whether a function return normally (0) or unwinding (1).
		// Start with unwinding.
		I32Const $1
		Set R1
	loop:
		Loop
			// Get PC_F & PC_B from -8(SP)
			Get SP
			I32Const $8
			I32Sub
			I32Load16U $2 // PC_F
			Tee R2

			Get R0
			I32Eq
			If // PC_F == R0, we're at the stop PC
				Get R1
				I32Eqz
				// Break if it is a normal return
				BrIf outer // actually jump to after the corresponding End
			End

			Get SP
			I32Const $8
			I32Sub
			I32Load16U $0 // PC_B

			Get R2 // PC_F
			CallIndirect $0
			Set R1 // save return/unwinding state for next iteration

			Get PAUSE
			I32Eqz
			BrIf loop
		End
	End

	I32Const $0
	Set PAUSE

	Return

TEXT wasm_export_lib(SB),NOSPLIT,$0
	UNDEF

TEXT runtime·pause(SB), NOSPLIT, $0-8
	MOVD newsp+0(FP), SP
	I32Const $1
	Set PAUSE
	RETUNWIND

// Called if a wasmexport function is called before runtime initialization
TEXT runtime·notInitialized(SB), NOSPLIT, $0
	MOVD $runtime·wasmStack+(m0Stack__size-16-8)(SB), SP
	I32Const $0 // entry PC_B
	Call runtime·notInitialized1(SB)
	Drop
	I32Const $0 // entry PC_B
	Call runtime·abort(SB)
	UNDEF
