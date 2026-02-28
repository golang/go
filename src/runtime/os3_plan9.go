// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/stringslite"
	"unsafe"
)

// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func sighandler(_ureg *ureg, note *byte, gp *g) int {
	gsignal := getg()
	mp := gsignal.m

	var t sigTabT
	var docrash bool
	var sig int
	var flags int
	var level int32

	c := &sigctxt{_ureg}
	notestr := gostringnocopy(note)

	// The kernel will never pass us a nil note or ureg so we probably
	// made a mistake somewhere in sigtramp.
	if _ureg == nil || note == nil {
		print("sighandler: ureg ", _ureg, " note ", note, "\n")
		goto Throw
	}
	// Check that the note is no more than ERRMAX bytes (including
	// the trailing NUL). We should never receive a longer note.
	if len(notestr) > _ERRMAX-1 {
		print("sighandler: note is longer than ERRMAX\n")
		goto Throw
	}
	if isAbortPC(c.pc()) {
		// Never turn abort into a panic.
		goto Throw
	}
	// See if the note matches one of the patterns in sigtab.
	// Notes that do not match any pattern can be handled at a higher
	// level by the program but will otherwise be ignored.
	flags = _SigNotify
	for sig, t = range sigtable {
		if stringslite.HasPrefix(notestr, t.name) {
			flags = t.flags
			break
		}
	}
	if flags&_SigPanic != 0 && gp.throwsplit {
		// We can't safely sigpanic because it may grow the
		// stack. Abort in the signal handler instead.
		flags = (flags &^ _SigPanic) | _SigThrow
	}
	if flags&_SigGoExit != 0 {
		exits((*byte)(add(unsafe.Pointer(note), 9))) // Strip "go: exit " prefix.
	}
	if flags&_SigPanic != 0 {
		// Copy the error string from sigtramp's stack into m->notesig so
		// we can reliably access it from the panic routines.
		memmove(unsafe.Pointer(mp.notesig), unsafe.Pointer(note), uintptr(len(notestr)+1))
		gp.sig = uint32(sig)
		gp.sigpc = c.pc()

		pc := c.pc()
		sp := c.sp()

		// If we don't recognize the PC as code
		// but we do recognize the top pointer on the stack as code,
		// then assume this was a call to non-code and treat like
		// pc == 0, to make unwinding show the context.
		if pc != 0 && !findfunc(pc).valid() && findfunc(*(*uintptr)(unsafe.Pointer(sp))).valid() {
			pc = 0
		}

		// IF LR exists, sigpanictramp must save it to the stack
		// before entry to sigpanic so that panics in leaf
		// functions are correctly handled. This will smash
		// the stack frame but we're not going back there
		// anyway.
		if usesLR {
			c.savelr(c.lr())
		}

		// If PC == 0, probably panicked because of a call to a nil func.
		// Not faking that as the return address will make the trace look like a call
		// to sigpanic instead. (Otherwise the trace will end at
		// sigpanic and we won't get to see who faulted).
		if pc != 0 {
			if usesLR {
				c.setlr(pc)
			} else {
				sp -= goarch.PtrSize
				*(*uintptr)(unsafe.Pointer(sp)) = pc
				c.setsp(sp)
			}
		}
		if usesLR {
			c.setpc(abi.FuncPCABI0(sigpanictramp))
		} else {
			c.setpc(abi.FuncPCABI0(sigpanic0))
		}
		return _NCONT
	}
	if flags&_SigNotify != 0 {
		if ignoredNote(note) {
			return _NCONT
		}
		if sendNote(note) {
			return _NCONT
		}
	}
	if flags&_SigKill != 0 {
		goto Exit
	}
	if flags&_SigThrow == 0 {
		return _NCONT
	}
Throw:
	mp.throwing = throwTypeRuntime
	mp.caughtsig.set(gp)
	startpanic_m()
	print(notestr, "\n")
	print("PC=", hex(c.pc()), "\n")
	print("\n")
	level, _, docrash = gotraceback()
	if level > 0 {
		goroutineheader(gp)
		tracebacktrap(c.pc(), c.sp(), c.lr(), gp)
		tracebackothers(gp)
		print("\n")
		dumpregs(_ureg)
	}
	if docrash {
		crash()
	}
Exit:
	goexitsall(note)
	exits(note)
	return _NDFLT // not reached
}

func sigenable(sig uint32) {
}

func sigdisable(sig uint32) {
}

func sigignore(sig uint32) {
}

func setProcessCPUProfiler(hz int32) {
}

func setThreadCPUProfiler(hz int32) {
	// TODO: Enable profiling interrupts.
	getg().m.profilehz = hz
}

// gsignalStack is unused on Plan 9.
type gsignalStack struct{}
