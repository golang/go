// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func sighandler(_ureg *ureg, note *byte, gp *g) int {
	_g_ := getg()
	var t sigTabT
	var docrash bool
	var length int
	var sig int
	var flags int

	c := &sigctxt{_ureg}

	// The kernel will never pass us a nil note or ureg so we probably
	// made a mistake somewhere in sigtramp.
	if _ureg == nil || note == nil {
		print("sighandler: ureg ", _ureg, " note ", note, "\n")
		goto Throw
	}
	// Check that the note is no more than ERRMAX bytes (including
	// the trailing NUL). We should never receive a longer note.
	length = findnull(note)
	if length > _ERRMAX-1 {
		print("sighandler: note is longer than ERRMAX\n")
		goto Throw
	}
	// See if the note matches one of the patterns in sigtab.
	// Notes that do not match any pattern can be handled at a higher
	// level by the program but will otherwise be ignored.
	flags = _SigNotify
	for sig, t = range sigtable {
		n := len(t.name)
		if length < n {
			continue
		}
		if strncmp(note, &t.name[0], uintptr(n)) == 0 {
			flags = t.flags
			break
		}
	}
	if flags&_SigGoExit != 0 {
		exits((*byte)(add(unsafe.Pointer(note), 9))) // Strip "go: exit " prefix.
	}
	if flags&_SigPanic != 0 {
		// Copy the error string from sigtramp's stack into m->notesig so
		// we can reliably access it from the panic routines.
		memmove(unsafe.Pointer(_g_.m.notesig), unsafe.Pointer(note), uintptr(length+1))
		gp.sig = uint32(sig)
		gp.sigpc = c.pc()
		// Only push sigpanic if PC != 0.
		//
		// If PC == 0, probably panicked because of a call to a nil func.
		// Not pushing that onto SP will make the trace look like a call
		// to sigpanic instead. (Otherwise the trace will end at
		// sigpanic and we won't get to see who faulted).
		if c.pc() != 0 {
			sp := c.sp()
			if regSize > ptrSize {
				sp -= ptrSize
				*(*uintptr)(unsafe.Pointer(sp)) = 0
			}
			sp -= ptrSize
			*(*uintptr)(unsafe.Pointer(sp)) = c.pc()
			c.setsp(sp)
		}
		c.setpc(funcPC(sigpanic))
		return _NCONT
	}
	if flags&_SigNotify != 0 {
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
	_g_.m.throwing = 1
	_g_.m.caughtsig = gp
	startpanic()
	print(gostringnocopy(note), "\n")
	print("PC=", hex(c.pc()), "\n")
	print("\n")
	if gotraceback(&docrash) > 0 {
		goroutineheader(gp)
		tracebacktrap(c.pc(), c.sp(), 0, gp)
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

func resetcpuprofiler(hz int32) {
	// TODO: Enable profiling interrupts.
	getg().m.profilehz = hz
}
