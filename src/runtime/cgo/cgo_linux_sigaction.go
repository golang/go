// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && linux && (386 || amd64 || arm64 || loong64 || ppc64 || ppc64le || riscv64)

package cgo

import "unsafe"

//go:linkname _cgo_libc_sigaction _cgo_libc_sigaction
var _cgo_libc_sigaction byte
var libc_sigaction = unsafe.Pointer(&_cgo_libc_sigaction)

//go:linknamestd _cgoSigaction runtime.cgoSigaction
var _cgoSigaction = cgoSigaction

// sigactionCall calls the C function fn with up to 3 arguments.
// It calls the runtime directly rather than through asmcgocall or
// asmcgocallNoG so that the nosplit chain from cgoSigaction stays
// within the nosplit stack limit even when compiled with -N -l.
//
//go:nosplit
//go:norace
//go:nocheckptr
func sigactionCall(noG bool, fn, a0, a1, a2 uintptr) uintptr {
	var f cgoCallFrame
	f.fn, f.a0, f.a1, f.a2 = fn, a0, a1, a2
	if noG {
		runtime_asmcgocall_no_g(cgoCallTrampolinePC(), unsafe.Pointer(&f))
	} else {
		runtime_asmcgocall(cgoCallTrampolinePC(), unsafe.Pointer(&f))
	}
	return f.r1
}

// restrictMask restricts act.mask, which must have been filled by
// sigfillset, to the signals in gomask. Starting from sigfillset rather
// than sigemptyset keeps the signals reserved by libc (such as glibc's
// SIGCANCEL and SIGSETXID), which sigfillset leaves out and sigaddset in
// gcc_sigaction.c refuses to add, out of the mask.
//
//go:nosplit
//go:norace
func (act *cSigaction) restrictMask(gomask uint64) {
	act.mask[0] &= uintptr(gomask)
	n := 1
	if unsafe.Sizeof(uintptr(0)) == 4 && len(act.mask) > 1 {
		act.mask[:][1] &= uintptr(gomask >> 32)
		n = 2
	}
	for i := n; i < len(act.mask); i++ {
		act.mask[i] = 0
	}
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoSigaction(signum uintptr, goact, oldgoact *goSigaction, noG bool) int32 {
	// SA_RESTORER is part of the kernel interface.
	// This is Linux i386/amd64 specific.
	// See the comment on fixSigactionForCgo.
	const _SA_RESTORER = 0x4000000

	tsanCall(tsanAcquireFnPtr, noG)

	var oldact cSigaction
	var actPtr, oldactPtr uintptr

	if goact != nil {
		var act cSigaction
		act.handler = goact.handler
		sigactionCall(noG, uintptr(libc_sigfillset), uintptr(unsafe.Pointer(&act.mask)), 0, 0)
		act.restrictMask(goact.mask)
		act.flags = int32(goact.flags &^ _SA_RESTORER)
		msanUnpoisonMode(noG, unsafe.Pointer(&act), unsafe.Sizeof(act))
		actPtr = uintptr(unsafe.Pointer(&act))
	}
	if oldgoact != nil {
		oldactPtr = uintptr(unsafe.Pointer(&oldact))
	}

	ret := int32(sigactionCall(noG, uintptr(libc_sigaction), uintptr(int32(signum)), actPtr, oldactPtr))
	if ret == -1 {
		// runtime.rt_sigaction expects _cgo_sigaction to return errno on error.
		errnoPtr := sigactionCall(noG, uintptr(libc_errno_location), 0, 0, 0)
		errno := *(*int32)(unsafe.Pointer(errnoPtr))
		tsanCall(tsanReleaseFnPtr, noG)
		return errno
	}

	if oldgoact != nil {
		oldgoact.handler = oldact.handler
		oldgoact.mask = uint64(oldact.mask[0])
		if unsafe.Sizeof(uintptr(0)) == 4 && len(oldact.mask) > 1 {
			oldgoact.mask |= uint64(oldact.mask[:][1]) << 32
		}
		oldgoact.flags = uintptr(uint32(oldact.flags))
		msanUnpoisonMode(noG, unsafe.Pointer(oldgoact), unsafe.Sizeof(*oldgoact))
	}

	tsanCall(tsanReleaseFnPtr, noG)
	return ret
}
