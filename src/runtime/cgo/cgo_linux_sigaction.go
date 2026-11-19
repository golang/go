// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && linux && (amd64 || arm64)

package cgo

import "unsafe"

//go:linkname _libc_sigaction libc_sigaction
var _libc_sigaction byte
var libc_sigaction = unsafe.Pointer(&_libc_sigaction)

// goSigaction matches runtime.sigactiont on linux/amd64 and linux/arm64.
type goSigaction struct {
	handler  uintptr
	flags    uint64
	restorer uintptr
	mask     uint64
}

//go:linknamestd _cgoSigaction runtime.cgoSigaction
var _cgoSigaction = cgoSigaction

//go:nosplit
//go:norace
func sigactionCall(noG bool, fn, a0, a1, a2 uintptr) uintptr {
	if noG {
		return asmcgocallNoG(fn, a0, a1, a2, 0, 0, 0)
	}
	return asmcgocall(fn, a0, a1, a2, 0, 0, 0)
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoSigaction(signum uintptr, goact, oldgoact *goSigaction, noG bool) int32 {
	// SA_RESTORER is part of the kernel interface.
	// This is Linux i386/amd64 specific.
	// See the comment on fixSigactionForCgo.
	const SA_RESTORER = 0x4000000

	if noG {
		tsanAcquireNoG()
	} else {
		tsanAcquire()
	}

	var oldact cSigaction
	var actPtr, oldactPtr uintptr

	if goact != nil {
		var act cSigaction
		act.handler = goact.handler
		sigactionCall(noG, uintptr(libc_sigfillset), uintptr(unsafe.Pointer(&act.mask)), 0, 0)
		act.mask[0] &= uintptr(goact.mask)
		act.flags = int32(goact.flags &^ SA_RESTORER)
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
		if noG {
			tsanReleaseNoG()
		} else {
			tsanRelease()
		}
		return errno
	}

	if oldgoact != nil {
		oldgoact.handler = oldact.handler
		oldgoact.mask = uint64(oldact.mask[0])
		oldgoact.flags = uint64(int64(oldact.flags))
		msanUnpoisonMode(noG, unsafe.Pointer(oldgoact), unsafe.Sizeof(*oldgoact))
	}

	if noG {
		tsanReleaseNoG()
	} else {
		tsanRelease()
	}
	return ret
}
