// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type libcFunc uintptr

var asmsysvicall6 libcFunc

//go:noescape
func sigfwd(fn uintptr, sig uint32, info *siginfo, ctx unsafe.Pointer)

//go:nosplit
func sysvicall0(fn *libcFunc) uintptr {
	libcall := &getg().m.libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 0
	libcall.args = uintptr(unsafe.Pointer(fn)) // it's unused but must be non-nil, otherwise crashes
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(libcall))
	return libcall.r1
}

//go:nosplit
func sysvicall1(fn *libcFunc, a1 uintptr) uintptr {
	libcall := &getg().m.libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 1
	// TODO(rsc): Why is noescape necessary here and below?
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(libcall))
	return libcall.r1
}

//go:nosplit
func sysvicall2(fn *libcFunc, a1, a2 uintptr) uintptr {
	libcall := &getg().m.libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 2
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(libcall))
	return libcall.r1
}

//go:nosplit
func sysvicall3(fn *libcFunc, a1, a2, a3 uintptr) uintptr {
	libcall := &getg().m.libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 3
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(libcall))
	return libcall.r1
}

//go:nosplit
func sysvicall4(fn *libcFunc, a1, a2, a3, a4 uintptr) uintptr {
	libcall := &getg().m.libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 4
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(libcall))
	return libcall.r1
}

//go:nosplit
func sysvicall5(fn *libcFunc, a1, a2, a3, a4, a5 uintptr) uintptr {
	libcall := &getg().m.libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 5
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(libcall))
	return libcall.r1
}

//go:nosplit
func sysvicall6(fn *libcFunc, a1, a2, a3, a4, a5, a6 uintptr) uintptr {
	libcall := &getg().m.libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 6
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(libcall))
	return libcall.r1
}
