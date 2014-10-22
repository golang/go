// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func setitimer(mode int32, new, old unsafe.Pointer)
func sigaction(sig int32, new, old unsafe.Pointer)
func sigaltstack(new, old unsafe.Pointer)
func sigprocmask(mode int32, new, old unsafe.Pointer)
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32
func getrlimit(kind int32, limit unsafe.Pointer)
func miniterrno(fn unsafe.Pointer)
func raise(sig int32)
func getcontext(ctxt unsafe.Pointer)
func tstart_sysvicall(mm unsafe.Pointer) uint32
func nanotime1() int64
func usleep1(usec uint32)
func osyield1()
func netpollinit()
func netpollopen(fd uintptr, pd *pollDesc) int32
func netpollclose(fd uintptr) int32
func netpollarm(pd *pollDesc, mode int)

type libcFunc byte

var asmsysvicall6 libcFunc

//go:nosplit
func sysvicall0(fn *libcFunc) uintptr {
	libcall := &getg().m.libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 0
	// TODO(rsc): Why is noescape necessary here and below?
	libcall.args = uintptr(noescape(unsafe.Pointer(&fn))) // it's unused but must be non-nil, otherwise crashes
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(libcall))
	return libcall.r1
}

//go:nosplit
func sysvicall1(fn *libcFunc, a1 uintptr) uintptr {
	libcall := &getg().m.libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 1
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
