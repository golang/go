// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/sys"
	"unsafe"
)

type mts struct {
	tv_sec  int64
	tv_nsec int64
}

type mscratch struct {
	v [6]uintptr
}

type mOS struct {
	waitsema uintptr // semaphore for parking on locks
	perrno   *int32  // pointer to tls errno
	// This is here to avoid using the G stack so the stack can move during the call.
	libcall libcall
	ts      mts
	scratch mscratch
}

type libcFunc uintptr

//go:linkname asmsysvicall6x runtime.asmsysvicall6
var asmsysvicall6x libcFunc // name to take addr of asmsysvicall6

func asmsysvicall6() // declared for vet; do NOT call

//go:nosplit
func sysvicall0(fn *libcFunc) uintptr {
	// Leave caller's PC/SP around for traceback.
	gp := getg()
	var mp *m
	if gp != nil {
		mp = gp.m
	}
	if mp != nil && mp.libcallsp == 0 {
		mp.libcallg.set(gp)
		mp.libcallpc = sys.GetCallerPC()
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = sys.GetCallerSP()
	} else {
		mp = nil // See comment in sys_darwin.go:libcCall
	}

	var libcall libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 0
	libcall.args = uintptr(unsafe.Pointer(fn)) // it's unused but must be non-nil, otherwise crashes
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&libcall))
	if mp != nil {
		mp.libcallsp = 0
	}
	return libcall.r1
}

//go:nosplit
func sysvicall1(fn *libcFunc, a1 uintptr) uintptr {
	r1, _ := sysvicall1Err(fn, a1)
	return r1
}

// sysvicall1Err returns both the system call result and the errno value.
// This is used by sysvicall1 and pipe.
//
//go:nosplit
func sysvicall1Err(fn *libcFunc, a1 uintptr) (r1, err uintptr) {
	// Leave caller's PC/SP around for traceback.
	gp := getg()
	var mp *m
	if gp != nil {
		mp = gp.m
	}
	if mp != nil && mp.libcallsp == 0 {
		mp.libcallg.set(gp)
		mp.libcallpc = sys.GetCallerPC()
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = sys.GetCallerSP()
	} else {
		mp = nil
	}

	var libcall libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 1
	// TODO(rsc): Why is noescape necessary here and below?
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&libcall))
	if mp != nil {
		mp.libcallsp = 0
	}
	return libcall.r1, libcall.err
}

//go:nosplit
func sysvicall2(fn *libcFunc, a1, a2 uintptr) uintptr {
	r1, _ := sysvicall2Err(fn, a1, a2)
	return r1
}

//go:nosplit
//go:cgo_unsafe_args

// sysvicall2Err returns both the system call result and the errno value.
// This is used by sysvicall2 and pipe2.
func sysvicall2Err(fn *libcFunc, a1, a2 uintptr) (uintptr, uintptr) {
	// Leave caller's PC/SP around for traceback.
	gp := getg()
	var mp *m
	if gp != nil {
		mp = gp.m
	}
	if mp != nil && mp.libcallsp == 0 {
		mp.libcallg.set(gp)
		mp.libcallpc = sys.GetCallerPC()
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = sys.GetCallerSP()
	} else {
		mp = nil
	}

	var libcall libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 2
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&libcall))
	if mp != nil {
		mp.libcallsp = 0
	}
	return libcall.r1, libcall.err
}

//go:nosplit
func sysvicall3(fn *libcFunc, a1, a2, a3 uintptr) uintptr {
	r1, _ := sysvicall3Err(fn, a1, a2, a3)
	return r1
}

//go:nosplit
//go:cgo_unsafe_args

// sysvicall3Err returns both the system call result and the errno value.
// This is used by sysvicall3 and write1.
func sysvicall3Err(fn *libcFunc, a1, a2, a3 uintptr) (r1, err uintptr) {
	// Leave caller's PC/SP around for traceback.
	gp := getg()
	var mp *m
	if gp != nil {
		mp = gp.m
	}
	if mp != nil && mp.libcallsp == 0 {
		mp.libcallg.set(gp)
		mp.libcallpc = sys.GetCallerPC()
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = sys.GetCallerSP()
	} else {
		mp = nil
	}

	var libcall libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 3
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&libcall))
	if mp != nil {
		mp.libcallsp = 0
	}
	return libcall.r1, libcall.err
}

//go:nosplit
//go:cgo_unsafe_args
func sysvicall4(fn *libcFunc, a1, a2, a3, a4 uintptr) uintptr {
	// Leave caller's PC/SP around for traceback.
	gp := getg()
	var mp *m
	if gp != nil {
		mp = gp.m
	}
	if mp != nil && mp.libcallsp == 0 {
		mp.libcallg.set(gp)
		mp.libcallpc = sys.GetCallerPC()
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = sys.GetCallerSP()
	} else {
		mp = nil
	}

	var libcall libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 4
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&libcall))
	if mp != nil {
		mp.libcallsp = 0
	}
	return libcall.r1
}

//go:nosplit
//go:cgo_unsafe_args
func sysvicall5(fn *libcFunc, a1, a2, a3, a4, a5 uintptr) uintptr {
	// Leave caller's PC/SP around for traceback.
	gp := getg()
	var mp *m
	if gp != nil {
		mp = gp.m
	}
	if mp != nil && mp.libcallsp == 0 {
		mp.libcallg.set(gp)
		mp.libcallpc = sys.GetCallerPC()
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = sys.GetCallerSP()
	} else {
		mp = nil
	}

	var libcall libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 5
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&libcall))
	if mp != nil {
		mp.libcallsp = 0
	}
	return libcall.r1
}

//go:nosplit
//go:cgo_unsafe_args
func sysvicall6(fn *libcFunc, a1, a2, a3, a4, a5, a6 uintptr) uintptr {
	// Leave caller's PC/SP around for traceback.
	gp := getg()
	var mp *m
	if gp != nil {
		mp = gp.m
	}
	if mp != nil && mp.libcallsp == 0 {
		mp.libcallg.set(gp)
		mp.libcallpc = sys.GetCallerPC()
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = sys.GetCallerSP()
	} else {
		mp = nil
	}

	var libcall libcall
	libcall.fn = uintptr(unsafe.Pointer(fn))
	libcall.n = 6
	libcall.args = uintptr(noescape(unsafe.Pointer(&a1)))
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&libcall))
	if mp != nil {
		mp.libcallsp = 0
	}
	return libcall.r1
}

func issetugid() int32 {
	return int32(sysvicall0(&libc_issetugid))
}
