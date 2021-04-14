// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build openbsd,!amd64
// +build openbsd,!arm64

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

//go:noescape
func tfork(param *tforkt, psize uintptr, mm *m, gg *g, fn uintptr) int32

// May run with m.p==nil, so write barriers are not allowed.
//go:nowritebarrier
func newosproc(mp *m) {
	stk := unsafe.Pointer(mp.g0.stack.hi)
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " id=", mp.id, " ostk=", &mp, "\n")
	}

	// Stack pointer must point inside stack area (as marked with MAP_STACK),
	// rather than at the top of it.
	param := tforkt{
		tf_tcb:   unsafe.Pointer(&mp.tls[0]),
		tf_tid:   nil, // minit will record tid
		tf_stack: uintptr(stk) - sys.PtrSize,
	}

	var oset sigset
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	ret := tfork(&param, unsafe.Sizeof(param), mp, mp.g0, funcPC(mstart))
	sigprocmask(_SIG_SETMASK, &oset, nil)

	if ret < 0 {
		print("runtime: failed to create new OS thread (have ", mcount()-1, " already; errno=", -ret, ")\n")
		if ret == -_EAGAIN {
			println("runtime: may need to increase max user processes (ulimit -p)")
		}
		throw("runtime.newosproc")
	}
}
