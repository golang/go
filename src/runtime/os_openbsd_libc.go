// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && !mips64

package runtime

import (
	"internal/abi"
	"unsafe"
)

// mstart_stub provides glue code to call mstart from pthread_create.
func mstart_stub()

// May run with m.p==nil, so write barriers are not allowed.
//
//go:nowritebarrierrec
func newosproc(mp *m) {
	if false {
		print("newosproc m=", mp, " g=", mp.g0, " id=", mp.id, " ostk=", &mp, "\n")
	}

	// Initialize an attribute object.
	var attr pthreadattr
	if err := pthread_attr_init(&attr); err != 0 {
		writeErrStr(failthreadcreate)
		exit(1)
	}

	// Find out OS stack size for our own stack guard.
	var stacksize uintptr
	if pthread_attr_getstacksize(&attr, &stacksize) != 0 {
		writeErrStr(failthreadcreate)
		exit(1)
	}
	mp.g0.stack.hi = stacksize // for mstart

	// Tell the pthread library we won't join with this thread.
	if pthread_attr_setdetachstate(&attr, _PTHREAD_CREATE_DETACHED) != 0 {
		writeErrStr(failthreadcreate)
		exit(1)
	}

	// Finally, create the thread. It starts at mstart_stub, which does some low-level
	// setup and then calls mstart.
	var oset sigset
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	err := retryOnEAGAIN(func() int32 {
		return pthread_create(&attr, abi.FuncPCABI0(mstart_stub), unsafe.Pointer(mp))
	})
	sigprocmask(_SIG_SETMASK, &oset, nil)
	if err != 0 {
		writeErrStr(failthreadcreate)
		exit(1)
	}

	pthread_attr_destroy(&attr)
}
