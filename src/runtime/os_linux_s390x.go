// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	_SS_DISABLE  = 2
	_NSIG        = 65
	_SI_USER     = 0
	_SIG_BLOCK   = 0
	_SIG_UNBLOCK = 1
	_SIG_SETMASK = 2
	_RLIMIT_AS   = 9
)

type sigset uint64

type rlimit struct {
	rlim_cur uintptr
	rlim_max uintptr
}

var sigset_all = sigset(^uint64(0))

func sigaddset(mask *sigset, i int) {
	if i > 64 {
		throw("unexpected signal greater than 64")
	}
	*mask |= 1 << (uint(i) - 1)
}

func sigdelset(mask *sigset, i int) {
	if i > 64 {
		throw("unexpected signal greater than 64")
	}
	*mask &^= 1 << (uint(i) - 1)
}

func sigfillset(mask *uint64) {
	*mask = ^uint64(0)
}

func sigcopyset(mask *sigset, m sigmask) {
	*mask = sigset(uint64(m[0]) | uint64(m[1])<<32)
}
