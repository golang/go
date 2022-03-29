// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The standard Linux sigset type on big-endian 64-bit machines.

//go:build linux && (ppc64 || s390x)

package runtime

const (
	_SS_DISABLE  = 2
	_NSIG        = 65
	_SI_USER     = 0
	_SIG_BLOCK   = 0
	_SIG_UNBLOCK = 1
	_SIG_SETMASK = 2
)

type sigset uint64

var sigset_all = sigset(^uint64(0))

//go:nosplit
//go:nowritebarrierrec
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

//go:nosplit
func sigfillset(mask *uint64) {
	*mask = ^uint64(0)
}
