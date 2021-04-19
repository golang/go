// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !mips
// +build !mipsle
// +build !mips64
// +build !mips64le
// +build !s390x
// +build !ppc64
// +build linux

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

// It's hard to tease out exactly how big a Sigset is, but
// rt_sigprocmask crashes if we get it wrong, so if binaries
// are running, this is right.
type sigset [2]uint32

type rlimit struct {
	rlim_cur uintptr
	rlim_max uintptr
}

var sigset_all = sigset{^uint32(0), ^uint32(0)}

//go:nosplit
//go:nowritebarrierrec
func sigaddset(mask *sigset, i int) {
	(*mask)[(i-1)/32] |= 1 << ((uint32(i) - 1) & 31)
}

func sigdelset(mask *sigset, i int) {
	(*mask)[(i-1)/32] &^= 1 << ((uint32(i) - 1) & 31)
}

func sigfillset(mask *uint64) {
	*mask = ^uint64(0)
}
