// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (mips || mipsle)

package runtime

func archauxv(tag, val uintptr) {
}

func osArchInit() {}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed fastrand().
	// nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}

const (
	_SS_DISABLE  = 2
	_NSIG        = 128 + 1
	_SI_USER     = 0
	_SIG_BLOCK   = 1
	_SIG_UNBLOCK = 2
	_SIG_SETMASK = 3
)

type sigset [4]uint32

var sigset_all = sigset{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}

//go:nosplit
//go:nowritebarrierrec
func sigaddset(mask *sigset, i int) {
	(*mask)[(i-1)/32] |= 1 << ((uint32(i) - 1) & 31)
}

func sigdelset(mask *sigset, i int) {
	(*mask)[(i-1)/32] &^= 1 << ((uint32(i) - 1) & 31)
}

//go:nosplit
func sigfillset(mask *[4]uint32) {
	(*mask)[0], (*mask)[1], (*mask)[2], (*mask)[3] = ^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)
}
