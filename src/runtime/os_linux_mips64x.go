// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build mips64 mips64le

package runtime

import "internal/cpu"

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_HWCAP:
		cpu.HWCap = uint(val)
	}
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
	_NSIG        = 129
	_SI_USER     = 0
	_SIG_BLOCK   = 1
	_SIG_UNBLOCK = 2
	_SIG_SETMASK = 3
)

type sigset [2]uint64

var sigset_all = sigset{^uint64(0), ^uint64(0)}

//go:nosplit
//go:nowritebarrierrec
func sigaddset(mask *sigset, i int) {
	(*mask)[(i-1)/64] |= 1 << ((uint32(i) - 1) & 63)
}

func sigdelset(mask *sigset, i int) {
	(*mask)[(i-1)/64] &^= 1 << ((uint32(i) - 1) & 63)
}

//go:nosplit
func sigfillset(mask *[2]uint64) {
	(*mask)[0], (*mask)[1] = ^uint64(0), ^uint64(0)
}
