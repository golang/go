// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build mips mipsle

package runtime

var randomNumber uint32

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_RANDOM:
		// sysargs filled in startupRandomData, but that
		// pointer may not be word aligned, so we must treat
		// it as a byte array.
		randomNumber = uint32(startupRandomData[4]) | uint32(startupRandomData[5])<<8 |
			uint32(startupRandomData[6])<<16 | uint32(startupRandomData[7])<<24
	}
}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed fastrand().
	// nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// randomNumber provides better seeding of fastrand1.
	return nanotime() + int64(randomNumber)
}

const (
	_SS_DISABLE  = 2
	_NSIG        = 128 + 1
	_SI_USER     = 0
	_SIG_BLOCK   = 1
	_SIG_UNBLOCK = 2
	_SIG_SETMASK = 3
	_RLIMIT_AS   = 6
)

type sigset [4]uint32

type rlimit struct {
	rlim_cur uintptr
	rlim_max uintptr
}

var sigset_all = sigset{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}

//go:nosplit
//go:nowritebarrierrec
func sigaddset(mask *sigset, i int) {
	(*mask)[(i-1)/32] |= 1 << ((uint32(i) - 1) & 31)
}

func sigdelset(mask *sigset, i int) {
	(*mask)[(i-1)/32] &^= 1 << ((uint32(i) - 1) & 31)
}

func sigfillset(mask *[4]uint32) {
	(*mask)[0], (*mask)[1], (*mask)[2], (*mask)[3] = ^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)
}
