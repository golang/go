// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !mips64
// +build !mips64le
// +build linux

package runtime

var sigset_all = sigset{^uint32(0), ^uint32(0)}

func sigaddset(mask *sigset, i int) {
	(*mask)[(i-1)/32] |= 1 << ((uint32(i) - 1) & 31)
}

func sigdelset(mask *sigset, i int) {
	(*mask)[(i-1)/32] &^= 1 << ((uint32(i) - 1) & 31)
}

func sigfillset(mask *uint64) {
	*mask = ^uint64(0)
}

func sigcopyset(mask *sigset, m sigmask) {
	copy((*mask)[:], m[:])
}
