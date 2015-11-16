// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le
// +build linux

package runtime

var sigset_all = sigset{^uint64(0), ^uint64(0)}

func sigaddset(mask *sigset, i int) {
	(*mask)[(i-1)/64] |= 1 << ((uint32(i) - 1) & 63)
}

func sigdelset(mask *sigset, i int) {
	(*mask)[(i-1)/64] &^= 1 << ((uint32(i) - 1) & 63)
}

func sigfillset(mask *[2]uint64) {
	(*mask)[0], (*mask)[1] = ^uint64(0), ^uint64(0)
}

func sigcopyset(mask *sigset, m sigmask) {
	(*mask)[0] = uint64(m[0]) | uint64(m[1])<<32
}
