// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix,ppc64

package cpu

import "golang.org/x/sys/unix"

const cacheLineSize = 128

const (
	// getsystemcfg constants
	_SC_IMPL     = 2
	_IMPL_POWER8 = 0x10000
	_IMPL_POWER9 = 0x20000
)

func init() {
	impl := unix.Getsystemcfg(_SC_IMPL)
	if impl&_IMPL_POWER8 != 0 {
		PPC64.IsPOWER8 = true
	}
	if impl&_IMPL_POWER9 != 0 {
		PPC64.IsPOWER9 = true
	}

	Initialized = true
}
