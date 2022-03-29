// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

package cpu

const (
	// getsystemcfg constants
	_SC_IMPL     = 2
	_IMPL_POWER9 = 0x20000
)

func osinit() {
	impl := getsystemcfg(_SC_IMPL)
	PPC64.IsPOWER9 = isSet(impl, _IMPL_POWER9)
}

// getsystemcfg is defined in runtime/os2_aix.go
func getsystemcfg(label uint) uint
