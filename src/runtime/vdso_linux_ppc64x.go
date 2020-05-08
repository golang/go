// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build ppc64 ppc64le

package runtime

const (
	// vdsoArrayMax is the byte-size of a maximally sized array on this architecture.
	// See cmd/compile/internal/ppc64/galign.go arch.MAXWIDTH initialization.
	vdsoArrayMax = 1<<50 - 1
)

var vdsoLinuxVersion = vdsoVersionKey{"LINUX_2.6.15", 0x75fcba5}

var vdsoSymbolKeys = []vdsoSymbolKey{
	{"__kernel_clock_gettime", 0xb0cd725, 0xdfa941fd, &vdsoClockgettimeSym},
}

// initialize with vsyscall fallbacks
var (
	vdsoClockgettimeSym uintptr = 0
)
