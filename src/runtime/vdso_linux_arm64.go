// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	// vdsoArrayMax is the byte-size of a maximally sized array on this architecture.
	// See cmd/compile/internal/arm64/galign.go arch.MAXWIDTH initialization.
	vdsoArrayMax = 1<<50 - 1
)

// key and version at man 7 vdso : aarch64
var vdsoLinuxVersion = vdsoVersionKey{"LINUX_2.6.39", 0x75fcb89}

var vdsoSymbolKeys = []vdsoSymbolKey{
	{"__kernel_clock_gettime", 0xb0cd725, 0xdfa941fd, &vdsoClockgettimeSym},
}

// initialize to fall back to syscall
var vdsoClockgettimeSym uintptr = 0
