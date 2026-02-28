// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	// vdsoArrayMax is the byte-size of a maximally sized array on this architecture.
	// See cmd/compile/internal/riscv64/galign.go arch.MAXWIDTH initialization.
	vdsoArrayMax = 1<<50 - 1
)

// key and version at man 7 vdso : riscv
var vdsoLinuxVersion = vdsoVersionKey{"LINUX_4.15", 0xae77f75}

var vdsoSymbolKeys = []vdsoSymbolKey{
	{"__vdso_clock_gettime", 0xd35ec75, 0x6e43a318, &vdsoClockgettimeSym},
}

// initialize to fall back to syscall
var vdsoClockgettimeSym uintptr = 0
