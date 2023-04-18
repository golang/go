// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && loong64

package runtime

const (
	// vdsoArrayMax is the byte-size of a maximally sized array on this architecture.
	// See cmd/compile/internal/loong64/galign.go arch.MAXWIDTH initialization.
	vdsoArrayMax = 1<<50 - 1
)

// not currently described in manpages as of May 2022, but will eventually
// appear
// when that happens, see man 7 vdso : loongarch
var vdsoLinuxVersion = vdsoVersionKey{"LINUX_5.10", 0xae78f70}

var vdsoSymbolKeys = []vdsoSymbolKey{
	{"__vdso_clock_gettime", 0xd35ec75, 0x6e43a318, &vdsoClockgettimeSym},
}

// initialize to fall back to syscall
var (
	vdsoClockgettimeSym uintptr = 0
)
