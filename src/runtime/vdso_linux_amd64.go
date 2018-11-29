// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	// vdsoArrayMax is the byte-size of a maximally sized array on this architecture.
	// See cmd/compile/internal/amd64/galign.go arch.MAXWIDTH initialization.
	vdsoArrayMax = 1<<50 - 1
)

var vdsoLinuxVersion = vdsoVersionKey{"LINUX_2.6", 0x3ae75f6}

var vdsoSymbolKeys = []vdsoSymbolKey{
	{"__vdso_gettimeofday", 0x315ca59, 0xb01bca00, &vdsoGettimeofdaySym},
	{"__vdso_clock_gettime", 0xd35ec75, 0x6e43a318, &vdsoClockgettimeSym},
}

// initialize with vsyscall fallbacks
var (
	vdsoGettimeofdaySym uintptr = 0xffffffffff600000
	vdsoClockgettimeSym uintptr = 0
)
