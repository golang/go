// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	// vdsoArrayMax is the byte-size of a maximally sized array on this architecture.
	// See cmd/compile/internal/amd64/galign.go arch.MAXWIDTH initialization.
	vdsoArrayMax = 1<<50 - 1
)

var sym_keys = []symbol_key{
	{"__vdso_gettimeofday", 0x315ca59, 0xb01bca00, &__vdso_gettimeofday_sym},
	{"__vdso_clock_gettime", 0xd35ec75, 0x6e43a318, &__vdso_clock_gettime_sym},
}

// initialize with vsyscall fallbacks
var (
	__vdso_gettimeofday_sym  uintptr = 0xffffffffff600000
	__vdso_clock_gettime_sym uintptr = 0
)
