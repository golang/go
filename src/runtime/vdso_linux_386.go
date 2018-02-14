// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	// vdsoArrayMax is the byte-size of a maximally sized array on this architecture.
	// See cmd/compile/internal/x86/galign.go arch.MAXWIDTH initialization, but must also
	// be constrained to max +ve int.
	vdsoArrayMax = 1<<31 - 1
)

var sym_keys = []symbol_key{
	{"__vdso_clock_gettime", 0xd35ec75, 0x6e43a318, &__vdso_clock_gettime_sym},
}

// initialize to fall back to syscall
var (
	__vdso_clock_gettime_sym uintptr = 0
)
