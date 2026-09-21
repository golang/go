// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package cgotest

/*
#include <stdio.h>

// Global function pointer to a dynamically-linked libc function.
// When compiled to a Mach-O object, this produces a RELOC_UNSIGNED
// relocation targeting the external symbol _puts.
int (*_cgo_test_dynref_puts)(const char *) = puts;

static int cgo_call_dynref(void) {
	if (_cgo_test_dynref_puts == 0) {
		return -1;
	}
	// Call the resolved function pointer. puts returns a non-negative
	// value on success.
	return _cgo_test_dynref_puts("cgo unsigned reloc test");
}
*/
import "C"

import "testing"

// unsignedRelocDynimport verifies that the Go internal linker correctly
// handles Mach-O UNSIGNED relocations targeting dynamic import symbols.
// The C preamble above contains a global function pointer initialized
// to puts, which produces a RELOC_UNSIGNED relocation to the external
// symbol _puts. If the linker can't handle this, the test binary
// won't link at all.
func unsignedRelocDynimport(t *testing.T) {
	got := C.cgo_call_dynref()
	if got < 0 {
		t.Fatal("C function pointer to puts not resolved")
	}
}
