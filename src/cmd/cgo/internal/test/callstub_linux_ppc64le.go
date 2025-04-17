// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// extern int notoc_func(void);
// int TestPPC64Stubs(void) {
//	return notoc_func();
// }
import "C"
import "testing"

func testPPC64CallStubs(t *testing.T) {
	// Verify the trampolines run on the testing machine. If they
	// do not, or are missing, a crash is expected.
	if C.TestPPC64Stubs() != 0 {
		t.Skipf("This test requires binutils 2.35 or newer.")
	}
}
