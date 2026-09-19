// asmcheck

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that optimized range memclr works when the clear target has a stable
// address. Pointer and slice fields must remain ordinary loops because their
// values can change through storage cleared by an earlier iteration.

package codegen

type T struct {
	a *[10]int
	b [10]int
	s []int
}

func (t *T) f() {
	for i := range t.a {
		// amd64:-".*runtime.memclrNoHeapPointers"
		// amd64:-`MOVUPS X15,`
		t.a[i] = 0
	}

	for i := range *t.a {
		// amd64:-".*runtime.memclrNoHeapPointers"
		// amd64:-`MOVUPS X15,`
		t.a[i] = 0
	}

	for i := range t.a {
		// amd64:-".*runtime.memclrNoHeapPointers"
		// amd64:-`MOVUPS X15,`
		(*t.a)[i] = 0
	}

	for i := range *t.a {
		// amd64:-".*runtime.memclrNoHeapPointers"
		// amd64:-`MOVUPS X15,`
		(*t.a)[i] = 0
	}

	// amd64:-".*runtime.memclrNoHeapPointers"
	// amd64:`MOVUPS X15,`
	for i := range t.b {
		t.b[i] = 0
	}

	// amd64:-".*runtime.memclrNoHeapPointers"
	for i := range t.s {
		t.s[i] = 0
	}
}
