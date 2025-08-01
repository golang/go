// asmcheck

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that optimized range memclr works with pointers to arrays.
// The clears get inlined, see https://github.com/golang/go/issues/56997

package codegen

type T struct {
	a *[10]int
	b [10]int
	s []int
}

func (t *T) f() {
	// amd64:-".*runtime.memclrNoHeapPointers"
	// amd64:`MOVUPS\tX15,`
	for i := range t.a {
		t.a[i] = 0
	}

	// amd64:-".*runtime.memclrNoHeapPointers"
	// amd64:`MOVUPS\tX15,`
	for i := range *t.a {
		t.a[i] = 0
	}

	// amd64:-".*runtime.memclrNoHeapPointers"
	// amd64:`MOVUPS\tX15,`
	for i := range t.a {
		(*t.a)[i] = 0
	}

	// amd64:-".*runtime.memclrNoHeapPointers"
	// amd64:`MOVUPS\tX15,`
	for i := range *t.a {
		(*t.a)[i] = 0
	}

	// amd64:-".*runtime.memclrNoHeapPointers"
	// amd64:`MOVUPS\tX15,`
	for i := range t.b {
		t.b[i] = 0
	}

	// amd64:".*runtime.memclrNoHeapPointers"
	for i := range t.s {
		t.s[i] = 0
	}
}
