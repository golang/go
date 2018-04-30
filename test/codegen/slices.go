// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains code generation tests related to the handling of
// slice types.

// ------------------ //
//      Clear         //
// ------------------ //

// Issue #5373 optimize memset idiom

func SliceClear(s []int) []int {
	// amd64:`.*memclrNoHeapPointers`
	for i := range s {
		s[i] = 0
	}
	return s
}

func SliceClearPointers(s []*int) []*int {
	// amd64:`.*memclrHasPointers`
	for i := range s {
		s[i] = nil
	}
	return s
}
