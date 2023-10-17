// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test cases for sort order of declarations.

package d

// C1 should be second.
const C1 = 1

// C0 should be first.
const C0 = 0

// V1 should be second.
var V1 uint

// V0 should be first.
var V0 uintptr

// CAx constants should appear after CBx constants.
const (
	CA2 = iota // before CA1
	CA1        // before CA0
	CA0        // at end
)

// VAx variables should appear after VBx variables.
var (
	VA2 int // before VA1
	VA1 int // before VA0
	VA0 int // at end
)

// T1 should be second.
type T1 struct{}

// T0 should be first.
type T0 struct{}

// F1 should be second.
func F1() {}

// F0 should be first.
func F0() {}
