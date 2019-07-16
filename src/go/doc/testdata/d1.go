// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test cases for sort order of declarations.

package d

// C2 should be third.
const C2 = 2

// V2 should be third.
var V2 int

// CBx constants should appear before CAx constants.
const (
	CB2 = iota // before CB1
	CB1        // before CB0
	CB0        // at end
)

// VBx variables should appear before VAx variables.
var (
	VB2 int // before VB1
	VB1 int // before VB0
	VB0 int // at end
)

const (
	// Single const declarations inside ()'s are considered ungrouped
	// and show up in sorted order.
	Cungrouped = 0
)

var (
	// Single var declarations inside ()'s are considered ungrouped
	// and show up in sorted order.
	Vungrouped = 0
)

// T2 should be third.
type T2 struct{}

// Grouped types are sorted nevertheless.
type (
	// TG2 should be third.
	TG2 struct{}

	// TG1 should be second.
	TG1 struct{}

	// TG0 should be first.
	TG0 struct{}
)

// F2 should be third.
func F2() {}
