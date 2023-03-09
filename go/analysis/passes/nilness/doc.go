// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package nilness inspects the control-flow graph of an SSA function
// and reports errors such as nil pointer dereferences and degenerate
// nil pointer comparisons.
//
// # Analyzer nilness
//
// nilness: check for redundant or impossible nil comparisons
//
// The nilness checker inspects the control-flow graph of each function in
// a package and reports nil pointer dereferences, degenerate nil
// pointers, and panics with nil values. A degenerate comparison is of the form
// x==nil or x!=nil where x is statically known to be nil or non-nil. These are
// often a mistake, especially in control flow related to errors. Panics with nil
// values are checked because they are not detectable by
//
//	if r := recover(); r != nil {
//
// This check reports conditions such as:
//
//	if f == nil { // impossible condition (f is a function)
//	}
//
// and:
//
//	p := &v
//	...
//	if p != nil { // tautological condition
//	}
//
// and:
//
//	if p == nil {
//		print(*p) // nil dereference
//	}
//
// and:
//
//	if p == nil {
//		panic(p)
//	}
package nilness
