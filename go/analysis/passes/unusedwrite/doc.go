// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unusedwrite checks for unused writes to the elements of a struct or array object.
//
// # Analyzer unusedwrite
//
// unusedwrite: checks for unused writes
//
// The analyzer reports instances of writes to struct fields and
// arrays that are never read. Specifically, when a struct object
// or an array is copied, its elements are copied implicitly by
// the compiler, and any element write to this copy does nothing
// with the original object.
//
// For example:
//
//	type T struct { x int }
//
//	func f(input []T) {
//		for i, v := range input {  // v is a copy
//			v.x = i  // unused write to field x
//		}
//	}
//
// Another example is about non-pointer receiver:
//
//	type T struct { x int }
//
//	func (t T) f() {  // t is a copy
//		t.x = i  // unused write to field x
//	}
package unusedwrite
