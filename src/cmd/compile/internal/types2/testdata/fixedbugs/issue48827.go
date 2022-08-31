// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type G[P any] int

type (
	_ G[int]
	_ G[G /* ERROR "cannot use.*without instantiation" */]
	_ bool /* ERROR "invalid operation: bool\[int\] \(bool is not a generic type\)" */ [int]
	_ bool /* ERROR "invalid operation: bool\[G\] \(bool is not a generic type\)" */[G]
)

// The example from the issue.
func _() {
	_ = &([10]bool /* ERROR "invalid operation.*bool is not a generic type" */ [1 /* ERROR expected type */ ]{})
}
