// run

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

func main() {
	a := &struct{ x int }{}
	b := &struct{ x int "" }{}

	ta := reflect.TypeOf(a)
	tb := reflect.TypeOf(b)

	// Ensure cmd/compile treats absent and empty tags as equivalent.
	a = b

	// Ensure package reflect treats absent and empty tags as equivalent.
	if !tb.AssignableTo(ta) {
		panic("fail")
	}
}
