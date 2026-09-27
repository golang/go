// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"

	"./a"
	"./b"
)

// Comparing two interfaces whose dynamic type the compiler knows is
// compiled to a direct comparison, so compare through a function that has
// to go through the type descriptor at run time.
//
//go:noinline
func eq(x, y any) bool { return x == y }

func main() {
	// main never mentions [2]a.K itself, so the only descriptors for it
	// are the ones in packages a and b.
	_ = a.Sink
	x, y := b.Get(), b.Get()
	if !reflect.TypeOf(x).Comparable() {
		panic("descriptor for [2]a.K has no equality function")
	}
	if !eq(x, y) {
		panic("[2]a.K values compare unequal")
	}
}
