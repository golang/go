// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

// complicated enough to require a compile-generated hash function
type K struct {
	a, b int32 // these get merged by the compiler into a single field, something typehash doesn't do
	c    float64
}

func main() {
	k := K{a: 1, b: 2, c: 3}

	// Make a reflect map.
	m := reflect.MakeMap(reflect.MapOf(reflect.TypeOf(K{}), reflect.TypeOf(true)))
	m.SetMapIndex(reflect.ValueOf(k), reflect.ValueOf(true))

	// The binary must not contain the type map[K]bool anywhere, or reflect.MapOf
	// will use that type instead of making a new one. So use an equivalent named type.
	type M map[K]bool
	var x M
	reflect.ValueOf(&x).Elem().Set(m)
	if !x[k] {
		panic("key not found")
	}
}
