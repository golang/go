// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The Go 1.18 frontend failed to disambiguate instantiations of
// different, locally defined generic types with the same name.
//
// The unified frontend also exposed the scope-disambiguation mangling
// to end users in reflect data.

package main

import (
	"reflect"
)

func one() any { type T[_ any] int; return T[int](0) }
func two() any { type T[_ any] int; return T[int](0) }

func main() {
	p, q := one(), two()

	// p and q have different dynamic types; this comparison should
	// evaluate false.
	if p == q {
		panic("bad type identity")
	}

	for _, x := range []any{p, q} {
		// The names here should not contain "·1" or "·2".
		if name := reflect.TypeOf(x).String(); name != "main.T[int]" {
			panic(name)
		}
	}
}
