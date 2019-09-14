// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzzymatch

func _() {
	var a struct {
		fabar  int
		fooBar string
	}

	a.fabar  //@item(fuzzFabarField, "a.fabar", "int", "field")
	a.fooBar //@item(fuzzFooBarField, "a.fooBar", "string", "field")

	afa //@complete(" //", fuzzFabarField, fuzzFooBarField)
	afb //@complete(" //", fuzzFooBarField, fuzzFabarField)

	fab //@complete(" //", fuzzFabarField)

	var myString string
	myString = af //@complete(" //", fuzzFooBarField, fuzzFabarField)

	var b struct {
		c struct {
			d struct {
				e struct {
					abc string
				}
				abc float32
			}
			abc bool
		}
		abc int
	}

	b.abc       //@item(fuzzABCInt, "b.abc", "int", "field")
	b.c.abc     //@item(fuzzABCbool, "b.c.abc", "bool", "field")
	b.c.d.abc   //@item(fuzzABCfloat, "b.c.d.abc", "float32", "field")
	b.c.d.e.abc //@item(fuzzABCstring, "b.c.d.e.abc", "string", "field")

	// in depth order by default
	abc //@complete(" //", fuzzABCInt, fuzzABCbool, fuzzABCfloat)

	// deep candidate that matches expected type should still ranked first
	var s string
	s = abc //@complete(" //", fuzzABCstring, fuzzABCInt, fuzzABCbool)
}
