// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decl

func a() {
	var b, c bool // want `b declared (and|but) not used`
	panic(c)

	if 1 == 1 {
		var s string // want `s declared (and|but) not used`
	}
}

func b() {
	// b is a variable
	var b bool // want `b declared (and|but) not used`
}

func c() {
	var (
		d string

		// some comment for c
		c bool // want `c declared (and|but) not used`
	)

	panic(d)
}
