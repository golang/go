// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 81990: folding the load of a type descriptor's Equal field
// marked the type:.eqfunc.* closure as used in an interface, which
// made the linker panic.

package main

import "reflect"

type Info struct {
	A any
	B any
}

func main() {
	if !reflect.TypeFor[Info]().Comparable() {
		panic("Info should be comparable")
	}
}
