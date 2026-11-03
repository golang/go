// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

type I1 interface{ M() }
type I2 interface{ M() }

type T struct{ x int }

func (T) M() {}

func main() {
	var want I2 = T{1}

	// Built for I1, read as an I2.
	var a I1 = T{1}
	if got := *(*I2)(&a); got != want {
		panic("value read through *I1 to *I2 does not compare equal")
	} else if _, ok := got.(T); !ok {
		panic("value read through *I1 to *I2 does not assert to T")
	} else {
		switch got.(type) {
		case T:
		default:
			panic("value read through *I1 to *I2 does not switch to T")
		}
	}

	// Same, with the I1 itab built by the runtime rather than the compiler.
	var e any = T{1}
	b := e.(I1)
	if got := *(*I2)(&b); got != want {
		panic("value read through *I1 to *I2 does not compare equal")
	}

	// An itab the runtime builds for I2 has to be the same one.
	if got := e.(I2); got != want {
		panic("value asserted from any to I2 does not compare equal")
	}

	// So does one that reflect builds.
	var c I2
	reflect.ValueOf(&c).Elem().Set(reflect.ValueOf(T{1}))
	if c != want {
		panic("value set through reflect does not compare equal")
	}
}
