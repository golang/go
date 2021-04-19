// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func bad used to fail to compile.

package p

type A [1]int

func bad(x A) {
	switch x {
	case A([1]int{1}):
	case A([1]int{1}):
	}
}

func good(x A) {
	y := A([1]int{1})
	z := A([1]int{1})
	switch x {
	case y:
	case z:
	}
}
