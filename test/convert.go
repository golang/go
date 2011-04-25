// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

func typeof(x interface{}) string { return reflect.TypeOf(x).String() }

func f() int { return 0 }

func g() int { return 0 }

type T func() int

var m = map[string]T{"f": f}

type A int
type B int

var a A = 1
var b B = 2
var x int

func main() {
	want := typeof(g)
	if t := typeof(f); t != want {
		println("type of f is", t, "want", want)
		panic("fail")
	}

	want = typeof(a)
	if t := typeof(+a); t != want {
		println("type of +a is", t, "want", want)
		panic("fail")
	}
	if t := typeof(a + 0); t != want {
		println("type of a+0 is", t, "want", want)
		panic("fail")
	}
}
