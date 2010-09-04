// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"
import "reflect"

type S struct {
	A, B, C, X, Y, Z int
}

type T struct {
	S
}

var a1 = S { 0, 0, 0, 1, 2, 3 }
var b1 = S { X: 1, Z: 3, Y: 2 }

var a2 = S { 0, 0, 0, 0, 0, 0, }
var b2 = S { }

var a3 = T { S { 1, 2, 3, 0, 0, 0, } }
var b3 = T { S: S{ A: 1, B: 2, C: 3 } }

var a4 = &[16]byte { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, }
var b4 = &[16]byte { 4: 1, 1, 1, 1, 12: 1, 1, }

var a5 = &[16]byte { 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, }
var b5 = &[16]byte { 1, 4: 1, 1, 1, 1, 12: 1, 1, }

var a6 = &[16]byte { 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, }
var b6 = &[...]byte { 1, 4: 1, 1, 1, 1, 12: 1, 1, 0, 0,}

type Same struct {
	a, b interface{}
}

var same = []Same {
	Same{ a1, b1 },
	Same{ a2, b2 },
	Same{ a3, b3 },
	Same{ a4, b4 },
	Same{ a5, b5 },
	Same{ a6, b6 },
}

func main() {
	ok := true
	for _, s := range same {
		if !reflect.DeepEqual(s.a, s.b) {
			ok = false
			fmt.Printf("not same: %v and %v\n", s.a, s.b)
		}
	}
	if !ok {
		fmt.Println("BUG: test/initialize")
	}
}
