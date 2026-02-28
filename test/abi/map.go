// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

type T [10]int

var m map[*T]int

//go:noinline
func F() {
	m = map[*T]int{
		K(): V(), // the key temp should be live across call to V
	}
}

//go:noinline
func V() int { runtime.GC(); runtime.GC(); runtime.GC(); return 123 }

//go:noinline
func K() *T {
	p := new(T)
	runtime.SetFinalizer(p, func(*T) { println("FAIL") })
	return p
}

func main() {
	F()
}
