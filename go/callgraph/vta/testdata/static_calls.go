// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface{}

func foo(i I) (I, I) {
	return i, i
}

func doWork(ii I) {}

func close(iii I) {}

func Baz(inp I) {
	a, b := foo(inp)
	defer close(a)
	go doWork(b)
}

// Relevant SSA:
// func Baz(inp I):
//   t0 = foo(inp)
//   t1 = extract t0 #0
//   t2 = extract t0 #1
//   defer close(t1)
//   go doWork(t2)
//   rundefers
//   ...
// func foo(i I) (I, I):
//   return i, i

// WANT:
// Local(inp) -> Local(i)
// Local(t1) -> Local(iii)
// Local(t2) -> Local(ii)
// Local(i) -> Local(t0[0]), Local(t0[1])
