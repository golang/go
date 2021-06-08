// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Mer interface{ M() }

func F[T Mer](expectPanic bool) {
	defer func() {
		err := recover()
		if (err != nil) != expectPanic {
			print("FAIL: (", err, " != nil) != ", expectPanic, "\n")
		}
	}()

	var t T
	T.M(t)
}

type MyMer int

func (MyMer) M() {}

func main() {
	F[Mer](true)
	F[struct{ Mer }](true)
	F[*struct{ Mer }](true)

	F[MyMer](false)
	F[*MyMer](true)
	F[struct{ MyMer }](false)
	F[struct{ *MyMer }](true)
	F[*struct{ MyMer }](true)
	F[*struct{ *MyMer }](true)
}
