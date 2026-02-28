// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure that literal value can be passed to struct
// blank field of array/struct type, see issue #38690.

package main

type A1 = [0]int
type A2 = [1]int

type S1 struct{}

type S2 struct {
	x int
}

type S3 = struct{}

type S4 = struct{ x int }

type S struct {
	x int
	_ [0]int
	_ [1]int
	_ A1
	_ A2
	_ S1
	_ S2
	_ S3
	_ S4
	_ [1]S4
}

var s = S{1, [0]int{}, [1]int{1}, A1{}, A2{1}, S1{}, S2{1}, S3{}, S4{1}, [1]S4{}}

func main() {
	f1()
	mustPanic(f2)
	mustPanic(f3)
}

func f1() {
	_ = S{1, [0]int{}, [1]int{1}, A1{}, A2{1}, S1{}, S2{1}, S3{}, S4{1}, [1]S4{}}
}

func f2() {
	_ = S{1, [0]int{}, [1]int{1}, A1{}, A2{1}, S1{}, S2{1}, S3{}, func() S4 { panic("") }(), [1]S4{}}
}

func f3() {
	_ = S{1, [0]int{}, [1]int{1}, A1{}, A2{1}, S1{}, S2{1}, S3{}, S4{1}, func() [1]S4 { panic("") }()}
}

func mustPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("expected panic, got nil")
		}
	}()
	f()
}
