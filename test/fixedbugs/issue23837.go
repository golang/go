// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f(p, q *struct{}) bool {
	return *p == *q
}

type T struct {
	x struct{}
	y int
}

//go:noinline
func g(p, q *T) bool {
	return p.x == q.x
}

//go:noinline
func h(p, q func() struct{}) bool {
	return p() == q()
}

func fi(p, q *struct{}) bool {
	return *p == *q
}

func gi(p, q *T) bool {
	return p.x == q.x
}

func hi(p, q func() struct{}) bool {
	return p() == q()
}

func main() {
	shouldPanic(func() { f(nil, nil) })
	shouldPanic(func() { g(nil, nil) })
	shouldPanic(func() { h(nil, nil) })
	shouldPanic(func() { fi(nil, nil) })
	shouldPanic(func() { gi(nil, nil) })
	shouldPanic(func() { hi(nil, nil) })
	n := 0
	inc := func() struct{} {
		n++
		return struct{}{}
	}
	h(inc, inc)
	if n != 2 {
		panic("inc not called")
	}
	hi(inc, inc)
	if n != 4 {
		panic("inc not called")
	}
}

func shouldPanic(x func()) {
	defer func() {
		if recover() == nil {
			panic("did not panic")
		}
	}()
	x()
}
