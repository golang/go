// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Recv <-chan int

type sliceOf[E any] interface {
	~[]E
}

func _Append[S sliceOf[T], T any](s S, t ...T) S {
	return append(s, t...)
}

func main() {
	recv := make(Recv)
	a := _Append([]Recv{recv}, recv)
	if len(a) != 2 || a[0] != recv || a[1] != recv {
		panic(a)
	}

	recv2 := make(chan<- int)
	a2 := _Append([]chan<- int{recv2}, recv2)
	if len(a2) != 2 || a2[0] != recv2 || a2[1] != recv2 {
		panic(a)
	}
}
