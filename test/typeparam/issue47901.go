// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Chan[T any] chan Chan[T]

func (ch Chan[T]) recv() Chan[T] {
	return <-ch
}

func main() {
	ch := Chan[int](make(chan Chan[int]))
	go func() {
		ch <- make(Chan[int])
	}()
	ch.recv()
}
