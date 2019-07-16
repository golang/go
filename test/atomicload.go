// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that we do loads exactly once. The SSA backend
// once tried to do the load in f twice, once sign extended
// and once zero extended.  This can cause problems in
// racy code, particularly sync/mutex.

package main

func f(p *byte) bool {
	x := *p
	a := int64(int8(x))
	b := int64(uint8(x))
	return a == b
}

func main() {
	var x byte
	const N = 1000000
	c := make(chan struct{})
	go func() {
		for i := 0; i < N; i++ {
			x = 1
		}
		c <- struct{}{}
	}()
	go func() {
		for i := 0; i < N; i++ {
			x = 2
		}
		c <- struct{}{}
	}()

	for i := 0; i < N; i++ {
		if !f(&x) {
			panic("non-atomic load!")
		}
	}
	<-c
	<-c
}
