// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test makes sure that we don't split a single
// load up into two separate loads.

package main

import "fmt"

//go:noinline
func read(b []byte) (uint16, uint16) {
	// There is only a single read of b[0].  The two
	// returned values must have the same low byte.
	v := b[0]
	return uint16(v), uint16(v) | uint16(b[1])<<8
}

const N = 100000

func main() {
	done := make(chan struct{})
	b := make([]byte, 2)
	go func() {
		for i := 0; i < N; i++ {
			b[0] = byte(i)
			b[1] = byte(i)
		}
		done <- struct{}{}
	}()
	go func() {
		for i := 0; i < N; i++ {
			x, y := read(b)
			if byte(x) != byte(y) {
				fmt.Printf("x=%x y=%x\n", x, y)
				panic("bad")
			}
		}
		done <- struct{}{}
	}()
	<-done
	<-done
}
