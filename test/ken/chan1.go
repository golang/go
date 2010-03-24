// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

const N = 1000 // sent messages
const M = 10   // receiving goroutines
const W = 2    // channel buffering
var h [N]int   // marking of send/recv

func r(c chan int, m int) {
	for {
		select {
		case r := <-c:
			if h[r] != 1 {
				println("r",
					"m=", m,
					"r=", r,
					"h=", h[r])
				panic("fail")
			}
			h[r] = 2
		}
	}
}

func s(c chan int) {
	for n := 0; n < N; n++ {
		r := n
		if h[r] != 0 {
			println("s")
			panic("fail")
		}
		h[r] = 1
		c <- r
	}
}

func main() {
	c := make(chan int, W)
	for m := 0; m < M; m++ {
		go r(c, m)
		runtime.Gosched()
	}
	runtime.Gosched()
	runtime.Gosched()
	s(c)
}
