// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	c := make(chan int, 1)
	dummy := make(chan int)
	v := 0x12345678
	for i := 0; i < 10; i++ {
		// 6g had a bug that caused select to pass &t to
		// selectrecv before allocating the memory for t,
		// which caused non-deterministic crashes.
		// This test looks for the bug by checking that the
		// value received actually ends up in t.
		// If the allocation happens after storing through
		// whatever garbage &t holds, the later reference
		// to t in the case body will use the new pointer and
		// not see the received value.
		v += 0x1020304
		c <- v
		select {
		case t := <-c:
			go func() {
				f(t)
			}()
			escape(&t)
			if t != v {
				println(i, v, t)
				panic("wrong values")
			}
		case dummy <- 1:
		}
	}
}

func escape(*int) {
}

func f(int) {
}

