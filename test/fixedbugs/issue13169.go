// run

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	a, b, c int
}

func usestack() {
	usestack1(32)
}
func usestack1(d int) byte {
	if d == 0 {
		return 0
	}
	var b [1024]byte
	usestack1(d - 1)
	return b[3]
}

const n = 100000

func main() {
	c := make(chan interface{})
	done := make(chan bool)

	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < n; j++ {
				c <- new(T)
			}
			done <- true
		}()
		go func() {
			for j := 0; j < n; j++ {
				_ = (<-c).(*T)
				usestack()
			}
			done <- true
		}()
	}
	for i := 0; i < 20; i++ {
		<-done
	}
}
