// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	c := make(chan bool, 1)
	ok := true
	for i := 0; i < 12; i++ {
		select {
		case _, ok = <-c:
			if i < 10 && !ok {
				panic("BUG")
			}
		default:
		}
		if i < 10 && !ok {
			panic("BUG")
		}
		if i >= 10 && ok {
			close(c)
		}
	}
}
