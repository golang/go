// asmcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func f() {
	ch1 := make(chan int)
	ch2 := make(chan int)
	for {
		// amd64:-`MOVQ\t[$]0, ""..autotmp_3`
		select {
		case <-ch1:
		case <-ch2:
		default:
		}
	}
}
