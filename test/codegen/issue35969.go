// asmcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func chanTempVar(ch chan int) {
	x := <-ch
	// amd64: -`MOVQ\tAX, ""\.\.autotmp_\d+\+\d+\(SP\)`
	ch <- x
}
