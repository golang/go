// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gccgo crashed compiling this.

package main

type S struct {
	f [2][]int
}

func F() (r [2][]int) {
	return
}

func main() {
	var a []S
	a[0].f = F()
}
