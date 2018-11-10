// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

func panicWhenNot(cond bool) {
	if cond {
		os.Exit(0)
	} else {
		panic("nilcheck elim failed")
	}
}

func main() {
	e := (*string)(nil)
	panicWhenNot(e == e)
	// Should never reach this line.
	panicWhenNot(*e == *e)
}
