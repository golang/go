// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 12133.  The CX register was getting clobbered
// because we did not keep track of its allocation correctly.

package main

import "fmt"

func main() {
	want := uint(48)
	got := f1(48)
	if got != want {
		fmt.Println("got", got, ", wanted", want)
		panic("bad")
	}
}

//go:noinline
func f1(v1 uint) uint {
	return v1 >> ((1 >> v1) + (1 >> v1))
}
