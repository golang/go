// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug344

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	// invalid use of goto.
	// do whatever you like, just don't crash.
	i := 42
	a := []*int{&i, &i, &i, &i}
	x := a[0]
	goto start
	for _, x = range a {
	start:
		fmt.Sprint(*x)
	}
}
