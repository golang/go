// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// used to panic because 6g didn't generate
// the code to fill in the ... argument to fmt.Sprint.

package main

import "fmt"

type T struct {
	a, b, c, d, e []int;
}

var t T

func main() {
	if fmt.Sprint("xxx", t) != "yyy" { 
	}
}
