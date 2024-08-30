// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for a garbage collection bug involving not
// marking x as having its address taken by &x[0]
// when x is an array value.

package main

import (
	"bytes"
	"fmt"
	"runtime"
)

func main() {
	var x = [4]struct{ x, y interface{} }{
		{"a", "b"},
		{"c", "d"},
		{"e", "f"},
		{"g", "h"},
	}

	var buf bytes.Buffer
	for _, z := range x {
		runtime.GC()
		fmt.Fprintf(&buf, "%s %s ", z.x.(string), z.y.(string))
	}

	if buf.String() != "a b c d e f g h " {
		println("BUG wrong output\n", buf.String())
	}
}
