// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=915

package main

type T struct {
	x int
}

var t = &T{42}
var i interface{} = t
var tt, ok = i.(*T)

func main() {
	if tt == nil || tt.x != 42 {
		println("BUG")
	}
}
