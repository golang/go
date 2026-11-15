// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

var pkgSink *a.Node

func main() {
	var x, y a.Node // ERROR "moved to heap: x" "moved to heap: y"
	pkgSink = &x
	a.Link(&x, &y)
	if x.Next != &y {
		panic("imported direct link mismatch") // ERROR "escapes to heap"
	}
	f := a.Link
	var u, v a.Node // ERROR "moved to heap: v"
	f(&u, &v)
	if u.Next != &v {
		panic("imported value link mismatch") // ERROR "escapes to heap"
	}
	_, _, _, _ = x, y, u, v
}
