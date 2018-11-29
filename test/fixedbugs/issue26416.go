// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type t1 struct {
	t1f1 int
	t1f2 int
}
type t2 struct {
	t2f1 int
	t2f2 int
	*t1
}
type t3 struct {
	t3f1 int
	*t2
}

var (
	_ = t2{t1f1: 600} // ERROR "cannot use promoted field t1.t1f1 in struct literal of type t2"
	_ = t3{t1f2: 800} // ERROR "cannot use promoted field t2.t1.t1f2 in struct literal of type t3"
	_ = t3{t2f1: 900} // ERROR "cannot use promoted field t2.t2f1 in struct literal of type t3"
)
