// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./p1"

type MyObject struct {
	p1.Fer
}

func main() {
	var b p1.Fer = &p1.Object{}
	p1.PrintFer(b)
	var c p1.Fer = &MyObject{b}
	p1.PrintFer(c)
}
