// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that method calls through an interface always
// call the the locally defined method localT.m independent
// at which embedding level it is and in which order
// embedding is done.

package main

import "./lib"

type localI interface {
	m() string
}

type localT struct{}

func (t *localT) m() string {
	return "main.localT.m"
}

type myT1 struct {
	localT
}

type myT2 struct {
	localT
	lib.T
}

type myT3 struct {
	lib.T
	localT
}

func main() {
	var i localI

	i = new(localT)
	if i.m() != "main.localT.m" {
		println("BUG: localT:", i.m(), "called")
	}

	i = new(myT1)
	if i.m() != "main.localT.m" {
		println("BUG: myT1:", i.m(), "called")
	}

	i = new(myT2)
	if i.m() != "main.localT.m" {
		println("BUG: myT2:", i.m(), "called")
	}

	i = new(myT3)
	if i.m() != "main.localT.m" {
		println("BUG: myT3:", i.m(), "called")
	}

}
