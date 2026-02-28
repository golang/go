// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) != 2 {
		fatal("usage: callback testname")
	}
	switch os.Args[1] {
	default:
		fatal("unknown test %q", os.Args[1])
	case "Call":
		testCall()
	case "Callback":
		testCallback()
	}
	println("OK")
}

func fatal(f string, args ...any) {
	fmt.Fprintln(os.Stderr, fmt.Sprintf(f, args...))
	os.Exit(1)
}

type GoCallback struct{}

func (p *GoCallback) Run() string {
	return "GoCallback.Run"
}

func testCall() {
	c := NewCaller()
	cb := NewCallback()

	c.SetCallback(cb)
	s := c.Call()
	if s != "Callback::run" {
		fatal("unexpected string from Call: %q", s)
	}
	c.DelCallback()
}

func testCallback() {
	c := NewCaller()
	cb := NewDirectorCallback(&GoCallback{})
	c.SetCallback(cb)
	s := c.Call()
	if s != "GoCallback.Run" {
		fatal("unexpected string from Call with callback: %q", s)
	}
	c.DelCallback()
	DeleteDirectorCallback(cb)
}
